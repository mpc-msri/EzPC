#pragma once
#include <sytorch/utils.h>
#include <llama/assert.h>
#include <sytorch/backend/default.h>
#include <string>

template <typename T>
class Layer {
public:
    std::string name;
    Tensor<T> activation;
    Backend<T> *backend = nullptr;

    // config options
    bool doTruncationForward = false;
    bool doPreSignExtension = false;
    bool doPostSignExtension = false;
    bool isFirst = false;
    u64 scale = 0;
    int mode = 0; // only used in ReLU in llama improved to decide between relu and reluext, might need something cleaner?
    int forwardTruncationMode = 0;
    bool useBias = true;
    bool isTrainingMode = false;
    std::string paramstring  = "";

    LayerGraphNode<T> *node = nullptr;

    Layer(const std::string &name) : activation({0}), name(name) {
        backend = defaultBackend<T>();
    }

    virtual void _initScale(u64 scale) {};
    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
        _initScale(scale);
    };
    
    virtual void _resize(const std::vector<std::vector<u64>> &shapes) {};
    void resize(const std::vector<std::vector<u64>> &shapes) {
        auto outdims = this->get_output_dims(shapes);
        activation.resize(outdims);
        _resize(shapes);
    }

    virtual void _forward(Tensor<T> &a) = 0;
    virtual void _forward(std::vector<Tensor<T> *> &a) { 
        if (a.size() != 1)
            throw std::runtime_error("variable input cardinality not supported in this layer");
       _forward(*a[0]); 
    };
    
    template <typename... Args>
    Tensor<T>& forward(Args & ... args) {
        std::vector<Tensor<T> *> a = collect(args...);
        return forward(a);
    }

    Tensor<T>& forward(std::vector<Tensor<T> *> &a) {
        if (a[0]->graphGenMode) {
            for (auto &i : a) {
                always_assert(i->graphGenMode);
            }
            node = new LayerGraphNode<T>();
            node->layer = this;
            node->allNodesInExecutionOrderRef = a[0]->graphNode->allNodesInExecutionOrderRef;
            node->allNodesInExecutionOrderRef->push_back(node);

            for(auto &i : a) {
                auto parentNode = i->graphNode;
                always_assert(parentNode->allNodesInExecutionOrderRef == node->allNodesInExecutionOrderRef);
                node->parents.push_back(parentNode);
                parentNode->children.push_back(node);
            }

            activation.graphNode = node;
            activation.graphGenMode = true;
            return activation;
        }

        // check if we have the graph generated already
        always_assert(node != nullptr);
        for(auto &i : a) {
            always_assert(i->graphNode != nullptr);
        }
        
        activation.graphGenMode = false;
        resize(getShapes(a));
        node->currTensor = &activation;
        activation.graphNode = node;

        if (doPreSignExtension) {
            for(auto &i : a) {
                this->backend->signext(*i, scale);
            }
        }
        _forward(a);
        if (doTruncationForward) {
            this->backend->truncateForward(activation, scale, forwardTruncationMode);
        }
        if (doPostSignExtension) {
            this->backend->signext(activation, scale);
        }
        for(auto &i : a) {
            i->graphNode->incrementAndGc();
        }
        return activation;
    }

    virtual TensorRef<T> getweights() { return TensorRef<T>(nullptr, 0); };
    virtual TensorRef<T> getbias() { return TensorRef<T>(nullptr, 0); };
    virtual std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) = 0;

    virtual void setBackend(Backend<T> *b) {
        backend = b;
    }

    void train() {
        isTrainingMode = true;
    }

    void eval() {
        isTrainingMode = false;
    }
};

template <typename T>
class Conv2D : public Layer<T> {
public:
    Tensor<T> inp;
    Tensor2D<T> filter;
    Tensor1D<T> bias;
    u64 ci, co;
    u64 fh, fw, padding, stride;

    Conv2D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("Conv2D"), ci(ci), co(co), fh(f), fw(f), 
        padding(padding), stride(stride), filter(co, f * f * ci), bias(co), inp({0,0,0,0})
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv2D(u64 ci, u64 co, const std::array<u64, 2> f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("Conv2D"), ci(ci), co(co), fh(f[0]), fw(f[1]), 
        padding(padding), stride(stride), filter(co, f[0] * f[1] * ci), bias(co), inp({0,0,0,0})
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale) {
        double xavier = 1.0 / sqrt(ci * fh * fw);
        filter.randomize(xavier * (1ULL<<scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL<<(2*scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        always_assert(shape[3] == ci);
        if (this->isTrainingMode)
            inp.resize(shape);
    }

    void _forward(Tensor<T> &a) {
        always_assert(a.shape.size() == 4);
        assert(a.shape[3] == ci);
        if (this->isTrainingMode)
            inp.copy(a, false);
        auto act_4d = this->activation.as_4d();
        this->backend->conv2D(fh, fw, padding, stride, ci, co, a.as_4d(), filter, act_4d);
        if (this->useBias)
            this->backend->addbias(this->activation, bias);
    }

    TensorRef<T> getweights() { return filter.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        always_assert(inShape[3] == ci);
        u64 newH = (((inShape[1] + 2*padding - fh)/stride) + 1);
        u64 newW = (((inShape[2] + 2*padding - fw)/stride) + 1);
        return {inShape[0], newH, newW, co};
    }
};

template <typename T>
class Conv3D : public Layer<T> {
public:
    Tensor<T> inp;
    Tensor2D<T> filter;
    Tensor1D<T> bias;
    u64 ci, co;
    u64 fd, fh, fw;
    u64 pd, ph, pw;
    u64 sd, sh, sw;
    u64 dd, dh, dw;

    Conv3D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, u64 dialation = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f), fh(f), fw(f), 
        pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), filter(co, f * f * f * ci), bias(co), inp({0,0,0,0,0})
    {
        always_assert(dialation == 1);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv3D(u64 ci, u64 co, const std::array<u64, 3> f, u64 padding = 0, u64 stride = 1, u64 dialation = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]), 
        pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), dd(dialation), dh(dialation), dw(dialation), filter(co, f[0] * f[1] * f[2] * ci), bias(co), inp({0,0,0,0,0})
    {
        always_assert(dialation == 1);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv3D(u64 ci, u64 co, const std::array<u64, 3> f, const std::array<u64, 3> padding = {0, 0, 0}, u64 stride = 1, u64 dialation = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]), 
        pd(padding[0]), ph(padding[1]), pw(padding[2]), sd(stride), sh(stride), sw(stride), dd(dialation), dh(dialation), dw(dialation), filter(co, f[0] * f[1] * f[2] * ci), bias(co), inp({0,0,0,0,0})
    {
        always_assert(dialation == 1);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv3D(u64 ci, u64 co, const std::array<u64, 3> f, const std::array<u64, 6> padding = {0, 0, 0, 0, 0, 0}, const std::array<u64, 3> stride = {1, 1, 1}, const std::array<u64, 3> dialation = {1, 1, 1}, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]), 
        pd(padding[0]), ph(padding[1]), pw(padding[2]), sd(stride[0]), sh(stride[1]), sw(stride[2]), dd(dialation[0]), dh(dialation[1]), dw(dialation[2]), filter(co, f[0] * f[1] * f[2] * ci), bias(co), inp({0,0,0,0,0})
    {
        always_assert(dialation[0] == 1);
        always_assert(dialation[1] == 1);
        always_assert(dialation[2] == 1);
        always_assert(padding[3] == padding[0]);
        always_assert(padding[4] == padding[1]);
        always_assert(padding[5] == padding[2]);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale) {
        double xavier = 1.0 / sqrt(ci * fd * fh * fw);
        filter.randomize(xavier * (1ULL<<scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL<<(2*scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 5);
        always_assert(shape[4] == ci);
        if (this->isTrainingMode)
            inp.resize(shape);
    }

    void _forward(Tensor<T> &a) {
        always_assert(a.shape.size() == 5);
        assert(a.shape[4] == ci);
        if (this->isTrainingMode)
            inp.copy(a, false);
        auto act_5d = this->activation.as_5d();
        this->backend->conv3D(fd, fh, fw, pd, ph, pw, sd, sh, sw, dd, dh, dw, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->backend->addbias(this->activation, bias);
    }

    TensorRef<T> getweights() { return filter.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 5);
        always_assert(inShape[4] == ci);
        u64 newD = (((inShape[1] + 2*pd - fd - (fd-1)*(dd-1))/sd) + 1);
        u64 newH = (((inShape[2] + 2*ph - fh - (fh-1)*(dh-1))/sh) + 1);
        u64 newW = (((inShape[3] + 2*pw - fw - (fw-1)*(dw-1))/sw) + 1);
        return {inShape[0], newD, newH, newW, co};
    }
};

template <typename T>
class AvgPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;

    AvgPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("AvgPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
    }

    void _forward(Tensor<T> &a) {
        always_assert(a.shape.size() == 4);
        auto act_4d = this->activation.as_4d();
        auto a_4d = a.as_4d();
        this->backend->avgPool2D(ks, padding, stride, a_4d, act_4d, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        u64 newH = (((inShape[1] + 2*padding - ks)/stride) + 1);
        u64 newW = (((inShape[2] + 2*padding - ks)/stride) + 1);
        return {inShape[0], newH, newW, inShape[3]};
    }
};

template <typename T>
class SumPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;

    SumPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("SumPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
    }

    void _forward(Tensor<T> &a) {
        this->backend->sumPool2D(ks, padding, stride, a.as_4d(), this->activation.as_4d());
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        u64 newH = (((inShape[1] + 2*padding - ks)/stride) + 1);
        u64 newW = (((inShape[2] + 2*padding - ks)/stride) + 1);
        return {inShape[0], newH, newW, inShape[3]};
    }
};

template <typename T>
class MaxPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;
    Tensor4D<u64> maxIndex;

    MaxPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("MaxPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride), maxIndex(0,0,0,0) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        this->maxIndex.resize(this->activation.shape);
    }

    void _forward(Tensor<T> &a) {
        auto a_4d = a.as_4d();
        auto act_4d = this->activation.as_4d();
        this->backend->maxPool2D(ks, padding, stride, a_4d, act_4d, maxIndex, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        u64 newH = (((inShape[1] + 2*padding - ks)/stride) + 1);
        u64 newW = (((inShape[2] + 2*padding - ks)/stride) + 1);
        return {inShape[0], newH, newW, inShape[3]};
    }
};

template <typename T>
class Flatten : public Layer<T> {
public:

    Flatten() : Layer<T>("Flatten") {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() >= 2);
    }

    void _forward(Tensor<T> &a) {
        if (a.shape.size() == 4) {
            auto a_4d = a.as_4d();
            auto act_2d = this->activation.as_2d();
            u64 d1 = a.shape[0];
            u64 d2 = a.shape[1];
            u64 d3 = a.shape[2];
            u64 d4 = a.shape[3];

            for (u64 i = 0; i < d1; i++) {
                for (u64 j = 0; j < d2; j++) {
                    for (u64 k = 0; k < d3; k++) {
                        for (u64 l = 0; l < d4; l++) {
                            // this->activation(i, j * d3 * d4 + k * d4 + l, 0, 0) = a(i, j, k, l);
                            act_2d(i, l * d2 * d3 + j * d3 + k) = a_4d(i, j, k, l);
                        }
                    }
                }
            }
        }
        else if (a.shape.size() == 5) {
            auto a_5d = a.as_5d();
            auto act_2d = this->activation.as_2d();
            u64 d1 = a.shape[0];
            u64 d2 = a.shape[1];
            u64 d3 = a.shape[2];
            u64 d4 = a.shape[3];
            u64 d5 = a.shape[4];

            for (u64 i = 0; i < d1; i++) {
                for (u64 j = 0; j < d2; j++) {
                    for (u64 k = 0; k < d3; k++) {
                        for (u64 l = 0; l < d4; l++) {
                            for (u64 m = 0; m < d5; ++m) {
                                act_2d(i, m * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l) = a_5d(i, j, k, l, m);
                            }
                        }
                    }
                }
            }
        }
        else {
            u64 sz = a.size();
            for (u64 i = 0; i < sz; i++) {
                this->activation.data[i] = a.data[i];
            }
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        u64 prod = 1;
        for(int i = 1; i < inShape.size(); i++) {
            prod *= inShape[i];
        }
        return {inShape[0], prod};
    }
};

template <typename T>
class FC : public Layer<T> {
public:
    Tensor<T> inp;
    Tensor2D<T> weight;
    Tensor1D<T> bias;
    u64 in, out;

    FC(u64 in, u64 out, bool useBias = false) : Layer<T>("FC"), in(in), out(out), weight(in, out), bias(out), inp({0,0,0,0}) {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale) {
        double xavier = 1.0 / sqrt(in);
        weight.randomize(xavier * (1ULL<<scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL<<(2*scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 2);
        always_assert(shape[1] == in);
        inp.resize(shape);
    }

    void _forward(Tensor<T> &a) {
        this->inp.copy(a, false);
        auto a_2d = a.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->matmul(a_2d, weight, act_2d);
        if (this->useBias)
            this->backend->addbias(this->activation, bias);
    }

    TensorRef<T> getweights() { return weight.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 2);
        assert(inShape[1] == in);
        return {inShape[0], out};
    }
};

template <typename T>
class ReLU: public Layer<T> {
public:
    Tensor<T> drelu;
    ReLU() :  Layer<T>("ReLU"), drelu({0}) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        this->drelu.resize(shape);
    }

    void _forward(Tensor<T> &a) {
        this->backend->relu(a, this->activation, this->drelu, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class LeakyReLU : public Layer<T>
{
public:
    Tensor<T> drelu;
    double alpha;
    LeakyReLU(double alpha) : Layer<T>("LeakyReLU"), drelu({0}), alpha(alpha) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        this->drelu.resize(shape);
        always_assert(this->alpha >= 0.0);
    }

    void _forward(Tensor<T> &a)
    {
        T alphaFix = type_cast<T>(alpha * (1LL << this->scale));
        this->backend->leakyRelu(a, this->activation, this->drelu, this->scale, this->mode, alphaFix);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class BatchNormInference : public Layer<T> {
public:
    Tensor1D<T> A; // scale = s
    Tensor1D<T> B; // scale = 2s

    BatchNormInference(u64 channels) : Layer<T>("BatchNormInference"), A(channels), B(channels) {
        this->A.fill(0);
        this->B.fill(0);
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        // always_assert(shape.size() == 4);
        always_assert(shape.back() == this->A.d1);
    }

    void _forward(Tensor<T> &a) {
        // always_assert(a.shape.size() == 4);
        assert(a.shape.back() == this->A.d1);
        if (this->isTrainingMode) {
            std::runtime_error("BatchNormInference should not be used in training mode");
        }
        else {
            this->backend->batchNormInference(this->A, this->B, a, this->activation, this->scale);
        }
    }

    TensorRef<T> getweights() { return A.ref(); }
    TensorRef<T> getbias() { return B.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        // always_assert(inShape.size() == 4);
        always_assert(inShape.back() == this->A.d1);
        return inShape;
    }
};

template <typename T>
class Identity: public Layer<T> {
public:
    Identity() :  Layer<T>("Identity") {}

    void _forward(Tensor<T> &a) {
        this->activation.copy(a, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class GlobalAvgPool2D : public Layer<T> {
public:

    GlobalAvgPool2D() : Layer<T>("GlobalAvgPool2D") {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        always_assert(shape[1] == shape[2]);
    }
    
    void _forward(Tensor<T> &a) {
        auto a_4d = a.as_4d();
        auto act_4d = this->activation.as_4d();
        this->backend->avgPool2D(a_4d.d2, 0, 1, a_4d, act_4d, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        always_assert(inShape[1] == inShape[2]);
        return {inShape[0], 1, 1, inShape[3]};
    }
};


template <typename T>
class ConvTranspose3D : public Layer<T> {
public:
    Tensor2D<T> filter;
    Tensor1D<T> bias;
    u64 ci, co;
    u64 fd, fh, fw;
    u64 pd, ph, pw;
    u64 sd, sh, sw;

    ConvTranspose3D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f), fh(f), fw(f), 
        pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), filter(co, f * f * f * ci), bias(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    ConvTranspose3D(u64 ci, u64 co, const std::array<u64, 3> f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]), 
        pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), filter(co, f[0] * f[1] * f[2] * ci), bias(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    ConvTranspose3D(u64 ci, u64 co, const std::array<u64, 3> f, const std::array<u64, 6> padding = {0, 0, 0, 0, 0, 0}, const std::array<u64, 3> stride = {1, 1, 1}, const std::array<u64, 3> dialation = {1, 1, 1}, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]), 
        pd(padding[0]), ph(padding[1]), pw(padding[2]), sd(stride[0]), sh(stride[1]), sw(stride[2]), filter(co, f[0] * f[1] * f[2] * ci), bias(co)
    {
        always_assert(dialation[0] == 1);
        always_assert(dialation[1] == 1);
        always_assert(dialation[2] == 1);
        always_assert(padding[3] == padding[0]);
        always_assert(padding[4] == padding[1]);
        always_assert(padding[5] == padding[2]);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale) {
        double xavier = 1.0 / sqrt(ci * fd * fh * fw);
        filter.randomize(xavier * (1ULL<<scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL<<(2*scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 5);
        always_assert(shape[4] == ci);
    }

    void _forward(Tensor<T> &a) {
        always_assert(a.shape.size() == 5);
        assert(a.shape[4] == ci);
        auto act_5d = this->activation.as_5d();
        this->backend->convTranspose3D(fd, fh, fw, pd, ph, pw, sd, sh, sw, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->backend->addbias(this->activation, bias);
    }

    TensorRef<T> getweights() { return filter.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 5);
        always_assert(inShape[4] == ci);
        u64 newD = (((inShape[1] - 1)*sd + fd - 2*pd));
        u64 newH = (((inShape[2] - 1)*sh + fh - 2*ph));
        u64 newW = (((inShape[3] - 1)*sw + fw - 2*pw));
        return {inShape[0], newD, newH, newW, co};
    }
};

template <typename T>
class ConvTranspose2D : public Layer<T>
{
public:
    Tensor2D<T> filter;
    Tensor1D<T> bias;
    u64 ci, co;
    u64 fh, fw;
    u64 ph, pw;
    u64 sh, sw;

    ConvTranspose2D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose2D"), ci(ci), co(co), fh(f), fw(f),
                                                                                                    ph(padding), pw(padding), sh(stride), sw(stride), filter(co, f * f * ci), bias(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    ConvTranspose2D(u64 ci, u64 co, const std::array<u64, 2> f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose2D"), ci(ci), co(co), fh(f[0]), fw(f[1]),
                                                                                                                         ph(padding), pw(padding), sh(stride), sw(stride), filter(co, f[0] * f[1] * ci), bias(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    ConvTranspose2D(u64 ci, u64 co, const std::array<u64, 2> f, const std::array<u64, 4> padding = {0, 0, 0, 0}, const std::array<u64, 2> stride = {1, 1}, const std::array<u64, 2> dialation = {1, 1}, bool useBias = false) : Layer<T>("ConvTranspose2D"), ci(ci), co(co), fh(f[0]), fw(f[1]),
                                                                                                                                                                                                                                ph(padding[0]), pw(padding[1]), sh(stride[0]), sw(stride[1]), filter(co, f[0] * f[1] * ci), bias(co)
    {
        always_assert(dialation[0] == 1);
        always_assert(dialation[1] == 1);
        always_assert(padding[2] == padding[0]);
        always_assert(padding[3] == padding[1]);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale)
    {
        double xavier = 1.0 / sqrt(ci * fh * fw);
        filter.randomize(xavier * (1ULL << scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL << (2 * scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        always_assert(shape[3] == ci);
    }

    void _forward(Tensor<T> &a)
    {
        always_assert(a.shape.size() == 4);
        assert(a.shape[3] == ci);
        auto act_4d = this->activation.as_4d();
        this->backend->convTranspose2D(fh, fw, ph, pw, sh, sw, ci, co, a.as_4d(), filter, act_4d);
        if (this->useBias)
            this->backend->addbias(this->activation, bias);
    }

    TensorRef<T> getweights() { return filter.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        always_assert(inShape[3] == ci);
        u64 newH = (((inShape[1] - 1) * sh + fh - 2 * ph));
        u64 newW = (((inShape[2] - 1) * sw + fw - 2 * pw));
        return {inShape[0], newH, newW, co};
    }
};

template <typename T>
class PlaceHolderLayer : public Layer<T> {
public:
    PlaceHolderLayer(const std::string &s) : Layer<T>(s) {
    }

    void initScale(u64 scale) {
        throw std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    void _forward(Tensor<T> &a) {
        throw std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class Add: public Layer<T> {
public:
    Add() :  Layer<T>("Add") {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        auto &shape0 = shapes[0];
        for (auto &shape : shapes) {
            always_assert(shape.size() == shape0.size());
            for (u64 i = 0; i < shape.size(); i++) {
                always_assert(shape[i] == shape0[i]);
            }
        }
    }

    void _forward(std::vector<Tensor<T> *> &a) {
        this->backend->add(a, this->activation);
    }

    void _forward(Tensor<T> &a) {
        this->activation.copy(a, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        auto &shape0 = inShapes[0];
        for (auto &shape : inShapes) {
            always_assert(shape.size() == shape0.size());
            for (u64 i = 0; i < shape.size(); i++) {
                assert(shape[i] == shape0[i]);
            }
        }
        auto &inShape = inShapes[0];
        return inShape;
    }
};

// concat along the last axis (channel)
template <typename T>
class Concat: public Layer<T> {
public:
    Concat() :  Layer<T>("Concat") {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        auto &shape0 = shapes[0];
        for (auto &shape : shapes) {
            for (u64 i = 0; i < shape.size() - 1; i++) {
                always_assert(shape[i] == shape0[i]);
            }
        }
    }

    void _forward(std::vector<Tensor<T> *> &arr) {
        u64 outchannels = 0;
        u64 sz = 0;
        for (auto &t : arr) {
            outchannels += t->shape.back();
            sz += t->size();
        }

        //#pragma omp parallel for
        for(int i = 0; i < sz; ++i)
        {
            u64 l = i % outchannels;
            u64 rest = i / outchannels;
            for(auto &a : arr) {
                if(l < a->shape.back()) {
                    this->activation.data[i] = a->data[rest * a->shape.back() + l];
                    break;
                }
                l -= a->shape.back();
            }
        }

    }

    void _forward(Tensor<T> &a) {
        this->activation.copy(a, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        auto &shape0 = inShapes[0];
        for (auto &shape : inShapes) {
            for (u64 i = 0; i < shape.size() - 1; i++) {
                always_assert(shape[i] == shape0[i]);
            }
        }

        std::vector<u64> outShape = shape0;
        outShape.back() = 0;
        for (auto &shape : inShapes) {
            outShape.back() += shape.back();
        }
        return outShape;
    }
};

template <typename T>
class GeLU: public Layer<T> {
public:
    GeLU() :  Layer<T>("GeLU") {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a) {
        this->backend->gelu(a, this->activation, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class SoftMax: public Layer<T> {
public:
    SoftMax() :  Layer<T>("SoftMax") {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        always_assert(shapes[0].size() == 2);
    }

    void _forward(Tensor<T> &a) {
        this->backend->softmax(a, this->activation, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        always_assert(inShapes[0].size() == 2);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class LayerNorm: public Layer<T> {
public:
    Tensor1D<T> A; // scale = s
    Tensor1D<T> B; // scale = 2s

    LayerNorm(u64 channels) : Layer<T>("LayerNorm"), A(channels), B(channels) {
        this->A.fill(0);
        this->B.fill(0);
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.back() == this->A.d1);
    }

    void _forward(Tensor<T> &a) {
        // always_assert(a.shape.size() == 4);
        assert(a.shape.back() == this->A.d1);
        this->backend->layernorm(this->A, this->B, a, this->activation, this->scale);
    }

    TensorRef<T> getweights() { return A.ref(); }
    TensorRef<T> getbias() { return B.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.back() == this->A.d1);
        return inShape;
    }
};

template <typename T>
class Split: public Layer<T> {
public:
    u64 n_splits;

    Split(u64 n_splits) :  Layer<T>("Split"), n_splits(n_splits) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.back() % n_splits == 0);
    }

    void _forward(Tensor<T> &a) {
        always_assert(a.shape.back() % n_splits == 0);
        u64 split_size = a.shape.back() / n_splits; // 3
        u64 rest_size = a.size() / a.shape.back(); // 2
        
        //#pragma omp parallel for
        for(u64 i = 0; i < a.size(); ++i) {
            u64 p = i / a.shape.back();
            u64 q = i % a.shape.back();
            u64 r = q / split_size;
            u64 s = q % split_size;
            this->activation.data[r * split_size * rest_size + p * split_size + s] = a.data[i];
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        always_assert(shape.back() % n_splits == 0);
        shape.back() /= n_splits;
        shape.insert(shape.begin(), n_splits);
        return shape;
    }
};

template <typename T>
class View: public Layer<T> {
public:
    i64 idx;

    View(i64 idx) :  Layer<T>("View"), idx(idx) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        // auto &shape = shapes[0];
        // always_assert(idx < shape[0]);
    }

    void _forward(Tensor<T> &a) {
        // always_assert(idx < a.shape[0]);
        // std::cout << (idx % a.shape[0]) << std::endl;
        u64 i = (idx + a.shape[0]) % a.shape[0];
        auto v = a.view(i);
        this->activation.copy(v, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        // always_assert(idx < shape[0]);
        shape.erase(shape.begin());
        return shape;
    }
};


template <typename T>
class Transpose: public Layer<T> {
public:
    std::vector<u64> perm;
    Transpose(const std::vector<u64> &perm) : Layer<T>("Transpose"), perm(perm) {}

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() >= 2);
    }

    void _forward(Tensor<T> &a) {
        if (a.shape.size() == 2)
        {
#pragma omp parallel for collapse(2)
            for (u64 i = 0; i < a.shape[0]; ++i)
            {
                for (u64 j = 0; j < a.shape[1]; ++j)
                {
                    this->activation.data[j * a.shape[perm[1]] + i] = a.data[i * a.shape[1] + j];
                }
            }
        }
        else if (a.shape.size() == 4)
        {
            auto a_4d = a.as_4d();
            auto out_4d = this->activation.as_4d();
#pragma omp parallel for collapse(4)
            for (int n = 0; n < a.shape[0]; ++n)
            {
                for (int h = 0; h < a.shape[1]; ++h)
                {
                    for (int w = 0; w < a.shape[2]; ++w)
                    {
                        for (int c = 0; c < a.shape[3]; ++c)
                        {
                            out_4d(perm[0] == 0 ? n : (perm[0] == 1 ? h : (perm[0] == 2 ? w : c)),
                                   perm[1] == 0 ? n : (perm[1] == 1 ? h : (perm[1] == 2 ? w : c)),
                                   perm[2] == 0 ? n : (perm[2] == 1 ? h : (perm[2] == 2 ? w : c)),
                                   perm[3] == 0 ? n : (perm[3] == 1 ? h : (perm[3] == 2 ? w : c))) = a_4d(n, h, w, c);
                        }
                    }
                }
            }
        }
        else if (a.shape.size() == 5)
        {
            auto a_5d = a.as_5d();
            auto out_5d = this->activation.as_5d();
#pragma omp parallel for collapse(5)
            for (int n = 0; n < a.shape[0]; ++n)
            {
                for (int h = 0; h < a.shape[1]; ++h)
                {
                    for (int w = 0; w < a.shape[2]; ++w)
                    {
                        for (int d = 0; d < a.shape[3]; ++d)
                        {
                            for (int c = 0; c < a.shape[4]; ++c)
                            {
                                out_5d(perm[0] == 0 ? n : (perm[0] == 1 ? h : (perm[0] == 2 ? w : (perm[0] == 3 ? d : c))),
                                       perm[1] == 0 ? n : (perm[1] == 1 ? h : (perm[1] == 2 ? w : (perm[1] == 3 ? d : c))),
                                       perm[2] == 0 ? n : (perm[2] == 1 ? h : (perm[2] == 2 ? w : (perm[2] == 3 ? d : c))),
                                       perm[3] == 0 ? n : (perm[3] == 1 ? h : (perm[3] == 2 ? w : (perm[3] == 3 ? d : c))),
                                       perm[4] == 0 ? n : (perm[4] == 1 ? h : (perm[4] == 2 ? w : (perm[4] == 3 ? d : c)))) = a_5d(n, h, w, d, c);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            throw std::runtime_error("supported only 2d, 4d, 5d tensors in transpose");
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        always_assert(perm.size() == shape.size());
        for (auto &p : perm)
        {
            if (p == 1)
                p = shape.size() - 1;
            else if (p > 1)
                p -= 1;
        }
        std::vector<u64> outShape;
        for (auto &p : perm)
        {
            outShape.push_back(shape[p]);
        }
        return outShape;
    }
};

template <typename T>
class _MatMul: public Layer<T> {
public:
    _MatMul() :  Layer<T>("_MatMul") {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape0[1] == shape1[0]);
    }

    void _forward(Tensor<T> &a) {
        throw std::runtime_error("single input not allowed in matmul");
    }

    void _forward(std::vector<Tensor<T> *> &a) {
        always_assert(a.size() == 2);
        auto &a0 = *a[0];
        auto a0_2d = a0.as_2d();
        auto &a1 = *a[1];
        auto a1_2d = a1.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->matmul(a0_2d, a1_2d, act_2d);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape0[1] == shape1[0]);
        return {shape0[0], shape1[1]};
    }
};


template <typename T>
class _ScalarMul: public Layer<T> {
public:
    double scalar;

    _ScalarMul(double scalar) :  Layer<T>("_ScalarMul"), scalar(scalar) {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a) {
        T scalarFix = scalar * (1LL << this->scale);
        this->backend->scalarmul(a, scalarFix, this->activation);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &shape0 = inShapes[0];
        return shape0;
    }
};
