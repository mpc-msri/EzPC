#pragma once
#include <sytorch/utils.h>
#include <llama/assert.h>
#include <sytorch/backend/cleartext.h>
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

    LayerGraphNode<T> *node = nullptr;

    Layer(const std::string &name) : activation({0}), name(name) {
        backend = new ClearText<T>();
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

    virtual Tensor2D<T>& getweights() { throw std::runtime_error("not implemented"); };
    virtual Tensor1D<T>& getbias() { throw std::runtime_error("not implemented"); };
    virtual Tensor<T> &getinput2() { throw std::runtime_error("not implemented"); };
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
            inp.copy(a);
        auto act_4d = this->activation.as_4d();
        this->backend->conv2D(fh, fw, padding, stride, ci, co, a.as_4d(), filter, act_4d);
        if (this->useBias)
            this->activation.as_4d().addBias(bias);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor1D<T>& getbias() { return bias; }

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
    u64 fd, fh, fw, padding, stride;

    Conv3D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f), fh(f), fw(f), 
        padding(padding), stride(stride), filter(co, f * f * f * ci), bias(co), inp({0,0,0,0,0})
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv3D(u64 ci, u64 co, const std::array<u64, 3> f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]), 
        padding(padding), stride(stride), filter(co, f[0] * f[1] * f[2] * ci), bias(co), inp({0,0,0,0,0})
    {
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
            inp.copy(a);
        auto act_5d = this->activation.as_5d();
        this->backend->conv3D(fd, fh, fw, padding, stride, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->activation.addBias(bias);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor1D<T>& getbias() { return bias; }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 5);
        always_assert(inShape[4] == ci);
        u64 newD = (((inShape[1] + 2*padding - fd)/stride) + 1);
        u64 newH = (((inShape[2] + 2*padding - fh)/stride) + 1);
        u64 newW = (((inShape[3] + 2*padding - fw)/stride) + 1);
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
        this->backend->avgPool2D(ks, padding, stride, a.as_4d(), this->activation.as_4d(), this->scale.as_4d());
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
        this->inp.copy(a);
        auto a_2d = a.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->matmul(a_2d, weight, act_2d);
        if (this->useBias)
            this->activation.as_2d().addBias2D(bias);
    }

    Tensor2D<T>& getweights() { return weight; }
    Tensor1D<T>& getbias() { return bias; }

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
class Sqrt : public Layer<T>
{
public:
    Tensor<T> dsqrt;
    Sqrt() : Layer<T>("Sqrt"), dsqrt({0}) {}

    void _resize(const std::vector<u64> &shape)
    {
        this->dsqrt.resize(shape);
    }

    void forward_internal(Tensor<T> &a, bool train = true)
    {
        this->backend->sqrt(a, this->activation, this->dsqrt, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape)
    {
        return inShape;
    }
};

template <typename T>
class Pow : public Layer<T>
{
public:
    Tensor<T> dpow;
    Tensor<T> exp;
    std::vector<u64> out_shape;
    Pow(const std::vector<u64> out_shape, const std::vector<u64> exp_shape) : Layer<T>("Pow"), dpow({0}), out_shape(out_shape), exp(exp_shape) {}

    void _resize(const std::vector<u64> &shape)
    {
        this->dpow.resize(shape);
    }

    void forward_internal(Tensor<T> &a, bool train = true)
    {
        this->backend->pow(a, this->exp, this->activation, this->dpow, this->scale, this->out_shape);
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape)
    {
        return inShape;
    }

    Tensor<T> &getinput2() { return exp; }
};

template <typename T>
class BatchNorm2dInference : public Layer<T> {
public:
    Tensor1D<T> A; // scale = s
    Tensor1D<T> B; // scale = 2s

    BatchNorm2dInference(u64 channels) : Layer<T>("BatchNorm2dInference"), A(channels), B(channels) {
        this->A.fill(0);
        this->B.fill(0);
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes) {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        always_assert(shape[3] == this->A.size);
    }

    void _forward(Tensor<T> &a) {
        always_assert(a.shape.size() == 4);
        assert(a.shape[3] == this->A.size);
        if (this->isTrainingMode) {
            std::runtime_error("BatchNorm2dInference should not be used in training mode");
        }
        else {
            this->backend->batchNorm2dInference(this->A, this->B, a.as_4d(), this->activation.as_4d(), this->scale);
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        always_assert(inShape[3] == this->A.size);
        return inShape;
    }
};

template <typename T>
class Identity: public Layer<T> {
public:
    Identity() :  Layer<T>("Identity") {}

    void _forward(Tensor<T> &a) {
        this->activation.copy(a);
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
    u64 fd, fh, fw, padding, stride;

    ConvTranspose3D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f), fh(f), fw(f), 
        padding(padding), stride(stride), filter(co, f * f * f * ci), bias(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    ConvTranspose3D(u64 ci, u64 co, const std::array<u64, 3> f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]), 
        padding(padding), stride(stride), filter(co, f[0] * f[1] * f[2] * ci), bias(co)
    {
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
        this->backend->convTranspose3D(fd, fh, fw, padding, stride, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->activation.addBias(bias);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor1D<T>& getbias() { return bias; }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 5);
        always_assert(inShape[4] == ci);
        u64 newD = (((inShape[1] - 1)*stride + fd - 2*padding));
        u64 newH = (((inShape[2] - 1)*stride + fh - 2*padding));
        u64 newW = (((inShape[3] - 1)*stride + fw - 2*padding));
        return {inShape[0], newD, newH, newW, co};
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

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) {
        auto &shape0 = inShapes[0];
        for (auto &shape : inShapes) {
            always_assert(shape.size() == shape0.size());
            for (u64 i = 0; i < shape.size(); i++) {
                always_assert(shape[i] == shape0[i]);
            }
        }
        auto &inShape = inShapes[0];
        return inShape;
    }
};
