#pragma once
#include <sytorch/utils.h>
#include <llama/assert.h>
#include <sytorch/backend/cleartext.h>
#include <string>

template <typename T>
class Layer {
public:
    std::string name;
    std::vector<u64> currentInputShape;
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

    LayerGraphNode<T> *node = nullptr;

    Layer(const std::string &name) : activation({0}), name(name) {
        backend = new ClearText<T>();
    }

    void init(const std::vector<u64> &shape, u64 scale) {
        initScale(scale);
        resize(shape);
    }

    virtual void _initScale(u64 scale) {};
    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
        _initScale(scale);
    };
    
    virtual void _resize(const std::vector<u64> &shape) {};
    void resize(const std::vector<u64> &shape) {
        currentInputShape = shape;
        auto outdims = this->get_output_dims(shape);
        activation.resize(outdims);
        _resize(shape);
    };

    virtual void forward_internal(Tensor<T> &a, bool train = true) = 0;
    
    Tensor<T>& forward(Tensor<T> &a, bool train = true) {
        if (a.graphGenMode) {
            node = new LayerGraphNode<T>();
            auto parentNode = a.graphNode;
            node->layer = this;
            
            node->allNodesInExecutionOrderRef = parentNode->allNodesInExecutionOrderRef;
            node->allNodesInExecutionOrderRef->push_back(node);
            
            node->parents.push_back(parentNode);
            parentNode->children.push_back(node);

            activation.graphNode = node;
            activation.graphGenMode = true;
            return activation;
        }

        activation.graphGenMode = false;
        resize(a.shape);
        if (node != nullptr) {
            node->currTensor = &activation;
            activation.graphNode = node;
        }
        if (doPreSignExtension) {
            this->backend->signext(a, scale);
        }
        forward_internal(a, train);
        if (doTruncationForward) {
            this->backend->truncateForward(activation, scale, forwardTruncationMode);
        }
        if (doPostSignExtension) {
            this->backend->signext(activation, scale);
        }
        if (a.graphNode != nullptr) {
            bool gcHappened = a.graphNode->incrementAndGc();
        }
        return activation;
    }

    virtual Tensor2D<T>& getweights() { throw std::runtime_error("not implemented"); };
    virtual Tensor1D<T>& getbias() { throw std::runtime_error("not implemented"); };
    virtual std::vector<u64> get_output_dims(const std::vector<u64> &inShape) = 0;

    virtual void setBackend(Backend<T> *b) {
        backend = b;
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 4);
        always_assert(shape[3] == ci);
        inp.resize(shape);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        always_assert(a.shape.size() == 4);
        assert(a.shape[3] == ci);
        inp.copy(a);
        auto act_4d = this->activation.as_4d();
        this->backend->conv2D(fh, fw, padding, stride, ci, co, a.as_4d(), filter, act_4d);
        if (this->useBias)
            this->activation.as_4d().addBias(bias);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor1D<T>& getbias() { return bias; }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 5);
        always_assert(shape[4] == ci);
        inp.resize(shape);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        always_assert(a.shape.size() == 5);
        assert(a.shape[4] == ci);
        inp.copy(a);
        auto act_5d = this->activation.as_5d();
        this->backend->conv3D(fd, fh, fw, padding, stride, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->activation.addBias(bias);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor1D<T>& getbias() { return bias; }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 4);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        always_assert(a.shape.size() == 4);
        this->backend->avgPool2D(ks, padding, stride, a.as_4d(), this->activation.as_4d(), this->scale.as_4d());
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 4);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        this->backend->sumPool2D(ks, padding, stride, a.as_4d(), this->activation.as_4d());
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 4);
        this->maxIndex.resize(this->activation.shape);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        auto a_4d = a.as_4d();
        auto act_4d = this->activation.as_4d();
        this->backend->maxPool2D(ks, padding, stride, a_4d, act_4d, maxIndex, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
        always_assert(inShape.size() == 4);
        u64 newH = (((inShape[1] + 2*padding - ks)/stride) + 1);
        u64 newW = (((inShape[2] + 2*padding - ks)/stride) + 1);
        return {inShape[0], newH, newW, inShape[3]};
    }
};

template <typename T>
class Flatten : public Layer<T> {
public:

    Flatten() : Layer<T>("Flatten") {}

    void forward_internal(Tensor<T> &a, bool train = true) {
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

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 2);
        always_assert(shape[1] == in);
        inp.resize(shape);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        this->inp.copy(a);
        auto a_2d = a.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->matmul(a_2d, weight, act_2d);
        if (this->useBias)
            this->activation.as_2d().addBias2D(bias);
    }

    Tensor2D<T>& getweights() { return weight; }
    Tensor1D<T>& getbias() { return bias; }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void _resize(const std::vector<u64> &shape) {
        this->drelu.resize(shape);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        this->backend->relu(a, this->activation, this->drelu, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
        return inShape;
    }
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 4);
        always_assert(shape[3] == this->A.size);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        always_assert(a.shape.size() == 4);
        assert(a.shape[3] == this->A.size);
        if (train) {
            std::runtime_error("BatchNorm2dInference should not be used in training mode");
        }
        else {
            this->backend->batchNorm2dInference(this->A, this->B, a.as_4d(), this->activation.as_4d(), this->scale);
        }
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
        always_assert(inShape.size() == 4);
        always_assert(inShape[3] == this->A.size);
        return inShape;
    }
};

template <typename T>
class Identity: public Layer<T> {
public:
    Identity() :  Layer<T>("Identity") {}

    void forward_internal(Tensor<T> &a, bool train = true) {
        this->activation.copy(a);
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
        return inShape;
    }
};

template <typename T>
class GlobalAvgPool2D : public Layer<T> {
public:

    GlobalAvgPool2D() : Layer<T>("GlobalAvgPool2D") {}

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 4);
        always_assert(shape[1] == shape[2]);
    }
    
    void forward_internal(Tensor<T> &a, bool train = true) {
        auto a_4d = a.as_4d();
        auto act_4d = this->activation.as_4d();
        this->backend->avgPool2D(a_4d.d2, 0, 1, a_4d, act_4d, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void _resize(const std::vector<u64> &shape) {
        always_assert(shape.size() == 5);
        always_assert(shape[4] == ci);
    }

    void forward_internal(Tensor<T> &a, bool train = true) {
        always_assert(a.shape.size() == 5);
        assert(a.shape[4] == ci);
        auto act_5d = this->activation.as_5d();
        this->backend->convTranspose3D(fd, fh, fw, padding, stride, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->activation.addBias(bias);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor1D<T>& getbias() { return bias; }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
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

    void forward_internal(Tensor<T> &a, bool train = true) {
        throw std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    std::vector<u64> get_output_dims(const std::vector<u64> &inShape) {
        return inShape;
    }
};
