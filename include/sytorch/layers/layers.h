#pragma once
#include <sytorch/utils.h>
#include <llama/assert.h>
#include <sytorch/backend/cleartext.h>
#include <string>

struct layer_dims {
    u64 n, h, w, c;

    u64 size() {
        return n * h * w * c;
    }
};

template <typename T>
class Layer {
public:
    std::string name;
    Tensor4D<T> activation;
    Tensor4D<T> inputDerivative;
    bool doTruncationForward = false;
    bool doPreSignExtension = false;
    bool doPostSignExtension = false;
    bool isFirst = false;
    u64 scale = 0;
    Backend<T> *backend = nullptr;
    static bool fakeExecution;
    int mode = 0; // only used in ReLU in llama improved to decide between relu and reluext, might need something cleaner?
    int forwardTruncationMode = 0;
    LayerGraphNode<T> *node = nullptr;

    Layer(const std::string &id) : activation(0,0,0,0), inputDerivative(0,0,0,0), name(id) {
        backend = new ClearText<T>();
    }
    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        initScale(scale);
        resize(d1, d2, d3, d4);
    }
    virtual void initScale(u64 scale) {};
    virtual void resize(u64 d1, u64 d2, u64 d3, u64 d4) = 0;
    virtual void forward_internal(Tensor4D<T> &a, bool train = true) = 0;
    Tensor4D<T>& forward(Tensor4D<T> &a, bool train = true) {
        if (fakeExecution) {
            activation.graphNode = new LayerGraphNode<T>();
            node = activation.graphNode;
            activation.graphNode->layer = this;
            activation.graphNode->parents.push_back(a.graphNode);
            a.graphNode->children.push_back(activation.graphNode);
            layer_dims indims = {a.d1, a.d2, a.d3, a.d4};
            layer_dims outdims = this->get_output_dims(indims);
            activation.resize(outdims.n, outdims.h, outdims.w, outdims.c);
            inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
            return activation;
        }
        if (a.d1 != inputDerivative.d1 || a.d2 != inputDerivative.d2 || a.d3 != inputDerivative.d3 || a.d4 != inputDerivative.d4) {
            resize(a.d1, a.d2, a.d3, a.d4);
        }
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
            // if (gcHappened) {
            //     std::cerr << "Output of " << a.graphNode->layer->name << " cleared" << std::endl;
            // }
        }
        return activation;
    }
    virtual void backward(const Tensor4D<T> &e) = 0;
    virtual Tensor2D<T>& getweights() { throw std::runtime_error("not implemented"); };
    virtual Tensor<T>& getbias() { throw std::runtime_error("not implemented"); };
    virtual struct layer_dims get_output_dims(struct layer_dims &in) = 0;
    virtual void setBackend(Backend<T> *b) {
        backend = b;
    }
};

template <typename T>
class Conv2D : public Layer<T> {
public:
    Tensor4D<T> inp;
    Tensor2D<T> filter;
    Tensor2D<T> filterGrad;
    Tensor2D<T> Vw;
    Tensor<T> bias;
    Tensor<T> biasGrad;
    Tensor<T> Vb;
    u64 ci, co, ks, padding, stride;

    Conv2D(u64 ci, u64 co, u64 ks, u64 padding = 0, u64 stride = 1) : Layer<T>("Conv2D"), ci(ci), co(co), ks(ks), padding(padding), 
        stride(stride), filter(co, ks * ks * ci), filterGrad(co, ks * ks * ci), Vw(co, ks * ks * ci), bias(co), biasGrad(co), Vb(co), inp(0,0,0,0)
    {
        this->doTruncationForward = true;
    }

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
        double xavier = 1.0 / sqrt(ci * ks * ks);
        filter.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
        Vw.fill(0);
        Vb.fill(0);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        always_assert(d4 == ci);
        this->inputDerivative.resize(d1, d2, d3, d4);
        inp.resize(d1, d2, d3, d4);
        u64 newH = (((d2 + 2*padding - ks)/stride) + 1);
        u64 newW = (((d3 + 2*padding - ks)/stride) + 1);
        this->activation.resize(d1, newH, newW, co);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        assert(a.d4 == ci);
        inp.copy(a);
        this->backend->conv2D(ks, ks, padding, stride, ci, co, a, filter, this->activation);
        this->activation.addBias(bias);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        this->backend->conv2DFilterGrad(ks, ks, padding, stride, ci, co, inp, filterGrad, e);
        this->backend->conv2DBiasGrad(e, biasGrad);
        if (!(this->isFirst)) {
            this->backend->conv2DInputGrad(ks, ks, padding, stride, ci, co, this->inputDerivative, filter, e);
            this->backend->truncate(this->inputDerivative, this->scale);
        }
        // this->backend->truncate(filterGrad, scale);
        this->backend->updateWeight(filter, filterGrad, Vw, this->scale);
        this->backend->updateBias(bias, biasGrad, Vb, this->scale);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor<T>& getbias() { return bias; }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        u64 newH = (((in.h + 2*padding - ks)/stride) + 1);
        u64 newW = (((in.w + 2*padding - ks)/stride) + 1);
        return {in.n, newH, newW, co};
    }
};

template <typename T>
class AvgPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;

    AvgPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("AvgPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride) {}

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, (d2 + 2*padding - ks)/stride + 1, (d3 + 2*padding - ks)/stride + 1, d4);
    }
    
    void forward_internal(Tensor4D<T> &a, bool train = true) {
        this->backend->avgPool2D(ks, padding, stride, a, this->activation, this->scale);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        this->backend->avgPool2DInputGrad(ks, padding, stride, this->inputDerivative, e, this->scale);
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        u64 newH = (((in.h + 2*padding - ks)/stride) + 1);
        u64 newW = (((in.w + 2*padding - ks)/stride) + 1);
        return {in.n, newH, newW, in.c};
    }
};

template <typename T>
class SumPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;

    SumPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("SumPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride) {}

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
    }
    
    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, (d2 + 2*padding - ks)/stride + 1, (d3 + 2*padding - ks)/stride + 1, d4);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        this->backend->sumPool2D(ks, padding, stride, a, this->activation);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        this->backend->sumPool2DInputGrad(ks, padding, stride, this->inputDerivative, e);
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        u64 newH = (((in.h + 2*padding - ks)/stride) + 1);
        u64 newW = (((in.w + 2*padding - ks)/stride) + 1);
        return {in.n, newH, newW, in.c};
    }
};

template <typename T>
class MaxPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;
    Tensor4D<u64> maxIndex;

    MaxPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("MaxPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride), maxIndex(0,0,0,0) {}

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, (d2 + 2*padding - ks)/stride + 1, (d3 + 2*padding - ks)/stride + 1, d4);
        this->maxIndex.resize(d1, (d2 + 2*padding - ks)/stride + 1, (d3 + 2*padding - ks)/stride + 1, d4);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        this->backend->maxPool2D(ks, padding, stride, a, this->activation, maxIndex, this->scale, this->mode);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        this->backend->maxPool2DInputGrad(ks, padding, stride, this->inputDerivative, e, maxIndex);
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        u64 newH = (((in.h + 2*padding - ks)/stride) + 1);
        u64 newW = (((in.w + 2*padding - ks)/stride) + 1);
        return {in.n, newH, newW, in.c};
    }
};

template <typename T>
class Flatten : public Layer<T> {
public:
    u64 d1, d2, d3, d4;

    Flatten() : Layer<T>("Flatten") {}

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->d1 = d1;
        this->d2 = d2;
        this->d3 = d3;
        this->d4 = d4;
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, d2 * d3 * d4, 1, 1);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        // this->activation(i, j * d3 * d4 + k * d4 + l, 0, 0) = a(i, j, k, l);
                        this->activation(i, l * d2 * d3 + j * d3 + k, 0, 0) = a(i, j, k, l);
                    }
                }
            }
        }
    }

    void backward(const Tensor4D<T> &e) {
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        this->inputDerivative(i, j, k, l) = e(i, l * d2 * d3 + j * d3 + k, 0, 0);
                    }
                }
            }
        }
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        return {in.n, in.h * in.w * in.c, 1, 1};
    }
};

template <typename T>
class FC : public Layer<T> {
public:
    Tensor4D<T> inp;
    Tensor2D<T> weight;
    Tensor2D<T> weightGrad;
    Tensor2D<T> Vw;
    Tensor<T> bias;
    Tensor<T> Vb;
    u64 in, out;

    FC(u64 in, u64 out) : Layer<T>("FC"), in(in), out(out), weight(in, out), bias(out), inp(0,0,0,0), weightGrad(in,out), Vw(in,out), Vb(out) {
        this->doTruncationForward = true;
    }

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
        double xavier = 1.0 / sqrt(in);
        weight.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
        Vw.fill(0);
        Vb.fill(0);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        always_assert(d2 == in);
        always_assert((d3 == 1) && (d4 == 1));
        inp.resize(d1, in, 1, 1);
        this->inputDerivative.resize(d1, in, 1, 1);
        this->activation.resize(d1, out, 1, 1);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        this->inp.copy(a);
        this->backend->matmul(a, weight, this->activation);
        this->activation.addBias2D(bias);
    }

    void backward(const Tensor4D<T> &e) {
        if (!(this->isFirst)) {
            this->backend->matmulTransposeB(e, weight, this->inputDerivative);
            this->backend->truncate(this->inputDerivative, this->scale);
        }
        this->backend->matmulTransposeA(inp, e, weightGrad);
        // this->backend->truncate(weightGrad, scale);
        Vw.resize(weightGrad.d1, weightGrad.d2);
        this->backend->updateWeight(weight, weightGrad, Vw, this->scale);
        this->backend->updateBias(bias, e, Vb, this->scale);
    }
    
    Tensor2D<T>& getweights() { return weight; }
    Tensor<T>& getbias() { return bias; }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        return {in.n, out, 1, 1};
    }
};

template <typename T>
class ReLU: public Layer<T> {
public:
    Tensor4D<T> drelu;
    ReLU() :  Layer<T>("ReLU"), drelu(0,0,0,0) {}

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, d2, d3, d4);
        this->drelu.resize(d1, d2, d3, d4);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        // std::cout << "== ReLU forward ==" << std::endl;
        // std::cout << "a: "; a.print();
        this->backend->relu(a, this->activation, this->drelu, this->scale, this->mode);
    }

    void backward(const Tensor4D<T> &e) {
        // std::cout << "== ReLU backward ==" << std::endl;
        // std::cout << "e: "; e.print();
        this->backend->select(e, this->drelu, this->inputDerivative);
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        return {in.n, in.h, in.w, in.c};
    }
};

template <typename T>
class BatchNorm2d : public Layer<T> {
public:
    Tensor<T> running_mean;
    Tensor<T> running_variance;
    Tensor<T> gamma;
    Tensor<T> beta;
    Tensor<T> Vgamma;
    Tensor<T> Vbeta;

    Tensor4D<T> x_normalized;
    Tensor<T> invstd;

    // Momentum is 0.1 and epsilon is 1e-5
    BatchNorm2d(u64 channels) : Layer<T>("BatchNorm2d"), running_mean(channels), running_variance(channels), gamma(channels), beta(channels),
    x_normalized(0, 0, 0, 0), invstd(channels), Vgamma(channels), Vbeta(channels) {
        this->running_mean.fill(0);
        this->beta.fill(0);
        this->Vgamma.fill(0);
        this->Vbeta.fill(0);
    }

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
        this->running_variance.fill(1ULL << scale);
        this->gamma.fill(1ULL << scale);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, d2, d3, d4);
        x_normalized.resize(d1, d2, d3, d4);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        assert(a.d4 == this->running_mean.size);
        if (train) {
            this->backend->batchNorm2dForwardTrain(a, this->activation, this->running_mean, this->running_variance, this->gamma, 
                this->beta, this->x_normalized, this->invstd, this->scale);
        }
        else {
            this->backend->batchNorm2dForwardTest(a, this->activation, this->running_mean, this->running_variance, this->gamma, 
                this->beta, this->scale);
        }
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d4 == this->running_mean.size);
        const u64 C = e.d4;
        Tensor<T> dgamma(C);
        Tensor<T> dbeta(C);
        this->backend->batchNorm2dBackward(this->inputDerivative, e, dgamma, dbeta, this->x_normalized, gamma, invstd, this->scale);
        // dgamma.print(scale);
        // dbeta.print(2*scale);
        // this->gamma.print(scale);
        this->backend->updateWeight(this->gamma, dgamma, Vgamma, this->scale);
        // this->gamma.print(scale);
        // this->beta.print(2*scale);
        this->backend->updateBias(this->beta, dbeta, Vbeta, this->scale);
        // this->beta.print(2*scale);
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        return {in.n, in.h, in.w, in.c};
    }
};

template <typename T>
class BatchNorm2dInference : public Layer<T> {
public:
    Tensor<T> A; // scale = s
    Tensor<T> B; // scale = 2s

    BatchNorm2dInference(u64 channels) : Layer<T>("BatchNorm2dInference"), A(channels), B(channels) {
        this->A.fill(0);
        this->B.fill(0);
        this->doTruncationForward = true;
    }

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, d2, d3, d4);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        assert(a.d4 == this->A.size);
        if (train) {
            std::runtime_error("BatchNorm2dInference should not be used in training mode");
        }
        else {
            this->backend->batchNorm2dInference(this->A, this->B, a, this->activation, this->scale);
        }
    }

    void backward(const Tensor4D<T> &e) {
        std::runtime_error("BatchNorm2dInference should not be used in training mode");
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        return {in.n, in.h, in.w, in.c};
    }
};

template <typename T>
class Identity: public Layer<T> {
public:
    Identity() :  Layer<T>("Identity") {}

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->activation.resize(d1, d2, d3, d4);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        this->activation.copy(a);
    }

    void backward(const Tensor4D<T> &e) {
        this->inputDerivative.copy(e);
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        return {in.n, in.h, in.w, in.c};
    }
};

template <typename T>
class PlaceHolderLayer : public Layer<T> {
public:
    PlaceHolderLayer(const std::string &s) : Layer<T>(s) {
    }

    void initScale(u64 scale) {
        std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    void backward(const Tensor4D<T> &e) {
        std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
        return {0, 0, 0, 0};
    }
};

template <typename T>
bool Layer<T>::fakeExecution = false;