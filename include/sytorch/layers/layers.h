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

    Layer(const std::string &id) : activation(0,0,0,0), inputDerivative(0,0,0,0), name(id) {
        backend = new ClearText<T>();
    }
    virtual void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) = 0;
    virtual void resize(u64 d1, u64 d2, u64 d3, u64 d4) = 0;
    virtual void forward_internal(Tensor4D<T> &a, bool train = true) = 0;
    Tensor4D<T>& forward(Tensor4D<T> &a, bool train = true) {
        if (fakeExecution) {
            activation.graphNode->layer = this;
            activation.graphNode->parents.push_back(a.graphNode);
            a.graphNode->children.push_back(activation.graphNode);
            return activation;
        }
        if (a.d1 != inputDerivative.d1 || a.d2 != inputDerivative.d2 || a.d3 != inputDerivative.d3 || a.d4 != inputDerivative.d4) {
            resize(a.d1, a.d2, a.d3, a.d4);
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

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale)
    {
        always_assert(std::is_integral<T>::value || scale == 0);
        always_assert(d4 == ci);
        this->scale = scale;
        double xavier = 1.0 / sqrt(ci * ks * ks);
        filter.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
        Vw.fill(0);
        Vb.fill(0);
        resize(d1, d2, d3, d4);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
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

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        resize(d1, d2, d3, d4);
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

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        resize(d1, d2, d3, d4);
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

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        resize(d1, d2, d3, d4);
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
    
    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        resize(d1, d2, d3, d4);
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

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        always_assert(d2 == in);
        always_assert((d3 == 1) && (d4 == 1));
        this->scale = scale;
        double xavier = 1.0 / sqrt(in);
        weight.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
        Vw.fill(0);
        Vb.fill(0);
        resize(d1, d2, d3, d4);
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

// template <typename T>
// class ReLUTruncate: public Layer<T> {
// public:
//     u64 shift;
//     Tensor4D<T> drelu;
//     ReLUTruncate(u64 shift) : Layer<T>("ReLUTruncate"), shift(shift), drelu(0,0,0,0) {}

//     void forward_internal(Tensor4D<T> &a, bool train = true) {
//         // std::cout << "== Truncate forward ==" << std::endl;
//         // std::cout << "a: "; a.print();
//         this->activation.resize(a.d1, a.d2, a.d3, a.d4);
//         this->drelu.resize(a.d1, a.d2, a.d3, a.d4);
//         this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
//         this->backend->relutruncate(a, this->activation, this->drelu, shift);
//     }

//     void backward(const Tensor4D<T> &e) {
//         // std::cout << "== ReLU backward ==" << std::endl;
//         // std::cout << "e: "; e.print();
//         this->backend->select(e, this->drelu, this->inputDerivative);
//         // std::cout << "== Truncate ==" << std::endl;
//         // std::cout << "e: "; this->inputDerivative.print();
//         // truncate(this->inputDerivative, this->inputDerivative, shift);
//     }

//     struct layer_dims get_output_dims(struct layer_dims &in) {
//         return {in.n, in.h, in.w, in.c};
//     }
// };

template <typename T>
class ReLU: public Layer<T> {
public:
    Tensor4D<T> drelu;
    ReLU() :  Layer<T>("ReLU"), drelu(0,0,0,0) {}

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        resize(d1, d2, d3, d4);
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

// template <typename T>
// class Truncate: public Layer<T> {
// public:
//     u64 shift;
//     Truncate(u64 shift) : Layer<T>("Truncate"), shift(shift) {}

//     void forward_internal(Tensor4D<T> &a, bool train = true) {
//         this->activation.resize(a.d1, a.d2, a.d3, a.d4);
//         this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
//         this->backend->truncate(a, this->activation, shift);
//     }

//     void backward(const Tensor4D<T> &e) {
//         // this->backend->truncate(e, this->inputDerivative, shift);
//         this->inputDerivative.copy(e);
//     }

//     struct layer_dims get_output_dims(struct layer_dims &in) {
//         return {in.n, in.h, in.w, in.c};
//     }
// };

template <typename T>
class Sequential : public Layer<T> {
public:
    std::vector<Layer<T>*> layers;

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        this->inputDerivative.resize(d1, d2, d3, d4);
        for(auto &l : layers) {
            l->init(d1, d2, d3, d4, scale);
            d1 = l->activation.d1;
            d2 = l->activation.d2;
            d3 = l->activation.d3;
            d4 = l->activation.d4;
        }
        this->activation.resize(d1, d2, d3, d4);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        for(auto &l : layers) {
            l->resize(d1, d2, d3, d4);
            d1 = l->activation.d1;
            d2 = l->activation.d2;
            d3 = l->activation.d3;
            d4 = l->activation.d4;
        }
        this->activation.resize(d1, d2, d3, d4);
    }

    void optimize(bool outermost)
    {
        int s = layers.size();
        // Set isFirst
        if (outermost)
        for(int i = 0; i < s; ++i) {
            if (layers[i]->name == "Conv2D" || layers[i]->name == "FC") {
                layers[i]->isFirst = true;
                break;
            }
        }
        
        // Optimization: ReLU-MaxPool
        for(int i = 0; i < s - 1; i++) {
            if (layers[i+1]->name == "MaxPool2D") {
                auto &n = layers[i]->name;
                if (n == "ReLU" || n == "ReLUTruncate" || n == "Truncate") {
                    std::swap(layers[i], layers[i+1]);
                }
            }
        }

    }

    Sequential(std::vector<Layer<T>*> _layers, bool outermost = true) : Layer<T>("Sequential"), layers(_layers) {
        optimize(outermost);
    }
    
    void forward_internal(Tensor4D<T> &a, bool train = true) {
        layers[0]->forward(a, train);
        u64 size = layers.size();
        for(u64 i = 1; i < size; i++) {
            layers[i]->forward(layers[i-1]->activation, train);
        }
        this->activation.copy(layers[size-1]->activation);

    }
    void backward(const Tensor4D<T> &e) {
        int size = layers.size();
        layers[size-1]->backward(e);
        for (int i = size - 2; i >= 0; i--) {
            layers[i]->backward(layers[i+1]->inputDerivative);
        }
        this->inputDerivative.copy(layers[0]->inputDerivative);
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        struct layer_dims res = in;
        u64 size = layers.size();
        for(u64 i = 0; i < size; i++) {
            res = layers[i]->get_output_dims(res);
        }
        return res;
    }

    virtual void setBackend(Backend<T> *b) {
        this->backend = b;
        for(auto &l : layers) {
            l->setBackend(b);
        }
    }
};

template <typename T>
class Identity: public Layer<T> {
public:
    Identity() :  Layer<T>("Identity") {}
    
    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        resize(d1, d2, d3, d4);
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
class BranchAdd : public Layer<T> {
public:
    Layer<T>* left;
    Layer<T>* right;

    BranchAdd(Layer<T>* left, Layer<T>* right) : Layer<T>("BranchAdd"), left(left), right(right) {
        
    }

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        this->inputDerivative.resize(d1, d2, d3, d4);
        left->init(d1, d2, d3, d4, scale);
        right->init(d1, d2, d3, d4, scale);
        always_assert(left->activation.d1 == right->activation.d1);
        always_assert(left->activation.d2 == right->activation.d2);
        always_assert(left->activation.d3 == right->activation.d3);
        always_assert(left->activation.d4 == right->activation.d4);
        this->activation.resize(left->activation.d1, left->activation.d2, left->activation.d3, left->activation.d4);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        left->resize(d1, d2, d3, d4);
        right->resize(d1, d2, d3, d4);
        always_assert(left->activation.d1 == right->activation.d1);
        always_assert(left->activation.d2 == right->activation.d2);
        always_assert(left->activation.d3 == right->activation.d3);
        always_assert(left->activation.d4 == right->activation.d4);
        this->activation.resize(left->activation.d1, left->activation.d2, left->activation.d3, left->activation.d4);
    }

    void forward_internal(Tensor4D<T> &a, bool train = true) {
        left->forward(a, train);
        right->forward(a, train);
        for(int i = 0; i < this->activation.d1; ++i) {
            for(int j = 0; j < this->activation.d2; ++j) {
                for(int k = 0; k < this->activation.d3; ++k) {
                    for(int l = 0; l < this->activation.d4; ++l) {
                        this->activation(i, j, k, l) = left->activation(i, j, k, l) + right->activation(i, j, k, l);
                    }
                }
            }
        }
    }

    void backward(const Tensor4D<T> &e) {
        left->backward(e);
        right->backward(e);
        for(int i = 0; i < this->inputDerivative.d1; ++i) {
            for(int j = 0; j < this->inputDerivative.d2; ++j) {
                for(int k = 0; k < this->inputDerivative.d3; ++k) {
                    for(int l = 0; l < this->inputDerivative.d4; ++l) {
                        this->inputDerivative(i, j, k, l) = left->inputDerivative(i, j, k, l) + right->inputDerivative(i, j, k, l);
                    }
                }
            }
        }
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        return left->get_output_dims(in);
    }

    virtual void setBackend(Backend<T> *b) {
        this->backend = b;
        left->setBackend(b);
        right->setBackend(b);
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

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        this->running_variance.fill(1ULL << scale);
        this->gamma.fill(1ULL << scale);
        resize(d1, d2, d3, d4);
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

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
        this->scale = scale;
        resize(d1, d2, d3, d4);
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
class PlaceHolderLayer : public Layer<T> {
public:
    PlaceHolderLayer(const std::string &s) : Layer<T>(s) {
    }

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale) {
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
void add(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor4D<T> &c)
{
    if (Layer<T>::fakeExecution) {
        c.graphNode->layer = new PlaceHolderLayer<T>("Add");
        c.graphNode->parents.push_back(a.graphNode);
        c.graphNode->parents.push_back(b.graphNode);
        a.graphNode->children.push_back(c.graphNode);
        b.graphNode->children.push_back(c.graphNode);
        return;
    }

    for (int i = 0; i < a.d1; ++i) {
        for (int j = 0; j < a.d2; ++j) {
            for (int k = 0; k < a.d3; ++k) {
                for (int l = 0; l < a.d4; ++l) {
                    c(i, j, k, l) = a(i, j, k, l) + b(i, j, k, l);
                }
            }
        }
    }
}

template <typename T>
Tensor4D<T> add(const Tensor4D<T> &a, const Tensor4D<T> &b)
{
    Tensor4D<T> c(a.d1, a.d2, a.d3, a.d4);
    add(a, b, c);
    return c;
}

template <typename T>
bool Layer<T>::fakeExecution = false;