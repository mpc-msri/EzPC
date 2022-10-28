#pragma once
#include "tensor.h"
#include "utils.h"
#include "backend/cleartext.h"
#include "backend/llama.h"
#include "backend/llamakey.h"
#include <string>

template <typename T>
#ifdef USE_CLEARTEXT
using DefaultBackend = ClearText<T>;
#else 
#ifdef USE_LLAMA
using DefaultBackend = Llama<T>;
#else
#ifdef USE_LLAMAKEY
using DefaultBackend = LlamaKey<T>;
#else
using DefaultBackend = ClearText<T>;
#endif
#endif
#endif

template <typename T>
class Layer {
public:
    std::string name;
    virtual void forward(const Tensor4D<T> &a) = 0;
    virtual void backward(const Tensor4D<T> &e) = 0;
    virtual void resize(u64 d1, u64 d2, u64 d3, u64 d4) = 0;
    Tensor4D<T> activation;
    Tensor4D<T> inputDerivative;
    Layer(const std::string &id) : activation(0,0,0,0), inputDerivative(0,0,0,0), name(id) {}
    virtual Tensor2D<T>& getweights() { throw std::runtime_error("not implemented"); };
    virtual Tensor<T>& getbias() { throw std::runtime_error("not implemented"); };
    bool isFirst = false;
};

template <typename T, u64 scale, class Backend = DefaultBackend<T>>
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
        static_assert(std::is_integral<T>::value || scale == 0);
        double xavier = 1.0 / sqrt(ci * ks * ks);
        filter.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
        Vw.fill(0);
        Vb.fill(0);
    }

    void forward(const Tensor4D<T> &a) {
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        assert(a.d4 == ci);
        u64 newH = (((a.d2 + 2*padding - ks)/stride) + 1);
        u64 newW = (((a.d3 + 2*padding - ks)/stride) + 1);
        inp.resize(a.d1, a.d2, a.d3, a.d4);
        inp.copy(a);
        this->activation.resize(a.d1, newH, newW, co);
        Backend::conv2D(ks, ks, padding, stride, ci, co, a, filter, this->activation);
        this->activation.addBias(bias);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        Backend::conv2DFilterGrad(ks, ks, padding, stride, ci, co, inp, filterGrad, e);
        Backend::conv2DBiasGrad(e, biasGrad);
        if (!(this->isFirst)) {
            Backend::conv2DInputGrad(ks, ks, padding, stride, ci, co, this->inputDerivative, filter, e);
            Backend::truncate(this->inputDerivative, scale);
        }
        // Backend::truncate(filterGrad, scale);
        Backend::updateWeight(filter, filterGrad, Vw, scale);
        Backend::updateBias(bias, biasGrad, Vb, scale);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        assert(d4 == ci);
        u64 newH = (((d2 + 2*padding - ks)/stride) + 1);
        u64 newW = (((d3 + 2*padding - ks)/stride) + 1);
        this->activation.resize(d1, newH, newW, co);
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->inp.resize(d1, d2, d3, d4);
    }

    Tensor2D<T>& getweights() { return filter; }
    Tensor<T>& getbias() { return bias; }
};

template <typename T, u64 scale, class Backend = DefaultBackend<T>>
class AvgPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;

    AvgPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("AvgPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride) {}

    void forward(const Tensor4D<T> &a) {
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        this->activation.resize(a.d1, (a.d2 + 2*padding - ks)/stride + 1, (a.d3 + 2*padding - ks)/stride + 1, a.d4);
        Backend::avgPool2D(ks, padding, stride, a, this->activation, scale);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        Backend::avgPool2DInputGrad(ks, padding, stride, this->inputDerivative, e, scale);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, (d2 + 2*padding - ks)/stride + 1, (d3 + 2*padding - ks)/stride + 1, d4);
        this->inputDerivative.resize(d1, d2, d3, d4);
    }
};

template <typename T, u64 scale, class Backend = DefaultBackend<T>>
class SumPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;

    SumPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("SumPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride) {}

    void forward(const Tensor4D<T> &a) {
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        this->activation.resize(a.d1, (a.d2 + 2*padding - ks)/stride + 1, (a.d3 + 2*padding - ks)/stride + 1, a.d4);
        Backend::sumPool2D(ks, padding, stride, a, this->activation);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        Backend::sumPool2DInputGrad(ks, padding, stride, this->inputDerivative, e);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, (d2 + 2*padding - ks)/stride + 1, (d3 + 2*padding - ks)/stride + 1, d4);
        this->inputDerivative.resize(d1, d2, d3, d4);
    }
};

template <typename T, class Backend = DefaultBackend<T>>
class MaxPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;
    Tensor4D<u64> maxIndex;

    MaxPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("MaxPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride), maxIndex(0,0,0,0) {}

    void forward(const Tensor4D<T> &a) {
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        this->activation.resize(a.d1, (a.d2 + 2*padding - ks)/stride + 1, (a.d3 + 2*padding - ks)/stride + 1, a.d4);
        this->maxIndex.resize(a.d1, (a.d2 + 2*padding - ks)/stride + 1, (a.d3 + 2*padding - ks)/stride + 1, a.d4);
        Backend::maxPool2D(ks, padding, stride, a, this->activation, maxIndex);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        Backend::maxPool2DInputGrad(ks, padding, stride, this->inputDerivative, e, maxIndex);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, (d2 + 2*padding - ks)/stride + 1, (d3 + 2*padding - ks)/stride + 1, d4);
        this->inputDerivative.resize(d1, d2, d3, d4);
    }
};

template <typename T>
class Flatten : public Layer<T> {
public:
    u64 d1, d2, d3, d4;

    Flatten() : Layer<T>("Flatten") {}

    void forward(const Tensor4D<T> &a) {
        d1 = a.d1;
        d2 = a.d2;
        d3 = a.d3;
        d4 = a.d4;
        this->activation.resize(d1, d2 * d3 * d4, 1, 1);
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        this->activation(i, j * d3 * d4 + k * d4 + l, 0, 0) = a(i, j, k, l);
                    }
                }
            }
        }
    }

    void backward(const Tensor4D<T> &e) {
        this->inputDerivative.resize(d1, d2, d3, d4);
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        this->inputDerivative(i, j, k, l) = e(i, j * d3 * d4 + k * d4 + l, 0, 0);
                    }
                }
            }
        }
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, d2 * d3 * d4, 1, 1);
        this->inputDerivative.resize(d1, d2, d3, d4);
    }
};

template <typename T, u64 scale, class Backend = DefaultBackend<T>>
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
        static_assert(std::is_integral<T>::value || scale == 0);
        double xavier = 1.0 / sqrt(in);
        weight.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
        Vw.fill(0);
        Vb.fill(0);
    }

    void forward(const Tensor4D<T> &a) {
        inp.resize(a.d1, a.d2, a.d3, a.d4);
        this->inp.copy(a);
        this->activation.resize(a.d1, weight.d2, 1, 1);
        this->weightGrad.resize(weight.d1, weight.d2);
        Backend::matmul(a, weight, this->activation);
        this->activation.addBias2D(bias);
    }

    void backward(const Tensor4D<T> &e) {
        if (!(this->isFirst)) {
            this->inputDerivative.resize(e.d1, weight.d1, 1, 1);
            Backend::matmulTransposeB(e, weight, this->inputDerivative);
            Backend::truncate(this->inputDerivative, scale);
        }
        Backend::matmulTransposeA(inp, e, weightGrad);
        // Backend::truncate(weightGrad, scale);
        Vw.resize(weightGrad.d1, weightGrad.d2);
        Backend::updateWeight(weight, weightGrad, Vw, scale);
        Backend::updateBias(bias, e, Vb, scale);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, out, 1, 1);
        this->inputDerivative.resize(d1, in, 1, 1);
        this->inp.resize(d1, d2, d3, d4);
    }
    
    Tensor2D<T>& getweights() { return weight; }
    Tensor<T>& getbias() { return bias; }
};

template <typename T, class Backend = DefaultBackend<T>>
class ReLUTruncate: public Layer<T> {
public:
    u64 shift;
    Tensor4D<T> drelu;
    ReLUTruncate(u64 shift) : Layer<T>("ReLUTruncate"), shift(shift), drelu(0,0,0,0) {}

    void forward(const Tensor4D<T> &a) {
        // std::cout << "== Truncate forward ==" << std::endl;
        // std::cout << "a: "; a.print();
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->drelu.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        Backend::relutruncate(a, this->activation, this->drelu, shift);
    }

    void backward(const Tensor4D<T> &e) {
        // std::cout << "== ReLU backward ==" << std::endl;
        // std::cout << "e: "; e.print();
        Backend::select(e, this->drelu, this->inputDerivative);
        // std::cout << "== Truncate ==" << std::endl;
        // std::cout << "e: "; this->inputDerivative.print();
        // truncate(this->inputDerivative, this->inputDerivative, shift);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, d2, d3, d4);
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->drelu.resize(d1, d2, d3, d4);
    }
};

template <typename T, class Backend = DefaultBackend<T>>
class ReLU: public Layer<T> {
public:
    Tensor4D<T> drelu;
    ReLU() :  Layer<T>("ReLU"), drelu(0,0,0,0) {}

    void forward(const Tensor4D<T> &a) {
        // std::cout << "== ReLU forward ==" << std::endl;
        // std::cout << "a: "; a.print();
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->drelu.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        Backend::relu(a, this->activation, this->drelu);
    }

    void backward(const Tensor4D<T> &e) {
        // std::cout << "== ReLU backward ==" << std::endl;
        // std::cout << "e: "; e.print();
        Backend::select(e, this->drelu, this->inputDerivative);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, d2, d3, d4);
        this->inputDerivative.resize(d1, d2, d3, d4);
        this->drelu.resize(d1, d2, d3, d4);
    }
};

template <typename T, class Backend = DefaultBackend<T>>
class Truncate: public Layer<T> {
public:
    u64 shift;
    Truncate(u64 shift) : Layer<T>("Truncate"), shift(shift) {}

    void forward(const Tensor4D<T> &a) {
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        Backend::truncate(a, this->activation, shift);
    }

    void backward(const Tensor4D<T> &e) {
        // Backend::truncate(e, this->inputDerivative, shift);
        this->inputDerivative.copy(e);
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, d2, d3, d4);
        this->inputDerivative.resize(d1, d2, d3, d4);
    }
};

template <typename T>
class Sequential : public Layer<T> {
public:
    std::vector<Layer<T>*> layers;
    
    Sequential(std::vector<Layer<T>*> _layers) : Layer<T>("Sequential"), layers(_layers) {
        int s = layers.size();
        // Set isFirst
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
    
    void forward(const Tensor4D<T> &a) {
        layers[0]->forward(a);
        u64 size = layers.size();
        for(u64 i = 1; i < size; i++) {
            layers[i]->forward(layers[i-1]->activation);
        }
        this->activation.resize(layers[size-1]->activation.d1, layers[size-1]->activation.d2, layers[size-1]->activation.d3, layers[size-1]->activation.d4);
        this->activation.copy(layers[size-1]->activation);

    }
    void backward(const Tensor4D<T> &e) {
        int size = layers.size();
        layers[size-1]->backward(e);
        for (int i = size - 2; i >= 0; i--) {
            layers[i]->backward(layers[i+1]->inputDerivative);
        }
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        this->activation.resize(d1, d2, d3, d4);
        this->inputDerivative.resize(d1, d2, d3, d4);
        u64 size = layers.size();
        for(u64 i = 0; i < size; i++) {
            layers[i]->resize(d1, d2, d3, d4);
            d1 = layers[i]->activation.d1;
            d2 = layers[i]->activation.d2;
            d3 = layers[i]->activation.d3;
            d4 = layers[i]->activation.d4;
        }
    }
};
