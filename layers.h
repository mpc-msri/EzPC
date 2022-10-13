#pragma once
#include "tensor.h"
#include "utils.h"
#include "relutruncate.h"

template <typename T>
class Layer {
public:
    virtual void forward(const Tensor4D<T> &a) = 0;
    virtual void backward(const Tensor4D<T> &e) = 0;
    Tensor4D<T> activation;
    Tensor4D<T> inputDerivative;
    Layer() : activation(0,0,0,0), inputDerivative(0,0,0,0) {}
};

template <typename T, u64 scale, bool isFirst = false>
class Conv2D : public Layer<T> {
public:
    Tensor4D<T> inp;
    Tensor2D<T> filter;
    Tensor2D<T> filterGrad;
    Tensor<T> bias;
    Tensor<T> biasGrad;
    u64 ci, co, ks, padding, stride;

    Conv2D(u64 ci, u64 co, u64 ks, u64 padding, u64 stride) : ci(ci), co(co), ks(ks), padding(padding), 
        stride(stride), filter(co, ks * ks * ci), filterGrad(co, ks * ks * ci), bias(co), biasGrad(co), inp(0,0,0,0)
    {
        static_assert(std::is_integral<T>::value || scale == 0);
        double xavier = 1.0 / sqrt(ci);
        filter.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
    }

    void forward(const Tensor4D<T> &a) {
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        assert(a.d4 == ci);
        u64 newH = (((a.d2 + 2*padding - ks)/stride) + 1);
        u64 newW = (((a.d3 + 2*padding - ks)/stride) + 1);
        inp.resize(a.d1, a.d2, a.d3, a.d4);
        inp.copy(a);
        this->activation.resize(a.d1, newH, newW, co);
        conv2D<T>(ks, ks, padding, stride, ci, co, a, filter, this->activation);
        this->activation.addBias(bias);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        conv2DFilterGrad<T>(ks, ks, padding, stride, ci, co, inp, filterGrad, e);
        conv2DBiasGrad<T>(e, biasGrad);
        if (!isFirst) {
            conv2DInputGrad<T>(ks, ks, padding, stride, ci, co, this->inputDerivative, filter, e);
            truncate(this->inputDerivative, scale);
        }
        truncate(filterGrad, scale);
        filter.updateWeight(filterGrad, 0.01);
        bias.updateBias(biasGrad, 0.01, scale);
    }
};

template <typename T, u64 scale>
class AvgPool2D : public Layer<T> {
public:
    u64 ks, padding, stride;

    AvgPool2D(u64 ks, u64 padding, u64 stride) : ks(ks), padding(padding), stride(stride) {}

    void forward(const Tensor4D<T> &a) {
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        this->activation.resize(a.d1, (a.d2 + 2*padding - ks)/stride + 1, (a.d3 + 2*padding - ks)/stride + 1, a.d4);
        avgPool2D<T>(ks, padding, stride, a, this->activation);
    }

    void backward(const Tensor4D<T> &e) {
        assert(e.d1 == this->activation.d1);
        assert(e.d2 == this->activation.d2);
        assert(e.d3 == this->activation.d3);
        assert(e.d4 == this->activation.d4);
        avgPool2DInputGrad<T>(ks, padding, stride, this->inputDerivative, e);
        truncate(this->inputDerivative, scale);
    }
};

template <typename T>
class Flatten : public Layer<T> {
public:
    u64 d1, d2, d3, d4;

    Flatten() {}

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
};

template <typename T, u64 scale, bool isFirst = false>
class FC : public Layer<T> {
public:
    Tensor4D<T> inp;
    Tensor2D<T> weight;
    Tensor2D<T> weightGrad;
    Tensor<T> bias;
    u64 in, out;

    FC(u64 in, u64 out) : in(in), out(out), weight(in, out), bias(out), inp(0,0,0,0), weightGrad(0,0) {
        static_assert(std::is_integral<T>::value || scale == 0);
        double xavier = 1.0 / sqrt(in);
        weight.randomize(xavier * (1ULL<<scale));
        bias.randomize(xavier * (1ULL<<(2*scale)));
    }

    void forward(const Tensor4D<T> &a) {
        // std::cout << "== FC forward ==" << std::endl;
        // std::cout << "a: "; a.print();
        inp.resize(a.d1, a.d2, a.d3, a.d4);
        this->inp.copy(a);
        this->activation.resize(a.d1, weight.d2, 1, 1);
        this->weightGrad.resize(weight.d1, weight.d2);
        matmul(a, weight, this->activation);
        this->activation.addBias2D(bias);
    }

    void backward(const Tensor4D<T> &e) {
        // std::cout << "== FC backward ==" << std::endl;
        // std::cout << "e: "; e.print();
        if (!isFirst) {
            this->inputDerivative.resize(e.d1, weight.d1, 1, 1);
            matmulTransposeB(e, weight, this->inputDerivative);
            truncate(this->inputDerivative, scale);
        }
        // std::cout << "r: "; r.print();
        // std::cout << "weight: "; weight.print();
        // inp.print();
        // inp.transpose2D();
        // auto g = matmul(inp, e);
        // truncate(g, scale);
        // weight.updateWeight(g, 0.01);
        // matmul(inp, e, weightGrad);
        matmulTransposeA(inp, e, weightGrad);
        truncate(weightGrad, scale);
        weight.updateWeight(weightGrad, 0.1);
        bias.updateBias(e, 0.1, scale);
        // bias.updateBias(e, 0.01, 0);
    }
};

template <typename T>
class ReLUTruncate: public Layer<T> {
public:
    u64 shift;
    Tensor4D<T> drelu;
    ReLUTruncate(u64 shift) : shift(shift), drelu(0,0,0,0) {}

    void forward(const Tensor4D<T> &a) {
        // std::cout << "== Truncate forward ==" << std::endl;
        // std::cout << "a: "; a.print();
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->drelu.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        relutruncate(a, this->activation, this->drelu, shift);
    }

    void backward(const Tensor4D<T> &e) {
        // std::cout << "== ReLU backward ==" << std::endl;
        // std::cout << "e: "; e.print();
        select(e, this->drelu, this->inputDerivative);
        // std::cout << "== Truncate ==" << std::endl;
        // std::cout << "e: "; this->inputDerivative.print();
        // truncate(this->inputDerivative, this->inputDerivative, shift);
    }
};

template <typename T>
class ReLU: public Layer<T> {
public:
    Tensor4D<T> drelu;
    ReLU() : drelu(0,0,0,0) {}

    void forward(const Tensor4D<T> &a) {
        // std::cout << "== ReLU forward ==" << std::endl;
        // std::cout << "a: "; a.print();
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->drelu.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        relu(a, this->activation, this->drelu);
    }

    void backward(const Tensor4D<T> &e) {
        // std::cout << "== ReLU backward ==" << std::endl;
        // std::cout << "e: "; e.print();
        select(e, this->drelu, this->inputDerivative);
    }
};

template <typename T>
class Truncate: public Layer<T> {
public:
    u64 shift;
    Truncate(u64 shift) : shift(shift) {}

    void forward(const Tensor4D<T> &a) {
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        truncate(a, this->activation, shift);
    }

    void backward(const Tensor4D<T> &e) {
        // truncate(e, this->inputDerivative, shift);
        this->inputDerivative.copy(e);
    }
};

template <typename T>
class Sequential : public Layer<T> {
public:
    std::vector<Layer<T>*> layers;
    
    Sequential(std::vector<Layer<T>*> layers) : layers(layers) {}
    
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
};
