#pragma once
#include "tensor.h"
#include "utils.h"
#include "relutruncate.h"

template <typename T>
class Layer {
public:
    virtual void forward(Tensor4D<T> &a) = 0;
    virtual void backward(Tensor4D<T> &e) = 0;
    Tensor4D<T> activation;
    Tensor4D<T> inputDerivative;
    Layer() : activation(0,0,0,0), inputDerivative(0,0,0,0) {}
};

template <typename T>
class Conv2D : public Layer<T> {
public:
    Tensor4D<T> filter;
    Tensor<T> bias;
    u64 ci, co, ks, padding, stride;

    Conv2D(u64 ci, u64 co, u64 ks, u64 padding, u64 stride) : ci(ci), co(co), ks(ks), padding(padding), 
        stride(stride), filter(ks, ks, ci, co), bias(co)
    {
        filter.randomize();
        bias.randomize();
    }

    void forward(Tensor4D<T> &a) {
        Tensor4D<T> r = conv2D<T>(padding, stride, a, filter);
        r.addBias(bias);
        this->activation.resize(r.d1, r.d2, r.d3, r.d4);
        this->activation.copy(r);
    }
    
    void backward(Tensor4D<T> &e) {
        assert(false);
    }
};

template <typename T>
class Flatten : public Layer<T> {
public:
    u64 d1, d2, d3, d4;

    Flatten() {}

    void forward(Tensor4D<T> &a) {
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

    void backward(Tensor4D<T> &e) {
    }
};

template <typename T>
class FC : public Layer<T> {
public:
    Tensor4D<T> inp;
    Tensor2D<T> weight;
    Tensor<T> bias;
    u64 in, out;

    FC(u64 in, u64 out) : in(in), out(out), weight(in, out), bias(out), inp(0,0,0,0) {
        weight.randomize();
        bias.randomize();
    }

    void forward(Tensor4D<T> &a) {
        // inp.resize(a.d1, a.d2, a.d3, a.d4);
        // this->inp.copy(a);
        Tensor4D<T> r = matmul(a, weight);
        r.addBias2D(bias);
        this->activation.resize(r.d1, r.d2, 1, 1);
        this->activation.copy(r);
    }

    void backward(Tensor4D<T> &e) {
        Tensor4D<T> r = matmul(e, weight, true);
        this->inputDerivative.resize(r.d1, r.d2, 1, 1);
        this->inputDerivative.copy(r);
        // inp.transpose2D();
        // auto g = matmul(inp, e);
        // weight.updateWeight(g, 0.01);
    }
};

template <typename T>
class ReLUTruncate: public Layer<T> {
public:
    u64 shift;
    Tensor4D<T> drelu;
    ReLUTruncate(u64 shift) : shift(shift), drelu(0,0,0,0) {}

    void forward(Tensor4D<T> &a) {
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->drelu.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        relutruncate(a, this->activation, this->drelu, shift);
    }

    void backward(Tensor4D<T> &e) {
        select(e, this->drelu, this->inputDerivative);
        truncate(this->inputDerivative, this->inputDerivative, shift);
    }
};

template <typename T>
class ReLU: public Layer<T> {
public:
    Tensor4D<T> drelu;
    ReLU() : drelu(0,0,0,0) {}

    void forward(Tensor4D<T> &a) {
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->drelu.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        relu(a, this->activation, this->drelu);
    }

    void backward(Tensor4D<T> &e) {
        select(e, this->drelu, this->inputDerivative);
    }
};

template <typename T>
class Truncate: public Layer<T> {
public:
    u64 shift;
    Truncate(u64 shift) : shift(shift) {}

    void forward(Tensor4D<T> &a) {
        this->activation.resize(a.d1, a.d2, a.d3, a.d4);
        this->inputDerivative.resize(a.d1, a.d2, a.d3, a.d4);
        truncate(a, this->activation, shift);
    }

    void backward(Tensor4D<T> &e) {
        truncate(e, this->inputDerivative, shift);
    }
};

template <typename T>
class Sequential : public Layer<T> {
public:
    std::vector<Layer<T>*> layers;
    
    Sequential(std::vector<Layer<T>*> layers) : layers(layers) {}
    
    void forward(Tensor4D<T> &a) {
        layers[0]->forward(a);
        u64 size = layers.size();
        for(u64 i = 1; i < size; i++) {
            layers[i]->forward(layers[i-1]->activation);
        }
        this->activation.resize(layers[size-1]->activation.d1, layers[size-1]->activation.d2, layers[size-1]->activation.d3, layers[size-1]->activation.d4);
        this->activation.copy(layers[size-1]->activation);

    }
    void backward(Tensor4D<T> &e) {
        int size = layers.size();
        layers[size-1]->backward(e);
        for (int i = size - 2; i >= 0; i--) {
            layers[i]->backward(layers[i+1]->inputDerivative);
        }
    }
};
