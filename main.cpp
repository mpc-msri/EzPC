#include <iostream>
#include <vector>
#include "utils.h"

template <typename T>
class Layer {
public:
    Layer<T> *next, *prev;
    virtual void forward(Tensor4D<T> &a) = 0;
    virtual Tensor4D<T> backward(Tensor4D<T> &e) = 0;
    Tensor4D<T> activation;
    Layer() : activation(0,0,0,0) {}
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
    
    Tensor4D<T> backward(Tensor4D<T> &e) {
        return e;
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
    Tensor4D<T> backward(Tensor4D<T> &e) {
        for(auto layer : layers) {
            e = layer->backward(e);
        }
        return e;
    }
};

int main() {
    // Tensor4D<u64> a(1, 1, 3, 3);
    // a[0][0][0][0] = 1;
    // a[0][0][0][1] = 2;
    // a[0][0][0][2] = 3;
    // a[0][0][1][0] = 4;
    // a[0][0][1][1] = 5;
    // a[0][0][1][2] = 6;
    // a[0][0][2][0] = 7;
    // a[0][0][2][1] = 8;

    // for(int i = 0; i < 3; i++) {
    //     for(int j = 0; j < 3; j++) {
    //         std::cout << a[0][0][i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Tensor2D<u64> x(3, 3);
    // x[0][0] = 1;
    // x[0][1] = 2;
    // x[0][2] = 3;
    // x[1][0] = 4;
    // x[1][1] = 5;
    // x[1][2] = 6;
    // x[2][0] = 7;
    // x[2][1] = 8;
    // x[2][2] = 9;

    // Tensor2D<u64> y(3, 1);
    // y[0][0] = 1;
    // y[1][0] = 2;
    // y[2][0] = 3;

    // auto z = matmul(x, y);

    // for(int i = 0; i < 3; i++) {
    //     std::cout << z[i][0] << std::endl;
    // }

    Tensor4D<u64> image(1, 32, 32, 3);
    Tensor4D<u64> filter(3, 3, 3, 64);
    
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            image[0][i][j][0] = 1;
            image[0][i][j][1] = 1;
            image[0][i][j][2] = 1;
        }
    }

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            for(int k = 0; k < 3; k++) {
                for(int l = 0; l < 64; l++) {
                    filter[i][j][k][l] = 1;
                }
            }
        }
    }

    Tensor2D<u64> imgR = reshapeInput(image, 1, 1, 3, 3);
    Tensor2D<u64> fR = reshapeFilter(filter);
    auto z = matmul(fR, imgR);

    // for(int i = 0; i < z.d1; ++i) {
    //     for(int j = 0; j < z.d2; ++j) {
    //         std::cout << z[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    Sequential<u64> model({
        new Conv2D<u64>(3ULL, 64ULL, 3ULL, 1ULL, 1ULL)
    });
    // Conv2D<u64> model(3, 64, 3, 1, 1);

    Tensor4D<u64> a(1, 32, 32, 3);

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            a[0][i][j][0] = 1;
            a[0][i][j][1] = 1;
            a[0][i][j][2] = 1;
        }
    }

    model.forward(image);

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            for(int k = 0; k < 64; k++) {
                std::cout << model.activation[0][i][j][k] << " ";
            }
        }
        std::cout << std::endl;
    }
}