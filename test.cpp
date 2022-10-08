#include <iostream>
#include <vector>
#include "layers.h"
#include "softmax.h"
#include "mnist.h"
#include <cmath>
#include <iomanip>

const u64 scale = 12;

void test_float() {
    auto model = Sequential<double>({
        new Flatten<double>(),
        new FC<double, 0>(784, 500),
        new ReLU<double>(),
        new FC<double, 0>(500, 10),
        new ReLU<double>(),
    });

    // ((FC<double, 0> *)(model.layers[3]))->weight.print();

    Tensor4D<double> image(1, 28, 28, 1);
    for(int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            image(0, i, j, 0) = 1.0;//train_image[0][i * 28 + j];
        }
    }

    model.forward(image);
    model.activation.print();
    std::cout << "== Forward Pass complete ==" << std::endl;
    Tensor4D<double> e(1, 10, 1, 1);
    double max = model.activation(0, 0, 0, 0);
    for(u64 j = 1; j < 10; ++j) {
        if(model.activation(0, j, 0, 0) > max) {
            max = model.activation(0, j, 0, 0);
        }
    }
    double den = 0.0;
    for(u64 j = 0; j < 10; ++j) {
        e(0, j, 0, 0) = model.activation(0, j, 0, 0) - max;
        den += exp(e(0, j, 0, 0));
    }

    for(u64 j = 0; j < 10; ++j) {
        double x = e(0, j, 0, 0);
        e(0, j, 0, 0) = exp(x) / den;
    }
    e(0, 0, 0, 0) -= 1.0;
    std::cout << "e: ";
    e.print();
    model.backward(e);
    // ((FC<double, 0> *)(model.layers[1]))->weightGrad.print();
    // ((FC<double, 0> *)(model.layers[1]))->weight.print();
    model.forward(image);
    model.activation.print();
}

void test_int() {
    auto model = Sequential<i64>({
        new Flatten<i64>(),
        new FC<i64, scale>(784, 500),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(500, 10),
        new ReLUTruncate<i64>(scale)
    });

    Tensor4D<i64> image(1, 28, 28, 1);
    for(int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            image(0, i, j, 0) = (1ULL << scale);
        }
    }

    model.forward(image);
    std::cout << "== Forward Pass complete ==" << std::endl;
    Tensor4D<i64> e(1, 10, 1, 1);
    i64 max = model.activation(0, 0, 0, 0);
    for(u64 j = 1; j < 10; ++j) {
        if(model.activation(0, j, 0, 0) > max) {
            max = model.activation(0, j, 0, 0);
        }
    }
    double den = 0.0;
    for(u64 j = 0; j < 10; ++j) {
        e(0, j, 0, 0) = model.activation(0, j, 0, 0) - max;
        double x = ((double)e(0, j, 0, 0)) / (1ULL << scale);
        den += exp(x);
    }

    for(u64 j = 0; j < 10; ++j) {
        double x = ((double)e(0, j, 0, 0)) / (1ULL << scale);
        e(0, j, 0, 0) = (i64)((exp(x) / den) * (1ULL << scale));
    }
    e(0, 0, 0, 0) -= (1ULL << scale);
    std::cout << "e: ";
    e.print();
    model.backward(e);
}