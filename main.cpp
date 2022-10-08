#include <iostream>
#include <vector>
#include "layers.h"
#include "softmax.h"
#include "mnist.h"
#include <cmath>
#include <iomanip>

const u64 trainLen = 60000;
const u64 scale = 12;
const u64 numEpochs = 20;
const u64 batchSize = 100;

void main_float() {

    std::cout << "=== Running Floating Point Training ===" << std::endl;
    Tensor4D<double> trainSet(trainLen, 28, 28, 1);

    for(u64 i = 0; i < trainLen; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                trainSet(i, j, k, 0) = train_image[i][j * 28 + k];
            }
        }
    }

    Tensor4D<double> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k];
            }
        }
    }

    auto fc1 = new FC<double, 0, true>(784, 512);
    auto fc2 = new FC<double, 0>(512, 256);
    auto fc3 = new FC<double, 0>(256, 10);

    auto model = Sequential<double>({
        new Flatten<double>(),
        fc1,
        new ReLU<double>(),
        fc2,
        new ReLU<double>(),
        fc3,
        new ReLU<double>(),
    });

    Tensor4D<double> e(batchSize, 10, 1, 1);
    Tensor4D<double> trainImage(batchSize, 28, 28, 1);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        for(u64 i = 0; i < trainLen; i += batchSize) {
            // fetch image
            for(int b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 28; ++j) {
                    for(u64 k = 0; k < 28; ++k) {
                        trainImage(b, j, k, 0) = train_image[i+b][j * 28 + k];
                    }
                }
            }
            model.forward(trainImage);
            softmax<double, 0>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, train_label[i+b], 0, 0) -= (1.0/batchSize);
            }
            model.backward(e);
        }
        model.forward(testSet);
        u64 correct = 0;
        for(u64 i = 0; i < 10000; i++) {
            if (model.activation.argmax(i) == test_label[i]) {
                correct++;
            }
        }
        std::cout << "Epoch: " << epoch << " Accuracy: " << correct/100.0 << std::endl;
    }
}

void main_int() {

    std::cout << "=== Running Fixed-Point Training ===" << std::endl;
    Tensor4D<i64> trainSet(trainLen, 28, 28, 1);

    for(u64 i = 0; i < trainLen; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                trainSet(i, j, k, 0) = train_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    auto model = Sequential<i64>({
        new Flatten<i64>(),
        new FC<i64, scale, true>(784, 500),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(500, 10),
        new ReLUTruncate<i64>(scale)
    });

    Tensor4D<i64> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    Tensor4D<i64> e(batchSize, 10, 1, 1);
    Tensor4D<i64> trainImage(batchSize, 28, 28, 1);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        for(u64 i = 0; i < trainLen; i += batchSize) {
            // fetch image
            for(int b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 28; ++j) {
                    for(u64 k = 0; k < 28; ++k) {
                        trainImage(b, j, k, 0) = train_image[i+b][j * 28 + k] * (1ULL << scale);
                    }
                }
            }
            model.forward(trainImage);
            softmax<i64, scale>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, train_label[i+b], 0, 0) -= ((1ULL<<scale)/batchSize);
            }
            model.backward(e);
        }
        model.forward(testSet);
        u64 correct = 0;
        for(u64 i = 0; i < 10000; i++) {
            if (model.activation.argmax(i) == test_label[i]) {
                correct++;
            }
        }
        std::cout << "Epoch: " << epoch << " Accuracy: " << correct/100.0 << std::endl;
    }
}

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

int main() {
    // std::cout << std::fixed;
    // std::cout << std::setprecision(20);
    load_mnist();
    // main_float();
    main_int();
    // test_float();
    // test_int();
}