#include <iostream>
#include <vector>
#include "layers.h"
#include "softmax.h"
#include "mnist.h"
#include <cmath>
#include <iomanip>

const u64 trainLen = 60000;
const u64 scale = 24;
const u64 numEpochs = 100;
const u64 batchSize = 100;

void main_float() {

    std::cout << "=== Running Floating Point Training ===" << std::endl;

    auto model = Sequential<double>({
        new Conv2D<double, 0, true>(1, 6, 5, 1, 1),
        new ReLU<double>(),
        new AvgPool2D<double>(2),
        new Conv2D<double, 0>(6, 16, 5, 1),
        new ReLU<double>(),
        new AvgPool2D<double>(2),
        new Conv2D<double, 0>(16, 120, 5),
        new ReLU<double>(),
        new Flatten<double>(),
        new FC<double, 0>(120, 84),
        new ReLU<double>(),
        new FC<double, 0>(84, 10),
    });

    Tensor4D<double> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k];
            }
        }
    }

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

    auto model = Sequential<i64>({
        new Conv2D<i64, scale, true>(1, 6, 5, 1, 1),
        new ReLUTruncate<i64>(scale),
        new AvgPool2D<i64>(2),
        new Conv2D<i64, scale>(6, 16, 5, 1),
        new ReLUTruncate<i64>(scale),
        new AvgPool2D<i64>(2),
        new Conv2D<i64, scale>(16, 120, 5),
        new ReLUTruncate<i64>(scale),
        new Flatten<i64>(),
        new FC<i64, scale>(120, 84),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(84, 10),
        new Truncate<i64>(scale),
    });

    Tensor4D<i64> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    u64 correct = 0;
    model.forward(testSet);
    for(u64 i = 0; i < 10000; i++) {
        if (model.activation.argmax(i) == test_label[i]) {
            correct++;
        }
    }
    std::cout << "Epoch: " << -1 << " Accuracy: " << correct/100.0 << std::endl;

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

void test_conv_float()
{
    std::cout << "=== Running Floating Point CNN Training ===" << std::endl;
    auto conv1 = new Conv2D<double, 0>(1, 1, 2, 1, 1);
    auto avgpool1 = new AvgPool2D<double>(2, 0, 2);

    auto model = Sequential<double>({
        conv1,
        avgpool1,
    });

    conv1->filter(0, 0) = 1;
    conv1->filter(0, 1) = 2;
    conv1->filter(0, 2) = 3;
    conv1->filter(0, 3) = 4;
    conv1->bias(0) = 0;

    // std::cout << "filter: ";
    // conv1->filter.print();
    Tensor4D<double> a(1, 3, 3, 1);
    a.fill(1);

    model.forward(a);

    // std::cout << "act: ";
    // model.activation.print();

    Tensor4D<double> e(1, 2, 2, 1);
    e(0, 0, 0, 0) = 1;
    e(0, 0, 1, 0) = 1;
    e(0, 1, 0, 0) = 1;
    e(0, 1, 1, 0) = 1;
    // // model.activation.print();
    model.backward(e);
    model.forward(a);
    model.backward(e);
    std::cout << "filter grad: ";
    conv1->filterGrad.print();
    std::cout << "filter: ";
    conv1->filter.print();
    std::cout << "input grad: ";
    conv1->inputDerivative.print();
}

int main() {
    // std::cout << std::fixed;
    // std::cout << std::setprecision(20);
    load_mnist();
    // main_float();
    main_int();
    // test_conv_float();
}