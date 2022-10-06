#include <iostream>
#include <vector>
#include "layers.h"
#include "mnist.h"
#include <cmath>

const u64 miniBatch = 60000;
const u64 scale = 12;
const u64 numEpochs = 20;

void main_float() {

    std::cout << "=== Running Floating Point Training ===" << std::endl;
    Tensor4D<double> trainSet(miniBatch, 28, 28, 1);

    for(u64 i = 0; i < miniBatch; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                trainSet(i, j, k, 0) = train_image[i][j * 28 + k];
            }
        }
    }

    auto model = Sequential<double>({
        new Flatten<double>(),
        new FC<double, 0, true>(784, 10),
        // new ReLU<double>(),
        // new FC<double, 0>(500, 10),
        // new ReLU<double>(),
    });

    Tensor4D<double> e(1, 10, 1, 1);
    Tensor4D<double> trainImage(1, 28, 28, 1);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        for(u64 i = 0; i < miniBatch; ++i) {
            // fetch image
            for(u64 j = 0; j < 28; ++j) {
                for(u64 k = 0; k < 28; ++k) {
                    trainImage(0, j, k, 0) = train_image[i][j * 28 + k];
                }
            }
            model.forward(trainImage);
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
            e(0, train_label[i], 0, 0) -= 1.0;
            model.backward(e);
        }

    }

    model.forward(trainSet);

    u64 correct = 0;
    for(u64 i = 0; i < miniBatch; i++) {
        if (model.activation.argmax(i) == train_label[i]) {
            correct++;
        }
    }
    std::cout << "Training Set Accuracy: " << correct << "/" << miniBatch << std::endl;

    Tensor4D<double> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k];
            }
        }
    }

    model.forward(testSet);
    correct = 0;
    for(u64 i = 0; i < 10000; i++) {
        if (model.activation.argmax(i) == test_label[i]) {
            correct++;
        }
    }
    std::cout << "Test Set Accuracy: " << correct << "/10000" << std::endl;
}

void main_int() {

    std::cout << "=== Running Fixed-Point Training ===" << std::endl;
    Tensor4D<i64> trainSet(miniBatch, 28, 28, 1);

    for(u64 i = 0; i < miniBatch; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                trainSet(i, j, k, 0) = train_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    auto model = Sequential<i64>({
        new Flatten<i64>(),
        new FC<i64, scale, true>(784, 10),
        // new ReLUTruncate<i64>(scale),
        // new FC<i64, scale>(500, 10),
        new Truncate<i64>(scale)
    });

    Tensor4D<i64> e(1, 10, 1, 1);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        for(u64 i = 0; i < miniBatch; ++i) {
            Tensor4D<i64> trainImage(1, 28, 28, 1);
            // fetch image
            for(u64 j = 0; j < 28; ++j) {
                for(u64 k = 0; k < 28; ++k) {
                    trainImage(0, j, k, 0) = train_image[i][j * 28 + k] * (1ULL << scale);
                }
            }
            model.forward(trainImage);
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
            e(0, train_label[i], 0, 0) -= (1ULL << scale);
            model.backward(e);
        }

    }

    model.forward(trainSet);

    u64 correct = 0;
    for(u64 i = 0; i < miniBatch; i++) {
        if (model.activation.argmax(i) == train_label[i]) {
            correct++;
        }
    }
    std::cout << "Training Set Accuracy: " << correct << "/" << miniBatch << std::endl;

    Tensor4D<i64> testSet(10000, 28, 28, 1);
    for(u64 i = 0; i < 10000; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    model.forward(testSet);
    correct = 0;
    for(u64 i = 0; i < 10000; i++) {
        if (model.activation.argmax(i) == test_label[i]) {
            correct++;
        }
    }
    std::cout << "Test Set Accuracy: " << correct << "/10000" << std::endl;
}

void test_float() {
    auto model = Sequential<double>({
        new Flatten<double>(),
        new FC<double, 0>(784, 500),
        new ReLU<double>(),
        new FC<double, 0>(500, 10),
        new ReLU<double>(),
    });

    Tensor4D<double> image(1, 28, 28, 1);
    for(int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            image(0, i, j, 0) = 1.0;//train_image[0][i * 28 + j];
        }
    }

    model.forward(image);
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
    load_mnist();
    main_float();
    main_int();
    // test_float();
    // test_int();
}