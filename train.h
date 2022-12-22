#pragma once
#include "layers.h"
#include "mnist.h"
#include "cifar10.hpp"
#include "softmax.h"

void printprogress(double percent);

template <typename T, u64 scale>
void train_mnist(Sequential<T> &model) {
    load_mnist();
    const u64 trainLen = 60000;
    const u64 testLen = 10000;
    const u64 batchSize = 100;
    const u64 numEpochs = 10;
    srand(time(NULL));
    std::cout << "=== Running Training on MNIST ===" << std::endl;
    std::cout << "> Scale = " << scale << std::endl;
    std::cout << "> Probablistic = " << (ClearText<T>::probablistic ? "true" : "false") << std::endl;

    Tensor4D<T> testSet(testLen, 28, 28, 1);
    for(u64 i = 0; i < testLen; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = (T)(test_image[i][j * 28 + k] * (1LL << scale));
            }
        }
    }

    Tensor4D<T> e(batchSize, 10, 1, 1);
    Tensor4D<T> trainImage(batchSize, 28, 28, 1);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch: " << epoch << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
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
            softmax<T, scale>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, train_label[i+b], 0, 0) -= (((T)(1LL<<scale))/batchSize);
            }
            model.backward(e);
            printprogress(((double)i) / trainLen);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        model.forward(testSet, false);
        u64 correct = 0;
        for(u64 i = 0; i < testLen; i++) {
            if (model.activation.argmax(i) == test_label[i]) {
                correct++;
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << std::endl;
        std::cout << " Accuracy: " << (correct*100.0) / testLen;
        std::cout << " Training Time: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << " seconds";
        std::cout << " Testing Time: " << std::chrono::duration_cast<std::chrono::seconds>(t3-t2).count() << " seconds";
        std::cout << std::endl;
    }
}

template <typename T, u64 scale>
void train_cifar10(Sequential<T> &model) {
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();
    const u64 batchSize = 100;
    const u64 numEpochs = 10;
    srand(time(NULL));
    std::cout << "=== Running Training on CIFAR10 ===" << std::endl;
    std::cout << "> Scale = " << scale << std::endl;
    std::cout << "> Probablistic = " << (ClearText<T>::probablistic ? "true" : "false") << std::endl;

    Tensor4D<T> testSet(batchSize, 32, 32, 3);
    Tensor4D<T> e(batchSize, 10, 1, 1);
    Tensor4D<T> trainImage(batchSize, 32, 32, 3);
    
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch: " << epoch << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        for(u64 i = 0; i < trainLen; i += batchSize) {
            // fetch image
            for(int b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 32; ++j) {
                    for(u64 k = 0; k < 32; ++k) {
                        for(u64 l = 0; l < 3; ++l) {
                            trainImage(b, j, k, l) = (T)((dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1LL << (scale)));
                        }
                    }
                }
            }
            model.forward(trainImage);
            softmax<T, scale>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, dataset.training_labels[i+b], 0, 0) -= (((T)(1LL<<scale))/batchSize);
            }
            model.backward(e);
            printprogress(((double)i) / trainLen);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        u64 correct = 0;
        for(u64 i = 0; i < testLen; i += batchSize) {
            for(u64 b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 32; ++j) {
                    for(u64 k = 0; k < 32; ++k) {
                        for(u64 l = 0; l < 3; ++l) {
                            testSet(b, j, k, l) = (T)((dataset.test_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1LL << (scale)));
                        }
                    }
                }
            }
            model.forward(testSet);
            for(u64 b = 0; b < batchSize; b++) {
                if (model.activation.argmax(b) == dataset.test_labels[i+b]) {
                    correct++;
                }
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        std::cout << std::endl;
        std::cout << " Accuracy: " << (correct*100.0) / testLen;
        std::cout << " Training Time: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << " seconds";
        std::cout << " Testing Time: " << std::chrono::duration_cast<std::chrono::seconds>(t3-t2).count() << " seconds";
        std::cout << std::endl;
    }
}

