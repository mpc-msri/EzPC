#pragma once
#include <sytorch/layers/layers.h>
#include <sytorch/sequential.h>
#include <sytorch/datasets/mnist.h>
#include <sytorch/datasets/cifar10.h>
#include <sytorch/softmax.h>

void printprogress(double percent);

// uncomment below line to calculate losses every 10 iterations and dump to file 
// #define PRINT_LOSSES_TO_FILE

// uncomment below line to use piranha's softmax in cifar training
// #define USE_PIRANHA_SOFTMAX

// this function differs from softmax in softmax.h in two ways
// 1. it's double only
// 2. it doesn't divide by batch size
template <typename T, u64 scale>
void softmax2(const Tensor4D<T> &in, Tensor4D<double> &out)
{
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == 1);
    assert(in.d4 == 1);
    assert(out.d3 == 1);
    assert(out.d4 == 1);
    assert(std::is_integral<T>::value || (scale == 0));
    auto batchSize = in.d1;
    auto numClasses = in.d2;
    for(int b = 0; b < batchSize; ++b) {
        T max = in(b, 0, 0, 0);
        for(u64 j = 1; j < numClasses; ++j) {
            if(in(b, j, 0, 0) > max) {
                max = in(b, j, 0, 0);
            }
        }
        double den = 0.0;
        double exps[numClasses];
        for(u64 j = 0; j < numClasses; ++j) {
            double x = in(b, j, 0, 0) - max;
            if constexpr (scale == 0) {
                exps[j] = std::exp(x);
            } else {
                exps[j] = std::exp(x / (1ULL << scale));
            }
            den += exps[j];
        }
        // den = den * batchSize;
        for(u64 j = 0; j < numClasses; ++j) {
            out(b, j, 0, 0) = exps[j] / den;
        }
    }
}

template <typename T, u64 scale>
void train_mnist(Sequential<T> &model) {
    load_mnist();
    const u64 trainLen = 60000;
    const u64 testLen = 10000;
    const u64 batchSize = 100;
    const u64 numEpochs = 10;
    model.init(batchSize, 28, 28, 1, scale);
#ifdef PRINT_LOSSES_TO_FILE
    std::ofstream losses("losses_cnn2.txt");
    Tensor4D<double> e_test(testLen, 10, 1, 1);
#endif
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
#ifdef PRINT_LOSSES_TO_FILE
            if (i % (10 * batchSize) == 0)
            {
                model.forward(testSet, false);
                softmax2<T, scale>(model.activation, e_test);
                double loss = 0;
                for(u64 b = 0; b < testLen; b++) {
                    double p = e_test(b, test_label[b], 0, 0);
                    // losses << "p: " << p << std::endl;
                    loss += (double) std::log(p);
                }
                losses << "Loss: " << -loss/testLen << std::endl;
            }
#endif
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
    model.init(batchSize, 32, 32, 3, scale);
    srand(time(NULL));
    std::cout << "=== Running Training on CIFAR10 ===" << std::endl;
    std::cout << "> Scale = " << scale << std::endl;
    std::cout << "> Truncation = ";
    if (ClearText<T>::localTruncationEmulation) {
        std::cout << "Local" << std::endl;
    } else if (ClearText<T>::probablistic) {
        std::cout << "Stochastic" << std::endl;
    } else {
        std::cout << "Correct" << std::endl;
    }

#ifdef USE_PIRANHA_SOFTMAX
    std::cout << "> Softmax = Piranha" << std::endl;
#else
    std::cout << "> Softmax = Accurate" << std::endl;
#endif

    Tensor4D<T> testSet(testLen, 32, 32, 3);
    for(u64 b = 0; b < testLen; ++b) {
        for(u64 j = 0; j < 32; ++j) {
            for(u64 k = 0; k < 32; ++k) {
                for(u64 l = 0; l < 3; ++l) {
                    testSet(b, j, k, l) = (T)((dataset.test_images[b][j * 32 + k + l * 32 * 32] / 255.0) * (1LL << (scale)));
                }
            }
        }
    }
    Tensor4D<T> e(batchSize, 10, 1, 1);
    Tensor4D<T> trainImage(batchSize, 32, 32, 3);
#ifdef PRINT_LOSSES_TO_FILE
    std::ofstream losses("losses.txt");
    Tensor4D<double> e_test(testLen, 10, 1, 1);
#endif
    
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch: " << epoch << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        for(u64 i = 0; i < trainLen; i += batchSize) {
#ifdef PRINT_LOSSES_TO_FILE
            if (i % (10 * batchSize) == 0)
            {
                model.forward(testSet, false);
                softmax2<T, scale>(model.activation, e_test);
                double loss = 0;
                for(u64 b = 0; b < testLen; b++) {
                    double p = e_test(b, dataset.test_labels[b], 0, 0);
                    // losses << "p: " << p << std::endl;
                    loss += (double) std::log(p);
                }
                losses << "Loss: " << -loss/testLen << std::endl;
            }
#endif
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
#ifdef USE_PIRANHA_SOFTMAX
            pirhana_softmax_ct(model.activation, e, scale);
#else
            softmax<T, scale>(model.activation, e);
#endif
            for(int b = 0; b < batchSize; ++b) {
                e(b, dataset.training_labels[i+b], 0, 0) -= (((T)(1LL<<scale))/batchSize);
            }
            model.backward(e);
            printprogress(((double)i) / trainLen);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        u64 correct = 0;
        model.forward(testSet);
        for(u64 b = 0; b < testLen; b++) {
            if (model.activation.argmax(b) == dataset.test_labels[b]) {
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

