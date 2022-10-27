#include <iostream>
#include <iomanip>
#include "cifar10.hpp"
#include "mnist.h"
#include "layers.h"
#include "softmax.h"

const u64 numEpochs = 100;
const u64 batchSize = 100;

#define useMaxPool true

void printprogress(double percent) {
    int val = (int) (percent * 100);
    int lpad = (int) (percent * 50);
    int rpad = 50 - lpad;
    std::cout << "\r" << "[" << std::setw(3) << val << "%] ["
              << std::setw(lpad) << std::setfill('=') << ""
              << std::setw(rpad) << std::setfill(' ') << "] ";
    std::cout.flush();
}


void cifar10_int() {
    const u64 scale = 16;
    std::cout << "> Scale = " << scale << std::endl;
    std::cout << "> Probablistic = " << (ClearText<i64>::probablistic ? "true" : "false") << std::endl;
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();
    srand(time(NULL));
    std::cout << "=== Running Fixed-Point Training (CIFAR-10) ===" << std::endl;

    Tensor4D<i64> testSet(batchSize, 32, 32, 3);

    auto model = Sequential<i64>({
        new Conv2D<i64, scale>(3, 64, 5, 1),
        new ReLUTruncate<i64>(scale),
#if useMaxPool
        new MaxPool2D<i64>(3, 0, 2),
#else
        new AvgPool2D<i64, scale>(3, 0, 2),
#endif
        new Conv2D<i64, scale>(64, 64, 5, 1),
        new ReLUTruncate<i64>(scale),
#if useMaxPool
        new MaxPool2D<i64>(3, 0, 2),
#else
        new AvgPool2D<i64, scale>(3, 0, 2),
#endif
        new Conv2D<i64, scale>(64, 64, 5, 1),
        new ReLUTruncate<i64>(scale),
#if useMaxPool
        new MaxPool2D<i64>(3, 0, 2),
#else
        new AvgPool2D<i64, scale>(3, 0, 2),
#endif
        new Flatten<i64>(),
        new FC<i64, scale>(64, 10),
        new Truncate<i64>(scale),
    });

    Tensor4D<i64> e(batchSize, 10, 1, 1);
    Tensor4D<i64> trainImage(batchSize, 32, 32, 3);
    for(int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        for(u64 i = 0; i < trainLen; i += batchSize) {
            // fetch image
            for(int b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 32; ++j) {
                    for(u64 k = 0; k < 32; ++k) {
                        for(u64 l = 0; l < 3; ++l) {
                            trainImage(b, j, k, l) = (dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1ULL << (scale));
                        }
                    }
                }
            }
            model.forward(trainImage);
            softmax<i64, scale>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, dataset.training_labels[i+b], 0, 0) -= ((1ULL<<scale)/batchSize);
            }
            model.backward(e);
            printprogress(((double)i) / trainLen);
        }
        u64 correct = 0;
        for(u64 i = 0; i < testLen; i += batchSize) {
            for(u64 b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 32; ++j) {
                    for(u64 k = 0; k < 32; ++k) {
                        for(u64 l = 0; l < 3; ++l) {
                            testSet(b, j, k, l) = (dataset.test_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1ULL << (scale));
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
        std::cout << " Accuracy: " << (correct*100.0) / testLen;
        std::cout << std::endl;
    }
}



void lenet_int() {

    const u64 trainLen = 60000;
    const u64 testLen = 10000;
    const u64 scale = 14;
    srand(time(NULL));
    srand(rand());
    srand(rand());
    std::cout << "=== Running Fixed-Point Training ===" << std::endl;

    auto model = Sequential<i64>({
        new Conv2D<i64, scale>(1, 6, 5, 1, 1),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(2),
        new Conv2D<i64, scale>(6, 16, 5, 1),
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(2),
        new Conv2D<i64, scale>(16, 120, 5),
        new ReLUTruncate<i64>(scale),
        new Flatten<i64>(),
        new FC<i64, scale>(120, 84),
        new ReLUTruncate<i64>(scale),
        new FC<i64, scale>(84, 10),
        new Truncate<i64>(scale),
    });

    Tensor4D<i64> testSet(testLen, 28, 28, 1);
    for(u64 i = 0; i < testLen; ++i) {
        for(u64 j = 0; j < 28; ++j) {
            for(u64 k = 0; k < 28; ++k) {
                testSet(i, j, k, 0) = test_image[i][j * 28 + k] * (1ULL << scale);
            }
        }
    }

    Tensor4D<i64> e(batchSize, 10, 1, 1);
    Tensor4D<i64> trainImage(batchSize, 28, 28, 1);
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
            softmax<i64, scale>(model.activation, e);
            for(int b = 0; b < batchSize; ++b) {
                e(b, train_label[i+b], 0, 0) -= ((1ULL<<scale)/batchSize);
            }
            model.backward(e);
            printprogress(((double)i) / trainLen);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        model.forward(testSet);
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

void lenet_float() {

    srand(time(NULL));
    std::cout << "=== Running Floating Point Training (MNIST) ===" << std::endl;

    const u64 trainLen = 60000;
    auto model = Sequential<double>({
        new Conv2D<double, 0>(1, 6, 5, 1, 1),
        new ReLU<double>(),
        new MaxPool2D<double>(2),
        new Conv2D<double, 0>(6, 16, 5, 1),
        new ReLU<double>(),
        new MaxPool2D<double>(2),
        new Conv2D<double, 0>(16, 120, 5),
        new ReLU<double>(),
        new Flatten<double>(),
        new FC<double, 0>(120, 84),
        new ReLU<double>(),
        new FC<double, 0>(84, 10),
    });

    const u64 testLen = 10000;
    Tensor4D<double> testSet(testLen, 28, 28, 1);
    for(u64 i = 0; i < testLen; ++i) {
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
            printprogress(((double)i) / trainLen);
        }
        model.forward(testSet);
        u64 correct = 0;
        for(u64 i = 0; i < testLen; i++) {
            if (model.activation.argmax(i) == test_label[i]) {
                correct++;
            }
        }
        std::cout << "Epoch: " << epoch << " Accuracy: " << (correct*100.0)/testLen << std::endl;
    }
}

void cifar10_float() {
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();

    std::cout << "=== Running Floating-Point Training (CIFAR-10) ===" << std::endl;
    
    auto fc3 = new FC<double, 0>(256, 10);
    auto model = Sequential<double>({
        /// 3 Layer from Gupta et al
        new Conv2D<double, 0>(3, 64, 5, 1),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Conv2D<double, 0>(64, 64, 5, 1),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Conv2D<double, 0>(64, 64, 5, 1),
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Flatten<double>(),
        new FC<double, 0>(64, 10),
        /// Lenet like
        // new Conv2D<double, 0>(3, 18, 5, 1, 1),
        // new ReLU<double>(),
        // new MaxPool2D<double>(2),
        // new Conv2D<double, 0>(18, 48, 5, 1),
        // new ReLU<double>(),
        // new MaxPool2D<double>(2),
        // new Conv2D<double, 0>(48, 360, 5),
        // new ReLU<double>(),
        // new MaxPool2D<double>(2),
        // new Flatten<double>(),
        // new FC<double, 0>(360, 120),
        // new ReLU<double>(),
        // new FC<double, 0>(120, 80),
        // new ReLU<double>(),
        // new FC<double, 0>(80, 10),
        /// FC lol
        // new Flatten<double>(),
        // new FC<double, 0>(3072, 1024),
        // new ReLU<double>(),
        // new FC<double, 0>(1024, 256),
        // new ReLU<double>(),
        // fc3,
    });

    Tensor4D<double> trainImage(batchSize, 32, 32, 3);
    Tensor4D<double> testImage(batchSize, 32, 32, 3);
    Tensor4D<double> e(batchSize, 10, 1, 1);
    
    for(u64 epoch = 0; epoch < numEpochs; ++epoch) {
        for(u64 i = 0; i < trainLen; i += batchSize) {
            for(u64 b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 32; ++j) {
                    for(u64 k = 0; k < 32; ++k) {
                        for(u64 l = 0; l < 3; ++l) {
                            trainImage(b, j, k, l) = dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0;
                        }
                    }
                }
            }
            model.forward(trainImage);
            softmax<double, 0>(model.activation, e);
            for(u64 b = 0; b < batchSize; ++b) {
                e(b, dataset.training_labels[i+b], 0, 0) -= (1.0/batchSize);
            }
            model.backward(e);
            printprogress(((double)i) / trainLen);
        }

        u64 correct = 0;
        for(u64 i = 0; i < testLen; i += batchSize) {
            for(u64 b = 0; b < batchSize; ++b) {
                for(u64 j = 0; j < 32; ++j) {
                    for(u64 k = 0; k < 32; ++k) {
                        for(u64 l = 0; l < 3; ++l) {
                            testImage(b, j, k, l) = dataset.test_images[i+b][j * 32 + k + l * 32 * 32] / 255.0;
                        }
                    }
                }
            }
            model.forward(testImage);
            for(u64 b = 0; b < batchSize; ++b) {
                if (model.activation.argmax(b) == dataset.test_labels[i+b]) {
                    correct++;
                }
            }
        }

        std::cout << "Epoch: " << epoch << " Accuracy: " << correct*100.0 / testLen << "% (" << correct << "/" << testLen << ")" << std::endl;

    }

}
