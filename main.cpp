#define USE_CLEARTEXT

#include <iostream>
#include <vector>
#include "layers.h"
#include "softmax.h"
#include "mnist.h"
#include <cmath>
#include <iomanip>
#include "cifar10.hpp"

void printprogress(double percent) {
    int val = (int) (percent * 100);
    int lpad = (int) (percent * 50);
    int rpad = 50 - lpad;
    std::cout << "\r" << "[" << std::setw(3) << val << "%] ["
              << std::setw(lpad) << std::setfill('=') << ""
              << std::setw(rpad) << std::setfill(' ') << "] ";
    std::cout.flush();
}

const u64 numEpochs = 100;
const u64 batchSize = 100;

#define useMaxPool true
void threelayer_keysize_llama() {

    const u64 scale = 16;
    LlamaKey<i64>::verbose = true;
    LlamaKey<i64>::probablistic = true;
    LlamaKey<i64>::bw = 64;
    const u64 minibatch = 100;

    auto model = Sequential<i64>({
        new Conv2D<i64, scale, LlamaKey<i64>>(3, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
#if useMaxPool
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
#else
        new AvgPool2D<i64, scale, LlamaKey<i64>>(3, 0, 2),
#endif
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
#if useMaxPool
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
#else
        new AvgPool2D<i64, scale, LlamaKey<i64>>(3, 0, 2),
#endif
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
#if useMaxPool
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
#else
        new AvgPool2D<i64, scale, LlamaKey<i64>>(3, 0, 2),
#endif
        new Flatten<i64>(),
        new FC<i64, scale, LlamaKey<i64>>(64, 10),
        new Truncate<i64, LlamaKey<i64>>(scale),
    });

    double gb = 1024ULL * 1024 * 1024 * 8ULL;

    Tensor4D<i64> trainImage(minibatch, 32, 32, 3);
    LlamaKey<i64>::serverkeysize = 0;
    LlamaKey<i64>::clientkeysize = 0;
    std::cout << ">>>>>>> FORWARD START<<<<<<<<" << std::endl;
    model.forward(trainImage);
    std::cout << ">>>>>>> FORWARD DONE <<<<<<<<" << std::endl;
    std::cout << "Server key size: " << LlamaKey<i64>::serverkeysize / gb << " GB" << std::endl;
    std::cout << "Client key size: " << LlamaKey<i64>::clientkeysize / gb << " GB" << std::endl;
    Tensor4D<i64> e(minibatch, model.activation.d2, model.activation.d3, model.activation.d4);
    LlamaKey<i64>::serverkeysize = 0;
    LlamaKey<i64>::clientkeysize = 0;
    std::cout << ">>>>>>> BACKWARD START <<<<<<<<" << std::endl;
    model.backward(e);
    std::cout << ">>>>>>> BACKWARD DONE <<<<<<<<" << std::endl;
    std::cout << "Server key size: " << LlamaKey<i64>::serverkeysize / gb << " GB" << std::endl;
    std::cout << "Client key size: " << LlamaKey<i64>::clientkeysize / gb << " GB" << std::endl;
}


void cifar10_int() {
    const u64 scale = 16;
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

void llama_test(int party) {
    srand(time(NULL));
    const u64 scale = 16;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto conv1 = new Conv2D<u64, scale, Llama<u64>>(3, 64, 5, 1);
    auto relu1 = new ReLUTruncate<u64, Llama<u64>>(scale);
    auto model = Sequential<u64>({
        conv1,
        relu1,
    });

    Tensor4D<u64> trainImage(2, 32, 32, 3); // 1 images with server and 1 with client

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    EndComputation();
    Llama<u64>::output(model.activation);
    Llama<u64>::finalize();
    if (LlamaConfig::party != 1)
        model.activation.print();
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


void cifar10_float_test() {
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();

    std::cout << "=== Running Floating-Point Training (CIFAR-10) ===" << std::endl;
    const u64 batchSize = 2;

    auto conv1 = new Conv2D<double, 0>(3, 64, 5, 1);
    auto conv2 = new Conv2D<double, 0>(64, 64, 5, 1);
    auto conv3 = new Conv2D<double, 0>(64, 64, 5, 1);
    auto fc1 = new FC<double, 0>(64, 10);
    auto model = Sequential<double>({
        /// 3 Layer from Gupta et al
        conv1,
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        conv2,
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        conv3,
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        new Flatten<double>(),
        fc1,
    });

    conv1->filter.fill(0.057);
    conv1->bias.fill(0.057);
    conv2->filter.fill(0.0125);
    conv2->bias.fill(0.0125);
    conv3->filter.fill(0.0125);
    conv3->bias.fill(0.0125);
    fc1->weight.fill(0.0625);
    fc1->bias.fill(0.0625);

    Tensor4D<double> trainImage(batchSize, 32, 32, 3);
    Tensor4D<double> testImage(batchSize, 32, 32, 3);
    Tensor4D<double> e(batchSize, 10, 1, 1);
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
        if (i == 0) {
            conv1->filter.print();
            break;
        }
    }
}

int main(int argc, char** argv) {
    prngWeights.SetSeed(osuCrypto::toBlock(time(NULL)));
#ifdef NDEBUG
    std::cout << "> Release Build" << std::endl;
#else
    std::cout << "> Debug Build" << std::endl;
#endif
    std::cout << "> Eigen will use " << Eigen::nbThreads() << " threads" << std::endl;
    // threelayer_keysize_llama();
    // load_mnist();
    // lenet_int();

    // int party = 0;
    // if (argc > 1) {
    //     party = atoi(argv[1]);
    // }
    // llama_test(party);

    cifar10_int();
}