#define USE_CLEARTEXT

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sytorch/layers/layers.h>
#include <sytorch/softmax.h>
#include <sytorch/networks.h>
#include <sytorch/datasets/cifar10.h>
#include <filesystem>
#include <Eigen/Dense>
#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_improved.h>
#include <sytorch/sequential.h>

template <typename T, u64 scale>
void cifar10_fill_images(Tensor4D<T>& trainImages, Tensor<u64> &trainLabels, int datasetOffset = 0) {
    int numImages = trainImages.d1;
    assert(trainImages.d2 == 32);
    assert(trainImages.d3 == 32);
    assert(trainImages.d4 == 3);
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    for(int b = 0; b < numImages; ++b) {
        for(u64 j = 0; j < 32; ++j) {
            for(u64 k = 0; k < 32; ++k) {
                for(u64 l = 0; l < 3; ++l) {
                    trainImages(b, j, k, l) = (T)((dataset.training_images[datasetOffset+b][j * 32 + k + l * 32 * 32] / 255.0) * (1LL << (scale)));
                }
            }
        }
        trainLabels(b) = dataset.training_labels[datasetOffset+b];
    }
}

void ct_test_3layer() {
    srand(time(NULL));
    const u64 scale = 24;
    const u64 bs = 100;

    auto conv1 = new Conv2D<i64>(3, 64, 5, 1, 1, true);
    auto conv2 = new Conv2D<i64>(64, 64, 5, 1, 1, true);
    auto conv3 = new Conv2D<i64>(64, 64, 5, 1, 1, true);
    auto fc1 = new FC<i64>(64, 10, true);
    auto model = Sequential<i64>({
        conv1,
        new ReLU<i64>(),
        new MaxPool2D<i64>(3, 0, 2),
        conv2,
        new ReLU<i64>(),
        new MaxPool2D<i64>(3, 0, 2),
        conv3,
        new ReLU<i64>(),
        new MaxPool2D<i64>(3, 0, 2),
        new Flatten<i64>(),
        fc1,
    });
    model.init(bs, 32, 32, 3, scale);
    // fc1->weight.print();
    // return;


    Tensor4D<i64> trainImages(bs, 32, 32, 3); // 1 images with server and 1 with client
    Tensor<u64> trainLabels(bs);
    Tensor4D<i64> e(bs, 10, 1, 1); // 1 images with server and 1 with client
    // trainImage.fill(1);
    int numIterations = 1;
    for(int i = 0; i < numIterations; ++i) {
        cifar10_fill_images<i64, scale>(trainImages, trainLabels, i * bs);
        model.forward(trainImages);
        softmax<i64, scale>(model.activation, e);
        for(int b = 0; b < bs; ++b) {
            e(b, trainLabels(b), 0, 0) -= (((1LL<<scale))/bs);
        }
        model.backward(e);
    }
    // model.activation.print<i64>();
    // conv1->activation.print<i64>();
    conv1->filter.print<i64>();
    conv2->filter.print<i64>();
    conv3->filter.print<i64>();
    fc1->weight.print<i64>();
    conv1->bias.print<i64>();
    conv2->bias.print<i64>();
    conv3->bias.print<i64>();
    fc1->bias.print<i64>();
}

void llama_test_3layer(int party) {
    if (party == 0) {
        return ct_test_3layer();
    }
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    // std::string ip = "172.31.45.174";
    llama->init(ip, false);
    if (party != 1) {
        secfloat_init(party - 1, ip);
    }
    const u64 bs = 100;
    
    auto conv1 = new Conv2D<u64>(3, 64, 5, 1, 1, true);
    auto conv2 = new Conv2D<u64>(64, 64, 5, 1, 1, true);
    auto conv3 = new Conv2D<u64>(64, 64, 5, 1, 1, true);
    auto fc1 = new FC<u64>(64, 10, true);
    auto model = Sequential<u64>({
        conv1,
        new ReLU<u64>(),
        new MaxPool2D<u64>(3, 0, 2),
        conv2,
        new ReLU<u64>(),
        new MaxPool2D<u64>(3, 0, 2),
        conv3,
        new ReLU<u64>(),
        new MaxPool2D<u64>(3, 0, 2),
        new Flatten<u64>(),
        fc1,
    });
    model.init(bs, 32, 32, 3, scale);
    model.setBackend(llama);
    model.optimize();

    Tensor4D<u64> trainImages(bs, 32, 32, 3);
    Tensor<u64> trainLabels(bs);
    Tensor4D<u64> e(bs, 10, 1, 1);
    // trainImage.fill(1);

    llama->initializeWeights(model); // dealer initializes the weights and sends to the parties
    llama::start();

    int numIterations = 1;
    for(int i = 0; i < numIterations; ++i) {
        cifar10_fill_images<u64, scale>(trainImages, trainLabels, i * bs);
        llama->inputA(trainImages);
        model.forward(trainImages);
        softmax_secfloat(model.activation, e, scale, party);
        if (party != 1) {
            for(int b = 0; b < bs; ++b) {
                e(b, trainLabels(b), 0, 0) -= (((1LL<<scale))/bs);
            }
        }
        model.backward(e);
    }
    llama::end();

    // auto op = conv1->activation;
    // llama->output(op);
    // if (party != 1) {
    //     blprint(op, LlamaConfig::bitlength - scale);
    // }
    // llama->output(conv1->filter);
    // llama->output(conv2->filter);
    // llama->output(conv3->filter);
    // llama->output(fc1->weight);
    // llama->output(conv1->bias);
    // llama->output(conv2->bias);
    // llama->output(conv3->bias);
    // llama->output(fc1->bias);
    // llama->output(model.activation);
    // llama->output(conv1->activation);
    // if (LlamaConfig::party != 1) {
    //     conv1->filter.print<i64>();
    //     conv2->filter.print<i64>();
    //     conv3->filter.print<i64>();
    //     fc1->weight.print<i64>();
    //     conv1->bias.print<i64>();
    //     conv2->bias.print<i64>();
    //     conv3->bias.print<i64>();
    //     fc1->bias.print<i64>();
    //     // model.activation.print<i64>();
    // }
    llama->finalize();
}

void llama_test_lenet_gupta(int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    llama->init(ip, false);
    if (party != 1) {
        secfloat_init(party - 1, ip);
    }
    const u64 bs = 100;

    auto model = Sequential<u64>({
        new Conv2D<u64>(1, 8, 5, 0, 1, true),
        new ReLU<u64>(),
        new MaxPool2D<u64>(2),
        new Conv2D<u64>(8, 16, 5, 0, 1, true),
        new ReLU<u64>(),
        new MaxPool2D<u64>(2),
        new Flatten<u64>(),
        new FC<u64>(256, 128, true),
        new ReLU<u64>(),
        new FC<u64>(128, 10, true),
    });

    model.init(bs, 28, 28, 1, scale);
    model.setBackend(llama);
    model.optimize();

    Tensor4D<u64> trainImages(bs, 28, 28, 1);
    trainImages.fill(1);
    Tensor<u64> trainLabels(bs);
    trainLabels.fill(1);
    Tensor4D<u64> e(bs, 10, 1, 1);
    // trainImage.fill(1);

    llama->initializeWeights(model); // dealer initializes the weights and sends to the parties
    llama::start();

    int numIterations = 1;
    for(int i = 0; i < numIterations; ++i) {
        llama->inputA(trainImages);
        model.forward(trainImages);
        softmax_secfloat(model.activation, e, scale, party);
        if (party != 1) {
            for(int b = 0; b < bs; ++b) {
                e(b, trainLabels(b), 0, 0) -= (((1LL<<scale))/bs);
            }
        }
        model.backward(e);
    }
    llama::end();
    llama->finalize();
}

void fptraining_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    // set floating point precision
    // std::cout << std::fixed  << std::setprecision(1);
#ifdef NDEBUG
    std::cerr << "> Release Build" << std::endl;
#else
    std::cerr << "> Debug Build" << std::endl;
#endif
    std::cerr << "> Eigen will use " << Eigen::nbThreads() << " threads" << std::endl;
}

int main(int argc, char** argv) {
    fptraining_init();
    int expnum = 2;

    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    if (expnum == 1)
        llama_test_lenet_gupta(party);
    else
        llama_test_3layer(party);

}