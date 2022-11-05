#define USE_CLEARTEXT

#include <iostream>
#include <vector>
#include "layers.h"
#include "softmax.h"
#include "mnist.h"
#include <cmath>
#include <iomanip>
#include "cifar10.hpp"
#include "networks.h"
#include "backend/llama_extended.h"

using LlamaVersion = Llama<u64>;

void llama_relutruncate_test(int party) {
    srand(time(NULL));
    const u64 scale = 2;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto relu1 = new ReLUTruncate<u64, LlamaVersion>(scale);
    auto model = Sequential<u64>({
        relu1,
    });

    Tensor4D<u64> trainImage(2, 2, 2, 1); // 1 images with server and 1 with client
    trainImage(0, 0, 0, 0) = 5; // 1 or 2
    trainImage(0, 0, 1, 0) = 7; // 1 or 2
    trainImage(0, 1, 0, 0) = 12; // 3
    trainImage(0, 1, 1, 0) = 15; // 3 or 4
    trainImage(1, 0, 0, 0) = -5;
    trainImage(1, 0, 1, 0) = -7;
    trainImage(1, 1, 0, 0) = -12;
    trainImage(1, 1, 1, 0) = -15;
    Tensor4D<u64> e(2, 2, 2, 1);
    e.fill(69);
    if (LlamaConfig::party == 1) {
        e.fill(0);
    }

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    model.backward(e);
    EndComputation();
    Llama<u64>::output(model.activation);
    // Llama<u64>::output(relu1->drelu);
    Llama<u64>::output(relu1->inputDerivative);
    Llama<u64>::finalize();
    if (LlamaConfig::party != 1) {
        model.activation.print();
        // relu1->drelu.print();
        relu1->inputDerivative.print();
    }
}

void llama_test_3layer(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaVersion::init();
    const u64 bs = 100;
    
    auto conv1 = new Conv2D<u64, scale, LlamaVersion>(3, 64, 5, 1);
    auto conv2 = new Conv2D<u64, scale, LlamaVersion>(64, 64, 5, 1);
    auto conv3 = new Conv2D<u64, scale, LlamaVersion>(64, 64, 5, 1);
    auto fc1 = new FC<u64, scale, LlamaVersion>(64, 10);
    auto model = Sequential<u64>({
        conv1,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        new MaxPool2D<u64, LlamaVersion>(3, 0, 2),
        conv2,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        new MaxPool2D<u64, LlamaVersion>(3, 0, 2),
        conv3,
        new ReLUTruncate<u64, LlamaVersion>(scale),
        new MaxPool2D<u64, LlamaVersion>(3, 0, 2),
        new Flatten<u64>(),
        fc1,
        new Truncate<u64, LlamaVersion>(scale),
    });

    auto conv1_ct = new Conv2D<i64, scale>(3, 64, 5, 1);
    auto conv2_ct = new Conv2D<i64, scale>(64, 64, 5, 1);
    auto conv3_ct = new Conv2D<i64, scale>(64, 64, 5, 1);
    auto fc1_ct = new FC<i64, scale>(64, 10);

    conv1_ct->filter.copy(conv1->filter);
    conv1_ct->bias.copy(conv1->bias);
    conv2_ct->filter.copy(conv2->filter);
    conv2_ct->bias.copy(conv2->bias);
    conv3_ct->filter.copy(conv3->filter);
    conv3_ct->bias.copy(conv3->bias);
    fc1_ct->weight.copy(fc1->weight);
    fc1_ct->bias.copy(fc1->bias);
    
    auto model_ct = Sequential<i64>({
        conv1_ct,
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        conv2_ct,
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        conv3_ct,
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        new Flatten<i64>(),
        fc1_ct,
        new Truncate<i64>(scale),
    });

    Tensor4D<u64> trainImage(bs, 32, 32, 3); // 1 images with server and 1 with client
    trainImage.fill((1ULL<<(scale+1)));
    Tensor4D<i64> trainImage_ct(bs, 32, 32, 3);
    trainImage_ct.copy(trainImage);
    Tensor4D<u64> e(bs, 10, 1, 1); // 1 images with server and 1 with client
    Tensor4D<i64> e_ct(bs, 10, 1, 1);

    LlamaVersion::initializeWeights(model); // dealer initializes the weights and sends to the parties
    LlamaVersion::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    EndComputation();
    LlamaVersion::output(model.activation);
    if (LlamaConfig::party != 1) {
        std::cout << "Secure Computation Output = \n";
        model.activation.print<i64>();
    }
    LlamaVersion::finalize();

    if (LlamaConfig::party == 1) {
        std::cout << "Cleartext Computation Output = \n";
        model_ct.forward(trainImage_ct);
        model_ct.activation.print<i64>();
    }
}

int main(int argc, char** argv) {
    prngWeights.SetSeed(osuCrypto::toBlock(time(NULL)));
    prng.SetSeed(osuCrypto::toBlock(time(NULL)));
#ifdef NDEBUG
    std::cout << "> Release Build" << std::endl;
#else
    std::cout << "> Debug Build" << std::endl;
#endif
    std::cout << "> Eigen will use " << Eigen::nbThreads() << " threads" << std::endl;
    // threelayer_keysize_llama();
    // load_mnist();
    // lenet_int();
    // lenet_float();
    // cifar10_int();

    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    llama_test_3layer(party);
    // llama_relutruncate_test(party);
    // llama_relu2round_test(party);
    // llama_relu_old_test(party);

    // cifar10_float_test();
}