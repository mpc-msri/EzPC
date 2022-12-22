#define USE_CLEARTEXT

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "layers.h"
#include "softmax.h"
#include "networks.h"
#include "backend/llama_extended.h"
#include "backend/llama.h"

void llama_test_3layer(int party) {
    using LlamaVersion = Llama<u64>;
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaVersion::init("172.31.45.85", false);
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
    e.fill(1ULL<<scale);
    e_ct.copy(e);
    if (LlamaConfig::party == 1) {
        e_ct.fill(0);
    }

    LlamaVersion::initializeWeights(model); // dealer initializes the weights and sends to the parties
    LlamaVersion::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    EndComputation();
    model.backward(e);
    EndComputation();
    LlamaVersion::output(model.activation);
    LlamaVersion::output(conv1->bias);
    LlamaVersion::output(conv1->filter);
    LlamaVersion::output(conv1->filterGrad);
    if (LlamaConfig::party != 1) {
        std::cout << "Secure Computation Output = \n";
        model.activation.print<i64>();
        conv1->bias.print<i64>();
        conv1->filter.print<i64>();
        conv1->filterGrad.print<i64>();
    }
    LlamaVersion::finalize();

    if (LlamaConfig::party == 1) {
        std::cout << "Cleartext Computation Output = \n";
        model_ct.forward(trainImage_ct);
        model_ct.backward(e_ct);
        model_ct.activation.print<i64>();
        conv1_ct->bias.print<i64>();
        conv1_ct->filter.print<i64>();
        conv1_ct->filterGrad.print<i64>();
    }
}

void fptraining_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(time(NULL)));
    prng.SetSeed(osuCrypto::toBlock(time(NULL)));
    // set floating point precision
    std::cout << std::fixed  << std::setprecision(1);
#ifdef NDEBUG
    std::cout << "> Release Build" << std::endl;
#else
    std::cout << "> Debug Build" << std::endl;
#endif
    std::cout << "> Eigen will use " << Eigen::nbThreads() << " threads" << std::endl;
}

int main(int argc, char** argv) {
    fptraining_init();
    // lenet_int();
    // lenet_float();
    // threelayer_int();
    // threelayer_float();

    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    // llama_test_3layer(party);

}