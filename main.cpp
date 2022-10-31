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

void threelayer_keysize_llama() {

    const u64 scale = 16;
    LlamaKey<i64>::verbose = true;
    LlamaKey<i64>::probablistic = true;
    LlamaKey<i64>::bw = 64;
    const u64 minibatch = 100;

    auto model = Sequential<i64>({
        new Conv2D<i64, scale, LlamaKey<i64>>(3, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
        new Conv2D<i64, scale, LlamaKey<i64>>(64, 64, 5, 1),
        new ReLUTruncate<i64, LlamaKey<i64>>(scale),
        new MaxPool2D<i64, LlamaKey<i64>>(3, 0, 2),
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


void llama_test_vgg(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto model = Sequential<u64>({
        new Conv2D<u64, scale, Llama<u64>>(3, 64, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(64, 64, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(64, 128, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(128, 128, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(128, 256, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(256, 256, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(256, 256, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(256, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new Conv2D<u64, scale, Llama<u64>>(512, 512, 3, 1),
        new SumPool2D<u64, scale, Llama<u64>>(2, 0, 2),
        new ReLUTruncate<u64, Llama<u64>>(scale+2),
        new Flatten<u64>(),
        new FC<u64, scale, Llama<u64>>(512, 256),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new FC<u64, scale, Llama<u64>>(256, 256),
        new ReLUTruncate<u64, Llama<u64>>(scale),
        new FC<u64, scale, Llama<u64>>(256, 10),
        new Truncate<u64, Llama<u64>>(scale),
    });

    // Tensor4D<u64> trainImage(2, 1, 2, 1); // 1 images with server and 1 with client
    Tensor4D<u64> trainImage(2, 32, 32, 3); // 1 images with server and 1 with client
    Tensor4D<u64> e(2, 10, 1, 1); // 1 images with server and 1 with client

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    pirhana_softmax(model.activation, e, scale);
    EndComputation();
    Llama<u64>::output(e);
    Llama<u64>::finalize();
    if (LlamaConfig::party != 1)
        e.print();
}


void llama_test_small(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    Llama<u64>::init();
    auto rt = new ReLUTruncate<u64, Llama<u64>>(scale);
    auto fc = new FC<u64, scale, Llama<u64>>(3, 3);
    auto model = Sequential<u64>({
        fc,
        rt
    });

    auto fc_ct = new FC<i64, scale>(3, 3);
    fc_ct->weight.copy(fc->weight);
    fc_ct->bias.copy(fc->bias);
    auto rt_ct = new ReLUTruncate<i64>(scale);

    auto model_ct = Sequential<i64>({
        fc_ct,
        rt_ct
    });

    // Tensor4D<u64> trainImage(2, 1, 2, 1); // 1 images with server and 1 with client
    Tensor4D<u64> trainImage(2, 3, 1, 1); // 1 images with server and 1 with client
    trainImage(0, 0, 0, 0) = (1ULL<<(scale+1));
    trainImage(0, 1, 0, 0) = (1ULL<<(scale+1));
    trainImage(0, 2, 0, 0) = (1ULL<<(scale+1));
    trainImage(1, 0, 0, 0) = (1ULL<<(scale+1));
    trainImage(1, 1, 0, 0) = -(1ULL<<(scale+1));
    trainImage(1, 2, 0, 0) = (1ULL<<(scale+1));
    Tensor4D<i64> trainImage_ct(2, 3, 1, 1);
    trainImage_ct.copy(trainImage);
    Tensor4D<u64> e(2, 3, 1, 1); // 1 images with server and 1 with client
    Tensor4D<i64> e_ct(2, 3, 1, 1);

    Llama<u64>::initializeWeights(model); // dealer initializes the weights and sends to the parties
    Llama<u64>::initializeData(trainImage, 1); // takes input from stdin
    StartComputation();
    model.forward(trainImage);
    pirhana_softmax(model.activation, e, scale);
    model.backward(e);
    EndComputation();
    // Llama<u64>::output(rt->drelu);
    // Llama<u64>::output(model.activation);
    // Llama<u64>::output(e);
    Llama<u64>::output(fc->weight);
    Llama<u64>::output(fc->bias);
    if (LlamaConfig::party != 1) {
        // rt->drelu.print();
        // model.activation.print<i64>();
        // e.print<i64>(); // eprint hehe
        fc->weight.print<i64>();
        fc->bias.print<i64>();
    }
    Llama<u64>::finalize();

    // comparison with ct
    model_ct.forward(trainImage_ct);
    pirhana_softmax_ct(model_ct.activation, e_ct, scale);
    model_ct.backward(e_ct);
    if (LlamaConfig::party == 1) {
        // model_ct.activation.print();
        // e_ct.print();
        fc_ct->weight.print();
        fc_ct->bias.print();
    }
}

void cifar10_float_test() {
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 trainLen = dataset.training_images.size();
    const u64 testLen = dataset.test_images.size();
    const u64 batchSize = 100;
    const u64 scale = 12;

    auto conv1 = new Conv2D<double, 0>(3, 64, 5, 1);
    auto conv2 = new Conv2D<double, 0>(64, 64, 5, 1);
    auto conv3 = new Conv2D<double, 0>(64, 64, 5, 1);
    auto fc1 = new FC<double, 0>(64, 10);
    auto rt3 = new ReLU<double>();
    auto model = Sequential<double>({
        /// 3 Layer from Gupta et al
        conv1,
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        conv2,
        new ReLU<double>(),
        new MaxPool2D<double>(3, 0, 2),
        conv3,
        rt3,
        new MaxPool2D<double>(3, 0, 2),
        new Flatten<double>(),
        fc1,
    });

    auto conv1_fp = new Conv2D<i64, scale>(3, 64, 5, 1);
    auto conv2_fp = new Conv2D<i64, scale>(64, 64, 5, 1);
    auto conv3_fp = new Conv2D<i64, scale>(64, 64, 5, 1);
    auto fc1_fp = new FC<i64, scale>(64, 10);
    auto rt3_fp = new ReLUTruncate<i64>(scale);
    auto t_last = new Truncate<i64>(scale);
    auto model_fp = Sequential<i64>({
        /// 3 Layer from Gupta et al
        conv1_fp,
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        conv2_fp,
        new ReLUTruncate<i64>(scale),
        new MaxPool2D<i64>(3, 0, 2),
        conv3_fp,
        rt3_fp,
        new MaxPool2D<i64>(3, 0, 2),
        new Flatten<i64>(),
        fc1_fp,
        t_last,
    });

    for(int i = 0; i < conv1_fp->filter.d1; ++i) {
        for(int j = 0; j < conv1_fp->filter.d2; ++j) {
            conv1_fp->filter(i, j) = conv1->filter(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < conv1_fp->bias.size; ++i) {
        conv1_fp->bias(i) = conv1->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    for(int i = 0; i < conv2_fp->filter.d1; ++i) {
        for(int j = 0; j < conv2_fp->filter.d2; ++j) {
            conv2_fp->filter(i, j) = conv2->filter(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < conv2_fp->bias.size; ++i) {
        conv2_fp->bias(i) = conv2->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    for(int i = 0; i < conv3_fp->filter.d1; ++i) {
        for(int j = 0; j < conv3_fp->filter.d2; ++j) {
            conv3_fp->filter(i, j) = conv3->filter(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < conv3_fp->bias.size; ++i) {
        conv3_fp->bias(i) = conv3->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    for(int i = 0; i < fc1_fp->weight.d1; ++i) {
        for(int j = 0; j < fc1_fp->weight.d2; ++j) {
            fc1_fp->weight(i, j) = fc1->weight(i, j) * (1ULL<<scale) + rand_float();
        }
    }

    for(int i = 0; i < fc1_fp->bias.size; ++i) {
        fc1_fp->bias(i) = fc1->bias(i) * (1ULL<<(2*scale)) + rand_float();
    }

    Tensor4D<double> trainImage(batchSize, 32, 32, 3);
    Tensor4D<i64> trainImage_fp(batchSize, 32, 32, 3);
    Tensor4D<double> e(batchSize, 10, 1, 1);
    Tensor4D<i64> e_fp(batchSize, 10, 1, 1);
    for(u64 i = 0; i < trainLen; i += batchSize) {
        for(u64 b = 0; b < batchSize; ++b) {
            for(u64 j = 0; j < 32; ++j) {
                for(u64 k = 0; k < 32; ++k) {
                    for(u64 l = 0; l < 3; ++l) {
                        trainImage(b, j, k, l) = dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0;
                        trainImage_fp(b, j, k, l) = (dataset.training_images[i+b][j * 32 + k + l * 32 * 32] / 255.0) * (1ULL<<(scale));
                    }
                }
            }
        }

        model.forward(trainImage);
        model_fp.forward(trainImage_fp);
        softmax<double, 0>(model.activation, e);
        softmax<i64, scale>(model_fp.activation, e_fp);
        for(u64 b = 0; b < batchSize; ++b) {
            e(b, dataset.training_labels[i+b], 0, 0) -= (1.0/batchSize);
            e_fp(b, dataset.training_labels[i+b], 0, 0) -= (1ULL<<(scale))/batchSize;
        }
        model.backward(e);
        model_fp.backward(e_fp);
        printprogress(((double)i) / trainLen);
    }

    std::cout << "FP = \n";
    std::cout << fc1_fp->weight(0, 0) / ((double)(1ULL<<scale)) << " ";
    std::cout << fc1_fp->weight(0, 1) / ((double)(1ULL<<scale)) << " ";
    std::cout << fc1_fp->weight(0, 2) / ((double)(1ULL<<scale)) << " ";
    std::cout << fc1_fp->weight(0, 3) / ((double)(1ULL<<scale)) << std::endl;

    std::cout << conv1_fp->filter(0, 0) / ((double)(1ULL<<scale)) << " ";
    std::cout << conv1_fp->filter(0, 1) / ((double)(1ULL<<scale)) << " ";
    std::cout << conv1_fp->filter(0, 2) / ((double)(1ULL<<scale)) << " ";
    std::cout << conv1_fp->filter(0, 3) / ((double)(1ULL<<scale)) << std::endl;

    std::cout << "Float = \n";
    std::cout << fc1->weight(0, 0) << " ";
    std::cout << fc1->weight(0, 1) << " ";
    std::cout << fc1->weight(0, 2) << " ";
    std::cout << fc1->weight(0, 3) << std::endl;

    std::cout << conv1->filter(0, 0) << " ";
    std::cout << conv1->filter(0, 1) << " ";
    std::cout << conv1->filter(0, 2) << " ";
    std::cout << conv1->filter(0, 3) << std::endl;
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
    // llama_test_vgg(party);
    llama_test_small(party);

    // cifar10_float_test();
}