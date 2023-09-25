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

void llama_test_vgg_imgnet(int party) {
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    llama->init("172.31.45.174", true);
    if (party != 1) {
        secfloat_init(party - 1, "172.31.45.174");
    }
    const u64 bs = 16;

    using T = u64;
    auto model = Sequential<T>({
        new Conv2D<T>(3, 64, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(64, 64, 3, 1),
        new MaxPool2D<T>(2, 0, 2),
        new ReLU<T>(),
        new Conv2D<T>(64, 128, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(128, 128, 3, 1),
        new MaxPool2D<T>(2, 0, 2),
        new ReLU<T>(),
        new Conv2D<T>(128, 256, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(256, 256, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(256, 256, 3, 1),
        new MaxPool2D<T>(2, 0, 2),
        new ReLU<T>(),
        new Conv2D<T>(256, 512, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(512, 512, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(512, 512, 3, 1),
        new MaxPool2D<T>(2, 0, 2),
        new ReLU<T>(),
        new Conv2D<T>(512, 512, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(512, 512, 3, 1),
        new ReLU<T>(),
        new Conv2D<T>(512, 512, 3, 1),
        new MaxPool2D<T>(2, 0, 2),
        new ReLU<T>(),
        new Flatten<T>(),
        new FC<T>(512 * 7 * 7, 4096),
        new ReLU<T>(),
        new FC<T>(4096, 4096),
        new ReLU<T>(),
        new FC<T>(4096, 1000),
    });
    model.init(bs, 224, 224, 3, scale);
    model.setBackend(llama);
    model.optimize();

    Tensor4D<u64> trainImages(bs, 224, 224, 3);
    Tensor<u64> trainLabels(bs);
    trainImages.fill(1);
    trainLabels.fill(1);
    Tensor4D<u64> e(bs, 1000, 1, 1);
    // trainImage.fill(1);

    llama->initializeWeights(model); // dealer initializes the weights and sends to the parties
    llama::start();
    llama->inputA(trainImages);
    model.forward(trainImages);
    // llama->output(model.activation);
    // if (party != 1)
    //     model.activation.print();
    llama::end();
    llama->finalize();
}

void llama_test_3layer(int party) {
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
    llama->init(ip, true);
    if (party != 1) {
        secfloat_init(party - 1, ip);
    }
    const u64 bs = 2;
    
    auto conv1 = new Conv2D<u64>(3, 64, 5, 1);
    auto conv2 = new Conv2D<u64>(64, 64, 5, 1);
    auto conv3 = new Conv2D<u64>(64, 64, 5, 1);
    auto fc1 = new FC<u64>(64, 10);
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
    llama->output(conv1->filter);
    llama->output(conv2->filter);
    llama->output(conv3->filter);
    llama->output(fc1->weight);
    llama->output(conv1->bias);
    llama->output(conv2->bias);
    llama->output(conv3->bias);
    llama->output(fc1->bias);
    // llama->output(model.activation);
    if (LlamaConfig::party != 1) {
        conv1->filter.print<i64>();
        conv2->filter.print<i64>();
        conv3->filter.print<i64>();
        fc1->weight.print<i64>();
        conv1->bias.print<i64>();
        conv2->bias.print<i64>();
        conv3->bias.print<i64>();
        fc1->bias.print<i64>();
        // model.activation.print<i64>();
    }
    llama->finalize();
}

void llama_test_lenet_gupta(int party) {
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
    llama->init(ip, true);
    if (party != 1) {
        secfloat_init(party - 1, ip);
    }
    const u64 bs = 100;

    auto model = Sequential<u64>({
        new Conv2D<u64>(1, 8, 5),
        new ReLU<u64>(),
        new MaxPool2D<u64>(2),
        new Conv2D<u64>(8, 16, 5),
        new ReLU<u64>(),
        new MaxPool2D<u64>(2),
        new Flatten<u64>(),
        new FC<u64>(256, 128),
        new ReLU<u64>(),
        new FC<u64>(128, 10),
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

void llama_test_lenet_minionn(int party) {
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    llama->init("172.31.45.174", true);
    if (party != 1) {
        secfloat_init(party - 1, "172.31.45.174");
    }
    const u64 bs = 100;

    auto model = Sequential<u64>({
        new Conv2D<u64>(1, 16, 5),
        new ReLU<u64>(),
        new MaxPool2D<u64>(2),
        new Conv2D<u64>(16, 16, 5),
        new ReLU<u64>(),
        new MaxPool2D<u64>(2),
        new Flatten<u64>(),
        new FC<u64>(256, 100),
        new ReLU<u64>(),
        new FC<u64>(100, 10),
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

void llama_test_falcon_alex(int party) {
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    llama->init("172.31.45.174", true);
    if (party != 1) {
        secfloat_init(party - 1, "172.31.45.174");
    }
    const u64 bs = 100;

    auto model = Sequential<u64>({
        new Conv2D<u64>(3, 96, 11, 9, 4),
        new MaxPool2D<u64>(3, 0, 2),
        new ReLU<u64>(),
        new Conv2D<u64>(96, 256, 5, 1, 1),
        new ReLU<u64>(),
        new MaxPool2D<u64>(2, 0, 1),
        new Conv2D<u64>(256, 384, 3, 1, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(384, 384, 3, 1, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(384, 256, 3, 1, 1),
        new ReLU<u64>(),
        new Flatten<u64>(),
        new FC<u64>(256, 256),
        new ReLU<u64>(),
        new FC<u64>(256, 256),
        new ReLU<u64>(),
        new FC<u64>(256, 10),
    });
    model.init(bs, 32, 32, 3, scale);
    model.setBackend(llama);
    model.optimize();

    Tensor4D<u64> trainImages(bs, 32, 32, 3);
    Tensor<u64> trainLabels(bs);
    trainImages.fill(1);
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

void ct_test_3layer() {
    srand(time(NULL));
    const u64 scale = 24;
    const u64 bs = 2;

    auto conv1 = new Conv2D<i64>(3, 64, 5, 1);
    auto conv2 = new Conv2D<i64>(64, 64, 5, 1);
    auto conv3 = new Conv2D<i64>(64, 64, 5, 1);
    auto fc1 = new FC<i64>(64, 10);
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
    conv1->filter.print<i64>();
    conv2->filter.print<i64>();
    conv3->filter.print<i64>();
    fc1->weight.print<i64>();
    conv1->bias.print<i64>();
    conv2->bias.print<i64>();
    conv3->bias.print<i64>();
    fc1->bias.print<i64>();
    // e.print<i64>();
    // conv2->filter.print<i64>();
    // conv3->filter.print<i64>();
}

void llama_test_pvgg(int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    llama->init("127.0.0.1", false);
    const u64 bs = 128;

    // this model is no more correct after removing Truncate and RT layers
    // the layers need a scale that is modifyable
    auto model = Sequential<u64>({
        new Conv2D<u64>(3, 64, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(64, 64, 3, 1),
        new SumPool2D<u64>(2, 0, 2),
        new ReLU<u64>(),
        new Conv2D<u64>(64, 128, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(128, 128, 3, 1),
        new SumPool2D<u64>(2, 0, 2),
        new ReLU<u64>(),
        new Conv2D<u64>(128, 256, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(256, 256, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(256, 256, 3, 1),
        new SumPool2D<u64>(2, 0, 2),
        new ReLU<u64>(),
        new Conv2D<u64>(256, 512, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(512, 512, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(512, 512, 3, 1),
        new SumPool2D<u64>(2, 0, 2),
        new ReLU<u64>(),
        new Conv2D<u64>(512, 512, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(512, 512, 3, 1),
        new ReLU<u64>(),
        new Conv2D<u64>(512, 512, 3, 1),
        new SumPool2D<u64>(2, 0, 2),
        new ReLU<u64>(),
        new Flatten<u64>(),
        new FC<u64>(512, 256),
        new ReLU<u64>(),
        new FC<u64>(256, 256),
        new ReLU<u64>(),
        new FC<u64>(256, 10),
    });
    model.init(bs, 32, 32, 3, scale);
    model.setBackend(llama);

    Tensor4D<u64> trainImage(bs, 32, 32, 3); // 1 images with server and 1 with client
    trainImage.fill((1ULL<<(scale+1)));
    Tensor4D<u64> e(bs, 10, 1, 1); // 1 images with server and 1 with client
    e.fill(1ULL<<scale);
    if (LlamaConfig::party == 1) {
        e.fill(0);
    }

    llama->initializeWeights(model); // dealer initializes the weights and sends to the parties
    llama->initializeData(trainImage, 1); // takes input from stdin
    llama::start();
    model.forward(trainImage);
    llama::end();
    model.backward(e);
    llama::end();
    // LlamaVersion::output(model.activation);
    // LlamaVersion::output(conv1->bias);
    // LlamaVersion::output(conv1->filter);
    // LlamaVersion::output(conv1->filterGrad);
    // // if (LlamaConfig::party != 1) {
    // //     std::cout << "Secure Computation Output = \n";
    // //     model.activation.print<i64>();
    // //     conv1->bias.print<i64>();
    // //     conv1->filter.print<i64>();
    // //     conv1->filterGrad.print<i64>();
    // // }
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

void llama_floattofix_test(int party)
{
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1",false);
    int numClasses = 1;
    int batchSize = 1;
    if (party != 1) {
        secfloat_init(party - 1, "127.0.0.1");
    }

    Tensor4D<u64> e(batchSize, numClasses, 1, 1);

    Tensor4D<u64> e_ct(batchSize, numClasses, 4, 1);
    e.fill(0);
    e_ct.fill(0);
    Tensor4D<u64> y(batchSize, numClasses, 1, 1);
    Tensor4D<u64> y_ct(batchSize, numClasses, 4, 1);
    y.fill(0);

    for(int i=0;i<numClasses;++i)
    {
        e(0,i,0,0) =1;
        e(1,i,0,0) = 0;
        e_ct(0,i,0,0) = 1;
        e_ct(1,i,0,0) = 0;
    }
    float input = 0.5f;
    uint64_t inp_com[4];
    uint64_t inp_com1[4];
    uint64_t inp_com2[4];

    union FloatRep{
        float f;
        uint32_t i;
    };

    FloatRep rep;
    rep.f = input;

    inp_com[0] = (rep.i & 0x7FFFFF | 0x800000); //mantissa 0x800000
    inp_com[1] = (rep.i >> 23) & 0xFF; //exponent
    inp_com[2] = (rep.i >> 31) & 0x1; //sign
    inp_com[3] = 0; //zero

    std::cout<<"input: "<<input<<"\n"; 
    std::cout<<"zero comp"<<inp_com[3]<<"\n";
    std::cout<<"exp comp"<<inp_com[1]<<"\n";
    std::cout<<"sign comp"<<inp_com[2]<<"\n";
    std::cout<<"mant comp"<<inp_com[0]<<"\n";
    
    llama->inputA(e);
    auto inp1=splitShare(inp_com[0],24);
    auto inp2=splitShare(inp_com[1],8);

    inp_com1[0]=inp1.first;
    inp_com2[0]=inp1.second;
    inp_com1[1]=inp2.first;
    inp_com2[1]=inp2.second;
    inp_com2[2]=inp_com[2];
    inp_com1[2]=inp_com[2];
    inp_com1[3]=inp_com[3];
    inp_com2[3]=inp_com[3];

    std::cout<<"secure output: \n";
    llama::start();

    if(LlamaConfig::party==2)
    {
        FloatToFix(1,inp_com1,y.data,scale);
    }
    else if(LlamaConfig::party==3)
    {
        FloatToFix(1,inp_com2,y.data,scale);
    }
    else
    {
        FloatToFix(1,inp_com,y.data,scale);
    }
    //std::cout<<"cleartext output: \n";
    //FloatToFixCt(numClasses*batchSize,e_ct.data,y_ct.data,scale);
    //y_ct.print();
    
    //FloatToFix(1,inp_com,y.data,scale);
    llama::end();
    llama->output(y);
    llama->finalize();
    if(LlamaConfig::party != 1)
    {
        y.print();
    }

}
void llama_fixtofloat_test(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    // LlamaExtended<u64>::init("172.31.45.158");
    // LlamaExtended<u64>::init("172.31.45.85", false);
    llama->init("127.0.0.1", false);
    int numClasses = 10;
    int batchSize = 100;
    if (party != 1) {
        // secfloat_init(party - 1, "172.31.45.158");
        secfloat_init(party - 1, "127.0.0.1");
    }
    
    Tensor4D<u64> e(batchSize, numClasses, 1, 1);
    Tensor4D<i64> e_ct(batchSize, numClasses, 1, 1);
    e.fill(0);
    e_ct.fill(0);
    Tensor4D<u64> y(batchSize, numClasses, 1, 1);
    Tensor4D<i64> y_ct(batchSize, numClasses, 1, 1);
    
    for(int i = 0; i < numClasses; ++i) {
        e(0, i, 0, 0) = i * (1ULL << scale);
        e(1, i, 0, 0) = 5 * (1ULL << scale);
        e_ct(0, i, 0, 0) = i * (1LL << scale);
        e_ct(1, i, 0, 0) = 5 * (1LL << scale);
    }
    
    llama->inputA(e);

    if (LlamaConfig::party == 1) {
        softmax<i64, scale>(e_ct, y_ct);
        y_ct.print();
    }

    llama::start();
    softmax_secfloat(e, y, scale, party);
    llama::end();
    llama->output(y);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        y.print();
    }
}


void test_reluextend(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 1000;
    Tensor4D<u64> x(size, 1, 1, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    Tensor4D<u64> expected_drelu(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = prngWeights.get<u64>();
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = ((i64)x(i, 0, 0, 0)) >> scale;
            if (((i64)expected(i, 0, 0, 0)) < 0) {
                expected(i, 0, 0, 0) = 0;
                expected_drelu(i, 0, 0, 0) = 0;
            }
            else {
                expected_drelu(i, 0, 0, 0) = 1;
            }
        }
    }

    llama->inputA(x);
    // local truncate reduce
    for(int i = 0; i < size; ++i) {
        x(i, 0, 0, 0) = x(i, 0, 0, 0) >> scale;
    }

    llama::start();
    Tensor4D<u64> y(size, 1, 1, 1);
    Tensor4D<u64> drelu(size, 1, 1, 1);
    ReluExtend(size, 64 - scale, 64, x.data, y.data, drelu.data);
    llama::end();
    llama->output(y);
    llama->output(drelu);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)(y(i, 0, 0, 0) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
        for(int i = 0; i < size; ++i) {
            std::cout << ((drelu(i, 0, 0, 0) - expected_drelu(i, 0, 0, 0)) % 2) << " ";
        }
        std::cout << std::endl;
    }
}

void test_ars(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 1000;
    Tensor4D<u64> x(size, 1, 1, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = prngWeights.get<u64>();
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = ((i64)x(i, 0, 0, 0)) >> scale;
        }
    }

    llama->inputA(x);

    llama::start();
    ARS(size, x.data, x.data, x.data, x.data, scale);
    llama::end();
    llama->output(x);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)(x(i, 0, 0, 0) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
    }
}

void test_rt(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 1000;
    Tensor4D<u64> x(size, 1, 1, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = prngWeights.get<u64>();
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = ((i64)x(i, 0, 0, 0)) >> scale;
            if (((i64)expected(i, 0, 0, 0)) < 0) {
                expected(i, 0, 0, 0) = 0;
            }
        }
    }

    llama->inputA(x);

    llama::start();
    Tensor4D<u64> drelu(size, 1, 1, 1);
    ReluTruncate(size, x.data, x.data, x.data, x.data, scale, drelu.data);
    llama::end();
    llama->output(x);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)(x(i, 0, 0, 0) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
    }
}

void test_r2(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 1000;
    Tensor4D<u64> x(size, 1, 1, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = prngWeights.get<u64>();
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = x(i, 0, 0, 0);
            if (((i64)x(i, 0, 0, 0)) < 0) {
                expected(i, 0, 0, 0) = 0;
            }
        }
    }

    llama->inputA(x);

    llama::start();
    Tensor4D<u64> drelu(size, 1, 1, 1);
    Relu2Round(size, x.data, x.data, x.data, x.data, drelu.data, 64);
    llama::end();
    llama->output(x);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)(x(i, 0, 0, 0) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
    }
}

void test_mul(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 1000;
    Tensor4D<u64> x(size, 1, 1, 1);
    Tensor4D<u64> y(size, 1, 1, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = prngWeights.get<u64>();
            y(i, 0, 0, 0) = prngWeights.get<u64>();
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = x(i, 0, 0, 0) * y(i, 0, 0, 0);
        }
    }

    llama->inputA(x);

    llama::start();
    ElemWiseSecretSharedVectorMult(size, x.data, x.data, y.data, y.data, x.data, x.data);
    llama::end();
    llama->output(x);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)(x(i, 0, 0, 0) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
    }
}

void test_reluspline(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 1000;
    Tensor4D<u64> x(size, 1, 1, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = prngWeights.get<u64>();
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = x(i, 0, 0, 0);
            if (((i64)x(i, 0, 0, 0)) < 0) {
                expected(i, 0, 0, 0) = 0;
            }
        }
    }

    llama->inputA(x);

    llama::start();
    Tensor4D<u64> drelu(size, 1, 1, 1);
    Relu(size, x.data, x.data, x.data, x.data, drelu.data);
    llama::end();
    llama->output(x);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)(x(i, 0, 0, 0) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
    }
}

void test_maxpool(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    const int bl = 40;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = bl;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 10;
    Tensor4D<u64> x(size, 2, 2, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = (1);
            x(i, 0, 1, 0) = (-4);
            x(i, 1, 0, 0) = (7);
            x(i, 1, 1, 0) = (5);
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = std::max(
                                        std::max(((i64)x(i, 0, 0, 0)), ((i64)x(i, 0, 1, 0))), 
                                        std::max(((i64)x(i, 1, 0, 0)), ((i64)x(i, 1, 1, 0)))
                                    );
        }
    }

    llama->inputA(x);

    llama::start();
    Tensor4D<u64> y(size, 1, 1, 1);
    Tensor4D<u64> maxbit(size * 3, 1, 1, 1);
    MaxPoolDouble(size, 1, 1, 1, 2, 2, 0, 0, 0, 0, 2, 2, size, 2, 2, 1, x.data, x.data, y.data, y.data, maxbit.data);
    llama::end();
    llama->output(y);
    llama->output(maxbit);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)((y(i, 0, 0, 0) % (1ULL << bl)) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
        for(int i = 0; i < size; ++i) {
            std::cout << (maxbit(0 * size + i, 0, 0, 0) % 2) << " ";
            std::cout << (maxbit(1 * size + i, 0, 0, 0) % 2) << " ";
            std::cout << (maxbit(2 * size + i, 0, 0, 0) % 2) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}


void test_signextend(int party) {
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    LlamaExtended<u64>* llama = new LlamaExtended<u64>();
    llama->init("127.0.0.1", true);

    int size = 1000;
    Tensor4D<u64> x(size, 1, 1, 1);
    Tensor4D<u64> expected(size, 1, 1, 1);
    if (party != 1)
    {
        for(int i = 0; i < size; ++i) {
            x(i, 0, 0, 0) = prngWeights.get<u64>();
        }

        for(int i = 0; i < size; ++i) {
            expected(i, 0, 0, 0) = ((i64)x(i, 0, 0, 0)) >> scale;
        }
    }

    llama->inputA(x);
    // local truncate reduce
    for(int i = 0; i < size; ++i) {
        x(i, 0, 0, 0) = x(i, 0, 0, 0) >> scale;
    }

    llama::start();
    Tensor4D<u64> drelu(size, 1, 1, 1);
    SignExtend2(size, 64 - scale, 64, x.data, x.data);
    llama::end();
    llama->output(x);
    llama->finalize();
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << ((i64)(x(i, 0, 0, 0) - expected(i, 0, 0, 0))) << " ";
        }
        std::cout << std::endl;
    }
}

void microbenchmark_conv(int party) {
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    // std::string ip = "127.0.0.1";
    std::string ip = "172.31.45.174";
    llama->init(ip, true);
    const u64 bs = 100;

    auto conv = new Conv2D<u64>(64, 64, 5, 1);
    conv->doTruncationForward = false;
    auto model = Sequential<u64>({
        conv,
    });
    model.setBackend(llama);

    llama->initializeWeights(model); // dealer initializes the weights and sends to the parties
    llama::start();

    u64 imgSize = 64;
    model.init(bs, imgSize, imgSize, 64, scale);
    Tensor4D<u64> trainImages(bs, imgSize, imgSize, 64);
    // llama->inputA(trainImages);
    model.forward(trainImages);
    llama::end();
    llama->finalize();
}


void microbenchmark_rt_llamaext(int party) {
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    // std::string ip = "127.0.0.1";
    std::string ip = "172.31.45.174";
    llama->init(ip, true);

    auto relu = new ReLU<u64>();
    relu->doTruncationForward = true;
    auto model = Sequential<u64>({
        relu,
    });
    model.setBackend(llama);

    llama::start();
    u64 numrelu = 10000000;
    model.init(1, numrelu, 1, 1, scale);
    Tensor4D<u64> trainImages(1, numrelu, 1, 1);
    // llama->inputA(trainImages);
    model.forward(trainImages);
    llama::end();
    llama->finalize();
}

void microbenchmark_rt_llamaimp(int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    // std::string ip = "127.0.0.1";
    std::string ip = "172.31.45.174";
    llama->init(ip, true);

    auto relu = new ReLU<u64>();
    auto model = Sequential<u64>({
        relu,
    });
    model.setBackend(llama);

    llama::start();
    u64 numrelu = 10000;
    model.init(1, numrelu, 1, 1, scale);
    Tensor4D<u64> trainImages(1, numrelu, 1, 1);

    llama->truncateForward(trainImages, scale);
    model.forward(trainImages);
    llama::end();
    llama->finalize();
}

void microbenchmark_maxpool_llamaimp(int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    u64 bs = 8;
    // std::string ip = "127.0.0.1";
    std::string ip = "172.31.45.174";
    llama->init(ip, true);

    u64 imgDim = 128;
    u64 filtersize = 11;

    auto relu = new ReLU<u64>();
    auto maxpool = new MaxPool2D<u64>(filtersize, 0, 1);
    auto model = Sequential<u64>({
        maxpool,
        relu,
    });
    model.setBackend(llama);

    llama::start();
    model.init(bs, imgDim, imgDim, 64, scale);
    Tensor4D<u64> trainImages(bs, imgDim, imgDim, 64);

    llama->truncateForward(trainImages, scale);
    model.forward(trainImages);
    llama::end();
    llama->finalize();
}

void piranha_microbenchmark(int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    // std::string ip = "127.0.0.1";
    std::string ip = "172.31.45.174";
    llama->init(ip, true);

    llama::start();
    Tensor4D<u64> trainImages(128, 10, 1, 1);
    PiranhaSoftmax(128, 10, trainImages.data, trainImages.data, trainImages.data, trainImages.data, scale);
    llama::end();
    llama->finalize();
}

template <typename T>
void loadFloatWeights(Sequential<T> &model, const std::string weightsFile, u64 scale) {
    size_t size_in_bytes = std::filesystem::file_size(weightsFile);
    always_assert(size_in_bytes % 4 == 0); // as it's float
    size_t numParameters = size_in_bytes / 4;
    float *floatWeights = new float[numParameters];
    
    std::ifstream file(weightsFile, std::ios::binary);
    file.read((char*) floatWeights, size_in_bytes);
    file.close();
    
    size_t wIdx = 0;
    for(int i = 0; i < model.layers.size(); i++) {
        // std::cout << "Loading " << model.layers[i]->name << std::endl;
        if(model.layers[i]->name.find("Conv2D") != std::string::npos || model.layers[i]->name.find("FC") != std::string::npos) {
            auto& weights = model.layers[i]->getweights();

            for (int j = 0; j < weights.d1; j++) {
                for(int k = 0; k < weights.d2; ++k) {
                    weights(j, k) = floatWeights[wIdx + weights.d2 * j + k] * (1LL << scale);
                }
            }
            
            auto wSize = weights.d1 * weights.d2;
            wIdx += wSize;

            auto& bias = model.layers[i]->getbias();

            for (int j = 0; j < bias.size; ++j) {
                bias(j) = floatWeights[wIdx + j] * (1LL << (2*scale));
            }

            wSize = bias.size;
            wIdx += wSize;
        }
    }
    always_assert(wIdx == numParameters);
    delete[] floatWeights;
}

template <typename T>
inline void loadImage(Tensor4D<T> &img, const std::string imgFile, u64 scale) {
    size_t size_in_bytes = std::filesystem::file_size(imgFile);
    // std::cout << imgFile << " " << size_in_bytes << std::endl;
    always_assert(size_in_bytes == 4 * img.d1 * img.d2 * img.d3 * img.d4);
    
    size_t numFeature = size_in_bytes / 4;
    float *floatImg = new float[numFeature];
    std::ifstream file(imgFile, std::ios::binary);
    file.read((char*) floatImg, size_in_bytes);
    file.close();

    for(int i = 0; i < img.d1; ++i) {
        for(int j = 0; j < img.d2; ++j) {
            for(int k = 0; k < img.d3; ++k) {
                for(int l = 0; l < img.d4; ++l) {
                    img(i, j, k, l) = floatImg[i * img.d2 * img.d3 * img.d4 + j * img.d3 * img.d4 + k * img.d4 + l] * (1LL << scale);
                }
            }
        }
    }

    delete[] floatImg;
}

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

void ct_vgg_imgnet()
{
    using T = i32;
    const u64 scale = 12;
    int numThreads = omp_thread_count();
    std::cout << "using threads = " << numThreads << std::endl;
    std::vector<Sequential<T> *> models;
    for(int i = 0; i < numThreads; ++i) {
        auto model = new Sequential<T>({
            new Conv2D<T>(3, 64, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(64, 64, 3, 1),
            new MaxPool2D<T>(2, 0, 2),
            new ReLU<T>(),
            new Conv2D<T>(64, 128, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(128, 128, 3, 1),
            new MaxPool2D<T>(2, 0, 2),
            new ReLU<T>(),
            new Conv2D<T>(128, 256, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(256, 256, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(256, 256, 3, 1),
            new MaxPool2D<T>(2, 0, 2),
            new ReLU<T>(),
            new Conv2D<T>(256, 512, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(512, 512, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(512, 512, 3, 1),
            new MaxPool2D<T>(2, 0, 2),
            new ReLU<T>(),
            new Conv2D<T>(512, 512, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(512, 512, 3, 1),
            new ReLU<T>(),
            new Conv2D<T>(512, 512, 3, 1),
            new MaxPool2D<T>(2, 0, 2),
            new ReLU<T>(),
            new Flatten<T>(),
            new FC<T>(512 * 7 * 7, 4096),
            new ReLU<T>(),
            new FC<T>(4096, 4096),
            new ReLU<T>(),
            new FC<T>(4096, 1000),
        });
        model->init(1, 224, 224, 3, scale);
        loadFloatWeights(*model, "vgg16-imgnet-float.dat", scale);
        models.push_back(model);
    }

    std::vector<Tensor4D<T> *> images(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        images[i] = new Tensor4D<T>(1, 224, 224, 3);
    }

    std::ofstream outfiles[numThreads];
    for(int i = 0; i < numThreads; ++i) {
        outfiles[i] = std::ofstream("thread_outputs/thread-" + std::to_string(i));
    }

    #pragma omp parallel for
    for (int i = 0; i < 50000; ++i) {
        int tid = omp_get_thread_num();
        std::string imgFile = "/data/kanav/dataset/ImageNet_ValData/vgg16-preprocessed-imgnetval/" + std::to_string(i) + ".dat";
        loadImage(*images[tid], imgFile, scale);
        models[tid]->forward(*images[tid], false);
        outfiles[tid] << (i+1) << " " << models[tid]->activation.argmax(0) + 1 << std::endl;
    }
}

void softmax_microbenchmark(int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    llama->init("172.31.45.174", true);
    if (party != 1) {
        secfloat_init(party - 1, "172.31.45.174");
    }
    const u64 bs = 100;

    Tensor4D<u64> moutput(bs, 10, 1, 1);
    Tensor4D<u64> e(bs, 10, 1, 1);

    llama::start();
    softmax_secfloat(moutput, e, scale, party);
    llama::end();
    llama->finalize();
}

int main(int argc, char** argv) {
    fptraining_init();
    // lenet_int();
    // lenet_float();
    // threelayer_int();
    // threelayer_float();
    // lenet_pirhana_int();
    // piranha_vgg_int();
    // piranha_vgg_float();
    // alexnet_float();
    // secureml_int();

    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    // llama_test_3layer(party);
    //llama_test_lenet_gupta(party);
    // gpu_main(argc, argv);
    // carvanha_compile();
    // llama_test_pvgg(party);
    //llama_floattofix_test(party);
    llama_fixtofloat_test(party);
    // if (party == 0) {
    //     ct_test_3layer();
    // }
    // else {
    //     llama_test_3layer(party);
    // }
    // llama_test_vgg_imgnet(party);
    // llama_test_lenet_minionn(party);
    // llama_test_falcon_alex(party);
    // microbenchmark_conv(party);
    // test_maxpool(party);
    // test_reluextend(party);
    // test_signextend(party);
    // test_ars(party);
    // test_rt(party);
    // test_r2(party);
    // test_mul(party);
    // test_reluspline(party);
    // microbenchmark_maxpool_llamaimp(party);
    // piranha_microbenchmark(party);
    // ct_vgg_imgnet();
    // softmax_microbenchmark(party);

}