#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_improved.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <sytorch/utils.h>

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

template <typename T>
class ResNet9: public SytorchModule<T> {
    using SytorchModule<T>::add;
public:
    Conv2D<T> *conv1;
    BatchNorm2dInference<T> *bn1;
    ReLU<T> *relu1;
    Conv2D<T> *conv2;
    BatchNorm2dInference<T> *bn2;
    ReLU<T> *relu2;
    MaxPool2D<T> *maxpool;
    Conv2D<T> *conv3;
    BatchNorm2dInference<T> *bn3;
    ReLU<T> *relu3;
    Conv2D<T> *conv4;
    BatchNorm2dInference<T> *bn4;
    ReLU<T> *relu4;
    Conv2D<T> *conv5;
    BatchNorm2dInference<T> *bn5;
    ReLU<T> *relu5;
    MaxPool2D<T> *maxpool2;
    Conv2D<T> *conv6;
    BatchNorm2dInference<T> *bn6;
    ReLU<T> *relu6;
    MaxPool2D<T> *maxpool3;
    Conv2D<T> *conv7;
    BatchNorm2dInference<T> *bn7;
    ReLU<T> *relu7;
    Conv2D<T> *conv8;
    BatchNorm2dInference<T> *bn8;
    ReLU<T> *relu8;
    MaxPool2D<T> *maxpool4;
    Flatten<T> *flatten;
    FC<T> *fc;

public:
    ResNet9()
    {
        conv1 = new Conv2D<T>(3, 64, 3, 1);
        bn1 = new BatchNorm2dInference<T>(64);
        relu1 = new ReLU<T>();
        
        conv2 = new Conv2D<T>(64, 128, 3, 1);
        bn2 = new BatchNorm2dInference<T>(128);
        relu2 = new ReLU<T>();
        maxpool = new MaxPool2D<T>(2, 0, 2);

        conv3 = new Conv2D<T>(128, 128, 3, 1);
        bn3 = new BatchNorm2dInference<T>(128);
        relu3 = new ReLU<T>();
        conv4 = new Conv2D<T>(128, 128, 3, 1);
        bn4 = new BatchNorm2dInference<T>(128);
        relu4 = new ReLU<T>();

        conv5 = new Conv2D<T>(128, 256, 3, 1);
        bn5 = new BatchNorm2dInference<T>(256);
        relu5 = new ReLU<T>();
        maxpool2 = new MaxPool2D<T>(2, 0, 2);

        conv6 = new Conv2D<T>(256, 512, 3, 1);
        bn6 = new BatchNorm2dInference<T>(512);
        relu6 = new ReLU<T>();
        maxpool3 = new MaxPool2D<T>(2, 0, 2);

        conv7 = new Conv2D<T>(512, 512, 3, 1);
        bn7 = new BatchNorm2dInference<T>(512);
        relu7 = new ReLU<T>();
        conv8 = new Conv2D<T>(512, 512, 3, 1);
        bn8 = new BatchNorm2dInference<T>(512);
        relu8 = new ReLU<T>();

        maxpool4 = new MaxPool2D<T>(4, 0, 4);
        flatten = new Flatten<T>();
        fc = new FC<T>(512, 10);
    }

    Tensor4D<T>& _forward(Tensor4D<T> &input)
    {
        // conv block
        auto &var1 = conv1->forward(input, false);
        auto &var2 = bn1->forward(var1, false);
        auto &var3 = relu1->forward(var2, false);

        // conv block
        auto &var4 = conv2->forward(var3, false);
        auto &var5 = bn2->forward(var4, false);
        auto &var6 = relu2->forward(var5, false);

        // maxpool
        auto &var7 = maxpool->forward(var6, false);

        // res block
        auto &var8 = conv3->forward(var7, false);
        auto &var9 = bn3->forward(var8, false);
        auto &var10 = relu3->forward(var9, false);
        auto &var11 = conv4->forward(var10, false);
        auto &var12 = bn4->forward(var11, false);
        auto &var13 = relu4->forward(var12, false);
        auto var14 = add(var7, var13);

        // conv block
        auto &var15 = conv5->forward(var14, false);
        auto &var16 = bn5->forward(var15, false);
        auto &var17 = relu5->forward(var16, false);

        // maxpool
        auto &var18 = maxpool2->forward(var17, false);

        // conv block
        auto &var19 = conv6->forward(var18, false);
        auto &var20 = bn6->forward(var19, false);
        auto &var21 = relu6->forward(var20, false);

        // maxpool
        auto &var22 = maxpool3->forward(var21, false);

        // res block
        auto &var23 = conv7->forward(var22, false);
        auto &var24 = bn7->forward(var23, false);
        auto &var25 = relu7->forward(var24, false);
        auto &var26 = conv8->forward(var25, false);
        auto &var27 = bn8->forward(var26, false);
        auto &var28 = relu8->forward(var27, false);
        auto var29 = add(var22, var28);

        // maxpool
        auto &var30 = maxpool4->forward(var29, false);

        // flatten
        auto &var31 = flatten->forward(var30, false);

        // fc
        auto &var32 = fc->forward(var31, false);

        return var32;
    }
};

template <typename T>
class Net : public SytorchModule<T>
{
    using SytorchModule<T>::concat;
    ReLU<T> *relu1;
    Identity<T> *iden;

public:
    Net()
    {
        relu1 = new ReLU<T>();
        iden = new Identity<T>();
    }

    Tensor4D<T>& _forward(Tensor4D<T> &input)
    {
        auto &var1 = relu1->forward(input, false);
        auto var2 = concat(input, var1);
        auto &var3 = iden->forward(var2, false);
        return var3;
    }
};

void module_test_clear()
{
    const u64 scale = 12;
    ResNet9<i64> resnet;
    // resnet.init(1, 32, 32, 3, scale);
    resnet.init(scale);
    resnet.load("cifar10_resnet9-float.dat");
    Tensor4D<i64> input(1, 32, 32, 3);
    input.fill(1LL << scale);
    // Tensor4D<i64>::trackAllocations = true;
    auto &res = resnet.forward(input);
    // Tensor4D<i64>::trackAllocations = false;
    resnet.activation.print();
}

void mini_test()
{
    const u64 scale = 12;
    Net<i64> net;
    net.init(scale);
    print_dot_graph(net.root);
    Tensor4D<i64> input(1, 1, 1, 2);
    input(0, 0, 0, 0) = -5;
    input(0, 0, 0, 1) = 5;
    auto &res = net.forward(input);
    net.activation.print();
}

void module_test_llama(int party)
{
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 12;
    LlamaConfig::bitlength = 32;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    llama->init(ip, true);

    ResNet9<u64> resnet;
    // resnet.init(1, 32, 32, 3, scale);
    resnet.init(scale);
    resnet.setBackend(llama);
    resnet.optimize();
    if (party != 1) {
        resnet.load("cifar10_resnet9-float.dat");
    }
    else {
        resnet.zero();
    }

    llama::start();
    Tensor4D<u64> input(1, 32, 32, 3);
    input.fill(1LL << scale);
    llama->inputA(input);

    resnet.forward(input);
    llama::end();
    auto &output = resnet.activation;
    llama->output(output);
    if (party != 1) {
        blprint(output, LlamaConfig::bitlength - scale);
    }
    llama->finalize();
}


int main(int argc, char** argv) {
    fptraining_init();

    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    if (party == 0) {
        module_test_clear();
        mini_test();
    }
    else {
        module_test_llama(party);
    }

}