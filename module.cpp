#include <sytorch/backend/llama_extended.h>
#include <sytorch/layers/layers.h>
#include <filesystem>

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
class ResNet9 {
public:
    union {
        struct {
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
        };
        Layer<T> *layers[30];
    };

    Tensor4D<T> activation;
    Backend<T> *backend = new ClearText<T>;
    LayerTreeNode<T> *root = nullptr;

public:
    ResNet9() : activation(1, 10, 1, 1)
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

    void init(u64 scale)
    {
        conv1->init(1, 32, 32, 3, scale);
        bn1->init(1, 32, 32, 64, scale);
        relu1->init(1, 32, 32, 64, scale);
        
        conv2->init(1, 32, 32, 64, scale);
        bn2->init(1, 32, 32, 128, scale);
        relu2->init(1, 32, 32, 128, scale);
        
        maxpool->init(1, 32, 32, 128, scale);

        conv3->init(1, 16, 16, 128, scale);
        bn3->init(1, 16, 16, 128, scale);
        relu3->init(1, 16, 16, 128, scale);
        conv4->init(1, 16, 16, 128, scale);
        bn4->init(1, 16, 16, 128, scale);
        relu4->init(1, 16, 16, 128, scale);

        conv5->init(1, 16, 16, 128, scale);
        bn5->init(1, 16, 16, 256, scale);
        relu5->init(1, 16, 16, 256, scale);
        
        maxpool2->init(1, 16, 16, 256, scale);

        conv6->init(1, 8, 8, 256, scale);
        bn6->init(1, 8, 8, 512, scale);
        relu6->init(1, 8, 8, 512, scale);
        
        maxpool3->init(1, 8, 8, 512, scale);

        conv7->init(1, 4, 4, 512, scale);
        bn7->init(1, 4, 4, 512, scale);
        relu7->init(1, 4, 4, 512, scale);
        conv8->init(1, 4, 4, 512, scale);
        bn8->init(1, 4, 4, 512, scale);
        relu8->init(1, 4, 4, 512, scale);

        maxpool4->init(1, 4, 4, 512, scale);
        flatten->init(1, 1, 1, 512, scale);
        fc->init(1, 512, 1, 1, scale);

        Tensor4D<T> ip(1, 32, 32, 3);
        ip.treeDat->curr = new PlaceHolderLayer<T>("Input");
        Layer<T>::treeInit = true;
        auto &res = this->forward(ip);
        Layer<T>::treeInit = false;
        root = ip.treeDat;
        // print_dot_graph(ip.treeDat);

    }

    void zero()
    {
        conv1->filter.fill(0);
        conv1->bias.fill(0);
        bn1->A.fill(0);
        bn1->B.fill(0);
        conv2->filter.fill(0);
        conv2->bias.fill(0);
        bn2->A.fill(0);
        bn2->B.fill(0);
        conv3->filter.fill(0);
        conv3->bias.fill(0);
        bn3->A.fill(0);
        bn3->B.fill(0);
        conv4->filter.fill(0);
        conv4->bias.fill(0);
        bn4->A.fill(0);
        bn4->B.fill(0);
        conv5->filter.fill(0);
        conv5->bias.fill(0);
        bn5->A.fill(0);
        bn5->B.fill(0);
        conv6->filter.fill(0);
        conv6->bias.fill(0);
        bn6->A.fill(0);
        bn6->B.fill(0);
        conv7->filter.fill(0);
        conv7->bias.fill(0);
        bn7->A.fill(0);
        bn7->B.fill(0);
        conv8->filter.fill(0);
        conv8->bias.fill(0);
        bn8->A.fill(0);
        bn8->B.fill(0);
        fc->weight.fill(0);
        fc->bias.fill(0);
    }

    void setBackend(Backend<T> *b)
    {
        for (int i = 0; i < 30; ++i) {
            layers[i]->setBackend(b);
        }
        backend = b;
    }

    void loadFloatWeights(const std::string weightsFile, u64 scale) {
        size_t size_in_bytes = std::filesystem::file_size(weightsFile);
        always_assert(size_in_bytes % 4 == 0); // as it's float
        size_t numParameters = size_in_bytes / 4;
        float *floatWeights = new float[numParameters];
        
        std::ifstream file(weightsFile, std::ios::binary);
        file.read((char*) floatWeights, size_in_bytes);
        file.close();
        
        size_t wIdx = 0;
        for(int i = 0; i < 30; i++) {
            // std::cout << "Loading " << layers[i]->name << std::endl;
            if(layers[i]->name.find("Conv2D") != std::string::npos || layers[i]->name.find("FC") != std::string::npos) {
                auto& weights = layers[i]->getweights();

                for (int j = 0; j < weights.d1; j++) {
                    for(int k = 0; k < weights.d2; ++k) {
                        weights(j, k) = floatWeights[wIdx + weights.d2 * j + k] * (1LL << scale);
                    }
                }
                
                auto wSize = weights.d1 * weights.d2;
                wIdx += wSize;

                auto& bias = layers[i]->getbias();

                for (int j = 0; j < bias.size; ++j) {
                    bias(j) = floatWeights[wIdx + j] * (1LL << (2*scale));
                }

                wSize = bias.size;
                wIdx += wSize;
            }
            else if (layers[i]->name.find("BatchNorm2dInference") != std::string::npos) {
                auto bn = (BatchNorm2dInference<T>*) layers[i];
                auto channel = bn->A.size;
                auto gammaPtr = floatWeights + wIdx;
                auto betaPtr = floatWeights + wIdx + channel;
                auto meanPtr = floatWeights + wIdx + 2 * channel;
                auto varPtr = floatWeights + wIdx + 3 * channel;
                for (int j = 0; j < channel; ++j) {
                    bn->A(j) = (gammaPtr[j] / std::sqrt(varPtr[j])) * (1LL << scale);
                    bn->B(j) = (betaPtr[j] - gammaPtr[j] * meanPtr[j] / std::sqrt(varPtr[j])) * (1LL << (2 * scale));
                }
                wIdx += 4 * channel;
            }
        }
        always_assert(wIdx == numParameters);
        delete[] floatWeights;
    }

    Tensor4D<T>& forward(Tensor4D<T> &input)
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

        activation.copy(var32);
        return activation;
    }

    void optimize()
    {
        backend->optimize(root);
    }
};

void module_test()
{
    const u64 scale = 12;
    ResNet9<i64> resnet;
    resnet.init(scale);
    resnet.loadFloatWeights("cifar10_resnet9-float.dat", scale);
    Tensor4D<i64> input(1, 32, 32, 3);
    input.fill(1LL << scale);
    auto &res = resnet.forward(input);
    resnet.activation.print();
}

template <typename T>
void blprint(const Tensor4D<T> &p, u64 bw)
{
    for (int i = 0; i < p.d1; ++i) {
        for (int j = 0; j < p.d2; ++j) {
            for (int k = 0; k < p.d3; ++k) {
                for (int l = 0; l < p.d4; ++l) {
                    i64 val;
                    if (bw == 64) {
                        val = p(i, j, k, l);
                    }
                    else {
                        val = (p(i, j, k, l) + (1LL << (bw - 1))) % (1LL << bw);
                        val -= (1LL << (bw - 1));
                    }
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

template <typename T>
void blprint(const Tensor4D<T> &p, u64 bw, u64 scale)
{
    for (int i = 0; i < p.d1; ++i) {
        for (int j = 0; j < p.d2; ++j) {
            for (int k = 0; k < p.d3; ++k) {
                for (int l = 0; l < p.d4; ++l) {
                    if (bw == 64) {
                        std::cout << ((double)p(i, j, k, l)) / (1LL << scale) << " ";
                        continue;
                    }
                    else {
                        i64 val = (p(i, j, k, l) + (1LL << (bw - 1))) % (1LL << bw);
                        val -= (1LL << (bw - 1));
                        std::cout << ((double)val) / (1LL << scale) << " ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void module_test_llama_ext(int party)
{
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 12;
    LlamaConfig::bitlength = 32;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    // std::string ip = "127.0.0.1";
    std::string ip = "127.0.0.1";
    llama->init(ip, true);

    ResNet9<u64> resnet;
    resnet.init(scale);
    resnet.setBackend(llama);
    resnet.optimize();
    if (party != 1) {
        resnet.loadFloatWeights("cifar10_resnet9-float.dat", scale);
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
        blprint(output, LlamaConfig::bitlength);
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
        module_test();
    }
    else {
        module_test_llama_ext(party);
    }

}