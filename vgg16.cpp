#include "sytorch/backend/llama_extended.h"
#include "sytorch/backend/llama_improved.h"
#include "sytorch/layers/layers.h"
#include "sytorch/module.h"
#include "sytorch/utils.h" 


template <typename T>
class Net: public SytorchModule<T> {
     using SytorchModule<T>::add;
public:
     Conv2D<T> *conv0;
     ReLU<T> *relu1;
     Conv2D<T> *conv2;
     MaxPool2D<T> *maxpool3;
     ReLU<T> *relu4;
     Conv2D<T> *conv5;
     ReLU<T> *relu6;
     Conv2D<T> *conv7;
     MaxPool2D<T> *maxpool8;
     ReLU<T> *relu9;
     Conv2D<T> *conv10;
     ReLU<T> *relu11;
     Conv2D<T> *conv12;
     ReLU<T> *relu13;
     Conv2D<T> *conv14;
     MaxPool2D<T> *maxpool15;
     ReLU<T> *relu16;
     Conv2D<T> *conv17;
     ReLU<T> *relu18;
     Conv2D<T> *conv19;
     ReLU<T> *relu20;
     Conv2D<T> *conv21;
     MaxPool2D<T> *maxpool22;
     ReLU<T> *relu23;
     Conv2D<T> *conv24;
     ReLU<T> *relu25;
     Conv2D<T> *conv26;
     ReLU<T> *relu27;
     Conv2D<T> *conv28;
     MaxPool2D<T> *maxpool29;
     ReLU<T> *relu30;
     Flatten<T> *reshape31;
     FC<T> *gemm32;
     ReLU<T> *relu33;
     FC<T> *gemm34;
     ReLU<T> *relu35;
     FC<T> *gemm36;
     


public:
     Net()
     {
          conv0 =    new Conv2D<T>(3, 64, 3, 1, 1, true);
          relu1 =    new ReLU<T>();
          conv2 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          maxpool3 =    new MaxPool2D<T>(2, 0, 2);
          relu4 =    new ReLU<T>();
          conv5 =    new Conv2D<T>(64, 128, 3, 1, 1, true);
          relu6 =    new ReLU<T>();
          conv7 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          maxpool8 =    new MaxPool2D<T>(2, 0, 2);
          relu9 =    new ReLU<T>();
          conv10 =    new Conv2D<T>(128, 256, 3, 1, 1, true);
          relu11 =    new ReLU<T>();
          conv12 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu13 =    new ReLU<T>();
          conv14 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          maxpool15 =    new MaxPool2D<T>(2, 0, 2);
          relu16 =    new ReLU<T>();
          conv17 =    new Conv2D<T>(256, 512, 3, 1, 1, true);
          relu18 =    new ReLU<T>();
          conv19 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu20 =    new ReLU<T>();
          conv21 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          maxpool22 =    new MaxPool2D<T>(2, 0, 2);
          relu23 =    new ReLU<T>();
          conv24 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu25 =    new ReLU<T>();
          conv26 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu27 =    new ReLU<T>();
          conv28 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          maxpool29 =    new MaxPool2D<T>(2, 0, 2);
          relu30 =    new ReLU<T>();
          reshape31 =    new Flatten<T>();
          gemm32 =    new FC<T>(25088, 4096, true);
          relu33 =    new ReLU<T>();
          gemm34 =    new FC<T>(4096, 4096, true);
          relu35 =    new ReLU<T>();
          gemm36 =    new FC<T>(4096, 1000, true);
     }

     Tensor4D<T>& _forward(Tensor4D<T> &input)
     {
          auto &var35 = conv0->forward(input, false);
          auto &var36 = relu1->forward(var35, false);
          auto &var37 = conv2->forward(var36, false);
          auto &var38 = maxpool3->forward(var37, false);
          auto &var39 = relu4->forward(var38, false);
          auto &var40 = conv5->forward(var39, false);
          auto &var41 = relu6->forward(var40, false);
          auto &var42 = conv7->forward(var41, false);
          auto &var43 = maxpool8->forward(var42, false);
          auto &var44 = relu9->forward(var43, false);
          auto &var45 = conv10->forward(var44, false);
          auto &var46 = relu11->forward(var45, false);
          auto &var47 = conv12->forward(var46, false);
          auto &var48 = relu13->forward(var47, false);
          auto &var49 = conv14->forward(var48, false);
          auto &var50 = maxpool15->forward(var49, false);
          auto &var51 = relu16->forward(var50, false);
          auto &var52 = conv17->forward(var51, false);
          auto &var53 = relu18->forward(var52, false);
          auto &var54 = conv19->forward(var53, false);
          auto &var55 = relu20->forward(var54, false);
          auto &var56 = conv21->forward(var55, false);
          auto &var57 = maxpool22->forward(var56, false);
          auto &var58 = relu23->forward(var57, false);
          auto &var59 = conv24->forward(var58, false);
          auto &var60 = relu25->forward(var59, false);
          auto &var61 = conv26->forward(var60, false);
          auto &var62 = relu27->forward(var61, false);
          auto &var63 = conv28->forward(var62, false);
          auto &var64 = maxpool29->forward(var63, false);
          auto &var65 = relu30->forward(var64, false);
          auto &var66 = reshape31->forward(var65, false);
          auto &var67 = gemm32->forward(var66, false);
          auto &var68 = relu33->forward(var67, false);
          auto &var69 = gemm34->forward(var68, false);
          auto &var70 = relu35->forward(var69, false);
          auto &var71 = gemm36->forward(var70, false);
          return var71;
     }

};


    
int main(int argc, char**argv){

    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    
    const u64 scale = 12;
    LlamaConfig::bitlength = 32;
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }

    if (party == 0) {
        Net<i64> net;
        net.init(scale);
        Tensor4D<i64> input(1, 224, 224, 3);
        input.fill(1LL << scale);
        print_dot_graph(net.root);
        net.forward(input);
        net.activation.print();
        return 0;
    }
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    //  _init(argc, argv); // This can read all command line arguments like party, ip, port, etc. and can be written as a part of backend.
    LlamaConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    
    llama->init(ip, true);

    Net<u64> net;
    net.init(scale);
    net.setBackend(llama);
    net.optimize();
    if (party != 1) {
        // net.load("weight_file.dat");
    }
    else {
        net.zero();
    }

    llama::start();
    Tensor4D<u64> input(1, 224, 224, 3);
    input.fill(1LL << scale);
    llama->inputA(input);

    net.forward(input);
    llama::end();
    auto &output = net.activation;
    llama->output(output);
    if (party != 1) {
        blprint(output, LlamaConfig::bitlength - scale);
    }
    llama->finalize();
}
    