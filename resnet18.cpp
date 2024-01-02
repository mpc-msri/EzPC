#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_improved.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <sytorch/utils.h>


template <typename T>
class Net: public SytorchModule<T> {
     using SytorchModule<T>::add;
     using SytorchModule<T>::concat;
public:
     Conv2D<T> *conv0;
     MaxPool2D<T> *maxpool1;
     ReLU<T> *relu2;
     Conv2D<T> *conv3;
     ReLU<T> *relu4;
     Conv2D<T> *conv5;
     ReLU<T> *relu7;
     Conv2D<T> *conv8;
     ReLU<T> *relu9;
     Conv2D<T> *conv10;
     ReLU<T> *relu12;
     Conv2D<T> *conv13;
     ReLU<T> *relu14;
     Conv2D<T> *conv15;
     Conv2D<T> *conv16;
     ReLU<T> *relu18;
     Conv2D<T> *conv19;
     ReLU<T> *relu20;
     Conv2D<T> *conv21;
     ReLU<T> *relu23;
     Conv2D<T> *conv24;
     ReLU<T> *relu25;
     Conv2D<T> *conv26;
     Conv2D<T> *conv27;
     ReLU<T> *relu29;
     Conv2D<T> *conv30;
     ReLU<T> *relu31;
     Conv2D<T> *conv32;
     ReLU<T> *relu34;
     Conv2D<T> *conv35;
     ReLU<T> *relu36;
     Conv2D<T> *conv37;
     Conv2D<T> *conv38;
     ReLU<T> *relu40;
     Conv2D<T> *conv41;
     ReLU<T> *relu42;
     Conv2D<T> *conv43;
     ReLU<T> *relu45;
     GlobalAvgPool2D<T> *globalaveragepool46;
     Flatten<T> *flatten47;
     FC<T> *gemm48;
     


public:
     Net()
     {
          conv0 =    new Conv2D<T>(3, 64, 7, 3, 2, true);
          maxpool1 =    new MaxPool2D<T>(3, 1, 2);
          relu2 =    new ReLU<T>();
          conv3 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu4 =    new ReLU<T>();
          conv5 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu7 =    new ReLU<T>();
          conv8 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu9 =    new ReLU<T>();
          conv10 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu12 =    new ReLU<T>();
          conv13 =    new Conv2D<T>(64, 128, 3, 1, 2, true);
          relu14 =    new ReLU<T>();
          conv15 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          conv16 =    new Conv2D<T>(64, 128, 1, 0, 2, true);
          relu18 =    new ReLU<T>();
          conv19 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu20 =    new ReLU<T>();
          conv21 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu23 =    new ReLU<T>();
          conv24 =    new Conv2D<T>(128, 256, 3, 1, 2, true);
          relu25 =    new ReLU<T>();
          conv26 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          conv27 =    new Conv2D<T>(128, 256, 1, 0, 2, true);
          relu29 =    new ReLU<T>();
          conv30 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu31 =    new ReLU<T>();
          conv32 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu34 =    new ReLU<T>();
          conv35 =    new Conv2D<T>(256, 512, 3, 1, 2, true);
          relu36 =    new ReLU<T>();
          conv37 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          conv38 =    new Conv2D<T>(256, 512, 1, 0, 2, true);
          relu40 =    new ReLU<T>();
          conv41 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu42 =    new ReLU<T>();
          conv43 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu45 =    new ReLU<T>();
          globalaveragepool46 =    new GlobalAvgPool2D<T>();
          flatten47 =    new Flatten<T>();
          gemm48 =    new FC<T>(512, 1000, true);
     }

     Tensor4D<T>& _forward(Tensor4D<T> &input)
     {
          auto &var44 = conv0->forward(input, false);
        //   return var44;
          auto &var45 = maxpool1->forward(var44, false);
          auto &var46 = relu2->forward(var45, false);
          auto &var47 = conv3->forward(var46, false);
          auto &var48 = relu4->forward(var47, false);
          auto &var49 = conv5->forward(var48, false);
          auto var50 = add(var49, var46);
          auto &var51 = relu7->forward(var50, false);
          auto &var52 = conv8->forward(var51, false);
          auto &var53 = relu9->forward(var52, false);
          auto &var54 = conv10->forward(var53, false);
          auto var55 = add(var54, var51);
          auto &var56 = relu12->forward(var55, false);
          auto &var57 = conv13->forward(var56, false);
          auto &var58 = relu14->forward(var57, false);
          auto &var59 = conv15->forward(var58, false);
          auto &var60 = conv16->forward(var56, false);
          auto var61 = add(var59, var60);
          auto &var62 = relu18->forward(var61, false);
          auto &var63 = conv19->forward(var62, false);
          auto &var64 = relu20->forward(var63, false);
          auto &var65 = conv21->forward(var64, false);
          auto var66 = add(var65, var62);
          auto &var67 = relu23->forward(var66, false);
          auto &var68 = conv24->forward(var67, false);
          auto &var69 = relu25->forward(var68, false);
          auto &var70 = conv26->forward(var69, false);
          auto &var71 = conv27->forward(var67, false);
          auto var72 = add(var70, var71);
          auto &var73 = relu29->forward(var72, false);
          auto &var74 = conv30->forward(var73, false);
          auto &var75 = relu31->forward(var74, false);
          auto &var76 = conv32->forward(var75, false);
          auto var77 = add(var76, var73);
          auto &var78 = relu34->forward(var77, false);
          auto &var79 = conv35->forward(var78, false);
          auto &var80 = relu36->forward(var79, false);
          auto &var81 = conv37->forward(var80, false);
          auto &var82 = conv38->forward(var78, false);
          auto var83 = add(var81, var82);
          auto &var84 = relu40->forward(var83, false);
          auto &var85 = conv41->forward(var84, false);
          auto &var86 = relu42->forward(var85, false);
          auto &var87 = conv43->forward(var86, false);
          auto var88 = add(var87, var84);
          auto &var89 = relu45->forward(var88, false);
        //   return var89;
          auto &var90 = globalaveragepool46->forward(var89, false);
        //   return var90;
          auto &var91 = flatten47->forward(var90, false);
          auto &var92 = gemm48->forward(var91, false);
          return var92;
     }

};

int main(int __argc, char**__argv){
    
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    
    const u64 scale = 10;

    if (party == 0) {
        Net<i64> net;
        net.init(scale);
        std::string weights_file = __argv[3];
        net.load(weights_file);
        Tensor4D<i64> input(1, 224, 224, 3);
        input.fill(1LL << scale);
        net.forward(input);
        net.activation.printshape();
        Tensor4D<i64> output2(1, 10, 1, 1);
        for(int i = 0; i < 10; ++i) {
            output2(0, i, 0, 0) = net.activation.data[i];
        }
        blprint(output2, LlamaConfig::bitlength - scale);
        return 0;
    }

    LlamaConfig::bitlength = 32;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    LlamaConfig::num_threads = 4;
    if(__argc > 2){
        ip = __argv[2];
    }
    llama->init(ip, true);

    Net<u64> net;
    net.init(scale);
    net.setBackend(llama);
    net.optimize();
    print_dot_graph(net.root);
    if(party == SERVER){
        std::string weights_file = __argv[3];
        net.load(weights_file);
    }
    else if(party == DEALER){
        net.zero();
    }
    llama->initializeInferencePartyA(net.root);

    Tensor4D<u64> input(1, 224, 224, 3);
    if(party == CLIENT){
         input.fill(1LL << scale);
    }
    llama->initializeInferencePartyB(input);

    llama::start();
    net.forward(input);
    llama::end();

    auto &output = net.activation;
    llama->outputA(output);
    if (party == CLIENT) {
        Tensor4D<u64> output2(1, 10, 1, 1);
        for(int i = 0; i < 10; ++i) {
            output2(0, i, 0, 0) = output.data[i];
        }
        blprint(output2, LlamaConfig::bitlength);
        blprint(output2, LlamaConfig::bitlength - scale);
    }
    llama->finalize();
}
    