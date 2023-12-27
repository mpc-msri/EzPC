#include "sytorch/backend/llama_extended.h"
#include "sytorch/backend/llama_improved.h"
#include "sytorch/layers/layers.h"
#include "sytorch/module.h"
#include "sytorch/utils.h" 


template <typename T>
class Net: public SytorchModule<T> {
     using SytorchModule<T>::add;
     using SytorchModule<T>::concat;
public:
     Conv2D<T> *conv0;
     MaxPool2D<T> *maxpool1;
     BatchNorm2dInference<T> *batchnormalization2;
     ReLU<T> *relu3;
     Conv2D<T> *conv4;
     Conv2D<T> *conv5;
     ReLU<T> *relu6;
     Conv2D<T> *conv7;
     ReLU<T> *relu8;
     Conv2D<T> *conv9;
     BatchNorm2dInference<T> *batchnormalization11;
     ReLU<T> *relu12;
     Conv2D<T> *conv13;
     ReLU<T> *relu14;
     Conv2D<T> *conv15;
     ReLU<T> *relu16;
     Conv2D<T> *conv17;
     BatchNorm2dInference<T> *batchnormalization19;
     ReLU<T> *relu20;
     Conv2D<T> *conv21;
     ReLU<T> *relu22;
     Conv2D<T> *conv23;
     ReLU<T> *relu24;
     Conv2D<T> *conv25;
     BatchNorm2dInference<T> *batchnormalization27;
     ReLU<T> *relu28;
     Conv2D<T> *conv29;
     Conv2D<T> *conv30;
     ReLU<T> *relu31;
     Conv2D<T> *conv32;
     ReLU<T> *relu33;
     Conv2D<T> *conv34;
     BatchNorm2dInference<T> *batchnormalization36;
     ReLU<T> *relu37;
     Conv2D<T> *conv38;
     ReLU<T> *relu39;
     Conv2D<T> *conv40;
     ReLU<T> *relu41;
     Conv2D<T> *conv42;
     BatchNorm2dInference<T> *batchnormalization44;
     ReLU<T> *relu45;
     Conv2D<T> *conv46;
     ReLU<T> *relu47;
     Conv2D<T> *conv48;
     ReLU<T> *relu49;
     Conv2D<T> *conv50;
     BatchNorm2dInference<T> *batchnormalization52;
     ReLU<T> *relu53;
     Conv2D<T> *conv54;
     ReLU<T> *relu55;
     Conv2D<T> *conv56;
     ReLU<T> *relu57;
     Conv2D<T> *conv58;
     BatchNorm2dInference<T> *batchnormalization60;
     ReLU<T> *relu61;
     Conv2D<T> *conv62;
     Conv2D<T> *conv63;
     ReLU<T> *relu64;
     Conv2D<T> *conv65;
     ReLU<T> *relu66;
     Conv2D<T> *conv67;
     BatchNorm2dInference<T> *batchnormalization69;
     ReLU<T> *relu70;
     Conv2D<T> *conv71;
     ReLU<T> *relu72;
     Conv2D<T> *conv73;
     ReLU<T> *relu74;
     Conv2D<T> *conv75;
     BatchNorm2dInference<T> *batchnormalization77;
     ReLU<T> *relu78;
     Conv2D<T> *conv79;
     ReLU<T> *relu80;
     Conv2D<T> *conv81;
     ReLU<T> *relu82;
     Conv2D<T> *conv83;
     BatchNorm2dInference<T> *batchnormalization85;
     ReLU<T> *relu86;
     Conv2D<T> *conv87;
     ReLU<T> *relu88;
     Conv2D<T> *conv89;
     ReLU<T> *relu90;
     Conv2D<T> *conv91;
     BatchNorm2dInference<T> *batchnormalization93;
     ReLU<T> *relu94;
     Conv2D<T> *conv95;
     ReLU<T> *relu96;
     Conv2D<T> *conv97;
     ReLU<T> *relu98;
     Conv2D<T> *conv99;
     BatchNorm2dInference<T> *batchnormalization101;
     ReLU<T> *relu102;
     Conv2D<T> *conv103;
     ReLU<T> *relu104;
     Conv2D<T> *conv105;
     ReLU<T> *relu106;
     Conv2D<T> *conv107;
     BatchNorm2dInference<T> *batchnormalization109;
     ReLU<T> *relu110;
     Conv2D<T> *conv111;
     Conv2D<T> *conv112;
     ReLU<T> *relu113;
     Conv2D<T> *conv114;
     ReLU<T> *relu115;
     Conv2D<T> *conv116;
     BatchNorm2dInference<T> *batchnormalization118;
     ReLU<T> *relu119;
     Conv2D<T> *conv120;
     ReLU<T> *relu121;
     Conv2D<T> *conv122;
     ReLU<T> *relu123;
     Conv2D<T> *conv124;
     BatchNorm2dInference<T> *batchnormalization126;
     ReLU<T> *relu127;
     Conv2D<T> *conv128;
     ReLU<T> *relu129;
     Conv2D<T> *conv130;
     ReLU<T> *relu131;
     Conv2D<T> *conv132;
     BatchNorm2dInference<T> *batchnormalization134;
     ReLU<T> *relu135;
     GlobalAvgPool2D<T> *globalaveragepool136;
     Flatten<T> *flatten137;
     FC<T> *gemm138;
     


public:
     Net()
     {
          conv0 =    new Conv2D<T>(3, 64, 7, 3, 2);
          maxpool1 =    new MaxPool2D<T>(3, 0, 2);
          batchnormalization2 =    new BatchNorm2dInference<T>(64);
          relu3 =    new ReLU<T>();
          conv4 =    new Conv2D<T>(64, 64, 1, 0, 1, true);
          conv5 =    new Conv2D<T>(64, 256, 1, 0, 1);
          relu6 =    new ReLU<T>();
          conv7 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu8 =    new ReLU<T>();
          conv9 =    new Conv2D<T>(64, 256, 1, 0, 1);
          batchnormalization11 =    new BatchNorm2dInference<T>(256);
          relu12 =    new ReLU<T>();
          conv13 =    new Conv2D<T>(256, 64, 1, 0, 1, true);
          relu14 =    new ReLU<T>();
          conv15 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu16 =    new ReLU<T>();
          conv17 =    new Conv2D<T>(64, 256, 1, 0, 1);
          batchnormalization19 =    new BatchNorm2dInference<T>(256);
          relu20 =    new ReLU<T>();
          conv21 =    new Conv2D<T>(256, 64, 1, 0, 1, true);
          relu22 =    new ReLU<T>();
          conv23 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu24 =    new ReLU<T>();
          conv25 =    new Conv2D<T>(64, 256, 1, 0, 1);
          batchnormalization27 =    new BatchNorm2dInference<T>(256);
          relu28 =    new ReLU<T>();
          conv29 =    new Conv2D<T>(256, 128, 1, 0, 1, true);
          conv30 =    new Conv2D<T>(256, 512, 1, 0, 2);
          relu31 =    new ReLU<T>();
          conv32 =    new Conv2D<T>(128, 128, 3, 1, 2, true);
          relu33 =    new ReLU<T>();
          conv34 =    new Conv2D<T>(128, 512, 1, 0, 1);
          batchnormalization36 =    new BatchNorm2dInference<T>(512);
          relu37 =    new ReLU<T>();
          conv38 =    new Conv2D<T>(512, 128, 1, 0, 1, true);
          relu39 =    new ReLU<T>();
          conv40 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu41 =    new ReLU<T>();
          conv42 =    new Conv2D<T>(128, 512, 1, 0, 1);
          batchnormalization44 =    new BatchNorm2dInference<T>(512);
          relu45 =    new ReLU<T>();
          conv46 =    new Conv2D<T>(512, 128, 1, 0, 1, true);
          relu47 =    new ReLU<T>();
          conv48 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu49 =    new ReLU<T>();
          conv50 =    new Conv2D<T>(128, 512, 1, 0, 1);
          batchnormalization52 =    new BatchNorm2dInference<T>(512);
          relu53 =    new ReLU<T>();
          conv54 =    new Conv2D<T>(512, 128, 1, 0, 1, true);
          relu55 =    new ReLU<T>();
          conv56 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu57 =    new ReLU<T>();
          conv58 =    new Conv2D<T>(128, 512, 1, 0, 1);
          batchnormalization60 =    new BatchNorm2dInference<T>(512);
          relu61 =    new ReLU<T>();
          conv62 =    new Conv2D<T>(512, 256, 1, 0, 1, true);
          conv63 =    new Conv2D<T>(512, 1024, 1, 0, 2);
          relu64 =    new ReLU<T>();
          conv65 =    new Conv2D<T>(256, 256, 3, 1, 2, true);
          relu66 =    new ReLU<T>();
          conv67 =    new Conv2D<T>(256, 1024, 1, 0, 1);
          batchnormalization69 =    new BatchNorm2dInference<T>(1024);
          relu70 =    new ReLU<T>();
          conv71 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu72 =    new ReLU<T>();
          conv73 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu74 =    new ReLU<T>();
          conv75 =    new Conv2D<T>(256, 1024, 1, 0, 1);
          batchnormalization77 =    new BatchNorm2dInference<T>(1024);
          relu78 =    new ReLU<T>();
          conv79 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu80 =    new ReLU<T>();
          conv81 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu82 =    new ReLU<T>();
          conv83 =    new Conv2D<T>(256, 1024, 1, 0, 1);
          batchnormalization85 =    new BatchNorm2dInference<T>(1024);
          relu86 =    new ReLU<T>();
          conv87 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu88 =    new ReLU<T>();
          conv89 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu90 =    new ReLU<T>();
          conv91 =    new Conv2D<T>(256, 1024, 1, 0, 1);
          batchnormalization93 =    new BatchNorm2dInference<T>(1024);
          relu94 =    new ReLU<T>();
          conv95 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu96 =    new ReLU<T>();
          conv97 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu98 =    new ReLU<T>();
          conv99 =    new Conv2D<T>(256, 1024, 1, 0, 1);
          batchnormalization101 =    new BatchNorm2dInference<T>(1024);
          relu102 =    new ReLU<T>();
          conv103 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu104 =    new ReLU<T>();
          conv105 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu106 =    new ReLU<T>();
          conv107 =    new Conv2D<T>(256, 1024, 1, 0, 1);
          batchnormalization109 =    new BatchNorm2dInference<T>(1024);
          relu110 =    new ReLU<T>();
          conv111 =    new Conv2D<T>(1024, 512, 1, 0, 1, true);
          conv112 =    new Conv2D<T>(1024, 2048, 1, 0, 2);
          relu113 =    new ReLU<T>();
          conv114 =    new Conv2D<T>(512, 512, 3, 1, 2, true);
          relu115 =    new ReLU<T>();
          conv116 =    new Conv2D<T>(512, 2048, 1, 0, 1);
          batchnormalization118 =    new BatchNorm2dInference<T>(2048);
          relu119 =    new ReLU<T>();
          conv120 =    new Conv2D<T>(2048, 512, 1, 0, 1, true);
          relu121 =    new ReLU<T>();
          conv122 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu123 =    new ReLU<T>();
          conv124 =    new Conv2D<T>(512, 2048, 1, 0, 1);
          batchnormalization126 =    new BatchNorm2dInference<T>(2048);
          relu127 =    new ReLU<T>();
          conv128 =    new Conv2D<T>(2048, 512, 1, 0, 1, true);
          relu129 =    new ReLU<T>();
          conv130 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu131 =    new ReLU<T>();
          conv132 =    new Conv2D<T>(512, 2048, 1, 0, 1);
          batchnormalization134 =    new BatchNorm2dInference<T>(2048);
          relu135 =    new ReLU<T>();
          globalaveragepool136 =    new GlobalAvgPool2D<T>();
          flatten137 =    new Flatten<T>();
          gemm138 =    new FC<T>(2048, 1001, true);
     }

     Tensor4D<T>& _forward(Tensor4D<T> &input)
     {
          auto &var157 = conv0->forward(input, false);
          auto &var158 = maxpool1->forward(var157, false);
          auto &var159 = batchnormalization2->forward(var158, false);
          auto &var160 = relu3->forward(var159, false);
          auto &var161 = conv4->forward(var160, false);
          auto &var162 = conv5->forward(var160, false);
          auto &var163 = relu6->forward(var161, false);
          auto &var164 = conv7->forward(var163, false);
          auto &var165 = relu8->forward(var164, false);
          auto &var166 = conv9->forward(var165, false);
          auto var167 = add(var166, var162);
          auto &var168 = batchnormalization11->forward(var167, false);
          auto &var169 = relu12->forward(var168, false);
          auto &var170 = conv13->forward(var169, false);
          auto &var171 = relu14->forward(var170, false);
          auto &var172 = conv15->forward(var171, false);
          auto &var173 = relu16->forward(var172, false);
          auto &var174 = conv17->forward(var173, false);
          auto var175 = add(var174, var167);
          auto &var176 = batchnormalization19->forward(var175, false);
          auto &var177 = relu20->forward(var176, false);
          auto &var178 = conv21->forward(var177, false);
          auto &var179 = relu22->forward(var178, false);
          auto &var180 = conv23->forward(var179, false);
          auto &var181 = relu24->forward(var180, false);
          auto &var182 = conv25->forward(var181, false);
          auto var183 = add(var182, var175);
          auto &var184 = batchnormalization27->forward(var183, false);
          auto &var185 = relu28->forward(var184, false);
          auto &var186 = conv29->forward(var185, false);
          auto &var187 = conv30->forward(var185, false);
          auto &var188 = relu31->forward(var186, false);
          auto &var189 = conv32->forward(var188, false);
          auto &var190 = relu33->forward(var189, false);
          auto &var191 = conv34->forward(var190, false);
          auto var192 = add(var191, var187);
          auto &var193 = batchnormalization36->forward(var192, false);
          auto &var194 = relu37->forward(var193, false);
          auto &var195 = conv38->forward(var194, false);
          auto &var196 = relu39->forward(var195, false);
          auto &var197 = conv40->forward(var196, false);
          auto &var198 = relu41->forward(var197, false);
          auto &var199 = conv42->forward(var198, false);
          auto var200 = add(var199, var192);
          auto &var201 = batchnormalization44->forward(var200, false);
          auto &var202 = relu45->forward(var201, false);
          auto &var203 = conv46->forward(var202, false);
          auto &var204 = relu47->forward(var203, false);
          auto &var205 = conv48->forward(var204, false);
          auto &var206 = relu49->forward(var205, false);
          auto &var207 = conv50->forward(var206, false);
          auto var208 = add(var207, var200);
          auto &var209 = batchnormalization52->forward(var208, false);
          auto &var210 = relu53->forward(var209, false);
          auto &var211 = conv54->forward(var210, false);
          auto &var212 = relu55->forward(var211, false);
          auto &var213 = conv56->forward(var212, false);
          auto &var214 = relu57->forward(var213, false);
          auto &var215 = conv58->forward(var214, false);
          auto var216 = add(var215, var208);
          auto &var217 = batchnormalization60->forward(var216, false);
          auto &var218 = relu61->forward(var217, false);
          auto &var219 = conv62->forward(var218, false);
          auto &var220 = conv63->forward(var218, false);
          auto &var221 = relu64->forward(var219, false);
          auto &var222 = conv65->forward(var221, false);
          auto &var223 = relu66->forward(var222, false);
          auto &var224 = conv67->forward(var223, false);
          auto var225 = add(var224, var220);
          auto &var226 = batchnormalization69->forward(var225, false);
          auto &var227 = relu70->forward(var226, false);
          auto &var228 = conv71->forward(var227, false);
          auto &var229 = relu72->forward(var228, false);
          auto &var230 = conv73->forward(var229, false);
          auto &var231 = relu74->forward(var230, false);
          auto &var232 = conv75->forward(var231, false);
          auto var233 = add(var232, var225);
          auto &var234 = batchnormalization77->forward(var233, false);
          auto &var235 = relu78->forward(var234, false);
          auto &var236 = conv79->forward(var235, false);
          auto &var237 = relu80->forward(var236, false);
          auto &var238 = conv81->forward(var237, false);
          auto &var239 = relu82->forward(var238, false);
          auto &var240 = conv83->forward(var239, false);
          auto var241 = add(var240, var233);
          auto &var242 = batchnormalization85->forward(var241, false);
          auto &var243 = relu86->forward(var242, false);
          auto &var244 = conv87->forward(var243, false);
          auto &var245 = relu88->forward(var244, false);
          auto &var246 = conv89->forward(var245, false);
          auto &var247 = relu90->forward(var246, false);
          auto &var248 = conv91->forward(var247, false);
          auto var249 = add(var248, var241);
          auto &var250 = batchnormalization93->forward(var249, false);
          auto &var251 = relu94->forward(var250, false);
          auto &var252 = conv95->forward(var251, false);
          auto &var253 = relu96->forward(var252, false);
          auto &var254 = conv97->forward(var253, false);
          auto &var255 = relu98->forward(var254, false);
          auto &var256 = conv99->forward(var255, false);
          auto var257 = add(var256, var249);
          auto &var258 = batchnormalization101->forward(var257, false);
          auto &var259 = relu102->forward(var258, false);
          auto &var260 = conv103->forward(var259, false);
          auto &var261 = relu104->forward(var260, false);
          auto &var262 = conv105->forward(var261, false);
          auto &var263 = relu106->forward(var262, false);
          auto &var264 = conv107->forward(var263, false);
          auto var265 = add(var264, var257);
          auto &var266 = batchnormalization109->forward(var265, false);
          auto &var267 = relu110->forward(var266, false);
          auto &var268 = conv111->forward(var267, false);
          auto &var269 = conv112->forward(var267, false);
          auto &var270 = relu113->forward(var268, false);
          auto &var271 = conv114->forward(var270, false);
          auto &var272 = relu115->forward(var271, false);
          auto &var273 = conv116->forward(var272, false);
          auto var274 = add(var273, var269);
          auto &var275 = batchnormalization118->forward(var274, false);
          auto &var276 = relu119->forward(var275, false);
          auto &var277 = conv120->forward(var276, false);
          auto &var278 = relu121->forward(var277, false);
          auto &var279 = conv122->forward(var278, false);
          auto &var280 = relu123->forward(var279, false);
          auto &var281 = conv124->forward(var280, false);
          auto var282 = add(var281, var274);
          auto &var283 = batchnormalization126->forward(var282, false);
          auto &var284 = relu127->forward(var283, false);
          auto &var285 = conv128->forward(var284, false);
          auto &var286 = relu129->forward(var285, false);
          auto &var287 = conv130->forward(var286, false);
          auto &var288 = relu131->forward(var287, false);
          auto &var289 = conv132->forward(var288, false);
          auto var290 = add(var289, var282);
          auto &var291 = batchnormalization134->forward(var290, false);
          auto &var292 = relu135->forward(var291, false);
          auto &var293 = globalaveragepool136->forward(var292, false);
          auto &var294 = flatten137->forward(var293, false);
          auto &var295 = gemm138->forward(var294, false);
          return var295;
     }

};


int ct_main(int __argc, char**__argv){
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));

    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();

    const u64 scale = 24;
    ResNet50NoBN<i64> net;
    net.init(scale);
    Tensor4D<i64> input(1, 224, 224, 3);

    // input.loadAsi64("../EzPC/sytorch/input_share1.dat");
    input.fill(1LL << scale);
    net.load("resnet50_no_bn_input_weights.dat");
    std::cout << ((u64)net.conv49->filter.data[0]) << " " << ((u64)net.conv49->filter.data[1]) << " " << ((u64)net.conv49->filter.data[2]) << " " << ((u64)net.conv49->filter.data[3]) << " " << ((u64)net.conv49->filter.data[4]) << std::endl;
    std::cout << ((u64)net.conv49->bias.data[0]) << " " << ((u64)net.conv49->bias.data[1]) << " " << ((u64)net.conv49->bias.data[2]) << " " << ((u64)net.conv49->bias.data[3]) << " " << ((u64)net.conv49->bias.data[4]) << std::endl;
    // std::cout << net.conv116->filter(0, 0) << std::endl;
    // input.dumpAsi64("input.dat");
    net.forward(input);
    //  for(int i = 0; i < 10; ++i)
    //       std::cout << net.activation.data[i] << std::endl;

    // blprint(net.activation, LlamaConfig::bitlength - scale);
    // net.setBackend(llama);
    // net.optimize();
    // net.dumpOrcaModel("resnet50nobn");
    // net.dumModelWeightsAsi64("resnet50nobn");
}
    
int main(int argc, char**argv){

    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    
    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    
    int party = 0;
    if (argc > 1) {
        party = atoi(argv[1]);
    }
    const u64 scale = 12;

    if (party == 0) {
        Net<i64> net;
        net.init(scale);
        // net.load("resnet50_weights.dat");
        Tensor4D<i64> input(1, 224, 224, 3);
        input.fill(1LL << scale);
        print_dot_graph(net.root);
        net.forward(input);
        blprint(net.activation, 64);
        return 0;
    }
    LlamaConfig::bitlength = 37;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    // _init(argc, argv); // This can read all command line arguments like party, ip, port, etc. and can be written as a part of backend.
    LlamaConfig::num_threads = 64;
    std::string ip = "172.31.45.229";
    
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
    