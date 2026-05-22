// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
     ReLU<T> *relu6;
     Conv2D<T> *conv7;
     Conv2D<T> *conv8;
     ReLU<T> *relu10;
     Conv2D<T> *conv11;
     ReLU<T> *relu12;
     Conv2D<T> *conv13;
     ReLU<T> *relu14;
     Conv2D<T> *conv15;
     ReLU<T> *relu17;
     Conv2D<T> *conv18;
     ReLU<T> *relu19;
     Conv2D<T> *conv20;
     ReLU<T> *relu21;
     Conv2D<T> *conv22;
     ReLU<T> *relu24;
     Conv2D<T> *conv25;
     ReLU<T> *relu26;
     Conv2D<T> *conv27;
     ReLU<T> *relu28;
     Conv2D<T> *conv29;
     Conv2D<T> *conv30;
     ReLU<T> *relu32;
     Conv2D<T> *conv33;
     ReLU<T> *relu34;
     Conv2D<T> *conv35;
     ReLU<T> *relu36;
     Conv2D<T> *conv37;
     ReLU<T> *relu39;
     Conv2D<T> *conv40;
     ReLU<T> *relu41;
     Conv2D<T> *conv42;
     ReLU<T> *relu43;
     Conv2D<T> *conv44;
     ReLU<T> *relu46;
     Conv2D<T> *conv47;
     ReLU<T> *relu48;
     Conv2D<T> *conv49;
     ReLU<T> *relu50;
     Conv2D<T> *conv51;
     ReLU<T> *relu53;
     Conv2D<T> *conv54;
     ReLU<T> *relu55;
     Conv2D<T> *conv56;
     ReLU<T> *relu57;
     Conv2D<T> *conv58;
     Conv2D<T> *conv59;
     ReLU<T> *relu61;
     Conv2D<T> *conv62;
     ReLU<T> *relu63;
     Conv2D<T> *conv64;
     ReLU<T> *relu65;
     Conv2D<T> *conv66;
     ReLU<T> *relu68;
     Conv2D<T> *conv69;
     ReLU<T> *relu70;
     Conv2D<T> *conv71;
     ReLU<T> *relu72;
     Conv2D<T> *conv73;
     ReLU<T> *relu75;
     Conv2D<T> *conv76;
     ReLU<T> *relu77;
     Conv2D<T> *conv78;
     ReLU<T> *relu79;
     Conv2D<T> *conv80;
     ReLU<T> *relu82;
     Conv2D<T> *conv83;
     ReLU<T> *relu84;
     Conv2D<T> *conv85;
     ReLU<T> *relu86;
     Conv2D<T> *conv87;
     ReLU<T> *relu89;
     Conv2D<T> *conv90;
     ReLU<T> *relu91;
     Conv2D<T> *conv92;
     ReLU<T> *relu93;
     Conv2D<T> *conv94;
     ReLU<T> *relu96;
     Conv2D<T> *conv97;
     ReLU<T> *relu98;
     Conv2D<T> *conv99;
     ReLU<T> *relu100;
     Conv2D<T> *conv101;
     Conv2D<T> *conv102;
     ReLU<T> *relu104;
     Conv2D<T> *conv105;
     ReLU<T> *relu106;
     Conv2D<T> *conv107;
     ReLU<T> *relu108;
     Conv2D<T> *conv109;
     ReLU<T> *relu111;
     Conv2D<T> *conv112;
     ReLU<T> *relu113;
     Conv2D<T> *conv114;
     ReLU<T> *relu115;
     Conv2D<T> *conv116;
     ReLU<T> *relu118;
     GlobalAvgPool2D<T> *globalaveragepool119;
     Flatten<T> *flatten120;
     FC<T> *gemm121;
     


public:
     Net()
     {
          conv0 =    new Conv2D<T>(3, 64, 7, 3, 2, true);
          maxpool1 =    new MaxPool2D<T>(3, 1, 2);
          relu2 =    new ReLU<T>();
          conv3 =    new Conv2D<T>(64, 64, 1, 0, 1, true);
          relu4 =    new ReLU<T>();
          conv5 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu6 =    new ReLU<T>();
          conv7 =    new Conv2D<T>(64, 256, 1, 0, 1, true);
          conv8 =    new Conv2D<T>(64, 256, 1, 0, 1, true);
          relu10 =    new ReLU<T>();
          conv11 =    new Conv2D<T>(256, 64, 1, 0, 1, true);
          relu12 =    new ReLU<T>();
          conv13 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu14 =    new ReLU<T>();
          conv15 =    new Conv2D<T>(64, 256, 1, 0, 1, true);
          relu17 =    new ReLU<T>();
          conv18 =    new Conv2D<T>(256, 64, 1, 0, 1, true);
          relu19 =    new ReLU<T>();
          conv20 =    new Conv2D<T>(64, 64, 3, 1, 1, true);
          relu21 =    new ReLU<T>();
          conv22 =    new Conv2D<T>(64, 256, 1, 0, 1, true);
          relu24 =    new ReLU<T>();
          conv25 =    new Conv2D<T>(256, 128, 1, 0, 1, true);
          relu26 =    new ReLU<T>();
          conv27 =    new Conv2D<T>(128, 128, 3, 1, 2, true);
          relu28 =    new ReLU<T>();
          conv29 =    new Conv2D<T>(128, 512, 1, 0, 1, true);
          conv30 =    new Conv2D<T>(256, 512, 1, 0, 2, true);
          relu32 =    new ReLU<T>();
          conv33 =    new Conv2D<T>(512, 128, 1, 0, 1, true);
          relu34 =    new ReLU<T>();
          conv35 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu36 =    new ReLU<T>();
          conv37 =    new Conv2D<T>(128, 512, 1, 0, 1, true);
          relu39 =    new ReLU<T>();
          conv40 =    new Conv2D<T>(512, 128, 1, 0, 1, true);
          relu41 =    new ReLU<T>();
          conv42 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu43 =    new ReLU<T>();
          conv44 =    new Conv2D<T>(128, 512, 1, 0, 1, true);
          relu46 =    new ReLU<T>();
          conv47 =    new Conv2D<T>(512, 128, 1, 0, 1, true);
          relu48 =    new ReLU<T>();
          conv49 =    new Conv2D<T>(128, 128, 3, 1, 1, true);
          relu50 =    new ReLU<T>();
          conv51 =    new Conv2D<T>(128, 512, 1, 0, 1, true);
          relu53 =    new ReLU<T>();
          conv54 =    new Conv2D<T>(512, 256, 1, 0, 1, true);
          relu55 =    new ReLU<T>();
          conv56 =    new Conv2D<T>(256, 256, 3, 1, 2, true);
          relu57 =    new ReLU<T>();
          conv58 =    new Conv2D<T>(256, 1024, 1, 0, 1, true);
          conv59 =    new Conv2D<T>(512, 1024, 1, 0, 2, true);
          relu61 =    new ReLU<T>();
          conv62 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu63 =    new ReLU<T>();
          conv64 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu65 =    new ReLU<T>();
          conv66 =    new Conv2D<T>(256, 1024, 1, 0, 1, true);
          relu68 =    new ReLU<T>();
          conv69 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu70 =    new ReLU<T>();
          conv71 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu72 =    new ReLU<T>();
          conv73 =    new Conv2D<T>(256, 1024, 1, 0, 1, true);
          relu75 =    new ReLU<T>();
          conv76 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu77 =    new ReLU<T>();
          conv78 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu79 =    new ReLU<T>();
          conv80 =    new Conv2D<T>(256, 1024, 1, 0, 1, true);
          relu82 =    new ReLU<T>();
          conv83 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu84 =    new ReLU<T>();
          conv85 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu86 =    new ReLU<T>();
          conv87 =    new Conv2D<T>(256, 1024, 1, 0, 1, true);
          relu89 =    new ReLU<T>();
          conv90 =    new Conv2D<T>(1024, 256, 1, 0, 1, true);
          relu91 =    new ReLU<T>();
          conv92 =    new Conv2D<T>(256, 256, 3, 1, 1, true);
          relu93 =    new ReLU<T>();
          conv94 =    new Conv2D<T>(256, 1024, 1, 0, 1, true);
          relu96 =    new ReLU<T>();
          conv97 =    new Conv2D<T>(1024, 512, 1, 0, 1, true);
          relu98 =    new ReLU<T>();
          conv99 =    new Conv2D<T>(512, 512, 3, 1, 2, true);
          relu100 =    new ReLU<T>();
          conv101 =    new Conv2D<T>(512, 2048, 1, 0, 1, true);
          conv102 =    new Conv2D<T>(1024, 2048, 1, 0, 2, true);
          relu104 =    new ReLU<T>();
          conv105 =    new Conv2D<T>(2048, 512, 1, 0, 1, true);
          relu106 =    new ReLU<T>();
          conv107 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu108 =    new ReLU<T>();
          conv109 =    new Conv2D<T>(512, 2048, 1, 0, 1, true);
          relu111 =    new ReLU<T>();
          conv112 =    new Conv2D<T>(2048, 512, 1, 0, 1, true);
          relu113 =    new ReLU<T>();
          conv114 =    new Conv2D<T>(512, 512, 3, 1, 1, true);
          relu115 =    new ReLU<T>();
          conv116 =    new Conv2D<T>(512, 2048, 1, 0, 1, true);
          relu118 =    new ReLU<T>();
          globalaveragepool119 =    new GlobalAvgPool2D<T>();
          flatten120 =    new Flatten<T>();
          gemm121 =    new FC<T>(2048, 1000, true);
     }

     Tensor<T>& _forward(Tensor<T> &input)
     {
          auto &var110 = conv0->forward(input);
          auto &var111 = maxpool1->forward(var110);
          auto &var112 = relu2->forward(var111);
          auto &var113 = conv3->forward(var112);
          auto &var114 = relu4->forward(var113);
          auto &var115 = conv5->forward(var114);
          auto &var116 = relu6->forward(var115);
          auto &var117 = conv7->forward(var116);
          auto &var118 = conv8->forward(var112);
          auto &var119 = add(var117, var118);
          auto &var120 = relu10->forward(var119);
          auto &var121 = conv11->forward(var120);
          auto &var122 = relu12->forward(var121);
          auto &var123 = conv13->forward(var122);
          auto &var124 = relu14->forward(var123);
          auto &var125 = conv15->forward(var124);
          auto &var126 = add(var125, var120);
          auto &var127 = relu17->forward(var126);
          auto &var128 = conv18->forward(var127);
          auto &var129 = relu19->forward(var128);
          auto &var130 = conv20->forward(var129);
          auto &var131 = relu21->forward(var130);
          auto &var132 = conv22->forward(var131);
          auto &var133 = add(var132, var127);
          auto &var134 = relu24->forward(var133);
          auto &var135 = conv25->forward(var134);
          auto &var136 = relu26->forward(var135);
          auto &var137 = conv27->forward(var136);
          auto &var138 = relu28->forward(var137);
          auto &var139 = conv29->forward(var138);
          auto &var140 = conv30->forward(var134);
          auto &var141 = add(var139, var140);
          auto &var142 = relu32->forward(var141);
          auto &var143 = conv33->forward(var142);
          auto &var144 = relu34->forward(var143);
          auto &var145 = conv35->forward(var144);
          auto &var146 = relu36->forward(var145);
          auto &var147 = conv37->forward(var146);
          auto &var148 = add(var147, var142);
          auto &var149 = relu39->forward(var148);
          auto &var150 = conv40->forward(var149);
          auto &var151 = relu41->forward(var150);
          auto &var152 = conv42->forward(var151);
          auto &var153 = relu43->forward(var152);
          auto &var154 = conv44->forward(var153);
          auto &var155 = add(var154, var149);
          auto &var156 = relu46->forward(var155);
          auto &var157 = conv47->forward(var156);
          auto &var158 = relu48->forward(var157);
          auto &var159 = conv49->forward(var158);
          auto &var160 = relu50->forward(var159);
          auto &var161 = conv51->forward(var160);
          auto &var162 = add(var161, var156);
          auto &var163 = relu53->forward(var162);
          auto &var164 = conv54->forward(var163);
          auto &var165 = relu55->forward(var164);
          auto &var166 = conv56->forward(var165);
          auto &var167 = relu57->forward(var166);
          auto &var168 = conv58->forward(var167);
          auto &var169 = conv59->forward(var163);
          auto &var170 = add(var168, var169);
          auto &var171 = relu61->forward(var170);
          auto &var172 = conv62->forward(var171);
          auto &var173 = relu63->forward(var172);
          auto &var174 = conv64->forward(var173);
          auto &var175 = relu65->forward(var174);
          auto &var176 = conv66->forward(var175);
          auto &var177 = add(var176, var171);
          auto &var178 = relu68->forward(var177);
          auto &var179 = conv69->forward(var178);
          auto &var180 = relu70->forward(var179);
          auto &var181 = conv71->forward(var180);
          auto &var182 = relu72->forward(var181);
          auto &var183 = conv73->forward(var182);
          auto &var184 = add(var183, var178);
          auto &var185 = relu75->forward(var184);
          auto &var186 = conv76->forward(var185);
          auto &var187 = relu77->forward(var186);
          auto &var188 = conv78->forward(var187);
          auto &var189 = relu79->forward(var188);
          auto &var190 = conv80->forward(var189);
          auto &var191 = add(var190, var185);
          auto &var192 = relu82->forward(var191);
          auto &var193 = conv83->forward(var192);
          auto &var194 = relu84->forward(var193);
          auto &var195 = conv85->forward(var194);
          auto &var196 = relu86->forward(var195);
          auto &var197 = conv87->forward(var196);
          auto &var198 = add(var197, var192);
          auto &var199 = relu89->forward(var198);
          auto &var200 = conv90->forward(var199);
          auto &var201 = relu91->forward(var200);
          auto &var202 = conv92->forward(var201);
          auto &var203 = relu93->forward(var202);
          auto &var204 = conv94->forward(var203);
          auto &var205 = add(var204, var199);
          auto &var206 = relu96->forward(var205);
          auto &var207 = conv97->forward(var206);
          auto &var208 = relu98->forward(var207);
          auto &var209 = conv99->forward(var208);
          auto &var210 = relu100->forward(var209);
          auto &var211 = conv101->forward(var210);
          auto &var212 = conv102->forward(var206);
          auto &var213 = add(var211, var212);
          auto &var214 = relu104->forward(var213);
          auto &var215 = conv105->forward(var214);
          auto &var216 = relu106->forward(var215);
          auto &var217 = conv107->forward(var216);
          auto &var218 = relu108->forward(var217);
          auto &var219 = conv109->forward(var218);
          auto &var220 = add(var219, var214);
          auto &var221 = relu111->forward(var220);
          auto &var222 = conv112->forward(var221);
          auto &var223 = relu113->forward(var222);
          auto &var224 = conv114->forward(var223);
          auto &var225 = relu115->forward(var224);
          auto &var226 = conv116->forward(var225);
          auto &var227 = add(var226, var221);
          auto &var228 = relu118->forward(var227);
          auto &var229 = globalaveragepool119->forward(var228);
          auto &var230 = flatten120->forward(var229);
          auto &var231 = gemm121->forward(var230);
          return var231;
     }
};

int main(int argc, char**__argv){

    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    srand(time(NULL));
    const u64 scale = 12;

    Net<i64> net;
    net.init(scale);
    net.load("resnet50_no_bn_input_weights.dat");

    Tensor<i64> input({1, 224, 224, 3});
    input.load("2.dat", scale);

    net.forward(input);
    
    auto act_2d = net.activation.as_2d();
//     print(net.conv0->activation, scale, 64);
//     printf("\n");
//     print(net.conv0->filter.as_nd(), scale, 64);
    std::cout << act_2d.argmax(0) + 1 << std::endl;
    std::cout << act_2d(0, act_2d.argmax(0)) << std::endl;
//     printf("%ld\n", net.conv0->activation.data[0]);
    return 0;

}
