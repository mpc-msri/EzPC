#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_improved.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <sytorch/utils.h>

template <typename T>
class VGG16: public SytorchModule<T> {
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
     VGG16()
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


template <typename T>
class ResNet50NoBN: public SytorchModule<T> {
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
     ResNet50NoBN()
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

     Tensor4D<T>& _forward(Tensor4D<T> &input)
     {
          auto &var110 = conv0->forward(input, false);
          auto &var111 = maxpool1->forward(var110, false);
          auto &var112 = relu2->forward(var111, false);
          auto &var113 = conv3->forward(var112, false);
          auto &var114 = relu4->forward(var113, false);
          auto &var115 = conv5->forward(var114, false);
          auto &var116 = relu6->forward(var115, false);
          auto &var117 = conv7->forward(var116, false);
          auto &var118 = conv8->forward(var112, false);
          auto var119 = add(var117, var118);
          auto &var120 = relu10->forward(var119, false);
          auto &var121 = conv11->forward(var120, false);
          auto &var122 = relu12->forward(var121, false);
          auto &var123 = conv13->forward(var122, false);
          auto &var124 = relu14->forward(var123, false);
          auto &var125 = conv15->forward(var124, false);
          auto var126 = add(var125, var120);
          auto &var127 = relu17->forward(var126, false);
          auto &var128 = conv18->forward(var127, false);
          auto &var129 = relu19->forward(var128, false);
          auto &var130 = conv20->forward(var129, false);
          auto &var131 = relu21->forward(var130, false);
          auto &var132 = conv22->forward(var131, false);
          auto var133 = add(var132, var127);
          auto &var134 = relu24->forward(var133, false);
          auto &var135 = conv25->forward(var134, false);
          auto &var136 = relu26->forward(var135, false);
          auto &var137 = conv27->forward(var136, false);
          auto &var138 = relu28->forward(var137, false);
          auto &var139 = conv29->forward(var138, false);
          auto &var140 = conv30->forward(var134, false);
          auto var141 = add(var139, var140);
          auto &var142 = relu32->forward(var141, false);
          auto &var143 = conv33->forward(var142, false);
          auto &var144 = relu34->forward(var143, false);
          auto &var145 = conv35->forward(var144, false);
          auto &var146 = relu36->forward(var145, false);
          auto &var147 = conv37->forward(var146, false);
          auto var148 = add(var147, var142);
          auto &var149 = relu39->forward(var148, false);
          auto &var150 = conv40->forward(var149, false);
          auto &var151 = relu41->forward(var150, false);
          auto &var152 = conv42->forward(var151, false);
          auto &var153 = relu43->forward(var152, false);
          auto &var154 = conv44->forward(var153, false);
          auto var155 = add(var154, var149);
          auto &var156 = relu46->forward(var155, false);
          auto &var157 = conv47->forward(var156, false);
          auto &var158 = relu48->forward(var157, false);
          auto &var159 = conv49->forward(var158, false);
          auto &var160 = relu50->forward(var159, false);
          auto &var161 = conv51->forward(var160, false);
          auto var162 = add(var161, var156);
          auto &var163 = relu53->forward(var162, false);
          auto &var164 = conv54->forward(var163, false);
          auto &var165 = relu55->forward(var164, false);
          auto &var166 = conv56->forward(var165, false);
          auto &var167 = relu57->forward(var166, false);
          auto &var168 = conv58->forward(var167, false);
          auto &var169 = conv59->forward(var163, false);
          auto var170 = add(var168, var169);
          auto &var171 = relu61->forward(var170, false);
          auto &var172 = conv62->forward(var171, false);
          auto &var173 = relu63->forward(var172, false);
          auto &var174 = conv64->forward(var173, false);
          auto &var175 = relu65->forward(var174, false);
          auto &var176 = conv66->forward(var175, false);
          auto var177 = add(var176, var171);
          auto &var178 = relu68->forward(var177, false);
          auto &var179 = conv69->forward(var178, false);
          auto &var180 = relu70->forward(var179, false);
          auto &var181 = conv71->forward(var180, false);
          auto &var182 = relu72->forward(var181, false);
          auto &var183 = conv73->forward(var182, false);
          auto var184 = add(var183, var178);
          auto &var185 = relu75->forward(var184, false);
          auto &var186 = conv76->forward(var185, false);
          auto &var187 = relu77->forward(var186, false);
          auto &var188 = conv78->forward(var187, false);
          auto &var189 = relu79->forward(var188, false);
          auto &var190 = conv80->forward(var189, false);
          auto var191 = add(var190, var185);
          auto &var192 = relu82->forward(var191, false);
          auto &var193 = conv83->forward(var192, false);
          auto &var194 = relu84->forward(var193, false);
          auto &var195 = conv85->forward(var194, false);
          auto &var196 = relu86->forward(var195, false);
          auto &var197 = conv87->forward(var196, false);
          auto var198 = add(var197, var192);
          auto &var199 = relu89->forward(var198, false);
          auto &var200 = conv90->forward(var199, false);
          auto &var201 = relu91->forward(var200, false);
          auto &var202 = conv92->forward(var201, false);
          auto &var203 = relu93->forward(var202, false);
          auto &var204 = conv94->forward(var203, false);
          auto var205 = add(var204, var199);
          auto &var206 = relu96->forward(var205, false);
          auto &var207 = conv97->forward(var206, false);
          auto &var208 = relu98->forward(var207, false);
          auto &var209 = conv99->forward(var208, false);
          auto &var210 = relu100->forward(var209, false);
          auto &var211 = conv101->forward(var210, false);
          auto &var212 = conv102->forward(var206, false);
          auto var213 = add(var211, var212);
          auto &var214 = relu104->forward(var213, false);
          auto &var215 = conv105->forward(var214, false);
          auto &var216 = relu106->forward(var215, false);
          auto &var217 = conv107->forward(var216, false);
          auto &var218 = relu108->forward(var217, false);
          auto &var219 = conv109->forward(var218, false);
          auto var220 = add(var219, var214);
          auto &var221 = relu111->forward(var220, false);
          auto &var222 = conv112->forward(var221, false);
          auto &var223 = relu113->forward(var222, false);
          auto &var224 = conv114->forward(var223, false);
          auto &var225 = relu115->forward(var224, false);
          auto &var226 = conv116->forward(var225, false);
          auto var227 = add(var226, var221);
          auto &var228 = relu118->forward(var227, false);
          auto &var229 = globalaveragepool119->forward(var228, false);
          auto &var230 = flatten120->forward(var229, false);
          auto &var231 = gemm121->forward(var230, false);
          return var231;
     }

};


template <typename T>
class ResNet18NoBN: public SytorchModule<T> {
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
     ResNet18NoBN()
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
          auto &var90 = globalaveragepool46->forward(var89, false);
          auto &var91 = flatten47->forward(var90, false);
          auto &var92 = gemm48->forward(var91, false);
          return var92;
     }

};

int main2(int __argc, char**__argv){
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
    net.load("../EzPC/sytorch/resnet50_no_bn_input_weights.dat");
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
    return 0;
}


int main(int __argc, char**__argv){
    
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using LlamaVersion = LlamaExtended<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    
    const u64 scale = 24;

    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaConfig::num_threads = 4;
    ip = "172.31.45.210";
    llama->init(ip, true);

    VGG16<u64> net;
    net.init(scale);
    net.setBackend(llama);
    net.optimize();

    if(party == SERVER){
        // net.load("resnet18_no_bn_input_weights.dat");
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
    return 0;
}
    