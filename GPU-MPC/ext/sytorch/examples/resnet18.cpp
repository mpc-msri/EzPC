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
class ResNet18 : public SytorchModule<T>
{
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
    ResNet18()
    {
        conv0 = new Conv2D<T>(3, 64, 7, 3, 2, true);
        maxpool1 = new MaxPool2D<T>(3, 1, 2);
        relu2 = new ReLU<T>();
        conv3 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        relu4 = new ReLU<T>();
        conv5 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        relu7 = new ReLU<T>();
        conv8 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        relu9 = new ReLU<T>();
        conv10 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        relu12 = new ReLU<T>();
        conv13 = new Conv2D<T>(64, 128, 3, 1, 2, true);
        relu14 = new ReLU<T>();
        conv15 = new Conv2D<T>(128, 128, 3, 1, 1, true);
        conv16 = new Conv2D<T>(64, 128, 1, 0, 2, true);
        relu18 = new ReLU<T>();
        conv19 = new Conv2D<T>(128, 128, 3, 1, 1, true);
        relu20 = new ReLU<T>();
        conv21 = new Conv2D<T>(128, 128, 3, 1, 1, true);
        relu23 = new ReLU<T>();
        conv24 = new Conv2D<T>(128, 256, 3, 1, 2, true);
        relu25 = new ReLU<T>();
        conv26 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        conv27 = new Conv2D<T>(128, 256, 1, 0, 2, true);
        relu29 = new ReLU<T>();
        conv30 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu31 = new ReLU<T>();
        conv32 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu34 = new ReLU<T>();
        conv35 = new Conv2D<T>(256, 512, 3, 1, 2, true);
        relu36 = new ReLU<T>();
        conv37 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        conv38 = new Conv2D<T>(256, 512, 1, 0, 2, true);
        relu40 = new ReLU<T>();
        conv41 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        relu42 = new ReLU<T>();
        conv43 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        relu45 = new ReLU<T>();
        globalaveragepool46 = new GlobalAvgPool2D<T>();
        flatten47 = new Flatten<T>();
        gemm48 = new FC<T>(512, 1000, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var44 = conv0->forward(input);
        //   return var44;
        auto &var45 = maxpool1->forward(var44);
        auto &var46 = relu2->forward(var45);
        auto &var47 = conv3->forward(var46);
        auto &var48 = relu4->forward(var47);
        auto &var49 = conv5->forward(var48);
        auto &var50 = add(var49, var46);
        auto &var51 = relu7->forward(var50);
        auto &var52 = conv8->forward(var51);
        auto &var53 = relu9->forward(var52);
        auto &var54 = conv10->forward(var53);
        auto &var55 = add(var54, var51);
        auto &var56 = relu12->forward(var55);
        auto &var57 = conv13->forward(var56);
        auto &var58 = relu14->forward(var57);
        auto &var59 = conv15->forward(var58);
        auto &var60 = conv16->forward(var56);
        auto &var61 = add(var59, var60);
        auto &var62 = relu18->forward(var61);
        auto &var63 = conv19->forward(var62);
        auto &var64 = relu20->forward(var63);
        auto &var65 = conv21->forward(var64);
        auto &var66 = add(var65, var62);
        auto &var67 = relu23->forward(var66);
        auto &var68 = conv24->forward(var67);
        auto &var69 = relu25->forward(var68);
        auto &var70 = conv26->forward(var69);
        auto &var71 = conv27->forward(var67);
        auto &var72 = add(var70, var71);
        auto &var73 = relu29->forward(var72);
        auto &var74 = conv30->forward(var73);
        auto &var75 = relu31->forward(var74);
        auto &var76 = conv32->forward(var75);
        auto &var77 = add(var76, var73);
        auto &var78 = relu34->forward(var77);
        auto &var79 = conv35->forward(var78);
        auto &var80 = relu36->forward(var79);
        auto &var81 = conv37->forward(var80);
        auto &var82 = conv38->forward(var78);
        auto &var83 = add(var81, var82);
        auto &var84 = relu40->forward(var83);
        auto &var85 = conv41->forward(var84);
        auto &var86 = relu42->forward(var85);
        auto &var87 = conv43->forward(var86);
        auto &var88 = add(var87, var84);
        auto &var89 = relu45->forward(var88);
        auto &var90 = globalaveragepool46->forward(var89);
        auto &var91 = flatten47->forward(var90);
        auto &var92 = gemm48->forward(var91);
        return var92;
    }
};

int main(int argc, char**__argv){

    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    srand(time(NULL));
    using T = i32;
    // ClearText<T>::bw = 64;
    const u64 scale = 12;//24

    ResNet18<T> net;
    net.init(scale);
    net.load("resnet18_no_bn_input_weights.dat");

    Tensor<T> input({1, 224, 224, 3});
    input.load("2.dat", scale);

    net.forward(input);
    
    auto act_2d = net.activation.as_2d();
    // printf("%ld\n", act_2d.data[0]);
    // for(int i = 200; i < 250; i++) printf("Act=%ld\n", act_2d.data[i]);
    // for(int i = 600; i < 650; i++) printf("Act=%ld\n", act_2d.data[i]);
//     print(net.conv0->activation, scale, 64);
//     printf("\n");
    // auto w = net.gemm48->weight.as_nd();
    // printf("%ld, %ld, %ld\n", w.data[0], w.data[1], w.data[w.size() - 1]);
    std::cout << act_2d.argmax(0) + 1 << std::endl;
    std::cout << act_2d(0, act_2d.argmax(0)) << std::endl;
    // printf("%ld\n", net.conv0->activation.data[0]);    
    
    return 0;

}
