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
class VGG16 : public SytorchModule<T>
{
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
        conv0 = new Conv2D<T>(3, 64, 3, 1, 1, true);
        relu1 = new ReLU<T>();
        conv2 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        maxpool3 = new MaxPool2D<T>(2, 0, 2);
        relu4 = new ReLU<T>();
        conv5 = new Conv2D<T>(64, 128, 3, 1, 1, true);
        relu6 = new ReLU<T>();
        conv7 = new Conv2D<T>(128, 128, 3, 1, 1, true);
        maxpool8 = new MaxPool2D<T>(2, 0, 2);
        relu9 = new ReLU<T>();
        conv10 = new Conv2D<T>(128, 256, 3, 1, 1, true);
        relu11 = new ReLU<T>();
        conv12 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu13 = new ReLU<T>();
        conv14 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        maxpool15 = new MaxPool2D<T>(2, 0, 2);
        relu16 = new ReLU<T>();
        conv17 = new Conv2D<T>(256, 512, 3, 1, 1, true);
        relu18 = new ReLU<T>();
        conv19 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        relu20 = new ReLU<T>();
        conv21 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        maxpool22 = new MaxPool2D<T>(2, 0, 2);
        relu23 = new ReLU<T>();
        conv24 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        relu25 = new ReLU<T>();
        conv26 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        relu27 = new ReLU<T>();
        conv28 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        maxpool29 = new MaxPool2D<T>(2, 0, 2);
        relu30 = new ReLU<T>();
        reshape31 = new Flatten<T>();
        gemm32 = new FC<T>(25088, 4096, true);
        relu33 = new ReLU<T>();
        gemm34 = new FC<T>(4096, 4096, true);
        relu35 = new ReLU<T>();
        gemm36 = new FC<T>(4096, 1000, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var35 = conv0->forward(input);
        auto &var36 = relu1->forward(var35);
        auto &var37 = conv2->forward(var36);
        auto &var38 = maxpool3->forward(var37);
        auto &var39 = relu4->forward(var38);
        auto &var40 = conv5->forward(var39);
        auto &var41 = relu6->forward(var40);
        auto &var42 = conv7->forward(var41);
        auto &var43 = maxpool8->forward(var42);
        auto &var44 = relu9->forward(var43);
        auto &var45 = conv10->forward(var44);
        auto &var46 = relu11->forward(var45);
        auto &var47 = conv12->forward(var46);
        auto &var48 = relu13->forward(var47);
        auto &var49 = conv14->forward(var48);
        auto &var50 = maxpool15->forward(var49);
        auto &var51 = relu16->forward(var50);
        auto &var52 = conv17->forward(var51);
        auto &var53 = relu18->forward(var52);
        auto &var54 = conv19->forward(var53);
        auto &var55 = relu20->forward(var54);
        auto &var56 = conv21->forward(var55);
        auto &var57 = maxpool22->forward(var56);
        auto &var58 = relu23->forward(var57);
        auto &var59 = conv24->forward(var58);
        auto &var60 = relu25->forward(var59);
        auto &var61 = conv26->forward(var60);
        auto &var62 = relu27->forward(var61);
        auto &var63 = conv28->forward(var62);
        auto &var64 = maxpool29->forward(var63);
        auto &var65 = relu30->forward(var64);
        auto &var66 = reshape31->forward(var65);
        auto &var67 = gemm32->forward(var66);
        auto &var68 = relu33->forward(var67);
        auto &var69 = gemm34->forward(var68);
        auto &var70 = relu35->forward(var69);
        auto &var71 = gemm36->forward(var70);
        return var71;
    }
};


int main(int argc, char**__argv){

    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    prngStr.SetSeed(osuCrypto::toBlock(time(NULL)));
    srand(time(NULL));
    // ClearText<i64>::bw = 64;
    const u64 scale = 24;

    VGG16<i64> net;
    net.init(scale);
    net.load("P-VGG16-imgnet-float.dat");

    Tensor<i64> input({1, 224, 224, 3});
    input.load("2.dat", scale);

    net.forward(input);
    
    auto act_2d = net.activation.as_2d();
    printf("%d\n", act_2d.data[0]);
//     print(net.conv0->activation, scale, 64);
//     printf("\n");
    // auto w = net.gemm48->weight.as_nd();
    // printf("%ld, %ld, %ld\n", w.data[0], w.data[1], w.data[w.size() - 1]);
    std::cout << act_2d.argmax(0) + 1 << std::endl;
    // printf("%ld\n", net.conv0->activation.data[0]);    
    
    return 0;

}
