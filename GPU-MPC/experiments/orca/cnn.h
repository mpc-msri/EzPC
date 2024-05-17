// 
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

#pragma once

#include <sytorch/module.h>

#include "utils/gpu_data_types.h"

#include "nn/orca/gpu_model.h"
#include "nn/orca/conv2d_layer.h"
#include "nn/orca/maxpool_layer.h"
#include "nn/orca/relu_layer.h"
#include "nn/orca/relu_extend_layer.h"
#include "nn/orca/avg_pool_layer.h"
#include "nn/orca/fc_layer.h"

#include "backend/orca.h"
#include "backend/piranha.h"

template <typename T>
class CNN2 : public SytorchModule<T>
{
    Conv2D<T> *conv1;
    ReLU<T> *relu1;
    MaxPool2D<T> *maxpool1;
    Conv2D<T> *conv2;
    ReLU<T> *relu2;
    MaxPool2D<T> *maxpool2;
    Flatten<T> *flatten3;
    FC<T> *fc4;
    ReLU<T> *relu4;
    FC<T> *fc5;

public:
    CNN2()
    {
        conv1 = new Conv2D<T>(1, 8, 5, 0, 1, true);
        relu1 = new ReLU<T>();
        maxpool1 = new MaxPool2D<T>(2, 0, 2);

        conv2 = new Conv2D<T>(8, 16, 5, 0, 1, true);
        relu2 = new ReLU<T>();
        maxpool2 = new MaxPool2D<T>(2, 0, 2);

        flatten3 = new Flatten<T>();
        fc4 = new FC<T>(256, 128, true);
        relu4 = new ReLU<T>();

        fc5 = new FC<T>(128, 10, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var1 = conv1->forward(input);
        auto &var2 = relu1->forward(var1);
        auto &var3 = maxpool1->forward(var2);
        auto &var4 = conv2->forward(var3);
        auto &var5 = relu2->forward(var4);
        auto &var6 = maxpool2->forward(var5);
        auto &var7 = flatten3->forward(var6);
        auto &var8 = fc4->forward(var7);
        auto &var9 = relu4->forward(var8);
        auto &var10 = fc5->forward(var9);
        return var10;
    }
};

template <typename T>
class PLenetNoReluAvgPool : public SytorchModule<T>
{
    Conv2D<T> *conv1;
    AvgPool2D<T> *avgpool1;
    ReLU<T> *relu1;
    Conv2D<T> *conv2;
    AvgPool2D<T> *avgpool2;
    ReLU<T> *relu2;
    Flatten<T> *flatten3;
    FC<T> *fc4;
    ReLU<T> *relu4;
    FC<T> *fc5;

public:
    PLenetNoReluAvgPool()
    {
        conv1 = new Conv2D<T>(1, 20, 5, 0, 1, false);
        avgpool1 = new AvgPool2D<T>(2, 0, 2);
        relu1 = new ReLU<T>();
        conv2 = new Conv2D<T>(20, 50, 5, 0, 1, false);
        avgpool2 = new AvgPool2D<T>(2, 0, 2);
        relu2 = new ReLU<T>();
        flatten3 = new Flatten<T>();
        fc4 = new FC<T>(800, 500, true);
        relu4 = new ReLU<T>();
        fc5 = new FC<T>(500, 10, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var1 = conv1->forward(input);
        auto &var2 = avgpool1->forward(var1);
        auto &var3 = relu1->forward(var2);
        auto &var4 = conv2->forward(var3);
        auto &var5 = avgpool2->forward(var4);
        auto &var6 = relu2->forward(var5);
        auto &var7 = flatten3->forward(var6);
        auto &var8 = fc4->forward(var7);
        auto &var9 = relu4->forward(var8);
        auto &var10 = fc5->forward(var9);
        return var10;
    }
};

template <typename T>
class MinionnLenet : public SytorchModule<T>
{
    Conv2D<T> *conv1;
    ReLU<T> *relu1;
    MaxPool2D<T> *maxpool1;
    Conv2D<T> *conv2;
    ReLU<T> *relu2;
    MaxPool2D<T> *maxpool2;
    Flatten<T> *flatten3;
    FC<T> *fc4;
    ReLU<T> *relu4;
    FC<T> *fc5;

public:
    MinionnLenet()
    {
        conv1 = new Conv2D<T>(1, 16, 5, 0, 1, true);
        relu1 = new ReLU<T>();
        maxpool1 = new MaxPool2D<T>(2, 0, 2);

        conv2 = new Conv2D<T>(16, 16, 5, 0, 1, true);
        relu2 = new ReLU<T>();
        maxpool2 = new MaxPool2D<T>(2, 0, 2);

        flatten3 = new Flatten<T>();
        fc4 = new FC<T>(256, 100, true);
        relu4 = new ReLU<T>();

        fc5 = new FC<T>(100, 10, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var1 = conv1->forward(input);
        auto &var2 = relu1->forward(var1);
        auto &var3 = maxpool1->forward(var2);
        auto &var4 = conv2->forward(var3);
        auto &var5 = relu2->forward(var4);
        auto &var6 = maxpool2->forward(var5);
        auto &var7 = flatten3->forward(var6);
        auto &var8 = fc4->forward(var7);
        auto &var9 = relu4->forward(var8);
        auto &var10 = fc5->forward(var9);
        return var10;
    }
};

template <typename T>
class PSecureMlNoRelu : public SytorchModule<T>
{
    FC<T> *fc1;
    ReLU<T> *relu1;
    FC<T> *fc2;
    ReLU<T> *relu2;
    FC<T> *fc3;

public:
    PSecureMlNoRelu()
    {
        fc1 = new FC<T>(784, 128, true);
        relu1 = new ReLU<T>();
        fc2 = new FC<T>(128, 128, true);
        relu2 = new ReLU<T>();
        fc3 = new FC<T>(128, 10, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var1 = fc1->forward(input);
        auto &var2 = relu1->forward(var1);
        auto &var3 = fc2->forward(var2);
        auto &var4 = relu2->forward(var3);
        auto &var5 = fc3->forward(var4);
        return var5;
    }
};

template <typename T>
class CNN3 : public SytorchModule<T>
{
    Conv2D<T> *conv1;
    ReLU<T> *relu1;
    MaxPool2D<T> *maxpool1;
    Conv2D<T> *conv2;
    ReLU<T> *relu2;
    MaxPool2D<T> *maxpool2;
    Conv2D<T> *conv3;
    ReLU<T> *relu3;
    MaxPool2D<T> *maxpool3;
    Flatten<T> *flatten4;
    FC<T> *fc5;

public:
    CNN3()
    {
        conv1 = new Conv2D<T>(3, 64, 5, 1, 1, true);
        relu1 = new ReLU<T>();
        maxpool1 = new MaxPool2D<T>(3, 0, 2);

        conv2 = new Conv2D<T>(64, 64, 5, 1, 1, true);
        relu2 = new ReLU<T>();
        maxpool2 = new MaxPool2D<T>(3, 0, 2);

        conv3 = new Conv2D<T>(64, 64, 5, 1, 1, true);
        relu3 = new ReLU<T>();
        maxpool3 = new MaxPool2D<T>(3, 0, 2);

        flatten4 = new Flatten<T>();
        fc5 = new FC<T>(64, 10, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var1 = conv1->forward(input);
        auto &var2 = relu1->forward(var1);
        auto &var3 = maxpool1->forward(var2);
        auto &var4 = conv2->forward(var3);
        auto &var5 = relu2->forward(var4);
        auto &var6 = maxpool2->forward(var5);
        auto &var7 = conv3->forward(var6);
        auto &var8 = relu3->forward(var7);
        auto &var9 = maxpool3->forward(var8);
        auto &var10 = flatten4->forward(var9);
        auto &var11 = fc5->forward(var10);
        return var11;
    }
};

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

template <typename T>
class ResNet50 : public SytorchModule<T>
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
    ResNet50()
    {
        conv0 = new Conv2D<T>(3, 64, 7, 3, 2, true);
        maxpool1 = new MaxPool2D<T>(3, 1, 2);
        relu2 = new ReLU<T>();
        conv3 = new Conv2D<T>(64, 64, 1, 0, 1, true);
        relu4 = new ReLU<T>();
        conv5 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        relu6 = new ReLU<T>();
        conv7 = new Conv2D<T>(64, 256, 1, 0, 1, true);
        conv8 = new Conv2D<T>(64, 256, 1, 0, 1, true);
        relu10 = new ReLU<T>();
        conv11 = new Conv2D<T>(256, 64, 1, 0, 1, true);
        relu12 = new ReLU<T>();
        conv13 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        relu14 = new ReLU<T>();
        conv15 = new Conv2D<T>(64, 256, 1, 0, 1, true);
        relu17 = new ReLU<T>();
        conv18 = new Conv2D<T>(256, 64, 1, 0, 1, true);
        relu19 = new ReLU<T>();
        conv20 = new Conv2D<T>(64, 64, 3, 1, 1, true);
        relu21 = new ReLU<T>();
        conv22 = new Conv2D<T>(64, 256, 1, 0, 1, true);
        relu24 = new ReLU<T>();
        conv25 = new Conv2D<T>(256, 128, 1, 0, 1, true);
        relu26 = new ReLU<T>();
        conv27 = new Conv2D<T>(128, 128, 3, 1, 2, true);
        relu28 = new ReLU<T>();
        conv29 = new Conv2D<T>(128, 512, 1, 0, 1, true);
        conv30 = new Conv2D<T>(256, 512, 1, 0, 2, true);
        relu32 = new ReLU<T>();
        conv33 = new Conv2D<T>(512, 128, 1, 0, 1, true);
        relu34 = new ReLU<T>();
        conv35 = new Conv2D<T>(128, 128, 3, 1, 1, true);
        relu36 = new ReLU<T>();
        conv37 = new Conv2D<T>(128, 512, 1, 0, 1, true);
        relu39 = new ReLU<T>();
        conv40 = new Conv2D<T>(512, 128, 1, 0, 1, true);
        relu41 = new ReLU<T>();
        conv42 = new Conv2D<T>(128, 128, 3, 1, 1, true);
        relu43 = new ReLU<T>();
        conv44 = new Conv2D<T>(128, 512, 1, 0, 1, true);
        relu46 = new ReLU<T>();
        conv47 = new Conv2D<T>(512, 128, 1, 0, 1, true);
        relu48 = new ReLU<T>();
        conv49 = new Conv2D<T>(128, 128, 3, 1, 1, true);
        relu50 = new ReLU<T>();
        conv51 = new Conv2D<T>(128, 512, 1, 0, 1, true);
        relu53 = new ReLU<T>();
        conv54 = new Conv2D<T>(512, 256, 1, 0, 1, true);
        relu55 = new ReLU<T>();
        conv56 = new Conv2D<T>(256, 256, 3, 1, 2, true);
        relu57 = new ReLU<T>();
        conv58 = new Conv2D<T>(256, 1024, 1, 0, 1, true);
        conv59 = new Conv2D<T>(512, 1024, 1, 0, 2, true);
        relu61 = new ReLU<T>();
        conv62 = new Conv2D<T>(1024, 256, 1, 0, 1, true);
        relu63 = new ReLU<T>();
        conv64 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu65 = new ReLU<T>();
        conv66 = new Conv2D<T>(256, 1024, 1, 0, 1, true);
        relu68 = new ReLU<T>();
        conv69 = new Conv2D<T>(1024, 256, 1, 0, 1, true);
        relu70 = new ReLU<T>();
        conv71 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu72 = new ReLU<T>();
        conv73 = new Conv2D<T>(256, 1024, 1, 0, 1, true);
        relu75 = new ReLU<T>();
        conv76 = new Conv2D<T>(1024, 256, 1, 0, 1, true);
        relu77 = new ReLU<T>();
        conv78 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu79 = new ReLU<T>();
        conv80 = new Conv2D<T>(256, 1024, 1, 0, 1, true);
        relu82 = new ReLU<T>();
        conv83 = new Conv2D<T>(1024, 256, 1, 0, 1, true);
        relu84 = new ReLU<T>();
        conv85 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu86 = new ReLU<T>();
        conv87 = new Conv2D<T>(256, 1024, 1, 0, 1, true);
        relu89 = new ReLU<T>();
        conv90 = new Conv2D<T>(1024, 256, 1, 0, 1, true);
        relu91 = new ReLU<T>();
        conv92 = new Conv2D<T>(256, 256, 3, 1, 1, true);
        relu93 = new ReLU<T>();
        conv94 = new Conv2D<T>(256, 1024, 1, 0, 1, true);
        relu96 = new ReLU<T>();
        conv97 = new Conv2D<T>(1024, 512, 1, 0, 1, true);
        relu98 = new ReLU<T>();
        conv99 = new Conv2D<T>(512, 512, 3, 1, 2, true);
        relu100 = new ReLU<T>();
        conv101 = new Conv2D<T>(512, 2048, 1, 0, 1, true);
        conv102 = new Conv2D<T>(1024, 2048, 1, 0, 2, true);
        relu104 = new ReLU<T>();
        conv105 = new Conv2D<T>(2048, 512, 1, 0, 1, true);
        relu106 = new ReLU<T>();
        conv107 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        relu108 = new ReLU<T>();
        conv109 = new Conv2D<T>(512, 2048, 1, 0, 1, true);
        relu111 = new ReLU<T>();
        conv112 = new Conv2D<T>(2048, 512, 1, 0, 1, true);
        relu113 = new ReLU<T>();
        conv114 = new Conv2D<T>(512, 512, 3, 1, 1, true);
        relu115 = new ReLU<T>();
        conv116 = new Conv2D<T>(512, 2048, 1, 0, 1, true);
        relu118 = new ReLU<T>();
        globalaveragepool119 = new GlobalAvgPool2D<T>();
        flatten120 = new Flatten<T>();
        gemm121 = new FC<T>(2048, 1000, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
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

template <typename T>
class PVGG16NoRelu : public SytorchModule<T>
{

public:
    Conv2D<T> *conv0;
    ReLU<T> *relu1;
    Conv2D<T> *conv2;
    AvgPool2D<T> *maxpool3;
    ReLU<T> *relu4;
    Conv2D<T> *conv5;
    ReLU<T> *relu6;
    Conv2D<T> *conv7;
    AvgPool2D<T> *maxpool8;
    ReLU<T> *relu9;
    Conv2D<T> *conv10;
    ReLU<T> *relu11;
    Conv2D<T> *conv12;
    ReLU<T> *relu13;
    Conv2D<T> *conv14;
    AvgPool2D<T> *maxpool15;
    ReLU<T> *relu16;
    Conv2D<T> *conv17;
    ReLU<T> *relu18;
    Conv2D<T> *conv19;
    ReLU<T> *relu20;
    Conv2D<T> *conv21;
    AvgPool2D<T> *maxpool22;
    ReLU<T> *relu23;
    Conv2D<T> *conv24;
    ReLU<T> *relu25;
    Conv2D<T> *conv26;
    ReLU<T> *relu27;
    Conv2D<T> *conv28;
    AvgPool2D<T> *maxpool29;
    ReLU<T> *relu30;
    Flatten<T> *reshape31;
    FC<T> *gemm32;
    ReLU<T> *relu33;
    FC<T> *gemm34;
    ReLU<T> *relu35;
    FC<T> *gemm36;

public:
    PVGG16NoRelu()
    {
        conv0 = new Conv2D<T>(3, 64, 3, 1, 1, false);
        relu1 = new ReLU<T>();
        conv2 = new Conv2D<T>(64, 64, 3, 1, 1, false);
        maxpool3 = new AvgPool2D<T>(2, 0, 2);
        relu4 = new ReLU<T>();
        conv5 = new Conv2D<T>(64, 128, 3, 1, 1, false);
        relu6 = new ReLU<T>();
        conv7 = new Conv2D<T>(128, 128, 3, 1, 1, false);
        maxpool8 = new AvgPool2D<T>(2, 0, 2);
        relu9 = new ReLU<T>();
        conv10 = new Conv2D<T>(128, 256, 3, 1, 1, false);
        relu11 = new ReLU<T>();
        conv12 = new Conv2D<T>(256, 256, 3, 1, 1, false);
        relu13 = new ReLU<T>();
        conv14 = new Conv2D<T>(256, 256, 3, 1, 1, false);
        maxpool15 = new AvgPool2D<T>(2, 0, 2);
        relu16 = new ReLU<T>();
        conv17 = new Conv2D<T>(256, 512, 3, 1, 1, false);
        relu18 = new ReLU<T>();
        conv19 = new Conv2D<T>(512, 512, 3, 1, 1, false);
        relu20 = new ReLU<T>();
        conv21 = new Conv2D<T>(512, 512, 3, 1, 1, false);
        maxpool22 = new AvgPool2D<T>(2, 0, 2);
        relu23 = new ReLU<T>();
        conv24 = new Conv2D<T>(512, 512, 3, 1, 1, false);
        relu25 = new ReLU<T>();
        conv26 = new Conv2D<T>(512, 512, 3, 1, 1, false);
        relu27 = new ReLU<T>();
        conv28 = new Conv2D<T>(512, 512, 3, 1, 1, false);
        maxpool29 = new AvgPool2D<T>(2, 0, 2);
        relu30 = new ReLU<T>();
        reshape31 = new Flatten<T>();
        gemm32 = new FC<T>(512, 256, true);
        relu33 = new ReLU<T>();
        gemm34 = new FC<T>(256, 256, true);
        relu35 = new ReLU<T>();
        gemm36 = new FC<T>(256, 10, true);
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

template <typename T>
class PAlexnetNoRelu : public SytorchModule<T>
{

public:
    Conv2D<T> *conv0;
    AvgPool2D<T> *maxpool1;
    ReLU<T> *relu2;
    Conv2D<T> *conv3;
    AvgPool2D<T> *maxpool4;
    ReLU<T> *relu5;
    Conv2D<T> *conv6;
    ReLU<T> *relu7;
    Conv2D<T> *conv8;
    ReLU<T> *relu9;
    Conv2D<T> *conv10;
    ReLU<T> *relu11;
    Flatten<T> *reshape12;
    FC<T> *gemm13;
    ReLU<T> *relu14;
    FC<T> *gemm15;
    ReLU<T> *relu16;
    FC<T> *gemm17;

public:
    PAlexnetNoRelu()
    {
        conv0 = new Conv2D<T>(3, 96, 11, 9, 4, false);
        maxpool1 = new AvgPool2D<T>(3, 0, 2);
        relu2 = new ReLU<T>();
        conv3 = new Conv2D<T>(96, 256, 5, 1, 1, false);
        maxpool4 = new AvgPool2D<T>(2, 0, 1);
        relu5 = new ReLU<T>();
        conv6 = new Conv2D<T>(256, 384, 3, 1, 1, false);
        relu7 = new ReLU<T>();
        conv8 = new Conv2D<T>(384, 384, 3, 1, 1, false);
        relu9 = new ReLU<T>();
        conv10 = new Conv2D<T>(384, 256, 3, 1, 1, false);
        relu11 = new ReLU<T>();
        reshape12 = new Flatten<T>();
        gemm13 = new FC<T>(256, 256, true);
        relu14 = new ReLU<T>();
        gemm15 = new FC<T>(256, 256, true);
        relu16 = new ReLU<T>();
        gemm17 = new FC<T>(256, 10, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var15 = conv0->forward(input);
        auto &var16 = maxpool1->forward(var15);
        auto &var17 = relu2->forward(var16);
        auto &var18 = conv3->forward(var17);
        auto &var19 = maxpool4->forward(var18);
        auto &var20 = relu5->forward(var19);
        auto &var21 = conv6->forward(var20);
        auto &var22 = relu7->forward(var21);
        auto &var23 = conv8->forward(var22);
        auto &var24 = relu9->forward(var23);
        auto &var25 = conv10->forward(var24);
        auto &var26 = relu11->forward(var25);
        auto &var27 = reshape12->forward(var26);
        auto &var28 = gemm13->forward(var27);
        auto &var29 = relu14->forward(var28);
        auto &var30 = gemm15->forward(var29);
        auto &var31 = relu16->forward(var30);
        auto &var32 = gemm17->forward(var31);
        return var32;
    }
};

template <typename T>
class FalconAlexnetNoRelu : public SytorchModule<T>
{

public:
    Conv2D<T> *conv0;
    MaxPool2D<T> *maxpool1;
    ReLU<T> *relu2;
    Conv2D<T> *conv3;
    MaxPool2D<T> *maxpool4;
    ReLU<T> *relu5;
    Conv2D<T> *conv6;
    ReLU<T> *relu7;
    Conv2D<T> *conv8;
    ReLU<T> *relu9;
    Conv2D<T> *conv10;
    ReLU<T> *relu11;
    Flatten<T> *reshape12;
    FC<T> *gemm13;
    ReLU<T> *relu14;
    FC<T> *gemm15;
    ReLU<T> *relu16;
    FC<T> *gemm17;

public:
    FalconAlexnetNoRelu()
    {
        conv0 = new Conv2D<T>(3, 96, 11, 9, 4, true);
        maxpool1 = new MaxPool2D<T>(3, 0, 2);
        relu2 = new ReLU<T>();
        conv3 = new Conv2D<T>(96, 256, 5, 1, 1, true);
        maxpool4 = new MaxPool2D<T>(2, 0, 1);
        relu5 = new ReLU<T>();
        conv6 = new Conv2D<T>(256, 384, 3, 1, 1, true);
        relu7 = new ReLU<T>();
        conv8 = new Conv2D<T>(384, 384, 3, 1, 1, true);
        relu9 = new ReLU<T>();
        conv10 = new Conv2D<T>(384, 256, 3, 1, 1, true);
        relu11 = new ReLU<T>();
        reshape12 = new Flatten<T>();
        gemm13 = new FC<T>(256, 256, true);
        relu14 = new ReLU<T>();
        gemm15 = new FC<T>(256, 256, true);
        relu16 = new ReLU<T>();
        gemm17 = new FC<T>(256, 10, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &var15 = conv0->forward(input);
        auto &var16 = maxpool1->forward(var15);
        auto &var17 = relu2->forward(var16);
        auto &var18 = conv3->forward(var17);
        auto &var19 = maxpool4->forward(var18);
        auto &var20 = relu5->forward(var19);
        auto &var21 = conv6->forward(var20);
        auto &var22 = relu7->forward(var21);
        auto &var23 = conv8->forward(var22);
        auto &var24 = relu9->forward(var23);
        auto &var25 = conv10->forward(var24);
        auto &var26 = relu11->forward(var25);
        auto &var27 = reshape12->forward(var26);
        auto &var28 = gemm13->forward(var27);
        auto &var29 = relu14->forward(var28);
        auto &var30 = gemm15->forward(var29);
        auto &var31 = relu16->forward(var30);
        auto &var32 = gemm17->forward(var31);
        return var32;
    }
};

template <typename T>
SytorchModule<T> *getCNN(std::string name)
{
    SytorchModule<T> *m;
    if (name.compare("CNN2") == 0)
    {
        m = new CNN2<T>();
    }
    else if (name.compare("CNN3") == 0)
    {
        m = new CNN3<T>();
    }
    else if (name.compare("ResNet18") == 0)
    {
        m = new ResNet18<T>();
    }
    else if (name.compare("ResNet50") == 0)
    {
        m = new ResNet50<T>();
    }
    else if (name.compare("VGG16") == 0)
    {
        m = new VGG16<T>();
    }
    else if (name.compare("P-LeNet") == 0)
    {
        m = new PLenetNoReluAvgPool<T>();
    }
    else if (name.compare("P-SecureML") == 0)
    {
        m = new PSecureMlNoRelu<T>();
    }
    else if (name.compare("P-VGG16") == 0)
    {
        m = new PVGG16NoRelu<T>();
    }
    else if (name.compare("P-AlexNet") == 0)
    {
        m = new PAlexnetNoRelu<T>();
    }
    else if (name.compare("AlexNet") == 0)
    {
        m = new FalconAlexnetNoRelu<T>();
    }
    else if (name.compare("ModelB") == 0)
    {
        m = new MinionnLenet<T>();
    }
    else
    {
        assert(0 && "unknown model");
    }
    return m;
}

template <typename T>
dcf::orca::GPUModel<T> *getGPUModel(std::string modelName, Tensor<T> inp)
{
    dcf::orca::GPUModel<T> *m;
    if (*(modelName.data()) == 'P')
    {
        m = getPiranhaCNN(modelName, inp);
    }
    else
    {
        m = getOrcaCNN(modelName, inp);
    }
    return m;
}

// in LlamaImproved, mode takes the value according to the following rule:
// 0: the layer takes as input \ell bits and outputs \ell bits
// 1: the layer takes as input \ell bits and outputs \ell - scale bits
// 2: the layer takes as input \ell - scale bits and outputs \ell bits
// 3: the layer takes as input \ell - scale bits and outputs \ell - scale bits

template <typename T>
dcf::orca::GPUModel<T> *getOrcaCNN(std::string modelName, Tensor<T> inp)
{
    auto m = getCNN<T>(modelName);
    m->init((u64)dcf::orca::global::scale, inp);
    m->train();
    auto b = new Orca<T>();
    m->setBackend(b);
    m->optimize();
    dcf::orca::GPUModel<T> *gpuModel = new dcf::orca::GPUModel<T>();
    for (auto n : m->allNodesInExecutionOrder)
    {
        auto layer = n->layer;
        if (layer->name == "Conv2D")
        {
            assert(layer->mode == 1);
            auto convLayer = (Conv2D<T> *)(layer);
            int N, h, w, c;
            N = convLayer->inputDerivative.shape[0];
            h = convLayer->inputDerivative.shape[1];
            w = convLayer->inputDerivative.shape[2];
            c = convLayer->inputDerivative.shape[3];
            assert(c == convLayer->ci);
            auto orcaConv2D = new dcf::orca::Conv2DLayer<T>((int)dcf::orca::global::bw, (int)dcf::orca::global::bw, N, h, w, (int)convLayer->ci, (int)convLayer->fh, (int)convLayer->fw, (int)convLayer->co, (int)convLayer->padding, (int)convLayer->padding, (int)convLayer->padding, (int)convLayer->padding, (int)convLayer->stride, (int)convLayer->stride, convLayer->useBias, dcf::TruncateType::StochasticTR, dcf::TruncateType::StochasticTruncate, !layer->isFirst, layer->isFirst);
            auto filter = convLayer->getweights();
            // memcpy(orcaConv2D->F, filter.data, filter.size * sizeof(T));
            if (convLayer->useBias)
            {
                auto bias = convLayer->getbias();
                // memcpy(orcaConv2D->b, bias.data, bias.size * sizeof(T));
            }
            gpuModel->layers.push_back(orcaConv2D);
        }

        else if (layer->name == "MaxPool2D")
        {
            assert(layer->mode == 3);
            auto maxPoolLayer = (MaxPool2D<T> *)(layer);
            int bwToUse = dcf::orca::global::bw;
            bwToUse -= dcf::orca::global::scale;
            int N, h, w, c;
            N = maxPoolLayer->inputDerivative.shape[0];
            h = maxPoolLayer->inputDerivative.shape[1];
            w = maxPoolLayer->inputDerivative.shape[2];
            c = maxPoolLayer->inputDerivative.shape[3];
            auto orcaMaxPool = new dcf::orca::MaxPool2DLayer<T>(bwToUse, bwToUse, dcf::orca::global::bw, N, h, w, c, maxPoolLayer->ks, maxPoolLayer->ks, maxPoolLayer->stride, maxPoolLayer->stride, maxPoolLayer->padding, maxPoolLayer->padding, maxPoolLayer->padding, maxPoolLayer->padding);
            gpuModel->layers.push_back(orcaMaxPool);
        }
        else if (layer->name == "FC")
        {
            assert(layer->mode == 1);
            auto fcLayer = (FC<T> *)(layer);
            auto orcaFC = new dcf::orca::FCLayer<T>(dcf::orca::global::bw, dcf::orca::global::bw, (int)fcLayer->inputDerivative.shape[0], (int)fcLayer->out, (int)fcLayer->in, dcf::TruncateType::StochasticTR, dcf::TruncateType::StochasticTruncate, fcLayer->useBias, !layer->isFirst, layer->isFirst);
            auto W = fcLayer->getweights();
            // memcpy(orcaFC->W, W.data, W.size * sizeof(T));
            if (fcLayer->useBias)
            {
                auto bias = fcLayer->getbias();
                // memcpy(orcaFC->Y, bias.data, bias.size * sizeof(T));
            }
            gpuModel->layers.push_back(orcaFC);
        }
        else if (layer->name == "ReLU")
        {
            assert(layer->mode == 2);
            auto reluLayer = (ReLU<T> *)(layer);
            int r = layer->activation.size();
            auto orcaRelu = new dcf::orca::ReluExtendLayer<T>(dcf::orca::global::bw - dcf::orca::global::scale, dcf::orca::global::bw, r);
            gpuModel->layers.push_back(orcaRelu);
        }
    }
    int l = m->allNodesInExecutionOrder.size();
    gpuModel->batchSz = inp.shape[0];
    gpuModel->inpSz = inp.size();
    gpuModel->classes = m->allNodesInExecutionOrder[l - 1]->currTensor->shape[1];
    return gpuModel;
}

template <typename T>
dcf::orca::GPUModel<T> *getPiranhaCNN(std::string modelName, Tensor<T> inp)
{
    auto m = getCNN<T>(modelName);
    if (modelName.compare("P-SecureML") == 0)
    {
        Tensor<T> temp(nullptr, {inp.shape[0], inp.size() / inp.shape[0]});
        m->init((u64)dcf::orca::global::scale, temp);
    }
    else
    {
        m->init((u64)dcf::orca::global::scale, inp);
    }
    m->train();
    auto b = new Piranha<T>();
    m->setBackend(b);
    m->optimize();
    dcf::orca::GPUModel<T> *gpuModel = new dcf::orca::GPUModel<T>();
    for (auto n : m->allNodesInExecutionOrder)
    {
        auto layer = n->layer;
        if (layer->name == "Conv2D")
        {
            auto convLayer = (Conv2D<T> *)(layer);
            assert(!convLayer->useBias);
            int N, h, w, c;
            N = convLayer->inputDerivative.shape[0];
            h = convLayer->inputDerivative.shape[1];
            w = convLayer->inputDerivative.shape[2];
            c = convLayer->inputDerivative.shape[3];
            assert(c == convLayer->ci);
            auto gpuLayer = new dcf::orca::Conv2DLayer<T>((int)dcf::orca::global::bw, (int)dcf::orca::global::bw, (int)N, (int)h, (int)w, (int)convLayer->ci, (int)convLayer->fh, (int)convLayer->fw, (int)convLayer->co, (int)convLayer->padding, (int)convLayer->padding, (int)convLayer->padding, (int)convLayer->padding, (int)convLayer->stride, (int)convLayer->stride, convLayer->useBias, dcf::TruncateType::LocalARS, dcf::TruncateType::LocalARS, !layer->isFirst, layer->isFirst);
            gpuModel->layers.push_back(gpuLayer);
        }
        else if (layer->name == "FC")
        {
            auto fcLayer = (FC<T> *)(layer);
            auto gpuLayer = new dcf::orca::FCLayer<T>((int)dcf::orca::global::bw, (int)dcf::orca::global::bw, (int)fcLayer->inputDerivative.shape[0], (int)fcLayer->out, (int)fcLayer->in, dcf::TruncateType::LocalARS, dcf::TruncateType::LocalARS, fcLayer->useBias, !layer->isFirst, layer->isFirst);
            gpuModel->layers.push_back(gpuLayer);
        }
        else if (layer->name == "ReLU")
        {
            auto reluLayer = (ReLU<T> *)(layer);
            int r = layer->activation.size();
            // printf("r=%lu\n", r);
            int inputBw = dcf::orca::global::bw - dcf::orca::global::scale - layer->mode;
            auto gpuLayer = new dcf::orca::ReluLayer<T>(inputBw, dcf::orca::global::bw, r);
            gpuModel->layers.push_back(gpuLayer);
        }
        else if (layer->name == "AvgPool2D")
        {
            auto avgPoolLayer = (AvgPool2D<T> *)(layer);
            assert(n->parents.size() == 1);
            auto p = n->parents[0];
            auto &a = p->layer->activation;
            assert(a.shape.size() == 4);
            int N, h, w, c;
            N = a.shape[0];
            h = a.shape[1];
            w = a.shape[2];
            c = a.shape[3];
            auto gpuLayer = new dcf::orca::AvgPool2DLayer<T>(dcf::orca::global::bw, dcf::orca::global::bw, dcf::orca::global::scale, N, h, w, c, avgPoolLayer->ks, avgPoolLayer->ks, avgPoolLayer->stride, avgPoolLayer->stride, avgPoolLayer->padding, avgPoolLayer->padding, avgPoolLayer->padding, avgPoolLayer->padding, dcf::TruncateType::LocalARS, dcf::TruncateType::LocalARS);
            gpuModel->layers.push_back(gpuLayer);
        }
    }
    int l = m->allNodesInExecutionOrder.size();
    printf("########Layers=%d\n", l);
    gpuModel->batchSz = inp.shape[0];
    gpuModel->inpSz = inp.size();
    gpuModel->classes = m->allNodesInExecutionOrder[l - 1]->currTensor->shape[1];
    return gpuModel;
}