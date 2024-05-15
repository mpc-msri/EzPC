/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <llama/keypack.h>

std::pair<ReluKeyPack, ReluKeyPack> keyGenRelu(int Bin, int Bout,
                        GroupElement rin, GroupElement rout, GroupElement routDrelu = 0);

// GroupElement evalRelu(int party, GroupElement x, const ReluKeyPack &k);
GroupElement evalRelu(int party, GroupElement x, const ReluKeyPack &k, GroupElement *drelu = nullptr);

std::pair<MaxpoolKeyPack, MaxpoolKeyPack> keyGenMaxpool(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement rout, GroupElement routBit);
GroupElement evalMaxpool(int party, GroupElement x, GroupElement y, const MaxpoolKeyPack &k, GroupElement &bit);

std::pair<Relu2RoundKeyPack, Relu2RoundKeyPack> keyGenRelu2Round(int effectiveBw, int bin, GroupElement rin, GroupElement routRelu, GroupElement rout);
GroupElement evalRelu2_drelu(int party, GroupElement x, const Relu2RoundKeyPack &key);
GroupElement evalRelu2_mult(int party, GroupElement x, GroupElement y, const Relu2RoundKeyPack &key);

std::pair<MaxpoolDoubleKeyPack, MaxpoolDoubleKeyPack> keyGenMaxpoolDouble(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement routBit, GroupElement rout);
GroupElement evalMaxpoolDouble_1(int party, GroupElement x, GroupElement y, const MaxpoolDoubleKeyPack &k);
GroupElement evalMaxpoolDouble_2(int party, GroupElement x, GroupElement y, GroupElement s, const MaxpoolDoubleKeyPack &k);

std::pair<SlothDreluKeyPack, SlothDreluKeyPack> keyGenSlothDrelu(int bin, GroupElement rin, GroupElement rout);
GroupElement evalSlothDrelu(int party, GroupElement x, const SlothDreluKeyPack &k);
