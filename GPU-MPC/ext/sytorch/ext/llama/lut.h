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

#pragma once

#include <llama/keypack.h>

std::pair<LUTKeyPack, LUTKeyPack> keyGenLUT(int bin, int bout, GroupElement rin, GroupElement rout);
std::pair<LUTSSKeyPack, LUTSSKeyPack> keyGenLUTSS(int bin, int bout, GroupElement rin, GroupElement rout);
std::pair<GroupElement, GroupElement> evalLUTSS_1(int party, GroupElement x, const std::vector<GroupElement> &tab, const LUTSSKeyPack &kp);
GroupElement evalLUTSS_2(int party, GroupElement res, GroupElement corr, const LUTSSKeyPack &kp);

std::pair<LUTDPFETKeyPack, LUTDPFETKeyPack> keyGenLUTDPFET(int bin, int bout, GroupElement rin, GroupElement routRes, GroupElement routCorr);
std::pair<GroupElement, GroupElement> evalLUTDPFET_1(int party, GroupElement x, const std::vector<GroupElement> &tab, LUTDPFETKeyPack &kp);
GroupElement evalLUTDPFET_2(int party, GroupElement res, GroupElement corr, const LUTDPFETKeyPack &kp);
