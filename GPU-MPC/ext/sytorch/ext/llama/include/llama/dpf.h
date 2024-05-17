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
#include <cryptoTools/Common/Defines.h>
#include <llama/group_element.h>
#include <llama/keypack.h>

std::pair<DPFKeyPack, DPFKeyPack> keyGenDPF(int bin, int bout, GroupElement idx, GroupElement payload);
GroupElement evalDPF_EQ(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_GT(int party, DPFKeyPack &key, GroupElement x);
GroupElement evalDPF_LT(int party, DPFKeyPack &key, GroupElement x);
void evalAll(int party, DPFKeyPack &key, GroupElement rightShift, GroupElement *out);
GroupElement evalAll_reduce(int party, DPFKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);

std::pair<DPFETKeyPack, DPFETKeyPack> keyGenDPFET(int bin, GroupElement idx);
std::pair<GroupElement, GroupElement> evalAll_reduce_et(int party, DPFETKeyPack &key, GroupElement rightShift, const std::vector<GroupElement> &tab);
GroupElement evalDPFET_LT(int party, const DPFETKeyPack &key, GroupElement x);
