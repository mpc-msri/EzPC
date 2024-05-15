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

std::pair<ScmpKeyPack, ScmpKeyPack> keyGenSCMP(int Bin, int Bout, GroupElement rin1, GroupElement rin2,
                                GroupElement rout);

GroupElement evalSCMP(int party, ScmpKeyPack key, GroupElement x, GroupElement y);

std::pair<ARSKeyPack, ARSKeyPack> keyGenARS(int Bin, int Bout, uint64_t shift, GroupElement rin, GroupElement rout);

GroupElement evalARS(int party, GroupElement x, uint64_t shift, const ARSKeyPack &k);

std::pair<EdabitsPrTruncKeyPack, EdabitsPrTruncKeyPack> keyGenEdabitsPrTrunc(int bw, int shift, GroupElement rin, GroupElement rout);

std::pair<TruncateReduceKeyPack, TruncateReduceKeyPack> keyGenTruncateReduce(int bin, int shift, GroupElement rin, GroupElement rout);
GroupElement evalTruncateReduce(int party, GroupElement x, const TruncateReduceKeyPack &k);

std::pair<SlothLRSKeyPack, SlothLRSKeyPack> keyGenSlothLRS(int bin, int shift, GroupElement rin, GroupElement rinWrap, GroupElement rout);
GroupElement evalSlothLRS(int party, GroupElement x, GroupElement w, const SlothLRSKeyPack &k);
