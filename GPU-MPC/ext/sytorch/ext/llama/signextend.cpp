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

#include "signextend.h"
#include <llama/dcf.h>

std::pair<SignExtend2KeyPack, SignExtend2KeyPack> keyGenSignExtend2(int bin, int bout, GroupElement rin, GroupElement rout)
{
    std::pair<SignExtend2KeyPack, SignExtend2KeyPack> keys;
    
    auto dcfKeys = keyGenDCF(bin, 1, rin, 1);
    keys.first.dcfKey = dcfKeys.first;
    keys.second.dcfKey = dcfKeys.second;

    GroupElement rw = random_ge(1);
    auto rw_split = splitShare(rw, 1);
    keys.first.rw = rw_split.first;
    keys.second.rw = rw_split.second;

    GroupElement p[2];
    if ((rw % 2) == 0) {
        p[0] = rout - rin - (1ULL << (bin - 1));
        p[1] = rout - rin + (1ULL << (bin - 1));
    } else {
        p[0] = rout - rin + (1ULL << (bin - 1));
        p[1] = rout - rin - (1ULL << (bin - 1));
    }

    for (int i = 0; i < 2; ++i) {
        auto p_split = splitShare(p[i], bout);
        keys.first.p[i] = p_split.first;
        keys.second.p[i] = p_split.second;
    }

    return keys;
}

std::pair<SlothSignExtendKeyPack, SlothSignExtendKeyPack> keyGenSlothSignExtend(int bin, int bout, GroupElement rin, GroupElement w, GroupElement rout)
{
    SlothSignExtendKeyPack k0, k1;
    k0.bin = bin;
    k1.bin = bin;
    k0.bout = bout;
    k1.bout = bout;

    auto rout_split = splitShare(rout - rin - (1LL << (bin - 1)), bout);
    k0.rout = rout_split.first;
    k1.rout = rout_split.second;

    mod(w, 1);
    auto w_split = splitShare(w, bout);
    k0.select = w_split.first;
    k1.select = w_split.second;

    return std::make_pair(k0, k1);
}

GroupElement evalSlothSignExtend(int party, GroupElement x, GroupElement w, const SlothSignExtendKeyPack &kp)
{
    mod(w, 1);
    return party * x + kp.rout + (1LL << kp.bin) * ((1 - w) * kp.select + w * (party - kp.select));
}
