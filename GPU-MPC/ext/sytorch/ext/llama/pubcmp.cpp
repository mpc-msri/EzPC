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

#include "pubcmp.h"
#include <llama/dcf.h>

std::pair<PubCmpKeyPack, PubCmpKeyPack> keyGenPubCmp(int bin, GroupElement rin, GroupElement rout)
{
    mod(rin, bin);
    GroupElement r1, r0;
    
    auto dcfN = keyGenDCF(bin, 1, rin, 1);

    GroupElement a = rout & 1;

    PubCmpKeyPack k0, k1;
    k0.bin = bin; k1.bin = bin;
    k0.dcfKey = dcfN.first; k1.dcfKey = dcfN.second;
    
    auto aSplit = splitShare(a, 1);
    k0.rout = aSplit.first; k1.rout = aSplit.second;
    return std::make_pair(k0, k1);
}

GroupElement evalPubCmp(int party, GroupElement x, GroupElement c, const PubCmpKeyPack &key) {
    GroupElement y = x - c;
    mod(y, key.bin);
    mod(x, key.bin);
    GroupElement t2 = 0;
    GroupElement t1 = 0;
    evalDCF(party, &t2, y, key.dcfKey);
    evalDCF(party, &t1, x, key.dcfKey);
    GroupElement res = t2 - t1 + key.rout;
    if (party == 1) {
        GroupElement N = -c;
        mod(N, key.bin);
        if (y >= N) {
            res += 1;
        }
    }
    mod(res, 1);
    return res;
}
