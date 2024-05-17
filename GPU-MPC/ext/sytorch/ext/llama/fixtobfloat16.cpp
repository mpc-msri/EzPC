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

#include "fixtobfloat16.h"
#include <llama/dcf.h>
#include <llama/dpf.h>

std::pair<F2BF16KeyPack, F2BF16KeyPack> keyGenF2BF16(int bin, GroupElement rin, GroupElement rout)
{
    mod(rin, bin);
    GroupElement r1, r0;
    
    auto dcfN = keyGenDCF(bin, bin, rin, 1);
    F2BF16KeyPack k0, k1;
    k0.bin = bin; k1.bin = bin;
    k0.dcfKey = dcfN.first; k1.dcfKey = dcfN.second;

    GroupElement rout_k = random_ge(bin);
    GroupElement rout_m = random_ge(bin);
    GroupElement rout_xm = random_ge(bin);
    GroupElement prod = rin * rout_m + rout_xm;

    auto dcfTruncate = keyGenDCF(bin - 8, 8, rout_xm % (1LL << (bin - 8)), 1);
    k0.dcfTruncate = dcfTruncate.first;
    k1.dcfTruncate = dcfTruncate.second;

    auto rout_k_split = splitShare(rout_k, bin);
    auto rout_m_split = splitShare(rout_m, bin);
    auto rin_split = splitShare(rin, bin);
    auto prod_split = splitShare(prod, bin);
    auto rProd_split = splitShare(rout_xm >> (bin - 8), bin);

    // GroupElement rout_real = rout - rout_k;
    auto rout_split = splitShare(rout - rout_k, 13);

    k0.rout_k = rout_k_split.first;
    k1.rout_k = rout_k_split.second;

    k0.rout_m = rout_m_split.first;
    k1.rout_m = rout_m_split.second;

    k0.rin = rin_split.first;
    k1.rin = rin_split.second;

    k0.prod = prod_split.first;
    k1.prod = prod_split.second;

    k0.rout = rout_split.first;
    k1.rout = rout_split.second;

    k0.rProd = rProd_split.first;
    k1.rProd = rProd_split.second;

    return std::make_pair(k0, k1);
}

std::pair<GroupElement, GroupElement> evalF2BF16_1(int party, GroupElement x, const F2BF16KeyPack &key)
{
    GroupElement t1 = 0;
    mod(x, key.bin);
    evalDCF(party, &t1, x, key.dcfKey);

    GroupElement res_prev = 0;
    GroupElement k_final = 0;
    GroupElement m_final = 0;
    for (int i = 1; i < key.bin; ++i)
    {
        GroupElement c = (1LL << i);
        GroupElement y = x - c;
        mod(y, key.bin);
        GroupElement t2 = 0;
        evalDCF(party, &t2, y, key.dcfKey);
        GroupElement res = t2 - t1;
        if (party == 1) {
            GroupElement N = -c;
            mod(N, key.bin);
            if (y >= N) {
                res += 1;
            }
        }
        // res contains  1{x < c}
        k_final += (i - 1) * (res - res_prev);
        m_final += (1LL << (key.bin - i)) * (res - res_prev);
        res_prev = res;
    }

    k_final += key.rout_k;
    m_final += key.rout_m;

    return std::make_pair(k_final, m_final);
}

GroupElement evalF2BF16_2(int party, GroupElement x, GroupElement k, GroupElement m, const F2BF16KeyPack &key)
{
    GroupElement res = -key.rin * m - x * key.rout_m + key.prod;
    if (party == 1)
    {
        res += m * x;
    }
    return res;
}

GroupElement evalF2BF16_3(int party, GroupElement k, GroupElement xm, const F2BF16KeyPack &key) {
    GroupElement t;
    evalDCF(party, &t, xm % (1LL << (key.bin - 8)), key.dcfTruncate);
    GroupElement res = party * (xm >> (key.bin - 8)) - key.rProd - t;
    if (party == 1)
    {
        res -= 128; // as the top bit is always 1
    }

    res = res << 6;
    if (party == 1)
        res = res + k;

    return res + key.rout;
}
