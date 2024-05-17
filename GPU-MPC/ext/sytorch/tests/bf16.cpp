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

#include "../ext/llama/fixtobfloat16.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 40;
    int numTrials = 255;
    
    for (int i = 1; i <= numTrials; ++i) {
        GroupElement rin = random_ge(bin);
        GroupElement rout = random_ge(14);
        GroupElement x = i;
        GroupElement xhat = x + rin;
        mod(xhat, bin);

        auto keys = keyGenF2BF16(bin, rin, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        auto t0 = evalF2BF16_1(0, xhat, key0);
        auto t1 = evalF2BF16_1(1, xhat, key1);
        GroupElement k = t0.first + t1.first;
        GroupElement m = t0.second + t1.second;

        auto xm0 = evalF2BF16_2(0, xhat, k, m, key0);
        auto xm1 = evalF2BF16_2(1, xhat, k, m, key1);
        GroupElement xm = xm0 + xm1;

        auto r0 = evalF2BF16_3(0, k, xm, key0);
        auto r1 = evalF2BF16_3(1, k, xm, key1);

        GroupElement r = r0 + r1 - rout;

        mod(r, 13);

        k = r % (1LL << 6);
        m = r >> 6;

        always_assert(((m+128) >> (7-k)) == x);
    }
}