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

#include "../ext/llama/wrap.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

void dpf_main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 64;
    int numTrials = 1000;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(1);
        GroupElement x = random_ge(bin);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenWrapDPF(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        GroupElement t0 = evalWrapDPF(0, xhat, key0);
        GroupElement t1 = evalWrapDPF(1, xhat, key1);
        GroupElement t = t0 ^ t1 ^ rout;
        if (xhat < r) {
            always_assert(t == 1);
        } else {
            always_assert(t == 0);
        }
    }
}

void ss_main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 6;
    int numTrials = 1000;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(1);
        GroupElement x = random_ge(bin);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenWrapSS(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        GroupElement t0 = evalWrapSS(0, xhat, key0);
        GroupElement t1 = evalWrapSS(1, xhat, key1);
        GroupElement t = t0 ^ t1 ^ rout;
        if (xhat < r) {
            always_assert(t == 1);
        } else {
            always_assert(t == 0);
        }
    }
}

int main()
{
    dpf_main();
    ss_main();
}
