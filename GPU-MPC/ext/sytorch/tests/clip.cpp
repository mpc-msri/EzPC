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

#include "../ext/llama/clip.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 40;
    int numTrials = 10000;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(bin);
        GroupElement x = random_ge(16);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenClip(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        auto routClip = keys.first.cmpKey.rout ^ keys.second.cmpKey.rout;
        GroupElement t0 = evalClip_1(0, 16, xhat, key0);
        GroupElement t1 = evalClip_1(1, 16, xhat, key1);
        GroupElement t = t0 ^ t1;
        always_assert((t ^ routClip) == 1);

        GroupElement t0_2 = evalClip_2(0, 16, t, xhat, key0);
        GroupElement t1_2 = evalClip_2(1, 16, t, xhat, key1);

        GroupElement t_2 = t0_2 + t1_2 - rout;
        mod(t_2, bin);

        always_assert(t_2 == x);
    }

    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(bin);
        GroupElement x = random_ge(bin);
        if (x < (1LL<<16)) {
            x += (1LL<<16);
        }
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenClip(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        auto routClip = keys.first.cmpKey.rout ^ keys.second.cmpKey.rout;
        GroupElement t0 = evalClip_1(0, 16, xhat, key0);
        GroupElement t1 = evalClip_1(1, 16, xhat, key1);
        GroupElement t = t0 ^ t1;
        always_assert((t ^ routClip) == 0);

        GroupElement t0_2 = evalClip_2(0, 16, t, xhat, key0);
        GroupElement t1_2 = evalClip_2(1, 16, t, xhat, key1);

        GroupElement t_2 = t0_2 + t1_2 - rout;
        mod(t_2, bin);

        always_assert(t_2 == ((1LL<<16) - 1));
    }
}
