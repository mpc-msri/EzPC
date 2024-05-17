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

#include "../ext/llama/lut.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 8;
    int bout = 64;
    std::vector<GroupElement> tab(256);
    for (int i = 0; i < 256; ++i)
    {
        tab[i] = i + 1;
    }

    for (int i = 0; i < 1000; ++i)
    {
        GroupElement rin = rand() % 256;
        GroupElement x = rand() % 256;
        GroupElement xhat = (x + rin) % 256;

        auto keys = keyGenLUTSS(bin, bout, rin, 0);
        auto y0 = evalLUTSS_1(0, xhat, tab, keys.first);
        auto y1 = evalLUTSS_1(1, xhat, tab, keys.second);
        GroupElement res = y0.first + y1.first;
        GroupElement corr = y0.second + y1.second;
        GroupElement f0 = evalLUTSS_2(0, res, corr, keys.first);
        GroupElement f1 = evalLUTSS_2(1, res, corr, keys.second);
        GroupElement y = f0 + f1;
        always_assert(y == tab[x]);
    }
}
