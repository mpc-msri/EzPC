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

#include <llama/dpf.h>
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 4;
    int bout = 64;

    for (int idx = 0; idx < 16; ++idx)
    {
        auto keys = keyGenDPF(bin, bout, idx, 1);
        auto& key0 = keys.first;
        auto& key1 = keys.second;

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_EQ(0, key0, i) ^ evalDPF_EQ(1, key1, i));
            if (i == idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_GT(0, key0, i) ^ evalDPF_GT(1, key1, i));
            if (i > idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        for (int i = 0; i < 16; ++i)
        {
            auto y = (evalDPF_LT(0, key0, i) ^ evalDPF_LT(1, key1, i));
            if (i < idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        GroupElement out0[16];
        GroupElement out1[16];
        evalAll(0, key0, 0, out0);
        evalAll(1, key1, 0, out1);
        for (int i = 0; i < 16; ++i)
        {
            auto y = (out0[i] + out1[i]);
            if (i == idx)
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        evalAll(0, key0, 7, out0);
        evalAll(1, key1, 7, out1);
        for (int i = 0; i < 16; ++i)
        {
            auto y = (out0[i] + out1[i]);
            if (i == ((idx+7)%16))
            {
                always_assert(y == 1);
            }
            else
            {
                always_assert(y == 0);
            }
        }

        GroupElement res0, res1;
        std::vector<GroupElement> tab(16);
        for (int i = 0; i < 16; ++i)
        {
            tab[i] = rand();
        }
        res0 = evalAll_reduce(0, key0, 0, tab);
        res1 = evalAll_reduce(1, key1, 0, tab);
        always_assert(res0 + res1 == tab[idx]);

    }
}