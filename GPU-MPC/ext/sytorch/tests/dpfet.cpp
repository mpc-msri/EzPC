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

    int bin = 16;
    int bout = 64;

    std::vector<GroupElement> tab(1LL<<bin);
    for (int i = 0; i < (1LL<<bin); ++i)
    {
        tab[i] = rand();
    }

    for (int i = 0; i < 1000; ++i)
    {
        GroupElement idx = rand() % (1LL<<bin);
        auto keys = keyGenDPFET(bin, idx);
        auto& key0 = keys.first;
        auto& key1 = keys.second;

        GroupElement rot = rand() % (1LL<<bin);

        auto res0 = evalAll_reduce_et(0, key0, rot, tab);
        auto res1 = evalAll_reduce_et(1, key1, rot, tab);
        // always_assert(res0 + res1 == tab[idx]);
        always_assert((res0.first +  res1.first) * (res0.second + res1.second) == tab[(idx+rot) % (1LL<<bin)]);

    }
}