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
    int samples = 18432;

    std::vector<GroupElement> tab(1LL<<bin);
    for (int i = 0; i < tab.size(); ++i)
    {
        tab[i] = rand();
    }

    DPFKeyPack *keys = new DPFKeyPack[samples];
    GroupElement *res = new GroupElement[samples];
    for (int idx = 0; idx < samples; ++idx)
    {
        auto keypair = keyGenDPF(bin, bout, rand(), 1);
        keys[idx] = keypair.first;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int idx = 0; idx < samples; ++idx)
    {
        res[idx] = evalAll_reduce(0, keys[idx], 0, tab);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << compute_time / 1000.0 << " ms" << std::endl;

    for (int idx = 0; idx < samples; ++idx)
    {
        std::cerr << res[idx] << std::endl;
    }
}