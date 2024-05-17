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

#include <llama/dcf.h>
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for (int i = 0; i < 256; ++i)
    {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 10000000;
    int bin = 64;
    int bout = 64;

    std::pair<DCFKeyPack, DCFKeyPack> *keys = new std::pair<DCFKeyPack, DCFKeyPack>[samples];
    GroupElement *r = new GroupElement[samples];
    GroupElement *op = new GroupElement[samples];

    for (int j = 0; j < samples; ++j)
    {
        r[j] = rand();
        // GroupElement idx = rand();
        //  % (1LL << bin);
        keys[j] = keyGenDCF(bin, bout, r[j], 1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < samples; ++j)
    {
        auto &key0 = keys[j].first;
        // auto &key1 = keys[j].second;
        GroupElement inp = rand();
        evalDCF(0, &op[j], inp, key0);
        // GroupElement res1;
        // evalDCF(1, &res1, inp, key1);
        // auto res = res0 + res1;
        // mod(res, bout);
        // if (inp < r[j])
        // {
        //     always_assert(res == 1);
        // }
        // else
        // {
        //     always_assert(res == 0);
        // }
    }
    std::cout << "Op=" << op[rand() % samples] << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken=" << elapsed.count() << std::endl;
}