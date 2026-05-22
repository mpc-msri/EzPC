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

/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>
#include <llama/config.h>
#include <llama/prng.h>
#include <omp.h>

using GroupElement = uint64_t;

inline void mod(GroupElement &a, int bw)
{
    if (bw != 64)
        a = a & ((uint64_t(1) << bw) - 1); 
}

inline GroupElement random_ge(int bw)
{
    GroupElement a;
    int tid = omp_get_thread_num();
    a = LlamaConfig::prngs[tid].get<uint64_t>();
    mod(a, bw);
    return a;
}

inline std::pair<GroupElement, GroupElement> splitShare(const GroupElement& a, int bw)
{
    GroupElement a1, a2;
    a1 = random_ge(bw);
    // a1 = 0;
    mod(a1, bw);
    a2 = a - a1;
    mod(a2, bw);
    return std::make_pair(a1, a2);
}

inline std::pair<GroupElement, GroupElement> splitShareXor(const GroupElement& a, int bw)
{
    GroupElement a1, a2;
    a1 = random_ge(bw);
    a2 = a ^ a1;
    return std::make_pair(a1, a2);
}

inline std::pair<GroupElement, GroupElement> splitShareCommonPRNG(const GroupElement& a, int bw)
{
    GroupElement a1, a2;
    a1 = prngShared.get<uint64_t>();
    // a1 = 0;
    mod(a1, bw);
    a2 = a - a1;
    mod(a2, bw);
    return std::make_pair(a1, a2);
}

inline GroupElement pow(GroupElement x, uint64_t e)
{
    if (e == 0)
    {
        return 1;
    }
    GroupElement res = pow(x, e / 2);
    if (e % 2 == 0)
    {
        return res * res;
    }
    else
    {
        return res * res * x;
    }
}

inline GroupElement msb(GroupElement a, int bw)
{
    return (a >> (bw - 1)) & 1;
}