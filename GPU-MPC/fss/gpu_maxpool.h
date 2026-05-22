// Author: Neha Jawalkar
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

#pragma once

#include "gpu_avgpool.h"
#include "gpu_relu.h"
#include "gpu_mul.h"

using MaxpoolParams = AvgPoolParams;

inline int getInSz(MaxpoolParams p)
{
    int sz;
    if (p.isLowerTriangular)
    {
        assert(p.imgH == p.imgW);
        assert(p.C == 1);
        sz = p.N * (p.imgH * (p.imgH + 1)) / 2;
    }
    else
    {
        sz = p.N * p.imgH * p.imgW * p.C;
    }
    return sz;
}

template <typename T>
struct GPUMaxpoolKey
{
    int rounds;
    GPUReluKey<T> *reluKey;
    // GPUAndKey* andKey;
};

template <typename T>
GPUMaxpoolKey<T> readGPUMaxpoolKey(MaxpoolParams p, u8 **key_as_bytes)
{
    GPUMaxpoolKey<T> k;
    k.rounds = *((int *)*key_as_bytes);
    *key_as_bytes += sizeof(int);
    k.reluKey = new GPUReluKey<T>[/*p.FH * p.FW*/ k.rounds];
    for (int i = 0; i < /*p.FH*/ k.rounds; i++)
    {
        k.reluKey[i] = readReluKey<T>(key_as_bytes);
    }
    return k;
}

#include "gpu_maxpool.cu"