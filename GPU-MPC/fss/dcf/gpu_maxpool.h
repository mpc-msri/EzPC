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

#include "fss/gpu_maxpool.h"
#include "fss/gpu_and.h"
#include "gpu_relu.h"

namespace dcf
{

    template <typename T>
    struct GPUMaxpoolKey
    {
        GPU2RoundReLUKey<T> *reluKey;
        GPUAndKey *andKey;
    };

    template <typename T>
    GPUMaxpoolKey<T> readGPUMaxpoolKey(MaxpoolParams p, u8 **key_as_bytes)
    {
        GPUMaxpoolKey<T> k;
        int rounds = p.FH * p.FW - 1;
        // printf("Rounds=%d\n", rounds);
        k.reluKey = new GPU2RoundReLUKey<T>[rounds + 1];
        for (int i = 0; i < rounds; i++)
        {
            k.reluKey[i + 1] = readTwoRoundReluKey<T>(key_as_bytes);
            // printf("Round %d=%d relus\n", i + 1, k.reluKey[i + 1].N);
        }
        return k;
    }
}

#include "gpu_maxpool.cu"