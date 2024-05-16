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

#include "utils/gpu_random.h"

#include "gpu_truncate.h"
#include "gpu_window.h"
#include "gpu_lut.h"

template <typename T>
struct GPUSqKey
{
    T *a, *c;
};

template <typename T>
GPUSqKey<T> readGPUSqKey(int N, u8** key_as_bytes)
{
    GPUSqKey<T> k;
    u64 memSz = N * sizeof(T);
    k.a = (T*) *key_as_bytes;
    *key_as_bytes += memSz;
    k.c = (T*) *key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}


template <typename T>
struct GPULayerNormKey
{
    GPUTruncateKey<T> muTrKey;
    GPUSqKey<T> sqKey;
    // GPULUTKey<T> invSqrtKey;
    GPUMulKey<T> wMulKey1;
    GPUMulKey<T> wMulKey2;
};

AvgPoolParams transposeWindow(AvgPoolParams p1) {
    assert(p1.FH == 1 && p1.FW == p1.imgW && p1.strideH == 1 && p1.strideW == p1.imgW);
    AvgPoolParams p2;
    memcpy(&p2, &p1, sizeof(AvgPoolParams));
    p2.FH = p1.imgH;
    p2.FW = 1;
    p2.strideH = p1.imgH;
    p2.strideW = 1;
    initPoolParams(p2);
    return p2;
}

template <typename T>
GPULayerNormKey<T> readGPULayerNormKey(AvgPoolParams p, u8** key_as_bytes, bool computeMu = true)
{
    GPULayerNormKey<T> k;
    auto inSz = getInSz(p);
    auto mSz = getMSz(p);
    if(computeMu) {
        k.muTrKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
        // printf("Num Truncations=%d\n", k.muTrKey.N);
    }
    k.sqKey = readGPUSqKey<T>(inSz, key_as_bytes);
    // printf("Num sq=%ld\n", inSz);
    // k.invSqrtKey = readGPULUTKey<T>(key_as_bytes);
    k.wMulKey1 = readGPUWindowMulKey<T>(p, TruncateType::TrWithSlack, key_as_bytes);
    // printf("here$$$\n");
    auto p2 = transposeWindow(p);
    k.wMulKey2 = readGPUWindowMulKey<T>(p2, TruncateType::TrWithSlack, key_as_bytes);
    // printf("here$$$\n");
    return k;
}

#include "gpu_layernorm.cu"
