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

#include "gpu_relu.h"
#include "gpu_lut.h"
#include "gpu_truncate.h"

template <typename T>
struct GPUGeluMuxKey
{
    T *c0, *c1;
};

template <typename T>
GPUGeluMuxKey<T> readGPUGeluMuxKey(uint8_t **key_as_bytes, int N)
{
    GPUGeluMuxKey<T> k;
    u64 memSz = 4 * N * sizeof(T); // num bytes
    k.c0 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    k.c1 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}

template <typename T, typename TClip>
struct GPUGeluKey
{
    int bw;
    GPUTruncateKey<T> trKey;
    GPUDReluKey dReluKey;
    u32 *icMask;
    GPUGeluMuxKey<TClip> muxKey;
    GPULUTKey<T> lutKey;
    GPUSelectKey<T> reluSelectKey;
};

template <typename T, typename TClip>
GPUGeluKey<T, TClip> readGpuGeluKey(uint8_t **key_as_bytes)
{
    GPUGeluKey<T, TClip> k;
    k.bw = *((int *)*key_as_bytes);
    *key_as_bytes += sizeof(int);
    k.trKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
    k.dReluKey = readGPUDReluKey(key_as_bytes);
    int N = k.trKey.N;
    // printf("###### Gelu N=%d\n", N);
    auto icMaskMemSize = ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    k.icMask = (u32 *)*key_as_bytes;
    *key_as_bytes += icMaskMemSize;
    k.muxKey = readGPUGeluMuxKey<TClip>(key_as_bytes, N);
    k.lutKey = readGPULUTKey<T>(key_as_bytes);
    k.reluSelectKey = readGPUSelectKey<T>(key_as_bytes, N);
    return k;
}


#include "gpu_gelu.cu"