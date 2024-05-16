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

#include <cuda.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_stats.h"
#include "utils/gpu_comms.h"

#include "gpu_relu.h"

using GPUMaskedDPFKey = GPUDReluKey;
constexpr auto readGPUMaskedDPFKey = readGPUDReluKey;

template <typename T>
struct GPUTrCorrKey
{
    GPUMaskedDPFKey mDpfKey;
    T *corr;
};

template <typename T>
struct GPUTruncateKey
{
    int bin, shift, bout, N;
    GPUTrCorrKey<T> lsbKey, msbKey;
};

enum TruncateType
{
    None,
    LocalLRS,
    LocalARS,
    TrWithSlack,
    TrFloor
};

template <typename T>
GPUTrCorrKey<T> readGPUTrCorrKey(u8 **key_as_bytes)
{
    GPUTrCorrKey<T> k;
    k.mDpfKey = readGPUMaskedDPFKey(key_as_bytes);
    size_t memSz = k.mDpfKey.dpfKey.M * sizeof(T);
    k.corr = (T *)*key_as_bytes;
    *key_as_bytes += 2 * memSz;
    return k;
}

template <typename T>
GPUTruncateKey<T> readGPUTrWithSlackKey(uint8_t **key_as_bytes)
{
    GPUTruncateKey<T> k;
    memcpy(&k, *key_as_bytes, 4 * sizeof(int));
    *key_as_bytes += 4 * sizeof(int);
    k.lsbKey = readGPUTrCorrKey<T>(key_as_bytes);
    // correct the msb only if needed
    size_t memSz = k.N * sizeof(T);
    k.msbKey.corr = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}

template <typename T>
GPUTruncateKey<T> readGPUTrFloorKey(uint8_t **key_as_bytes)
{
    GPUTruncateKey<T> k;
    memcpy(&k, *key_as_bytes, 4 * sizeof(int));
    *key_as_bytes += 4 * sizeof(int);

    k.lsbKey = readGPUTrCorrKey<T>(key_as_bytes);
    if (k.bout > k.bin - k.shift)
        k.msbKey = readGPUTrCorrKey<T>(key_as_bytes);
    return k;
}

template <typename T>
GPUTruncateKey<T> readGPUTruncateKey(TruncateType t, uint8_t **key_as_bytes)
{
    GPUTruncateKey<T> k;
    switch (t)
    {
    case TruncateType::TrWithSlack:
        k = readGPUTrWithSlackKey<T>(key_as_bytes);
        break;
    case TruncateType::TrFloor:
        k = readGPUTrFloorKey<T>(key_as_bytes);
        break;
    default:
        assert(t == TruncateType::None || t == TruncateType::LocalARS || t == TruncateType::LocalLRS);
    }
    return k;
}

template <typename T>
void checkTrFloor(int bin, int bout, int shift, int N, T *h_masked_A, T *h_mask_A, T *h_A_ct)
{
    // printf("N=%d\n", N);
    for (int i = 0; i < N; i++)
    {
        auto truncated_A = cpuArs(h_A_ct[i], bin, shift);
        cpuMod(truncated_A, bout);
        auto output = h_masked_A[i] - h_mask_A[i];
        cpuMod(output, bout);
        auto diff = output - truncated_A;
        if (i < 10 || diff != T(0))
            printf("%d: %ld %ld %ld %ld, %ld, %ld\n", i, u64(output), u64(truncated_A), u64(h_A_ct[i]), h_masked_A[i], h_mask_A[i], h_A_ct[i]);
        assert(diff == T(0));
    }
}

template <typename T>
void checkTrStochastic(int bin, int bout, int shift, int N, T *h_masked_O, T *h_mask_O, T *h_I, T *h_r)
{
    for (int i = 0; i < N; i++)
    {
        auto unmasked_o = h_masked_O[i] - h_mask_O[i];
        cpuMod(unmasked_o, bout);
        auto trInp = (h_I[i] + (1ULL << (bin - 1))) >> shift;
        cpuMod(trInp, bin - shift);
        T temp = h_I[i];
        cpuMod(temp, shift);
        if (h_r[i] < temp)
        {
            trInp += 1;
            cpuMod(trInp, bin - shift);
        }
        trInp -= (1ULL << (bin - shift - 1));
        cpuMod(trInp, bout);
        if (i < 10 || unmasked_o != trInp)
            printf("%d=%lu %lu %lu %lu %u\n", i, unmasked_o, trInp, h_r[i], temp, h_r[i] < temp);
        assert(unmasked_o == trInp); // <= 1);
    }
}

#include "gpu_truncate.cu"