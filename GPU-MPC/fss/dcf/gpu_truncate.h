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

#include "gpu_dcf.h"
#include "gpu_relu.h"
#include "utils/gpu_stats.h"
#include "utils/gpu_comms.h"

namespace dcf
{
    enum TruncateType
    {
        None,
        // LocalLRS,
        StochasticTR,
        LocalARS,
        StochasticTruncate
    };

    using GPUMaskedDCFKey = GPUDReluKey;

    template <typename T>
    struct GPUStTRKey
    {
        int bin, bout, shift, N;
        GPUMaskedDCFKey lsbKey;
        T *lsbCorr;
    };

    template <typename T>
    struct GPUSignExtendKey
    {
        int bin, bout, N;
        GPUMaskedDCFKey dcfKey;
        T *t, *p;
    };

    template <typename T>
    struct GPUTruncateKey
    {
        GPUStTRKey<T> stTRKey;
        GPUSignExtendKey<T> signExtendKey;
    };

    const auto readGPUMaskedDCFKey = readGPUDReluKey;

    template <typename T>
    GPUSignExtendKey<T> readGPUSignExtendKey(uint8_t **key_as_bytes)
    {
        GPUSignExtendKey<T> k;
        k.bin = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.bout = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.N = *((int *)*key_as_bytes);
        *key_as_bytes += sizeof(int);

        k.dcfKey = readGPUMaskedDCFKey(key_as_bytes);
        // change this ugly name
        size_t memSz = k.dcfKey.dcfKey.M * sizeof(T);
        // printf("Sign extend=%d\n", k.dcfKey.dcfKey.M);
        k.t = (T *)*key_as_bytes;
        *key_as_bytes += memSz;
        k.p = (T *)*key_as_bytes;
        *key_as_bytes += 2 * memSz;
        return k;
    }

    template <typename T>
    GPUStTRKey<T> readGPUStTRKey(u8 **key_as_bytes)
    {
        GPUStTRKey<T> k;
        memcpy(&k, *key_as_bytes, 4 * sizeof(int));
        *key_as_bytes += 4 * sizeof(int);
        k.lsbKey = readGPUMaskedDCFKey(key_as_bytes);
        size_t memSz = k.N * sizeof(T);
        k.lsbCorr = (T *)*key_as_bytes;
        *key_as_bytes += 2 * memSz;
        return k;
    }

    template <typename T>
    GPUTruncateKey<T> readGPUTrStochasticKey(u8 **key_as_bytes)
    {
        GPUTruncateKey<T> k;
        k.stTRKey = readGPUStTRKey<T>(key_as_bytes);
        k.signExtendKey = readGPUSignExtendKey<T>(key_as_bytes);
        return k;
    }

    template <typename T>
    GPUTruncateKey<T> readGPUTruncateKey(TruncateType t, uint8_t **key_as_bytes)
    {
        GPUTruncateKey<T> k;
        switch (t)
        {
        case TruncateType::StochasticTruncate:
            k = readGPUTrStochasticKey<T>(key_as_bytes);
            break;
        case TruncateType::StochasticTR:
            k.stTRKey = readGPUStTRKey<T>(key_as_bytes);
            break;
        default:
            assert(t == TruncateType::None || t == TruncateType::LocalARS || t == TruncateType::StochasticTR);
        }
        return k;
    }
}

#include "gpu_truncate.cu"