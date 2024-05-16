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

#include "utils/gpu_data_types.h"
#include "gpu_truncate.h"

struct MatmulParams
{
    // multiply two matrices of input bitwidth bw
    // truncate by shift and return the output in output bitwidth bw
    int bw, shift;
    int batchSz;
    int M, K, N;
    int size_A, size_B, size_C;
    int ld_A, ld_B, ld_C;
    int stride_A, stride_B, stride_C;
    bool rowMaj_A, rowMaj_B, rowMaj_C;
    bool cIsLowerTriangular = false;
};

template <typename T>
struct GPUMatmulKey
{
    // MatmulParams p;
    u64 mem_size_A, mem_size_B, mem_size_C;
    T *A, *B, *C;
    GPUTruncateKey<T> trKey;
};

template <typename T>
GPUMatmulKey<T> readGPUMatmulKey(MatmulParams p, TruncateType t, uint8_t **key_as_bytes)
{
    GPUMatmulKey<T> k;
    k.mem_size_A = p.size_A * sizeof(T);
    k.mem_size_B = p.size_B * sizeof(T);
    k.mem_size_C = p.size_C * sizeof(T);
    k.A = (T *)*key_as_bytes;
    *key_as_bytes += k.mem_size_A;
    k.B = (T *)*key_as_bytes;
    *key_as_bytes += k.mem_size_B;
    k.C = (T *)*key_as_bytes;
    *key_as_bytes += k.mem_size_C;
    k.trKey = readGPUTruncateKey<T>(t, key_as_bytes);
    return k;
}

void stdInit(MatmulParams &p, int bw, int scale)
{
    p.bw = bw;
    p.shift = scale;

    p.ld_A = p.K;
    p.ld_B = p.N;
    p.ld_C = p.N;

    p.stride_A = p.M * p.K;
    p.stride_B = p.K * p.N;
    p.stride_C = p.M * p.N;

    p.size_A = p.batchSz * p.M * p.K;
    p.size_B = p.batchSz * p.K * p.N;

    if (p.cIsLowerTriangular)
    {
        assert(p.M == p.N);
        p.size_C = p.batchSz * (p.M * (p.M + 1)) / 2;
    }
    else
    {
        p.size_C = p.batchSz * p.M * p.N;
    }

    p.rowMaj_A = true;
    p.rowMaj_B = true;
    p.rowMaj_C = true;
}

#include "gpu_matmul.cu"
