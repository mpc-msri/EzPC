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

template <typename T>
struct GPUMulKey {
    u64 szA, szB, szC;
    T *a, *b, *c;
    GPUTruncateKey<T> trKey;
};

template <typename T>
GPUMulKey<T> readGPUMulKey(u8** key_as_bytes, u64 szA, u64 szB, u64 szC, TruncateType t) {
    // printf("Inside mul key, N=%lu, %lu, %lu, %lx\n", szA, szB, szC, *key_as_bytes);
    GPUMulKey<T> k;
    k.szA = szA;
    k.szB = szB;
    k.szC = szC;
    k.a = (T*) *key_as_bytes;
    *key_as_bytes += (szA * sizeof(T));
    k.b = (T*) *key_as_bytes;
    *key_as_bytes += (szB * sizeof(T));
    k.c = (T*) *key_as_bytes;
    *key_as_bytes += (szC * sizeof(T));
    k.trKey = readGPUTruncateKey<T>(t, key_as_bytes);
    return k;
}

#include "gpu_mul.cu"