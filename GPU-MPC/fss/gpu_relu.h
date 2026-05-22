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

#include "gpu_select.h"
#include "gpu_dpf.h"

// using GPUDReluKey = GPUMaskedDCFKey;
using u32 = uint32_t;

struct GPUDReluKey
{
    GPUDPFKey dpfKey;
    u32 *mask;
};

template <typename T>
struct GPUReluKey
{
    int bin, bout, numRelus;
    GPUDReluKey dreluKey;
    GPUSelectKey<T> selectKey;
};

GPUDReluKey readGPUDReluKey(uint8_t **key_as_bytes)
{
    GPUDReluKey k;
    k.dpfKey = readGPUDPFKey(key_as_bytes);
    int N = k.dpfKey.M;
    k.mask = (uint32_t *)*key_as_bytes;
    // number of 32-bit integers * sizeof(int)
    // only works for bout = 1
    *key_as_bytes += ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

// const auto readGPUDReluWithDCFKey = readGPUMaskedDCFKey;

template <typename T>
GPUReluKey<T> readReluKey(uint8_t **key_as_bytes)
{
    GPUReluKey<T> k;
    memcpy(&k, *key_as_bytes, 3 * sizeof(int));
    *key_as_bytes += 3 * sizeof(int);

    k.dreluKey = readGPUDReluKey(key_as_bytes);
    k.selectKey = readGPUSelectKey<T>(key_as_bytes, k.numRelus);
    return k;
}

#include "gpu_relu.cu"