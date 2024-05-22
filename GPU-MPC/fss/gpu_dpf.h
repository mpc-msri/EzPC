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
#include "gpu_aes_shm.h"
#include "gpu_sstab.h"

// using u32 = u32;

struct GPUDPFTreeKey
{
    int bin, N, evalAll;
    AESBlock *scw;
    AESBlock *l0, *l1;
    u32 *tR;
    u64 szScw, memSzScw, memSzL, memSzT, memSzOut;
};

struct GPUDPFKey
{
    // if bin <= 7, populate ss, else ss = NULL
    int bin, M, B;
    u64 memSzOut;
    GPUDPFTreeKey *dpfTreeKey;
    GPUSSTabKey ssKey;
};

GPUDPFTreeKey readGPUDPFTreeKey(u8 **key_as_bytes)
{
    GPUDPFTreeKey k;

    std::memcpy((char *)&k, *key_as_bytes, 3 * sizeof(int));
    *key_as_bytes += 3 * sizeof(int);

    k.szScw = k.N * (k.bin - LOG_AES_BLOCK_LEN);
    k.memSzScw = k.szScw * sizeof(AESBlock);
    k.scw = (AESBlock *)*key_as_bytes;

    *key_as_bytes += k.memSzScw;
    k.memSzL = k.N * sizeof(AESBlock);
    k.l0 = (AESBlock *)*key_as_bytes;
    *key_as_bytes += k.memSzL;
    k.l1 = (AESBlock *)*key_as_bytes;
    *key_as_bytes += k.memSzL;

    if (k.evalAll)
        k.memSzT = k.N * sizeof(u32);
    else
        k.memSzT = ((k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (k.bin - LOG_AES_BLOCK_LEN);
    k.tR = (u32 *)*key_as_bytes;
    *key_as_bytes += k.memSzT;
    k.memSzOut = ((k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

GPUDPFKey readGPUDPFKey(u8 **key_as_bytes)
{
    GPUDPFKey k;
    k.bin = *((int *)*key_as_bytes);
    if (k.bin <= 7)
    {
        k.ssKey = readGPUSSTabKey(key_as_bytes);
        k.M = k.ssKey.N;
        k.B = 1;
        k.memSzOut = k.ssKey.memSzOut;
    }
    else
    {
        memcpy(&k, *key_as_bytes, 3 * sizeof(int));
        *key_as_bytes += (3 * sizeof(int));

        k.dpfTreeKey = new GPUDPFTreeKey[k.B];
        k.memSzOut = 0;
        for (int b = 0; b < k.B; b++)
        {
            k.dpfTreeKey[b] = readGPUDPFTreeKey(key_as_bytes);
            k.memSzOut += k.dpfTreeKey[b].memSzOut;
        }
    }
    return k;
}

const auto readGPUDcfKey = readGPUDPFKey;

#include "gpu_dpf.cu"