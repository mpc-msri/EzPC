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
#include "utils/misc_utils.h"


__host__ __device__ void printAESBlock(AESBlock *b)
{
    auto bAsInt = (u8 *)b;
    for (int i = 15; i >= 0; i--)
        printf("%02x", bAsInt[i]);
    printf("\n");
}

template <typename T>
__device__ inline u8 lsb(T b)
{
    return u8(b) & 1;
}

template <typename T>
__device__ inline T gpuMsb(T x, int bin)
{
    return ((x >> (bin - 1)) & T(1));
}

__device__ u64 getVCW(int bout, u32 *vcw, int num_dcfs, int i)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (bout == 1 || bout == 2)
    {
        int intsPerLevel = (bout * num_dcfs - 1) / PACKING_SIZE + 1;
        int payloadPerIntMask = (PACKING_SIZE / bout) - 1;
        auto vAsInt = ((u32 *)vcw)[intsPerLevel * i + ((bout * threadId) / PACKING_SIZE)];
        vAsInt >>= (bout * (threadIdx.x & payloadPerIntMask));
        u64 v = static_cast<u64>(vAsInt);
        gpuMod(v, bout);
        return v;
    }
    else
    {
        return ((u64 *)vcw)[i * num_dcfs + threadId];
    }
}

__device__ void writeVCW(int bout, u32 *vcwArr, u64 vcw, int i, int N)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(FULL_MASK, threadId < N);
    int laneId = threadIdx.x & 0x1f;
    gpuMod(vcw, bout);
    if (bout == 1)
    {
        int intsPerLevel = (bout * N - 1) / PACKING_SIZE + 1;
        int vcwAsInt = static_cast<int>(vcw);
        vcwAsInt <<= laneId;
        for (int j = 16; j >= 1; j /= 2)
            vcwAsInt += __shfl_down_sync(mask, vcwAsInt, j, 32);
        if (laneId == 0)
        {
            // printf("Writing to vcw[%d]=%u\n", i * intsPerLevel + (threadId / PACKING_SIZE), u32(vcwAsInt));
            ((u32 *)vcwArr)[i * intsPerLevel + (threadId / PACKING_SIZE)] = static_cast<u32>(vcwAsInt);
        }
    }
    else if (bout == 2)
    {
        // assert(0 && "bout = 2 not supported yet!");
        // number of 32-bit ints per level even though we're storing 64-bit integers
        int intsPerLevel = (bout * N - 1) / PACKING_SIZE + 1;
        vcw <<= 2 * laneId;
        for (int j = 16; j >= 1; j /= 2)
            vcw += __shfl_down_sync(mask, vcw, j, 32);
        if (laneId == 0)
        {
            // thread 0 in each warp will write two integers
            int localIdx = 2 * (threadId / PACKING_SIZE);
            int idx = i * intsPerLevel + localIdx;
            ((u32 *)vcwArr)[idx] = static_cast<u32>(vcw);
            if (localIdx + 1 < intsPerLevel)
                ((u32 *)vcwArr)[idx + 1] = static_cast<u32>(vcw >> 32);
            // else printf("not writing one integer\n");
            //     vcwArr[i * longIntsPerLevel + (threadId / PACKING_SIZE)] = vcw;
        }
    }
    else
    {
        ((u64 *)vcwArr)[i * N + threadId] = vcw;
    }
}

u32 *moveMasks(u64 memSz, std::vector<u32 *> *h_masks, Stats *s)
{
    // assert(h_masks);
    u32 *d_out = NULL;
    if (h_masks)
    {
        d_out = (u32 *)gpuMalloc(h_masks->size() * memSz);
        auto d_outTemp = d_out;
        assert(memSz % sizeof(u32) == 0);
        int numInts = memSz / sizeof(u32);
        // printf("numInts=%d\n", numInts);
        for (int i = 0; i < h_masks->size(); i++)
        {
            // printf("Masks=%u\n", (*h_masks)[i][0]);
            moveIntoGPUMem((u8 *)d_outTemp, (u8 *)(*h_masks)[i], memSz, s);
            d_outTemp += numInts;
        }
    }
    else
    {
        d_out = (u32 *)gpuMalloc(memSz);
    }
    return d_out;
}
