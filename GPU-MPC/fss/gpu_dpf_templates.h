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

#include "utils/misc_utils.h"
#include "utils/gpu_data_types.h"

#include "gpu_fss_helper.h"

typedef void (*dpfPrologue)(int party, int bin, int N,
                            u64 x,
                            u64 *o);
typedef void (*dpfEpilogue)(int party, int bin, int N,
                            u64 x,
                            u8 *o, u32 *out, u64 oStride);

__device__ void idPrologue(int party, int bin, int N,
                           u64 x,
                           u64 *o)
{
    // printf("Inside truncate=%ld\n", x);
    o[0] = x;
    // gpuMod(o[0], bin);
}

template <u64 p>
__device__ void dReluPrologue(int party, int bin, int N,
                              u64 x,
                              u64 *o)
{
    o[0] = p - x - 1;
}

template <u64 p, u64 q>
__device__ void geluPrologue(int party, int bin, int N,
                             u64 x,
                             u64 *o)
{
    o[0] = -x - 1;
    o[1] = p - x - 1;
    o[2] = q - x - 1;
}

template <u64 p, bool flip>
__device__ void dReluEpilogue(int party, int bin, int N,
                              u64 x,
                              u8 *o, u32 *out, u64 oStride)
{
    auto o1 = u64(*o);
    auto mask = getVCW(1, out, N, 0);
    o1 ^= mask;
    if (party == SERVER1)
        o1 ^= (gpuMsb(x - p, bin + 1) ^ u64(flip));
    // gpuMod(o, 1);
    //  ^ o ^ mask;
    // printf("Epilogue: %ld, %ld, %ld\n", mask, u64(*o), gpuMsb(x, bin + 1));
    writePackedOp(out, o1, 1, N);
}

template <u64 p, u64 q>
__device__ void geluEpilogue(int party, int bin, int N,
                             u64 x,
                             u8 *o, u32 *out, u64 oStride)
{
    auto o1 = u64(o[0]);
    auto dReluMask = getVCW(1, out, N, 0);
    o1 ^= dReluMask;
    if (party == SERVER1)
        o1 ^= gpuMsb(x, bin + 1);
    // gpuMod(o, 1);
    //  ^ o ^ mask;
    // printf("Epilogue: %lu, %lu\n", dReluMask, o1);
    writePackedOp(out, o1, 1, N);
    // writeVCW(1, out, o1, 0, oStride);

    o1 = u64(o[1]);
    auto o2 = u64(o[2]);
    auto icMask = getVCW(1, out + oStride, N, 0);
    o1 ^= (o2 ^ icMask);
    if (party == SERVER1)
    {
        auto xp = x - p;
        // gpuMod(xp, bin + 1);
        auto xq = x - q;
        // gpuMod(xq, bin + 1);
        o1 ^= (gpuMsb(xp, bin + 1) ^ gpuMsb(xq, bin + 1));
    }
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if(i == 0) printf("Epilogue %d: %ld, %ld\n", i, mask, o, gpuMsb(x, bin + 1));
    // gpuMod(o, 1);
    //  ^ o ^ mask;
    //
    // writeVCW(1, out, o1, 1, N);
    writePackedOp(out + oStride, o1, 1, N);
    // printf("icMask=%lu, ic=%lu\n", icMask, o1);

}

__device__ void maskEpilogue(int party, int bin, int N,
                             u64 x,
                             u8 *o, u32 *out, u64 oStride)
{
    auto o1 = u64(*o);
    auto mask = getVCW(1, out, N, 0);
    // printf("Mask: %ld, output: %ld\n", mask, o);
    o1 = o1 ^ mask;
    writePackedOp(out, o1, 1, N);
}

__device__ void idEpilogue(int party, int bin, int N,
                           u64 x,
                           u8 *o, u32 *out, u64 oStride)
{

    writePackedOp(out, u64(*o), 1, N);
}