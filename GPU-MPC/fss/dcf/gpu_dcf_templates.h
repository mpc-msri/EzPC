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
#include "utils/gpu_stats.h"
#include "utils/gpu_mem.h"

#include "fss/gpu_fss_helper.h"

#include <vector>

// using namespace std;
namespace dcf
{

    typedef void (*dcfPrologue)(int party, int bin, int N,
                                u64 x,
                                u64 *o);
    typedef void (*dcfEpilogue)(int party, int bin, int bout, int N,
                                u64 x,
                                u64 *o_l, u32 *out_g, u64 oStride);

    __device__ void idPrologue(int party, int bin, int N,
                               u64 x,
                               u64 *o)
    {
        o[0] = x;
    }

    __device__ void idEpilogue(int party, int bin, int bout, int N,
                               u64 x,
                               u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = u64(*o_l);
        writePackedOp(out_g, o1, bout, N);
    }

    __device__ void maskEpilogue(int party, int bin, int bout, int N,
                                 u64 x,
                                 u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = u64(*o_l);
        auto mask = getVCW(bout, out_g, N, 0);
        // printf("Mask: %ld, output: %ld\n", mask, o);
        o1 = o1 + mask;
        gpuMod(o1, bout);
        writePackedOp(out_g, o1, bout, N);
    }

    __device__ void dReluPrologue(int party, int bin, int N,
                                  u64 x,
                                  u64 *o)
    {
        o[0] = x;
        o[1] = (x + (1ULL << (bin - 1)));
    }

    template <bool returnXLtRin>
    __device__ void dReluEpilogue(int party, int bin, int bout, int N,
                                  u64 x,
                                  u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = o_l[0];
        auto o2 = o_l[1];
        auto mask = getVCW(bout, out_g, N, 0);
        auto o = o2 - o1 + mask;
        // printf("o1=%lu, o2=%lu, mask=%lu, o=%lu\n", o1, o2, mask, o);
        if (party == SERVER1)
        {
            auto x2 = (x + (1ULL << (bin - 1)));
            gpuMod(x2, bin);
            o += (x2 >= (1ULL << (bin - 1)));
        }
        gpuMod(o, bout);
        writePackedOp(out_g, o, bout, N);
        // writeVCW(bout, out_g, o, 0, N);
        if (returnXLtRin)
        {
            o1 += getVCW(bout, out_g + oStride, N, 0);
            gpuMod(o1, bout);
            writePackedOp(out_g + oStride, o1, bout, N);
            // writeVCW(bout, out_g, o1, 1, N);
        }
    }

}