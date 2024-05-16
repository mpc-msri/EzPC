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
#include "fss/gpu_sstab.h"
#include "gpu_dcf_templates.h"

namespace dcf
{
    template <typename T, int E, dcfPrologue pr, dcfEpilogue ep>
    __global__ void lookupSSTable(int party, int bin, int N,
                                  T *in, u8 *ss, u32 *out)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < N)
        {
            auto x = u64(in[tid]);
            gpuMod(x, bin);
            int tabSz = (1ULL << (max(0, bin - 3))); // number of bytes occupied by the table
            // printf("Table start idx %d=%d\n", tid, tabSz * tid);
            u8 *localSS = &ss[tid * tabSz];
            u64 x1[E];
            // populate the input
            pr(party, bin, N, x, x1);
            u64 o[E];
            for (int e = 0; e < E; e++)
            {
                gpuMod(x1[e], bin);
                // printf("X[%d]=%ld, %ld, %d, %d\n", e, x, x1[e], int(lookup(localSS, x1[e])), int(localSS[x1[e] / 8]));
                o[e] = lookup(localSS, x1[e]);
            }
            ep(party, bin, 1, N, x, o, out, 0);
        }
    }

    template <typename T, int E, dcfPrologue pr, dcfEpilogue ep>
    u32 *gpuLookupSSTable(GPUSSTabKey &k, int party, T *d_in, Stats *s, std::vector<u32 *> *h_masks = NULL)
    {
        auto d_out = moveMasks(k.memSzOut, h_masks, s);
        // printf("Bin=%d, Memsz=%ld\n", k.bin, k.memSzSS);
        auto d_ss = (u8 *)moveToGPU((u8 *)k.ss, k.memSzSS, s);
        dcf::lookupSSTable<T, E, pr, ep><<<(k.N - 1) / 128 + 1, 128>>>(party, k.bin, k.N, d_in, d_ss, d_out);
        checkCudaErrors(cudaDeviceSynchronize());
        gpuFree(d_ss);
        return d_out;
    }
}