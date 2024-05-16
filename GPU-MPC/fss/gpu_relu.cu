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

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include "utils/gpu_data_types.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_mem.h"

#include "gpu_select.h"
#include "gpu_and.h"
#include "gpu_relu.h"

using namespace std;

template <typename T>
__global__ void keyGenDReluKernel(int party, int bin, int N, T *rinArr, T *rout, T *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        auto mRin = -rinArr[i];
        gpuMod(mRin, bin);
        auto correctionTerm = gpuMsb(mRin, bin) ^ 1 ^ rout[i];
        assert(correctionTerm == 0 || correctionTerm == 1);
        out[i] = correctionTerm;
    }
}

// drelu(x-p) where x-p is guaranteed to be small
template <typename T>
void genDcfKeyForDRelu(uint8_t **key_as_bytes, int party, int bin, int N, T *d_rin, AESGlobalContext *gaes)
{
    auto d_mRin = (T *)gpuMalloc(N * sizeof(T));
    gpuLinearComb(bin - 1, N, d_mRin, -T(1), d_rin);
    gpuKeyGenDCF(key_as_bytes, party, bin - 1, N, d_mRin, gaes);
    gpuFree(d_mRin);
}

// need to check this
// drelu mask is used as input mask for the next set of protocols
// do we need something better than GPUGroupElement?
template <typename T, u64 p, u64 q>
T *gpuKeyGenIc(uint8_t **key_as_bytes, int party, int bin, int N, T *d_rin, bool genDcfKey, AESGlobalContext *gaes)
{
    if (genDcfKey)
        genDcfKeyForDRelu(key_as_bytes, party, bin, N, d_rin, gaes);
    auto d_icMask = randomGEOnGpu<T>(N, 1);
    // checkCudaErrors(cudaMemset(d_icMask, 0, N * sizeof(T)));
    writeShares<T, T>(key_as_bytes, party, N, d_icMask, 1);
    return d_icMask;
}

template <typename T>
T *gpuKeyGenDRelu(uint8_t **key_as_bytes, int party, int bin, int N, T *d_rin, AESGlobalContext *gaes)
{
    genDcfKeyForDRelu(key_as_bytes, party, bin, N, d_rin, gaes);
    auto d_dReluMask = randomGEOnGpu<T>(N, 1);
    // checkCudaErrors(cudaMemset(d_dReluMask, 0, N * sizeof(T)));
    auto d_modifiedMask = (T *)gpuMalloc(N * sizeof(T));
    keyGenDReluKernel<<<(N - 1) / 128 + 1, 128>>>(party, bin, N, d_rin, d_dReluMask, d_modifiedMask);
    writeShares<T, T>(key_as_bytes, party, N, d_modifiedMask, 1);
    gpuFree(d_modifiedMask);
    return d_dReluMask;
}

// relu(x-p1) + p2 where x-p1 is guaranteed to be small
template <typename TIn, typename TOut, u64 p, u64 q, bool flipDRelu>
TOut *gpuGenReluKey(uint8_t **key_as_bytes, int party, int bin, int bout, int N, TIn *d_inputMask, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, bout);
    writeInt(key_as_bytes, N);
    // printf("Writing bout=%d, N=%d\n", bout, N);
    auto d_dreluMask = gpuKeyGenDRelu(key_as_bytes, party, bin, N, d_inputMask, gaes);
    auto d_reluMask = gpuKeyGenSelect<TIn, TOut>(key_as_bytes, party, N, d_inputMask, d_dreluMask, bout);
    return d_reluMask;
}

// Relu(x-p) + q, where x-p is guaranteed to be small
template <typename TIn, typename TOut, u64 p, u64 q, bool flipDRelu>
TOut *gpuRelu(SigmaPeer *peer, int party, GPUReluKey<TOut> &k, TIn *d_I, AESGlobalContext *gaes, Stats *s)
{
    auto &dreluKey = k.dreluKey;
    std::vector<u32 *> h_mask({dreluKey.mask});
    auto d_drelu = gpuDcf<TIn, 1, dReluPrologue<p>, dReluEpilogue<p, flipDRelu>>(dreluKey.dpfKey, party, d_I, gaes, s, &h_mask);
    peer->reconstructInPlace(d_drelu, 1, k.numRelus, s);
    auto d_relu = gpuSelect<TIn, TOut, p, q>(peer, party, k.bout, k.selectKey, (u32 *)d_drelu, d_I, s);
    return d_relu;
}
