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

#include "utils/helper_cuda.h"

template <typename TIn, typename TOut>
using localTrFunc = TOut (*)(int party, int bin, int shift, TIn x);

template <typename TIn, typename TOut>
__device__ TOut lrs(int party, int bin, int shift, TIn x)
{
    return TOut(x >> shift);
}

template <typename TIn, typename TOut>
__device__ TOut ars(int party, int bin, int shift, TIn x)
{
    x += TIn(1ULL << (bin - 1));
    gpuMod(x, bin);
    auto trX = TOut((x >> shift) - (1ULL << (bin - shift - 1)));
    return trX;
}

template <typename TIn, typename TOut, localTrFunc<TIn, TOut> tf>
__global__ void localTrKernel(int party, int bin, int shift, int N, TIn *x, TOut *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        y[i] = (TOut)tf(party, bin, shift, x[i]);
    }
}

template <typename TIn, typename TOut, localTrFunc<TIn, TOut> tf>
TOut *gpuLocalTr(int party, int bin, int shift, int N, TIn *d_I, bool inPlace = false)
{
    assert(bin >= shift);
    TOut* d_O = inPlace ? (TOut*) d_I : (TOut*) gpuMalloc(N * sizeof(TOut));
    localTrKernel<TIn, TOut, tf><<<(N - 1) / 128 + 1, 128>>>(party, bin, shift, N, d_I, d_O);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_O;
}