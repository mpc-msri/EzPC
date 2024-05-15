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