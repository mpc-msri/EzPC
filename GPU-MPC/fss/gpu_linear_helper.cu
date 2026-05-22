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

#include <cstdlib>
#include <cassert>
#include <stdio.h>

#include "utils/gpu_data_types.h"
#include "utils/helper_cuda.h"
#include "utils/misc_utils.h"
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>
#include "cutlass/reduction/device/tensor_reduce.h"


template <typename T>
inline cutlass::TensorRef<T, cutlass::layout::TensorNHWC> getTensorRef(
    T *ptr, int n, int h, int w, int c)
{
    return cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(
        ptr,
        cutlass::layout::TensorNHWC::packed({n, h, w, c}));
}

template <typename T>
inline cutlass::TensorRef<T, cutlass::layout::TensorNHWC> getTensorRefBias(
    T *ptr)
{

    return cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(
        ptr,
        cutlass::layout::TensorNHWC::Stride(0));
}



template <typename T>
__global__ void addBiasKernel(int batchSz, int M, int N, int bw, T *A, T *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batchSz * M * N)
    {
        int s = i / (M * N);
        int t = i % (M * N);
        int bIdx = t % N;
        A[i] += b[s * N + bIdx];
        gpuMod(A[i], bw);
    }
}

template <typename T>
void gpuAddBias(int batchSz, int M, int N, int bw, T *d_A, T *h_b, Stats *s)
{
    // assert(bw == sizeof(T) * 8);
    size_t memSizeB = batchSz * N * sizeof(T);
    auto d_b = (T *)moveToGPU((uint8_t *)h_b, memSizeB, s);
    addBiasKernel<<<(batchSz * M * N - 1) / 128 + 1, 128>>>(batchSz, M, N, bw, d_A, d_b);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_b);
}

template <typename T>
void gpuAddBiasWrapper(int batchSz, int M, int N, int bw, T *h_A, T *h_b)
{ // check this once
    size_t memSizeA = batchSz * M * N * sizeof(T);
    auto d_A = (T *)moveToGPU((uint8_t *)h_A, memSizeA, NULL);
    gpuAddBias(batchSz, M, N, bw, d_A, h_b, NULL);
    moveIntoCPUMem((uint8_t *)h_A, (uint8_t *)d_A, memSizeA, NULL);
    gpuFree(d_A);
}

// bias is an M vector
template <typename T>
T *getBiasGrad(int N, int M, int bw, T *d_A)
{
    assert(bw == sizeof(T) * 8);
    T *d_b = (T *)gpuMalloc(M * sizeof(T));
    const int kV = 1;
    using TensorReduction = cutlass::reduction::device::TensorReduction<
        T,                           // output
        T,                           // source
        cutlass::layout::TensorNHWC, // Layout
        cutlass::plus<T>,            // Functor
        kV,                          // kV
        T                            // ElementCompute
        >;

    auto t_A = getTensorRef(d_A, 1, 1, N, M);
    auto t_b = getTensorRef(d_b, 1, 1, 1, M);

    TensorReduction reduction(/*t_A.extent()*/ {1, 1, N, M}, 2);

    uint8_t *workspace = gpuMalloc(reduction.workspace_size());

    cutlass::Status status = reduction.reduce(
        t_b /*.device_ref()*/, // dst_tensor
        t_A /*.device_ref()*/, // src_tensor
        workspace,             // device_workspace
        T(0)                   // reduction_identity
    );
    CUTLASS_CHECK(status);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(workspace);

    return d_b;
}

template <typename T>
T *getBiasGradWrapper(int N, int M, int bw, T *h_A)
{
    size_t memSizeA = N * M * sizeof(T);
    auto d_A = (T *)moveToGPU((uint8_t *)h_A, memSizeA, NULL);
    auto d_b = getBiasGrad(N, M, bw, d_A);
    size_t memSizeB = M * sizeof(T);
    auto h_b = (T *)moveToCPU((uint8_t *)d_b, memSizeB, NULL);
    gpuFree(d_A);
    gpuFree(d_b);
    return h_b;
}

template <typename T>
__global__ void leftShiftAndAddKernel(T *A, T *B, T *C, int shift, T alpha, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = (A[i] << shift) + alpha * B[i];
        // if(i == 1) printf("%u %u %u %u %d\n", A[i], B[i], alpha, C[i], shift);
    }
}

template <typename T>
void gpuLeftShiftAndAdd(int N, T *d_A, T *d_B, T *d_C, int shift, T alpha)
{
    assert(shift < sizeof(T) * 64);
    leftShiftAndAddKernel<<<(N - 1) / 256 + 1, 256>>>(d_A, d_B, d_C, shift, alpha, N);
    checkCudaErrors(cudaDeviceSynchronize());
}

template <typename T>
void gpuLeftShiftAndAddWrapper(int N, T *d_A, T *h_B, T *d_C, int shift, T alpha)
{
    size_t memSize = N * sizeof(T);
    // auto d_A = (T*) moveToGPU((uint8_t*) h_A, memSize, NULL);
    auto d_B = (T *)moveToGPU((uint8_t *)h_B, memSize, NULL);
    gpuLeftShiftAndAdd(N, d_A, d_B, d_A, shift, alpha);
    // moveIntoCPUMem((uint8_t*) h_C, (uint8_t*) d_A, memSize, NULL);
    gpuFree(d_A);
    gpuFree(d_B);
}
