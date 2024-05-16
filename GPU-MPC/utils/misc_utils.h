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

#include <cstddef>
#include <cassert>
#include <cstdint>
#include <omp.h>
#include <unistd.h>

#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "helper_cuda.h"

#include "gpu_file_utils.h"

#include <numeric>

template <typename T>
__global__ void xorKernel(T *A, T *B, int N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        A[j] ^= B[j];
    }
}

template <typename T>
__device__ inline void gpuMod(T &x, int bw)
{
    if (bw < sizeof(T) * 8)
        x &= ((T(1) << bw) - 1);
}

template <typename T>
__device__ void linearComb(int i, T c, T d_A)
{
    static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    return c * d_A[i];
}

template <typename T, typename... Args>
__device__ T linearComb(int i, T c)
{
    // static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    return c;
}

template <typename T, typename... Args>
__device__ T linearComb(int i, T c, T *A)
{
    // static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    // if(i == 0) printf("Linear comb: %ld, %ld\n", c, A[i]);
    return c * A[i];
}

template <typename T, typename... Args>
__device__ T linearComb(int i, T c, T *A, Args... args)
{
    // if(i == 0) printf("Linear comb: %ld, %ld\n", c, A[i]);
    // static_assert(std::is_arithmetic<T>::value, "Only arithmetic types supported");
    return c * A[i] + linearComb(i, args...);
}

template <typename T, typename... Args>
__global__ void linearCombWrapper(int bw, int N, T *d_O, Args... args)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_O[i] = linearComb(i, args...);
        gpuMod(d_O[i], bw);
        // if(i == 0) printf("Op=%lu\n", d_O[i]);
    }
}

template <typename T, typename... Arguments>
void gpuLinearComb(int bw, int N, T *d_O, Arguments... args)
{
    const int thread_block_size = 128;
    linearCombWrapper<<<(N - 1) / thread_block_size + 1, thread_block_size>>>(bw, N, d_O, args...);
    checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaGetLastError());
}

template <typename T>
void gpuXor(T *d_A, T *d_B, int N, Stats *s)
{
    const int thread_block_size = 128;
    xorKernel<<<(N - 1) / thread_block_size + 1, thread_block_size>>>(d_A, d_B, N);
    checkCudaErrors(cudaDeviceSynchronize());
}

void writeInt(u8 **key_as_bytes, int N)
{
    memcpy(*key_as_bytes, &N, sizeof(int));
    *key_as_bytes += sizeof(int);
}

template <typename T>
__global__ void unmaskKernel(int bw, int N, T *A, T *mask_A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        A[i] = A[i] - mask_A[i];
        gpuMod(A[i], bw);
        A[i] -= ((A[i] >> (bw - 1)) << bw);
    }
}

template <typename T>
void unmaskValues(int bw, int N, T *d_A, T *h_mask_A, Stats *s)
{
    auto d_mask_A = (T *)moveToGPU((u8 *)h_mask_A, N * sizeof(T), s);
    unmaskKernel<<<(N - 1) / 256 + 1, 256>>>(bw, N, d_A, d_mask_A);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_mask_A);
    // return d_A;
}

template <typename T>
__device__ void writePackedOp(u32 *dReluOutput, T dReluBit, int bout, u64 N)
{
    u64 threadId = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(FULL_MASK, /*threadIdx.x*/ threadId < N);
    int laneId = threadIdx.x & 0x1f;
    if (bout == 1)
    {
        int dreluAsInt = static_cast<int>(dReluBit);
        dreluAsInt <<= laneId;
        for (int j = 16; j >= 1; j /= 2)
            dreluAsInt += __shfl_down_sync(mask, dreluAsInt, j, 32);
        if (laneId == 0)
        {
            // if(threadId == 0 || (threadId / 32 == 71088)) printf("bit=%u, %lx\n", dreluAsInt, dReluOutput);
            ((u32 *)dReluOutput)[threadId / 32] = static_cast<u32>(dreluAsInt);
        }
    }
    else if (bout == 2)
    {
        auto dreluAsLongInt = u64(dReluBit);
        dreluAsLongInt <<= (2 * laneId);
        for (int j = 16; j >= 1; j /= 2)
            dreluAsLongInt += __shfl_down_sync(mask, dreluAsLongInt, j, 32);
        if (laneId == 0)
        {
            ((u32 *)dReluOutput)[threadId / 16] = static_cast<u32>(dreluAsLongInt);
            if (N - threadId > 16)
            {
                ((u32 *)dReluOutput)[threadId / 16 + 1] = static_cast<u32>(dreluAsLongInt >> 32);
            }
        }
    }
    else
    {
        ((T *)dReluOutput)[threadId] = dReluBit;
    }
}

template <typename T>
__global__ void modKernel(u64 N, T *d_data, int bw)
{
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < N)
    {
        gpuMod(d_data[i], bw);
    }
}

template <typename TIn, typename TOut>
__global__ void getPackedSharesKernel(u64 N, int party, TIn *d_A, TOut *d_A0, u32 *d_packed_A, int bw)
{
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < N)
    {
        TOut share_A = TOut(d_A[i]);
        if (d_A0)
            share_A = party == SERVER0 ? d_A0[i] : TOut(d_A[i]) - d_A0[i];
        gpuMod(share_A, bw);
        // if(i == 1) printf("%lu: share_A = %u, %u, %d\n", i, share_A, d_A[i], party);
        writePackedOp(d_packed_A, share_A, bw, N);
    }
}

template <typename T>
inline void cpuMod(T &x, int bw)
{
    if (bw < sizeof(T) * 8)
        x &= ((T(1) << bw) - 1);
}

template <typename T>
T cpuArs(T x, int bin, int shift)
{
    x -= (1ULL << (bin - 1));
    cpuMod(x, bin);
    x >>= shift;
    x -= (1ULL << (bin - shift - 1));
    // T msb = (x & (T(1) << (bin - 1))) >> (bin - 1);
    // T signMask = (((T(1) << shift) - msb) << (8 * sizeof(T) - shift));
    // x = (x >> shift) | signMask;
    return x;
}

inline double asFloat(u64 x, int bw, int scale)
{
    return ((i64)cpuArs(x << (64 - bw), 64, 64 - bw)) / (double)(1ULL << scale);
}

void dropOSPageCache()
{
    printf("Dropping the OS page cache\n");
    assert(0 == system("sudo sh -c \"echo  3 > /proc/sys/vm/drop_caches\"") && "could not drop page caches!");
    sync();
}
