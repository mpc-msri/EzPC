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

// Utilities and system includes
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include "utils/gpu_data_types.h"
#include "utils/helper_string.h" // helper for shared functions common to CUDA Samples

// CUDA and CUBLAS functions
#include "utils/helper_functions.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_stats.h"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"

#include "gpu_linear_helper.h"
#include "gpu_matmul.h"

const int block_sz = 256;


using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

template <typename T>
using GemmRRR = cutlass::gemm::device::Gemm<T,         // Data-type of A matrix
                                            RowMajor,  // Layout of A matrix
                                            T,         // Data-type of B matrix
                                            RowMajor,  // Layout of B matrix
                                            T,         // Data-type of C matrix
                                            RowMajor>; // Layout of C matrix

template <typename T>
using GemmRCR = cutlass::gemm::device::Gemm<T,           // Data-type of A matrix
                                            RowMajor,    // Layout of A matrix
                                            T,           // Data-type of B matrix
                                            ColumnMajor, // Layout of B matrix
                                            T,           // Data-type of C matrix
                                            RowMajor>;   // Layout of C matrix

template <typename T>
using GemmCRR = cutlass::gemm::device::Gemm<T,           // Data-type of A matrix
                                            ColumnMajor, // Layout of A matrix
                                            T,           // Data-type of B matrix
                                            RowMajor,    // Layout of B matrix
                                            T,           // Data-type of C matrix
                                            RowMajor>;   // Layout of C matrix

template <typename T>
T *cutlassMatmul(MatmulParams p, T *d_A, T *d_B, T *d_C, bool cIsBias = false)
{
    T *d_D = (T *)gpuMalloc(p.M * p.N * sizeof(T));
    cutlass::Status status;
    if (p.rowMaj_A && p.rowMaj_B && p.rowMaj_C)
    {
        GemmRRR<T> gemm_operator;
        typename GemmRRR<T>::Arguments args({p.M, p.N, p.K},                             // Gemm Problem dimensions
                                            {d_A, p.K},                                  // Tensor-ref for source matrix A
                                            {d_B, p.N},                                  // Tensor-ref for source matrix B
                                            {d_C ? d_C : d_D, d_C && cIsBias ? 0 : p.N}, // Tensor-ref for source matrix C
                                            {d_D, p.N},                                  // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                            {T(1), d_C ? T(1) : T(0)});                  // Scalars used in the Epilogue
        status = gemm_operator(args);
    }
    else if (p.rowMaj_A && !p.rowMaj_B && p.rowMaj_C)
    { /* M x K x N */
        GemmRCR<T> gemm_operator;
        typename GemmRCR<T>::Arguments args({p.M, p.N, p.K},            // Gemm Problem dimensions
                                            {d_A, p.K},                 // Tensor-ref for source matrix A
                                            {d_B, p.K},                 // Tensor-ref for source matrix B
                                            {d_C ? d_C : d_D, p.N},     // Tensor-ref for source matrix C
                                            {d_D, p.N},                 // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                            {T(1), d_C ? T(1) : T(0)}); // Scalars used in the Epilogue
        status = gemm_operator(args);
    }
    else if (!p.rowMaj_A && p.rowMaj_B && p.rowMaj_C)
    { /* M x K x N */
        GemmCRR<T> gemm_operator;
        // printf("%d %d %d\n", k.M, k.N, k.K);
        typename GemmCRR<T>::Arguments args({p.M, p.N, p.K},            // Gemm Problem dimensions
                                            {d_A, p.M},                 // Tensor-ref for source matrix A
                                            {d_B, p.N},                 // Tensor-ref for source matrix B
                                            {d_C ? d_C : d_D, p.N},     // Tensor-ref for source matrix C
                                            {d_D, p.N},                 // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                            {T(1), d_C ? T(1) : T(0)}); // Scalars used in the Epilogue
        status = gemm_operator(args);
    }
    else
    {
        assert(false && "no option matches!");
    }
    CUTLASS_CHECK(status);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_D;
}

template <typename T>
T *cutlassMatmulWrapper(MatmulParams p, T *d_A, T *d_B, T *d_C, bool cIsBias = false, bool reduceBw = false)
{
    T *d_O;
    if (p.batchSz == 1)
    {
        d_O = cutlassMatmul(p, d_A, d_B, d_C, cIsBias);
    }
    else
    {
        // assert(!p.cIsLowerTriangular);
        d_O = cutlassBatchedMatmul(p, d_A, d_B, d_C, cIsBias);
    }
    if (p.cIsLowerTriangular)
    {
        assert(!d_C);
        auto d_temp = packLowerTriangularMatrix(p, d_O);
        gpuFree(d_O);
        d_O = d_temp;
    }
    if (reduceBw && p.bw < sizeof(T) * 8)
    {
        modKernel<<<(p.size_C - 1) / 128 + 1, 128>>>(p.size_C, d_O, p.bw);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    return d_O;
}

template <typename T>
T *cutlassBatchedMatmul(MatmulParams p, T *d_A, T *d_B, T *d_C, bool cIsBias = false)
{
    auto d_D = (T *)gpuMalloc(p.batchSz * p.M * p.N * sizeof(T));
    using BatchedGemmRRR = cutlass::gemm::device::GemmBatched<
        T, cutlass::layout::RowMajor,
        T, cutlass::layout::RowMajor,
        T, cutlass::layout::RowMajor>;

    using BatchedGemmRCR = cutlass::gemm::device::GemmBatched<
        T, cutlass::layout::RowMajor,
        T, cutlass::layout::ColumnMajor,
        T, cutlass::layout::RowMajor>;

    assert(p.rowMaj_A && p.rowMaj_C);
    cutlass::Status status;
    if (p.rowMaj_B)
    {
        BatchedGemmRRR gemm_op;
        status = gemm_op({{p.M, p.N, p.K},
                          {d_A, p.ld_A},
                          p.stride_A,
                          {d_B, p.ld_B},
                          p.stride_B,
                          {d_C ? d_C : d_D, d_C && cIsBias ? 0 : p.ld_C}, //
                          cIsBias ? p.N : p.stride_C,
                          {d_D, p.ld_C},
                          p.stride_C,
                          {T(1), d_C ? T(1) : T(0)},
                          p.batchSz});
    }
    else
    {
        // printf("ld_A=%d, ld_B=%d, ld_C=%d, stride_A=%d, stride_B=%d, stride_C=%d, M=%d, N=%d, K=%d, d_C=%ld\n", p.ld_A, p.ld_B, p.ld_C, p.stride_A, p.stride_B, p.stride_C, p.M, p.N, p.K, d_C);
        assert(!d_C || !cIsBias);
        BatchedGemmRCR gemm_op;
        status = gemm_op({{p.M, p.N, p.K},
                          {d_A, p.ld_A}, // 786
                          p.stride_A,
                          {d_B, p.ld_B}, // 786*3
                          p.stride_B,
                          {d_C ? d_C : d_D, p.ld_C}, //
                          p.stride_C,
                          {d_D, p.ld_C},
                          p.stride_C,
                          {T(1), d_C ? T(1) : T(0)},
                          p.batchSz});
    }
    CUTLASS_CHECK(status);
    return d_D;
}

template <typename T>
T *gpuMatmulPlaintext(MatmulParams p, T *d_A, T *d_B, T *d_C, bool cIsBias)
{
    // Need to fix this
    auto d_D = cutlassMatmulWrapper<T>(p, d_A, d_B, d_C, cIsBias, true);
    return d_D;
}

template <typename T>
__global__ void packLowerTriangularKernel(MatmulParams p, T *A, T *packed_A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p.size_C)
    {
        int t = i;
        int imgElems = (p.M * (p.M + 1)) / 2;
        int n = t / imgElems;
        t = t % imgElems;
        // figure out the index of the matrix
        int j = int(floor((-1.0f + __fsqrt_rn(1 + 8.0f * t)) / 2.0f));
        int k = t - ((j * (j + 1)) / 2);
        if (k == j + 1)
        {
            j++;
            k = 0;
        }
        // printf("t=%d, k=%d, j=%d, sqrt=%f, sqrt2=%f\n", t, k, j, __fsqrt_rd(1 + 8.0f * t), __fsqrt_rd(1 + 8.0f * (t+1)));
        assert(k < j + 1);
        // printf("%d <- (%d, %d, %d), %ld, %d\n", i, n, j, k, A[n * imgElems + j * p.N + k], n * imgElems + j * p.N + k);
        packed_A[i] = A[n * p.M * p.N + j * p.N + k];
    }
}

template <typename T>
T *packLowerTriangularMatrix(MatmulParams p, T *d_A)
{
    assert(p.M == p.N);
    // printf("Packing matrix=%d\n", p.size_C);
    auto d_packed_A = (T *)gpuMalloc(p.size_C * sizeof(T));
    packLowerTriangularKernel<<<(p.size_C - 1) / 128 + 1, 128>>>(p, d_A, d_packed_A);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_packed_A;
}

template <typename T>
T *gpuKeygenMatmul(u8 **key_as_bytes, int party, MatmulParams p, T *d_mask_X, T *h_mask_W, T *h_mask_Y, TruncateType t, AESGlobalContext *gaes, bool wIsOnGpu = false, T *d_mask_Z = NULL)
{
    // printf("w is on gpu=%d, X addr=%lx\n", wIsOnGpu, d_mask_X);
    assert(!p.cIsLowerTriangular || !h_mask_Y);
    auto d_mask_W = h_mask_W;
    if (!wIsOnGpu)
        d_mask_W = (T *)moveToGPU((u8 *)h_mask_W, p.size_B * sizeof(T), NULL);
    writeShares<T, T>(key_as_bytes, party, p.size_A, d_mask_X, p.bw);
    writeShares<T, T>(key_as_bytes, party, p.size_B, d_mask_W, p.bw);
    // T *d_mask_Z = NULL;
    if (p.cIsLowerTriangular)
    {
        assert(p.size_C == p.batchSz * (p.M * (p.M + 1)) / 2);
        auto d_Z = gpuMatmulPlaintext(p, d_mask_X, d_mask_W, (T *)NULL, false);
        d_mask_Z = randomGEOnGpu<T>(p.size_C, p.bw);
        // checkCudaErrors(cudaMemset(d_mask_Z, 0, p.size_C * sizeof(T)));
        gpuLinearComb(p.bw, p.size_C, d_Z, T(1), d_Z, T(1), d_mask_Z);
        writeShares<T, T>(key_as_bytes, party, p.size_C, d_Z, p.bw);
        gpuFree(d_Z);
    }
    else
    {
        if (!d_mask_Z)
            d_mask_Z = randomGEOnGpu<T>(p.size_C, p.bw);
        // checkCudaErrors(cudaMemset(d_mask_Z, 0, p.size_C * sizeof(T)));
        auto d_masked_Z = gpuMatmulPlaintext(p, d_mask_X, d_mask_W, d_mask_Z, false);
        writeShares<T, T>(key_as_bytes, party, p.size_C, d_masked_Z, p.bw);
        gpuFree(d_masked_Z);
        if (h_mask_Y)
            gpuAddBias(p.batchSz, p.M, p.N, p.bw, d_mask_Z, h_mask_Y, NULL);
    }
    if (!wIsOnGpu)
        gpuFree(d_mask_W);

    auto d_mask_truncated_Z = genGPUTruncateKey<T, T>(key_as_bytes, party, t, p.bw, p.bw, p.shift, p.size_C, d_mask_Z, gaes);

    if (d_mask_Z != d_mask_truncated_Z)
        gpuFree(d_mask_Z);
    return d_mask_truncated_Z;
}

template <typename T>
void gmwGpuKeygenMatmul(u8 **key_as_bytes, int party, MatmulParams p, T *h_W)
{
    auto d_mask_X = randomGEOnGpu<T>(p.size_A, p.bw);
    auto d_mask_W = (T *)moveToGPU((u8 *)h_W, p.size_B * sizeof(T), NULL);
    // (T*) gpuMalloc(p.size_B * sizeof(T));
    // checkCudaErrors(cudaMemset(d_mask_W, 0, p.size_B * sizeof(T)));
    // randomGEOnGpu<T>(p.size_B, p.bw);

    writeShares<T, T>(key_as_bytes, party, p.size_A, d_mask_X, p.bw);
    writeShares<T, T>(key_as_bytes, party, p.size_B, d_mask_W, p.bw);
    assert(!p.cIsLowerTriangular);
    auto d_Z = gpuMatmulPlaintext(p, d_mask_X, d_mask_W, (T *)NULL, false);
    writeShares<T, T>(key_as_bytes, party, p.size_C, d_Z, p.bw);

    gpuFree(d_mask_X);
    gpuFree(d_mask_W);
    gpuFree(d_Z);
}

template <typename T>
T *gpuMatmulBeaver(MatmulParams p, GPUMatmulKey<T> k, int party, T *d_A, T *d_B, T *d_r0, T *d_r1, T *d_bias, Stats *s)
{
    T *d_C1, *d_C2;
    if (party == SERVER0)
    {
        gpuLinearComb(p.bw, p.size_B, d_r1, T(1), d_B, T(-1), d_r1);
    }
    d_C1 = cutlassMatmulWrapper<T>(p, d_A, d_r1, d_bias, true);
    d_C2 = cutlassMatmulWrapper<T>(p, d_r0, d_B, NULL);
    T *d_C = (T *)moveToGPU((u8 *)k.C, k.mem_size_C, s);
    gpuLinearComb(p.bw, p.size_C, d_C, T(1), d_C, party == SERVER0 ? T(1) : T(-1), d_C1, T(-1), d_C2);
    gpuFree(d_C1);
    gpuFree(d_C2);
    return d_C;
}

template <typename T>
T *gpuMatmul(SigmaPeer *peer, int party, MatmulParams p, GPUMatmulKey<T> &k, T *d_X, T *h_W, T *h_Y, TruncateType t, AESGlobalContext *gaes, Stats *s, bool wIsOnGpu = false, T* d_mask_X = nullptr)
{
    // printf("X=%lx, %lu\n", d_X, k.mem_size_A);
    u64 b0 = peer->bytesSent() + peer->bytesReceived();
    if (!d_mask_X)
        d_mask_X = (T *)moveToGPU((u8 *)k.A, k.mem_size_A, s);
    auto d_W = h_W;
    if (!wIsOnGpu)
        d_W = (T *)moveToGPU((u8 *)h_W, k.mem_size_B, s);
    auto d_mask_W = (T *)moveToGPU((u8 *)k.B, k.mem_size_B, s);
    T *d_Y = NULL;
    // printf("N=%d, batchSz=%d\n", p.N, p.batchSz);
    if (party == SERVER0 && h_Y)
        d_Y = (T *)moveToGPU((u8 *)h_Y, p.batchSz * p.N * sizeof(T), s);

    auto d_Z = gpuMatmulBeaver(p, k, party, d_X, d_W, d_mask_X, d_mask_W, d_Y, s);
    // printf("Finished matmul\n");
    gpuFree(d_mask_X);
    if (!wIsOnGpu)
        gpuFree(d_W);
    gpuFree(d_mask_W);
    if (d_Y)
        gpuFree(d_Y);

    peer->reconstructInPlace(d_Z, p.bw, p.size_C, s);

    auto d_truncatedZ = gpuTruncate<T, T>(p.bw, p.bw, t, k.trKey, p.shift, peer, party, p.size_C, d_Z, gaes, s); //, true);
    if (d_Z != d_truncatedZ)
        gpuFree(d_Z);

    u64 b1 = peer->bytesSent() + peer->bytesReceived();
    s->linear_comm_bytes += (b1 - b0);
    return d_truncatedZ;
}

template <typename T>
T *gpuMatmulGmw(SigmaPeer *peer, int party, MatmulParams p, GPUMatmulKey<T> &k, T *d_X, T *d_mask_X, T *d_W, T *d_mask_W, T *h_Y, Stats *s)
{
    T *d_Y = NULL;
    printf("Adding bias=%lx\n", h_Y);
    if (h_Y)
    {
        printf("here\n");
        d_Y = (T *)moveToGPU((u8 *)h_Y, p.batchSz * p.N * sizeof(T), s);
    }
    auto d_Z = gpuMatmulBeaver(p, k, party, d_X, d_W, d_mask_X, d_mask_W, d_Y, s);
    if (d_Y)
        gpuFree(d_Y);
    return d_Z;
}

template <typename T>
T *gpuMatmulWrapper(MatmulParams p, T *h_A, T *h_B, T *h_C, bool cIsBias)
{
    size_t memSzBias = p.N * sizeof(T);
    size_t memSzC = p.size_C * sizeof(T);
    T *d_A = (T *)moveToGPU((u8 *)h_A, p.size_A * sizeof(T), NULL);
    T *d_B = (T *)moveToGPU((u8 *)h_B, p.size_B * sizeof(T), NULL);
    T *d_C = NULL;
    if (h_C)
        d_C = (T *)moveToGPU((u8 *)h_C, cIsBias ? memSzBias : memSzC, NULL);
    // Need to fix this
    auto d_D = cutlassMatmulWrapper<T>(p, d_A, d_B, d_C, cIsBias, true);
    gpuFree(d_A);
    gpuFree(d_B);
    if (d_C)
        gpuFree(d_C);
    T *h_D = (T *)moveToCPU((u8 *)d_D, memSzC, NULL);
    gpuFree(d_D);
    return h_D;
}
