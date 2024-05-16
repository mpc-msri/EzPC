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

#include <llama/api.h>

#include <cutlass/reduction/device/tensor_reduce.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/vector.h>
#include <cutlass/coord.h>

#include "utils/helper_cutlass.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_random.h"

#include "gpu_layernorm.h"
#include "gpu_maxpool.h"

typedef u64 (*pFunc)(int party, int N, int i, u64 x, u8 *bytes);

template <typename T>
T *gpuSum(int bw, int M, int N, T *d_X)
{
    // printf("Inside gpuSum=%d, %d\n", M, N);
    // auto h_X = (T*) moveToCPU((u8*) d_X, 4 * sizeof(T), NULL);
    // printf("%ld, %ld, %ld, %ld\n", h_X[0], h_X[1], h_X[2], h_X[3]);
    assert(bw <= sizeof(T) * 8);
    T *d_Y = (T *)gpuMalloc(M * sizeof(T));
    // const int kV = 1;
    using TensorReduction = cutlass::reduction::device::TensorReductionAffineContiguous<
        2, 1, T, T, cutlass::plus<T>>;

    auto t_X = cutlass::TensorRef<T, cutlass::layout::RowMajor>(
        d_X,
        cutlass::layout::RowMajor::packed({M, N}));
    auto t_Y = cutlass::TensorRef<T, cutlass::layout::RowMajor>(
        d_Y,
        cutlass::layout::RowMajor::packed({M, 1}));

    TensorReduction reduction(cutlass::Coord<2>({M, N}));

    uint8_t *workspace = gpuMalloc(reduction.workspace_size());
    i64 dstStride = 1;
    i64 srcStride = i64(N);
    cutlass::Status status = reduction.reduce(d_Y, &dstStride, d_X, &srcStride, workspace, T(0));
    CUTLASS_CHECK(status);
    if (bw < 8 * sizeof(T))
        modKernel<<<(M - 1) / 128 + 1, 128>>>(M, d_Y, bw);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(workspace);
    // auto h_Y = (T*) moveToCPU((u8*) d_Y, 1 * sizeof(T), NULL);
    // printf("%ld\n", h_Y[0]);
    return d_Y;
}

template <typename T>
__device__ u64 squareKeygen(int party, int N, int i, u64 x, u8 *bytes)
{
    u64 r = u64(((T *)bytes)[i]);
    // printf("Inside sq keygen %d: a=%ld, r=%ld\n", i, x, r);
    return x * x + r;
}

template <typename T>
__device__ u64 square(int party, int N, int i, u64 x, u8 *bytes)
{
    u64 a = u64(((T *)bytes)[i]);
    u64 c = u64(((T *)bytes)[N + i]);
    // printf("Inside sq %d: x=%ld, a=%ld, c=%ld\n", i, x, a, c);
    return (party == SERVER1) * x * x - 2 * a * x + c;
}

template <typename T, pFunc p>
__global__ void applyPointFunc(int party, int bw, int N, T *d_X, T *d_O, u8 *bytes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        u64 x = u64(d_X[i]);
        auto o = (T)p(party, N, i, x, bytes);
        gpuMod(o, bw);
        d_O[i] = o;
        // printf("Inside point func %d: %ld\n", i, d_O[i]);
    }
}

template <typename T, pFunc p>
T *pointFunc(int party, int bw, int N, T *d_X, u8 *bytes)
{
    T *d_O = (T *)gpuMalloc(N * sizeof(T));
    applyPointFunc<T, p><<<(N - 1) / 128 + 1, 128>>>(party, bw, N, d_X, d_O, bytes);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_O;
}

template <typename T>
T *gpuKeygenLayerNorm(u8 **key_as_bytes, int party, AvgPoolParams p, T *d_mask_A, T *d_mask_B, T *d_mask_X, AESGlobalContext *gaes, bool computeMu = true)
{
    // return d_mask_X;
    assert(p.N == 1);
    int inSz = getInSz(p);
    int mSz = getMSz(p);
    T *d_mask_xMMu = d_mask_X;
    if (computeMu)
    {
        auto d_sumMask = gpuSum(p.bw, p.imgH, p.imgW, d_mask_X);
        // return d_sumMask;
        T *d_mask_mu;
        // printf("Bw=%d\n", p.bw);
        if ((p.imgW & (p.imgW - 1)) == 0)
        {
            // printf("######## Shift=%d\n", int(log2(p.imgW)));
            d_mask_mu = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, p.bw, p.bw, int(log2(p.imgW)), mSz, d_sumMask, gaes);
        }
        else
        {
            gpuLinearComb(p.bw, mSz, d_sumMask, T((1LL << p.scale) / (double)p.imgW), d_sumMask);
            d_mask_mu = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrFloor, p.bw, p.bw, p.scale, mSz, d_sumMask, gaes);
        }
        // return d_mask_mu;
        // printf("boo\n");
        gpuFree(d_sumMask);
        d_mask_xMMu = windowFunc<T, xPlusM<u64(1), u64(-1)>>(party, p, d_mask_X, d_mask_mu);
        // return d_mask_xMMu;
    }
    // printf("boo\n");
    auto d_mask_sq = randomGEOnGpu<T>(inSz, p.bw);
    // checkCudaErrors(cudaMemset(d_mask_sq, 0, inSz * sizeof(T)));
    auto d_mask_sqModified = pointFunc<T, squareKeygen<T>>(party, p.bw, inSz, d_mask_xMMu, (u8 *)d_mask_sq);
    writeShares<T, T>(key_as_bytes, party, inSz, d_mask_xMMu, p.bw);
    writeShares<T, T>(key_as_bytes, party, inSz, d_mask_sqModified, p.bw);
    gpuFree(d_mask_sqModified);

    // printf("boo##\n");
    auto d_mask_sumSq = gpuSum(p.bw, p.imgH, p.imgW, d_mask_sq);
    gpuFree(d_mask_sq);
    // return d_mask_sumSq;

    auto h_mask_sumSq = (T *)moveToCPU((u8 *)d_mask_sumSq, p.imgH * sizeof(T), NULL);
    Rsqrt(p.imgH, h_mask_sumSq, h_mask_sumSq, p.imgW, p.scale, "LayerNorm::");
    // F2BF16(mSz, h_mask_sumSq, h_mask_sumSq, "Rsqrt::");
    moveIntoGPUMem((u8 *)d_mask_sumSq, (u8 *)h_mask_sumSq, mSz * sizeof(T), NULL);
    // printf("boo\n");
    // can potentially make this u16 once we move f2bf16 to the gpu
    // auto d_mask_invSqrt = gpuKeyGenLUT<T, T>(key_as_bytes, party, 13, p.bw, mSz, d_mask_sumSq, gaes);
    // gpuFree(d_mask_sumSq);
    // (x - mu)*var
    // printf("boo\n");
    auto d_mask_invSqrt = d_mask_sumSq;
    // return d_mask_invSqrt;
    auto d_mask_normX = keygenWindowMul(key_as_bytes, party, p, d_mask_xMMu, d_mask_invSqrt, TruncateType::TrWithSlack, gaes);
    gpuFree(d_mask_invSqrt);
    // return d_mask_normX;
    // doesn't matter even if we change things since we're copying things anyway
    // p.FH = p.imgH;
    // p.FW = 1;
    // p.strideH = p.imgH;
    // p.strideW = 1;
    // initPoolParams(p);
    // printf("boo\n");
    auto p2 = transposeWindow(p);
    // printf("Prefinal boo\n");
    auto d_mask_layerNorm = keygenWindowMul(key_as_bytes, party, p2, d_mask_normX, d_mask_A, TruncateType::TrWithSlack, gaes, d_mask_B);
    // printf("Final boo\n");
    gpuFree(d_mask_normX);
    return d_mask_layerNorm;
    // no truncate?
}

template <typename T>
T *gpuLayerNorm(SigmaPeer *peer, int party, AvgPoolParams p, GPULayerNormKey<T> k, T *d_A, T *d_B, T *d_X, std::vector<GroupElement> *invSqrtTab, AESGlobalContext *gaes, Stats *s, bool computeMu = true)
{
    assert(8 * sizeof(T) == 64);
    int inSz = getInSz(p);
    int mSz = getMSz(p);
    T *d_xMMu = d_X;
    if (computeMu)
    {
        auto d_sum = gpuSum(p.bw, p.imgH, p.imgW, d_X);
        T *d_mu;
        if ((p.imgW & (p.imgW - 1)) == 0)
        {
            d_mu = gpuTruncate<T, T>(p.bw, p.bw, TruncateType::TrFloor, k.muTrKey, int(log2(p.imgW)), peer, party, mSz, d_sum, gaes, s);
        }
        else
        {
            gpuLinearComb(p.bw, mSz, d_sum, T((1LL << p.scale) / (double)p.imgW), d_sum);
            d_mu = gpuTruncate<T, T>(p.bw, p.bw, TruncateType::TrFloor, k.muTrKey, p.scale, peer, party, mSz, d_sum, gaes, s);
        }
        gpuFree(d_sum);
        d_xMMu = windowFunc<T, xPlusM<u64(1), u64(-1)>>(party, p, d_X, d_mu);
    }
    auto d_sqKey = (u8 *)moveToGPU((u8 *)k.sqKey.a, (2 * inSz) * sizeof(T), s);
    auto d_sq = pointFunc<T, square<T>>(party, p.bw, inSz, d_xMMu, (u8 *)d_sqKey);
    gpuFree(d_sqKey);

    auto d_sumSq = gpuSum(p.bw, p.imgH, p.imgW, d_sq);
    gpuFree(d_sq);
    peer->reconstructInPlace(d_sumSq, p.bw, mSz, s);

    auto h_sumSq = (T *)moveToCPU((u8 *)d_sumSq, p.imgH * sizeof(T), s);
    Rsqrt(p.imgH, h_sumSq, h_sumSq, p.imgW, p.scale, "LayerNorm::", invSqrtTab);
    moveIntoGPUMem((u8 *)d_sumSq, (u8 *)h_sumSq, mSz * sizeof(T), s);
    // (x - mu)*var
    auto d_invSqrt = d_sumSq;
    auto d_normX = windowMul(peer, party, p, k.wMulKey1, d_xMMu, d_invSqrt, TruncateType::TrWithSlack, gaes, s);
    gpuFree(d_invSqrt);
    auto p2 = transposeWindow(p);
    auto d_layerNorm = windowMul(peer, party, p2, k.wMulKey2, d_normX, d_A, TruncateType::TrWithSlack, gaes, s, d_B);
    gpuFree(d_normX);
    return d_layerNorm;
}