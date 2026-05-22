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

#include "gpu_maxpool.h"
#include "gpu_gelu.h"

template <typename TIn, typename TOut>
__global__ void keyGenGeluMuxKernel(int party, int bin, int bout, TOut *linFunc, int N, TIn *b0Mask, TIn *b1Mask, TIn *mask_X, TOut *outMask, TOut *c0, TOut *c1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        auto muxMask = 2 * b1Mask[i] + b0Mask[i];
        for (int j = 0; j < 4; j++)
        {
            auto idx = 4 * i + j ^ muxMask;
            auto temp = -linFunc[4 + j] * mask_X[i];
            gpuMod(temp, bin);
            c0[idx] = linFunc[j] + /*- linFunc[4 + j] * mask_X[i]*/ temp + outMask[i];
            c1[idx] = linFunc[4 + j];

            gpuMod(c0[idx], bout);
            gpuMod(c1[idx], bout);
        }
        // if(i == 0) printf("MuxKernel %d: Input=%ld, Output=%ld\n", i, mask_X[i], outMask[i]);
    }
}

template <typename TIn, typename TOut>
TOut *keyGenGeluMux(u8 **key_as_bytes, int party, int bin, int bout, const TOut linFunc[2][4], int N, TIn *d_b0Mask, TIn *d_b1Mask, TIn *d_mask_X)
{
    assert(bin <= 8 * sizeof(TIn));
    assert(bout <= 8 * sizeof(TOut));
    auto d_outMask = randomGEOnGpu<TOut>(N, bout);
    // checkCudaErrors(cudaMemset(d_outMask, 0, N * sizeof(TOut)));
    u64 memSzC = 4 * N * sizeof(TOut);
    auto d_c0 = (TOut *)gpuMalloc(memSzC);
    auto d_c1 = (TOut *)gpuMalloc(memSzC);
    auto d_linFunc = (TOut *)moveToGPU((u8 *)linFunc, 8 * sizeof(TOut), NULL);
    keyGenGeluMuxKernel<TIn, TOut><<<(N - 1) / 128 + 1, 128>>>(party, bin, bout, d_linFunc, N, d_b0Mask, d_b1Mask, d_mask_X, d_outMask, d_c0, d_c1);
    writeShares<TOut, TOut>(key_as_bytes, party, 4 * N, d_c0, bout);
    writeShares<TOut, TOut>(key_as_bytes, party, 4 * N, d_c1, bout);
    gpuFree(d_c0);
    gpuFree(d_c1);
    gpuFree(d_linFunc);
    return d_outMask;
}

template <typename TIn, typename TOut>
__global__ void geluMuxKernel(int party, int bin, int bout, int N, u32 *drelu_g, u32 *ic_g, TIn *Xt, TOut *out, TOut *c0_g, TOut *c1_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        // assert(bout == 8);
        auto drelu = (drelu_g[i / 32] >> (i % 32)) & u32(1);
        auto ic = (ic_g[i / 32] >> (i % 32)) & u32(1);
        // if(i == 0) printf("dReluBit=%u, ze bit=%u\n", ic, drelu);
        auto c = 4 * i + 2 * ic + drelu;
        auto c0 = TIn(c0_g[c]);
        auto c1 = TIn(c1_g[c]);
        out[i] = c1 * Xt[i] + c0;
        gpuMod(out[i], bout);
        // if(i == 0) printf("MuxKernel %d: Input=%ld, Output=%ld\n", i, Xt[i], out[i]);
    }
}

template <typename TIn, typename TOut>
TOut *geluMux(SigmaPeer *peer, int party, GPUGeluMuxKey<TOut> k, int bin, int bout, int N, u32 *d_drelu, u32 *d_ic, TIn *d_Xt, Stats *s)
{
    // assert(bout == 8);
    assert(bout <= 8 * sizeof(TOut));
    auto d_out = (TOut *)gpuMalloc(N * sizeof(TOut));
    u64 memSzC = 4 * N * sizeof(TOut);
    auto d_c0 = (TOut *)moveToGPU((u8 *)k.c0, memSzC, s);
    auto d_c1 = (TOut *)moveToGPU((u8 *)k.c1, memSzC, s);
    geluMuxKernel<TIn, TOut><<<(N - 1) / 128 + 1, 128>>>(party, bin, bout, N, d_drelu, d_ic, d_Xt, d_out, d_c0, d_c1);
    gpuFree(d_c0);
    gpuFree(d_c1);
    peer->reconstructInPlace(d_out, bout, N, s);
    return d_out;
}

template <typename T, typename TClip, int clipBw>
T *gpuKeyGenGelu(uint8_t **key_as_bytes, int party, int bw, int bin, int scale, int N, T *d_mask_X, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, bw);
    int bwXt = bin - scale + 6 + 1;
    // truncated X = Xt
    auto d_mask_Xt = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrWithSlack, bw, bwXt, scale - 6, N, d_mask_X, gaes);
    auto d_dReluMask = gpuKeyGenDRelu(key_as_bytes, party, bwXt, N, d_mask_Xt, gaes);
    assert(bwXt > 7);
    // printf("ClipBW=%d\n", clipBw);
    assert(8 * sizeof(TClip) >= clipBw);
    const u64 max = (1ULL << clipBw) - 1;
    auto d_icMask = gpuKeyGenIc<T, max, -max>(key_as_bytes, party, bwXt, N, d_mask_Xt, false, gaes);
    const TClip linFunc[2][4] = {
        {TClip(max), TClip(max), 0, 0},
        {0, 0, TClip(-1), 1}};
    auto d_clipMask = keyGenGeluMux<T, TClip>(key_as_bytes, party, bwXt, clipBw, linFunc, N, d_dReluMask, d_icMask, d_mask_Xt);
    auto d_lutMask = gpuKeyGenLUT<TClip, T>(key_as_bytes, party, clipBw, bw, N, d_clipMask, gaes);
    gpuFree(d_clipMask);

    // auto d_reluMask = randomGEOnGpu<T>(N, bw);
    auto d_reluMask = gpuKeyGenSelect<T, T>(key_as_bytes, party, N, d_mask_X, d_dReluMask, bw);

    gpuLinearComb(bw, N, d_reluMask, T(1), d_reluMask, -T(1), d_lutMask);
    gpuFree(d_lutMask);
    return d_reluMask;
}

// clip happens in place
template <typename T, typename TClip, int clipBw>
T *gpuGelu(SigmaPeer *peer, int party, GPUGeluKey<T, TClip> &k, int bw, int bin, int scale, int N, T *d_X, T *d_geluSubRelu, AESGlobalContext *gaes, Stats *s)
{
    assert(8 * sizeof(TClip) >= clipBw);
    assert(bin > scale - 6);
    int bwXt = bin - scale + 6 + 1;
    // do a truncate reduce
    auto d_Xt = gpuTruncate(bw, bwXt, TruncateType::TrWithSlack, k.trKey, scale - 6, peer, party, N, d_X, gaes, s);
    // the -1 doesn't matter because anything larger is anyway set to (1 << clipBw) - 1
    const u64 clipVal = (1ULL << clipBw) - 1;
    std::vector<u32 *> h_masks({k.dReluKey.mask, k.icMask});
    u32 *d_res = gpuDcf<T, 3, geluPrologue<clipVal, -clipVal>, geluEpilogue<clipVal, -clipVal>>(k.dReluKey.dpfKey, party, d_Xt, gaes, s, &h_masks);
    int numInts = ((N - 1) / PACKING_SIZE + 1);
    peer->reconstructInPlace(d_res, 1, 2 * numInts * 32, s);

    u32 *d_dRelu = d_res;
    u32 *d_ic = d_res + numInts;
    auto d_clippedX = geluMux<T, TClip>(peer, party, k.muxKey, bwXt, clipBw, N, d_dRelu, d_ic, d_Xt, s);
    gpuFree(d_Xt);
    auto d_reluSubGelu = gpuDpfLUT<TClip, T>(k.lutKey, peer, party, d_clippedX, d_geluSubRelu, gaes, s, false);
    gpuFree(d_clippedX);
    T *d_relu = gpuSelect<T, T, 0, 0>(peer, party, bw, k.reluSelectKey, d_dRelu, d_X, s, false);
    gpuFree(d_res);
    gpuLinearComb(bw, N, d_relu, T(1), d_relu, -T(1), d_reluSubGelu);
    gpuFree(d_reluSubGelu);
    peer->reconstructInPlace(d_relu, bw, N, s);
    return d_relu;
}