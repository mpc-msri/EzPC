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

#include <cstdint>

#include "fss/gpu_truncate.h"
#include "fss/gpu_softmax.h"
#include "fss/gpu_matmul.h"

struct MHAParams
{
    int n_seq, n_embed, n_heads, dim_W;
    bool selfAttn, doNormQKt, rotEmb;
};

struct MHAMulParams
{
    MatmulParams pQKV, pQKt, pSmQKtV, pProj;
    MaxpoolParams pMPool;
};

template <typename T>
struct GPUMHAKey
{
    GPUMatmulKey<T> mmKeyQKV, mmKeyQKt, mmKeySmQKtV, mmKeyProj;
    GPUSoftMaxKey<T> softmaxKey;
    GPUTruncateKey<T> reQTrKey, reKTrKey, normQKtTrKey;
};

template <typename T>
struct MHATables
{
    T *d_nExpMsbTab = NULL, *d_nExpLsbTab = NULL, *d_invTab = NULL;
};

template <typename T>
inline MHATables<T> initMHATables(int n_seq, int scale)
{
    MHATables<T> mhaTab;
    mhaTab.d_nExpMsbTab = genLUT<T, nExpMsb<T>>(8, 4, scale);
    mhaTab.d_nExpLsbTab = genLUT<T, nExpLsb<T>>(8, 12, scale);
    mhaTab.d_invTab = genLUT<T, inv<T>>(int(ceil(log2(n_seq))) + 6, 6, scale);
    return mhaTab;
}

inline MatmulParams initPQKV(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pQKV;
    pQKV.M = pMHA.n_seq;
    pQKV.K = pMHA.n_embed;
    pQKV.N = pMHA.dim_W;
    pQKV.batchSz = 3 * pMHA.n_heads;
    stdInit(pQKV, bw, scale);
    pQKV.size_A = pQKV.M * pQKV.K;
    pQKV.stride_A = 0;
    pQKV.ld_B = 3 * pMHA.n_embed;
    pQKV.stride_B = pMHA.dim_W;
    return pQKV;
}

inline MatmulParams initPQKt(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pQKt;
    pQKt.M = pMHA.n_seq;
    pQKt.K = pMHA.dim_W;
    pQKt.N = pMHA.n_seq;
    pQKt.batchSz = pMHA.n_heads;
    if (pMHA.selfAttn)
        pQKt.cIsLowerTriangular = true;

    stdInit(pQKt, bw, scale);
    if (pMHA.doNormQKt && int(log2(pMHA.dim_W)) % 2 == 0)
    {
        // assert(int(log2(dim_W)) % 2 == 0);
        // printf("Shift=%d\n", int(log2(pMHA.dim_W) / 2));
        pQKt.shift += int(log2(pMHA.dim_W) / 2);
    }
    // K is stored in column-major form
    pQKt.rowMaj_B = false;
    pQKt.ld_B = pQKt.K;
    return pQKt;
}

inline MatmulParams initPSmQKtV(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pSmQKtV;
    pSmQKtV.M = pMHA.n_seq;
    pSmQKtV.K = pMHA.n_seq;
    pSmQKtV.N = pMHA.dim_W;
    pSmQKtV.batchSz = pMHA.n_heads;
    stdInit(pSmQKtV, bw, scale);
    pSmQKtV.ld_C = pMHA.n_heads * pMHA.dim_W;
    pSmQKtV.stride_C = pMHA.dim_W;
    return pSmQKtV;
}

inline MatmulParams initPProj(MHAParams pMHA, int bw, int scale)
{
    MatmulParams pProj;
    pProj.M = pMHA.n_seq;
    pProj.K = pMHA.n_embed;
    pProj.N = pMHA.n_embed;
    pProj.batchSz = 1;
    stdInit(pProj, bw, scale);
    return pProj;
}

inline MaxpoolParams initPMaxpool(MHAParams pMHA, int bw, int scale)
{
    MaxpoolParams pMPool;
    pMPool.N = pMHA.n_heads;
    pMPool.imgH = pMHA.n_seq;
    pMPool.imgW = pMHA.n_seq;
    pMPool.C = 1;
    pMPool.FH = 1;
    pMPool.FW = pMHA.n_seq;
    pMPool.strideH = 1;
    pMPool.strideW = pMHA.n_seq;
    pMPool.zPadHLeft = 0;
    pMPool.zPadWLeft = 0;
    pMPool.zPadHRight = 0;
    pMPool.zPadWRight = 0;
    pMPool.bw = bw;
    pMPool.bin = bw - scale;
    pMPool.scale = scale;
    pMPool.scaleDiv = 0;
    initPoolParams(pMPool);
    pMPool.bwBackprop = 0;
    if (pMHA.selfAttn)
        pMPool.isLowerTriangular = true;
    return pMPool;
}

MHAMulParams initMHAMulParams(MHAParams pMHA, int bw, int scale) {
    MHAMulParams pMHAMul;
    pMHAMul.pQKV = initPQKV(pMHA, bw, scale);
    pMHAMul.pQKt = initPQKt(pMHA, bw, scale);
    pMHAMul.pMPool = initPMaxpool(pMHA, bw, scale);
    pMHAMul.pSmQKtV = initPSmQKtV(pMHA, bw, scale);
    pMHAMul.pProj = initPProj(pMHA, bw, scale);
    return pMHAMul;
}

template <typename T>
GPUMHAKey<T> readGPUMHAKey(MHAParams pMHA, MHAMulParams pMHAMul, u8 **key_as_bytes)
{
    GPUMHAKey<T> k;
    k.mmKeyQKV = readGPUMatmulKey<T>(pMHAMul.pQKV, TruncateType::TrFloor, key_as_bytes);
    if (pMHA.rotEmb)
    {
        k.reQTrKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
        k.reKTrKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
    }
    k.mmKeyQKt = readGPUMatmulKey<T>(pMHAMul.pQKt, TruncateType::TrFloor, key_as_bytes);
    if (pMHA.doNormQKt && int(log2(pMHA.dim_W)) % 2 == 1)
        k.normQKtTrKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
    k.softmaxKey = readGPUSoftMaxKey<T>(pMHAMul.pMPool, key_as_bytes);
    k.mmKeySmQKtV = readGPUMatmulKey<T>(pMHAMul.pSmQKtV, TruncateType::TrFloor, key_as_bytes);
    k.mmKeyProj = readGPUMatmulKey<T>(pMHAMul.pProj, TruncateType::TrFloor, key_as_bytes);
    return k;
}

#include "gpu_mha.cu"