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

#include "gpu_truncate.h"
#include "utils/misc_utils.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"

#include "gpu_dcf_templates.h"
#include "fss/gpu_local_truncate.h"
#include <cassert>

namespace dcf
{

    template <typename T>
    __global__ void signExtendKeyKernel(int bin, int bout, int N, T *inMask, u8 *dcfMask, T *t, T *p, T *outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            t[i] = outMask[i] - inMask[i] - (T(1) << (bin - 1));
            gpuMod<T>(t[i], bout);
            assert(dcfMask[i] == 0 || dcfMask[i] == 1);
            int idx0 = dcfMask[i];
            int idx1 = 1 - idx0;
            p[2 * i + idx0] = 0;
            p[2 * i + idx1] = (T(1) << bin);
        }
    }

    template <typename T>
    T *genSignExtendKey(uint8_t **key_as_bytes, int party, int bin, int bout, int N, T *d_inputMask, AESGlobalContext *gaes)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);
        gpuKeyGenDCF<T>(key_as_bytes, party, bin, 1, N, d_inputMask, T(1), gaes);
        auto d_dcfMask = randomGEOnGpu<u8>(N, 1);
        writeShares<u8, u8>(key_as_bytes, party, N, d_dcfMask, 1);
        auto d_outputMask = randomGEOnGpu<T>(N, bout);
        auto d_T = (T *)gpuMalloc(N * sizeof(T));
        auto d_p = (T *)gpuMalloc(2 * N * sizeof(T));
        signExtendKeyKernel<<<(N - 1) / 256 + 1, 256>>>(bin, bout, N, d_inputMask, d_dcfMask, d_T, d_p, d_outputMask);
        writeShares<T, T>(key_as_bytes, party, N, d_T, bout);
        writeShares<T, T>(key_as_bytes, party, 2 * N, d_p, bout);
        gpuFree(d_dcfMask);
        gpuFree(d_T);
        gpuFree(d_p);
        return d_outputMask;
    }

    template <typename T>
    __global__ void keygenStTRKernel(int party, int bin, int bout, int shift, int N, T *inputMask, T *rHat, u8 *lsbMask, T *lsbCorr, T *outMask)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            auto temp = inputMask[i];
            gpuMod(temp, shift);
            auto corr = 1 - 1 * (rHat[i] < temp) - (inputMask[i] >> shift) + outMask[i];
            gpuMod(corr, bout);
            auto corrM1 = corr - 1;
            gpuMod(corrM1, bout);
            lsbCorr[2 * i + lsbMask[i]] = corr;
            lsbCorr[2 * i + (lsbMask[i] ^ 1)] = corrM1;
        }
    }

    template <typename T>
    T *genGPUStTRKey(uint8_t **key_as_bytes, int party, int bin, int bout, int shift, int N, T *d_inputMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, shift);
        writeInt(key_as_bytes, N);
        // printf("shift=%d, %lx\n", shift, h_r);
        auto d_rHat = randomGEOnGpu<T>(N, shift);
        if (h_r)
            moveIntoCPUMem((u8 *)h_r, (u8 *)d_rHat, N * sizeof(T), NULL);
        gpuLinearComb(shift, N, d_rHat, T(1), d_rHat, T(1), d_inputMask);
        gpuKeyGenDCF(key_as_bytes, party, shift, 1, N, d_rHat, T(1), gaes, true);
        auto d_lsbMask = randomGEOnGpu<u8>(N, 1);
        writeShares<u8, u8>(key_as_bytes, party, N, d_lsbMask, 1);
        auto d_outMask = randomGEOnGpu<T>(N, bout);
        auto d_lsbCorr = (T *)gpuMalloc(2 * N * sizeof(T));
        keygenStTRKernel<<<(N - 1) / 128 + 1, 128>>>(party, bin, bout, shift, N, d_inputMask, d_rHat, d_lsbMask, d_lsbCorr, d_outMask);
        writeShares<T, T>(key_as_bytes, party, 2 * N, d_lsbCorr, bout);
        gpuFree(d_inputMask);
        gpuFree(d_rHat);
        gpuFree(d_lsbMask);
        gpuFree(d_lsbCorr);
        return d_outMask;
    }

    template <typename T>
    T *genGPUStochasticTruncateKey(uint8_t **key_as_bytes, int party, int bin, int bout, int shift, int N, T *d_inputMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        auto d_trMask = genGPUStTRKey(key_as_bytes, party, bin, bin - shift, shift, N, d_inputMask, gaes, h_r);
        // this free happens inside genGPUStTRKey()
        // gpuFree(d_inputMask);
        auto d_outputMask = genSignExtendKey(key_as_bytes, party, bin - shift, bout, N, d_trMask, gaes);
        gpuFree(d_trMask);
        return d_outputMask;
    }

    template <typename T>
    T *genGPUTruncateKey(uint8_t **key_as_bytes, int party, TruncateType t, int bin, int bout, int shift, int N, T *d_inMask, AESGlobalContext *gaes, T *h_r = NULL)
    {
        T *d_outMask;
        switch (t)
        {
        case TruncateType::StochasticTruncate:
            d_outMask = genGPUStochasticTruncateKey(key_as_bytes, party, bin, bout, shift, N, d_inMask, gaes, h_r);
            break;
        case TruncateType::LocalARS:
            gpuLocalTr<T, T, ars>(party, bin, shift, N, d_inMask, true);
            d_outMask = d_inMask;
            break;
        case TruncateType::StochasticTR:
            bout = bin - shift;
            d_outMask = genGPUStTRKey(key_as_bytes, party, bin, bout, shift, N, d_inMask, gaes, h_r);
            break;
        default:
            d_outMask = d_inMask;
            assert(t == TruncateType::None);
        }
        return d_outMask;
    }


    template <typename T>
    __global__ void selectForTruncateKernel(T *x, u32 *maskedDcfBit, T *outMask, T *p, int N, int party)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            // can remove the cast to u32* for maskedDcfBit
            int dcfBit = (((u32 *)maskedDcfBit)[i / 32] >> (threadIdx.x & 0x1f)) & 1;
            x[i] = (party == SERVER1) * x[i] + outMask[i] + p[2 * i + dcfBit];
        }
    }

    // no memory leak
    template <typename T>
    void gpuSelectForTruncate(int party, int N, T *d_I, u32 *d_maskedDcfBit, T *h_outMask, T *h_p, Stats *s)
    {
        size_t memSz = N * sizeof(T);
        auto d_outMask = (T *)moveToGPU((u8 *)h_outMask, memSz, s);
        auto d_p = (T *)moveToGPU((u8 *)h_p, 2 * memSz, s);
        selectForTruncateKernel<T><<<(N - 1) / 128 + 1, 128>>>(d_I, d_maskedDcfBit, d_outMask, d_p, N, party);
        checkCudaErrors(cudaDeviceSynchronize());
        gpuFree(d_outMask);
        gpuFree(d_p);
    }

    // no memory leaks
    template <typename T>
    void gpuSignExtend(GPUSignExtendKey<T> k, int party, SigmaPeer *peer, T *d_I, AESGlobalContext *g, Stats *s)
    {
        gpuLinearComb(k.bin, k.N, d_I, T(1), d_I, T(1ULL << (k.bin - 1)));
        std::vector<u32 *> h_dcfMask = {k.dcfKey.dReluMask};
        auto d_maskedDcfBit = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::maskEpilogue>(k.dcfKey.dcfKey, party, d_I, g, s, &h_dcfMask);
        peer->reconstructInPlace(d_maskedDcfBit, 1, k.N, s);
        gpuSelectForTruncate(party, k.N, d_I, d_maskedDcfBit, k.t, k.p, s);
        peer->reconstructInPlace(d_I, k.bout, k.N, s);
        gpuFree(d_maskedDcfBit);
    }

    template <typename T>
    __global__ void stochasticTRKernel(int party, int bin, int bout, int shift, int N, T *d_I, u32 *d_dcf, T *lsbCorr)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
        {
            T lsb = (T)((d_dcf[i / PACKING_SIZE] >> (threadIdx.x & 0x1f)) & 1);
            d_I[i] = (party == SERVER1) * (d_I[i] >> shift) + lsbCorr[2 * i + lsb];
            gpuMod(d_I[i], bout);
        }
    }

    template <typename T>
    void gpuStochasticTR(GPUStTRKey<T> k, int party, SigmaPeer *peer, T *d_I, AESGlobalContext *g, Stats *s)
    {
        std::vector<u32 *> h_mask = {k.lsbKey.dReluMask};
        auto d_dcf = dcf::gpuDcf<T, 1, idPrologue, maskEpilogue>(k.lsbKey.dcfKey, party, d_I, g, s, &h_mask);
        peer->reconstructInPlace(d_dcf, 1, k.N, s);
        auto d_lsbCorr = (T *)moveToGPU((u8 *)k.lsbCorr, 2 * k.N * sizeof(T), s);
        stochasticTRKernel<<<(k.N - 1) / 128 + 1, 128>>>(party, k.bin, k.bout, k.shift, k.N, d_I, d_dcf, d_lsbCorr);
        peer->reconstructInPlace(d_I, k.bout, k.N, s);
        gpuFree(d_dcf);
        gpuFree(d_lsbCorr);
    }

    template <typename T>
    void gpuStochasticTruncate(GPUTruncateKey<T> k, int party, SigmaPeer *peer, T *d_I, AESGlobalContext *g, Stats *s)
    {
        gpuStochasticTR(k.stTRKey, party, peer, d_I, g, s);
        gpuSignExtend(k.signExtendKey, party, peer, d_I, g, s);
    }

    template <typename T>
    void gpuTruncate(int bin, int bout, TruncateType t, GPUTruncateKey<T> k, int shift, SigmaPeer *peer, int party, int N, T *d_I, AESGlobalContext *gaes, Stats *s)
    {
        switch (t)
        {
        case TruncateType::StochasticTR:
            // assert(bout == bin - shift);
            bout = bin - shift;
            gpuStochasticTR(k.stTRKey, party, peer, d_I, gaes, s);
            break;
        case TruncateType::LocalARS:
            gpuLocalTr<T, T, ars>(party, bin, shift, N, d_I, true);
            break;
        case TruncateType::StochasticTruncate:
            gpuStochasticTruncate(k, party, peer, d_I, gaes, s);
            break;
        default:
            assert(t == TruncateType::None);
        }
        return;
    }

    // check via tolerance bounds
    template <typename T>
    void checkTrStWithTol(int bin, int bout, int shift, int N, T *h_masked_A, T *h_mask_A, T *h_A_ct)
    {
        for (int i = 0; i < N; i++)
        {
            auto temp = h_A_ct[i] + T(1ULL << (bin - 1));
            cpuMod(temp, bin);
            auto truncated_A = temp >> shift;
            auto truncated_A_plus1 = truncated_A + 1;
            cpuMod(truncated_A_plus1, bin - shift);
            truncated_A -= T(1ULL << (bin - shift - 1));
            cpuMod(truncated_A, bout);
            truncated_A_plus1 -= T(1ULL << (bin - shift - 1));
            cpuMod(truncated_A_plus1, bout);
            auto output = h_masked_A[i] - h_mask_A[i];
            cpuMod(output, bout);
            if (i < 10)
                printf("%lu %lu %lu\n", h_A_ct[i], u64(output), u64(truncated_A));
            if (output != truncated_A && output != truncated_A_plus1)
                printf("%lu %lu %lu %lu\n", h_A_ct[i], u64(output), u64(truncated_A), u64(truncated_A_plus1));
            assert(output == truncated_A || output == truncated_A_plus1);
        }
    }

}