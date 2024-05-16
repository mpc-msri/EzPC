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

#include <cassert>
#include "utils/gpu_mem.h"

#include "gpu_sgd.h"

namespace dcf
{

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
        leftShiftAndAddKernel<<<(N - 1) / 128 + 1, 128>>>(d_A, d_B, d_C, shift, alpha, N);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    template <typename T>
    void genGpuSGDWithMomentumKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *h_W, T *d_W,
                                  T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t, AESGlobalContext *gaes, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        auto d_Vw = (T *)moveToGPU((u8 *)h_Vw, memSizeW, NULL);
        int shift = orca::mom_scale + scaleVw - scaledW;
        // the d_dW mask got moved to the left by shift
        gpuLeftShiftAndAdd(N, d_dW, d_Vw, d_Vw, shift, T(orca::mom_fp));
        d_Vw = genGPUTruncateKey(key_as_bytes, party, t, bin, bout, orca::mom_scale, N, d_Vw, gaes);
        moveIntoCPUMem((u8 *)h_Vw, (u8 *)d_Vw /*d_dW*/, memSizeW, NULL);

        bool dWWasNull = false;
        if (d_W == NULL)
        {
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, NULL);
            dWWasNull = true;
        }
        shift = orca::lr_scale[epoch] + scaleVw - scaleW;
        // this is wrong it needs to be -lr
        auto d_new_W = (T *)gpuMalloc(memSizeW);
        gpuLeftShiftAndAdd(N, d_W, d_Vw, d_new_W, shift, -T(orca::lr_fp));
        if (shift > 0)
            d_new_W = genGPUTruncateKey(key_as_bytes, party, t, bin, bout, shift, N, d_new_W, gaes);
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_new_W, memSizeW, NULL);
        gpuFree(d_new_W);
        if (dWWasNull)
            gpuFree(d_W);
        gpuFree(d_Vw);
        // // dW = W << (scale + orca::lr_scale[epoch]) + orca::lr_fp * Vw;
        // shift = scaleVw + orca::lr_scale[epoch] - scaleW;
        // // Neha: this is wrong. it needs to be -lr
        // gpuLeftShiftAndAddWrapper(N, W, Vw, dW, shift, -orca::lr_fp);
        // genGPUTruncateKey(f1, f2, shift > 0 ? t : TruncateType::None, bin, bout, shift, N, dW, W);
    }

    template <typename T>
    void readGpuSGDWithMomentumKey(TruncateType t, GPUTruncateKey<T> *truncateKeyVw, GPUTruncateKey<T> *truncateKeyW, u8 **key_as_bytes, int scaleW, int scaleVw, int scaledW, int epoch)
    {
        *truncateKeyVw = readGPUTruncateKey<T>(t, key_as_bytes);
        int shift = orca::lr_scale[epoch] + scaleVw - scaleW;
        if (shift > 0)
            *truncateKeyW = readGPUTruncateKey<T>(t, key_as_bytes);
    }

    template <typename T>
    void gpuSgdWithMomentum(int bin, int bout, int N, T *h_W, T *d_W,
                            T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, dcf::TruncateType t,
                            dcf::GPUTruncateKey<T> truncateKeyVw, GPUTruncateKey<T> truncateKeyW, int party, SigmaPeer *peer, AESGlobalContext *gaes, Stats *s, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        auto d_Vw = (T *)moveToGPU((u8 *)h_Vw, memSizeW, s);
        int shift = orca::mom_scale + scaleVw - scaledW;
        // printf("h_Vw=%ld\n", h_Vw[0]);
        // the d_dW mask got moved to the left by shift
        gpuLeftShiftAndAdd(N, d_dW, d_Vw, d_Vw, shift, T(orca::mom_fp));
        dcf::gpuTruncate(bin, bout, t, truncateKeyVw, orca::mom_scale, peer, party, N, d_Vw, gaes, s);
        moveIntoCPUMem((u8 *)h_Vw, (u8 *)d_Vw /*d_dW*/, memSizeW, s);

        bool dWWasNull = false;
        if (d_W == NULL)
        {
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, s);
            dWWasNull = true;
        }
        shift = orca::lr_scale[epoch] + scaleVw - scaleW;
        // this is wrong it needs to be -lr
        gpuLeftShiftAndAdd(N, d_W, d_Vw, d_W, shift, -T(orca::lr_fp));
        if (shift > 0)
            dcf::gpuTruncate(bin, bout, t, truncateKeyW, shift, peer, party, N, d_W, gaes, s);
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_W, memSizeW, s);
        if (dWWasNull)
            gpuFree(d_W);
        gpuFree(d_Vw);
    }

    template <typename T>
    void checkSgdWithMomentum(int bin, int bout, int N,
                              T *h_W, T *h_Vw, T *h_dW,
                              T *h_masked_W, T *h_masked_Vw,
                              T *h_mask_W, T *h_mask_Vw,
                              int scaleW, int scaleVw, int scaledW, int epoch)
    {
        int shiftdW = scaleVw + orca::mom_scale - scaledW;
        int shiftW = orca::lr_scale[epoch] + scaleVw - scaleW;
        for (int i = 0; i < N; i++)
        {
            auto vw = h_masked_Vw[i] - h_mask_Vw[i];
            auto vw_ct = cpuArs((h_dW[i] << shiftdW) + T(orca::mom_fp) * h_Vw[i], bin, orca::mom_scale);
            // if(i < 10) printf("%lu %lu\n", u64(vw), u64(vw_ct));
            assert(vw - vw_ct <= 1);
            auto w_ct = cpuArs((h_W[i] << shiftW) - T(orca::lr_fp) * vw_ct, bin, shiftW);
            // this is the new masked f
            auto w = h_masked_W[i] - h_mask_W[i];
            // need to test this when the starting vf is non-zero
            auto diff = abs(static_cast<int64_t>(u64(w) - u64(w_ct)));
            if (i < 10)
                printf("%lu %lu %ld\n", u64(w), u64(w_ct), diff);
            // the two is important
            assert(/*abs(static_cast<int64_t>(w - w_ct))*/ diff <= 2);
        }
    }

    template <typename T>
    T *gpuMultiplyByConstant(T *d_A, T x, int N)
    {
        auto d_B = (T *)gpuMalloc(N * sizeof(T));
        gpuLinearComb(sizeof(T) * 8, N, d_B, x, d_A);
        return d_B;
    }

    template <typename T>
    void genGpuSGDKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *h_W, T *d_W,
                      T *d_dW, int scaleW, int scaledW, TruncateType t, AESGlobalContext *gaes, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        auto d_delta = gpuMultiplyByConstant(d_dW, -T(orca::lr_fp), N);
        int rightShift = scaledW + orca::lr_scale[epoch] - scaleW;
        bool dWWasNull = false;
        if (rightShift > 0)
        {
            assert(rightShift == orca::global::scale + orca::lr_scale[epoch]);
            d_delta = genGPUTruncateKey(key_as_bytes, party, t, bin, bout, rightShift, N, d_delta, gaes);
            gpuLinearComb(bin, N, d_W, T(1), d_W, T(1), d_delta);
        }
        else
        {
            int leftShift = scaleW - orca::lr_scale[epoch] - scaledW;
            assert(leftShift == orca::global::scale - orca::lr_scale[epoch]);
            assert(d_W == NULL);
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, NULL);
            dWWasNull = true;
            gpuLeftShiftAndAdd(N, d_delta, d_W, d_W, leftShift, T(1));
        }
        gpuFree(d_delta);
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_W, memSizeW, NULL);
        if (dWWasNull)
            gpuFree(d_W);
    }

    template <typename T>
    void readGpuSGDKey(TruncateType t, int scaleW, int scaledW, GPUTruncateKey<T> *truncateKeyW, u8 **key_as_bytes, int epoch)
    {
        int rightShift = scaledW + orca::lr_scale[epoch] - scaleW;
        if (rightShift > 0)
        {
            *truncateKeyW = readGPUTruncateKey<T>(t, key_as_bytes);
        }
    }

    template <typename T>
    void gpuSgd(int bin, int bout, int N, T *h_W, T *d_W,
                T *d_dW, int scaleW, int scaledW, TruncateType t,
                GPUTruncateKey<T> truncateKeyW, int party, SigmaPeer *peer, AESGlobalContext *gaes, Stats *s, int epoch)
    {
        size_t memSizeW = N * sizeof(T);
        // the d_dW mask got moved to the left by shift
        auto d_delta = gpuMultiplyByConstant(d_dW, -T(orca::lr_fp), N);
        int rightShift = orca::lr_scale[epoch] + scaledW - scaleW;
        bool dWWasNull = false;
        if (rightShift > 0)
        {
            assert(rightShift == orca::global::scale + orca::lr_scale[epoch]);
            dcf::gpuTruncate(bin, bout, t, truncateKeyW, rightShift, peer, party, N, d_delta, gaes, s);
            gpuLinearComb(bin, N, d_W, T(1), d_W, T(1), d_delta);
        }
        else
        {
            int leftShift = scaleW - orca::lr_scale[epoch] - scaledW;
            assert(leftShift == orca::global::scale - orca::lr_scale[epoch]);
            assert(d_W == NULL);
            d_W = (T *)moveToGPU((u8 *)h_W, memSizeW, NULL);
            dWWasNull = true;
            gpuLeftShiftAndAdd(N, d_delta, d_W, d_W, leftShift, T(1));
        }
        gpuFree(d_delta);
        moveIntoCPUMem((u8 *)h_W, (u8 *)d_W, memSizeW, s);
        if (dWWasNull)
            gpuFree(d_W);
    }

    template <typename T>
    void checkSgd(int bin, int bout, int N,
                  T *h_W, T *h_dW, T *h_masked_W,
                  T *h_mask_W, int scaleW, int scaledW, int epoch)
    {
        int rightShift = orca::lr_scale[epoch] + scaledW - scaleW;
        if (rightShift > 0)
        {
            assert(rightShift == orca::global::scale + orca::lr_scale[epoch]);
            for (int i = 0; i < N; i++)
            {
                auto w_ct = h_W[i] - cpuArs(T(orca::lr_fp) * h_dW[i], bin, rightShift);
                // this is the new masked f
                auto w = h_masked_W[i] - h_mask_W[i];
                // need to test this when the starting vf is non-zero
                auto diff = abs(static_cast<int32_t>(w - w_ct));
                if (i < 10)
                    printf("%lu %lu %d\n", u64(w), u64(w_ct), diff);
                assert(diff <= 1);
            }
        }
        else
        {
            int leftShift = scaleW - orca::lr_scale[epoch] - scaledW;
            assert(leftShift == orca::global::scale - orca::lr_scale[epoch]);
            for (int i = 0; i < N; i++)
            {
                auto w_ct = h_W[i] - T(orca::lr_fp) * h_dW[i] * (T(1) << leftShift);
                // this is the new masked f
                auto w = h_masked_W[i] - h_mask_W[i];
                // need to test this when the starting vf is non-zero
                auto diff = abs(static_cast<int32_t>(w - w_ct));
                if (i < 10)
                    printf("%lu %lu %ld\n", w, w_ct, diff);
                assert(diff == 0);
            }
        }
    }

    template <typename T>
    void genOptimizerKey(u8 **key_as_bytes, int party, int bin, int bout, int N, T *h_W, T *d_W,
                         T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t, bool useMomentum, AESGlobalContext *gaes, int epoch)
    {
        if (useMomentum)
        {
            genGpuSGDWithMomentumKey(key_as_bytes, party, bin, bout, N, h_W, d_W, h_Vw, d_dW, scaleW, scaleVw, scaledW, t, gaes, epoch);
        }
        else
        {
            genGpuSGDKey(key_as_bytes, party, bin, bout, N, h_W, d_W, d_dW, scaleW, scaledW, t, gaes, epoch);
        }
    }

    template <typename T>
    void readOptimizerKey(TruncateType t, GPUTruncateKey<T> *truncateKeyVw, GPUTruncateKey<T> *truncateKeyW, u8 **key_as_bytes, int scaleW, int scaleVw, int scaledW, bool useMomentum, int epoch)
    {
        if (useMomentum)
        {
            readGpuSGDWithMomentumKey(t, truncateKeyVw, truncateKeyW, key_as_bytes, scaleW, scaleVw, scaledW, epoch);
        }
        else
        {
            readGpuSGDKey(t, scaleW, scaledW, truncateKeyW, key_as_bytes, epoch);
        }
    }

    template <typename T>
    void optimize(int bin, int bout, int N, T *h_W, T *d_W,
                  T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t,
                  GPUTruncateKey<T> truncateKeyVw, GPUTruncateKey<T> truncateKeyW, int party, SigmaPeer *peer, bool useMomentum, AESGlobalContext *gaes, Stats *s, int epoch)
    {
        if (useMomentum)
        {
            gpuSgdWithMomentum(bin, bout, N, h_W, d_W, h_Vw, d_dW, scaleW, scaleVw, scaledW, t, truncateKeyVw, truncateKeyW, party, peer, gaes, s, epoch);
        }
        else
        {
            gpuSgd(bin, bout, N, h_W, d_W, d_dW, scaleW, scaledW, t, truncateKeyW, party, peer, gaes, s, epoch);
        }
    }

    template <typename T>
    void checkOptimizer(int bin, int bout, int N,
                        T *h_W, T *h_Vw, T *h_dW,
                        T *h_masked_W, T *h_masked_Vw,
                        T *h_mask_W, T *h_mask_Vw,
                        int scaleW, int scaleVw, int scaledW, bool useMomentum, int epoch)
    {
        if (useMomentum)
        {
            checkSgdWithMomentum(bin, bout, N, h_W, h_Vw, h_dW, h_masked_W, h_masked_Vw, h_mask_W, h_mask_Vw, scaleW, scaleVw, scaledW, epoch);
        }
        else
        {
            checkSgd(bin, bout, N, h_W, h_dW, h_masked_W, h_mask_W, scaleW, scaledW, epoch);
        }
    }

}