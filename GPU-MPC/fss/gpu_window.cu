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

#include "gpu_window.h"

struct WindowArgs
{
    int party, i, j, N, M;
};

typedef u64 (*fMax)(WindowArgs w, u64 x, u64 max, u8 *bytes);

template <u64 a, u64 b>
__device__ u64 xPlusM(WindowArgs w, u64 x, u64 m, u8 *bytes)
{
    // if(w.j <= 10)//>= 4 && i < 8)
    // printf("Inside max %d=%lu, %lu, %lu\n", w.i, m, x, m - x);
    return a * x + b * m;
}

template <typename T>
__device__ u64 keygenMul(WindowArgs w, u64 x, u64 max, u8 *bytes)
{
    auto r = ((T *)bytes)[w.i];
    // printf("beaverKeygen=%ld, %ld, %ld, %ld\n", x, max, x*max, r);
    return x * max + r;
}

template <typename T>
__device__ u64 beaverMul(WindowArgs w, u64 x, u64 max, u8 *bytes)
{
    auto a = ((T *)bytes)[w.i];
    auto b = ((T *)bytes)[w.N + w.j];
    auto c = ((T *)bytes)[w.N + w.M + w.i];
    // printf("%ld, %ld, %ld, %ld, %ld, %d, %d, %d, %d\n", a, b, c, x, max, M, N, i, j);
    // (x - a) * (y - b) =
    return (w.party == SERVER1) * (x * max) - x * b - a * max + c;
}

template <typename T, fMax f>
__global__ void windowFuncKernel(int party, MaxpoolParams p, T *d_X, T *d_max, T *d_out, int N, int M, u8 *bytes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        int t = i;
        int j;
        if (p.isLowerTriangular)
        {
            int n, h;
            assert(p.imgH == p.imgW);
            assert(p.C == 1);
            n = t / ((p.imgH * (p.imgH + 1)) / 2);
            t = t % ((p.imgH * (p.imgH + 1)) / 2);
            h = int(floor((-1.0f + sqrt(1 + 8.0f * t)) / 2.0f));
            // w = t - ((h * (h + 1)) / 2);
            j = n * p.imgH + h;
            // printf("%d=%d\n", t, j);
        }
        else
        {
            int n, h, w, c;
            n = t / (p.imgH * p.imgW * p.C);
            t = t % (p.imgH * p.imgW * p.C);
            h = t / (p.imgW * p.C);
            t = t % (p.imgW * p.C);
            w = t / (p.C);
            c = t % (p.C);
            j = n * p.H * p.W * p.C + (h / p.strideH) * p.W * p.C + (w / p.strideW) * p.C + c;
        }
        WindowArgs wa{party, i, j, N, M};
        auto o = T(f(wa, u64(d_X[i]), u64(d_max[j]), bytes));
        gpuMod(o, p.bw);
        d_out[i] = o;
    }
}

template <typename T, fMax f>
T *windowFunc(int party, MaxpoolParams p, T *d_X, T *d_M, u8 *d_bytes = NULL, bool inPlace = false)
{
    int inSz = getInSz(p);
    int mSz = getMSz(p);
    T *d_out = d_X;
    if (!inPlace)
        d_out = (T *)gpuMalloc(inSz * sizeof(T));
    // printf("%d, %d, %d, %d\n", p.strideH, p.FH, p.strideW, p.FW);
    assert(p.strideH == p.FH && p.strideW == p.FW);
    assert(p.zPadHLeft == 0 && p.zPadHRight == 0 && p.zPadWLeft == 0 && p.zPadWRight == 0);
    windowFuncKernel<T, f><<<(inSz - 1) / 128 + 1, 128>>>(party, p, d_X, d_M, d_out, inSz, mSz, d_bytes);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_out;
}

template <typename T>
T *keygenWindowMul(u8 **key_as_bytes, int party, MaxpoolParams p, T *d_mask_X, T *d_mask_M, TruncateType t, AESGlobalContext *gaes, T *d_mask_B = NULL)
{
    int inSz = getInSz(p);
    int mSz = getMSz(p);
    auto d_mulMask = randomGEOnGpu<T>(inSz, p.bw);
    // checkCudaErrors(cudaMemset(d_mulMask, 0, inSz * sizeof(T)));
    auto d_mulMask1 = windowFunc<T, keygenMul<T>>(party, p, d_mask_X, d_mask_M, (u8 *)d_mulMask);
    writeShares<T, T>(key_as_bytes, party, inSz, d_mask_X, p.bw);
    writeShares<T, T>(key_as_bytes, party, mSz, d_mask_M, p.bw);
    writeShares<T, T>(key_as_bytes, party, inSz, d_mulMask1, p.bw);
    gpuFree(d_mulMask1);
    if (d_mask_B)
    {
        auto d_tempMask = windowFunc<T, xPlusM<u64(1), u64(1)>>(party, p, d_mulMask, d_mask_B, NULL, true);
        assert(d_tempMask == d_mulMask);
    }
    // truncate X*M + B as is correct
    auto d_truncateMask = genGPUTruncateKey<T, T>(key_as_bytes, party, t, p.bw, p.bw, p.scale, inSz, d_mulMask, gaes);
    if (d_truncateMask != d_mulMask)
        gpuFree(d_mulMask);
    return d_truncateMask;
}

template <typename T>
T *windowMul(SigmaPeer *peer, int party, MaxpoolParams p, GPUMulKey<T> &k, T *d_X, T *d_M, TruncateType t, AESGlobalContext *gaes, Stats *s, T *d_B = NULL)
{
    auto inSz = getInSz(p);
    auto mSz = getMSz(p);
    auto d_mulKey = (u8 *)moveToGPU((u8 *)k.a, (2 * inSz + mSz) * sizeof(T), s);
    auto d_mulOut = windowFunc<T, beaverMul<T>>(party, p, d_X, d_M, (u8 *)d_mulKey);
    gpuFree(d_mulKey);
    peer->reconstructInPlace(d_mulOut, p.bw, inSz, s);
    if (d_B /*&& party == SERVER1*/)
    {
        auto d_temp = windowFunc<T, xPlusM<u64(1), u64(1)>>(party, p, d_mulOut, d_B, NULL, true);
        assert(d_mulOut == d_temp);
    }
    auto d_truncated_O = gpuTruncate<T, T>(p.bw, p.bw, t, k.trKey, p.scale, peer, party, inSz, d_mulOut, gaes, s);
    if (d_truncated_O != d_mulOut)
        gpuFree(d_mulOut);
    return d_truncated_O;
}