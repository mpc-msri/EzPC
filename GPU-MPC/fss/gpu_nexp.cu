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

#include "gpu_lut.h"
#include "gpu_truncate.h"
#include "gpu_mul.h"
#include "gpu_relu.h"

template <typename T>
T *gpuKeygenNExp(u8 **key_as_bytes, int party, int bw, int bin, int scale, int N, T *d_mask_X, AESGlobalContext *gaes)
{
    const u64 p = (1ULL << 16) - 1;
    // assert(bin + 1 <= bw);
    // small gap between x and p
    // flip(relu(x - p)) + p
    // this is wrong, can't arbitrarily do bin + 1 whenever you please
    // the input is a 39 bit input
    auto d_clipMask = gpuGenReluKey<T, u16, p, p, true>(key_as_bytes, party, bin, 16, N, d_mask_X, gaes);
    // generate the output in the full bw and scale
    auto d_lsbLutMask = gpuKeyGenLUT<u16, T>(key_as_bytes, party, 8, bw, N, d_clipMask, gaes);
    auto d_msbMask = genGPUTruncateKey<u16, u8>(key_as_bytes, party, TruncateType::TrWithSlack, 16, 8, 8, N, d_clipMask, gaes);
    gpuFree(d_clipMask);
    auto d_msbLutMask = gpuKeyGenLUT<u8, T>(key_as_bytes, party, 8, bw, N, d_msbMask, gaes);
    gpuFree(d_msbMask);
    // clipMask is lsb mask
    auto d_nExpMask = gpuKeygenMul(key_as_bytes, party, bw, scale, N, d_msbLutMask, d_lsbLutMask, TruncateType::TrWithSlack, gaes);
    gpuFree(d_msbLutMask);
    gpuFree(d_lsbLutMask);
    // auto d_nExpMask = genGPUTruncateKey<T, T>(key_as_bytes, party, TruncateType::TrWithSlack, bw, bw, scale, N, d_mulMask, gaes);
    // gpuFree(d_mulMask);
    return d_nExpMask;
}

template <typename T>
T *gpuNExp(SigmaPeer* peer, int party, int bw, int bin, int scale, int N, GPUNExpKey<T> k, T *d_X, T* d_nExpMsbTab, T* d_nExpLsbTab, AESGlobalContext *gaes, Stats *s)
{
    const u64 p = (1ULL << 16) - 1;
    auto d_clippedX = gpuRelu<T, u16, p, p, true>(peer, party, k.reluKey, d_X, gaes, s);
    // printf("Starting LSB LUT=%d, %d\n", N, k.N);
    auto d_lsbLookup = gpuDpfLUT<u16, T>(k.lsbLutKey, peer, party, d_clippedX, d_nExpLsbTab, gaes, s);
    auto d_msb = gpuTruncate<u16, u8>(16, 8, TruncateType::TrWithSlack, k.trKey, 8, peer, party, k.N, d_clippedX, gaes, s);
    gpuFree(d_clippedX);
    // printf("Starting MSB LUT\n");
    auto d_msbLookup = gpuDpfLUT<u8, T>(k.msbLutKey, peer, party, d_msb, d_nExpMsbTab, gaes, s);
    gpuFree(d_msb);
    // don't add comm here?
    auto d_nExp = gpuMul(peer, party, bw, scale, k.N, k.mulKey, d_msbLookup, d_lsbLookup, TruncateType::TrWithSlack, gaes, NULL);
    gpuFree(d_lsbLookup);
    gpuFree(d_msbLookup);
    // printf("N=%d\n", N);
    // auto d_nExp = gpuTruncate<T, T>(bw, bw, TruncateType::TrWithSlack, k.mulTrKey, scale, peer, party, N, d_nExpMul, gaes, s);
    // gpuFree(d_nExpMul);
    return d_nExp;
}

// auto d_dReluMask = gpuKeyGenDRelu(key_as_bytes, party, bin, N, d_mask_X, gaes);
// writeShares(key_as_bytes, party, N, d_dReluMask, 1);
// auto d_clipMask = randomGEOnGpu<T>(u16, 16);
// gpuLinearComb(16, N, d_clipMask, u16(1), d_clipMask, u16(-1));
// reconstruct = true
// gpuKeyGenSelect<T, u16>(key_as_bytes, party, N, d_mask_X, d_dReluMask, d_clipMask, 16);
