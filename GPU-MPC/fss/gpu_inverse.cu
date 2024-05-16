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

#include "gpu_inverse.h"

template <typename T>
T* gpuKeygenLUTInverse(u8** key_as_bytes, int party, int bw, int bin, int scale, int N, T* d_mask_X, AESGlobalContext* gaes) {
    // only consider the last 16 bits of the input
    assert(bin <= bw);
    assert(bin - 6 <= 16);
    assert(scale == 12);
    // assert(bin > 16);
    // printf("here %d, %d\n", bin, scale);
    auto d_trMask = genGPUTruncateKey<T, u16>(key_as_bytes, party, TruncateType::TrWithSlack, bin, bin - 6, /*std::max(bin - 13, 0)*/6, N, d_mask_X, gaes);
    // printf("here inside LUT, Bin=%d\n", );
    auto d_invMask = gpuKeyGenLUT<u16, T>(key_as_bytes, party, /*13*/bin - 6, bw, N, d_trMask, gaes);
    // printf("here\n");
    gpuFree(d_trMask);
    return d_invMask;
}

template <typename T>
T* gpuLUTInverse(SigmaPeer* peer, int party, int bw, int bin, int scale, int N, GPULUTInverseKey<T> k, T* d_X, T* d_invTab, AESGlobalContext* gaes, Stats* s) {
    assert(bin - 6 <= 16);
    assert(scale == 12);
    auto d_truncated_X = gpuTruncate<T, u16>(bin, bin - 6, TruncateType::TrWithSlack, k.trKey, /*std::max(bin - 13, 0)*/6, peer, party, k.N, d_X, gaes, s);
    auto d_invX = gpuDpfLUT<u16, T>(k.lutKey, peer, party, d_truncated_X, d_invTab, gaes, s);
    gpuFree(d_truncated_X);
    return d_invX;
}

