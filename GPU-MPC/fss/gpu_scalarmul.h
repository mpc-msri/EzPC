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

#include "utils/gpu_data_types.h"
#include "gpu_truncate.h"

// Z = aX, where a is a public scalar
template <typename T>
T *gpuKeygenScalarMul(u8 **key_as_bytes, int party, int bw, int N, T a, T *d_mask_X, TruncateType t, int shift, AESGlobalContext *gaes)
{
    auto d_mask_Z = (T *)gpuMalloc(N * sizeof(T));
    gpuLinearComb(bw, N, d_mask_Z, a, d_mask_X);
    auto d_mask_truncated_Z = genGPUTruncateKey<T, T>(key_as_bytes, party, t, bw, bw, shift, N, d_mask_X, gaes);
    if (d_mask_truncated_Z != d_mask_Z)
        gpuFree(d_mask_Z);
    return d_mask_truncated_Z;
}

template <typename T>
T *gpuScalarMul(SigmaPeer *peer, int party, int bw, int N, GPUTruncateKey<T> k, T a, T *d_X, TruncateType t, int shift, AESGlobalContext *gaes, Stats *s)
{
    u64 b0 = peer->bytesSent() + peer->bytesReceived();
    auto d_Z = (T *)gpuMalloc(N * sizeof(T));
    gpuLinearComb(bw, N, d_Z, a, d_X);
    auto d_truncated_Z = gpuTruncate<T, T>(bw, bw, t, k, shift, peer, party, N, d_Z, gaes, s);
    if (d_truncated_Z != d_Z)
        gpuFree(d_Z);
    u64 b1 = peer->bytesSent() + peer->bytesReceived();
    s->linear_comm_bytes += (b1 - b0);
    return d_truncated_Z;
}
