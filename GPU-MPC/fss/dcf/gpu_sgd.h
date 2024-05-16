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

#include "fss/dcf/gpu_truncate.h"

namespace dcf
{
    namespace orca
    {
        static const uint64_t lr_fp = 1;
        static const int lr_scale[5] = {6, 6, 6, 9, 9};
        static const uint64_t mom_fp = 29;
        static const int mom_scale = 5;
    }

    template <typename T>
    void genOptimizerKey(uint8_t **key_as_bytes, int party, int bin, int bout, int N, T *h_W, T *d_W,
                         T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t, bool useMomentum, AESGlobalContext *gaes, int epoch);

    template <typename T>
    void readOptimizerKey(TruncateType t, GPUSignExtendKey<T> *truncateKeyVw, GPUSignExtendKey<T> *truncateKeyW, uint8_t **key_as_bytes, int scaleW, int scaleVw, int scaledW, bool useMomentum, int epoch);

    template <typename T>
    void optimize(int bin, int bout, int N, T *h_W, T *d_W,
                  T *h_Vw, T *d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t,
                  GPUSignExtendKey<T> truncateKeyVw, GPUSignExtendKey<T> truncateKeyW, int party, Peer *peer, bool useMomentum, AESGlobalContext *gaes, Stats *s, int epoch);

    template <typename T>
    void checkOptimizer(int bin, int bout, int N,
                        T *h_W, T *h_Vw, T *h_dW,
                        T *h_masked_W, T *h_masked_Vw,
                        T *h_mask_W, T *h_mask_Vw,
                        int scaleW, int scaleVw, int scaledW, bool useMomentum, int epoch);

}

#include "gpu_sgd.cu"
