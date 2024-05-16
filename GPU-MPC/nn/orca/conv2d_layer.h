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

#include "utils/gpu_stats.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_conv2d.h"
#include "fss/dcf/gpu_truncate.h"

#include "gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class Conv2DLayer : public GPULayer<T>
        {

        private:
            void initConvKey();
            void initConvKeydI();
            void initConvKeydF();

        public:
            Conv2DParams p;
            GPUConv2DKey<T> convKey, convKeydI, convKeydF;
            bool inputIsShares;
            T *I, *F, *Vf, *b, *Vb;
            // Stats s;
            bool useBias, computedI;
            TruncateType tf, tb;
            GPUTruncateKey<T> truncateKeyC, truncateKeydI, truncateKeyF, truncateKeyVf, truncateKeyb, truncateKeyVb;

            // using these variables for keygen
            T *mask_I, *d_mask_I, *mask_F, *mask_Vf, *mask_b, *mask_Vb;

            Conv2DLayer(int bin, int bout, int N, int H, int W, int CI, int FH, int FW, int CO,
                        int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, int strideH, int strideW, bool useBias, dcf::TruncateType tf, dcf::TruncateType tb, bool computedI, bool inputIsShares);
            T *genForwardKey(u8 **key_as_bytes, int party, T *mask_I, AESGlobalContext *gaes);
            T *genBackwardKey(u8 **key_as_bytes, int party, T *mask_grad, AESGlobalContext *gaes, int epoch);
            void readForwardKey(u8 **key_as_bytes);
            void readBackwardKey(u8 **key_as_bytes, int epoch);
            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes);
            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *gaes, int epoch);
            void initWeights(u8 **weights, bool floatWeights);
            void dumpWeights(std::ofstream &f);
        };
    }
}
#include "conv2d_layer.cu"
