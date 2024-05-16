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

#include "utils/gpu_stats.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_matmul.h"
#include "fss/dcf/gpu_truncate.h"

#include "gpu_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        class FCLayer : public GPULayer<T>
        {
        private:
            void initMulParamsdW();
            void initMulParamsdX();
            void initMemSz(MatmulParams p, GPUMatmulKey<T> *k)
            {
                k->mem_size_A = p.size_A * sizeof(T);
                k->mem_size_B = p.size_B * sizeof(T);
                k->mem_size_C = p.size_C * sizeof(T);
            }

        public:
            MatmulParams p, pdW, pdX;
            GPUMatmulKey<T> mmKey, mmKeydX, mmKeydW;
            // Z = XW + Y
            // X = N * something, which means that Y is a row vector
            T *X, *W, *Y, *Vw, *Vy;
            T *mask_X, *mask_W, *mask_Z, *mask_Y = NULL, *mask_dX, *mask_dW, *mask_dY = NULL, *mask_Vw, *mask_Vy = NULL;
            dcf::TruncateType tf, tb;
            GPUTruncateKey<T> truncateKeyZ, truncateKeydX, truncateKeyW, truncateKeyVw, truncateKeyY, truncateKeyVy;
            // Stats s;
            bool useBias;
            bool computedX;
            bool inputIsShares;

            FCLayer(int bin, int bout, int M, int N, int K, dcf::TruncateType tf, dcf::TruncateType tb, bool useBias, bool computedX, bool inputIsShares);
            T *genForwardKey(uint8_t **key_as_bytes, int party, T *d_mask_X, AESGlobalContext *gaes);
            T *genBackwardKey(uint8_t **key_as_bytes, int party, T *d_mask_grad, AESGlobalContext *gaes, int epoch);
            void readForwardKey(uint8_t **key_as_bytes);
            void readBackwardKey(uint8_t **key_as_bytes, int epoch);
            T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes);
            T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *gaes, int epoch);
            void initWeights(uint8_t **weights, bool floatWeights);
            void dumpWeights(std::ofstream &f);
        };
    }
}

#include "fc_layer.cu"