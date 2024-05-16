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

#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "fss/gpu_avgpool.h"
#include "fss/dcf/gpu_truncate.h"

#include "avg_pool_layer.h"

namespace dcf
{
    namespace orca
    {
        template <typename T>
        AvgPool2DLayer<T>::AvgPool2DLayer(int bin, int bout, int scaleDiv, int N, int imgH, int imgW, int C, int FH, int FW, int strideH,
                                          int strideW, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, TruncateType tf, TruncateType tb)
        {
            assert(bin == bout);
            // assert(bin == sizeof(T) * 8);
            assert(zPadHLeft == zPadHRight && zPadWLeft == zPadWRight && zPadHLeft == zPadWLeft && zPadHLeft == 0 && "padding is not supported in avgpool!");
            this->name = "AvgPool2D";
            p = {bin, bin, 0, scaleDiv, bin, N, imgH, imgW, C, FH, FW, strideH, strideW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, 0, 0, false};
            assert(imgH >= FH);
            assert(imgW >= FW);
            assert(zPadHLeft == 0 && zPadHRight == 0 && zPadWLeft == 0 && zPadWRight == 0);
            initPoolParams(p);
            inSz = getInSz(p);
            outSz = getMSz(p);
            this->tf = tf;
            this->tb = tb;
        }

        template <typename T>
        T *AvgPool2DLayer<T>::genForwardKey(u8 **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *gaes)
        {
            auto d_mask_O = gpuAddPool(p, d_inputMask, &(this->s));
            // gpuFree(d_inputMask);
            auto d_mask_truncated_O = dcf::genGPUTruncateKey(key_as_bytes, party, tf, p.bw, p.bw, p.scaleDiv, outSz, d_mask_O, gaes); /*, mask_truncated_C);*/
            return d_mask_truncated_O;
        }

        template <typename T>
        T *AvgPool2DLayer<T>::genBackwardKey(u8 **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *gaes, int epoch)
        {
            auto d_mask_dI = gpuAddPoolBackProp(p, d_incomingGradMask, &(this->s));
            gpuFree(d_incomingGradMask);
            auto d_mask_truncated_dI = dcf::genGPUTruncateKey(key_as_bytes, party, tb, p.bw, p.bw, p.scaleDiv, inSz, d_mask_dI, gaes); /*, mask_truncated_C);*/
            return d_mask_truncated_dI;
        }

        template <typename T>
        void AvgPool2DLayer<T>::readForwardKey(u8 **key_as_bytes)
        {
            truncateKeyF = dcf::readGPUTruncateKey<T>(tf, key_as_bytes);
        }

        template <typename T>
        void AvgPool2DLayer<T>::readBackwardKey(u8 **key_as_bytes, int epoch)
        {
            truncateKeyB = dcf::readGPUTruncateKey<T>(tb, key_as_bytes);
        }

        template <typename T>
        T *AvgPool2DLayer<T>::forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g)
        {
            auto d_O = gpuAddPool(p, d_I, &(this->s));
            // gpuFree(d_I);
            dcf::gpuTruncate(p.bw, p.bw, tf, truncateKeyF, p.scaleDiv, peer, party, outSz, d_O, g, &(this->s));
            return d_O;
        }

        template <typename T>
        T *AvgPool2DLayer<T>::backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch)
        {
            auto d_dI = gpuAddPoolBackProp(p, d_incomingGrad, &(this->s));
            gpuFree(d_incomingGrad);
            dcf::gpuTruncate(p.bw, p.bw, tb, truncateKeyB, p.scaleDiv, peer, party, inSz, d_dI, g, &(this->s));
            return d_dI;
        }
    }
}
