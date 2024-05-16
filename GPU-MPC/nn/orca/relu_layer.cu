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
#include <cstdint>

#include "relu_layer.h"
#include "utils/gpu_mem.h"
#include "utils/misc_utils.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_random.h"

#include "fss/dcf/gpu_dcf.h"
#include "fss/dcf/gpu_relu.h"

namespace dcf
{
    namespace orca
    {
        template <typename T>
        ReluLayer<T>::ReluLayer(int bin, int bout, int numRelus)
        { //, int shift) {
            // assert(bin == bout);
            this->name = "Two Round ReLU";
            this->bin = bin;
            this->bout = bout;
            this->numRelus = numRelus;
            dReluMask = (u8 *)cpuMalloc(numRelus);
            drelu = (u32 *)cpuMalloc(((numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE));
            // printf("Two round ReLU: %d relus\n", numRelus);
        }

        // have we freed memory in this place?
        template <typename T>
        T *ReluLayer<T>::genForwardKey(u8 **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *gaes)
        {
            // auto d_outputMask = randomGEOnGpu<T>(numRelus, bout);
            auto d_oMasks = dcf::gpuGenTwoRoundReluKey(key_as_bytes, party, bin, bout, numRelus, d_inputMask, gaes);
            auto d_dreluMask = d_oMasks.first;
            auto d_reluMask = d_oMasks.second;
            if (this->train)
                moveIntoCPUMem((u8 *)dReluMask, (u8 *)d_dreluMask, numRelus, NULL);
            gpuFree(d_dreluMask);
            // gpuFree(d_inputMask);
            return d_reluMask;
        }

        template <typename T>
        T *ReluLayer<T>::genBackwardKey(u8 **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *gaes, int epoch)
        {
            this->checkIfTrain();
            auto d_dreluMask = (u8 *)moveToGPU((u8 *)dReluMask, numRelus, NULL);
            // auto d_outgoingGradMask = randomGEOnGpu<T>(numRelus, bout);
            auto d_outgoingGradMask = gpuKeyGenSelect<T, T, u8>(key_as_bytes, party, numRelus, d_incomingGradMask, d_dreluMask, bout);
            gpuFree(d_incomingGradMask);
            gpuFree(d_dreluMask);
            return d_outgoingGradMask;
        }

        template <typename T>
        void ReluLayer<T>::readForwardKey(u8 **key_as_bytes)
        {
            reluKey = dcf::readTwoRoundReluKey<T>(key_as_bytes);
        }

        template <typename T>
        void ReluLayer<T>::readBackwardKey(u8 **key_as_bytes, int epoch)
        {
            backpropSelectKey = readGPUSelectKey<T>(key_as_bytes, numRelus);
        }

        template <typename T>
        T *ReluLayer<T>::forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes)
        {
            auto res = dcf::gpuTwoRoundRelu(peer, party, reluKey, d_I, gaes, &(this->s));
            auto d_drelu = res.first;
            auto d_relu = res.second;
            if (this->train)
                moveIntoCPUMem((u8 *)drelu, (u8 *)d_drelu, ((numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE), &(this->s));
            gpuFree(d_drelu);
            // gpuFree(d_I);
            return d_relu;
        }

        template <typename T>
        T *ReluLayer<T>::backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch)
        {
            this->checkIfTrain();
            auto d_drelu = (u32 *)moveToGPU((u8 *)drelu, ((numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE), &(this->s));
            auto d_selectOutput = gpuSelect<T, T, 0, 0>(peer, party, bout, backpropSelectKey, d_drelu, d_incomingGrad, &(this->s));
            gpuFree(d_drelu);
            gpuFree(d_incomingGrad);
            return d_selectOutput;
        }

    }
}