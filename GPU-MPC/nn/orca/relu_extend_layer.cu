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

#include <cassert>
#include <omp.h>

#include "utils/gpu_mem.h"
#include "utils/misc_utils.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_random.h"

#include "relu_extend_layer.h"

namespace dcf
{
    namespace orca
    {
        template <typename T>
        ReluExtendLayer<T>::ReluExtendLayer(int bin, int bout, int numRelus)
        {
            this->name = "ReLU Extend";
            this->bin = bin;
            this->bout = bout;
            this->numRelus = numRelus;
            dReluMask = (u8 *)cpuMalloc(numRelus);
            drelu = (u32 *)cpuMalloc(((numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE));
        }

        template <typename T>
        ReluExtendLayer<T>::~ReluExtendLayer()
        {
            cpuFree(dReluMask);
            cpuFree(drelu);
        }

        // have we feed memory in this place?
        template <typename T>
        T *ReluExtendLayer<T>::genForwardKey(u8 **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *gaes)
        {
            auto res = gpuKeygenReluExtend(key_as_bytes, party, bin, bout, numRelus, d_inputMask, gaes);
            auto d_dreluMask = res.first;
            auto d_randomOutMask = res.second;
            if (this->train)
                moveIntoCPUMem((u8 *)dReluMask, (u8 *)d_dreluMask, numRelus, NULL);
            gpuFree(d_dreluMask);
            return d_randomOutMask;
        }

        template <typename T>
        T *ReluExtendLayer<T>::genBackwardKey(u8 **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *gaes, int epoch)
        {
            this->checkIfTrain();
            auto d_dreluMask = (u8 *)moveToGPU((u8 *)dReluMask, numRelus, NULL);
            auto d_outgoingGradMask = gpuKeyGenSelect<T, T, u8>(key_as_bytes, party, numRelus, d_incomingGradMask, d_dreluMask, bout);
            gpuFree(d_incomingGradMask);
            gpuFree(d_dreluMask);
            return d_outgoingGradMask;
        }

        template <typename T>
        void ReluExtendLayer<T>::readForwardKey(u8 **key_as_bytes)
        {
            reluExtendKey = readGPUReluExtendKey<T>(key_as_bytes);
        }

        template <typename T>
        void ReluExtendLayer<T>::readBackwardKey(u8 **key_as_bytes, int epoch)
        {
            backpropSelectKey = readGPUSelectKey<T>(key_as_bytes, numRelus);
        }

        // no memory leak
        template <typename T>
        T *ReluExtendLayer<T>::forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes)
        {
            auto res = gpuReluExtend(peer, party, reluExtendKey, d_I, gaes, &(this->s));
            auto d_dcf = res.first;
            auto d_drelu = d_dcf;
            auto d_xLTRin = (u32 *)(((u8 *)d_dcf) + reluExtendKey.dReluKey.dcfKey.memSzOut);
            auto d_O = res.second;
            if (this->train)
                moveIntoCPUMem((u8 *)drelu, (u8 *)d_xLTRin, ((numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE), &(this->s));
            gpuFree(d_drelu);
            return d_O;
        }

        // no memory leak
        template <typename T>
        T *ReluExtendLayer<T>::backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch)
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