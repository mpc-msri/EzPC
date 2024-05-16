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

#include "utils/helper_cuda.h"
#include "utils/gpu_mem.h"
#include "utils/misc_utils.h"
#include "utils/gpu_file_utils.h"

#include "maxpool_layer.h"

namespace dcf
{
    namespace orca
    {

        template <typename T>
        MaxPool2DLayer<T>::MaxPool2DLayer(int bin, int bout, int bwBackprop, int N, int imgH, int imgW, int C, int FH, int FW, int strideH,
                                          int strideW, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight)
        {
            assert(bin == bout);
            // assert(zPadHLeft == zPadHRight && zPadWLeft == zPadWRight && zPadHLeft == zPadWLeft && zPadHLeft == 0 && "padding is not supported in maxpool!");
            this->name = "MaxPool2D";
            p.bw = bin;
            p.bin = bin;
            // p.bout = bout;
            p.bwBackprop = bwBackprop;
            p.N = N;
            p.imgH = imgH;
            p.imgW = imgW;
            p.C = C;
            p.FH = FH;
            p.FW = FW;
            assert(imgH >= FH);
            assert(imgW >= FW);
            p.strideH = strideH;
            p.strideW = strideW;
            // no padding for now
            //  assert(zPadHLeft == 0 && zPadHRight == 0 && zPadWLeft == 0 && zPadWRight == 0);
            p.zPadHLeft = zPadHLeft;
            p.zPadHRight = zPadHRight;
            p.zPadWLeft = zPadWLeft;
            p.zPadWRight = zPadWRight;
            initPoolParams(p);
            int outSz = p.N * p.H * p.W * p.C;
            oneHotOutputMask = (u8 *)cpuMalloc(p.FH * p.FW * outSz);
            int numInts = ((outSz * p.FH * p.FW - 1) / PACKING_SIZE + 1);
            oneHot = (u32 *)cpuMalloc(numInts * sizeof(u32));
        }

        template <typename T>
        MaxPool2DLayer<T>::~MaxPool2DLayer()
        {
            cpuFree(oneHotOutputMask);
            cpuFree(oneHot);
        }

        template <typename T>
        T *MaxPool2DLayer<T>::genForwardKey(uint8_t **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *gaes)
        {
            int outSz = p.N * p.H * p.W * p.C;
            // T* d_curMax = (T*) gpuMalloc(outSz * sizeof(T));
            size_t oneHotSize = outSz * p.FH * p.FW;
            u8 *d_oneHotMask = NULL;
            if (this->train)
            {
                d_oneHotMask = (u8 *)gpuMalloc(oneHotSize);
                checkCudaErrors(cudaMemset(d_oneHotMask, 0, oneHotSize));
            }
            auto d_maxMask = gpuKeygenMaxpool(key_as_bytes, party, p, d_inputMask, d_oneHotMask, gaes);
            // gpuFree(d_inputMask);
            if (this->train)
            {
                moveIntoCPUMem((u8 *)oneHotOutputMask, (u8 *)d_oneHotMask, oneHotSize, NULL);
                gpuFree(d_oneHotMask);
            }
            return d_maxMask;
        }

        template <typename T>
        T *MaxPool2DLayer<T>::genBackwardKey(uint8_t **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *gaes, int epoch)
        {
            this->checkIfTrain();
            size_t oneHotSize = p.N * p.H * p.W * p.C * p.FH * p.FW;
            auto d_oneHotMask = (u8 *)moveToGPU((u8 *)oneHotOutputMask, oneHotSize, NULL);
            auto d_outgoingGradMask = keyGenMaxpoolBackProp(key_as_bytes, party, p, d_oneHotMask, d_incomingGradMask);
            gpuFree(d_oneHotMask);
            gpuFree(d_incomingGradMask);
            return d_outgoingGradMask;
        }

        template <typename T>
        void MaxPool2DLayer<T>::readForwardKey(uint8_t **key_as_bytes)
        {
            // maxpoolKey.p = p;
            // gaes = g;
            maxpoolKey.reluKey = new GPU2RoundReLUKey<T>[p.FH * p.FW];
            maxpoolKey.andKey = new GPUAndKey[p.FH * p.FW];
            for (int i = 0; i < p.FH; i++)
            {
                for (int j = 0; j < p.FW; j++)
                {
                    if (i == 0 && j == 0)
                        continue;
                    maxpoolKey.reluKey[i * p.FW + j] = readTwoRoundReluKey<T>(key_as_bytes);
                    if (this->train)
                        maxpoolKey.andKey[i * p.FW + j] = readGPUAndKey(key_as_bytes);
                }
            }
        }

        template <typename T>
        void MaxPool2DLayer<T>::readBackwardKey(uint8_t **key_as_bytes, int epoch)
        {
            int numSelects = p.N * p.H * p.W * p.C * p.FH * p.FW;
            backpropSelectKey = readGPUSelectKey<T>(key_as_bytes, numSelects);
        }

        // no memory leak
        template <typename T>
        T *MaxPool2DLayer<T>::forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *gaes)
        {
            int outSz = getMSz(p);
            // T* d_curMax = (T*) gpuMalloc(outSz * sizeof(T));
            u32 *d_oneHot = NULL;
            int numInts = ((outSz * p.FH * p.FW - 1) / PACKING_SIZE + 1);
            if (this->train)
            {
                d_oneHot = (u32 *)gpuMalloc(numInts * sizeof(u32));
            }
            auto d_curMax = gpuMaxPool(peer, party, p, maxpoolKey, d_I, d_oneHot, gaes, &(this->s));
            if (this->train)
            {
                moveIntoCPUMem((uint8_t *)oneHot, (uint8_t *)d_oneHot, numInts * sizeof(u32), &(this->s));
                gpuFree(d_oneHot);
            }
            return d_curMax;
        }

        // no memory leak
        template <typename T>
        T *MaxPool2DLayer<T>::backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch)
        {

            this->checkIfTrain();
            int incomingGradSize = getMSz(p);
            int outgoingGradSize = getInSz(p);
            int oneHotSize = incomingGradSize * p.FH * p.FW;
            int numInts = (oneHotSize - 1) / PACKING_SIZE + 1;

            auto d_oneHot = (u32 *)moveToGPU((uint8_t *)oneHot, numInts * sizeof(u32), &(this->s));
            auto d_outgoingGradExpanded = gpuSelectForMaxpoolBackprop(p, backpropSelectKey, d_oneHot, d_incomingGrad,
                                                                      party, &(this->s));
            gpuFree(d_incomingGrad);
            auto d_outgoingGrad = gpuCollectGradients(p, d_outgoingGradExpanded, &(this->s));
            peer->reconstructInPlace(d_outgoingGrad, p.bwBackprop, outgoingGradSize, &(this->s));
            gpuFree(d_oneHot);
            gpuFree(d_outgoingGradExpanded);
            return d_outgoingGrad;
        }
    }
}