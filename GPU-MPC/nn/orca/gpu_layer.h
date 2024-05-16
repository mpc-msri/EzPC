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

#include "fss/gpu_aes_shm.h"
#include "utils/gpu_comms.h"
#include <cassert>
#include <vector>
// #include <mutex>
// #include <condition_variable>
namespace dcf
{
    namespace orca
    {

        template <typename T>
        class GPULayer
        {
        public:
            std::string name = "";
            bool train = false;
            bool useMomentum = true;
            bool loadedWeights = false;
            Stats s;
            virtual void setTrain(bool useMomentum)
            {
                train = true;
                this->useMomentum = useMomentum;
            }
            void checkIfTrain()
            {
                assert(train && "train is not set!");
            }

            virtual T *genForwardKey(u8 **key_as_bytes, int party, T *d_inputMask, AESGlobalContext *g) = 0;
            virtual T *genBackwardKey(u8 **key_as_bytes, int party, T *d_incomingGradMask, AESGlobalContext *g, int epoch) = 0;
            virtual void readForwardKey(u8 **key_as_bytes) = 0;
            virtual void readBackwardKey(u8 **key_as_bytes, int epoch) = 0;
            virtual T *forward(SigmaPeer *peer, int party, T *d_I, AESGlobalContext *g) = 0;
            virtual T *backward(SigmaPeer *peer, int party, T *d_incomingGrad, AESGlobalContext *g, int epoch) = 0;
            virtual void initWeights(u8 **weights, bool floatWeights) {}
            virtual void dumpWeights(std::ofstream &f) {}
        };

    }
}