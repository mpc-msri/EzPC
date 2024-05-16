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

#include "gpu_data_types.h"
#include <curand.h>


template <typename T>
T *randomGEOnGpu(const u64 n, int bw);
template <typename T>
void randomGEOnCpu(const u64 n, int bw, T *h_data);
template <typename T>
T *randomGEOnCpu(const u64 n, int bw);
AESBlock *randomAESBlockOnGpu(const int n);
void initGPURandomness();
void destroyGPURandomness();
void initCPURandomness();
void destroyCPURandomness();
template <typename T>
T *getMaskedInputOnGpu(int N, int bw, T *d_mask_I, T **h_I, bool smallInputs = false, int smallBw = 0);
template <typename T>
T *getMaskedInputOnCpu(int N, int bw, T *h_mask_I, T **h_I, bool smallInputs = false, int smallBw = 0);
template <typename TIn, typename TOut>
void writeShares(u8 **key_as_bytes, int party, u64 N, TIn *d_A, int bw, bool randomShares = true);


#include "gpu_random.cu"