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

#include "gpu_maxpool.h"
#include "gpu_nexp.h"
#include "gpu_lut.h"
#include "gpu_inverse.h"
#include "gpu_window.h"


template <typename T>
struct GPUSoftMaxKey
{
    GPUMaxpoolKey<T> maxPoolKey;
    GPUNExpKey<T> nExpKey;
    GPULUTInverseKey<T> invKey;
    GPUMulKey<T> wMulKey;
};

template <typename T>
GPUSoftMaxKey<T> readGPUSoftMaxKey(MaxpoolParams p, u8 **key_as_bytes)
{
    GPUSoftMaxKey<T> k;
    assert(p.C == 1);
    assert(p.strideH == 1);
    assert(p.strideW == p.FW);

    u64 inSz = getInSz(p);
    u64 mSz = getMSz(p);
    k.maxPoolKey = readGPUMaxpoolKey<T>(p, key_as_bytes);
    k.nExpKey = readGPUNExpKey<T>(key_as_bytes);
    k.invKey = readGPULUTInverseKey<T>(key_as_bytes);
    k.wMulKey = readGPUWindowMulKey<T>(p, TruncateType::TrWithSlack, key_as_bytes);
    return k;
}



#include "gpu_softmax.cu"