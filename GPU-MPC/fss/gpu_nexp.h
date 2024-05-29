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

#include "gpu_lut.h"
#include "gpu_truncate.h"
#include "gpu_mul.h"
#include "gpu_relu.h"

template <typename T>
struct GPUNExpKey
{
    int N;
    GPUReluKey<u16> reluKey;
    GPULUTKey<T> lsbLutKey;
    GPUTruncateKey<u8> trKey;
    GPULUTKey<T> msbLutKey;
    GPUMulKey<T> mulKey;
    // GPUTruncateKey<T> mulTrKey;
};

template <typename T>
GPUNExpKey<T> readGPUNExpKey(u8 **key_as_bytes)
{
    GPUNExpKey<T> k;
    k.reluKey = readReluKey<u16>(key_as_bytes);
    k.N = k.reluKey.numRelus;
    k.lsbLutKey = readGPULUTKey<T>(key_as_bytes);
    k.trKey = readGPUTruncateKey<u8>(TruncateType::TrWithSlack, key_as_bytes);
    k.msbLutKey = readGPULUTKey<T>(key_as_bytes);
    k.mulKey = readGPUMulKey<T>(key_as_bytes, (u64)k.N, (u64)k.N, (u64)k.N, TruncateType::TrWithSlack);
    return k;
}

#include "gpu_nexp.cu"