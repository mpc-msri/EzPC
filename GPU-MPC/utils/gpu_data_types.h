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

// #ifndef GPU_DATA_TYPES_H
// #define GPU_DATA_TYPES_H

#pragma once

#include <utility>
#include <stdint.h>
#include <cstddef>

#include <sytorch/tensor.h>

#include "gpu_stats.h"

typedef unsigned __int128 AESBlock;

#define SERVER0 0
#define SERVER1 1
#define AES_BLOCK_LEN_IN_BITS 128
#define FULL_MASK 0xffffffff
#define LOG_AES_BLOCK_LEN 7

#define PACKING_SIZE 32
#define PACK_TYPE uint32_t

#define NUM_SHARED_MEM_BANKS 32

using orcaTemplateClass = u64;

namespace dcf
{
    namespace orca
    {
        namespace global
        {
            static const int bw = 64;
            static const int scale = 24;
        }
    }
}