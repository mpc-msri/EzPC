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

#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_stats.h"
// #include <sys/types.h>
// extern cudaMemPool_t mempool;

extern "C" uint8_t *gpuMalloc(size_t size_in_bytes);
extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin = true);
extern "C" void cpuFree(void *h_a, bool pinned = true);
extern "C" void gpuFree(void *d_a);
extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *);
extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s);
extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *);
extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s);
extern "C" void initGPUMemPool();