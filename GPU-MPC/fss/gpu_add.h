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

#include "utils/gpu_data_types.h"

template <typename T> 
__global__ void addManyArrays(int bw, int numPtrs, int N, T** ptrs, T* out) { 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    if (j < N) { 
        out[j] = T(0); 
        for(int i = 0; i < numPtrs; i++) out[j] += ptrs[i][j]; 
        gpuMod(out[j], bw); 
        // if(j < 3) printf("Add %d, %d=%ld, %ld, %ld\n", N, j, out[j], ptrs[0][j], ptrs[1][j]);
    } 
}

template <typename T>
T* gpuAdd(int bw, int N, std::vector<T*> &h_dPtrsOnHost) { 
    int numDPtrs = h_dPtrsOnHost.size();
    T** d_dPtrs = (T**) moveToGPU((uint8_t*) h_dPtrsOnHost.data(), numDPtrs * sizeof(T*), NULL); 
    T* d_out = (T*) gpuMalloc(N * sizeof(T)); 
    addManyArrays<<<(N - 1) / 128 + 1, 128>>>(bw, numDPtrs, N, d_dPtrs, d_out); 
    checkCudaErrors(cudaDeviceSynchronize()); 
    return d_out; 
}
