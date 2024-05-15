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
