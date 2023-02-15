#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"
#include "gpu_stats.h"

// #include <sys/types.h>

extern "C" uint8_t* gpuMalloc(size_t size_in_bytes) {
    uint8_t* d_a;
    checkCudaErrors(cudaMalloc(&d_a, size_in_bytes));
    return d_a;
}

extern "C" uint8_t* cpuMalloc(size_t size_in_bytes) {
    uint8_t* h_a;
    checkCudaErrors(cudaMallocHost(&h_a, size_in_bytes));
    return h_a;
}

extern "C" void gpuFree(void* d_a) {
    checkCudaErrors(cudaFree(d_a));
}


extern "C" void cpuFree(void* h_a) {
    checkCudaErrors(cudaFreeHost(h_a));
}


extern "C" uint8_t* moveToCPU(uint8_t* d_a, size_t size_in_bytes, Stats* s) {
    uint8_t* h_a = cpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();  
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();  
    auto elapsed = end - start;
    if(s) s->transfer_time += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    // cout << "move to cpu " << s->transfer_time << endl;
    return h_a;
}

extern "C" uint8_t* moveIntoCPUMem(uint8_t* h_a, uint8_t* d_a, size_t size_in_bytes, Stats* s) {
    // uint8_t* h_a = cpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();  
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();  
    auto elapsed = end - start;
    if(s) s->transfer_time += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    // cout << "move to cpu " << s->transfer_time << endl;
    return h_a;
}

extern "C" uint8_t* moveToGPU(uint8_t* h_a, size_t size_in_bytes, Stats* s) {
    uint8_t* d_a = gpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();  
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();  
    auto elapsed = end - start;
    if(s) s->transfer_time += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    return d_a;
}
