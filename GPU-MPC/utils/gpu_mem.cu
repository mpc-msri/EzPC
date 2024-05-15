#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"
#include "gpu_stats.h"
#include <cassert>

// #include <sys/types.h>

cudaMemPool_t mempool;

extern "C" void initGPUMemPool()
{
    int isMemPoolSupported = 0;
    int device = 0;
    // is it okay to use device=0?
    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
                                           cudaDevAttrMemoryPoolsSupported, device));
    // printf("%d\n", isMemPoolSupported);
    assert(isMemPoolSupported);
    /* implicitly assumes that the device is 0 */

    checkCudaErrors(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    checkCudaErrors(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    uint64_t *d_dummy_ptr;
    uint64_t bytes = 40 * (1ULL << 30);
    checkCudaErrors(cudaMallocAsync(&d_dummy_ptr, bytes, 0));
    checkCudaErrors(cudaFreeAsync(d_dummy_ptr, 0));
    uint64_t reserved_read, threshold_read;
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_read));
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold_read));
    printf("reserved memory: %lu %lu\n", reserved_read, threshold_read);
}

extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    checkCudaErrors(cudaMallocAsync(&d_a, size_in_bytes, 0));
    return d_a;
}


extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin)
{
    uint8_t *h_a;
    int err = posix_memalign((void **)&h_a, 32, size_in_bytes);
    assert(err == 0 && "posix memalign");
    if (pin)
        checkCudaErrors(cudaHostRegister(h_a, size_in_bytes, cudaHostRegisterDefault));
    return h_a;
}

extern "C" void gpuFree(void *d_a)
{
    checkCudaErrors(cudaFreeAsync(d_a, 0));
}

extern "C" void cpuFree(void *h_a, bool pinned)
{
    if (pinned)
        checkCudaErrors(cudaHostUnregister(h_a));
    free(h_a);
}

extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *h_a = cpuMalloc(size_in_bytes, true);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *d_a = gpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return d_a;
}
