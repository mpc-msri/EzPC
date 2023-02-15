#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"
#include <cassert>
#include "gpu_ctx.h"

extern "C" GPUContext* initialize_gpu_ctx() {
    GPUContext* c = new GPUContext;
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaStreamCreateWithFlags(&(c->stream), cudaStreamNonBlocking));
    int isMemPoolSupported = 0;

    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
                                         cudaDevAttrMemoryPoolsSupported, 0));
    // printf("%d\n", isMemPoolSupported);
    assert(isMemPoolSupported);
    /* implicitly assumes that the device is 0 */    
    checkCudaErrors(cudaDeviceGetDefaultMemPool(&(c->mempool), 0));
    uint64_t threshold = UINT64_MAX;
    checkCudaErrors(cudaMemPoolSetAttribute(c->mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    uint64_t* d_dummy_ptr;
    uint64_t bytes = 15 * (1ULL << 30);
    checkCudaErrors(cudaMallocAsync(&d_dummy_ptr, bytes, c->stream));
    checkCudaErrors(cudaFreeAsync(d_dummy_ptr, c->stream));    
    uint64_t reserved_read, threshold_read;
    checkCudaErrors(cudaMemPoolGetAttribute(c->mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_read));
    checkCudaErrors(cudaMemPoolGetAttribute(c->mempool, cudaMemPoolAttrReleaseThreshold, &threshold_read));
    printf("reserved memory: %lu %lu\n", reserved_read, threshold_read);
    return c;
}
