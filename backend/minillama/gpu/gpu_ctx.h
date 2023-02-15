#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

struct GPUContext {
    cudaMemPool_t mempool;
    cudaStream_t stream;
};
