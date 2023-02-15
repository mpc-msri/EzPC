#include <cuda.h>
#include <cuda_runtime.h>
#include "gpu_stats.h"
// #include <sys/types.h>

extern "C" uint8_t* gpuMalloc(size_t size_in_bytes);
extern "C" uint8_t* cpuMalloc(size_t size_in_bytes);
extern "C" void cpuFree(void* h_a);
extern "C" void gpuFree(void* d_a);
extern "C" uint8_t* moveToGPU(uint8_t* h_a, size_t size_in_bytes, Stats*);
extern "C" uint8_t* moveToCPU(uint8_t* d_a, size_t size_in_bytes, Stats*);
extern "C" uint8_t* moveIntoCPUMem(uint8_t* h_a, uint8_t* d_a, size_t size_in_bytes, Stats* s);
