#pragma once

#include "utils/gpu_data_types.h"
#include <cassert>
#include <omp.h>


struct GPUAndKey {
    int N;
    uint32_t *b0, *b1, *b2;
};


GPUAndKey readGPUAndKey(uint8_t** key_as_bytes) {
    GPUAndKey k;
    k.N = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);
    int num_ints = (k.N - 1) / PACKING_SIZE + 1;
    size_t size_in_bytes = num_ints * sizeof(uint32_t);
    k.b0 = (uint32_t*) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b1 = (uint32_t*) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b2 = (uint32_t*) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    return k;
}

#include "gpu_and.cu"