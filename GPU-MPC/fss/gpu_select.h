#pragma once

#include "utils/gpu_data_types.h"

template <typename T>
struct GPUSelectKey
{
    int N;
    T *a, *b, *c, *d1, *d2;
};

template <typename T>
GPUSelectKey<T> readGPUSelectKey(uint8_t** key_as_bytes, int N) {
    GPUSelectKey<T> k;
    k.N = N;

    size_t size_in_bytes = N * sizeof(T);

    k.a = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.c = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d1 = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d2 = (T *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    return k;
}


// template <typename T>
// GPUSelectKey<T> readGPUSelectKey(uint8_t **key_as_bytes, int N);

#include "gpu_select.cu"