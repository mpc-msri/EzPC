#pragma once

#include "utils/gpu_data_types.h"
#include "gpu_truncate.h"

template <typename T>
struct GPUMulKey {
    u64 szA, szB, szC;
    T *a, *b, *c;
    GPUTruncateKey<T> trKey;
};

template <typename T>
GPUMulKey<T> readGPUMulKey(u8** key_as_bytes, u64 szA, u64 szB, u64 szC, TruncateType t) {
    // printf("Inside mul key, N=%lu, %lu, %lu, %lx\n", szA, szB, szC, *key_as_bytes);
    GPUMulKey<T> k;
    k.szA = szA;
    k.szB = szB;
    k.szC = szC;
    k.a = (T*) *key_as_bytes;
    // printf("a=%ld\n", *k.a);
    *key_as_bytes += (szA * sizeof(T));
    k.b = (T*) *key_as_bytes;
    // printf("b=%ld\n", *k.b);
    *key_as_bytes += (szB * sizeof(T));
    k.c = (T*) *key_as_bytes;
    // printf("c=%ld\n", *k.c);
    *key_as_bytes += (szC * sizeof(T));
    printf("Reading truncate key######\n");
    k.trKey = readGPUTruncateKey<T>(/*TruncateType::TrWithSlack*/t, key_as_bytes);
    return k;
}

#include "gpu_mul.cu"