#pragma once

#include "gpu_relu.h"
#include "gpu_lut.h"
#include "gpu_truncate.h"

template <typename T>
struct GPUGeluMuxKey
{
    T *c0, *c1;
};

template <typename T>
GPUGeluMuxKey<T> readGPUGeluMuxKey(uint8_t **key_as_bytes, int N)
{
    GPUGeluMuxKey<T> k;
    u64 memSz = 4 * N * sizeof(T); // num bytes
    k.c0 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    k.c1 = (T *)*key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}

template <typename T, typename TClip>
struct GPUGeluKey
{
    int bw;
    GPUTruncateKey<T> trKey;
    GPUDReluKey dReluKey;
    u32 *icMask;
    GPUGeluMuxKey<TClip> muxKey;
    GPULUTKey<T> lutKey;
    GPUSelectKey<T> reluSelectKey;
};

template <typename T, typename TClip>
GPUGeluKey<T, TClip> readGpuGeluKey(uint8_t **key_as_bytes)
{
    GPUGeluKey<T, TClip> k;
    k.bw = *((int *)*key_as_bytes);
    *key_as_bytes += sizeof(int);
    k.trKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
    k.dReluKey = readGPUDReluKey(key_as_bytes);
    int N = k.trKey.N;
    // printf("###### Gelu N=%d\n", N);
    auto icMaskMemSize = ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    k.icMask = (u32 *)*key_as_bytes;
    *key_as_bytes += icMaskMemSize;
    k.muxKey = readGPUGeluMuxKey<TClip>(key_as_bytes, N);
    k.lutKey = readGPULUTKey<T>(key_as_bytes);
    k.reluSelectKey = readGPUSelectKey<T>(key_as_bytes, N);
    return k;
}


#include "gpu_gelu.cu"