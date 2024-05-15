#pragma once

#include "utils/gpu_data_types.h"

#include "gpu_select.h"
#include "gpu_dpf.h"

// using GPUDReluKey = GPUMaskedDCFKey;
using u32 = uint32_t;

struct GPUDReluKey
{
    GPUDPFKey dpfKey;
    u32 *mask;
};

template <typename T>
struct GPUReluKey
{
    int bin, bout, numRelus;
    GPUDReluKey dreluKey;
    GPUSelectKey<T> selectKey;
};

GPUDReluKey readGPUDReluKey(uint8_t **key_as_bytes)
{
    GPUDReluKey k;
    k.dpfKey = readGPUDPFKey(key_as_bytes);
    int N = k.dpfKey.M;
    k.mask = (uint32_t *)*key_as_bytes;
    // number of 32-bit integers * sizeof(int)
    // only works for bout = 1
    *key_as_bytes += ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

// const auto readGPUDReluWithDCFKey = readGPUMaskedDCFKey;

template <typename T>
GPUReluKey<T> readReluKey(uint8_t **key_as_bytes)
{
    GPUReluKey<T> k;
    memcpy(&k, *key_as_bytes, 3 * sizeof(int));
    *key_as_bytes += 3 * sizeof(int);

    k.dreluKey = readGPUDReluKey(key_as_bytes);
    k.selectKey = readGPUSelectKey<T>(key_as_bytes, k.numRelus);
    return k;
}

#include "gpu_relu.cu"