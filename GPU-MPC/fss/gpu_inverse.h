#pragma once

#include "gpu_lut.h"
#include "gpu_truncate.h"

template <typename T>
struct GPULUTInverseKey {
    int N;
    GPUTruncateKey<u16> trKey;
    GPULUTKey<T> lutKey;
};

template <typename T>
GPULUTInverseKey<T> readGPULUTInverseKey(u8** key_as_bytes) {
    GPULUTInverseKey<T> k;
    k.trKey = readGPUTruncateKey<u16>(TruncateType::TrWithSlack, key_as_bytes);
    k.N = k.trKey.N;
    k.lutKey = readGPULUTKey<T>(key_as_bytes);
    return k;
}

#include "gpu_inverse.cu"