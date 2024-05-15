#pragma once

#include "gpu_lut.h"
#include "gpu_truncate.h"
#include "gpu_mul.h"
#include "gpu_relu.h"

template <typename T>
struct GPUNExpKey
{
    int N;
    GPUReluKey<u16> reluKey;
    GPULUTKey<T> lsbLutKey;
    GPUTruncateKey<u8> trKey;
    GPULUTKey<T> msbLutKey;
    GPUMulKey<T> mulKey;
    // GPUTruncateKey<T> mulTrKey;
};

template <typename T>
GPUNExpKey<T> readGPUNExpKey(u8 **key_as_bytes)
{
    GPUNExpKey<T> k;
    k.reluKey = readReluKey<u16>(key_as_bytes);
    printf("##Reading Relu key=%d\n", k.reluKey.bout);
    k.N = k.reluKey.numRelus;
    k.lsbLutKey = readGPULUTKey<T>(key_as_bytes);
    k.trKey = readGPUTruncateKey<u8>(TruncateType::TrWithSlack, key_as_bytes);
    k.msbLutKey = readGPULUTKey<T>(key_as_bytes);
    k.mulKey = readGPUMulKey<T>(key_as_bytes, (u64)k.N, (u64)k.N, (u64)k.N, TruncateType::TrWithSlack);
    // printf("Done reading nexp key\n");
    // k.mulTrKey = readGPUTruncateKey<T>(TruncateType::TrWithSlack, key_as_bytes);
    return k;
}

#include "gpu_nexp.cu"