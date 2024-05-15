#pragma once

#include "gpu_maxpool.h"

template <typename T>
GPUMulKey<T> readGPUWindowMulKey(MaxpoolParams p, TruncateType t, u8 **key_as_bytes)
{
    GPUMulKey<T> k;
    u64 inSz = getInSz(p);
    u64 mSz = getMSz(p);
    printf("%d, %d\n", inSz, mSz);
    k = readGPUMulKey<T>(key_as_bytes, inSz, mSz, inSz, t);
    return k;
}

#include "gpu_window.cu"