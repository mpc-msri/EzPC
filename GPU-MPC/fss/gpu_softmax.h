#pragma once

#include "gpu_maxpool.h"
#include "gpu_nexp.h"
#include "gpu_lut.h"
#include "gpu_inverse.h"
#include "gpu_window.h"


template <typename T>
struct GPUSoftMaxKey
{
    GPUMaxpoolKey<T> maxPoolKey;
    GPUNExpKey<T> nExpKey;
    GPULUTInverseKey<T> invKey;
    GPUMulKey<T> wMulKey;
};

template <typename T>
GPUSoftMaxKey<T> readGPUSoftMaxKey(MaxpoolParams p, u8 **key_as_bytes)
{
    GPUSoftMaxKey<T> k;
    assert(p.C == 1);
    assert(p.strideH == 1);
    assert(p.strideW == p.FW);

    u64 inSz = getInSz(p);
    u64 mSz = getMSz(p);
    k.maxPoolKey = readGPUMaxpoolKey<T>(p, key_as_bytes);
    k.nExpKey = readGPUNExpKey<T>(key_as_bytes);
    k.invKey = readGPULUTInverseKey<T>(key_as_bytes);
    k.wMulKey = readGPUWindowMulKey<T>(p, TruncateType::TrWithSlack, key_as_bytes);
    return k;
}



#include "gpu_softmax.cu"