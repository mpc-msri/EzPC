#pragma once

#include "utils/gpu_data_types.h"
#include "gpu_truncate.h"

// Z = aX, where a is a public scalar
template <typename T>
T *gpuKeygenScalarMul(u8 **key_as_bytes, int party, int bw, int N, T a, T *d_mask_X, TruncateType t, int shift, AESGlobalContext *gaes)
{
    auto d_mask_Z = (T *)gpuMalloc(N * sizeof(T));
    gpuLinearComb(bw, N, d_mask_Z, a, d_mask_X);
    printf("Truncate type=%d\n", t);
    auto d_mask_truncated_Z = genGPUTruncateKey<T, T>(key_as_bytes, party, t, bw, bw, shift, N, d_mask_X, gaes);
    if (d_mask_truncated_Z != d_mask_Z)
        gpuFree(d_mask_Z);
    return d_mask_truncated_Z;
}

template <typename T>
T *gpuScalarMul(SigmaPeer *peer, int party, int bw, int N, GPUTruncateKey<T> k, T a, T *d_X, TruncateType t, int shift, AESGlobalContext *gaes, Stats *s)
{
    u64 b0 = peer->bytesSent() + peer->bytesReceived();
    auto d_Z = (T *)gpuMalloc(N * sizeof(T));
    gpuLinearComb(bw, N, d_Z, a, d_X);
    printf("Truncate type=%d\n", t);
    auto d_truncated_Z = gpuTruncate<T, T>(bw, bw, t, k, shift, peer, party, N, d_Z, gaes, s); //, true);
    gpuFree(d_Z);
    u64 b1 = peer->bytesSent() + peer->bytesReceived();
    s->linear_comm_bytes += (b1 - b0);
    return d_truncated_Z;
}
