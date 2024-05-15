#include "gpu_mul.h"

template <typename T>
__global__ void keygenBeaver(int bw, int N, T *A, T *B, T *C, T *C1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C1[i] = A[i] * B[i] + C[i];
        gpuMod(C1[i], bw);
    }
}

template <typename T>
__global__ void doBeaverMul(int party, int bw, int N, T *X, T *Y, T *a, T *b, T *c, T *Z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        Z[i] = (party == SERVER1) * (X[i] * Y[i]) - X[i] * b[i] - a[i] * Y[i] + c[i];
        gpuMod(Z[i], bw);
        // printf("%ld, %ld, %ld\n", X[i], Y[i], Z[i]);
    }
}

template <typename T>
T *gpuKeygenMul(u8 **key_as_bytes, int party, int bw, int scale, int N, T *d_mask_A, T *d_mask_B, TruncateType t, AESGlobalContext *gaes)
{
    auto d_mask_C = randomGEOnGpu<T>(N, bw);
    // checkCudaErrors(cudaMemset(d_mask_C, 0, N * sizeof(T)));
    auto d_mask_C1 = (T *)gpuMalloc(N * sizeof(T));
    keygenBeaver<<<(N - 1) / 128 + 1, 128>>>(bw, N, d_mask_A, d_mask_B, d_mask_C, d_mask_C1);
    writeShares<T, T>(key_as_bytes, party, N, d_mask_A, bw);
    writeShares<T, T>(key_as_bytes, party, N, d_mask_B, bw);
    writeShares<T, T>(key_as_bytes, party, N, d_mask_C1, bw);
    gpuFree(d_mask_C1);
    printf("##Num truncations: %d\n", N);
    auto d_mask_truncated_C = genGPUTruncateKey<T, T>(key_as_bytes, party, /*TruncateType::TrWithSlack*/t, bw, bw, scale, N, d_mask_C, gaes);
    gpuFree(d_mask_C);
    return d_mask_truncated_C;
}

template <typename T>
T *gpuMul(SigmaPeer *peer, int party, int bw, int scale, int N, GPUMulKey<T> k, T *d_X, T *d_Y, TruncateType t, AESGlobalContext *gaes, Stats *s)
{
    u64 b0 = peer->bytesSent() + peer->bytesReceived();
    auto d_a = (T *)moveToGPU((u8 *)k.a, 3 * N * sizeof(T), s);
    auto d_b = d_a + N;
    auto d_c = d_b + N;
    auto d_Z = (T *)gpuMalloc(N * sizeof(T));
    doBeaverMul<<<(N - 1) / 128 + 1, 128>>>(party, bw, N, d_X, d_Y, d_a, d_b, d_c, d_Z);
    gpuFree(d_a);
    peer->reconstructInPlace(d_Z, bw, N, s);
    auto d_truncated_Z = gpuTruncate<T, T>(bw, bw, /*TruncateType::TrWithSlack*/t, k.trKey, scale, peer, party, N, d_Z, gaes, s); //, true);
    gpuFree(d_Z);
    u64 b1 = peer->bytesSent() + peer->bytesReceived();
    if (s)
        s->linear_comm_bytes += (b1 - b0);
    printf("Comm inside Mul=%ld\n", b1 - b0);
    return d_truncated_Z;
}
