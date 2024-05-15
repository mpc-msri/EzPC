#include "utils/gpu_data_types.h"
// #include "utils/misc_utils.h"

template <typename T>
__global__ void keyGenAndKernel(int N, T *b0, T *b1, T *randomMaskOut, T *maskOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        maskOut[i] = (b0[i] * b1[i] + randomMaskOut[i]) & 1ULL;
    }
}

template <typename T>
void writeAndKey(u8 **key_as_bytes, int party, int N, T *d_b0, T *d_b1, T *d_maskOut, int bout)
{
    assert(bout == 1);
    writeInt(key_as_bytes, N);
    writeShares<T, T>(key_as_bytes, party, N, d_b0, bout);
    writeShares<T, T>(key_as_bytes, party, N, d_b1, bout);
    writeShares<T, T>(key_as_bytes, party, N, d_maskOut, bout);
}

template <typename T>
T *gpuKeyGenAnd(u8 **key_as_bytes, int party, int bout, int N, T *d_b0, T *d_b1)
{
    assert(bout == 1);
    auto d_randomMaskOut = randomGEOnGpu<T>(N, 1);
    // checkCudaErrors(cudaMemset(d_randomMaskOut, 0, N * sizeof(u64)));
    auto d_maskOut = (T *)gpuMalloc(N * sizeof(T));
    keyGenAndKernel<<<(N - 1) / 256 + 1, 256>>>(N, d_b0, d_b1, d_randomMaskOut, d_maskOut);
    writeAndKey(key_as_bytes, party, N, d_b0, d_b1, d_maskOut, bout);
    gpuFree(d_maskOut);
    return d_randomMaskOut;
}
