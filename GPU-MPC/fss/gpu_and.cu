// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
