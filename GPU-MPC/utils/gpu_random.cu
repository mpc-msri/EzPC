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

/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cassert>

#include <cuda_runtime.h>
// #include <curand.h>

#include "curand_utils.h"
#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "misc_utils.h"

#include "gpu_random.h"

// using data_type = u32;
// cudaStream_t stream = NULL;
curandGenerator_t gpuGen[2];
curandGenerator_t cpuGen[2];
curandRngType_t rng = CURAND_RNG_PSEUDO_XORWOW;
curandOrdering_t order = CURAND_ORDERING_PSEUDO_BEST;

void randomUIntsOnGpu(const u64 n, u32 *d_data)
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandGenerate(gpuGen[device], d_data, n));
}

void randomUIntsOnCpu(const u64 n, u32 *h_data)
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandGenerate(cpuGen[device], h_data, n));
}

template <typename T>
T *randomGEOnGpu(const u64 n, int bw)
{
  u64 numUInts = (n * sizeof(T) - 1) / (sizeof(u32)) + 1;
  // printf("random n=%lu, ints=%lu, bw=%d\n", n, numUInts, bw);
  // assert((n * sizeof(T)) % sizeof(u32) == 0);
  auto d_data = (u32 *)gpuMalloc(numUInts * sizeof(u32));
  randomUIntsOnGpu(/*2 * n*/ numUInts, /*(u32*)*/ d_data);
  modKernel<<<(n - 1) / 256 + 1, 256>>>(n, (T *)d_data, bw);
  return (T *)d_data;
}

// extern "C" void randomGEOnGpu(const int n) {
//   randomUIntsOnGpu(2 * n, (u32*) d_data);
// }

template <typename T>
void randomGEOnCpu(const u64 n, int bw, T *h_data)
{
  u64 numUInts = (n * sizeof(T)) / (sizeof(u32));
  assert((n * sizeof(T)) % sizeof(u32) == 0);
  randomUIntsOnCpu(numUInts, (u32 *)h_data);
  if (bw < sizeof(T) * 8)
  {
    for (u64 i = 0; i < n; i++)
    {
      h_data[i] &= ((T(1) << bw) - 1);
    }
  }
}

template <typename T>
T *randomGEOnCpu(const u64 n, int bw)
{
  // printf("n=%lu\n", n);
  auto h_data = (T *)cpuMalloc(n * sizeof(T));
  randomGEOnCpu(n, bw, h_data);
  return h_data;
}

AESBlock *randomAESBlockOnGpu(const int n)
{
  AESBlock *d_data = (AESBlock *)gpuMalloc(n * sizeof(AESBlock));
  randomUIntsOnGpu(4 * n, (u32 *)d_data);
  return d_data;
}

void initGPURandomness()
{
  const unsigned long long offset = 0ULL;
  const unsigned long long seed = 12345ULL;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandCreateGenerator(&(gpuGen[device]), CURAND_RNG_PSEUDO_XORWOW));
  CURAND_CHECK(curandSetGeneratorOffset(gpuGen[device], offset));
  CURAND_CHECK(curandSetGeneratorOrdering(gpuGen[device], order));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gpuGen[device], seed));
}

void initCPURandomness()
{
  const unsigned long long offset = 0ULL;
  const unsigned long long seed = 1234567890ULL;
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  printf("CPU randomness, seed: %llu, offset: %llu\n", seed, offset);
  CURAND_CHECK(curandCreateGeneratorHost(&(cpuGen[device]), CURAND_RNG_PSEUDO_XORWOW));
  CURAND_CHECK(curandSetGeneratorOffset(cpuGen[device], offset));
  CURAND_CHECK(curandSetGeneratorOrdering(cpuGen[device], order));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(cpuGen[device], seed));
}

void destroyGPURandomness()
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandDestroyGenerator(gpuGen[device]));
  // CUDA_CHECK(cudaDeviceReset());
}

void destroyCPURandomness()
{
  int device;
  checkCudaErrors(cudaGetDevice(&device));
  CURAND_CHECK(curandDestroyGenerator(cpuGen[device]));
  // CUDA_CHECK(cudaDeviceReset());
}

template <typename TIn, typename TOut>
void writeShares(u8 **key_as_bytes, int party, u64 N, TIn *d_A, int bw, bool randomShares)
{
  assert(bw <= 8 * sizeof(TOut));
  TOut *d_A0 = NULL;
  if (randomShares)
    d_A0 = randomGEOnGpu<TOut>(N, bw);
  // checkCudaErrors(cudaMemset(d_A0, 0, N * sizeof(TOut)));
  size_t memSzA;
  if (bw == 1 || bw == 2)
  {
    auto numInts = ((bw * N - 1) / PACKING_SIZE + 1);
    memSzA = numInts * sizeof(PACK_TYPE);
  }
  else
    memSzA = N * sizeof(TOut);
  auto d_packed_A = (u32 *)gpuMalloc(memSzA);

  getPackedSharesKernel<<<(N - 1) / 256 + 1, 256>>>(N, party, d_A, d_A0, d_packed_A, bw);
  checkCudaErrors(cudaDeviceSynchronize());

  moveIntoCPUMem((u8 *)*key_as_bytes, (u8 *)d_packed_A, memSzA, NULL);
  *key_as_bytes += memSzA;

  if (d_A0)
    gpuFree(d_A0);
  gpuFree(d_packed_A);
}

template <typename T>
T *getMaskedInputOnGpu(int N, int bw, T *d_mask_I, T **h_I, bool smallInputs, int smallBw)
{
  size_t memSzI = N * sizeof(T);
  // int smallBw = 38; //15;
  // printf("small inputs=%d\n", smallInputs);
  // keep a gap here
  auto d_I = randomGEOnGpu<T>(N, smallInputs ? std::min(smallBw, bw) : bw /*- 1*/);
  // checkCudaErrors(cudaMemset(d_I, 0, memSzI));
  // printf("%ld\n", -(T(1) << (smallInputs ? 14 : bw - 2)));
  // uncomment this for small negative inputs
  if (smallInputs)
    gpuLinearComb(64 /*bw*/, N, d_I, T(1), d_I, -(T(1) << (smallBw - 1)));
  *h_I = (T *)moveToCPU((u8 *)d_I, memSzI, NULL);
  // printf("Input: %ld, %ld\n", i64((*h_I)[0]), i64((*h_I)[1]));
  gpuLinearComb(bw, N, d_I, T(1), d_I, T(1), d_mask_I);
  // gpuAddSharesInPlace<T>(d_I, d_mask_I, bw, N);
  return d_I;
}

template <typename T>
T *getMaskedInputOnCpu(int N, int bw, T *h_mask_I, T **h_I, bool smallInputs, int smallBw)
{
  size_t memSzI = N * sizeof(T);
  auto d_mask_I = (T *)moveToGPU((u8 *)h_mask_I, memSzI, NULL);
  auto d_masked_I = getMaskedInputOnGpu<T>(N, bw, d_mask_I, h_I, smallInputs, smallBw);
  gpuFree(d_mask_I);
  auto h_masked_I = (T *)moveToCPU((u8 *)d_masked_I, memSzI, NULL);
  gpuFree(d_masked_I);
  return h_masked_I;
}