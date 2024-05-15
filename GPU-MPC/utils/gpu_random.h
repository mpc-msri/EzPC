#pragma once

#include "gpu_data_types.h"
#include <curand.h>


template <typename T>
T *randomGEOnGpu(const u64 n, int bw);
template <typename T>
void randomGEOnCpu(const u64 n, int bw, T *h_data);
template <typename T>
T *randomGEOnCpu(const u64 n, int bw);
AESBlock *randomAESBlockOnGpu(const int n);
void initGPURandomness();
void destroyGPURandomness();
void initCPURandomness();
void destroyCPURandomness();
template <typename T>
T *getMaskedInputOnGpu(int N, int bw, T *d_mask_I, T **h_I, bool smallInputs = false, int smallBw = 0);
template <typename T>
T *getMaskedInputOnCpu(int N, int bw, T *h_mask_I, T **h_I, bool smallInputs = false, int smallBw = 0);
template <typename TIn, typename TOut>
void writeShares(u8 **key_as_bytes, int party, u64 N, TIn *d_A, int bw, bool randomShares = true);


#include "gpu_random.cu"