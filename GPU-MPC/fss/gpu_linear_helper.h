#pragma once

#include "utils/gpu_data_types.h"
#include <cutlass/cutlass.h>

#define CUTLASS_CHECK(status)                                                                      \
{                                                                                                  \
    cutlass::Status error = status;                                                                \
    if (error != cutlass::Status::kSuccess) {                                                      \
        std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                  << std::endl;                                                                    \
    }                                                                                              \
}

template <typename T> void gpuAddBias(int batchSz, int N, int M, int bw, T* d_A, T* h_b, Stats* s);

#include "gpu_linear_helper.cu"