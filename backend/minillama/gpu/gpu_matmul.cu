// Utilities and system includes
#include <assert.h>
#include "gpu_data_types.h"
#include "helper_string.h" // helper for shared functions common to CUDA Samples

#include <chrono>

#include "cutlass/gemm/device/gemm.h"

#include <math.h>
#include <stdlib.h>


// CUDA and CUBLAS functions
#include "helper_functions.h"
#include "helper_cuda.h"
#include "gpu_linear_helper.cuh"
#include "gpu_ctx.h"
#include "gpu_mem.h"
#include "gpu_stats.h"

#include <iostream>
#include <fstream>
#include <string>

const int block_size = 256;

#define CUTLASS_CHECK(status)                                                                      \
{                                                                                                  \
    cutlass::Status error = status;                                                                \
    if (error != cutlass::Status::kSuccess) {                                                      \
        std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                  << std::endl;                                                                    \
    }                                                                                              \
}

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

using CutlassGemmRRR = cutlass::gemm::device::Gemm<GPUGroupElement,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  GPUGroupElement,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  GPUGroupElement,        // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix

using CutlassGemmRCR = cutlass::gemm::device::Gemm<GPUGroupElement,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  GPUGroupElement,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  GPUGroupElement,        // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix


using CutlassGemmCRR = cutlass::gemm::device::Gemm<GPUGroupElement,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  GPUGroupElement,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  GPUGroupElement,        // Data-type of C matrix
                                                  RowMajor>; // Layout of C matrix




GPUGroupElement* cutlass_matmul(GPUGroupElement* d_A, GPUGroupElement* d_B, GPUGroupElement* d_C, GPUMatmulKey k, /*bool rowMajA, bool rowMajB, bool rowMajC,*/ bool cIsBias=false) {
    GPUGroupElement* d_D = (GPUGroupElement*) gpuMalloc(k.M * k.N * sizeof(GPUGroupElement));
    cutlass::Status status;
    if(k.rowMajA && k.rowMajB && k.rowMajC) {
        // printf("%lu %d\n", d_C, cIsBias);
        CutlassGemmRRR gemm_operator;
        CutlassGemmRRR::Arguments args({k.M, k.N, k.K},  // Gemm Problem dimensions
                              {d_A, k.K},    // Tensor-ref for source matrix A
                              {d_B, k.N},    // Tensor-ref for source matrix B
                              {d_C ? d_C : d_D, d_C && cIsBias ? 0 : k.N},    // Tensor-ref for source matrix C
                              {d_D, k.N},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {1ULL, d_C ? 1ULL : 0ULL}); // Scalars used in the Epilogue
        status = gemm_operator(args);
    } else if(k.rowMajA && !k.rowMajB && k.rowMajC) { /* M x K x N */
        CutlassGemmRCR gemm_operator;
        CutlassGemmRCR::Arguments args({k.M, k.N, k.K},  // Gemm Problem dimensions
                              {d_A, k.K},    // Tensor-ref for source matrix A
                              {d_B, k.K},    // Tensor-ref for source matrix B
                              {d_C ? d_C : d_D, k.N},    // Tensor-ref for source matrix C
                              {d_D, k.N},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {1ULL, d_C ? 1ULL : 0ULL}); // Scalars used in the Epilogue
        status = gemm_operator(args);
    } else if(!k.rowMajA && k.rowMajB && k.rowMajC) { /* M x K x N */
        CutlassGemmCRR gemm_operator;
        // printf("%d %d %d\n", k.M, k.N, k.K);
        CutlassGemmCRR::Arguments args({k.M, k.N, k.K},  // Gemm Problem dimensions
                              {d_A, k.M},    // Tensor-ref for source matrix A
                              {d_B, k.N},    // Tensor-ref for source matrix B
                              {d_C ? d_C : d_D, k.N},    // Tensor-ref for source matrix C
                              {d_D, k.N},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {1ULL, d_C ? 1ULL : 0ULL}); // Scalars used in the Epilogue
        status = gemm_operator(args);
    } else {
        assert(false && "no option matches!");
    }
    CUTLASS_CHECK(status);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_D;
}


extern "C" GPUGroupElement *gpu_matmul(GPUMatmulKey k, int party, GPUGroupElement *d_A, GPUGroupElement *d_B, GPUGroupElement* d_X, GPUGroupElement* d_Y, GPUGroupElement* h_bias,/*bool rowMajA, bool rowMajB, bool rowMajC,*/ Stats* s/*, GPUContext* c*/)
{
    GPUGroupElement *d_C1, *d_C2, *d_bias = NULL;
    if (party == 0)
    {
        subtractInPlace<<<(k.size_B - 1) / block_size + 1, block_size>>>(d_B, d_Y, k.size_B);
        checkCudaErrors(cudaDeviceSynchronize());
        if(h_bias != NULL) d_bias = (GPUGroupElement*) moveToGPU((uint8_t*) h_bias, k.N * sizeof(GPUGroupElement), s);
    }
    d_C1 = cutlass_matmul(d_A, d_Y, d_bias, k/*, c , false, true, rowMajA, rowMajB, rowMajC*/, true);
    d_C2 = cutlass_matmul(d_X, d_B, NULL, k/*, c , false, false, rowMajA, rowMajB, rowMajC*/);
    GPUGroupElement *d_C = (GPUGroupElement*) moveToGPU((uint8_t*) k.C, k.mem_size_C, s);
    beaver_add_group_elements<<<(k.size_C - 1) / block_size + 1, block_size, 0/*, c->stream*/>>>(d_C1, d_C2, d_C, k.size_C, party);
    // if(party == 0)
    gpuFree(d_C1);
    gpuFree(d_C2);
    // gpuFree(d_C3);
    return d_C;
}


extern "C" GPUGroupElement *gpuMatmulWrapper(GPUMatmulKey k, GPUGroupElement* h_A, GPUGroupElement* h_B, GPUGroupElement* h_C, /*bool rowMajA, bool rowMajB, bool rowMajC,*/ bool cIsBias)
{
    size_t mem_size_bias = k.N * sizeof(GPUGroupElement);
    // printf("matmul wrapper: %lu %lu\n", k.mem_size_A, k.mem_size_B);
    GPUGroupElement* d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, k.mem_size_A, NULL);
    // printf("here\n");
    GPUGroupElement* d_B = (GPUGroupElement*) moveToGPU((uint8_t*) h_B, k.mem_size_B, NULL);
    // printf("here\n");
    GPUGroupElement* d_C = NULL;
    if(h_C) d_C = (GPUGroupElement*) moveToGPU((uint8_t*) h_C, cIsBias ? mem_size_bias : k.mem_size_C, NULL);
    // Need to fix this
    auto d_D = cutlass_matmul(d_A, d_B, d_C, k, /*NULL , 0 , true, false, k.rowMajA, k.rowMajB, k.rowMajC,*/ cIsBias);
    gpuFree(d_A);
    gpuFree(d_B);
    if(d_C) gpuFree(d_C);
    GPUGroupElement *h_D = (GPUGroupElement*) moveToCPU((uint8_t*) d_D, k.mem_size_C, NULL);
    gpuFree(d_D);
    return h_D;
}
