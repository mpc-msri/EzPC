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


// Utilities and system includes
#include <assert.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "utils/gpu_data_types.h"
#include "utils/helper_string.h" // helper for shared functions common to CUDA Samples

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>
#include "cutlass/reduction/device/tensor_reduce.h"

// CUDA and CUBLAS functions
#include "utils/helper_functions.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_stats.h"
#include "utils/gpu_random.h"

#include "fss/gpu_linear_helper.h"

#include "gpu_conv2d.h"


const int block_size = 256;
// cudnnHandle_t cudnn;

template <typename T>
using Conv2DKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, // accumulator, might be overkill for small bitwidths but that's okay
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm86,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

template <typename T>
using Conv2DImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2DKernel<T>>;

template <typename T>
T *cutlass_conv2d(GPUConv2DKey<T> k, T *d_I, T *d_F, T *d_C /*, GPUContext* c*/, bool cIsBias)
{
    // auto start = std::chrono::high_resolution_clock::now();
    auto A = getTensorRef(d_I, k.p.N, k.p.H, k.p.W, k.p.CI);
    auto B = getTensorRef(d_F, k.p.CO, k.p.FH, k.p.FW, k.p.CI);
    auto C = getTensorRef(d_C, k.p.N, k.p.OH, k.p.OW, k.p.CO);
    if (cIsBias)
        C = getTensorRefBias(d_C);
    auto D = getTensorRef((T *)gpuMalloc(k.mem_size_O), k.p.N, k.p.OH, k.p.OW, k.p.CO);
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(k.p.N, k.p.H, k.p.W, k.p.CI);
    cutlass::Tensor4DCoord filter_size(k.p.CO, k.p.FH, k.p.FW, k.p.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(k.p.zPadHLeft, k.p.zPadHRight, k.p.zPadWLeft, k.p.zPadWRight);
    cutlass::MatrixCoord conv_stride(k.p.strideH, k.p.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(k.p.N, k.p.OH, k.p.OW, k.p.CO);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        1 // split_k_slices
    );

    typename Conv2DImplicitGemm<T>::Arguments arguments{
        problem_size,
        A,
        B,
        d_C ? C : D,
        D,
        {T(1), d_C ? T(1) : T(0)},
    };

    Conv2DImplicitGemm<T> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    // printf("Allocating gpu workspace\n");
    uint8_t *workspace = gpuMalloc(workspace_size);
    // printf("Allocation done\n");
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    CUTLASS_CHECK(status);

    gpuFree(workspace);

    checkCudaErrors(cudaDeviceSynchronize());
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // std::cout << "Time for cutlass conv2d in microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() << std::endl;

    return D.data();
}

template <typename T>
using ConvDGradKernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm86,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    // cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kUnity>::Kernel;

template <typename T>
using ConvDGradImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<ConvDGradKernel<T>>;

template <typename T>
T *cutlass_conv_dgrad(GPUConv2DKey<T> k, T *d_incomingGrad, T *d_F, T *d_I)
{

    auto A = getTensorRef(d_incomingGrad, k.p.N, k.p.OH, k.p.OW, k.p.CO);
    auto B = getTensorRef(d_F, k.p.CO, k.p.FH, k.p.FW, k.p.CI);
    auto C = getTensorRef(d_I, k.p.N, k.p.H, k.p.W, k.p.CI);
    auto D = getTensorRef((T *)gpuMalloc(k.mem_size_O), k.p.N, k.p.H, k.p.W, k.p.CI);

    assert(k.mem_size_I == k.p.N * k.p.OH * k.p.OW * k.p.CO * sizeof(T));
    assert(k.mem_size_F == k.p.CO * k.p.FH * k.p.FW * k.p.CI * sizeof(T));
    assert(k.mem_size_O == k.p.N * k.p.H * k.p.W * k.p.CI * sizeof(T));

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(k.p.N, k.p.H, k.p.W, k.p.CI);
    cutlass::Tensor4DCoord filter_size(k.p.CO, k.p.FH, k.p.FW, k.p.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(k.p.zPadHLeft, k.p.zPadHRight, k.p.zPadWLeft, k.p.zPadWRight);
    cutlass::MatrixCoord conv_stride(k.p.strideH, k.p.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(k.p.N, k.p.OH, k.p.OW, k.p.CO);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size, // output_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size, // input_size,
        mode,
        1 // split_k_slices
    );
    // if(k.strideH == 1 && k.strideW == 1) {
    typename ConvDGradImplicitGemm<T>::Arguments arguments{
        problem_size,
        A,
        B,
        d_I ? C : D,
        D,
        {T(1), d_I ? T(1) : T(0)}};
    ConvDGradImplicitGemm<T> implicit_gemm_op;
    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    uint8_t *workspace = gpuMalloc(workspace_size);
    auto status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);
    status = implicit_gemm_op.initialize(arguments, workspace);
    CUTLASS_CHECK(status);
    status = implicit_gemm_op();
    CUTLASS_CHECK(status);
    gpuFree(workspace);

    return D.data();
}

template <typename T>
using ConvWGradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm86,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

template <typename T>
using ConvWGradImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<ConvWGradKernel<T>>;

template <typename T>
T *cutlass_conv_fgrad(GPUConv2DKey<T> k, T *d_grad, T *d_I, T *d_F)
{
    auto A = getTensorRef(d_grad, k.p.N, k.p.OH, k.p.OW, k.p.CO);
    auto B = getTensorRef(d_I, k.p.N, k.p.H, k.p.W, k.p.CI);
    auto C = getTensorRef(d_F, k.p.CO, k.p.FH, k.p.FW, k.p.CI);
    auto D = getTensorRef((T *)gpuMalloc(k.mem_size_O), k.p.CO, k.p.FH, k.p.FW, k.p.CI);

    assert(k.mem_size_I == k.p.N * k.p.OH * k.p.OW * k.p.CO * sizeof(T));
    assert(k.mem_size_F == k.p.N * k.p.H * k.p.W * k.p.CI * sizeof(T));
    assert(k.mem_size_O == k.p.CO * k.p.FH * k.p.FW * k.p.CI * sizeof(T));

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(k.p.N, k.p.H, k.p.W, k.p.CI);
    cutlass::Tensor4DCoord filter_size(k.p.CO, k.p.FH, k.p.FW, k.p.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(k.p.zPadHLeft, k.p.zPadHRight, k.p.zPadWLeft, k.p.zPadWRight);
    cutlass::MatrixCoord conv_stride(k.p.strideH, k.p.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(k.p.N, k.p.OH, k.p.OW, k.p.CO);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size, // output_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        1 // split_k_slices
    );

    typename ConvWGradImplicitGemm<T>::Arguments arguments{
        problem_size,
        A,
        B,
        d_F ? C : D,
        D,
        {T(1), d_F ? T(1) : T(0) /*options.alpha, options.beta*/}};

    ConvWGradImplicitGemm<T> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    uint8_t *workspace = gpuMalloc(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    CUTLASS_CHECK(status);

    gpuFree(workspace);

    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // std::cout << "Time for cutlass conv2d wgrad in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

    return D.data();
    // return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

template <typename T>
T *cutlassConv2D(GPUConv2DKey<T> k, T *d_I, T *d_F, T *d_C, char op, bool cIsBias = false, bool reduceBw = false)
{
    T *d_O;
    switch (op)
    {
    case 0:
        d_O = cutlass_conv2d(k, d_I, d_F, d_C, cIsBias);
        break;
    case 1:
        d_O = cutlass_conv_dgrad(k, d_I, d_F, d_C);
        break;
    case 2:
        d_O = cutlass_conv_fgrad(k, d_I, d_F, d_C);
        // unnecessary but ok
        break;
    }
    if (reduceBw && k.p.bout < sizeof(T) * 8)
    {
        modKernel<<<(k.p.size_O - 1) / block_size + 1, block_size>>>(k.p.size_O, d_O, k.p.bout);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    return d_O;
}

template <typename T>
T *gpuConv2DBeaver(GPUConv2DKey<T> k, int party, T *d_I, T *d_F, T *d_a, T *d_b, T *h_bias, Stats *s, char op)
{
    T *d_O1, *d_O2, *d_bias = NULL;
    if (op == 0 && h_bias != NULL)
    {
        d_bias = (T *)moveToGPU((uint8_t *)h_bias, k.p.CO * sizeof(T), s);
    }
    if (party == SERVER0)
    {
        gpuLinearComb(k.p.bout, k.p.size_F, d_b, T(1), d_F, -T(1), d_b);
    }
    d_O1 = cutlassConv2D(k, d_I, d_b, d_bias, op, true);
    d_O2 = cutlassConv2D(k, d_a, d_F, (T *)NULL, op);

    T *d_O = (T *)moveToGPU((uint8_t *)k.O, k.mem_size_O, s);
    gpuLinearComb(k.p.bout, k.p.size_O, d_O, T(1), d_O, party == SERVER0 ? T(1) : T(-1), d_O1, T(-1), d_O2);
    // beaverAdd<<<(k.p.size_O - 1) / block_size + 1, block_size>>>(k.p.size_O, party, k.p.bout, d_O1, d_O2, d_O);
    // checkCudaErrors(cudaDeviceSynchronize());
    if (d_bias)
        gpuFree(d_bias);
    gpuFree(d_O1);
    gpuFree(d_O2);
    return d_O;
}

template <typename T>
T *gpuConv2DWrapper(GPUConv2DKey<T> k, T *h_I, T *h_F, T *h_C, char op, bool cIsBias)
{
    T *d_I, *d_F, *d_C = NULL;
    d_I = (T *)moveToGPU((uint8_t *)h_I, k.mem_size_I, NULL);
    d_F = (T *)moveToGPU((uint8_t *)h_F, k.mem_size_F, NULL);
    if (h_C)
    {
        if (cIsBias)
            d_C = (T *)moveToGPU((uint8_t *)h_C, k.p.CO * sizeof(T), NULL);
        else
            d_C = (T *)moveToGPU((uint8_t *)h_C, k.mem_size_O, NULL);
    }

    // Need to fix this
    auto d_O = cutlassConv2D(k, d_I, d_F, d_C, op, cIsBias, true);
    gpuFree(d_I);
    gpuFree(d_F);
    if (h_C)
        gpuFree(d_C);

    T *h_O = (T *)moveToCPU((uint8_t *)d_O, k.mem_size_O, NULL);

    gpuFree(d_O);
    return h_O;
}

template <typename T>
T *gpuConv2DPlaintext(GPUConv2DKey<T> k, T *d_I, T *d_F, T *d_C, char op, bool cIsBias)
{
    // printf("here\n");
    auto d_O = cutlassConv2D(k, d_I, d_F, d_C, op, cIsBias, true);
    // if(k.Bout < sizeof(T) * 8) modKernel<<<(k.size_O - 1) / block_size + 1, block_size>>>(k.size_O, d_O, k.Bout);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_O;
}

template <typename T>
T *gpuKeygenConv2D(u8 **key_as_bytes, int party, GPUConv2DKey<T> k, T *d_mask_I = NULL, T *h_mask_F = NULL, bool maskOutput = false, T* d_mask_C = NULL)
{
    bool mask_I_was_null = false;
    if (!d_mask_I)
    {
        d_mask_I = randomGEOnGpu<T>(k.p.size_I, k.p.bin);
        // checkCudaErrors(cudaMemset(d_mask_I, 0, k.mem_size_I));
        mask_I_was_null = true;
    }
    // T *d_mask_F = NULL;
    // if (h_mask_F)
    // {
    T* d_mask_F = (T *)moveToGPU((u8 *)h_mask_F, k.mem_size_F, NULL);
    // }
    // else
    // {
        // printf("%lu\n", k.p.size_F);
        // d_mask_F = (T*) gpuMalloc(k.p.size_F * sizeof(T));
        // checkCudaErrors(cudaMemset(d_mask_F, 0, k.mem_size_F));
        // randomGEOnGpu<T>(k.p.size_F, k.p.bin);
    // }

    // T *d_mask_C = NULL;
    if (maskOutput && !d_mask_C)
    {
        d_mask_C = randomGEOnGpu<T>(k.p.size_O, k.p.bout);
        // checkCudaErrors(cudaMemset(d_mask_C, 0, k.p.size_O * sizeof(T)));
    }

    auto d_masked_C = gpuConv2DPlaintext(k, d_mask_I, d_mask_F, d_mask_C, 0, false);
    // printf("Writing shares=%lu, %lu, %lu\n", k.p.size_I, k.p.size_F, k.p.size_O);
    writeShares<T, T>(key_as_bytes, party, k.p.size_I, d_mask_I, k.p.bout);
    writeShares<T, T>(key_as_bytes, party, k.p.size_F, d_mask_F, k.p.bout);
    writeShares<T, T>(key_as_bytes, party, k.p.size_O, d_masked_C, k.p.bout);
    // printf("Done writing shares\n");
    if (mask_I_was_null)
        gpuFree(d_mask_I);
    gpuFree(d_mask_F);
    gpuFree(d_masked_C);
    // printf("Returning mask\n");
    return d_mask_C;
}