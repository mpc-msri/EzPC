// Utilities and system includes
#include <assert.h>
#include "gpu_data_types.h"
#include "helper_string.h" // helper for shared functions common to CUDA Samples

#include <chrono>

// CUDA runtime
#include <cstddef>
#include <cstdint>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
// #include <cublas_v2.h>

#include "cutlass/reduction/device/tensor_reduce.h"


#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>
#include <math.h>
#include <stdlib.h>


// CUDA and CUBLAS functions
#include "helper_functions.h"
#include "helper_cuda.h"
#include "gpu_linear_helper.cuh"
#include "gpu_ctx.h"
#include "gpu_mem.h"
#include "gpu_stats.h"

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/util/device_memory.h>

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

inline cutlass::TensorRef<GPUGroupElement, cutlass::layout::TensorNHWC> toTensorRef(
        GPUGroupElement *ptr, int n, int h, int w, int c) {

    return cutlass::TensorRef<GPUGroupElement, cutlass::layout::TensorNHWC>(
        ptr,
        cutlass::layout::TensorNHWC::packed({n, h, w, c})
    );
}

inline cutlass::TensorRef<GPUGroupElement, cutlass::layout::TensorNHWC> toTensorRefBias(
        GPUGroupElement *ptr) {

    return cutlass::TensorRef<GPUGroupElement, cutlass::layout::TensorNHWC>(
        ptr,
        cutlass::layout::TensorNHWC::Stride(0)
    );
}

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        GPUGroupElement,
        1,
        GPUGroupElement,
        GPUGroupElement
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

using FpropImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


GPUGroupElement* cutlass_conv2d(GPUGroupElement *d_I, GPUGroupElement *d_F, GPUGroupElement* d_C, GPUConv2DKey k/*, GPUContext* c*/, bool cIsBias) 
{
    // auto start = std::chrono::high_resolution_clock::now();
    auto A = toTensorRef(d_I, k.N, k.H, k.W, k.CI);
    auto B = toTensorRef(d_F, k.CO, k.FH, k.FW, k.CI);
    auto C = toTensorRef(d_C, k.N, k.OH, k.OW, k.CO);
    if(cIsBias) C = toTensorRefBias(d_C);
    auto D = toTensorRef((GPUGroupElement*) gpuMalloc(k.mem_size_O), k.N, k.OH, k.OW, k.CO);
    
    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(k.N, k.H, k.W, k.CI);
    cutlass::Tensor4DCoord filter_size(k.CO, k.FH, k.FW, k.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(k.zPadHLeft, k.zPadHRight, k.zPadWLeft, k.zPadWRight);
    cutlass::MatrixCoord conv_stride(k.strideH, k.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(k.N, k.OH, k.OW, k.CO);

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

    typename FpropImplicitGemm::Arguments arguments {
        problem_size,
        A, B, d_C ? C : D, D,
        {1ULL, d_C ? 1ULL : 0ULL},
    };

    FpropImplicitGemm implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    uint8_t* workspace = gpuMalloc(workspace_size);
    // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

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
    // std::cout << "Time for cutlass conv2d in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

    return D.data();
}


using Conv2dDgradKernel = typename cutlass::conv::kernel::DefaultConv2dDgrad<
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        GPUGroupElement,
        1,
        GPUGroupElement,
        GPUGroupElement
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    // cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic,
    cutlass::conv::StrideSupport::kUnity
>::Kernel;

using DgradImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

GPUGroupElement* cutlass_conv_dgrad(
    GPUGroupElement* d_incomingGrad, GPUGroupElement* d_F, GPUGroupElement* d_I, GPUConv2DKey k) {
    // printf("mem alloc: %lu %lu %lu\n", k.mem_size_I, k.mem_size_F, k.mem_size_O);
    // cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();

    auto A = toTensorRef(d_incomingGrad, k.N, k.OH, k.OW, k.CO);
    auto B = toTensorRef(d_F, k.CO, k.FH, k.FW, k.CI);
    auto C = toTensorRef(d_I, k.N, k.H, k.W, k.CI);
    auto D = toTensorRef((GPUGroupElement*) gpuMalloc(k.mem_size_O), k.N, k.H, k.W, k.CI);

    assert(k.mem_size_I == k.N * k.OH * k.OW * k.CO * sizeof(GPUGroupElement));
    assert(k.mem_size_F == k.CO * k.FH * k.FW * k.CI * sizeof(GPUGroupElement));
    assert(k.mem_size_O == k.N * k.H * k.W * k.CI * sizeof(GPUGroupElement));

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(k.N, k.H, k.W, k.CI);
    cutlass::Tensor4DCoord filter_size(k.CO, k.FH, k.FW, k.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(k.zPadHLeft, k.zPadHRight, k.zPadWLeft, k.zPadWRight);
    cutlass::MatrixCoord conv_stride(k.strideH, k.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(k.N, k.OH, k.OW, k.CO);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size, // output_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,// input_size,
        mode,
        1 // split_k_slices
    );     
    typename DgradImplicitGemm::Arguments arguments {
        problem_size,
        A, B, d_I ? C : D, D,
        {1ULL, d_I ? 1ULL : 0ULL} 
    };

    DgradImplicitGemm implicit_gemm_op;
    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

    uint8_t* workspace = gpuMalloc(workspace_size);
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
    // std::cout << "Time for cutlass conv2d dgrad in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;


    return D.data();
}

using Conv2dWgradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement, cutlass::layout::TensorNHWC,
    GPUGroupElement,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        GPUGroupElement,
        1,
        GPUGroupElement,
        GPUGroupElement
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

using WgradImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

GPUGroupElement* cutlass_conv_fgrad(GPUGroupElement* d_grad, GPUGroupElement* d_I, GPUGroupElement* d_F, GPUConv2DKey k) {
    // cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();
//f = grad * input
    auto A = toTensorRef(d_grad, k.N, k.OH, k.OW, k.CO);
    auto B = toTensorRef(d_I, k.N, k.H, k.W, k.CI);
    auto C = toTensorRef(d_F, k.CO, k.FH, k.FW, k.CI);
    auto D = toTensorRef((GPUGroupElement*) gpuMalloc(k.mem_size_O), k.CO, k.FH, k.FW, k.CI);

    assert(k.mem_size_I == k.N * k.OH * k.OW * k.CO * sizeof(GPUGroupElement));
    assert(k.mem_size_F == k.N * k.H * k.W * k.CI * sizeof(GPUGroupElement));
    assert(k.mem_size_O == k.CO * k.FH * k.FW * k.CI * sizeof(GPUGroupElement));

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::Tensor4DCoord input_size(k.N, k.H, k.W, k.CI);
    cutlass::Tensor4DCoord filter_size(k.CO, k.FH, k.FW, k.CI);
    // check these initializations later
    cutlass::Tensor4DCoord padding(k.zPadHLeft, k.zPadHRight, k.zPadWLeft, k.zPadWRight);
    cutlass::MatrixCoord conv_stride(k.strideH, k.strideW);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(k.N, k.OH, k.OW, k.CO);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,// output_size,
        filter_size,
        padding,
        conv_stride,
        dilation,
        output_size,
        mode,
        1 // split_k_slices
    ); 

    typename WgradImplicitGemm::Arguments arguments {
        problem_size,
        A, B, d_F ? C : D, D,
        {1ULL, d_F ? 1ULL : 0ULL/*options.alpha, options.beta*/} 
    };

    WgradImplicitGemm implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    uint8_t* workspace = gpuMalloc(workspace_size);

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




GPUGroupElement* execute_op(GPUGroupElement* d_I, GPUGroupElement* d_F, GPUGroupElement* d_C, GPUConv2DKey k, char op, bool cIsBias=false) {
    // return cutlass_conv2d(d_I, d_F, k);
    switch(op) {
        case 0: 
            return cutlass_conv2d(d_I, d_F, d_C, k, cIsBias);
        case 1:
            return cutlass_conv_dgrad(d_I, d_F, d_C, k);
        case 2:
            return cutlass_conv_fgrad(d_I, d_F, d_C, k);
    }
}


extern "C" GPUGroupElement *gpu_conv2d(GPUConv2DKey k, int party, GPUGroupElement *d_I, GPUGroupElement *d_F, GPUGroupElement* d_a, GPUGroupElement* d_b, GPUGroupElement* h_bias, Stats* s, char op)
{
    GPUGroupElement *d_O1, *d_O2, *d_bias = NULL;
    // printf("filter size: %d\n", k.size_F);
    if (party == 0)
    {
        subtractInPlace<<<(k.size_F - 1) / block_size + 1, block_size>>>(d_F, d_b, k.size_F);
        if(op == 0 /*&& party == 0*/ && h_bias != NULL) {
            d_bias = (GPUGroupElement*) moveToGPU((uint8_t*) h_bias, k.CO * sizeof(GPUGroupElement), s);
        } 
    }
    d_O1 = execute_op(d_I, d_b, d_bias, k/*, c , false, true*/, op, true);
    d_O2 = execute_op(d_a, d_F, NULL, k/*, c , false, false*/, op);

    GPUGroupElement *d_O = (GPUGroupElement*) moveToGPU((uint8_t*) k.O, k.mem_size_O, s);
    beaver_add_group_elements<<<(k.size_O - 1) / block_size + 1, block_size>>>(d_O1, d_O2, d_O, k.size_O, party);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    gpuFree(d_bias);
    gpuFree(d_O1);
    gpuFree(d_O2);
    return d_O;
}

// extern "C" GPUGroupElement *gpuConv2DProtocolWrapper(GPUConv2DKey k, int party, GPUGroupElement *h_I, GPUGroupElement *h_F) {
//     GPUGroupElement *d_I = (GPUGroupElement*) moveToGPU((uint8_t*) h_I, k.mem_size_I, NULL);
//     GPUGroupElement *d_F = (GPUGroupElement*) moveToGPU((uint8_t*) h_F, k.mem_size_F, NULL);

//     // cudaMalloc(&d_I, k.mem_size_I);
//     // cudaMalloc(&d_F, k.mem_size_F);

//     // cudaMemcpy(d_I, h_I, k.mem_size_I, cudaMemcpyHostToDevice);
//     // cudaMemcpy(d_F, h_F, k.mem_size_F, cudaMemcpyHostToDevice);

//     auto d_O = gpu_conv2d(k, party, d_I, d_F, NULL);

//     GPUGroupElement *h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, k.mem_size_O, NULL);
//     // cudaMallocHost(&h_O, k.mem_size_O);
//     // cudaMemcpy(h_O, d_O, k.mem_size_O, cudaMemcpyDeviceToHost);

//     return h_O;
// }


extern "C" GPUGroupElement *gpuConv2DWrapper(GPUConv2DKey k, GPUGroupElement* h_I, GPUGroupElement* h_F, GPUGroupElement* h_C, char op, bool cIsBias)
{
    GPUGroupElement *d_I, *d_F, *d_C = NULL;
    d_I = (GPUGroupElement*) moveToGPU((uint8_t*) h_I, k.mem_size_I, NULL);
    d_F = (GPUGroupElement*) moveToGPU((uint8_t*) h_F, k.mem_size_F, NULL);
    if(h_C) {
        if(cIsBias) d_C = (GPUGroupElement*) moveToGPU((uint8_t*) h_C, k.CO * sizeof(GPUGroupElement), NULL);
        else d_C = (GPUGroupElement*) moveToGPU((uint8_t*) h_C, k.mem_size_O, NULL);
    }

    // Need to fix this
    auto d_O = execute_op(d_I, d_F, d_C, k, op, cIsBias);
    gpuFree(d_I);
    gpuFree(d_F);
    if(h_C) gpuFree(d_C);

    GPUGroupElement *h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, k.mem_size_O, NULL);

    gpuFree(d_O);
    return h_O;
}


extern "C" GPUGroupElement* gpuAddShares(GPUGroupElement* d_A, GPUGroupElement* d_B, int N) {
    GPUGroupElement* d_C = (GPUGroupElement*) gpuMalloc(N * sizeof(GPUGroupElement));
    // cudaMalloc(&d_C, N * sizeof(GPUGroupElement));
    const int thread_block_size = 128;
    add_group_elements<<<(N - 1) / thread_block_size + 1, thread_block_size>>>(d_A, d_B, d_C, N);
    checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaGetLastError());
    // gpuFree(d_A);
    // gpuFree(d_B);
    return d_C;
}

extern "C" void gpuAddSharesInPlace(GPUGroupElement* d_A, GPUGroupElement* d_B, int bw, int N) {
    const int thread_block_size = 128;
    addInPlace<<<(N - 1) / thread_block_size + 1, thread_block_size>>>(d_A, d_B, bw, N);
    checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaGetLastError());
}

extern "C" void gpuAddSharesModN(int numBits, uint32_t* d_A, uint32_t* d_B, int N) {
    const int thread_block_size = 128;
    assert(numBits == 2);
    int numInts = (numBits * N - 1) / 32 + 1;
    // printf("numInts: %d\n", numInts);
    addModN<<<(numInts - 1) / thread_block_size + 1, thread_block_size>>>(numBits, d_A, d_B, numInts);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

extern "C" void gpuXor(uint32_t* d_A, uint32_t* d_B, int N, Stats* s) {
    const int thread_block_size = 128;
    xorBits<<<(N - 1) / thread_block_size + 1, thread_block_size>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}


extern "C" void embedElementsWrapper(GPUGroupElement *h_A, double *h_A1,
                                                 double *h_A2, double *h_A3, double *h_A4, int N)
{
    GPUGroupElement *d_A;
    double *d_A1, *d_A2, *d_A3, *d_A4;

    unsigned long mem_size = N * sizeof(GPUGroupElement);

    cudaMalloc(&d_A, mem_size);
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMalloc((void **)&d_A1, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_A2, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_A3, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_A4, mem_size));

    embed_group_elements<<<(N - 1) / 128 + 1, 128>>>(d_A, d_A1, d_A2, d_A3, d_A4, N);

    cudaMemcpy(h_A1, d_A1, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A2, d_A2, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A3, d_A3, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A4, d_A4, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_A1);
    cudaFree(d_A2);
    cudaFree(d_A3);
    cudaFree(d_A4);
}

extern "C" void extractElementsWrapper(GPUGroupElement *h_A, double *h_A1,
                                                 double *h_A2, double *h_A3, double *h_A4, int N)
{
    GPUGroupElement *d_A;
    double *d_A1, *d_A2, *d_A3, *d_A4;

    unsigned long mem_size = N * sizeof(GPUGroupElement);

    cudaMalloc(&d_A, mem_size);

    checkCudaErrors(cudaMalloc((void **)&d_A1, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_A2, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_A3, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_A4, mem_size));

    cudaMemcpy(d_A1, h_A1, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2, h_A2, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A3, h_A3, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A4, h_A4, mem_size, cudaMemcpyHostToDevice);

    extract_group_elements<<<(N - 1) / 128 + 1, 128>>>(d_A, d_A1, d_A2, d_A3, d_A4, N);

    cudaMemcpy(h_A, d_A, mem_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_A1);
    cudaFree(d_A2);
    cudaFree(d_A3);
    cudaFree(d_A4);
}
// cudamemcpy, free, malloc

extern "C" void checkOutput(GPUGroupElement* h_O1, GPUGroupElement* h_O2, GPUGroupElement* h_R, int N) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto d_O1 = (GPUGroupElement*) moveToGPU((uint8_t*) h_O1, size_in_bytes, NULL);
    auto d_O2 = (GPUGroupElement*) moveToGPU((uint8_t*) h_O2, size_in_bytes, NULL);
    auto d_R = (GPUGroupElement*) moveToGPU((uint8_t*) h_R, size_in_bytes, NULL);
    check_output<<<(N-1)/128 + 1, 128>>>(h_O1, h_O2, h_R, N);
    cudaDeviceSynchronize();
}


extern "C" GPUGroupElement* gpuAddPool(GPUGroupElement* d_A, GPUConv2DKey k) {
    int H = k.OH / 2;
    int W = k.OW / 2;
    int num_elems = k.size_O / 4;
    assert(k.size_O == k.OH * k.OW * k.N * k.CO);
    auto d_B = (GPUGroupElement*) gpuMalloc(num_elems * sizeof(GPUGroupElement));
    add_pool<<<(num_elems-1)/128 + 1, 128>>>(d_A, d_B, k.N, k.CO, H, W, num_elems);
    return d_B;
}

extern "C" GPUGroupElement* gpuAddPoolBackProp(GPUGroupElement* d_A, GPUConv2DKey k) {
    int H = k.OH / 2;
    int W = k.OW / 2;
    int num_elems = k.size_O / 4;
    assert(k.size_O == k.OH * k.OW * k.N * k.CO);
    auto d_B = (GPUGroupElement*) gpuMalloc(4 * num_elems * sizeof(GPUGroupElement));
    add_pool_backprop<<<(num_elems-1)/128 + 1, 128>>>(d_A, d_B, k.N, k.CO, H, W, num_elems);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return d_B;
}


extern "C" GPUGroupElement* addPoolWrapper(GPUGroupElement* h_A, GPUConv2DKey k) {
    auto d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, k.mem_size_O, NULL);   
    auto d_B = gpuAddPool(d_A, k);
    auto h_B = (GPUGroupElement*) moveToCPU((uint8_t*) d_B, k.mem_size_O / 4, NULL);
    return h_B;
}

__device__ inline void mod(GPUGroupElement& x, int bw) {
    if(bw < 64) x &= ((1ULL << bw) - 1);
}

__global__ void addBiasKernel(int N, int M, int bw, GPUGroupElement* A, GPUGroupElement* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N * M) {
        int bIdx = i % M;
        A[i] += b[bIdx];
        mod(A[i], bw);
    }
}


extern "C" void gpuAddBias(int N, int M, int bw, GPUGroupElement* d_A, GPUGroupElement* h_b, Stats* s) {
    size_t memSizeB = M * sizeof(GPUGroupElement);
    auto d_b = (GPUGroupElement*) moveToGPU((uint8_t*) h_b, memSizeB, s);
    addBiasKernel<<<(M * N - 1) / block_size + 1, block_size>>>(N, M, bw, d_A, d_b);
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void gpuAddBiasWrapper(int N, int M, int bw, GPUGroupElement* h_A, GPUGroupElement* h_b) {
    size_t memSizeA = N * M * sizeof(GPUGroupElement);
    auto d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, memSizeA, NULL);
    gpuAddBias(N, M, bw, d_A, h_b, NULL);
    moveIntoCPUMem((uint8_t*) h_A, (uint8_t*) d_A, memSizeA, NULL);
}
// bias is an M vector
extern "C" GPUGroupElement* getBiasGrad(int N, int M, int bw, GPUGroupElement* d_A) {
    assert(bw == 64);
    GPUGroupElement* d_b = (GPUGroupElement*) gpuMalloc(M * sizeof(GPUGroupElement));
    const int kV = 1;
    using TensorReduction = cutlass::reduction::device::TensorReduction<
        GPUGroupElement, // output
        GPUGroupElement, // source
        cutlass::layout::TensorNHWC, // Layout
        cutlass::plus<GPUGroupElement>, //Functor
        kV, //kV
        GPUGroupElement //ElementCompute
    >;
    
    auto t_A = toTensorRef(d_A, 1, 1, N, M);
    auto t_b = toTensorRef(d_b, 1, 1, 1, M);

    TensorReduction reduction(/*t_A.extent()*/{1, 1, N, M}, 2);
    
    uint8_t* workspace = gpuMalloc(reduction.workspace_size());

    cutlass::Status status = reduction.reduce(
            t_b/*.device_ref()*/, // dst_tensor
            t_A/*.device_ref()*/, // src_tensor
            workspace, // device_workspace
            GPUGroupElement(0) //reduction_identity
          );
    CUTLASS_CHECK(status);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(workspace);

    return d_b;
}

extern "C" GPUGroupElement* getBiasGradWrapper(int N, int M, int bw, GPUGroupElement* h_A) {
    size_t memSizeA = N * M * sizeof(GPUGroupElement);
    auto d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, memSizeA, NULL);
    auto d_b = getBiasGrad(N, M, bw, d_A);
    size_t memSizeB = M * sizeof(GPUGroupElement);
    auto h_b = (GPUGroupElement*) moveToCPU((uint8_t*) d_b, memSizeB, NULL);
    gpuFree(d_A);
    gpuFree(d_b);
    return h_b;
}

__global__ void leftShiftAndAddKernel(GPUGroupElement* A, GPUGroupElement* B, GPUGroupElement* C, int shift, GPUGroupElement alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        C[i] = (A[i] << shift) + alpha * B[i];
    }
}


extern "C" void gpuLeftShiftAndAdd(int N, GPUGroupElement* d_A, GPUGroupElement* d_B, GPUGroupElement* d_C, int shift, GPUGroupElement alpha) {
    leftShiftAndAddKernel<<<(N - 1) / block_size + 1, block_size>>>(d_A, d_B, d_C, shift, alpha, N);
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void gpuLeftShiftAndAddWrapper(int N, GPUGroupElement* h_A, GPUGroupElement* h_B, GPUGroupElement* h_C, int shift, GPUGroupElement alpha) {
    size_t memSize = N * sizeof(GPUGroupElement);
    auto d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, memSize, NULL);
    auto d_B = (GPUGroupElement*) moveToGPU((uint8_t*) h_B, memSize, NULL);
    gpuLeftShiftAndAdd(N, d_A, d_B, d_A, shift, alpha);
    moveIntoCPUMem((uint8_t*) h_C, (uint8_t*) d_A, memSize, NULL);
    gpuFree(d_A);
    gpuFree(d_B);
}