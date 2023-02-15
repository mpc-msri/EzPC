// #include "group_element.h"
#include <cryptoTools/Common/Defines.h>
// #include "gpu_aes.cu"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
// #include "gpu_data_types.h"
#include "helper_cuda.h"
#include "gpu_dcf.cu"
#include "gpu_mem.h"
#include "maxpool_layer.h"
#include "gpu_select.h"

using namespace std;

typedef uint64_t GPUGroupElement;
typedef unsigned __int128 AESBlock;

#define SERVER1 1
#define AES_BLOCK_LEN_IN_BITS 128
// #define FULL_MASK 0xffffffff

__device__ void traversePathForTwoEvals(int Bin, int Bout, int output_vec_len, int party,
                                        GPUGroupElement x1,
                                        GPUGroupElement x2,
                                        AESBlock *scw,
                                        GPUGroupElement *v1,
                                        GPUGroupElement *v2,
                                        GPUGroupElement *vcw,
                                        bool geq,
                                        int evalGroupIdxStart,
                                        int evalGroupIdxLen,
                                        AESBlock *s1, AESBlock *s2, int num_dcfs, AESSharedContext* saes)
{
    // can probaby save one AES evaluation here
    *s1 = scw[0];
    *s2 = scw[0];

    x1 = __brevll(x1) >> (64 - Bin);
    x2 = __brevll(x2) >> (64 - Bin);

    for (int i = 0; i < Bin; ++i)
    {
        const osuCrypto::u8 keep1 = lsb(x1);
        auto vcwI = getVCW(Bout, vcw, num_dcfs, i);
        *s1 = traverseOneDCF(Bin, Bout, output_vec_len, party, *s1,
                             scw[(i + 1) * num_dcfs], keep1, v1, /*vcw[i * num_dcfs]*/ vcwI, i, geq, evalGroupIdxStart, evalGroupIdxLen, saes);
        x1 = x1 >> 1;

        const osuCrypto::u8 keep2 = lsb(x2);
        *s2 = traverseOneDCF(Bin, Bout, output_vec_len, party, *s2,
                             scw[(i + 1) * num_dcfs], keep2, v2, /*vcw[i * num_dcfs]*/ vcwI, i, geq, evalGroupIdxStart, evalGroupIdxLen, saes);
        x2 = x2 >> 1;
    }
    return; 
}

// __device__ void writeDReluOutput(GPUGroupElement* dReluOutput, GPUGroupElement dReluBit, int bout, int N) {
//     int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < N);
//     assert(threadId < N);
//         int laneId = threadIdx.x & 0x1f;
//         if(bout == 1) {
//             int dreluAsInt = static_cast<int>(dReluBit);
//             dreluAsInt <<= laneId;
//             for (int j = 16; j >= 1; j /= 2)
//                 dreluAsInt += __shfl_down_sync(mask, dreluAsInt, j, 32);
//             if (laneId == 0) {
//                 ((uint32_t*) dReluOutput)[threadId / 32] = static_cast<uint32_t>(dreluAsInt);
//             }
//         } else if(bout == 2) {
//             // printf("%d: %lu\n", threadId, dReluBit);
//             dReluBit <<= (2 * laneId);
//             for(int j = 16; j >= 1; j /= 2)
//                 dReluBit += __shfl_down_sync(mask, dReluBit, j, 32);
//             if(laneId == 0) {
//                 dReluOutput[threadId / 32] = dReluBit;
//                 // printf("%d: %lu\n", threadId, dReluBit);
//             }
//         } else if(bout == 64) {
//             dReluOutput[threadId] = dReluBit;
//         }
// }

/* out needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
__global__ void dReluKernel(int Bin, int Bout, int output_vec_len,
                            GPUGroupElement *v1, GPUGroupElement *dReLUOutput, // groupSize
                            int party, GPUGroupElement *dcf_input,    // might want to pass a pointer to this later
                            AESBlock *scw,                            // k.Bin + 1
                            GPUGroupElement *vcw,                     // k.Bin * groupSize
                            GPUGroupElement *dReLUMask,
                            GPUGroupElement *xLTRinMask,
                            bool geq /*= false*/, int evalGroupIdxStart /*= 0*/,
                            int evalGroupIdxLen /*= -1*/, int num_dcfs, AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < num_dcfs);
    if(thread_id < num_dcfs) {
        scw = &scw[thread_id/* * (Bin + 1)*/];
        // vcw = &vcw[thread_id/* * (Bin + 1)*/];

        /* v_share needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
        /* need to make v_share a local variable and avoid writing to global memory */
        /* might change this later if it adversely impacts performance */
        // v_share = &v_share[thread_id];
        GPUGroupElement local_v1 = 0, local_v2 = 0;

        GPUGroupElement x = dcf_input[thread_id];
        GPUGroupElement x1 = x & ((1ULL << Bin) - 1);
        GPUGroupElement x2 = (x + (1ULL << (Bin - 1))) & ((1ULL << Bin) - 1);
        
        AESBlock s1, s2;
        traversePathForTwoEvals(Bin, Bout,
                            output_vec_len, party, x1, x2, scw, &local_v1, &local_v2, vcw, geq,
                            evalGroupIdxStart, evalGroupIdxLen, &s1, &s2, num_dcfs, &saes);
    // need to optimize the use of memory here
        auto vcwBin = getVCW(Bout, vcw, num_dcfs, Bin);
        getDCFOutput(Bout, s1, vcwBin/*vcw[num_dcfs * Bin */, party, &local_v1);
        getDCFOutput(Bout, s2, vcwBin/*vcw[num_dcfs * Bin]*/, party, &local_v2);
        // printf("dcf output %d: %lu %lu\n", thread_id, local_v1, local_v2);

        if(v1) {
            GPUGroupElement v1Mask = 0;
            if(xLTRinMask) v1Mask = getVCW(Bout, xLTRinMask, num_dcfs, 0);
            // printf("v1 mask: %lu\n", v1Mask);
            auto maskedV1 = (local_v1 + v1Mask) & output_bit_mask;
            writeDReluOutput(v1, maskedV1, Bout, num_dcfs);//v1[thread_id] = local_v1;
        }
        auto mask = getVCW(Bout, dReLUMask, num_dcfs, 0);
        // printf("dcf output %d: %lu %lu %lu\n", thread_id, local_v1, local_v2, mask);
        if (party == SERVER1)
            local_v2 = local_v2 + (x2 >= (1ULL << (Bin - 1)));
        auto dreluBit = (local_v2 - local_v1 + /*dReLUMask[thread_id]*/ mask) & output_bit_mask;//static_cast<GPUGroupElement>(1);
        writeDReluOutput(dReLUOutput, dreluBit, Bout, num_dcfs);
        // printf("dcf output %d: %lu\n", thread_id, dreluBit);
        // int drelu_as_int = static_cast<int>(dReLUOutput);
        // int laneId = threadIdx.x & 0x1f;
        // drelu_as_int <<= laneId;
        // for (int j = 16; j >= 1; j /= 2)
        //     drelu_as_int += __shfl_down_sync(mask, drelu_as_int, j, 32);
        // if (laneId == 0)
        // {
        //     dReLU[thread_id / 32] = drelu_as_int;
        // }

    }
}

/* out needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
__global__ void computeDiffWithCurMax(MaxpoolParams p, int fh, int fw, GPUGroupElement* curMax, GPUGroupElement* img, GPUGroupElement* diff, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < N) {
        int t = thread_id;
        int n = t / (p.H * p.W * p.C);
        t = t % (p.H * p.W * p.C);
        int h = t / (p.W * p.C);
        t = t % (p.W * p.C);
        int w = t / p.C;
        int c = t % p.C;
        int leftTopCornerH = h * p.strideH - p.zPadHLeft;
        int leftTopCornerW = w * p.strideW - p.zPadWLeft;
        int posH = leftTopCornerH + fh;
        int posW = leftTopCornerW + fw;
        assert(posH >= 0 && posH <= p.imgH);
        assert(posW >= 0 && posW <= p.imgW);
        GPUGroupElement toCmp1;
        if(fh == 0 && fw == 1) {
            toCmp1 = img[n * p.imgH * p.imgW * p.C + leftTopCornerH * p.imgW * p.C + leftTopCornerW * p.C + c];
            curMax[thread_id] = toCmp1;
        } else toCmp1 = curMax[thread_id];
        GPUGroupElement toCmp2 = img[n * p.imgH * p.imgW * p.C + posH * p.imgW * p.C + posW * p.C + c];
        // printf("%d: %lu %lu\n", thread_id, toCmp1, toCmp2);
        diff[thread_id] = (toCmp2 - toCmp1) & ((1ULL << p.bin) - 1); 
    }
}



__global__ void evalDCFForlrs(int lrsBin, int dcfBin, int Bout, int output_vec_len,
                              /*GPUGroupElement *v,     */               // groupSize
                              int party, GPUGroupElement *dcf_input, // might want to pass a pointer to this later
                              AESBlock *scw,                         // k.Bin + 1
                              GPUGroupElement *vcw,                  // k.Bin * groupSize
                              GPUGroupElement *lrs_tn,
                              GPUGroupElement *lrsMask,
                              bool geq /*= false*/, int evalGroupIdxStart /*= 0*/,
                              int evalGroupIdxLen /*= -1*/, int shift, int num_dcfs, AESGlobalContext gaes)
{
    // need to add a < N check here
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < num_dcfs) {
        scw = &scw[thread_id/* * (Bin + 1)*/];
        // vcw = &vcw[thread_id/* * (Bin + 1)*/];

        GPUGroupElement local_v = 0;
        GPUGroupElement x = dcf_input[thread_id];
        GPUGroupElement x0 = x % (1ULL << shift);
        GPUGroupElement x1 = x / (1ULL << shift);
        GPUGroupElement xs = (1ULL << shift) - 1 - x0;

        AESBlock s;
        /*auto s =*/ traversePathDCF(dcfBin, Bout,
                             output_vec_len, party, xs, scw, &local_v, vcw, geq,
                             evalGroupIdxStart, evalGroupIdxLen, &s, num_dcfs, &saes);
        auto vcwBin = getVCW(Bout, vcw, num_dcfs, dcfBin);
        getDCFOutput(Bout, s, /*vcw[num_dcfs * Bin]*/ vcwBin, party, &local_v);

        if (party == SERVER1)
            local_v += x1;

        lrsMask[thread_id] += ((1ULL << (lrsBin - shift)) * lrs_tn[thread_id] + local_v);
    }
}
// no memory leak
extern "C" std::pair<GPUGroupElement*, GPUGroupElement*> evalDRelu(GPUDCFKey k, int party, GPUGroupElement *d_in, GPUGroupElement* h_dReLUMask, GPUGroupElement* h_xLTRinMask, AESGlobalContext* gaes, Stats* stats, bool returnXLTRin)
{   
    const int tb_size = 256;
    int num_thread_blocks = ((k.num_dcfs - 1) / tb_size) + 1;

    // printf("drelu params: %d %d\n", k.Bin, k.Bout);
    GPUGroupElement h_output_bit_mask = static_cast<GPUGroupElement>(-1) >> (64 - k.Bout);
    // printf("output bit mask %lu\n", h_output_bit_mask);
    checkCudaErrors(cudaMemcpyToSymbol(output_bit_mask, &h_output_bit_mask, sizeof(GPUGroupElement)));
    AESBlock* d_scw = (AESBlock*) moveToGPU((uint8_t*) k.scw, k.mem_size_scw, stats);
    GPUGroupElement* d_vcw = (GPUGroupElement*) moveToGPU((uint8_t*) k.vcw, k.mem_size_vcw, stats);

    GPUGroupElement *d_v1 = NULL, *d_xLTRinMask = NULL;
    assert(k.mem_size_vcw % (k.Bin + 1) == 0);
    size_t size_in_bytes = k.mem_size_vcw / (k.Bin + 1);//k.num_dcfs * sizeof(GPUGroupElement);
    if(returnXLTRin) {
        d_v1 = (GPUGroupElement*) gpuMalloc(size_in_bytes);
        if(h_xLTRinMask) d_xLTRinMask = (GPUGroupElement*) moveToGPU((uint8_t*) h_xLTRinMask, size_in_bytes, stats);
    }
    
    // size_t drelu_size_in_bytes = ((k.num_dcfs - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    GPUGroupElement *d_dReLU = (GPUGroupElement*) gpuMalloc(size_in_bytes);
    GPUGroupElement *d_dReLUMask = (GPUGroupElement*) moveToGPU((uint8_t*) h_dReLUMask, size_in_bytes, stats);
    dReluKernel<<<num_thread_blocks, tb_size>>>(k.Bin, k.Bout, k.out_vec_len,
                                                d_v1, d_dReLU, party, d_in, d_scw, d_vcw, d_dReLUMask, d_xLTRinMask, false, 0, -1, k.num_dcfs, *gaes);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    gpuFree(d_vcw);
    gpuFree(d_scw);
    gpuFree(d_dReLUMask);
    if(d_xLTRinMask) gpuFree(d_xLTRinMask);

    return std::make_pair(d_v1, d_dReLU);
}

GPUGroupElement *evalLrs(int lrsBin, GPUDCFKey k, int party, GPUGroupElement *dcf_input, GPUGroupElement* d_lrs_tn, GPUGroupElement *h_lrsMask, int shift, AESGlobalContext* gaes, Stats* stats)
{

    const int tb_size = 256;
    int num_thread_blocks = ((k.num_dcfs - 1) / tb_size) + 1;

    GPUGroupElement h_output_bit_mask = static_cast<GPUGroupElement>(-1) >> (64 - k.Bout);
    checkCudaErrors(cudaMemcpyToSymbol(output_bit_mask, &h_output_bit_mask, sizeof(GPUGroupElement)));

    // might be allocating more memory here than necessary
    uint64_t out_size_in_bytes = k.num_dcfs * sizeof(GPUGroupElement);

    AESBlock* d_scw = (AESBlock*) moveToGPU((uint8_t*) k.scw, k.mem_size_scw, stats);
    GPUGroupElement* d_vcw = (GPUGroupElement*) moveToGPU((uint8_t*) k.vcw, k.mem_size_vcw, stats);
    GPUGroupElement* d_lrsMask = (GPUGroupElement*) moveToGPU((uint8_t*) h_lrsMask, out_size_in_bytes, stats);

    evalDCFForlrs<<<num_thread_blocks, tb_size>>>(lrsBin, k.Bin, k.Bout, k.out_vec_len,
                                                  /*d_v,*/ party, dcf_input, d_scw, d_vcw, d_lrs_tn, d_lrsMask, false, 0, -1, shift, k.num_dcfs, *gaes);

    // implicit synchronization
    gpuFree(d_vcw);
    gpuFree(d_scw);
    return d_lrsMask;
}

extern "C" void gpu_relu_truncate(GPUReLUTruncateKey k,
                                  int party, GPUGroupElement *d_in,
                                  GPURTContext *rtc, AESGlobalContext* gaes, Stats* stats)
{
    auto dReLUOutput = evalDRelu(k.dcfKeyN, party, d_in, /*d_dReLUMask*/ k.a, NULL, gaes, stats, true);
    GPUGroupElement *d_lrs_tn = dReLUOutput.first;
    uint32_t *d_drelu = (uint32_t*) dReLUOutput.second;

    auto d_lrs = evalLrs(k.Bin, k.dcfKeyS, party, d_in, d_lrs_tn, k.zTruncate, k.shift, gaes, stats);
    gpuFree(d_in);

    rtc->d_lrs0 = d_lrs;
    rtc->d_drelu0 = d_drelu;
}

__device__ GPUGroupElement evalArs(GPUGroupElement x, int bin, int shift) {
    GPUGroupElement msb = (x & (1ULL << (bin - 1))) >> (bin - 1);
    GPUGroupElement signMask = (((1ULL << shift) - msb) << (64 - shift));
    x = (x >> shift) | signMask;
    return x;
}

__global__ void evalArsKernel(GPUGroupElement* x, int bin, int shift, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < N) {
        x[thread_id] = evalArs(x[thread_id], bin, shift);
    }
}

extern "C" void gpuLocalARS(int bin, int shift, int N, GPUGroupElement* d_I) {
    assert(bin >= shift);
    evalArsKernel<<<(N - 1) / 128 + 1, 128>>>(d_I, bin, shift, N);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

extern "C" void gpuLocalARSWrapper(int bin, int shift, int N, GPUGroupElement* h_I, GPUGroupElement* h_O) {
    assert(bin >= shift);
    size_t memSizeI = N * sizeof(GPUGroupElement);
    auto d_I = (GPUGroupElement*) moveToGPU((uint8_t*) h_I, memSizeI, NULL);
    evalArsKernel<<<(N - 1) / 128 + 1, 128>>>(d_I, bin, shift, N);
    checkCudaErrors(cudaDeviceSynchronize());
    moveIntoCPUMem((uint8_t*) h_O, (uint8_t*) d_I, memSizeI, NULL);
    gpuFree(d_I);
}


extern "C" void gpu_local_truncate_relu(GPUReLUTruncateKey k,
                                  int party, /*int bin,*/ GPUGroupElement *d_in,
                                  GPURTContext *rtc, AESGlobalContext* gaes, Stats* stats)
{
    assert(k.Bin >= k.shift);
    evalArsKernel<<<(k.num_rts - 1) / 256 + 1, 256>>>(d_in, k.Bin, k.shift, k.num_rts);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    auto dReLUOutput = evalDRelu(k.dcfKeyN, party, d_in, /*d_dReLUMask*/ k.a2, NULL, gaes, stats, false);
    uint32_t *d_drelu = (uint32_t*) dReLUOutput.second;
    rtc->d_lrs0 = d_in;
    rtc->d_drelu0 = d_drelu;
}
// no memory leak
extern "C" std::pair<uint32_t*, GPUGroupElement*> gpuDReluForMaxPool(MaxpoolParams p, GPUDReluKey k, int party, int fh, int fw,
                                  GPUGroupElement* d_curMax, GPUGroupElement *d_in,
                                  AESGlobalContext* gaes, Stats* stats)
{
    int out_size = p.N * p.H * p.W * p.C;
    // printf("allocating diff: %d\n", out_size);
    // if(!d_curMax) d_curMax = (GPUGroupElement*) gpuMalloc(out_size * sizeof(GPUGroupElement));
    GPUGroupElement* d_diff = (GPUGroupElement*) gpuMalloc(out_size * sizeof(GPUGroupElement));
    int tb_size = 256;
    computeDiffWithCurMax<<<(out_size - 1) / tb_size + 1, tb_size>>>(p, fh, fw, d_curMax, d_in, d_diff, out_size);
    auto dReLUOutput = evalDRelu(k.dcfKey, party, d_diff, (GPUGroupElement*) k.dReluMask, NULL, gaes, stats, false);
    uint32_t *d_drelu = (uint32_t*) dReLUOutput.second;
    return std::make_pair(d_drelu, d_diff);
}

extern "C" void gpu_local_truncate(GPUReLUTruncateKey k,
                                  /*int bin,*/ GPUGroupElement *d_in,
                                  Stats* stats)
{
    evalArsKernel<<<(k.num_rts - 1) / 256 + 1, 256>>>(d_in, k.Bin, k.shift, k.num_rts);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}


extern "C" std::pair<GPUGroupElement *, uint32_t *> gpuReluTruncateWrapper(GPUReLUTruncateKey k,
                                                                           int party, GPUGroupElement *rt_input, AESGlobalContext* gaes)
{
    GPURTContext rtc;
    GPUGroupElement* d_in;

    size_t lrs_size_in_bytes = k.num_rts * sizeof(GPUGroupElement);    
    size_t drelu_size_in_bytes = (k.num_rts - 1) / 8 + 1;

    cudaMalloc(&d_in, lrs_size_in_bytes);
    cudaMemcpy(d_in, rt_input, lrs_size_in_bytes, cudaMemcpyHostToDevice);
    
    gpu_relu_truncate(k, party, d_in, &rtc, gaes, NULL);

    GPUGroupElement* h_lrs0;
    uint32_t *h_drelu0;

    cudaMallocHost(&h_lrs0, lrs_size_in_bytes);
    cudaMallocHost(&h_drelu0, drelu_size_in_bytes);

    cudaMemcpy(h_lrs0, rtc.d_lrs0, lrs_size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_drelu0, rtc.d_drelu0, drelu_size_in_bytes, cudaMemcpyDeviceToHost);

    gpuFree(d_in);
    gpuFree(rtc.d_lrs0);
    gpuFree(rtc.d_drelu0);

    return std::make_pair(h_lrs0, h_drelu0);
}

__global__ void ReLUTruncateMult(uint32_t *x1, uint32_t *x2, 
                                GPUGroupElement *y1, GPUGroupElement *y2,
                                GPUGroupElement *a, GPUGroupElement *b, 
                                GPUGroupElement *c, GPUGroupElement *d1, 
                                GPUGroupElement *d2, int party, int N, 
                                bool truncateY, bool saveX, int shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < N);
    if(i < N) {
        int laneId = threadIdx.x & 0x1f;
        GPUGroupElement x1_as_group_element = (x1[i / 32] >> laneId) & static_cast<GPUGroupElement>(1);
        GPUGroupElement x2_as_group_element = 0;
        if(x2) x2_as_group_element = (x2[i / 32] >> laneId) & static_cast<GPUGroupElement>(1);
        GPUGroupElement x = (x1_as_group_element + x2_as_group_element) & static_cast<GPUGroupElement>(1);

        GPUGroupElement y = y1[i] + (y2 ? y2[i] : 0);
        GPUGroupElement is_zero_x = (x == 0);
        a[i] = -a[i] * y - b[i] * x + c[i] + y * is_zero_x * d1[i] + is_zero_x * d2[i] + (party == SERVER1) * x * y;
        if(saveX) {
            int drelu_as_int = static_cast<int>(x);
            // int laneId = threadIdx.x & 0x1f;
            drelu_as_int <<= laneId;
            for (int j = 16; j >= 1; j /= 2)
                drelu_as_int += __shfl_down_sync(mask, drelu_as_int, j, 32);
            if (laneId == 0)
            {
                x1[i / 32] = drelu_as_int;
            }
        }
    }
}

__global__ void selectForMaxpoolBackpropKernel(MaxpoolParams p, uint32_t *oneHot, 
                                GPUGroupElement *incomingGrad, 
                                GPUGroupElement *out,
                                GPUGroupElement *a, GPUGroupElement *b, 
                                GPUGroupElement *e, GPUGroupElement *d1, 
                                GPUGroupElement *d2, int party, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        int laneId = threadIdx.x & 0x1f;
        int t = i;
        int n = t / (p.H * p.W * p.C * p.FH * p.FW);
        t = t % (p.H * p.W * p.C * p.FH * p.FW);
        int h = t / (p.W * p.C * p.FH * p.FW);
        t = t % (p.W * p.C * p.FH * p.FW);
        int w = t / (p.C * p.FH * p.FW);
        t = t % (p.C * p.FH * p.FW);
        int c = t / (p.FH * p.FW);
        // t = t % (p.FH * p.FW);
        // int fh = t / p.FW;
        // int fw = t % p.FW;

        // int leftTopCornerH = h * p.strideH - p.zPadHLeft;
        // int leftTopCornerW = w * p.strideW - p.zPadWLeft;

        // int curPosH = leftTopCornerH + fh;
        // int curPosW = leftTopCornerW + fw;

        // int imgIdx = n * p.imgH * p.imgW * p.C + curPosH * p.imgW * p.C + curPosW * p.C + c;
        // printf("%d %d %d %d <-- %d %d %d %d %d %d\n", n, curPosH, curPosW, c, n, h, w, c, fh, fw);

        GPUGroupElement x = (oneHot[i / 32] >> laneId) & static_cast<GPUGroupElement>(1);
        // printf("%d: %lu, %d: %lu\n", i, x, imgIdx, outgoingGrad[imgIdx]);
        GPUGroupElement is_zero_x = (x == 0);
        int j = n * p.H * p.W * p.C + h * p.W * p.C + w * p.C + c;
        GPUGroupElement y = incomingGrad[j];
        // printf("%d: %lu %lu\n", imgIdx, (-a[i] * y - b[i] * x + e[i] + y * is_zero_x * d1[i] + is_zero_x * d2[i] + (party == SERVER1) * (x * y)), outgoingGrad[imgIdx]);
        out[i] = (-a[i] * y - b[i] * x + e[i] + y * is_zero_x * d1[i] + is_zero_x * d2[i] + (party == SERVER1) * (x * y));
    }
}

__device__ GPUGroupElement select(uint32_t *xArr, 
                                GPUGroupElement *yArr,
                                GPUGroupElement *a, GPUGroupElement *b, 
                                GPUGroupElement *c, GPUGroupElement *d1, 
                                GPUGroupElement *d2, int party, int N, int i)
{
    assert(i < N);
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if(i < N) {
    int laneId = threadIdx.x & 0x1f;
    GPUGroupElement x = (xArr[i / 32] >> laneId) & 1ULL;
    GPUGroupElement is_zero_x = (x == 0);
    GPUGroupElement y = yArr[i];
    GPUGroupElement selectOutput = -a[i] * y - b[i] * x + c[i] + y * is_zero_x * d1[i] + 
                is_zero_x * d2[i] + (party == SERVER1) * (x * y);
    return selectOutput;
    // }
}

__global__ void selectForMaxpoolKernel(uint32_t *drelu, 
                                GPUGroupElement *diff, /*GPUGroupElement *curMax,*/
                                GPUGroupElement *a, GPUGroupElement *b, 
                                GPUGroupElement *c, GPUGroupElement *d1, 
                                GPUGroupElement *d2, int party, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        auto selectOutput = select(drelu, diff, a, b, c, d1, d2, party, N, i);
        // if(party == SERVER1 && curMax != NULL) selectOutput += curMax[i];
        a[i] = selectOutput;
    }
}



__global__ void andForMaxpoolKernel(MaxpoolParams p, int pos, uint32_t* dreluBits, uint32_t* oneHotBits,
    uint32_t* b0Bits, uint32_t* b1Bits, uint32_t* b2Bits, int party, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < N);
    if(i < N) {
        int laneId = threadIdx.x & 0x1f;
        int t = i;
        int n = t / (p.H * p.W * p.C * p.FH * p.FW);
        t = t % (p.H * p.W * p.C * p.FH * p.FW);
        int h = t / (p.W * p.C * p.FH * p.FW);
        t = t % (p.W * p.C * p.FH * p.FW);
        int w = t / (p.C * p.FH * p.FW);
        t = t % (p.C * p.FH * p.FW);
        int c = t / (p.FH * p.FW);
        int q = t % (p.FH * p.FW);
        int newOneHot = 0;
        int idx = i / 32;
        if(q < pos) {
            // need to check this once
            int dreluIndex = n * p.H * p.W * p.C + h * p.W * p.C + w * p.C + c;
            // printf("%d: %d %d\n", i, dreluIndex / 32, idx);
            uint32_t drelu = (dreluBits[dreluIndex / 32] >> (dreluIndex % 32)) & 1; 
            // int idx = (dreluIndex * p.FH * p.FW + q) / 32;
            uint32_t oneHot;// = 0;
            if(pos == 2 && q == 0) oneHot = 1; 
            else if(pos == 2) oneHot = 0;
            else oneHot = (oneHotBits[idx] >> laneId) & 1;
            uint32_t incomingOneHot = (q == pos - 1 ? 1 : 0);
            uint32_t diff = (incomingOneHot - oneHot) & 1;
            int keyNum = dreluIndex * pos + q;
            int keyIdx = keyNum / 32;
            int keyPos = keyNum % 32;
            uint32_t b0 = (b0Bits[keyIdx] >> keyPos) & 1;
            uint32_t b1 = (b1Bits[keyIdx] >> keyPos) & 1;
            uint32_t b2 = (b2Bits[keyIdx] >> keyPos) & 1;
            // printf("drelu: %d diff: %d b0: %d b1: %d b2: %d curOneHot: %d %u %u %u\n", drelu, diff, b0, b1, b2, oneHot, b0Bits[idx], b1Bits[idx], b2Bits[idx]);
            newOneHot = (-b0 * diff - drelu * b1 + b2 + (party == SERVER1) * (drelu * diff + oneHot)) & 1;
            // printf("%d %d %d %d %d %d %d: %d %d %d %d %d %d\n", n, h, w, c, q, pos, dreluIndex, drelu, oneHot, incomingOneHot, newOneHot, keyIdx, laneId);
        }
        // printf("%d: %d\n", i, newOneHot);
        assert(newOneHot == 0 || newOneHot == 1);
        newOneHot <<= laneId;
        for (int j = 16; j >= 1; j /= 2)
            newOneHot += __shfl_down_sync(mask, newOneHot, j, 32);
        if (laneId == 0)
        {
            // printf("new one hot: %d %d\n", idx, newOneHot);
            oneHotBits[idx] = static_cast<uint32_t>(newOneHot);
        }
        // update uint32_t in tandem
    }
}

__global__ void gpuCollectGradientsKernel(MaxpoolParams p, GPUGroupElement* outgoingGradExpanded, GPUGroupElement* outgoingGrad, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        int t = i;
        int n = t / (p.imgH * p.imgW * p.C);
        t = t % (p.imgH * p.imgW * p.C);
        int h = t / (p.imgW * p.C);
        t = t % (p.imgW * p.C);
        int w = t / (p.C);
        int c = t % (p.C);
        GPUGroupElement sumGrads = 0;
        for(int fh = 0; fh < p.FH; fh++) {
            for(int fw = 0; fw < p.FW; fw++) {
                int leftTopCornerH = h - fh;
                int leftTopCornerW = w - fw;
                int rightTopCornerH = leftTopCornerH;
                int rightTopCornerW = leftTopCornerW + p.FW - 1;
                int leftBottomCornerH = leftTopCornerH + p.FH - 1;
                int leftBottomCornerW = leftTopCornerW;;
                int rightBottomCornerH = leftTopCornerH + p.FH - 1;
                int rightBottomCornerW = leftTopCornerW + p.FW - 1;
                if(leftTopCornerH >= 0 && leftTopCornerW >= 0 && 
                   rightTopCornerH >= 0 && rightTopCornerW < p.imgW &&
                   leftBottomCornerH < p.imgH && leftBottomCornerW >= 0 &&
                   rightBottomCornerH < p.imgH && rightBottomCornerW < p.imgW &&
                   leftTopCornerH % p.strideH == 0 && leftTopCornerW % p.strideW == 0) {
                    int gradH = leftTopCornerH / p.strideH;
                    int gradW = leftTopCornerW / p.strideW;
                    int idx = n * p.H * p.W * p.C * p.FH * p.FW + gradH * p.W * p.C * p.FH * p.FW + gradW * p.C * p.FH * p.FW + c * p.FH * p.FW + fh * p.FW + fw;
                    sumGrads += outgoingGradExpanded[idx];
                }
            }
        }
        outgoingGrad[i] = (sumGrads & ((1ULL << p.bin) - 1));
    }
}

__global__ void reluSignExtendMuxKernel(int party, int bin, /*int f,*/ int N, GPUGroupElement* x, GPUGroupElement* oneHot, GPUGroupElement* outMask, GPUGroupElement* drelu, GPUGroupElement* xLTRin) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(j < N) {
        int posInBlock = threadIdx.x & 0xf;
        uint32_t d = (((uint32_t*) drelu)[j / 16] >> (2 * posInBlock)) & 3;
        uint32_t w = (((uint32_t*) xLTRin)[j / 16] >> (2 * posInBlock)) & 3;
        uint32_t i = (2 * d + w) & 3;
        // should i store this table transposed instead?
        // will always access sequential elements so might benefit from locality within a thread
        GPUGroupElement rotatedP3 = oneHot[4 * j + ((2 - i) & 3)];
        GPUGroupElement rotatedP4 = oneHot[4 * j + ((3 - i) & 3)];
        GPUGroupElement xIn = x[j];
        // printf("mux %d: %d %d\n", j, ((2 - i) & 3), ((3 - i) & 3));
        // printf("drelu %d: %u %u %u %u %lu %lu %lu %d %lu\n", j, d, w, i, 4 * j + ((2 - i) & 3), rotatedP3, rotatedP4, outMask[2 * j + (d & 1)], bin, xIn);
        x[j] = xIn * rotatedP3 + (xIn + (1ULL << (bin))) * rotatedP4 + outMask[2 * j + (d & 1)];
        GPUGroupElement dreluBit = static_cast<GPUGroupElement>(d & 1);
        writeDReluOutput(xLTRin, dreluBit, 1, N);
    }
}



/* need to coalesce global memory accesses */

extern "C" std::pair<GPUGroupElement*, uint32_t*> finish_relu_truncate(GPUReLUTruncateKey k,
                                                 uint32_t *d_x1, GPUGroupElement *d_y1,/* GPUGroupElement *d_a,*/
                                                 uint32_t *d_x2, GPUGroupElement *d_y2, int party, bool truncateY, bool saveX, Stats* stats)
{
    size_t size_in_bytes = k.num_rts * sizeof(GPUGroupElement);

    GPUGroupElement *d_a = (GPUGroupElement*) moveToGPU((uint8_t*) k.a, size_in_bytes, stats);
    GPUGroupElement *d_b = (GPUGroupElement*) moveToGPU((uint8_t*) k.b, size_in_bytes, stats);
    GPUGroupElement *d_c = (GPUGroupElement*) moveToGPU((uint8_t*) k.c, size_in_bytes, stats);
    GPUGroupElement *d_d1 = (GPUGroupElement*) moveToGPU((uint8_t*) k.d1, size_in_bytes, stats);
    GPUGroupElement *d_d2 = (GPUGroupElement*) moveToGPU((uint8_t*) k.d2, size_in_bytes, stats);

    const int tb_size = 256;

    ReLUTruncateMult<<<(k.num_rts - 1) / tb_size + 1, tb_size>>>(d_x1, d_x2, d_y1, d_y2, d_a, d_b, d_c, d_d1, d_d2, party, k.num_rts, truncateY, saveX, k.shift);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    if(!saveX) gpuFree(d_x1);
    gpuFree(d_y1);
    if(d_x2) gpuFree(d_x2);
    if(d_y2) gpuFree(d_y2);
    gpuFree(d_b);
    gpuFree(d_c);
    gpuFree(d_d1);
    gpuFree(d_d2);
    
    return std::make_pair(d_a, d_x1);
}

extern "C" GPUGroupElement* gpuSelectForMaxpool(GPUSelectKey k,
                                                uint32_t *d_drelu,
                                                GPUGroupElement *d_diff, /*GPUGroupElement* d_curMax,*/
                                                int party, Stats* stats)
{
    // printf("selectKey.N: %d\n", k.N);
    size_t size_in_bytes = k.N * sizeof(GPUGroupElement);

    GPUGroupElement *d_a = (GPUGroupElement*) moveToGPU((uint8_t*) k.a, size_in_bytes, stats);
    GPUGroupElement *d_b = (GPUGroupElement*) moveToGPU((uint8_t*) k.b, size_in_bytes, stats);
    GPUGroupElement *d_c = (GPUGroupElement*) moveToGPU((uint8_t*) k.c, size_in_bytes, stats);
    GPUGroupElement *d_d1 = (GPUGroupElement*) moveToGPU((uint8_t*) k.d1, size_in_bytes, stats);
    GPUGroupElement *d_d2 = (GPUGroupElement*) moveToGPU((uint8_t*) k.d2, size_in_bytes, stats);

    const int tb_size = 256;
    selectForMaxpoolKernel<<<(k.N - 1) / tb_size + 1, tb_size>>>(d_drelu, 
    d_diff, /*d_curMax,*/ d_a, d_b, d_c, d_d1, d_d2, party, k.N);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // gpuFree(d_a);
    gpuFree(d_b);
    gpuFree(d_c);
    gpuFree(d_d1);
    gpuFree(d_d2);
    
    return d_a;
}

// no memory leak
extern "C" GPUGroupElement* gpuSelectForMaxpoolBackprop(MaxpoolParams p, GPUSelectKey k,
                                                uint32_t *d_oneHot, 
                                                GPUGroupElement *d_incomingGrad, 
                                                int party, Stats* stats)
{
    size_t size_in_bytes = k.N * sizeof(GPUGroupElement);

    // int outgoingGradSize = p.N * p.imgH * p.imgW * p.C;
    // size_t outgoingGradMemSize = outgoingGradSize * sizeof(GPUGroupElement);
    GPUGroupElement *d_out = (GPUGroupElement*) gpuMalloc(size_in_bytes);
    // checkCudaErrors(cudaMemset(d_outgoingGrad, 0, outgoingGradMemSize));
    GPUGroupElement *d_a, *d_b, *d_c, *d_d1, *d_d2;
    d_a = (GPUGroupElement*) moveToGPU((uint8_t*) k.a, 5 * size_in_bytes, stats);
    d_b = d_a + k.N;
    d_c = d_b + k.N;
    d_d1 = d_c + k.N;
    d_d2 = d_d1 + k.N;
    // GPUGroupElement *d_b = (GPUGroupElement*) moveToGPU((uint8_t*) k.b, size_in_bytes, stats);
    // GPUGroupElement *d_c = (GPUGroupElement*) moveToGPU((uint8_t*) k.c, size_in_bytes, stats);
    // GPUGroupElement *d_d1 = (GPUGroupElement*) moveToGPU((uint8_t*) k.d1, size_in_bytes, stats);
    // GPUGroupElement *d_d2 = (GPUGroupElement*) moveToGPU((uint8_t*) k.d2, size_in_bytes, stats);

    const int tb_size = 256;

    selectForMaxpoolBackpropKernel<<<(k.N - 1) / tb_size + 1, tb_size>>>(p, d_oneHot, 
    d_incomingGrad, d_out, d_a, d_b, d_c, d_d1, d_d2, party, k.N);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    gpuFree(d_a);
    // gpuFree(d_b);
    // gpuFree(d_c);
    // gpuFree(d_d1);
    // gpuFree(d_d2);
    
    return d_out;//d_outgoingGrad;
}

// no memory leak
extern "C" void gpuAndForMaxpool(MaxpoolParams p, int pos, GPUAndKey k,
                                            uint32_t *d_drelu,/* uint32_t *d_drelu2,*/ 
                                            uint32_t *d_oneHot,
                                            int party, Stats* stats)
{
    // printf("selectKey.N: %d\n", k.N);
    int num_ints = (k.N - 1) / PACKING_SIZE + 1;
    size_t size_in_bytes = num_ints * sizeof(uint32_t);

    uint32_t *d_b0, *d_b1, *d_b2;
    d_b0 = (uint32_t*) moveToGPU((uint8_t*) k.b0, 3 * size_in_bytes, stats);
    d_b1 = d_b0 + num_ints;
    d_b2 = d_b1 + num_ints;
    // uint32_t *d_b1 = (uint32_t*) moveToGPU((uint8_t*) k.b1, size_in_bytes, stats);
    // uint32_t *d_b2 = (uint32_t*) moveToGPU((uint8_t*) k.b2, size_in_bytes, stats);

    const int tb_size = 256;
    // printf("N for maxpool and: %d %d %d\n", k.N, num_ints, size_in_bytes);
    int numElems = p.N * p.H * p.W * p.C * p.FH * p.FW;
    andForMaxpoolKernel<<<(numElems - 1) / tb_size + 1, tb_size>>>(p, pos, d_drelu, d_oneHot,
    d_b0, d_b1, d_b2, party, numElems);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    gpuFree(d_b0);
    // gpuFree(d_b1);
    // gpuFree(d_b2);
    // gpuFree(d_d1);
    // gpuFree(d_d2);
    
    // return d_oneHot;
}

extern "C" GPUGroupElement* gpuCollectGradients(MaxpoolParams p, GPUGroupElement* d_outgoingGradExpanded, Stats* s) {
    size_t outgoingGradSize = p.N * p.imgH * p.imgW * p.C;
    size_t outgoingGradMemSize = outgoingGradSize * sizeof(GPUGroupElement);
    GPUGroupElement* d_outgoingGrad = (GPUGroupElement*) gpuMalloc(outgoingGradMemSize);
    // checkCudaErrors(cudaMemset(d_outgoingGrad, 0, outgoingGradMemSize));
    const int tbSize = 256;
    assert(p.zPadHLeft == 0 && p.zPadHRight == 0 && p.zPadWLeft == 0 && p.zPadWRight == 0);
    gpuCollectGradientsKernel<<<(outgoingGradSize - 1) / tbSize + 1, tbSize>>>(p, d_outgoingGradExpanded, d_outgoingGrad, outgoingGradSize);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return d_outgoingGrad;
}

extern "C" void gpuReluSignExtendMux(int party, int bin, int N, 
GPUGroupElement* d_I, GPUGroupElement* h_oneHot, GPUGroupElement* h_outMask, GPUGroupElement* d_drelu, 
GPUGroupElement* d_xLTRin, Stats* s) {
    auto d_oneHot = (GPUGroupElement*) moveToGPU((uint8_t*) h_oneHot, 4 * N * sizeof(GPUGroupElement), s);
    auto d_outMask = (GPUGroupElement*) moveToGPU((uint8_t*) h_outMask, 2 * N * sizeof(GPUGroupElement), s);
    // printf("here\n");
    reluSignExtendMuxKernel<<<(N - 1) / 128 + 1, 128>>>(party, bin, N, d_I, d_oneHot, d_outMask, d_drelu, d_xLTRin);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_oneHot);
    gpuFree(d_outMask);
}



// extern "C" GPUGroupElement *finishRTWrapper(GPUReLUTruncateKey k,
//                                             uint32_t *h_x1, GPUGroupElement *h_y1,
//                                             uint32_t *h_x2, GPUGroupElement *h_y2, int party)
// {
//     GPUGroupElement *d_y1, *d_y2, *d_a;
//     uint32_t *d_x1, *d_x2;

//     unsigned long drelu_size_in_bytes = (k.num_rts - 1) / 8 + 1;
//     unsigned long size_in_bytes = k.num_rts * sizeof(GPUGroupElement);

//     cudaMalloc(&d_x1, drelu_size_in_bytes);
//     cudaMalloc(&d_x2, drelu_size_in_bytes);
//     cudaMalloc(&d_y1, size_in_bytes);
//     cudaMalloc(&d_y2, size_in_bytes);
//     cudaMalloc(&d_a, size_in_bytes);

//     cudaMemcpy(d_x1, h_x1, drelu_size_in_bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_x2, h_x2, drelu_size_in_bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y1, h_y1, size_in_bytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_y2, h_y2, size_in_bytes, cudaMemcpyHostToDevice);    
//     cudaMemcpy(d_a, k.a, size_in_bytes, cudaMemcpyHostToDevice);

//     auto d_rt = finish_relu_truncate(k, d_x1, d_y1, d_a, h_x2, h_y2, party, NULL);

//     GPUGroupElement* h_rt;
//     cudaMallocHost(&h_rt, size_in_bytes);
//     cudaMemcpy(h_rt, d_rt, size_in_bytes, cudaMemcpyDeviceToHost);

//     cudaFree(d_x1);
//     cudaFree(d_x2);
//     cudaFree(d_y1);
//     cudaFree(d_y2);
//     cudaFree(d_a);
//     cudaFree(d_rt);

//     return h_rt;
// }
__global__ void  plaintext_relu_truncate(GPUGroupElement* d_A, int N, int bw, int shift) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < N) {
        d_A[thread_id] = (d_A[thread_id] < (1ULL << (bw - 1))) * (d_A[thread_id] >> shift);
    }
}

extern "C" GPUGroupElement *plaintextRTWRapper(GPUGroupElement* h_A, int N, int bw, int shift) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, size_in_bytes, NULL);
    plaintext_relu_truncate<<<(N - 1) / 256 + 1, 256>>>(d_A, N, bw, shift);
    auto h_out = (GPUGroupElement*) moveToCPU((uint8_t*) d_A, size_in_bytes, NULL);
    gpuFree(d_A);
    return h_out;
}



// extern "C" GPUGroupElementPair evalDReluWrapper(GPUDCFKey k, int party, GPUGroupElement *dcf_input, int shift, AESGlobalContext* gaes)
// {
//     GPUGroupElement* d_in;
//     size_t in_size_in_bytes = k.num_dcfs * sizeof(GPUGroupElement);
//     checkCudaErrors(cudaMalloc(&d_in, in_size_in_bytes));
//     checkCudaErrors(cudaMemcpy(d_in, dcf_input, in_size_in_bytes, cudaMemcpyHostToDevice));

//     auto device_ctx = evalDRelu(k, party, d_in, shift, gaes, NULL);
//     GPUGroupElement *d_v1 = device_ctx.first;
//     GPUGroupElement *d_v2 = device_ctx.second;
//     // GPUGroupElement *d_in = device_ctx.second;

//     cudaFree(d_in);

//     GPUGroupElement *h_v1 = new GPUGroupElement[k.num_dcfs];
//     GPUGroupElement *h_v2 = new GPUGroupElement[k.num_dcfs];

//     // unsigned long long size_in_bytes = k.num_dcfs * sizeof(GPUGroupElement);

//     cudaMemcpy(h_v1, d_v1, in_size_in_bytes, cudaMemcpyDeviceToHost);
//     cudaMemcpy(h_v2, d_v2, in_size_in_bytes, cudaMemcpyDeviceToHost);

//     cudaFree(d_v1);
//     cudaFree(d_v2);

//     // printf("%llu %llu\n", h_v1[0], h_v2[0]);
//     return std::make_pair(h_v1, h_v2);
// }


// extern "C" GPUGroupElement *evalLrsWrapper(GPUDCFKey k, int party, GPUGroupElement *dcf_input, GPUGroupElement* d_lrs_tn, GPUGroupElement* h_lrsMask, int shift, AESGlobalContext* gaes)
// {
//     GPUGroupElement *d_in, *d_v, *h_v;
//     unsigned long mem_size_in_bytes = k.num_dcfs * sizeof(GPUGroupElement);
//     cudaMalloc(&d_in, mem_size_in_bytes);
//     cudaMemcpy(d_in, dcf_input, mem_size_in_bytes, cudaMemcpyHostToDevice);
//     d_v = evalLrs(k, party, d_in, d_lrs_tn, h_lrsMask, shift, gaes, NULL);
//     h_v = new GPUGroupElement[k.num_dcfs];
//     cudaMemcpy(h_v, d_v, mem_size_in_bytes, cudaMemcpyDeviceToHost);
//     return h_v;
// }

// __global__ void computeReLUTruncateResult(int n, int shift, GPUGroupElement *d_lrs_tn,
//                                           /*GPUGroupElement *d_lrs_ts,
//                                           GPUGroupElement *d_zTruncate,*/
//                                           GPUGroupElement *d_drelu_t2,
//                                           GPUGroupElement *d_a,
//                                           uint32_t *d_drelu, int num_rts)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < num_rts);
//     // printf("mask: %u\n", mask);
//     if (i < num_rts)
//     {
//         // d_zTruncate[i] += ((1ULL << (n - shift)) * d_lrs_tn[i] + d_lrs_ts[i]);
//         GPUGroupElement val = (d_drelu_t2[i] + d_a[i] - d_lrs_tn[i]) & static_cast<GPUGroupElement>(1);
//         int val_as_int = static_cast<int>(val);
//         int laneId = threadIdx.x & 0x1f;
//         val_as_int <<= laneId;
//         // printf("gpu %d %d\n", laneId, val_as_int);
//         for (int j = 16; j >= 1; j /= 2)
//             val_as_int += __shfl_down_sync(mask, val_as_int, j, 32);
//         if (laneId == 0)
//         {
//             d_drelu[i / 32] = val_as_int;
//             // printf("gpu val being stored: %d\n", val_as_int);
//         }
//     }
//     // d_drelu_t2[i] += (d_a[i] - d_lrs_tn[i]);
//     // neha: need to uncomment this
//     // d_drelu_t2[i] &= static_cast<GPUGroupElement>(1);
//     // need to write a kernel to pack these bits efficiently
// }

// extern "C" void gpuTwoRoundRelu(GPULocalTruncateReluKey k,
//                                   int party, GPUGroupElement *d_in,
//                                   GPURTContext *rtc, AESGlobalContext* gaes, Stats* stats)
// {
//     auto d_mem = evalDRelu(k.rinDcfKey, party, d_in, k.shift, gaes, stats);
//     GPUGroupElement *d_lrs_tn = d_mem.first;
//     GPUGroupElement *d_drelu_t2 = d_mem.second;

//     auto d_lrs_ts = evalLrs(k.dcfKeyS, party, d_in, k.shift, gaes, stats);
//     checkCudaErrors(cudaFree(d_in));

//     const int tb_size = 256; 
//     size_t size_in_bytes = k.num_rts * sizeof(GPUGroupElement);

//     GPUGroupElement *d_zTruncate = (GPUGroupElement*) moveToGPU((uint8_t*) k.zTruncate, size_in_bytes, stats);
//     GPUGroupElement *d_a = (GPUGroupElement*) moveToGPU((uint8_t*) k.a, size_in_bytes, stats);

//     uint32_t *d_drelu;
//     // cudaMalloc(&d_zTruncate, size_in_bytes);
//     // cudaMemcpy(d_zTruncate, k.zTruncate, size_in_bytes, cudaMemcpyHostToDevice);
//     // cudaMalloc(&d_a, size_in_bytes);
//     size_t drelu_mem_size = (k.num_rts - 1) / 8 + 1;
//     checkCudaErrors(cudaMalloc(&d_drelu, drelu_mem_size)); // one bit per rt
//     // checkCudaErrors(cudaMemcpy(d_a, k.a, size_in_bytes, cudaMemcpyHostToDevice));
//     computeReLUTruncateResult<<<(k.num_rts - 1) / tb_size + 1, tb_size>>>(k.Bin,
//                                                                           k.shift, d_lrs_tn, d_lrs_ts, d_zTruncate, d_drelu_t2, d_a, d_drelu, k.num_rts);
//     // checkCudaErrors(cudaDeviceSynchronize());
//     // GPUGroupElement *h_lrs;
//     // uint32_t *h_drelu;
//     // checkCudaErrors(cudaMallocHost(&h_lrs, size_in_bytes));
//     // checkCudaErrors(cudaMallocHost(&h_drelu, drelu_mem_size));
//     // checkCudaErrors(cudaMemcpy(h_lrs, d_zTruncate, size_in_bytes, cudaMemcpyDeviceToHost));
//     // can prolly get rid of this
//     // checkCudaErrors(cudaMemcpy(h_drelu, d_drelu, drelu_mem_size, cudaMemcpyDeviceToHost));

//     rtc->d_lrs0 = d_zTruncate;
//     rtc->d_drelu0 = d_drelu;
//     rtc->d_a = d_a;
//     // context->h_lrs0 = h_lrs;
//     // context->h_drelu0 = h_drelu;
// }
