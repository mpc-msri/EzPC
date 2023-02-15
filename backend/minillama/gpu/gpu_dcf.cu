// #include "group_element.h"
#include <cryptoTools/Common/Defines.h>
// #include "gpu_aes.cu"
#include "gpu_data_types.h"

#include "gpu_aes_shm.cu"
#include "gpu_mem.h"
#include "gpu_truncate.h"
#include <assert.h>
#include <cstdint>
// #include <cuda_device_runtime_api.h>
#include <iostream>
#include <fstream>
#include <string>

#include "helper_cuda.h"
// #include "gpu_utils.cpp"

using namespace std;

typedef uint64_t GPUGroupElement;
typedef unsigned __int128 AESBlock;

#define SERVER1 1
#define AES_BLOCK_LEN_IN_BITS 128
#define FULL_MASK 0xffffffff


// #define tb_size 128;
// #define bit_length 64


__device__ __constant__ GPUGroupElement output_bit_mask;

TORCH_CSPRNG_HOST_DEVICE inline AESBlock lsb(AESBlock b)
{
    return b & 1;
}

TORCH_CSPRNG_HOST_DEVICE GPUGroupElement getGroupElementFromAESBlock(AESBlock b, int Bout, int vector_elem_pos)
{
    /* returning the last 64 bits */
    // int output_bits = Bout * vector_elem_pos;
    // int aes_block_num = output_bits / AES_BLOCK_LEN_IN_BITS;
    // b = b ^ aes_block_num;
    // int offset_in_aes_block = output_bits % AES_BLOCK_LEN_IN_BITS;
    // b = b >> offset_in_aes_block;
    GPUGroupElement g = static_cast<GPUGroupElement>(b);
    /* this will not cause any divergence because the entire protocol
    will be run with a single output bit length */
    if (Bout < 64)
        g &= output_bit_mask;
    return g;
}

// __host__ __device__ void printAESBlock2(uint8_t *b)
// {
//     for (int i = 0; i < 16; i++)
//         printf("%02X", b[i]);
//     printf("\n");
// }

__device__ AESBlock traverseOneDCF(int Bin, int Bout,
                                                 int output_vec_len, int party,
                                                 const AESBlock &s,
                                                 const AESBlock &cw,
                                                 const osuCrypto::u8 keep,
                                                 GPUGroupElement *v_share,
                                                 GPUGroupElement vcw,
                                                 uint64_t level,
                                                 bool geq,
                                                 int evalGroupIdxStart,
                                                 int evalGroupIdxLen, AESSharedContext* c)

{
    // /* these need to be written to constant memory */
    const AESBlock notThreeAESBlock = ~3; //toAESBlock(~0, ~3);
    // const AESBlock TwoAESBlock = 2;       //toAESBlock(0, 2);
    // const AESBlock ThreeAESBlock = 3;     //toAESBlock(0, 3);
    const AESBlock zeroAndAllOne[2] = {0, static_cast<AESBlock>(~0)};
    const AESBlock OneAESBlock = 1; //toAESBlock(0, 2);

    AESBlock tau = 0, cur_v = 0, stcw; //, to_encrypt_0, to_encrypt_1;
    // to_encrypt_0 = keep;
    // to_encrypt_1 = keep + 2;
    /* the t which determines if we should add in the control word
    for this level */
    osuCrypto::u8 t_previous = lsb(s);
    /* remove the last two bits from the AES seed */
    auto ss = s & notThreeAESBlock;
    // tau = keep; //ss;
    /* get the seed for this level (tau) */
    // apply_aes_prg((uint8_t *)&ss, keep, (uint8_t *)&tau, (uint8_t *)&cur_v);
    applyAESPRG(c, (uint32_t *)&ss, keep, (uint32_t *)&tau, (uint32_t *)&cur_v);
    // printAESBlock2((uint8_t *)&ss);
    // printAESBlock2((uint8_t *)&tau);
    // printAESBlock2((uint8_t *)&cur_v);
    // printf("%d\n", keep);

    /* why does this xor always exist? */
    /* notice that the last two bits of tau are 
    not necessarily zero and we retain those bits on XOR-ing with ss */
    // tau = tau ^ ss;

    // AESBlock v_this_level = ss;
    // gpu_aes_encrypt((uint8_t *)&v_this_level, keep + 2);
    // v_this_level ^= ss;

    /* zero out the last two bits of the correction word for s because
    they must contain the corrections for t0 and t1 */
    const auto scw = (cw & notThreeAESBlock);
    /* separate the correction bits for t0 and t1 and place them 
    in the lsbs of two AES blocks */
    AESBlock ds[] = {((cw >> 1) & OneAESBlock), (cw & OneAESBlock)};
    const auto mask = zeroAndAllOne[t_previous];

    /* QUESTION: why are we xoring (0 || the CW for the `keep' bit) with tau,
    whose last two bits have the form (b0 || b1)? */
    /* tau has the seed and t for the next level in its lsb. This means
    that we don't need to `select' the correct bit from tau in the same way that 
    we need to select it from the correction word. Tau has *two* spare bits because
    the seed only needs 126 bits, so we can assume wlog that the last bit is t */
    /* correct the seed for the next level if necessary */
    stcw = tau ^ ((scw ^ ds[keep]) & mask);

    uint64_t sign = (party == SERVER1) ? -1 : 1;
    GPUGroupElement v = getGroupElementFromAESBlock(cur_v, Bout, 0);
    /* OPT: there is nothing stopping us from replacing this
    MUL with an &. In fact, we should probably replace all MULs (since they are
    only with 0/1 or 1/-1 with their + counterparts) */
    // *v_share += (sign * (v + t_previous * vcw));
    *v_share += (sign * (v + (static_cast<GPUGroupElement>(mask) & vcw)));
    // printAESBlock2((uint8_t *)&cur_v);
    // printf("gpu: %llu %llu %d\n", v, *v_share, t_previous);
    return stcw;
}


__device__ GPUGroupElement getVCW(int Bout, GPUGroupElement* vcw, int num_dcfs, int i) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // int laneId = threadIdx.x & 0x1f;
    if(Bout == 1 || Bout == 2) {
        // uint32_t mask = Bout == 1 ? 1 : 3;
        // int blocks_per_level = ((num_dcfs - 1) / PACKING_SIZE + 1);
        // return static_cast<GPUGroupElement>((((uint32_t*) vcw)[blocks_per_level * i + (thread_id / PACKING_SIZE)] >> laneId) & 1);
    // } else if(Bout == 2) {
        int intsPerLevel = (Bout * num_dcfs - 1) / PACKING_SIZE + 1;
        int payloadPerIntMask = (PACKING_SIZE / Bout) - 1;
        // printf("mask: %d\n", payloadPerIntMask);
        // printf("reading vcw %d %d %d %u\n", i, intsPerLevel, intsPerLevel * i + ((Bout * threadId) / PACKING_SIZE), ((uint32_t*) vcw)[intsPerLevel * i + ((Bout * threadId) / PACKING_SIZE)]);
        // it makes sense to read fewer bits even when bout is 2
        return static_cast<GPUGroupElement>((((uint32_t*) vcw)[intsPerLevel * i + ((Bout * threadId) / PACKING_SIZE)] >> (Bout * (threadIdx.x & payloadPerIntMask))) & output_bit_mask);
    }
    else {
        return vcw[i * num_dcfs + threadId];
    }
}

__device__ /*AESBlock*/ void traversePathDCF(int Bin, int Bout, int output_vec_len, int party,
                                    GPUGroupElement dcf_input,
                                    AESBlock *scw,
                                    GPUGroupElement *v_share,
                                    GPUGroupElement *vcw,
                                    bool geq,
                                    int evalGroupIdxStart,
                                    int evalGroupIdxLen, AESBlock* s, int num_dcfs, AESSharedContext* c)
{
    // AESBlock cur_scw = scw[0];
    // AESBlock 
    *s = scw[0];
    dcf_input = __brevll(dcf_input) >> (64 - Bin);
    for (int i = 0; i < Bin; ++i)
    {
        const osuCrypto::u8 keep = lsb(dcf_input);
        auto vcwI = getVCW(Bout, vcw, num_dcfs, i);
        *s = traverseOneDCF(Bin, Bout, output_vec_len, party, *s,
                           scw[(i + 1) * num_dcfs], keep, v_share, /*vcw[i * num_dcfs]*/ vcwI, i, geq, evalGroupIdxStart, evalGroupIdxLen, c);
        dcf_input = dcf_input >> 1;
    }
    return;
}

// think about when to pass pointers to large amounts of data like AESBlocks
__device__ void getDCFOutput(int Bout, AESBlock s, GPUGroupElement vcw, int party, GPUGroupElement *v)
{
    auto t = lsb(s);
    const AESBlock notThreeAESBlock = ~3;
    GPUGroupElement g = getGroupElementFromAESBlock(s & notThreeAESBlock, Bout, 0);

    /* this is okay to do because vcw now points to the local set of correction
    words */
    g += (t * vcw);
    /* this if condition will never diverge because each party gets its own GPU */
    if (party == SERVER1)
    {
        g = -g;
    }
    *v += g;
    if (Bout < 64)
        *v &= output_bit_mask;
}

__device__ void writeDReluOutput(GPUGroupElement* dReluOutput, GPUGroupElement dReluBit, int bout, int N) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(FULL_MASK, /*threadIdx.x*/ threadId < N);
    int laneId = threadIdx.x & 0x1f;
    if(bout == 1) {
        int dreluAsInt = static_cast<int>(dReluBit);
        dreluAsInt <<= laneId;
        for (int j = 16; j >= 1; j /= 2)
            dreluAsInt += __shfl_down_sync(mask, dreluAsInt, j, 32);
        if (laneId == 0) {
            ((uint32_t*) dReluOutput)[threadId / 32] = static_cast<uint32_t>(dreluAsInt);
        }
    } else if(bout == 2) {
        dReluBit <<= (2 * laneId);
        for(int j = 16; j >= 1; j /= 2)
            dReluBit += __shfl_down_sync(mask, dReluBit, j, 32);
        if(laneId == 0) {
            dReluOutput[threadId / 32] = dReluBit;
        }
    } else if(bout == 64) {
        dReluOutput[threadId] = dReluBit;
    }
}

/* out needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
__global__ void evalDCF(int Bin, int Bout, int output_vec_len,
                        GPUGroupElement *v_share,              // groupSize
                        int party, GPUGroupElement *dcf_input, // might want to pass a pointer to this later
                        AESBlock *scw,                         // k.Bin + 1
                        GPUGroupElement *vcw,                  // k.Bin * groupSize
                        bool geq /*= false*/, int evalGroupIdxStart /*= 0*/,
                        int evalGroupIdxLen /*= -1*/, int num_dcfs, AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < num_dcfs)
    {
        scw = &scw[thread_id/* * (Bin + 1)*/];
        // vcw = &vcw[thread_id/* * (Bin + 1)*/];

        /* v_share needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
        /* need to make v_share a local variable and avoid writing to global memory */
        /* might change this later if it adversely impacts performance */
        // v_share = &v_share[thread_id];
        GPUGroupElement local_v_share = 0;
        GPUGroupElement x = dcf_input[thread_id] & ((1ULL << Bin) - 1);
        // printf("%llu\n", local_dcf_input);
        AESBlock s;
        /*auto s =*/ traversePathDCF(Bin, Bout,
                                 output_vec_len, party, x, scw, &local_v_share, vcw, geq,
                                 evalGroupIdxStart, evalGroupIdxLen, &s, num_dcfs, &saes);
        auto vcwBin = getVCW(Bout, vcw, num_dcfs, Bin);
        getDCFOutput(Bout, s, vcwBin, party, &local_v_share);
        v_share[thread_id] = local_v_share;
    }
}
// fix the ballot sync bug
__global__ void maskedDCFKernel(int Bin, int Bout, int output_vec_len,
                        GPUGroupElement *v_share,              // groupSize
                        int party, GPUGroupElement *dcf_input, // might want to pass a pointer to this later
                        AESBlock *scw,                         // k.Bin + 1
                        GPUGroupElement *vcw,                  // k.Bin * groupSize
                        GPUGroupElement *dcfMask,
                        bool geq /*= false*/, int evalGroupIdxStart /*= 0*/,
                        int evalGroupIdxLen /*= -1*/, int num_dcfs, AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // GPUGroupElement dcfBit = 0;
    if (thread_id < num_dcfs)
    {
        scw = &scw[thread_id/* * (Bin + 1)*/];
        // vcw = &vcw[thread_id/* * (Bin + 1)*/];
        /* v_share needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
        /* need to make v_share a local variable and avoid writing to global memory */
        /* might change this later if it adversely impacts performance */
        // v_share = &v_share[thread_id];
        GPUGroupElement local_v_share = 0;
        GPUGroupElement x = dcf_input[thread_id] & ((1ULL << Bin) - 1);
        
        // printf("%llu\n", bias);
        AESBlock s;
        /*auto s =*/ traversePathDCF(Bin, Bout,
                                 output_vec_len, party, x, scw, &local_v_share, vcw, geq,
                                 evalGroupIdxStart, evalGroupIdxLen, &s, num_dcfs, &saes);
        auto vcwBin = getVCW(Bout, vcw, num_dcfs, Bin);
        getDCFOutput(Bout, s, vcwBin, party, &local_v_share);
        auto mask = getVCW(Bout, dcfMask, num_dcfs, 0);
        auto dcfBit = (local_v_share + mask) & ((1ULL << Bout) - 1);
        // will keep this here for now because ballot sync returns false for exited threads by default
        writeDReluOutput(v_share, dcfBit, Bout, num_dcfs);
    }
    // writeDReluOutput(v_share, dcfBit, Bout, num_dcfs);
}

__global__ void selectForTruncateKernel(GPUGroupElement* x, GPUGroupElement* maskedDcfBit, GPUGroupElement* outMask, GPUGroupElement* p, int N, int party) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        uint32_t dcfBit = (((uint32_t*) maskedDcfBit)[i / 32] >> (threadIdx.x & 0x1f)) & 1;
        int j = (dcfBit /*+ 1*/) & 1;
        x[i] = (party == SERVER1) * x[i] + outMask[i] + p[2*i + j];
    }
}

__global__ void localLRSKernel(int shift, GPUGroupElement* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        x[i] >>= shift;
    }
    
}

__global__ void addConstantKernel(int bw, GPUGroupElement* x, GPUGroupElement c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        x[i] = (x[i] + c) & ((1ULL << bw) - 1);
    }
}


double to_gb(uint64_t b)
{
    return b / (double)(1 << 30);
}

// #define DEBUG 0

extern "C" GPUGroupElement *gpu_dcf(GPUDCFKey k, int party, GPUGroupElement *dcf_input, AESGlobalContext* g)
{

    const int tb_size = 256;//k.num_dcfs >= 128 ? 128 : k.num_dcfs;
    int num_thread_blocks = ((k.num_dcfs - 1) / tb_size) + 1;

    // // initialize_gpu_aes();
    AESBlock *d_scw;
    GPUGroupElement *h_out, *d_vcw, *d_out, *d_in;

    GPUGroupElement h_output_bit_mask = static_cast<GPUGroupElement>(-1) >> (64 - k.Bout);
    checkCudaErrors(cudaMemcpyToSymbol(output_bit_mask, &h_output_bit_mask, sizeof(GPUGroupElement)));

    // // might be allocating more memory here than necessary
    uint64_t out_size_in_bytes = k.num_dcfs * sizeof(GPUGroupElement);
    uint64_t in_size_in_bytes = k.num_dcfs * sizeof(GPUGroupElement);

    cudaMallocHost(&h_out, out_size_in_bytes);
    // h_out = new GPUGroupElement[k.num_dcfs];

    checkCudaErrors(cudaMalloc(&d_scw, k.mem_size_scw));
    checkCudaErrors(cudaMalloc(&d_vcw, k.mem_size_vcw));
    checkCudaErrors(cudaMalloc(&d_out, out_size_in_bytes));
    checkCudaErrors(cudaMalloc(&d_in, in_size_in_bytes));

    checkCudaErrors(cudaMemcpy(d_scw, k.scw, k.mem_size_scw, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vcw, k.vcw, k.mem_size_vcw, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in, dcf_input, in_size_in_bytes, cudaMemcpyHostToDevice));

    evalDCF<<<num_thread_blocks, tb_size>>>(k.Bin, k.Bout, k.out_vec_len, d_out, party, d_in, d_scw, d_vcw, false, 0, -1, k.num_dcfs, *g);

    checkCudaErrors(cudaMemcpy(h_out, d_out, out_size_in_bytes, cudaMemcpyDeviceToHost));

    cudaFree(&d_scw);
    cudaFree(&d_vcw);
    cudaFree(&d_out);
    cudaFree(&d_in);

    return h_out;
}

// no memory leak
extern "C" GPUGroupElement *gpuMaskedDcf(GPUMaskedDCFKey k, int party, GPUGroupElement *d_in, AESGlobalContext* g, Stats* s)
{
    // do not change tb size it is needed to load the sbox
    const int tb_size = 256;//k.num_dcfs >= 128 ? 128 : k.num_dcfs;
    GPUDCFKey dcfKey = k.dcfKey;
    int N = dcfKey.num_dcfs;
    int num_thread_blocks = (N - 1) / tb_size + 1;
    AESBlock *d_scw;
    GPUGroupElement *d_out, *d_vcw, *d_mask;
    GPUGroupElement h_output_bit_mask = static_cast<GPUGroupElement>(-1) >> (64 - dcfKey.Bout);
    checkCudaErrors(cudaMemcpyToSymbol(output_bit_mask, &h_output_bit_mask, sizeof(GPUGroupElement)));
    assert(dcfKey.mem_size_vcw % (dcfKey.Bin + 1) == 0);
    size_t memSizeOut = dcfKey.mem_size_vcw / (dcfKey.Bin + 1);

    d_out = (GPUGroupElement*) gpuMalloc(memSizeOut);
    d_scw = (AESBlock*) moveToGPU((uint8_t*) dcfKey.scw, dcfKey.mem_size_scw, s);
    d_vcw = (GPUGroupElement*) moveToGPU((uint8_t*) dcfKey.vcw, dcfKey.mem_size_vcw, s);
    d_mask = (GPUGroupElement*) moveToGPU((uint8_t*) k.dReluMask, memSizeOut, s);

    // printf("%lu %lu %lu %d %d %d %d %d\n", dcfKey.mem_size_scw, dcfKey.mem_size_vcw, memSizeOut, dcfKey.Bin, dcfKey.Bout, dcfKey.num_dcfs, num_thread_blocks, tb_size);
    maskedDCFKernel<<<num_thread_blocks, tb_size>>>(dcfKey.Bin, dcfKey.Bout, dcfKey.out_vec_len, d_out, party, d_in, d_scw, d_vcw, d_mask, false, 0, -1, dcfKey.num_dcfs, *g);

    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_scw);
    gpuFree(d_vcw);
    gpuFree(d_mask);
    return d_out;
}

// no memory leak
extern "C" void gpuSelectForTruncate(int party, int N, GPUGroupElement* d_I, GPUGroupElement* d_maskedDcfBit, GPUGroupElement* h_outMask, GPUGroupElement* h_p, Stats* s) {
    size_t memSize = N * sizeof(GPUGroupElement);
    auto d_outMask = (GPUGroupElement*) moveToGPU((uint8_t*) h_outMask, memSize, s);
    auto d_p = (GPUGroupElement*) moveToGPU((uint8_t*) h_p, 2*memSize, s);
    selectForTruncateKernel<<<(N - 1) / 128 + 1, 128>>>(d_I, d_maskedDcfBit, d_outMask, d_p, N, party);
    checkCudaErrors(cudaDeviceSynchronize());
    gpuFree(d_outMask);
    gpuFree(d_p);
}

extern "C" void gpuLocalLRS(int N, int shift, GPUGroupElement* d_I) {
    localLRSKernel<<<(N - 1) / 128 + 1, 128>>>(shift, d_I, N);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

extern "C" void gpuLocalLRSWrapper(int bin, int shift, int N, GPUGroupElement* h_I, GPUGroupElement* h_O) {
    assert(bin >= shift);
    size_t memSizeI = N * sizeof(GPUGroupElement);
    auto d_I = (GPUGroupElement*) moveToGPU((uint8_t*) h_I, memSizeI, NULL);
    localLRSKernel<<<(N - 1) / 128 + 1, 128>>>(shift, d_I, N);
    checkCudaErrors(cudaDeviceSynchronize());
    moveIntoCPUMem((uint8_t*) h_O, (uint8_t*) d_I, memSizeI, NULL);
    gpuFree(d_I);
}


extern "C" void gpuAddConstant(int bw, GPUGroupElement* d_I, GPUGroupElement c, int N) {
    addConstantKernel<<<(N - 1) / 128 + 1, 128>>>(bw, d_I, c, N);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}


/* need to coalesce global memory accesses */

// int num_threads = 1;

// AESBlock tau[num_threads], cur_v[num_threads];
// AESBlock *d_scw, *d_tau, *d_cur_v;
// cudaMalloc(&d_scw, sizeof(AESBlock) * num_threads);
// cudaMalloc(&d_tau, sizeof(AESBlock) * num_threads);
// cudaMalloc(&d_cur_v, sizeof(AESBlock) * num_threads);

// AESBlock h_scw[num_threads];
// for (int i = 0; i < num_threads; i++)
//     h_scw[i] = k.scw[0];

// cudaMemcpy(d_scw, h_scw, sizeof(AESBlock) * num_threads, cudaMemcpyHostToDevice);
// printf("calling a function on the gpu\n");
// apply_aes_prg_wrapper<<<1, 1>>>((uint8_t *)d_scw, 0, (uint8_t *)d_tau, (uint8_t *)d_cur_v);

// cudaMemcpy(&tau, d_tau, sizeof(AESBlock) * num_threads, cudaMemcpyDeviceToHost);
// cudaMemcpy(&cur_v, d_cur_v, sizeof(AESBlock) * num_threads, cudaMemcpyDeviceToHost);

// printf("from the gpu\n");
// printAESBlock((uint8_t *)&tau[1]);
// printAESBlock((uint8_t *)&cur_v[1]);
// printf("boo\n");
// return 0;
