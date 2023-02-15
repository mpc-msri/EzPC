#include "gpu_fss_utils.h"
#include <cryptoTools/Common/Defines.h>
#include <../group_element.h>
#include <cryptoTools/Crypto/PRNG.h>
#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "helper_cuda.h"
#include <cassert>
#include <cstdint>
#include <omp.h>

extern "C" GPUGroupElement* gpuAddShares(GPUGroupElement* d_A, GPUGroupElement* d_B, int N, Stats*);
extern "C" void gpuAddSharesInPlace(GPUGroupElement* d_A, GPUGroupElement* d_B, int bw, int N);
extern "C" void gpuAddSharesModN(int numBits, uint32_t* A, uint32_t* B, int N);
extern "C" GPUGroupElement* gpuXor(uint32_t* d_A, uint32_t* d_B, int N, Stats*);

osuCrypto::PRNG* prngs;
int numPrngs = 0;

void initCPURandomness(/*int numThreads*/) {
    // int numThreads = omp_get_num_threads();
    int numThreads = 4;
    omp_set_num_threads(numThreads);
    printf("%d\n", numThreads);
    prngs = new osuCrypto::PRNG[numThreads];
    // for(int i = 0; i < numThreads; i++) prngs[i].SetSeed(osuCrypto::toBlock(0, time(NULL)));
    for(int i = 0; i < numThreads; i++) prngs[i].SetSeed(osuCrypto::toBlock(0, i)); // Kanav: insecure! for debugging
    numPrngs = numThreads;
}

GPUGroupElement randomGE(int bw) {
    GPUGroupElement a;
    // printf("%d\n", omp_get_thread_num());
    int tid = omp_get_thread_num();
    assert(tid < numPrngs);
    a = prngs[tid].get<uint64_t>();
    mod(a, bw);
    // printf("random ge: %d, %lu, %d\n", omp_get_thread_num(), a, bw);
    return a;
    // return random_ge(bw);
}

int32_t randomInt() {
    // GPUGroupElement a;
    // printf("%d\n", omp_get_thread_num());
    int tid = omp_get_thread_num();
    assert(tid < numPrngs);
    return prngs[tid].get<int32_t>();
    // mod(a, bw);
    // printf("random ge: %d, %lu, %d\n", omp_get_thread_num(), a, bw);
    // return a;
    // return random_ge(bw);
}

std::array<osuCrypto::block, 2> getRandomAESBlockPair() {
    int tid = omp_get_thread_num();
    assert(tid < numPrngs);
    auto randomBlock = prngs[tid].get<std::array<osuCrypto::block, 2>>();
    // std::cout << "aes block from new prngs: " << tid << " " << randomBlock[0] << std::endl;
    // printf("random block: %lu %lu\n", randomBlock[0], randomBlock[1]);
    return randomBlock;
}

GPUGroupElement* gpuReconstruct(GPUGroupElement* h_A0, int N, Peer* peer, int party, Stats* s) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto h_A1 = (GPUGroupElement*) exchangeShares(peer, (uint8_t*) h_A0, size_in_bytes, party, s);
    
    auto d_A0 = (GPUGroupElement*) moveToGPU((uint8_t*)h_A0, size_in_bytes, s);
    auto d_A1 = (GPUGroupElement*) moveToGPU((uint8_t*)h_A1, size_in_bytes, s);
    
    cpuFree(h_A1);
    auto d_A = gpuAddShares(d_A0, d_A1, N, s);    
    gpuFree(d_A0);
    gpuFree(d_A1);
    return d_A;
}

void gpuReconstructInPlace(GPUGroupElement* d_A0, int numBits, int N, Peer* peer, int party, Stats* s) {
    size_t size_in_bytes, numInts = 0; 
    if(numBits > 32) size_in_bytes = N * sizeof(GPUGroupElement);
    else {
        assert(numBits == 1 || numBits == 2);
        numInts = ((numBits * N - 1) / 32 + 1);
        size_in_bytes = numInts * sizeof(uint32_t);
    }
    auto h_A0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_A0, size_in_bytes, s);
    auto h_A1 = (GPUGroupElement*) exchangeShares(peer, (uint8_t*) h_A0, size_in_bytes, party, s);
    cpuFree(h_A0);
    auto d_A1 = (GPUGroupElement*) moveToGPU((uint8_t*) h_A1, size_in_bytes, s);
    cpuFree(h_A1);
    if(numBits == 1) 
        gpuXor((uint32_t*) d_A0, (uint32_t*) d_A1, numInts, s);
    else if(numBits == 2)
        gpuAddSharesModN(numBits, (uint32_t*) d_A0, (uint32_t*) d_A1, N);
    else if(numBits > 32)
        gpuAddSharesInPlace(d_A0, d_A1, numBits, N);    
    gpuFree(d_A1);
}

void gpuReconstructBits(uint32_t* d_A0, int N, Peer* peer, int party, Stats* s) {
    size_t size_in_bytes = N * sizeof(uint32_t);
    auto h_A0 = (uint32_t*) moveToCPU((uint8_t*) d_A0, size_in_bytes, s); 
    auto h_A1 = (uint32_t*) exchangeShares(peer, (uint8_t*) h_A0, size_in_bytes, party, s);
    // printf("%d %d\n", *h_A0, *h_A1);
    cpuFree(h_A0);
    auto d_A1 = (uint32_t*) moveToGPU((uint8_t*)h_A1, size_in_bytes, s);    
    cpuFree(h_A1);
    gpuXor(d_A0, d_A1, N, s);    
    gpuFree(d_A1);
}
// use this function to generate random input that must be transferred to the gpu
// free this memory using cpuFree()
GPUGroupElement* initRandom(int N, int bw) {
    GPUGroupElement* random = (GPUGroupElement*) cpuMalloc(N * sizeof(GPUGroupElement));//new GPUGroupElement[N];
    // #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        random[i] = randomGE(bw);
    }
    return random;
}

void initRandomInPlace(GPUGroupElement* random, int N, int bw) {
    // #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        random[i] = randomGE(bw);
    }
}

GPUGroupElement* initWithConst(int N, int bw, GPUGroupElement x) {
    GPUGroupElement* A = new GPUGroupElement[N];
    // #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        A[i] = x;
    }
    return A;
}

GPUGroupElement* getMaskedInputOnCpu(int N, int bw, int party, Peer* peer, GPUGroupElement* h_mask_I, GPUGroupElement** h_I) {
    size_t memSizeI = N * sizeof(GPUGroupElement);
    GPUGroupElement* h_masked_I = (GPUGroupElement*) cpuMalloc(memSizeI);
    // else *h_I = initWithConst(N, bw, 0);
    // there can be an overflow when bw < 64 but does it matter?
    // fixed the overflow by having one party generate only zeros
    if(party == 0) {
        *h_I = (GPUGroupElement*) cpuMalloc(memSizeI);
        // #pragma omp parallel for
        for(int i = 0; i < N; i++) {
            (*h_I)[i] = randomGE(bw);
            h_masked_I[i] = (*h_I)[i] + h_mask_I[i];
            mod(h_masked_I[i], bw);
        }
        send_bytes(peer, (uint8_t*) h_masked_I, memSizeI);
    } else {
        assert(party == 1);
        recv_bytes(peer, (uint8_t*) h_masked_I, memSizeI);
    }
    return h_masked_I;
}

GPUGroupElement* getMaskedInputOnGpu(int N, int bw, int party, Peer* peer, GPUGroupElement* h_mask_I, GPUGroupElement** h_I) {
    size_t memSizeI = N * sizeof(GPUGroupElement);
    auto h_masked_I = getMaskedInputOnCpu(N, bw, party, peer, h_mask_I, h_I);
    auto d_masked_I = (GPUGroupElement*) moveToGPU((uint8_t*) h_masked_I, memSizeI, NULL);
    cpuFree(h_masked_I);
    return d_masked_I;
}

// assumes that P0 has all the input
std::pair<GPUGroupElement*, GPUGroupElement*> maskInput(int N, int bw, int party, Peer* peer, GPUGroupElement* h_I, GPUGroupElement* h_mask_I, Stats* s) {
    size_t memSizeI = N * sizeof(GPUGroupElement);
    GPUGroupElement *d_I, *d_mask_I;
    // if(party == 0) 
    d_I = (GPUGroupElement*) moveToGPU((uint8_t*) h_I, memSizeI, s);
    // else {
        // d_I = (GPUGroupElement*) gpuMalloc(N * sizeof(GPUGroupElement));
        // checkCudaErrors(cudaMemset(d_I, 0, N * sizeof(GPUGroupElement)));
    // }
    d_mask_I = (GPUGroupElement*) moveToGPU((uint8_t*) h_mask_I, memSizeI, s);
    gpuAddSharesInPlace(d_I, d_mask_I, bw, N);
    gpuReconstructInPlace(d_I, bw, N, peer, party, s);
    return std::make_pair(d_I, d_mask_I);
}

void initXavier(GPUGroupElement* W, int N, double range) {
    for(int i = 0; i < N; i++) {
        double r = (double) randomInt();
        W[i] = (GPUGroupElement)((r / (1LL << 31)) * range);
    }
}

void initWeightsHelper(GPUGroupElement* W, int N, double range, int party, Peer* peer) {
    if(party == 0) {
        initXavier(W, N, range);
        send_bytes(peer, (uint8_t*) W, N * sizeof(GPUGroupElement));
    } else {
        recv_bytes(peer, (uint8_t*) W, N * sizeof(GPUGroupElement));
    }
}

        // h_I is the dataset that is stored in memory
        // h_mask_I is the mask which is part of the key
        // we cannot free or overwrite either because we need the input as-is in successive epochs
        // and we need the mask as-is to execute the conv protocol
    //     auto h_masked_I0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_I, memSizeI, s);
    //     h_masked_I1 = exchangeShares(peer, h_masked_I0, memSizeI, party, s);
    //     auto d_masked_I1 = (GPUGroupElement*) moveToGPU(h_masked_I1, memSizeI, &s);
    //     gpuAddSharesInPlace(d_I, d_masked_I1, bw, N);
    //     gpuFree(d_masked_I1);
    // } else {
    //     assert(party == 1);
    //     d_mask_I = d_I;
    //     h_masked_I1 = exchangeShares(peer, h_mask_I, memSizeI, party, s);
    //     auto d_masked_I1 = (GPUGroupElement*) moveToGPU(h_masked_I1, memSizeI, &s);
    //     auto d_masked_I = gpuAddShares(d_I, d_masked_I1, bw, N);
    //     gpuFree(d_I);
    //     d_I = d_masked_I;
    //     gpuFree(d_masked_I1);
    // }
