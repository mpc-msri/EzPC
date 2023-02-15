#pragma once

#include "fss.h"
#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "gpu_stats.h"

uint8_t* exchangeShares(Peer*, uint8_t*, size_t, int, Stats*);
// GPUGroupElement* addShares(GPUGroupElement*, GPUGroupElement*, int N);
GPUGroupElement* gpuReconstruct(GPUGroupElement* h_A0, int N, Peer* peer, int party, Stats*);
// GPUGroupElement* reconstructMasked(GPUGroupElement* h_A0, GPUGroupElement* h_mask_A0, int N, Peer* peer, int party, Stats*);
// GPUGroupElement* GPUConv2D(Peer*, GPUConv2DKey k, GPUGroupElement* d_I, GPUGroupElement* d_F, int party, Stats*);
// GPUGroupElement* GPUMatmul(Peer*, GPUMatmulKey k, GPUGroupElement* d_A, GPUGroupElement* d_B, int party, Stats*);
GPUGroupElement* GPUReluTruncate(Peer*, GPUReLUTruncateKey k, GPUGroupElement* d_I, int party, AESGlobalContext* gaes, Stats*);
GPUGroupElement* GPULocalTruncateRelu(Peer* peer, int bin, GPUReLUTruncateKey k, GPUGroupElement* d_I, int party, AESGlobalContext* g, Stats* s);

GPUConv2DKey readGPUConv2DKey(/*std::istream&*/ uint8_t**);
GPUGroupElement* readGPUConv2DInput(std::istream& f, GPUConv2DKey k);
GPUGroupElement* readGPUConv2DFilter(std::istream& f, GPUConv2DKey k);
GPUReLUTruncateKey readGPUReLUTruncateKey(/*std::istream&*/ uint8_t**);
GPUReLUTruncateKey readGPULocalTruncateReLUKey(/*std::istream&*/ uint8_t**, bool);
GPUMatmulKey readGPUMatmulKey(/*std::istream&*/ uint8_t**);

Peer* connectToPeer(int party, std::string addr);

// uint8_t* moveToGPU(uint8_t* h_a, size_t size_in_bytes);
// uint8_t* moveToCPU(uint8_t* d_a, size_t size_in_bytes);

GPUGroupElement* initRandom(int N, int bitlength);

// void cpuFree(void* h_a);
// void gpuFree(void* d_a);

extern "C" void initAESContext(AESGlobalContext* g);
extern "C" void freeAESGlobalContext(AESGlobalContext* g);


void initGPUFSS(AESGlobalContext* g);

extern "C" GPUGroupElement *plaintextRTWRapper(GPUGroupElement* h_A, int N, int bw, int shift);
extern "C" GPUGroupElement *gpuConv2DWrapper(GPUConv2DKey k, GPUGroupElement* h_I, GPUGroupElement* h_F, GPUGroupElement* h_C, char op);
extern "C" GPUGroupElement *gpuMatmulWrapper(GPUMatmulKey k, GPUGroupElement* h_A, GPUGroupElement* h_B, GPUGroupElement* h_C, bool, bool, bool);
extern "C" void checkOutput(GPUGroupElement* h_O1, GPUGroupElement* h_O2, GPUGroupElement* h_R, int N);
extern "C" GPUGroupElement* gpuAddShares(GPUGroupElement* d_A, GPUGroupElement* d_B, int N, Stats*);
extern "C" GPUGroupElement* gpuAddPool(GPUGroupElement* d_A, GPUConv2DKey k);
extern "C" GPUGroupElement* gpuAddPoolBackProp(GPUGroupElement* d_A, GPUConv2DKey k);
extern "C" GPUGroupElement* addPoolWrapper(GPUGroupElement* h_A, GPUConv2DKey k);

uint8_t *readFile(std::string filename, size_t* input_size);
