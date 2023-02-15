#include "gpu_comms.h"
#include "gpu_data_types.h"
#include <../group_element.h>

GPUGroupElement* gpuReconstruct(GPUGroupElement* h_A0, int N, Peer* peer, int party, Stats*);
GPUGroupElement* getMaskedInputOnCpu(int N, int bw, int party, Peer* peer, GPUGroupElement* h_mask_I, GPUGroupElement** h_I);
GPUGroupElement* getMaskedInputOnGpu(int N, int bw, int party, Peer* peer, GPUGroupElement* mask_I, GPUGroupElement** h_I);
GPUGroupElement* initRandom(int N, int bw);
void initRandomInPlace(GPUGroupElement* random, int N, int bw);
GPUGroupElement* initWithConst(int N, int bw, GPUGroupElement x);
void gpuReconstructInPlace(GPUGroupElement* d_A0, int numBits, int N, Peer* peer, int party, Stats* s);
void gpuReconstructBits(uint32_t* d_A0, int N, Peer* peer, int party, Stats* s);
GPUGroupElement randomGE(int bw);
void initCPURandomness();
std::array<osuCrypto::block, 2> getRandomAESBlockPair();
std::pair<GPUGroupElement*, GPUGroupElement*> maskInput(int N, int bw, int party, Peer* peer, GPUGroupElement* h_I, GPUGroupElement* h_mask_I, Stats* s);
void initWeightsHelper(GPUGroupElement* W, int N, double range, int party, Peer* peer);

