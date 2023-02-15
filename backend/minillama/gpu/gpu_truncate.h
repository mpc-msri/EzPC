#pragma once

#include "gpu_data_types.h"
#include "gpu_dcf.h"
#include "gpu_relu.h"
#include "gpu_stats.h"
#include "gpu_comms.h"

typedef GPUDReluKey GPUMaskedDCFKey;

struct GPUSignExtendKey {
    int bin, bout, N;
    GPUMaskedDCFKey dcfKey;
    GPUGroupElement *t, *p;
};

enum TruncateType {
    None,
    LocalLRS,
    LocalARS,
    StochasticTruncate
};

void genGPUStochasticTruncateKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int shift, int N, GPUGroupElement* inMask, GPUGroupElement* outMask);
void genGPUSignExtendKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int N, GPUGroupElement* inMask, GPUGroupElement* outMask);
GPUSignExtendKey readGPUSignExtendKey(uint8_t** key_as_bytes);
void gpuSignExtend(GPUSignExtendKey k, int party, Peer* peer, GPUGroupElement* d_I, AESGlobalContext* g, Stats* s);
void gpuStochasticTruncate(GPUSignExtendKey k, /*int bin, int bout,*/ int shift, int party, Peer* peer, GPUGroupElement* d_I, AESGlobalContext* g, Stats* s);
const auto genGPUMaskedDCFKey = genGPUDReluKey;
const auto readGPUMaskedDCFKey = readGPUDReluKey;
GPUGroupElement cpuArs(GPUGroupElement x, int bin, int shift);
void gpuTruncate(int bin, int bout, TruncateType t, GPUSignExtendKey signExtendKey, int shift, Peer* peer, int party, int N, GPUGroupElement* d_I, AESGlobalContext* gaes, Stats* s);
void genGPUTruncateKey(std::ostream& f1, std::ostream& f2, TruncateType t, int bin, int bout, int shift, int N, GPUGroupElement *inMask, GPUGroupElement *outMask);
void readGPUTruncateKey(TruncateType t, GPUSignExtendKey *truncateKey, uint8_t** key_as_bytes);
void checkStochasticTruncate(int bin, int bout, int shift, int N, GPUGroupElement* h_masked_A, GPUGroupElement* h_mask_A, GPUGroupElement* h_A_ct);
