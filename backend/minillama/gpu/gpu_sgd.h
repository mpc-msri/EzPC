#pragma once

#include "gpu_data_types.h"
#include "gpu_truncate.h"

static const GPUGroupElement lr_fp = 1;
static const GPUGroupElement lr_scale = 6;
static const GPUGroupElement mom_fp = 29;
static const GPUGroupElement mom_scale = 5;
static const GPUGroupElement scale = 24;

void genGpuSGDWithMomentumKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int N, GPUGroupElement* W, GPUGroupElement* Vw, GPUGroupElement* dW, int scaleW, int scaleVw, int scaledW, TruncateType t);
void readGpuSGDWithMomentumKey(TruncateType t, GPUSignExtendKey* truncateKeyVw, GPUSignExtendKey* truncateKeyW, uint8_t** key_as_bytes, int scaleW, int scaleVw, int scaledW);
void gpuSgdWithMomentum(int bin, int bout, int N, GPUGroupElement* h_W, GPUGroupElement* d_W, 
GPUGroupElement* h_Vw, GPUGroupElement* d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t,
GPUSignExtendKey truncateKeyVw, GPUSignExtendKey truncateKeyW, int party, Peer* peer, AESGlobalContext* gaes, Stats* s);
void checkSgdWithMomentum(int bin, int bout, int N, 
GPUGroupElement* h_W, GPUGroupElement* h_Vw, GPUGroupElement* h_dW,
GPUGroupElement* h_masked_W, GPUGroupElement* h_masked_Vw, 
GPUGroupElement* h_mask_W, GPUGroupElement* h_mask_Vw,  
int scaleW, int scaleVw, int scaledW);