#include "gpu_sgd.h"
#include "gpu_data_types.h"
#include "gpu_mem.h"
#include <cassert>

extern "C" void gpuLeftShiftAndAddWrapper(int N, GPUGroupElement* h_A, GPUGroupElement* h_B, GPUGroupElement* h_C, int shift, GPUGroupElement alpha);
extern "C" void gpuLeftShiftAndAdd(int N, GPUGroupElement* d_A, GPUGroupElement* d_b, GPUGroupElement* d_C, int shift, GPUGroupElement alpha);


void genGpuSGDWithMomentumKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int N, GPUGroupElement* W, GPUGroupElement* Vw, GPUGroupElement* dW, int scaleW, int scaleVw, int scaledW, TruncateType t) {
    // Vw = dW << mom_scale + mom_fp * Vw
    int shift = scaleVw + mom_scale - scaledW;
    assert(shift > 0);
    // the dW mask gets moved to the left by shift and written to dW
    gpuLeftShiftAndAddWrapper(N, dW, Vw, dW, shift, mom_fp);
    // genTruncateKey will generate the new Vw mask
    genGPUTruncateKey(f1, f2, t, bin, bout, mom_scale, N, dW, Vw);
    // dW = W << (scale + lr_scale) + lr_fp * Vw;
    shift = scaleVw + lr_scale - scaleW;
    // Neha: this is wrong. it needs to be -lr
    gpuLeftShiftAndAddWrapper(N, W, Vw, dW, shift, -lr_fp);
    genGPUTruncateKey(f1, f2, shift > 0 ? t : TruncateType::None, bin, bout, shift, N, dW, W);
}   

void readGpuSGDWithMomentumKey(TruncateType t, GPUSignExtendKey* truncateKeyVw, GPUSignExtendKey* truncateKeyW, uint8_t** key_as_bytes, int scaleW, int scaleVw, int scaledW) {
    readGPUTruncateKey(t, truncateKeyVw, key_as_bytes);
    int shift = lr_scale + scaleVw - scaleW;
    if(shift > 0) readGPUTruncateKey(t, truncateKeyW, key_as_bytes);
}

void gpuSgdWithMomentum(int bin, int bout, int N, GPUGroupElement* h_W, GPUGroupElement* d_W, 
GPUGroupElement* h_Vw, GPUGroupElement* d_dW, int scaleW, int scaleVw, int scaledW, TruncateType t,
GPUSignExtendKey truncateKeyVw, GPUSignExtendKey truncateKeyW, int party, Peer* peer, AESGlobalContext* gaes, Stats* s) {
    size_t memSizeW = N * sizeof(GPUGroupElement);
    auto d_Vw = (GPUGroupElement*) moveToGPU((uint8_t*) h_Vw, memSizeW, s);
    int shift = mom_scale + scaleVw - scaledW;
    // the d_dW mask got moved to the left by shift
    gpuLeftShiftAndAdd(N, d_dW, d_Vw, d_Vw, shift, mom_fp);
    gpuTruncate(bin, bout, t, truncateKeyVw, mom_scale, peer, party, N, d_Vw, gaes, s);
    moveIntoCPUMem((uint8_t*) h_Vw, (uint8_t*) d_Vw/*d_dW*/, memSizeW, s);
    bool dWWasNull = false;
    if(d_W == NULL) {
        d_W = (GPUGroupElement*) moveToGPU((uint8_t*) h_W, memSizeW, s);
        dWWasNull = true;
    }
    shift = lr_scale + scaleVw - scaleW;
    // this is wrong it needs to be -lr
    gpuLeftShiftAndAdd(N, d_W, d_Vw, d_W, shift, -lr_fp);
    if(shift > 0) gpuTruncate(bin, bout, t, truncateKeyW, shift, peer, party, N, d_W, gaes, s);
    moveIntoCPUMem((uint8_t*) h_W, (uint8_t*) d_W, memSizeW, s);
    if(dWWasNull) gpuFree(d_W);
}

void checkSgdWithMomentum(int bin, int bout, int N, 
GPUGroupElement* h_W, GPUGroupElement* h_Vw, GPUGroupElement* h_dW,
GPUGroupElement* h_masked_W, GPUGroupElement* h_masked_Vw,
GPUGroupElement* h_mask_W, GPUGroupElement* h_mask_Vw,
int scaleW, int scaleVw, int scaledW) {
    int shiftdW = scaleVw + mom_scale - scaledW;
    int shiftW = lr_scale + scaleVw - scaleW;
    for(int i = 0; i < N; i++) {
            auto vw = h_masked_Vw[i] - h_mask_Vw[i];
            auto vw_ct = cpuArs((h_dW[i] << shiftdW) + mom_fp * h_Vw[i], bin, mom_scale);
            if(i < 10) printf("%lu %lu\n", vw, vw_ct);
            assert(vw - vw_ct <= 1);
            auto w_ct = cpuArs((h_W[i] << shiftW) - lr_fp * vw_ct, bin, shiftW);
            // this is the new masked f
            auto w = h_masked_W[i] - h_mask_W[i];
            // need to test this when the starting vf is non-zero
            if(i < 10) printf("%lu %lu\n", w, w_ct);
            // the two is important
            assert(abs(static_cast<int64_t>(w - w_ct)) <= 2);
        }
}
