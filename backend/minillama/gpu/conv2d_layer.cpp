#include "conv2d_layer.h"
#include "gpu_data_types.h"
#include "gpu_sgd.h"
#include "gpu_mem.h"
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include "gpu_truncate.h"
#include "gpu_sgd.h"
#include <cstdint>
#include <cassert>
#include <cmath>

// #include "gpu_data_types.h"

extern "C" GPUGroupElement *gpuConv2DWrapper(GPUConv2DKey k, GPUGroupElement* h_I, GPUGroupElement* h_F, GPUGroupElement* h_C, char op, bool cIsBias);
extern "C" GPUGroupElement *gpu_conv2d(GPUConv2DKey k, int party, GPUGroupElement *d_I, GPUGroupElement *d_F, GPUGroupElement* d_a, GPUGroupElement* d_b, GPUGroupElement* h_bias, Stats* s, char op);
extern "C" void gpuAddBiasWrapper(int N, int M, int bw, GPUGroupElement* h_A, GPUGroupElement* h_b);
extern "C" GPUGroupElement* getBiasGrad(int N, int M, int bw, GPUGroupElement* d_A);
extern "C" GPUGroupElement* getBiasGradWrapper(int N, int M, int bw, GPUGroupElement* h_A);

// need to fix this so dI is not always computed
Conv2DLayer::Conv2DLayer(int bin, int bout, int N, int H, int W, int CI, int FH, int FW, int CO, 
        int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, int strideH, int strideW, bool useBias, TruncateType tf, TruncateType tb, bool computedI) {
    assert(bin == bout && bin == 64);
    p.bin = bin;
    p.bout = bout;
    p.N = N;
    p.H = H;
    p.W = W;
    p.CI = CI;
    p.FH = FH;
    p.FW = FW;
    p.CO = CO;
    p.zPadHLeft = zPadHLeft;
    p.zPadHRight = zPadHRight;
    p.zPadWLeft = zPadWLeft;
    p.zPadWRight = zPadWRight;
    p.strideH = strideH;
    p.strideW = strideW;
    p.OH = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    p.OW = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    // printf("output H, W: %d %d\n", p.OH, p.OW);
    p.size_I = N * H * W * CI;
    p.size_F = CO * FH * FW * CI;
    p.size_O = N * p.OH * p.OW * CO;
    this->useBias = useBias;
    s.comm_time = 0;
    s.transfer_time = 0;
    this->tf = tf;
    this->tb = tb;
    size_t memSizeI = p.size_I * sizeof(GPUGroupElement);
    size_t memSizeF = p.size_F * sizeof(GPUGroupElement);
    size_t memSizeO = p.size_O * sizeof(GPUGroupElement);

    I = (GPUGroupElement*) cpuMalloc(memSizeI);
    F = (GPUGroupElement*) cpuMalloc(memSizeF);
    Vf = (GPUGroupElement*) cpuMalloc(memSizeF);
    memset(F, 0, memSizeF);
    memset(Vf, 0, memSizeF);
    // this is okay to do because we will never use I and mask_I
    // at the same time
    mask_I = I; 
    d_mask_I = NULL;
    mask_F = F;
    // initRandomInPlace(mask_F, p.size_F, p.bin);
    // mask_C = (GPUGroupElement*) cpuMalloc(memSizeO);
    mask_Vf = Vf;
    
    // mask_dI comes from outside
    // mask_dF = (GPUGroupElement*) cpuMalloc(memSizeF);
    // mask_C is used in the forward pass and mask_dI is used in the backward pass
    // dI and C are not the same size
    // mask_dI = (GPUGroupElement*) cpuMalloc(memSizeI);
    if(useBias) {
        size_t memSizeB = p.CO * sizeof(GPUGroupElement);
        b = (GPUGroupElement*) cpuMalloc(memSizeB);
        Vb = (GPUGroupElement*) cpuMalloc(memSizeB);
        memset(b, 0, memSizeB);
        memset(Vb, 0, memSizeB);
        // mask_db = (GPUGroupElement*) cpuMalloc(memSizeB);
    } else {
        b = NULL;
        Vb = NULL;
    }
    mask_b = b;
    mask_Vb = Vb;
    this->computedI = computedI;
}

// void Conv2DLayer::clear() {
//     size_t memSizeF = p.size_F * sizeof(GPUGroupElement);
//     memset(Vf, 0, memSizeF);
//     size_t memSizeB = p.CO * sizeof(GPUGroupElement);
//     memset(Vb, 0, memSizeB);
// }

void Conv2DLayer::initConvKey() {
    memcpy(&convKey, &p, 17 * sizeof(int));
    convKey.size_I = p.size_I;
    convKey.size_F = p.size_F;
    convKey.size_O = p.size_O;
    convKey.mem_size_I = p.size_I * sizeof(GPUGroupElement);
    convKey.mem_size_F = p.size_F * sizeof(GPUGroupElement);
    convKey.mem_size_O = p.size_O * sizeof(GPUGroupElement);
}

void Conv2DLayer::initConvKeydI() {
    memcpy(&convKeydI, &p, 17 * sizeof(int));
    convKeydI.size_I = convKey.size_O;
    convKeydI.mem_size_I = convKey.mem_size_O;
    convKeydI.size_F = convKey.size_F;
    convKeydI.mem_size_F = convKey.mem_size_F;
    convKeydI.size_O = convKey.size_I;
    convKeydI.mem_size_O = convKey.mem_size_I;
}

void Conv2DLayer::initConvKeydF() {
    memcpy(&convKeydF, &p, 17 * sizeof(int));
    convKeydF.size_I = p.size_O;
    convKeydF.mem_size_I = convKey.mem_size_O;
    convKeydF.size_F = p.size_I;
    convKeydF.mem_size_F = convKey.mem_size_I;
    convKeydF.size_O = p.size_F;
    convKeydF.mem_size_O = convKey.mem_size_F;
}
    // the last weights update has already given us a mask for F
    // need to change this so the output gets written to the same memory as before
    // need to add the bias mask to the output mask
    // because the bias will get added to the output

void Conv2DLayer::genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_I, GPUGroupElement* mask_truncated_C) {
    initConvKey(); 
    // the copy might not be necessary
    memcpy(this->mask_I, mask_I, convKey.mem_size_I);
    
    auto mask_C = initRandom(p.size_O, p.bout);
    auto masked_C = gpuConv2DWrapper(convKey, mask_I, mask_F, mask_C, 0, false);

    writeSecretSharesToFile(f1, f2, p.bout, p.size_I, mask_I);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_F, mask_F);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_O, masked_C);

    cpuFree(masked_C);
    
    // bias has scale 2s
    if(useBias) gpuAddBiasWrapper(p.size_O / p.CO, p.CO, p.bout, mask_C, mask_b);
    genGPUTruncateKey(f1, f2, tf, p.bin, p.bout, scale, p.size_O, mask_C, mask_truncated_C);
    cpuFree(mask_C);
}
// need to truncate dI

void Conv2DLayer::genBackwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_grad, GPUGroupElement* mask_truncated_dI) {
    // need to free all the leaked memory
    initConvKeydF();
    auto mask_dF = initRandom(p.size_F, p.bin);
    // is mask_F missing? it shouldn't be but is it?
    auto masked_dF = gpuConv2DWrapper(convKeydF, mask_grad, mask_I, mask_dF, 2, false);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_O, mask_grad);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_F, masked_dF);
    cpuFree(masked_dF);
    // this needs to be computed here because ow the contents of mask_F will change
    if(computedI) {
        initConvKeydI();
        auto mask_dI = initRandom(p.size_I, p.bin);
        auto masked_dI = gpuConv2DWrapper(convKeydI, mask_grad, mask_F, mask_dI, 1, false);
        writeSecretSharesToFile(f1, f2, p.bout, p.size_I, masked_dI);
        cpuFree(masked_dI);
        genGPUTruncateKey(f1, f2, tb, p.bin, p.bout, scale, p.size_I, mask_dI, mask_truncated_dI);
        cpuFree(mask_dI);
    }
    genGpuSGDWithMomentumKey(f1, f2, p.bin, p.bout, p.size_F, mask_F, mask_Vf, mask_dF, scale, 2*scale, 2*scale, tb);
    if(useBias) {
        auto mask_db = getBiasGradWrapper(p.size_O / p.CO, p.CO, p.bin, mask_grad);
        genGpuSGDWithMomentumKey(f1, f2, p.bin, p.bout, p.CO, mask_b, mask_Vb, mask_db, 2*scale, 2*scale - lr_scale, scale, tb);
        cpuFree(mask_db);
    }
    cpuFree(mask_dF);
}


void Conv2DLayer::readForwardKey(uint8_t** key_as_bytes) {
    initConvKey(); 
    convKey.I = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += convKey.mem_size_I;
    convKey.F = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += convKey.mem_size_F;
    convKey.O = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += convKey.mem_size_O;
    readGPUTruncateKey(tf, &truncateKeyC, key_as_bytes);
}

void Conv2DLayer::readBackwardKey(uint8_t** key_as_bytes) {
    GPUGroupElement* mask_grad = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += convKey.mem_size_O;
    GPUGroupElement* mask_dF = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += convKey.mem_size_F;

    // grad * input
    initConvKeydF();
    convKeydF.I = mask_grad;
    convKeydF.F = convKey.I;
    convKeydF.O = mask_dF;

    if(computedI) {
        GPUGroupElement* mask_dI = (GPUGroupElement*) *key_as_bytes;
        *key_as_bytes += convKey.mem_size_I;
        // grad * F
        initConvKeydI();
        convKeydI.I = mask_grad;
        convKeydI.F = convKey.F;
        convKeydI.O = mask_dI;

         // should refactor this later to look pretty
        readGPUTruncateKey(tb, &truncateKeydI, key_as_bytes); 
    }
    // readGpuSGDWithMomentumKey(tb, &truncateKeyVf, &truncateKeyF, &truncateKeyVb, key_as_bytes, useBias);
    readGpuSGDWithMomentumKey(tb, &truncateKeyVf, &truncateKeyF, key_as_bytes, scale, 2*scale, 2*scale);
    if(useBias) readGpuSGDWithMomentumKey(tb, &truncateKeyVb, &truncateKeyb, key_as_bytes, 2*scale, 2*scale - lr_scale, scale);
}

GPUGroupElement* Conv2DLayer::forward(Peer* peer, int party, GPUGroupElement* d_I, AESGlobalContext* gaes) {
    GPUGroupElement /**d_mask_I,*/ *d_F, *d_mask_F;
    moveIntoCPUMem((uint8_t*) I, (uint8_t*) d_I, convKey.mem_size_I, &s);
    if(!d_mask_I) d_mask_I = (GPUGroupElement*) moveToGPU((uint8_t*) convKey.I, convKey.mem_size_I, &s);
    d_F = (GPUGroupElement*) moveToGPU((uint8_t*) F, convKey.mem_size_F, &s);
    d_mask_F = (GPUGroupElement*) moveToGPU((uint8_t*) convKey.F, convKey.mem_size_F, &s);
    auto d_C = gpu_conv2d(convKey, party, d_I, d_F, d_mask_I, d_mask_F, b, &s, 0);

    // should not be freeing d_I who knows where else it is being used
    gpuFree(d_I);
    gpuFree(d_F);
    gpuFree(d_mask_I);
    d_mask_I = NULL;
    gpuFree(d_mask_F);

    gpuReconstructInPlace(d_C, p.bout, p.size_O, peer, party, &s);
    gpuTruncate(p.bin, p.bout, tf, truncateKeyC, scale, peer, party, p.size_O, d_C, gaes, &s);
    return d_C;
}

GPUGroupElement* Conv2DLayer::backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* gaes) {
    auto d_mask_incomingGrad = (GPUGroupElement*) moveToGPU((uint8_t*) convKeydF.I, convKeydF.mem_size_I, &s);
    auto d_mask_I = (GPUGroupElement*) moveToGPU((uint8_t*) convKey.I, convKey.mem_size_I, &s);
    auto d_I = (GPUGroupElement*) moveToGPU((uint8_t*) I, convKey.mem_size_I, &s);
    auto d_F = (GPUGroupElement*) moveToGPU((uint8_t*) F, convKey.mem_size_F, &s);

    GPUGroupElement* d_dI = NULL;
    if(computedI) {
        auto d_mask_F = (GPUGroupElement*) moveToGPU((uint8_t*) convKey.F, convKey.mem_size_F, &s);
        d_dI = gpu_conv2d(convKeydI, party, d_incomingGrad, d_F, d_mask_incomingGrad, d_mask_F, NULL, &s, 1);
        gpuFree(d_mask_F);
        gpuReconstructInPlace(d_dI, p.bin, p.size_I, peer, party, &s);
        gpuTruncate(p.bin, p.bout, tb, truncateKeydI, scale, peer, party, p.size_I, d_dI, gaes, &s);
    }
    
    auto d_dF = gpu_conv2d(convKeydF, party, d_incomingGrad, d_I, d_mask_incomingGrad, d_mask_I, NULL, &s, 2);
    gpuReconstructInPlace(d_dF, p.bin, p.size_F, peer, party, &s);
    gpuSgdWithMomentum(p.bin, p.bout, p.size_F, F, d_F, Vf, d_dF, scale, 2*scale, 2*scale, tb, truncateKeyVf, truncateKeyF, party, peer, gaes, &s);

    if(useBias) {
        auto d_db = getBiasGrad(p.size_O / p.CO, p.CO, p.bout, d_incomingGrad);
        gpuSgdWithMomentum(p.bin, p.bout, p.CO, b, NULL, Vb, d_db, 2*scale, 2*scale - lr_scale, scale, tb, truncateKeyVb, truncateKeyb, 
        party, peer, gaes, &s);
        gpuFree(d_db);
    }

    gpuFree(d_incomingGrad);
    gpuFree(d_I);
    gpuFree(d_F);
    gpuFree(d_mask_incomingGrad);
    gpuFree(d_mask_I);
    gpuFree(d_dF);

    return d_dI;
}

void Conv2DLayer::initWeights(Peer* peer, int party) {
    // printf("!!!!!!!!!!!!!inside init weights for conv\n");
    double xavier = 1.0 / sqrt(p.CI * p.FH * p.FW);
    initWeightsHelper(F, p.size_F, xavier * (1ULL<<scale), party, peer);
    if(useBias) initWeightsHelper(b, p.CO, xavier * (1ULL<<(2*scale)), party, peer);
}