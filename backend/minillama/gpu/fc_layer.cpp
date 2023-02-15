#include "fc_layer.h"
#include "gpu_data_types.h"
#include "gpu_mem.h"
#include "gpu_fss_utils.h"
#include "gpu_file_utils.h"
#include "gpu_truncate.h"
#include "gpu_sgd.h"
#include <cstdint>
#include <cassert>
#include <cmath>


extern "C" GPUGroupElement *gpuMatmulWrapper(GPUMatmulKey k, GPUGroupElement* h_A, GPUGroupElement* h_B, GPUGroupElement* h_C, /*bool rowMajA, bool rowMajB, bool rowMajC,*/ bool cIsBias);
extern "C" GPUGroupElement *gpu_matmul(GPUMatmulKey k, int party, GPUGroupElement *d_A, GPUGroupElement *d_B, GPUGroupElement* d_mask_A, GPUGroupElement* d_mask_B, GPUGroupElement* h_bias, /*bool rowMajA, bool rowMajB, bool rowMajC,*/ Stats* s/*, GPUContext* c*/);
extern "C" void gpuAddBiasWrapper(int N, int M, int bw, GPUGroupElement* h_A, GPUGroupElement* h_b);
extern "C" GPUGroupElement* getBiasGrad(int N, int M, int bw, GPUGroupElement* d_A);
extern "C" GPUGroupElement* getBiasGradWrapper(int N, int M, int bw, GPUGroupElement* h_A);


FCLayer::FCLayer(int bin, int bout, int M, int N, int K, TruncateType tf, TruncateType tb, bool useBias) {
    assert(bin == bout && bin == 64);
    p.bin = bin;
    p.bout = bout;
    p.M = M;
    p.N = N;
    p.K = K;
    p.size_W = K * N;
    p.size_X = M * K;
    p.size_Y = N;
    p.size_Z = M * N;
    mask_X = (GPUGroupElement*) cpuMalloc(p.size_X * sizeof(GPUGroupElement));
    mask_W = (GPUGroupElement*) cpuMalloc(p.size_W * sizeof(GPUGroupElement));
    mask_Z = (GPUGroupElement*) cpuMalloc(p.size_Z * sizeof(GPUGroupElement));
    mask_dX = (GPUGroupElement*) cpuMalloc(p.size_X * sizeof(GPUGroupElement));
    mask_dW = (GPUGroupElement*) cpuMalloc(p.size_W * sizeof(GPUGroupElement));
    mask_Vw = (GPUGroupElement*) cpuMalloc(p.size_W * sizeof(GPUGroupElement));
    this->tf = tf;
    this->tb = tb;
    this->useBias = useBias;
    if(useBias) {
        printf("init bias mem size: %u %d %d %d\n", p.size_Y * sizeof(GPUGroupElement), p.M, p.K, p.N);
        mask_Y = (GPUGroupElement*) cpuMalloc(p.size_Y * sizeof(GPUGroupElement));
        mask_Vy = (GPUGroupElement*) cpuMalloc(p.size_Y * sizeof(GPUGroupElement));
        mask_dY = (GPUGroupElement*) cpuMalloc(p.size_Y * sizeof(GPUGroupElement));
    }
    X = mask_X;
    W = mask_W;
    Y = mask_Y;
    Vw = mask_Vw;
    Vy = mask_Vy;
}

void initMemSize(GPUMatmulKey* k) {
    k->mem_size_A = k->size_A * sizeof(GPUGroupElement);
    k->mem_size_B = k->size_B * sizeof(GPUGroupElement);
    k->mem_size_C = k->size_C * sizeof(GPUGroupElement);
}

void FCLayer::initMatmulKey() {
    memcpy(&matmulKey, &p, 5 * sizeof(int));
    matmulKey.rowMajA = true;
    matmulKey.rowMajB = true;
    matmulKey.rowMajC = true;
    matmulKey.size_A = p.size_X;
    matmulKey.size_B = p.size_W;
    matmulKey.size_C = p.size_Z;
    initMemSize(&matmulKey);
}

void FCLayer::initMatmulKeydW() {
    matmulKeydW.M = p.K;
    matmulKeydW.K = p.M;
    matmulKeydW.N = p.N;
    matmulKeydW.rowMajA = false;
    matmulKeydW.rowMajB = true;
    matmulKeydW.rowMajC = true;
    matmulKeydW.size_A = p.size_X;
    matmulKeydW.size_B = p.size_Z;
    matmulKeydW.size_C = p.size_W;
    initMemSize(&matmulKeydW);
}

void FCLayer::initMatmulKeydX(){
    matmulKeydX.M = p.M;
    matmulKeydX.K = p.N;
    matmulKeydX.N = p.K;
    matmulKeydX.rowMajA = true;
    matmulKeydX.rowMajB = false;
    matmulKeydX.rowMajC = true;
    matmulKeydX.size_A = p.size_Z;
    matmulKeydX.size_B = p.size_W;
    matmulKeydX.size_C = p.size_X;
    initMemSize(&matmulKeydX);
}
// neha: to fix: maxpool, and make it so the conv2d output is 40 bits???? (bout == 40????)
void FCLayer::genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_X, GPUGroupElement* mask_truncated_Z) {
    initMatmulKey();
    memcpy(this->mask_X, mask_X, matmulKey.mem_size_A);
    initRandomInPlace(mask_Z, p.size_Z, p.bin);
    auto masked_Z = gpuMatmulWrapper(matmulKey, mask_X, mask_W, mask_Z, false);//, true, true, true);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_X, mask_X);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_W, mask_W);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_Z, masked_Z);
    cpuFree(masked_Z);
    if(useBias) gpuAddBiasWrapper(p.M, p.N, p.bin, mask_Z, mask_Y);
    genGPUTruncateKey(f1, f2, tf, p.bin, p.bout, scale, p.size_Z, mask_Z, mask_truncated_Z);
}

void FCLayer::genBackwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* mask_grad, GPUGroupElement* mask_truncated_dX) {
    initMatmulKeydX();
    initRandomInPlace(mask_dX, p.size_X, p.bin);
    auto masked_dX = gpuMatmulWrapper(matmulKeydX, mask_grad, mask_W, mask_dX, false/*, true, false, true*/);
    
    initMatmulKeydW();
    initRandomInPlace(mask_dW, p.size_W, p.bout);
    auto masked_dW = gpuMatmulWrapper(matmulKeydW, mask_X, mask_grad, mask_dW, false/*false, true, true*/);

    writeSecretSharesToFile(f1, f2, p.bout, p.size_Z, mask_grad);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_X, masked_dX);
    writeSecretSharesToFile(f1, f2, p.bout, p.size_W, masked_dW);

    cpuFree(masked_dX);
    cpuFree(masked_dW);

    genGPUTruncateKey(f1, f2, tb, p.bin, p.bout, scale, p.size_X, mask_dX, mask_truncated_dX);
    genGpuSGDWithMomentumKey(f1, f2, p.bin, p.bout, p.size_W, mask_W, mask_Vw, mask_dW, scale, 2*scale, 2*scale, tb);
    if(useBias) {
        auto mask_dY = getBiasGradWrapper(p.M, p.N, p.bin, mask_grad);
        genGpuSGDWithMomentumKey(f1, f2, p.bin, p.bout, p.N, mask_Y, mask_Vy, mask_dY, 2*scale, 2*scale - lr_scale, scale, tb);
        cpuFree(mask_dY);
    }
}

void FCLayer::readForwardKey(uint8_t** key_as_bytes) {
    initMatmulKey();
    matmulKey.A = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += matmulKey.mem_size_A;
    
    matmulKey.B = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += matmulKey.mem_size_B;

    matmulKey.C = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += matmulKey.mem_size_C;

    readGPUTruncateKey(tf, &truncateKeyZ, key_as_bytes);
}

void FCLayer::readBackwardKey(uint8_t** key_as_bytes) {
    GPUGroupElement* mask_grad = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += matmulKey.mem_size_C;

    GPUGroupElement* mask_dX = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += matmulKey.mem_size_A;

    GPUGroupElement* mask_dW = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += matmulKey.mem_size_B;

    initMatmulKeydW();
    matmulKeydW.A = matmulKey.A;
    matmulKeydW.B = mask_grad;
    matmulKeydW.C = mask_dW;

    initMatmulKeydX();
    matmulKeydX.A = mask_grad;
    matmulKeydX.B = matmulKey.B;
    matmulKeydX.C = mask_dX;

    readGPUTruncateKey(tb, &truncateKeydX, key_as_bytes); 
    readGpuSGDWithMomentumKey(tb, &truncateKeyVw, &truncateKeyW, key_as_bytes, scale, 2*scale, 2*scale);
    if(useBias) readGpuSGDWithMomentumKey(tb, &truncateKeyVy, &truncateKeyY, key_as_bytes, 2*scale, 2*scale - lr_scale, scale);
}
        
GPUGroupElement* FCLayer::forward(Peer* peer, int party, GPUGroupElement* d_X, AESGlobalContext* gaes) {
    moveIntoCPUMem((uint8_t*) X, (uint8_t*) d_X, matmulKey.mem_size_A, &s);
    auto d_mask_X = (GPUGroupElement*) moveToGPU((uint8_t*) matmulKey.A, matmulKey.mem_size_A, &s);
    auto d_W = (GPUGroupElement*) moveToGPU((uint8_t*) W, matmulKey.mem_size_B, &s);
    auto d_mask_W = (GPUGroupElement*) moveToGPU((uint8_t*) matmulKey.B, matmulKey.mem_size_B, &s);
    auto d_Z = gpu_matmul(matmulKey, party, d_X, d_W, d_mask_X, d_mask_W, /*true, true, true,*/ Y, &s);

    gpuFree(d_X);
    gpuFree(d_mask_X);
    gpuFree(d_W);
    gpuFree(d_mask_W);

    gpuReconstructInPlace(d_Z, p.bout, p.size_Z, peer, party, &s);
    gpuTruncate(p.bin, p.bout, tf, truncateKeyZ, scale, peer, party, p.size_Z, d_Z, gaes, &s);
    return d_Z;
}

GPUGroupElement* FCLayer::backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* gaes) {
    auto d_mask_grad = (GPUGroupElement*) moveToGPU((uint8_t*) matmulKeydX.A, matmulKeydX.mem_size_A, &s);
    auto d_W = (GPUGroupElement*) moveToGPU((uint8_t*) W, matmulKeydX.mem_size_B, &s);  
    auto d_mask_W = (GPUGroupElement*) moveToGPU((uint8_t*) matmulKeydX.B, matmulKeydX.mem_size_B, &s);
    auto d_dX = gpu_matmul(matmulKeydX, party, d_incomingGrad, d_W, d_mask_grad, d_mask_W, NULL,/*true, false, true,*/ &s);
    gpuReconstructInPlace(d_dX, p.bout, p.size_X, peer, party, &s);
    gpuTruncate(p.bin, p.bout, tb, truncateKeydX, scale, peer, party, p.size_X, d_dX, gaes, &s);    
    gpuFree(d_mask_W);

    auto d_X = (GPUGroupElement*) moveToGPU((uint8_t*) X, matmulKeydW.mem_size_A, &s);
    auto d_mask_X = (GPUGroupElement*) moveToGPU((uint8_t*) matmulKeydW.A, matmulKeydW.mem_size_A, &s);
    auto d_dW = gpu_matmul(matmulKeydW, party, d_X, d_incomingGrad, d_mask_X, d_mask_grad, NULL,/*false, true, true,*/ &s);
    gpuReconstructInPlace(d_dW, p.bout, p.size_W, peer, party, &s);

    gpuFree(d_X);
    gpuFree(d_mask_X);

    gpuSgdWithMomentum(p.bin, p.bout, p.size_W, W, d_W, Vw, d_dW, scale, 2*scale, 2*scale, tb, truncateKeyVw, truncateKeyW, party, peer, gaes, &s);
    gpuFree(d_W);
    gpuFree(d_dW);
    if(useBias) {
        // printf("getting the bias grad\n");
        auto d_dY = getBiasGrad(p.M, p.N, p.bout, d_incomingGrad);
        // printf("starting sgd with momentum for bias\n");
        gpuSgdWithMomentum(p.bin, p.bout, p.N, Y, NULL, Vy, d_dY, 2*scale, 2*scale - lr_scale, scale, tb, truncateKeyVy, truncateKeyY, 
        party, peer, gaes, &s);
        gpuFree(d_dY);
    }
    gpuFree(d_incomingGrad);
    gpuFree(d_mask_grad);
    return d_dX;
}

void FCLayer::initWeights(Peer* peer, int party) {
    double xavier = 1.0 / sqrt(p.K);
    initWeightsHelper(W, p.size_W, xavier * (1ULL<<scale), party, peer);
    if(useBias) initWeightsHelper(Y, p.size_Y, xavier * (1ULL<<(2*scale)), party, peer);
}