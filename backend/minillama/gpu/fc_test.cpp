#include "fc_layer.h"
#include "gpu_data_types.h"
#include "cpu_fss.h"
// #include "gpu_truncate.h"
#include "gpu_sgd.h"
#include "../input_prng.h"
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include <cassert>
#include <cstdint>

extern "C" void initAESContext(AESGlobalContext* g);

extern "C" GPUGroupElement *gpuMatmulWrapper(GPUMatmulKey k, GPUGroupElement* h_A, GPUGroupElement* h_B, GPUGroupElement* h_C, bool cIsBias);
extern "C" GPUGroupElement* getBiasGradWrapper(int N, int M, int bw, GPUGroupElement* h_A);


int main(int argc, char *argv[]) {
    prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    initCPURandomness();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64, bout = 64, M = 100, N = 10, K = 64;
    int party = atoi(argv[1]);
    auto fc_layer = FCLayer(bin, bout, M, N, K, TruncateType::StochasticTruncate, TruncateType::StochasticTruncate, true);
    GPUGroupElement *h_X, *h_W, *h_Y, *h_Z, *h_grad, *h_Vw, *h_Vy;
    GPUGroupElement *h_mask_X, *h_mask_W, *h_mask_Y, *h_mask_Z, *h_mask_grad, *h_mask_dX, *h_mask_Vw, *h_mask_Vy, *h_mask_new_Vw, *h_mask_new_Vy, *h_mask_new_W, *h_mask_new_Y;
// check: have you reconstructed the masked output in the protocol?
    if(party == 0) {
        std::ofstream f1("matmul_key1.dat"), f2("matmul_key2.dat"); 
        h_mask_X = initRandom(fc_layer.p.size_X, bin);
        h_mask_W = initRandom(fc_layer.p.size_W, bin);
        h_mask_Y = initRandom(N, bin);
        h_mask_Z = (GPUGroupElement*) cpuMalloc(fc_layer.p.size_Z * sizeof(GPUGroupElement));
        h_mask_grad = initRandom(fc_layer.p.size_Z, bin);
        h_mask_dX = (GPUGroupElement*) cpuMalloc(fc_layer.p.size_X * sizeof(GPUGroupElement));
        h_mask_Vw = initRandom(fc_layer.p.size_W, bin);
        h_mask_Vy = initRandom(N, bin);
        // matmulKey.mem_size_F hasn't been initialized yet
        memcpy(fc_layer.mask_W, h_mask_W, fc_layer.p.size_W * sizeof(GPUGroupElement));
        memcpy(fc_layer.mask_Vw, h_mask_Vw, fc_layer.p.size_W * sizeof(GPUGroupElement));
        
        // uncomment for bias
        memcpy(fc_layer.mask_Y, h_mask_Y, N * sizeof(GPUGroupElement));
        memcpy(fc_layer.mask_Vy, h_mask_Vy, N * sizeof(GPUGroupElement));
        
        fc_layer.genForwardKey(f1, f2, h_mask_X, h_mask_Z);
        fc_layer.genBackwardKey(f1, f2, h_mask_grad, h_mask_dX);
        h_mask_new_Vw = (GPUGroupElement*) cpuMalloc(fc_layer.p.size_W * sizeof(GPUGroupElement));
        h_mask_new_Vy = (GPUGroupElement*) cpuMalloc(N * sizeof(GPUGroupElement));
        h_mask_new_W = (GPUGroupElement*) cpuMalloc(fc_layer.p.size_W * sizeof(GPUGroupElement));
        h_mask_new_Y = (GPUGroupElement*) cpuMalloc(N * sizeof(GPUGroupElement));
        memcpy(h_mask_new_Vw, fc_layer.mask_Vw, fc_layer.p.size_W * sizeof(GPUGroupElement));
        memcpy(h_mask_new_W, fc_layer.mask_W, fc_layer.p.size_W * sizeof(GPUGroupElement));
        
        // uncomment for bias
        memcpy(h_mask_new_Vy, fc_layer.mask_Vy, N * sizeof(GPUGroupElement));
        memcpy(h_mask_new_Y, fc_layer.mask_Y, N * sizeof(GPUGroupElement));

        f1.close();
        f2.close();
    }
    Peer* peer = connectToPeer(party, argv[2]);
    size_t file_size;
    uint8_t* key_as_bytes = readFile("matmul_key" + std::to_string(party+1) + ".dat", &file_size);
    fc_layer.readForwardKey(&key_as_bytes);
    fc_layer.readBackwardKey(&key_as_bytes);
    auto d_masked_X = getMaskedInputOnGpu(fc_layer.p.size_X, bin, party, peer, h_mask_X, &h_X);
    auto h_masked_W = getMaskedInputOnCpu(fc_layer.p.size_W, bin, party, peer, h_mask_W, &h_W);
    memcpy(fc_layer.W, h_masked_W, fc_layer.matmulKey.mem_size_B);
    
    // uncomment for bias
    auto h_masked_Y = getMaskedInputOnCpu(N, bin, party, peer, h_mask_Y, &h_Y);
    memcpy(fc_layer.Y, h_masked_Y, N * sizeof(GPUGroupElement));
    
    auto d_masked_Z = fc_layer.forward(peer, party, d_masked_X, &g);
    auto d_masked_grad = getMaskedInputOnGpu(fc_layer.p.size_Z, bout, party, peer, h_mask_grad, &h_grad);
    
    auto h_masked_Vw = getMaskedInputOnCpu(fc_layer.p.size_W, bout, party, peer, h_mask_Vw, &h_Vw);
    memcpy(fc_layer.Vw, h_masked_Vw, fc_layer.matmulKey.mem_size_B);

    //uncommment for bias
    auto h_masked_Vy = getMaskedInputOnCpu(N, bout, party, peer, h_mask_Vy, &h_Vy);
    memcpy(fc_layer.Vy, h_masked_Vy, N * sizeof(GPUGroupElement));
    
    auto d_masked_dX = fc_layer.backward(peer, party, d_masked_grad, &g);
    if(party == 0) {
        auto h_masked_Z = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_Z, fc_layer.matmulKey.mem_size_C, NULL);
        auto h_masked_dX = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_dX, fc_layer.matmulKey.mem_size_A, NULL);
        auto h_Z_ct = gpuMatmulWrapper(fc_layer.matmulKey, h_X, h_W, h_Y, true);
        checkStochasticTruncate(bin, bout /*- scale*/, scale, fc_layer.p.size_Z, h_masked_Z, h_mask_Z, h_Z_ct);
        auto h_dX_ct = gpuMatmulWrapper(fc_layer.matmulKeydX, h_grad, h_W, NULL, false);
        checkStochasticTruncate(bin, bout, scale, fc_layer.p.size_X, h_masked_dX, h_mask_dX, h_dX_ct);
        auto h_dW_ct = gpuMatmulWrapper(fc_layer.matmulKeydW, h_X, h_grad, NULL, false);
        printf("checking sgd with momentum for W\n");
        checkSgdWithMomentum(bin, bout, fc_layer.p.size_W, h_W, h_Vw, h_dW_ct, fc_layer.W, fc_layer.Vw, 
        h_mask_new_W, h_mask_new_Vw, scale, 2*scale, 2*scale);
        auto h_dY_ct = getBiasGradWrapper(M, N, bout, h_grad);
        printf("checking sgd with momentum for Y\n");
        checkSgdWithMomentum(bin, bout, N, h_Y, h_Vy, h_dY_ct, fc_layer.Y, fc_layer.Vy, h_mask_new_Y, h_mask_new_Vy,  2*scale, 2*scale - lr_scale, scale);
    }
    return 0;
}
