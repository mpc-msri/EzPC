#include "conv2d_layer.h"
#include "gpu_data_types.h"
#include "cpu_fss.h"
#include "gpu_truncate.h"
#include "gpu_sgd.h"
#include <../input_prng.h>
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include <cassert>
#include <cstdint>

extern "C" void initAESContext(AESGlobalContext* g);

extern "C" GPUGroupElement *gpuConv2DWrapper(GPUConv2DKey k, GPUGroupElement* h_I, GPUGroupElement* h_F, GPUGroupElement* h_C, char op, bool cIsBias);

int main(int argc, char *argv[]) {
    prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    initCPURandomness();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64;
    int bout = 64;
    int N = 128;
    int H = 32;
    int W = 32;
    int CI = 3;
    int FH = 5;
    int FW = 5;
    int CO = 64;
    int zPadHLeft = 1;
    int zPadHRight = 1;
    int zPadWLeft = 1;
    int zPadWRight = 1;
    int strideH = 1;
    int strideW = 1;

    int party = atoi(argv[1]);
    auto conv2d_layer = Conv2DLayer(bin, bout, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, false);
    GPUGroupElement *h_I, *h_F, *h_b, *h_C, *h_grad, *h_dI, *h_Vb, *h_Vf;
    GPUGroupElement *h_mask_I, *h_mask_F, *h_mask_b, *h_mask_C, *h_mask_grad, *h_mask_dI, *h_mask_Vf, *h_mask_Vb, *h_mask_new_F, *h_mask_new_b, *h_mask_new_Vf, *h_mask_new_Vb;
// check: have you reconstructed the masked output in the protocol?
    if(party == 0) {
        std::ofstream f1("conv2d_key1.dat"), f2("conv2d_key2.dat"); 
        h_mask_I = initRandom(conv2d_layer.p.size_I, bin);
        h_mask_F = initRandom(conv2d_layer.p.size_F, bin);
        h_mask_b = initRandom(CO, bin);
        h_mask_C = (GPUGroupElement*) cpuMalloc(conv2d_layer.p.size_O * sizeof(GPUGroupElement));
        h_mask_grad = initRandom(conv2d_layer.p.size_O, bin);
        h_mask_dI = (GPUGroupElement*) cpuMalloc(conv2d_layer.p.size_I * sizeof(GPUGroupElement));
        h_mask_Vf = initRandom(conv2d_layer.p.size_F, bin);
        h_mask_Vb = initRandom(CO, bin);
        // convKey.mem_size_F hasn't been initialized yet
        memcpy(conv2d_layer.mask_F, h_mask_F, conv2d_layer.p.size_F * sizeof(GPUGroupElement));
        memcpy(conv2d_layer.mask_Vf, h_mask_Vf, conv2d_layer.p.size_F * sizeof(GPUGroupElement));
        memcpy(conv2d_layer.mask_b, h_mask_b, CO * sizeof(GPUGroupElement));
        memcpy(conv2d_layer.mask_Vb, h_mask_Vb, CO * sizeof(GPUGroupElement));
        // conv2d_layer.clear();
        conv2d_layer.genForwardKey(f1, f2, h_mask_I, h_mask_C);
        conv2d_layer.genBackwardKey(f1, f2, h_mask_grad, h_mask_dI);
        // it says dF but actually means updated F
        h_mask_new_Vf = (GPUGroupElement*) cpuMalloc(conv2d_layer.p.size_F * sizeof(GPUGroupElement));
        h_mask_new_Vb = (GPUGroupElement*) cpuMalloc(CO * sizeof(GPUGroupElement));
        h_mask_new_F = (GPUGroupElement*) cpuMalloc(conv2d_layer.p.size_F * sizeof(GPUGroupElement));
        h_mask_new_b = (GPUGroupElement*) cpuMalloc(CO * sizeof(GPUGroupElement));
        memcpy(h_mask_new_Vf, conv2d_layer.mask_Vf, conv2d_layer.p.size_F * sizeof(GPUGroupElement));
        memcpy(h_mask_new_Vb, conv2d_layer.mask_Vb, CO * sizeof(GPUGroupElement));
        // printf("Vf in test: %lu\n", h_mask_Vf[0]);
        memcpy(h_mask_new_F, conv2d_layer.mask_F, conv2d_layer.p.size_F * sizeof(GPUGroupElement));
        memcpy(h_mask_new_b, conv2d_layer.mask_b, CO * sizeof(GPUGroupElement));
        f1.close();
        f2.close();
    }
    Peer* peer = connectToPeer(party, argv[2]);
    size_t file_size;
    uint8_t* key_as_bytes = readFile("conv2d_key" + std::to_string(party+1) + ".dat", &file_size);
    conv2d_layer.readForwardKey(&key_as_bytes);
    conv2d_layer.readBackwardKey(&key_as_bytes);
    auto d_masked_I = getMaskedInputOnGpu(conv2d_layer.p.size_I, bin, party, peer, h_mask_I, &h_I);
    auto h_masked_F = getMaskedInputOnCpu(conv2d_layer.p.size_F, bin, party, peer, h_mask_F, &h_F);
    memcpy(conv2d_layer.F, h_masked_F, conv2d_layer.convKey.mem_size_F);
    auto h_masked_b = getMaskedInputOnCpu(CO, bin, party, peer, h_mask_b, &h_b);
    memcpy(conv2d_layer.b, h_masked_b, CO * sizeof(GPUGroupElement));
    // conv2d_layer.clear();
    auto d_C = conv2d_layer.forward(peer, party, d_masked_I, &g);
    auto d_masked_grad = getMaskedInputOnGpu(conv2d_layer.p.size_O, bout, party, peer, h_mask_grad, &h_grad);
    auto h_masked_Vf = getMaskedInputOnCpu(conv2d_layer.p.size_F, bout, party, peer, h_mask_Vf, &h_Vf);
    auto h_masked_Vb = getMaskedInputOnCpu(CO, bout, party, peer, h_mask_Vb, &h_Vb);
    memcpy(conv2d_layer.Vf, h_masked_Vf, conv2d_layer.convKey.mem_size_F);
    memcpy(conv2d_layer.Vb, h_masked_Vb, CO * sizeof(GPUGroupElement));
    auto d_masked_dI = conv2d_layer.backward(peer, party, d_masked_grad, &g);
    if(party == 0) {
        auto h_C = (GPUGroupElement*) moveToCPU((uint8_t*) d_C, conv2d_layer.convKey.mem_size_O, NULL);
        // auto h_masked_dI = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_dI, conv2d_layer.convKey.mem_size_I, NULL);
        auto h_C_ct = gpuConv2DWrapper(conv2d_layer.convKey, h_I, h_F, h_b, 0, true);
        for(int i = 0; i < conv2d_layer.p.size_O; i++) {
            auto output = h_C[i] - h_mask_C[i];
            mod(output, bin - scale);
            h_C_ct[i] >>= scale;
            // if(output - h_C_ct[i] > 1) printf("%lu %lu\n", output, h_C_ct[i]);
            auto diff = output - h_C_ct[i];
            // this is for the case when due to an overflow output is
            // 2**40 and C_ct is 2**40 - 1
            mod(diff, bin - scale);
            assert(diff <= 1);
            if(i < 10) printf("%lu %lu %lu\n", output, h_C_ct[i], diff);
            // if(output - h_C_ct[i] > 1) printf("%d: %lu %lu\n", i, output, h_C_ct[i]);
            // assert(abs(output - h_C_ct[i]) <= 1); //assert(h_C[i] - h_mask_C[i] == h_C_ct[i]);
        }
        // auto h_dI_ct = gpuConv2DWrapper(conv2d_layer.convKeydI, h_grad, h_F, NULL, 1, false);
        // for(int i = 0; i < conv2d_layer.p.size_I; i++) {
            // auto truncated_dI = /*h_dI_ct[i];*/cpuArs(h_dI_ct[i], bin, scale);
            // assert(h_masked_dI[i] - h_mask_dI[i] - truncated_dI <= 1);
        // }
        auto h_dF_ct = gpuConv2DWrapper(conv2d_layer.convKeydF, h_grad, h_I, NULL, 2, false);
        // auto vf_ct = new GPUGroupElement[conv2d_layer.p.size_F];
        for(int i = 0; i < conv2d_layer.p.size_F; i++) {
            auto vf = conv2d_layer.Vf[i] - h_mask_new_Vf[i];
            auto vf_ct = cpuArs((h_dF_ct[i] << mom_scale) + mom_fp * h_Vf[i], bin, mom_scale);
            assert(vf - vf_ct <= 1);
            auto new_f_ct = cpuArs((h_F[i] << (scale + lr_scale)) - lr_fp * vf_ct, bin, scale + lr_scale);
            // this is the new masked f
            auto new_f = conv2d_layer.F[i] - h_mask_new_F[i];
            // need to test this when the starting vf is non-zero
            assert(new_f - new_f_ct <= 2);
        }
        // incoming grad has scale s
        // Vb has scale 2s - lr
        // s - lr + mom_scale
        // output has scale 2s - lr + mom_scale, need to shift by mom_scale
        auto db_ct = new GPUGroupElement[CO];
        for(int i = 0; i < CO; i++) {
            GPUGroupElement db_ct = 0;
            for(int j = 0; j < conv2d_layer.p.size_O / CO; j++) {
                db_ct += h_grad[j * CO + i];
            }
            auto vb = conv2d_layer.Vb[i] - h_mask_new_Vb[i];
            auto vb_ct = cpuArs(((db_ct << (scale - lr_scale + mom_scale)) + mom_fp * h_Vb[i]), bin, mom_scale);
            assert(vb - vb_ct <= 1);
            auto new_b_ct = h_b[i] - lr_fp * vb_ct;
            // this is the new masked bias
            auto new_b = conv2d_layer.b[i] - h_mask_new_b[i];
            assert(new_b_ct - new_b <= 2);
            // printf("%d: %lu %lu\n", i, new_b, new_b_ct);
        }
    }
    return 0;
}