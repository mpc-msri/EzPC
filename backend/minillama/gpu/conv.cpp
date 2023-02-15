#include <iostream>
#include <chrono>
#include <cstdlib>
#include <chrono>


#include "fss.h"
#include "gpu_fss.h"
#include "gpu_stats.h"
#include "gpu_keygen.h"
#include "layers.h"

using namespace std;

int bitlength = 64; // = 64;
Peer *peer;
int party;



std::pair<GPUGroupElement*, GPUGroupElement*> getInputAndInputShare(int N, int party, Peer* peer) {
    auto h_I0 = /*CPUToGPUGroupElement(init_with_const(N, 64, 1), N);*/initRandom(N, 64);
    auto d_I = gpuReconstruct(h_I0, N, peer, party, NULL);
    auto h_I = (GPUGroupElement*) moveToCPU((uint8_t*) d_I, N * sizeof(GPUGroupElement), NULL);
    return std::make_pair(h_I, h_I0);
}

GPUGroupElement* getMaskedInput(int N, int party, Peer* peer, GPUGroupElement* mask_I) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto h_I = initRandom(N, 64);//CPUToGPUGroupElement(init_with_const(N, 64, 1), N);//initRandom(N, 64);
    // GPUGroupElement* h_I = new GPUGroupElement[N];
    // for(int i = 0; i < N; i++) h_I[i] &= ((1ULL << 40) - static_cast<GPUGroupElement>(1));
    if(party == 0) {
        for(int i = 0; i < N; i++) {
            h_I[i] += mask_I[i];
        }
    }
    auto d_I = gpuReconstruct(h_I, N, peer, party, NULL);
    return d_I;
}


int main(int argc, char *argv[])
{
    AESGlobalContext gaes;
    initGPUFSS(&gaes);

    party = atoi(argv[1]);
    bitlength = atoi(argv[2]);
    int bin = atoi(argv[2]);
    int bout = atoi(argv[3]);
    // int N = atoi(argv[4]);
    // int faithfulTruncations = atoi(argv[5]);
    // printf("%d %d %d %d\n", bin, bout, N, party);
    int N = 128, H = 32, W = 32, CI = 3, FH = 3, FW = 3, CO = 64, OH = 32, OW = 32;
    // int padding = 1, stride = 1;
    int size_I = getConv2DInputSize(N, H, W, CI);
    int size_F = getConv2DFilterSize(CI, FH, FW, CO);
    int size_O = getConv2DOutputSize(N, H, W, CI, FH, FW, CO, 1, 1, 1, 1, 1, 1);
    assert(size_O == OH * OW * N * CO);

    GPUGroupElement *I = new GPUGroupElement[size_I];  
    GPUGroupElement *F = new GPUGroupElement[size_F];
    GPUGroupElement *dI = new GPUGroupElement[size_I];  
    GPUGroupElement *dF = new GPUGroupElement[size_F];
    GPUGroupElement *C = new GPUGroupElement[size_O];  
    GPUGroupElement *incomingGrad = new GPUGroupElement[size_O];
    GPUGroupElement *mask_I, *mask_F, *mask_C, *mask_incomingGrad, *mask_dI, *mask_dF;

    if(party == 0) {
        std::ofstream f1("conv_key1.dat", ios::binary), f2("conv_key2.dat", ios::binary);
        GroupElement* mask_I_cpu = /*init_with_const(size_I, 64, 1);*/init_random(size_I, 64);
        GroupElement* mask_F_cpu = /*init_with_const(size_F, 64, 1);*/init_random(size_F, 64);
        // mask_F_cpu[0].value = 12;
        auto mask_C_cpu = gpuWriteKeyConv2D(f1, f2, bin, bout, N, H, W, CI, FH, FW, CO, mask_I_cpu, size_I, 
        mask_F_cpu, size_F, size_O);
        auto mask_incomingGrad_cpu = init_random(size_O, 64);
        auto mask_dI_cpu = /*init_with_const(size_I, 64, 1);*/init_random(size_I, 64);
        auto mask_dF_cpu = /*init_with_const(size_F, 64, 1);*/init_random(size_F, 64);
        gpuWriteKeyConv2DBackProp(f1, f2, bin, bout, N, H, W, CI, FH, FW, CO,
        mask_incomingGrad_cpu, size_O, mask_I_cpu, size_I, mask_F_cpu, size_F,
        mask_dI_cpu, mask_dF_cpu);
        printf("done\n");
        f1.close();
        f2.close();
        mask_I = CPUToGPUGroupElement(mask_I_cpu, size_I);
        mask_F = getFilter(CO, FH, FW, CI, mask_F_cpu, size_F);
        mask_C = CPUToGPUGroupElement(mask_C_cpu, size_O);
        mask_incomingGrad = CPUToGPUGroupElement(mask_incomingGrad_cpu, size_O);
        mask_dI = CPUToGPUGroupElement(mask_dI_cpu, size_I);
        mask_dF = CPUToGPUGroupElement(mask_dF_cpu, size_F);
    }
    Peer* peer = connectToPeer(party, argv[4]);
    size_t file_size;
    uint8_t* key_as_bytes = readFile("conv_key" + std::to_string(party+1) + ".dat", &file_size);
    printf("file size: %lu\n", file_size);
    Conv2DLayer conv_layer;
    conv_layer.init(&key_as_bytes);
    conv_layer.initBProp(&key_as_bytes);
    auto d_masked_I = getMaskedInput(size_I, party, peer, mask_I);
    auto h_masked_I = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_I, size_I * sizeof(GPUGroupElement), NULL);

    auto res = getInputAndInputShare(size_F, party, peer);
    F = res.first;
    conv_layer.F = res.second;
    auto d_O = conv_layer.forward(peer, party, d_masked_I);
    auto h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, size_O * sizeof(GPUGroupElement), NULL);
    if(party == 0) {
        for(int i = 0; i < size_I; i++) I[i] = h_masked_I[i] - mask_I[i];
        for(int i = 0; i < size_O; i++) C[i] = h_O[i] - mask_C[i];
        auto conv_o = gpuConv2DWrapper(conv_layer.key, I, F, NULL, 0);
        for(int i = 0; i < size_O; i++) 
        // printf("%lu %lu\n", conv_o[i], C[i]);
        assert(conv_o[i] == C[i]);
    }
    auto d_masked_grad = getMaskedInput(size_O, party, peer, mask_incomingGrad);
    auto h_masked_grad = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_grad, size_O * sizeof(GPUGroupElement), NULL);
    d_O = conv_layer.backward(peer, party, d_masked_grad);
    h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, size_I * sizeof(GPUGroupElement), NULL);
    auto d_reconstructedF = gpuReconstruct(conv_layer.F, size_F, peer, party, NULL);
    auto h_reconstructedF = (GPUGroupElement*) moveToCPU((uint8_t*) d_reconstructedF, size_F * sizeof(GPUGroupElement), NULL);
    if(party == 0) {
        for(int i = 0; i < size_O; i++) incomingGrad[i] = h_masked_grad[i] - mask_incomingGrad[i];
        for(int i = 0; i < size_I; i++) dI[i] = h_O[i] - mask_dI[i];
        auto di_o = gpuConv2DWrapper(conv_layer.keydI, incomingGrad, F, NULL, 1);
        for(int i = 0; i < size_I; i++) assert(di_o[i] == dI[i]);
        auto df_o = gpuConv2DWrapper(conv_layer.keydF, incomingGrad, I, NULL, 2);
        for(int i = 0; i < size_F; i++) assert(h_reconstructedF[i] == F[i] + mask_F[i] + df_o[i] + mask_dF[i]);
    }
    return 0;
}