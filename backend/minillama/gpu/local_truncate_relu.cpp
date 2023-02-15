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


GPUGroupElement* getMaskedInput(int N, int party, Peer* peer, GPUGroupElement* mask_I) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto h_I = initRandom(N, 64);
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
    int N = atoi(argv[4]);
    int faithfulTruncations = atoi(argv[5]);
    printf("%d %d %d %d\n", bin, bout, N, party);

    GPUGroupElement *mask_I, *mask_RT, *mask_dRelu, *mask_incomingGrad, *mask_gradRelu;

    if(party == 0) {
        std::ofstream f1("local_truncate_relu_key1.dat", ios::binary), f2("local_truncate_relu_key2.dat", ios::binary);
        GroupElement* mask_I_cpu = init_random(N, 64);
        auto res = gpuWriteKeyLocalTruncateRelu(f1, f2, bin, bout, 24, mask_I_cpu, N);
        auto mask_RT_cpu = res.first;
        auto mask_dRelu_cpu = res.second;
        auto mask_incomingGrad_cpu = init_random(N, 64);
        auto mask_gradRelu_cpu = gpuWriteKeyLocalTruncateReluBackProp(f1, f2, bin, bout, 24, mask_incomingGrad_cpu, mask_dRelu_cpu, N);
        f1.close();
        f2.close();
        mask_I = CPUToGPUGroupElement(mask_I_cpu, N);
        mask_RT = CPUToGPUGroupElement(mask_RT_cpu, N);
        mask_dRelu = CPUToGPUGroupElement(mask_dRelu_cpu, N);
        mask_incomingGrad = CPUToGPUGroupElement(mask_incomingGrad_cpu, N);
        mask_gradRelu = CPUToGPUGroupElement(mask_gradRelu_cpu, N);
    }
    Peer* peer = connectToPeer(party, argv[6]);
    size_t file_size;
    uint8_t* key_as_bytes = readFile("local_truncate_relu_key" + std::to_string(party+1) + ".dat", &file_size);
    printf("file size: %lu\n", file_size);
    RTLayer rt_layer;
    printf("%d\n", faithfulTruncations);
    rt_layer.init(&key_as_bytes, &gaes, faithfulTruncations);
    rt_layer.initBProp(&key_as_bytes);
    auto d_masked_I = getMaskedInput(N, party, peer, mask_I);
    auto h_masked_I = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_I, N * sizeof(GPUGroupElement), NULL);
    // auto compute_start = std::chrono::high_resolution_clock::now();
    auto d_O = rt_layer.forward(peer, party, d_masked_I);
    auto h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, N * sizeof(GPUGroupElement), NULL);
    // freeAESGlobalContext(&gaes);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - compute_start;
    // std::cout << "Total time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    if(party == 0) {
        int count = 0;
        for(int i = 0; i < N; i++) {
            auto unmasked_o = h_O[i] - mask_RT[i];
            auto shifted_I = (h_masked_I[i] - mask_I[i]) >> 24;
            int dReLU = ((h_masked_I[i] - mask_I[i]) < (1ULL << (bin - 1)));
            // printf("%d: %lu %d %lu\n", i, unmasked_o, dReLU, shifted_I);
            int64_t tol = faithfulTruncations ? 0 : 1;
            if(abs(static_cast<int64_t>(dReLU * shifted_I - unmasked_o)) > tol) count++;
            // assert(unmasked_o == shifted_I * (shifted_I < (1ULL << (bin - 1))));
        }
        printf("num errors: %d\n", count);
    }
    auto d_masked_grad = getMaskedInput(N, party, peer, mask_incomingGrad);
    auto h_masked_grad = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_grad, N * sizeof(GPUGroupElement), NULL);
    // auto compute_start = std::chrono::high_resolution_clock::now();
    d_O = rt_layer.backward(peer, party, d_masked_grad);
    h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, N * sizeof(GPUGroupElement), NULL);
    freeAESGlobalContext(&gaes);
    if(party == 0) {
        int count = 0;
        for(int i = 0; i < N; i++) {
            auto unmasked_o = h_O[i] - mask_gradRelu[i];
            auto shifted_grad = (h_masked_grad[i] - mask_incomingGrad[i]) >> 24;
            int dReLU = ((h_masked_I[i] - mask_I[i]) < (1ULL << (bin - 1)));
            // printf("%d: %lu %d %lu\n", i, unmasked_o, dReLU, shifted_grad);
            int64_t tol = faithfulTruncations ? 0 : 1;
            if(abs(static_cast<int64_t>(dReLU * shifted_grad - unmasked_o)) > tol) count++;
            // assert(unmasked_o == shifted_I * (shifted_I < (1ULL << (bin - 1))));
        }
        printf("num errors: %d\n", count);
    }
    return 0;
}