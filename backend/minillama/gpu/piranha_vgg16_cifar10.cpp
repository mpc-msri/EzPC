#include <iostream>
#include <chrono>
#include <cstdlib>
#include <chrono>

#include "fss.h"
#include "gpu_fss.h"
#include "gpu_stats.h"
#include "layers.h"

using namespace std;

int bitlength = 64; // = 64;
Peer *peer;
int party;


GPUGroupElement* cpuAddShares(GPUGroupElement* h_A, GPUGroupElement* h_B, int N, Stats* s) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, size_in_bytes, s);
    auto d_B = (GPUGroupElement*) moveToGPU((uint8_t*) h_B, size_in_bytes, s);
    auto d_C = gpuAddShares(d_A, d_B, N, s);
    gpuFree(d_A);
    gpuFree(d_B);
    auto h_C = (GPUGroupElement*) moveToCPU((uint8_t*) d_C, size_in_bytes, s);
    // gpuFree(d_A);
    // gpuFree(d_B);
    gpuFree(d_C);
    return h_C;
}

struct VGG16OnCifar10 {
    Conv2DLayer conv_layer[13];
    RTLayer rt_layer[15];
    MatmulLayer matmul_layer[3];

    GPUGroupElement* i;
    GPUGroupElement* f[13];
    GPUGroupElement* b[3];
};

GPUGroupElement* randomInput(int N, int party, Peer* peer, GPUGroupElement** i0) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    *i0 = initRandom(N, 64);
    auto d_i = gpuReconstruct(*i0, N, peer, party, NULL);
    auto i = (GPUGroupElement*) moveToCPU((uint8_t*) d_i, size_in_bytes, NULL);
    gpuFree(d_i);
    return i;
}

void prepareInput(VGG16OnCifar10* m, Peer* peer, int party) {
    // might overflow an int
    int N;
    N = m->conv_layer[0].key.size_I;
    m->i = randomInput(N, party, peer, &m->conv_layer[0].I);
    for(int i = 0; i < 13; i++) {
        N = m->conv_layer[i].key.size_F;
        m->f[i] = randomInput(N, party, peer, &m->conv_layer[i].F);
    }
    for(int i = 0; i < 3; i++) {
        N = m->matmul_layer[i].key.size_B;
        m->b[i] = randomInput(N, party, peer, &m->matmul_layer[i].B);
    }
}

GPUGroupElement* reconstructInput(Peer* peer, int party, int N, GPUGroupElement* share_x, GPUGroupElement* share_mask_x, Stats* s) {
    auto share_masked_x = cpuAddShares(share_x, share_mask_x, N, s);
    cpuFree(share_x);
    auto masked_x = gpuReconstruct(share_masked_x, N, peer, party, s);
    cpuFree(share_masked_x);
    return masked_x;
}


void readConvRTLayer(VGG16OnCifar10* m, Peer* peer, int party, uint8_t** key_as_bytes/*std::istream& key_file*/, int i, AESGlobalContext* gaes) {
    m->conv_layer[i].init(key_as_bytes);
    m->rt_layer[i].init(key_as_bytes, gaes, true);
}

void readMatmulRTLayer(VGG16OnCifar10* m, Peer* peer, int party, /*std::istream& key_file*/uint8_t** key_as_bytes, int i, AESGlobalContext* gaes) {
    m->matmul_layer[i].init(/*key_file*/key_as_bytes);
    if(i < 2) {
        m->rt_layer[13 + i].init(key_as_bytes, gaes, true);
    }
}

VGG16OnCifar10 readKeys(Peer* peer, int party, AESGlobalContext* gaes) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t file_size;
    uint8_t* key_as_bytes = readFile("vgg16_cifar10_key" + std::to_string(party+1) + ".dat", &file_size);
    VGG16OnCifar10 m;
    for(int i = 0; i < 13; i++) {
        readConvRTLayer(&m, peer, party, &key_as_bytes, i, gaes);
    }
    for(int i = 0; i < 3; i++) {
        readMatmulRTLayer(&m, peer, party, &key_as_bytes, i, gaes);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time for reading keys in ms*: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    return m;
}


int main(int argc, char *argv[])
{
    AESGlobalContext gaes;
    initGPUFSS(&gaes);
    party = atoi(argv[1]);
    Peer* peer = connectToPeer(party, /*"172.31.45.173"*/ "172.31.45.158");
    auto m = readKeys(peer, party, &gaes);
    prepareInput(&m, peer, party);
    GPUGroupElement* d_O = NULL;
    auto compute_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 13; i++) {
        printf("Layer %d:\n", i+1);
        d_O = m.conv_layer[i].forward(peer, party, d_O);
        if(i == 1 || i == 3 || i == 6 || i == 9 || i == 12)
            d_O = gpuAddPool(d_O, m.conv_layer[i].key);
        d_O = m.rt_layer[i].forward(peer, party, d_O);    
    }
    for(int i = 0; i < 3; i++) {
        printf("Layer %d:\n", i+14);
        d_O = m.matmul_layer[i].forward(peer, party, d_O);
        if(i < 2) {
            d_O = m.rt_layer[13 + i].forward(peer, party, d_O);
        }
    }

    auto h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, m.matmul_layer[2].key.mem_size_C, NULL);

    freeAESGlobalContext(&gaes);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - compute_start;
    std::cout << "Total time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    if(party == 0) {
        auto conv_i = m.i;
        for(int i = 0; i < 13; i++) {
            printf("Computing layer %d\n", i+1);
            auto conv_o = gpuConv2DWrapper(m.conv_layer[i].key, conv_i, m.f[i], NULL, 0);
            GPUGroupElement* rt_o;
            if(i == 1 || i == 3 || i == 6 || i == 9 || i == 12)
            {
                auto addpool_o = addPoolWrapper(conv_o, m.conv_layer[i].key);
                rt_o = plaintextRTWRapper(addpool_o, m.rt_layer[i].key.num_rts, 64, 26);
            } 
            else rt_o = plaintextRTWRapper(conv_o, m.rt_layer[i].key.num_rts, 64, 24);
            conv_i = rt_o;
        }
        auto matmul_i = conv_i;
        GPUGroupElement* matmul_o;
        for(int i = 0; i < 3; i++) {
            printf("Computing layer %d\n", i+14);
            matmul_o = gpuMatmulWrapper(m.matmul_layer[i].key, matmul_i, m.b[i], NULL, true, true, true);
            if(i < 2) {
                auto rt_o = plaintextRTWRapper(matmul_o, m.rt_layer[13+i].key.num_rts, 64, 24);
                matmul_i = rt_o;
            }   
        }
        size_t rout_size;
        auto rout = (GPUGroupElement*) readFile("vgg16_cifar10_rout.dat", &rout_size);
        assert(rout_size == m.matmul_layer[2].key.mem_size_C);
        checkOutput(matmul_o, h_O, rout, m.matmul_layer[2].key.size_C);
    }
    return 0;
}