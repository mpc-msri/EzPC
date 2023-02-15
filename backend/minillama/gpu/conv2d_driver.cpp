#include <iostream>
#include <chrono>
#include <cstdlib>
#include <chrono>


#include "fss.h"
#include "gpu_fss.h"
#include "gpu_stats.h"

using namespace std;

int bitlength = 64; // = 64;
Peer *peer;
int party;


GPUGroupElement* cpuAddShares(GPUGroupElement* h_A, GPUGroupElement* h_B, int N, Stats* s) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto d_A = (GPUGroupElement*) moveToGPU((uint8_t*) h_A, size_in_bytes, s);
    auto d_B = (GPUGroupElement*) moveToGPU((uint8_t*) h_B, size_in_bytes, s);
    auto d_C = gpuAddShares(d_A, d_B, N, s);
    auto h_C = (GPUGroupElement*) moveToCPU((uint8_t*) d_C, size_in_bytes, s);
    gpuFree(d_A);
    gpuFree(d_B);
    gpuFree(d_C);
    return h_C;
}

struct VGG16OnCifar10 {
    GPUConv2DKey conv_key[13];
    GPUReLUTruncateKey rt_key[13];

    GPUGroupElement* i;
    GPUGroupElement* f[13];

    GPUGroupElement* i0;
    GPUGroupElement* f0[13];

    Stats conv_stats[13];
    Stats rt_stats[13];
};

GPUGroupElement* randomInput(int N, int party, Peer* peer, GPUGroupElement** i0) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    *i0 = initRandom(N);
    auto d_i = gpuReconstruct(*i0, N, peer, party, NULL);
    auto i = (GPUGroupElement*) moveToCPU((uint8_t*) d_i, size_in_bytes, NULL);
    gpuFree(d_i);
    return i;
}

void prepareInput(VGG16OnCifar10* m, Peer* peer, int party) {
    // might overflow an int
    int N;
    N = m->conv_key[0].size_I;
    m->i = randomInput(N, party, peer, &m->i0);
    for(int i = 0; i < 13; i++) {
        N = m->conv_key[i].size_F;
        m->f[i] = randomInput(N, party, peer, &m->f0[i]);
    }
    printf("%lu %lu\n", m->i[0], m->f[0][0]);
}
void reconstructInput(VGG16OnCifar10* m, Peer* peer, int party, int i, Stats* s) {
    int N;
    size_t size_in_bytes;
    if(i == 0) {
        N = m->conv_key[i].size_I;
        size_in_bytes = m->conv_key[i].mem_size_I;
        auto masked_i0 = cpuAddShares(m->i0, m->conv_key[i].I, N, s);
        cpuFree(m->i0);
        m->i0 = gpuReconstruct(masked_i0, N, peer, party, s);
        cpuFree(masked_i0);
    }
    N = m->conv_key[i].size_F;
    size_in_bytes = m->conv_key[i].mem_size_F;
    auto masked_f0 = cpuAddShares(m->f0[i], m->conv_key[i].F, N, s);
    cpuFree(m->f0[i]);
    m->f0[i] = gpuReconstruct(masked_f0, N, peer, party, s);
    cpuFree(masked_f0);
}


void readConvRTLayer(VGG16OnCifar10* m, Peer* peer, int party, std::istream& key_file, int i) {
    m->conv_key[i] = readGPUConv2DKey(key_file);
    m->rt_key[i] = readGPUReLUTruncateKey(key_file);
}

VGG16OnCifar10 readKeys(Peer* peer, int party) {
    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream key_file("vgg16_cifar10_key" + std::to_string(party+1) + ".dat");
    VGG16OnCifar10 m;
    // memset(&m.conv_stats, 0, 13 * sizeof(Stats));
    // memset(&m.rt_stats, 0, 13 * sizeof(Stats));
    for(int i = 0; i < 13; i++) {
        m.conv_stats[i].transfer_time = 0;
        m.conv_stats[i].comm_time = 0;

        m.rt_stats[i].transfer_time = 0;
        m.rt_stats[i].comm_time = 0;

        readConvRTLayer(&m, peer, party, key_file, i);
    }
    key_file.close();
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
    Peer* peer = connectToPeer(party);
    auto m = readKeys(peer, party);
    prepareInput(&m, peer, party);
    printf("prepped input\n");
    GPUGroupElement* d_O;
    auto compute_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 13; i++) {
        printf("Layer %d:\n", i+1);
        auto start = std::chrono::high_resolution_clock::now();
        reconstructInput(&m, peer, party, i, &m.conv_stats[i]);
        // readConvRTLayer(&m, peer, party, key, i);
        // need to fix this later by populating device input in d_O elsewhere
        if(i == 0) d_O = m.i0;
        // auto start = std::chrono::high_resolution_clock::now();
        d_O = GPUConv2D(peer, m.conv_key[i], d_O, m.f0[i], party, &m.conv_stats[i]);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "Time for computing Conv2D in ms*: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

        start = end;//std::chrono::high_resolution_clock::now();
        d_O = GPUReluTruncate(peer, m.rt_key[i], d_O, party, &gaes, &m.rt_stats[i]);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Time for computing RT in ms*: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    
        if(i == 1 || i == 3 || i == 6 || i == 9 || i == 12)
            d_O = gpuAddPool(d_O, m.conv_key[i]);
    }
    auto h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, m.conv_key[12].mem_size_O / 4, NULL);

    freeAESGlobalContext(&gaes);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - compute_start;
    std::cout << "Total time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    for(int i = 0; i < 13; i++) {
        printf("%d: %lu %lu\n", i, m.conv_stats[i].comm_time, m.conv_stats[i].transfer_time);
        printf("%d: %lu %lu\n", i, m.rt_stats[i].comm_time, m.rt_stats[i].transfer_time);
    }
    if(party == 0) {
        auto conv_i = m.i;
        for(int i = 0; i < 13; i++) {
            printf("Computing layer %d\n", i+1);
            auto conv_o = gpuConv2DWrapper(m.conv_key[i], conv_i, m.f[i]);
            auto rt_o = plaintextRTWRapper(conv_o, m.conv_key[i].size_O, 64, 24);
            conv_i = rt_o;
            if(i == 1 || i == 3 || i == 6 || i == 9 || i == 12)
            {
                auto addPool_o = addPoolWrapper(rt_o, m.conv_key[i]);
                conv_i = addPool_o;
            } 
        }
        size_t rout_size;
        auto rout = (GPUGroupElement*) readFile("vgg16_cifar10_rout.dat", &rout_size);
        assert(rout_size == m.conv_key[12].mem_size_O / 4);
        checkOutput(conv_i, h_O, rout, m.conv_key[12].size_O / 4);
    }
    return 0;
}