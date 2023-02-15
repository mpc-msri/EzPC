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

struct VGG16OnCifar10 {
    Conv2DLayer conv_layer[13];
    RTLayer rt_layer[15];
    MatmulLayer matmul_layer[3];
    SoftMaxLayer softmax_layer;

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

void readConvRTLayer(VGG16OnCifar10* m, Peer* peer, int party, uint8_t** key_as_bytes/*std::istream& key_file*/, int i, AESGlobalContext* gaes) {
    m->conv_layer[i].init(key_as_bytes);
    m->rt_layer[i].init(key_as_bytes, gaes, false);
}

void readMatmulRTLayer(VGG16OnCifar10* m, Peer* peer, int party, /*std::istream& key_file*/uint8_t** key_as_bytes, int i, AESGlobalContext* gaes) {
    m->matmul_layer[i].init(/*key_file*/key_as_bytes);
    if(i < 2) {
        m->rt_layer[13 + i].init(key_as_bytes, gaes, false);
    }
}

VGG16OnCifar10 readKeys(Peer* peer, int party, AESGlobalContext* gaes) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t file_size;
    uint8_t* key_as_bytes = readFile("vgg16_cifar10_local_truncations_key" + std::to_string(party+1) + ".dat", &file_size);
    printf("start: %lu\n", key_as_bytes);
    VGG16OnCifar10 m;
    for(int i = 0; i < 13; i++) {
        readConvRTLayer(&m, peer, party, &key_as_bytes, i, gaes);
    }
    for(int i = 0; i < 3; i++) {
        readMatmulRTLayer(&m, peer, party, &key_as_bytes, i, gaes);
    }
    m.matmul_layer[2].initBProp(&key_as_bytes);
    
    m.rt_layer[14].initBProp(&key_as_bytes);
    m.matmul_layer[1].initBProp(&key_as_bytes);
    printf("14 done\n");
    
    m.rt_layer[13].initBProp(&key_as_bytes);
    m.matmul_layer[0].initBProp(&key_as_bytes);
    printf("13 done\n");

    m.rt_layer[12].initBProp(&key_as_bytes);
    m.conv_layer[12].initBProp(&key_as_bytes);
    printf("12 done\n");

    m.rt_layer[11].initBProp(&key_as_bytes);
    m.conv_layer[11].initBProp(&key_as_bytes);
    printf("11 done\n");

    m.rt_layer[10].initBProp(&key_as_bytes);
    m.conv_layer[10].initBProp(&key_as_bytes);
    printf("10 done\n");

    m.rt_layer[9].initBProp(&key_as_bytes);
    m.conv_layer[9].initBProp(&key_as_bytes);
    printf("9 done\n");
    printf("%lu\n", key_as_bytes);

    m.rt_layer[8].initBProp(&key_as_bytes);
    m.conv_layer[8].initBProp(&key_as_bytes);
    printf("8 done\n");
    printf("%lu\n", key_as_bytes);

    m.rt_layer[7].initBProp(&key_as_bytes);
    m.conv_layer[7].initBProp(&key_as_bytes);
    printf("7 done\n");

    m.rt_layer[6].initBProp(&key_as_bytes);
    m.conv_layer[6].initBProp(&key_as_bytes);
    printf("6 done\n");

    m.rt_layer[5].initBProp(&key_as_bytes);
    m.conv_layer[5].initBProp(&key_as_bytes);
    printf("5 done\n");

    m.rt_layer[4].initBProp(&key_as_bytes);
    m.conv_layer[4].initBProp(&key_as_bytes);
    printf("4 done\n");

    m.rt_layer[3].initBProp(&key_as_bytes);

    m.conv_layer[3].initBProp(&key_as_bytes);
    printf("3 done\n");

    m.rt_layer[2].initBProp(&key_as_bytes);
    m.conv_layer[2].initBProp(&key_as_bytes);
    printf("2 done\n");

    m.rt_layer[1].initBProp(&key_as_bytes);
    m.conv_layer[1].initBProp(&key_as_bytes);
    printf("1 done\n");

    m.rt_layer[0].initBProp(&key_as_bytes);
    m.conv_layer[0].initBProp(&key_as_bytes);
    printf("0 done\n");


    printf("%lu\n", key_as_bytes);
    // m.softmax_layer.init(10, 128, 26);
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
    
    d_O = m.matmul_layer[2].backward(peer, party, d_O);

    d_O = m.rt_layer[14].backward(peer, party, d_O);
    d_O = m.matmul_layer[1].backward(peer, party, d_O);

    d_O = m.rt_layer[13].backward(peer, party, d_O);
    d_O = m.matmul_layer[0].backward(peer, party, d_O);

    d_O = m.rt_layer[12].backward(peer, party, d_O);
    d_O = gpuAddPoolBackProp(d_O, m.conv_layer[12].key);
    d_O = m.conv_layer[12].backward(peer, party, d_O);

    printf("layer 11\n");
    d_O = m.rt_layer[11].backward(peer, party, d_O);
    d_O = m.conv_layer[11].backward(peer, party, d_O);

    printf("layer 10\n");
    d_O = m.rt_layer[10].backward(peer, party, d_O);
    d_O = m.conv_layer[10].backward(peer, party, d_O);

    printf("layer 9\n");
    d_O = m.rt_layer[9].backward(peer, party, d_O);
    d_O = gpuAddPoolBackProp(d_O, m.conv_layer[9].key);
    d_O = m.conv_layer[9].backward(peer, party, d_O);

    printf("layer 8\n");
    d_O = m.rt_layer[8].backward(peer, party, d_O);
    d_O = m.conv_layer[8].backward(peer, party, d_O);

    printf("layer 7\n");
    d_O = m.rt_layer[7].backward(peer, party, d_O);
    d_O = m.conv_layer[7].backward(peer, party, d_O);

    printf("layer 6\n");
    d_O = m.rt_layer[6].backward(peer, party, d_O);
    d_O = gpuAddPoolBackProp(d_O, m.conv_layer[6].key);
    d_O = m.conv_layer[6].backward(peer, party, d_O);

    printf("layer 5\n");
    d_O = m.rt_layer[5].backward(peer, party, d_O);
    d_O = m.conv_layer[5].backward(peer, party, d_O);

    printf("layer 4\n");
    d_O = m.rt_layer[4].backward(peer, party, d_O);
    d_O = m.conv_layer[4].backward(peer, party, d_O);

    printf("layer 3\n");
    d_O = m.rt_layer[3].backward(peer, party, d_O);
    d_O = gpuAddPoolBackProp(d_O, m.conv_layer[3].key);
    d_O = m.conv_layer[3].backward(peer, party, d_O);

    printf("layer 2\n");
    d_O = m.rt_layer[2].backward(peer, party, d_O);
    d_O = m.conv_layer[2].backward(peer, party, d_O);

    printf("layer 1\n");
    d_O = m.rt_layer[1].backward(peer, party, d_O);
    d_O = gpuAddPoolBackProp(d_O, m.conv_layer[1].key);
    d_O = m.conv_layer[1].backward(peer, party, d_O);

    printf("layer 0\n");
    d_O = m.rt_layer[0].backward(peer, party, d_O);
    d_O = m.conv_layer[0].backward(peer, party, d_O);
    printf("comm time: %lu\n", m.conv_layer[0].s.comm_time);

    // auto h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, m.matmul_layer[2].key.mem_size_C, NULL);
    // m.softmax_layer.backward(peer, party, h_O);
    freeAESGlobalContext(&gaes);
    // d_O = m.matmul_layer[2].backward(peer, party, d_O);
    // d_O = m.rt_layer[14].backward(peer, party, d_O);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - compute_start;
    std::cout << "Total time in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    if(party == 0) {
    //     auto conv_i = m.i;
    //     for(int i = 0; i < 13; i++) {
    //         printf("Computing layer %d\n", i+1);
    //         auto conv_o = gpuConv2DWrapper(m.conv_layer[i].key, conv_i, m.f[i]);
    //         GPUGroupElement* rt_o;
    //         if(i == 1 || i == 3 || i == 6 || i == 9 || i == 12)
    //         {
    //             auto addpool_o = addPoolWrapper(conv_o, m.conv_layer[i].key);
    //             rt_o = plaintextRTWRapper(addpool_o, m.rt_layer[i].key.num_rts, 64, 26);
    //         } 
    //         else rt_o = plaintextRTWRapper(conv_o, m.rt_layer[i].key.num_rts, 64, 24);
    //         conv_i = rt_o;
    //     }
    //     auto matmul_i = conv_i;
    //     GPUGroupElement* matmul_o;
    //     for(int i = 0; i < 3; i++) {
    //         printf("Computing layer %d\n", i+14);
    //         matmul_o = gpuMatmulWrapper(m.matmul_layer[i].key, matmul_i, m.b[i]);
    //         if(i < 2) {
    //             auto rt_o = plaintextRTWRapper(matmul_o, m.rt_layer[13+i].key.num_rts, 64, 24);
    //             matmul_i = rt_o;
    //         }   
    //     }
    //     size_t rout_size;
    //     auto rout = (GPUGroupElement*) readFile("vgg16_cifar10_rout.dat", &rout_size);
    //     assert(rout_size == m.matmul_layer[2].key.mem_size_C);
    //     checkOutput(matmul_o, h_O, rout, m.matmul_layer[2].key.size_C);
    }
    return 0;
}