#include <chrono>

#include "fss.h"
#include "gpu_data_types.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include "gpu_stats.h"
#include "gpu_fss.h"



// extern "C" GPUGroupElement *gpu_conv2d(GPUConv2DKey k,
//                                        int party, GPUGroupElement *,
//                                        GPUGroupElement *, Stats*);

extern "C" GPUGroupElement *gpu_conv2d(GPUConv2DKey k, int party, GPUGroupElement *d_I, GPUGroupElement *d_F, GPUGroupElement* d_a, GPUGroupElement* d_b, Stats* s, char op);


extern "C" GPUGroupElement *gpu_matmul(GPUMatmulKey k,
                                       int party, GPUGroupElement *,
                                       GPUGroupElement *, bool, bool, bool, Stats*);

extern "C" void gpu_relu_truncate(GPUReLUTruncateKey k,
                                  int party, GPUGroupElement *d_in,
                                  GPURTContext *rtc, AESGlobalContext* gaes, Stats*);

extern "C" void gpu_local_truncate_relu(GPUReLUTruncateKey k,
                                  int party, GPUGroupElement *d_in,
                                  GPURTContext *rtc, AESGlobalContext* gaes, Stats* stats);

                                  
extern "C" std::pair<GPUGroupElement*, uint32_t*> finish_relu_truncate(GPUReLUTruncateKey k,
                                                 uint32_t *d_x1, GPUGroupElement *d_y1, /*GPUGroupElement *d_a,*/
                                                 uint32_t *h_x2, GPUGroupElement *h_y2, int party, bool, bool, Stats*);


extern "C" GPUGroupElement* gpuAddShares(GPUGroupElement* d_A, GPUGroupElement* d_B, int N, Stats*);



void initGPUFSS(AESGlobalContext* g) {
    fss_init();

    AES aesSeed(toBlock(0, time(NULL)));
    auto commonSeed = aesSeed.ecbEncBlock(ZeroBlock);
    prngShared.SetSeed(commonSeed);

    initAESContext(g);
}




GPUGroupElement* initRandom(int N, int bitlength) {
    GPUGroupElement* A = (GPUGroupElement*) cpuMalloc(N * sizeof(GPUGroupElement));
    for (int i = 0; i < N; i++)
        A[i] = random_ge(bitlength).value;
    return A;
}

// GPUGroupElement* reconstructMasked(GPUGroupElement* h_A0, GPUGroupElement* h_mask_A0, int N, Peer* peer, int party, Stats* s) {
//     size_t size_in_bytes = N * sizeof(GPUGroupElement);
//     GPUGroupElement* d_A0 = (GPUGroupElement*) moveToGPU((uint8_t*) h_A0, size_in_bytes, s);
//     GPUGroupElement* d_mask_A0 = (GPUGroupElement*) moveToGPU((uint8_t*) h_mask_A0, size_in_bytes, s);
//     GPUGroupElement* d_masked_A0 = gpuAddShares(d_A0, d_mask_A0, N, s);
//     // gpuFree(d_A0);
//     // gpuFree(d_mask_A0);
    
//     GPUGroupElement* h_masked_A0 = (GPUGroupElement*) moveToCPU((uint8_t*)d_masked_A0, size_in_bytes, s);
    
//     auto h_masked_A1 = (GPUGroupElement*) exchangeShares(peer, (uint8_t*) h_masked_A0, size_in_bytes, party, s);
//     cpuFree(h_masked_A0);
//     auto d_masked_A1 = (GPUGroupElement*) moveToGPU((uint8_t*)h_masked_A1, size_in_bytes, s);
//     cpuFree(h_masked_A1);

//     auto d_masked_A = gpuAddShares(d_masked_A0, d_masked_A1, N, s);    
//     return d_masked_A;
// }

GPUGroupElement* gpuReconstruct(GPUGroupElement* h_A0, int N, Peer* peer, int party, Stats* s) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);
    auto h_A1 = (GPUGroupElement*) exchangeShares(peer, (uint8_t*) h_A0, size_in_bytes, party, s);
    
    auto d_A0 = (GPUGroupElement*) moveToGPU((uint8_t*)h_A0, size_in_bytes, s);
    auto d_A1 = (GPUGroupElement*) moveToGPU((uint8_t*)h_A1, size_in_bytes, s);
    
    cpuFree(h_A1);
    auto d_A = gpuAddShares(d_A0, d_A1, N, s);    
    gpuFree(d_A0);
    gpuFree(d_A1);
    return d_A;
}

// GPUGroupElement* GPUConv2D(Peer* peer, GPUConv2DKey k, GPUGroupElement* d_I, GPUGroupElement* d_F, int party, Stats* s) {
//     auto d_C0 = gpu_conv2d(k, party, d_I, d_F, s);
//     auto h_C0 = moveToCPU((uint8_t*) d_C0, k.mem_size_O, s);
//     auto h_C1 = exchangeShares(peer, (uint8_t*) h_C0, k.mem_size_O, party, s);
//     cpuFree(h_C0);
//     auto d_C1 = (GPUGroupElement*) moveToGPU(h_C1, k.mem_size_O, s);
//     cpuFree(h_C1);
//     auto d_C = gpuAddShares(d_C0, d_C1, k.size_O, s); 
//     gpuFree(d_C0);
//     gpuFree(d_C1);
//     return d_C;
// }

// GPUGroupElement* GPUMatmul(Peer* peer, GPUMatmulKey k, GPUGroupElement* d_A, GPUGroupElement* d_B, int party, Stats* s) {
//     auto d_C0 = gpu_matmul(k, party, d_A, d_B, s);
//     auto h_C0 = moveToCPU((uint8_t*) d_C0, k.mem_size_C, s);
//     auto h_C1 = exchangeShares(peer, (uint8_t*) h_C0, k.mem_size_C, party, s);
//     cpuFree(h_C0);
//     auto d_C1 = (GPUGroupElement*) moveToGPU(h_C1, k.mem_size_C, s);
//     cpuFree(h_C1);
//     auto d_C = gpuAddShares(d_C0, d_C1, k.size_C, s); 
//     gpuFree(d_C0);
//     gpuFree(d_C1);
//     return d_C;
// }

GPUGroupElement* GPUReluTruncate(Peer* peer, GPUReLUTruncateKey k, GPUGroupElement* d_I, int party, AESGlobalContext* g, Stats* s) {
    
    // GPURTContext c;
    // // auto start = std::chrono::high_resolution_clock::now();
    // gpu_relu_truncate(k, party, d_I, &c, g, s);
    // size_t lrs_mem_size = k.num_rts * sizeof(GPUGroupElement);
    // size_t drelu_mem_size = (k.num_rts - 1) / 8 + 1;

    // auto h_lrs0 = moveToCPU((uint8_t*) c.d_lrs0, lrs_mem_size, s);
    // auto h_drelu0 = moveToCPU((uint8_t*) c.d_drelu0, drelu_mem_size, s);

    // auto h_lrs1 = exchangeShares(peer, (uint8_t*) h_lrs0, lrs_mem_size, party, s);
    // auto h_drelu1 = exchangeShares(peer, (uint8_t*) h_drelu0, drelu_mem_size, party, s);
    
    // auto d_lrs1 = (GPUGroupElement*) moveToGPU(h_lrs1, lrs_mem_size, s);
    // auto d_drelu1 = (uint32_t*) moveToGPU(h_drelu1, drelu_mem_size, s);

    // cpuFree(h_lrs1);
    // cpuFree(h_drelu1);
    // auto d_res0 = finish_relu_truncate(k, c.d_drelu0, c.d_lrs0, /*c.d_a,*/ d_drelu1, d_lrs1, party, s);
    // auto h_res0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_res0, lrs_mem_size, s);
    // auto d_res = gpuReconstruct(h_res0, k.num_rts, peer, party, s);
    // return d_res;
}



// GPUGroupElement* GPULocalTruncateRelu(Peer* peer, int bin, GPUReLUTruncateKey k, GPUGroupElement* d_I, int party, AESGlobalContext* g, Stats* s) {
    
//     GPURTContext c;
//     gpu_local_truncate_relu(k, party, bin, d_I, &c, g, s);

//     size_t lrs_mem_size = k.num_rts * sizeof(GPUGroupElement);
//     size_t drelu_mem_size = ((k.num_rts - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);

//     auto h_drelu0 = moveToCPU((uint8_t*) c.d_drelu0, drelu_mem_size, s);
//     auto h_drelu1 = exchangeShares(peer, (uint8_t*) h_drelu0, drelu_mem_size, party, s);
//     auto d_drelu1 = (uint32_t*) moveToGPU(h_drelu1, drelu_mem_size, s);
//     cpuFree(h_drelu1);

//     auto d_res0 = finish_relu_truncate(k, c.d_drelu0, c.d_lrs0, /*c.d_a,*/ d_drelu1, NULL, party, s);
//     auto h_res0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_res0, lrs_mem_size, s);

//     auto d_res = gpuReconstruct(h_res0, k.num_rts, peer, party, s);
//     return d_res;
// }

