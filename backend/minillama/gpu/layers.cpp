#include "gpu_data_types.h"
// #include "gpu_stats.h"
#include "layers.h"
#include "gpu_fss.h"
#include "fss.h"
// #include "piranha_softmax.h"


extern "C" GPUGroupElement *gpu_matmul(GPUMatmulKey k,
                                       int party, GPUGroupElement *,
                                       GPUGroupElement *, GPUGroupElement *,  
                                       GPUGroupElement *, bool, bool, bool, Stats*);

extern "C" GPUGroupElement *gpu_conv2d(GPUConv2DKey k, int party, 
GPUGroupElement *d_I, GPUGroupElement *d_F, GPUGroupElement* d_a, GPUGroupElement* d_b, Stats* s, char op);


extern "C" void gpu_relu_truncate(GPUReLUTruncateKey k,
                                  int party, GPUGroupElement *d_in,
                                  GPURTContext *rtc, AESGlobalContext* gaes, Stats*);

extern "C" void gpu_local_truncate_relu(GPUReLUTruncateKey k,
                                  int party, /*int bin,*/ GPUGroupElement *d_in,
                                  GPURTContext *rtc, AESGlobalContext* gaes, Stats* stats);

extern "C" void gpu_local_truncate(GPUReLUTruncateKey k, GPUGroupElement *d_in,
                                  Stats* stats);

extern "C" std::pair<GPUGroupElement*, GPUGroupElement*> finish_relu_truncate(GPUReLUTruncateKey k,
                                                 uint32_t *d_x1, GPUGroupElement *d_y1, /*GPUGroupElement *d_a,*/
                                                 uint32_t *h_x2, GPUGroupElement *h_y2, int party, bool, bool, Stats*);


void Conv2DLayer::init(uint8_t** key_as_bytes) {
        key = readGPUConv2DKey(key_as_bytes);
        s.comm_time = 0;
        s.transfer_time = 0;
    }

void Conv2DLayer::initBProp(uint8_t** key_as_bytes) {

        GPUGroupElement* mask_grad = (GPUGroupElement*) *key_as_bytes;
        *key_as_bytes += key.mem_size_O;
        GPUGroupElement* mask_dI = (GPUGroupElement*) *key_as_bytes;
        *key_as_bytes += key.mem_size_I;
        GPUGroupElement* mask_dF = (GPUGroupElement*) *key_as_bytes;
        *key_as_bytes += key.mem_size_F;

        keydI = key;
        // grad * F
        keydI.I = mask_grad;
        keydI.F = key.F;
        keydI.O = mask_dI;

        keydI.size_I = key.size_O;
        keydI.mem_size_I = key.mem_size_O;

        keydI.size_F = key.size_F;
        keydI.mem_size_F = key.mem_size_F;

        keydI.size_O = key.size_I;
        keydI.mem_size_O = key.mem_size_I;
        // need to read the gradient mask
        // need to read the output mask

        // grad * input
        keydF = key;
        keydF.I = mask_grad;
        keydF.F = key.I;
        keydF.O = mask_dF;
        
        keydF.size_I = key.size_O;
        keydF.mem_size_I = key.mem_size_O;

        keydF.size_F = key.size_I;
        keydF.mem_size_F = key.mem_size_I;

        keydF.size_O = key.size_F;
        keydF.mem_size_O = key.mem_size_F;

        // need to read the output mask

        // s.comm_time = 0;
        // s.transfer_time = 0;
    }

std::pair<GPUGroupElement*, GPUGroupElement*> reconstructMasked(
    GPUGroupElement* h_A0, GPUGroupElement* h_mask_A0, int N, 
    Peer* peer, int party, Stats* s) {
    size_t size_in_bytes = N * sizeof(GPUGroupElement);

    GPUGroupElement* d_A0 = (GPUGroupElement*) moveToGPU((uint8_t*) h_A0, size_in_bytes, s);
    GPUGroupElement* d_mask_A0 = (GPUGroupElement*) moveToGPU((uint8_t*) h_mask_A0, size_in_bytes, s);
    GPUGroupElement* d_masked_A0 = gpuAddShares(d_A0, d_mask_A0, N, s);
    gpuFree(d_A0);
    
    GPUGroupElement* h_masked_A0 = (GPUGroupElement*) moveToCPU((uint8_t*)d_masked_A0, size_in_bytes, s);
    
    auto h_masked_A1 = (GPUGroupElement*) exchangeShares(peer, (uint8_t*) h_masked_A0, size_in_bytes, party, s);
    cpuFree(h_masked_A0);
    auto d_masked_A1 = (GPUGroupElement*) moveToGPU((uint8_t*)h_masked_A1, size_in_bytes, s);
    cpuFree(h_masked_A1);

    auto d_masked_A = gpuAddShares(d_masked_A0, d_masked_A1, N, s);    
    gpuFree(d_masked_A0);
    gpuFree(d_masked_A1);
    return std::make_pair(d_masked_A, d_mask_A0);
}

GPUGroupElement* Conv2DLayer::forward(Peer* peer, int party, GPUGroupElement* d_I) {
        auto start = std::chrono::high_resolution_clock::now();
        GPUGroupElement *d_mask_I, *d_F, *d_mask_F;
        if(!d_I) {
            auto output = reconstructMasked(I, key.I, key.size_I, peer, party, &s);
            d_I = output.first;
            d_mask_I = output.second;
        } else {
            I = (GPUGroupElement*) moveToCPU((uint8_t*) d_I, key.mem_size_I, &s);
            d_mask_I = (GPUGroupElement*) moveToGPU((uint8_t*) key.I, key.mem_size_I, &s);
        }
        auto output = reconstructMasked(F, key.F, key.size_F, peer, party, &s);
        d_F = output.first;
        d_mask_F = output.second;

        F = (GPUGroupElement*) moveToCPU((uint8_t*) d_F, key.mem_size_F, &s);
        auto d_C0 = gpu_conv2d(key, party, d_I, d_F, d_mask_I, d_mask_F, &s, 0);

        gpuFree(d_I);
        gpuFree(d_F);
        gpuFree(d_mask_I);
        gpuFree(d_mask_F);

        auto h_C0 = moveToCPU((uint8_t*) d_C0, key.mem_size_O, &s);
        auto h_C1 = exchangeShares(peer, (uint8_t*) h_C0, key.mem_size_O, party, &s);
        cpuFree(h_C0);
        auto d_C1 = (GPUGroupElement*) moveToGPU(h_C1, key.mem_size_O, &s);
        cpuFree(h_C1);
        auto d_C = gpuAddShares(d_C0, d_C1, key.size_O, &s); 
        gpuFree(d_C0);
        gpuFree(d_C1);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "Time for Conv2D in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
        return d_C;
    }

GPUGroupElement* Conv2DLayer::backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad) {
    auto start = std::chrono::high_resolution_clock::now();

    auto d_mask_incomingGrad = (GPUGroupElement*) moveToGPU((uint8_t*) keydI.I, keydI.mem_size_I, &s);
    auto d_mask_I = (GPUGroupElement*) moveToGPU((uint8_t*) key.I, key.mem_size_I, &s);
    auto d_mask_F = (GPUGroupElement*) moveToGPU((uint8_t*) key.F, key.mem_size_F, &s);
    auto d_I = (GPUGroupElement*) moveToGPU((uint8_t*) I, key.mem_size_I, &s);
    auto d_F = (GPUGroupElement*) moveToGPU((uint8_t*) F, key.mem_size_F, &s);
    auto d_dI0 = gpu_conv2d(keydI, party, d_incomingGrad, d_F, d_mask_incomingGrad, d_mask_F, &s, 1);
    auto h_dI0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_dI0, key.mem_size_I, &s);
    auto d_dI = gpuReconstruct(h_dI0, key.size_I, peer, party, &s);
    
    // need to update the gradient here
    // auto d_dF = gpu_conv2d(keydF, party, d_incomingGrad, d_I, d_mask_incomingGrad, d_mask_I, &s, 2);
    auto d_dF0 = gpu_conv2d(keydF, party, d_incomingGrad, d_I, d_mask_incomingGrad, d_mask_I, &s, 2);
    // auto h_dF0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_dF0, key.mem_size_F, &s);
    // auto d_dF = gpuReconstruct(h_dF0, key.size_F, peer, party, &s);
    GPUGroupElement* d_F0; 

    if(party == 0)
        d_F0 = gpuAddShares(d_F, d_dF0, key.size_F, &s);
    else 
        d_F0 = d_dF0;

    // need to update gradient before moving to cpu
    auto h_F0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_F0, key.mem_size_F, &s);
    F = h_F0;

    gpuFree(d_incomingGrad);
    gpuFree(d_I);
    gpuFree(d_F);
    gpuFree(d_mask_incomingGrad);
    gpuFree(d_mask_I);
    gpuFree(d_mask_F);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time for Conv2D back in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

    // F = F - dF;
    return d_dI;
}


void MatmulLayer::init(uint8_t** key_as_bytes) {
        key = readGPUMatmulKey(key_as_bytes);
        s.comm_time = 0;
        s.transfer_time = 0;
    }

void MatmulLayer::initBProp(uint8_t** key_as_bytes) {
    int M = *((int*) *key_as_bytes);
    int N = *((int*) (*key_as_bytes + sizeof(int)));
    int K = *((int*) (*key_as_bytes + 2 * sizeof(int)));
    // std::memcpy((char*) &keydW, *key_as_bytes, 3 * sizeof(int));
    // printf("matmul dims: %d %d %d %lu %lu %lu\n", M, N, K, key.mem_size_A, key.mem_size_B, key.mem_size_C);
    *key_as_bytes += 3 * sizeof(int);

    auto mask_grad = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += key.mem_size_C;

    auto mask_dW = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += key.mem_size_B;

    auto mask_dX = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += key.mem_size_A;

    keydW.Bin = key.Bin;
    keydW.Bout = key.Bout;
    keydW.M = key.K;
    keydW.K = key.M;
    keydW.N = key.N;
    keydW.A = key.A;
    keydW.B = mask_grad;
    keydW.C = mask_dW;
    keydW.size_A = key.size_A;
    keydW.size_B = key.size_C;
    keydW.size_C = key.size_B;
    keydW.mem_size_A = key.mem_size_A;
    keydW.mem_size_B = key.mem_size_C;
    keydW.mem_size_C = key.mem_size_B;

    keydX.Bin = key.Bin;
    keydX.Bout = key.Bout;
    keydX.M = key.M;
    keydX.K = key.N;
    keydX.N = key.K;
    keydX.A = mask_grad;
    keydX.B = key.B;
    keydX.C = mask_dX;
    keydX.size_A = key.size_C;
    keydX.size_B = key.size_B;
    keydX.size_C = key.size_A;
    keydX.mem_size_A = key.mem_size_C;
    keydX.mem_size_B = key.mem_size_B;
    keydX.mem_size_C = key.mem_size_A;
}

GPUGroupElement* matmul(GPUMatmulKey key, int party, Peer* peer, GPUGroupElement* d_A, GPUGroupElement* d_mask_A, GPUGroupElement* d_B, GPUGroupElement* d_mask_B, bool rowMajA, bool rowMajB, bool rowMajC, Stats* s) {
        auto d_C0 = gpu_matmul(key, party, d_A, d_B, d_mask_A, d_mask_B, rowMajA, rowMajB, rowMajC, s);
        auto h_C0 = moveToCPU((uint8_t*) d_C0, key.mem_size_C, s);
        auto h_C1 = exchangeShares(peer, (uint8_t*) h_C0, key.mem_size_C, party, s);
        cpuFree(h_C0);
        auto d_C1 = (GPUGroupElement*) moveToGPU(h_C1, key.mem_size_C, s);
        cpuFree(h_C1);
        auto d_C = gpuAddShares(d_C0, d_C1, key.size_C, s); 
        gpuFree(d_C0);
        gpuFree(d_C1);
        return d_C;
}

GPUGroupElement* MatmulLayer::forward(Peer* peer, int party, GPUGroupElement* d_A) {
        auto start = std::chrono::high_resolution_clock::now();
        A = (GPUGroupElement*) moveToCPU((uint8_t*) d_A, key.mem_size_A, &s);
        GPUGroupElement *d_B, *d_mask_B;
        auto output = reconstructMasked(B, key.B, key.size_B, peer, party, &s);
        d_B = output.first;
        d_mask_B = output.second;

        B = (GPUGroupElement*) moveToCPU((uint8_t*) d_B, key.mem_size_B, &s);
        GPUGroupElement* d_mask_A = (GPUGroupElement*) moveToGPU((uint8_t*) key.A, key.mem_size_A, &s);
        auto d_C = matmul(key, party, peer, d_A, d_mask_A, d_B, d_mask_B, true, true, true, &s);

        gpuFree(d_A);
        gpuFree(d_B);
        gpuFree(d_mask_A);
        gpuFree(d_mask_B);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "Time for matmul in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

        return d_C;
    }

GPUGroupElement* MatmulLayer::backward(Peer* peer, int party, GPUGroupElement* d_grad) {
    auto start = std::chrono::high_resolution_clock::now();
    // dW = X^T delta
    auto d_X = (GPUGroupElement*) moveToGPU((uint8_t*) A, keydW.mem_size_A, &s);
    auto d_mask_X = (GPUGroupElement*) moveToGPU((uint8_t*) keydW.A, keydW.mem_size_A, &s);
    auto d_mask_grad = (GPUGroupElement*) moveToGPU((uint8_t*) keydW.B, keydW.mem_size_B, &s);

    // need to update this properly later
    auto d_dW0 = gpu_matmul(keydW, party, d_X, d_grad, d_mask_X, d_mask_grad, false, true, true, &s);
    auto d_W = (GPUGroupElement*) moveToGPU((uint8_t*) B, key.mem_size_B, &s);
    cpuFree(B);
    if(party == 0) {
        auto d_W0 = gpuAddShares(d_W, d_dW0, key.size_B, &s);
        B = (GPUGroupElement*) moveToCPU((uint8_t*) d_W0, key.mem_size_B, &s);
    } else {
        B = (GPUGroupElement*) moveToCPU((uint8_t*) d_dW0, key.mem_size_B, &s);
    }
    // matmul(keydW, party, peer, d_X, d_mask_X, d_grad, d_mask_grad, false, true, true, &s);
    gpuFree(d_X);
    gpuFree(d_mask_X); 

    // dX = delta W^T
    // auto d_W = (GPUGroupElement*) moveToGPU((uint8_t*) B, keydX.mem_size_B, &s);
    auto d_mask_W = (GPUGroupElement*) moveToGPU((uint8_t*) keydX.B, keydX.mem_size_B, &s);
    auto d_dX = matmul(keydX, party, peer, d_grad, d_mask_grad, d_W, d_mask_W, true, false, true, &s);

    gpuFree(d_grad);
    gpuFree(d_mask_grad);
    gpuFree(d_W);
    gpuFree(d_mask_W);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time for matmul back in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

    return d_dX;
}


void RTLayer::init(uint8_t** key_as_bytes, AESGlobalContext* g, bool faithfulTruncations) {
    this->faithfulTruncations = faithfulTruncations;
    if(faithfulTruncations) {
        key = readGPUReLUTruncateKey(key_as_bytes);
    } else {
        key = readGPULocalTruncateReLUKey(key_as_bytes, true);
    }
    s.comm_time = 0;
    s.transfer_time = 0;
    gaes = g;
}

void RTLayer::initBProp(uint8_t** key_as_bytes) {
    // size_t size_in_bytes = key.num_rts * sizeof(GPUGroupElement);

    // keyBackProp.Bin = key.Bin;
    // keyBackProp.Bout = key.Bout;
    // keyBackProp.shift = key.shift;
    // keyBackProp.num_rts = key.num_rts;

    // keyBackProp.b = (GPUGroupElement *) *key_as_bytes;
    // *key_as_bytes += size_in_bytes;

    // keyBackProp.c = (GPUGroupElement *) *key_as_bytes;
    // *key_as_bytes += size_in_bytes;

    // keyBackProp.d1 = (GPUGroupElement *) *key_as_bytes;
    // *key_as_bytes += size_in_bytes;

    // keyBackProp.d2 = (GPUGroupElement *) *key_as_bytes;
    // *key_as_bytes += size_in_bytes;

    keyBackProp = readGPULocalTruncateReLUKey(key_as_bytes, false);
    printf("num rts: %d\n", keyBackProp.num_rts);
    keyBackProp.a = key.a;
}

GPUGroupElement* RTLayer::backward(Peer* peer, int party, GPUGroupElement* d_grad) {
    gpu_local_truncate(keyBackProp, d_grad, &s);
    auto d_drelu = (uint32_t*) moveToGPU((uint8_t*) dReLU, ((key.num_rts - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE), &s);
    // need to add a truncate kernel here
    auto d_grad0 = finish_relu_truncate(keyBackProp, d_drelu, d_grad, NULL, NULL, party, false, false, &s);
    auto d_dX0 = d_grad0.first;
    // printf("num rts: %d %lu %d\n", key.num_rts, key.num_rts * sizeof(GPUGroupElement), keyBackProp.num_rts);
    auto h_dX0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_dX0, key.num_rts * sizeof(GPUGroupElement), &s);
    auto d_dX = gpuReconstruct(h_dX0, key.num_rts, peer, party, &s);
    return d_dX;
}





GPUGroupElement* RTLayer::forward(Peer* peer, int party, GPUGroupElement* d_I) {
    auto start = std::chrono::high_resolution_clock::now();
    
    GPURTContext c;
    // do something better for this later
    if(faithfulTruncations) {
        gpu_relu_truncate(key, party, d_I, &c, gaes, &s);
    } else {
        gpu_local_truncate_relu(key, party, d_I, &c, gaes, &s);
    }
    size_t lrs_mem_size = key.num_rts * sizeof(GPUGroupElement);
    size_t drelu_mem_size = ((key.num_rts - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    uint8_t *h_lrs0, *h_drelu0, *h_lrs1, *h_drelu1;
    GPUGroupElement *d_lrs1 = NULL;
    uint32_t *d_drelu1;
    if(faithfulTruncations)
    {
        h_lrs0 = moveToCPU((uint8_t*) c.d_lrs0, lrs_mem_size, &s);
        h_lrs1 = exchangeShares(peer, (uint8_t*) h_lrs0, lrs_mem_size, party, &s);
        d_lrs1 = (GPUGroupElement*) moveToGPU(h_lrs1, lrs_mem_size, &s);
        cpuFree(h_lrs1);
    }
    h_drelu0 = moveToCPU((uint8_t*) c.d_drelu0, drelu_mem_size, &s);
    h_drelu1 = exchangeShares(peer, (uint8_t*) h_drelu0, drelu_mem_size, party, &s);
    d_drelu1 = (uint32_t*) moveToGPU(h_drelu1, drelu_mem_size, &s);
    cpuFree(h_drelu1);
    auto d_res = finish_relu_truncate(key, c.d_drelu0, c.d_lrs0, /*c.d_a,*/ d_drelu1, d_lrs1, party, !faithfulTruncations, true, &s);

    auto d_relu0 = d_res.first;
    auto d_drelu = d_res.second;
    auto h_relu0 = (GPUGroupElement*) moveToCPU((uint8_t*) d_relu0, lrs_mem_size, &s);

    auto d_relu = gpuReconstruct(h_relu0, key.num_rts, peer, party, &s);

    // save dRelu for the backward pass
    dReLU = (uint32_t*) moveToCPU((uint8_t*) d_drelu, drelu_mem_size, &s);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time for RT in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

    return d_relu;
}
