// #include "backend/minillama/gpu/conv2d_layer.h"
// #include "softmax.h"
// #include "backend/minillama/gpu/maxpool_layer.h"
// #include "backend/minillama/gpu/relu_sign_extend_layer.h"
// #include "backend/minillama/gpu/fc_layer.h"
// #include "backend/minillama/gpu/gpu_data_types.h"
// #include "backend/minillama/gpu/gpu_truncate.h"
// #include "backend/minillama/gpu/gpu_sgd.h"
// #include "backend/minillama/input_prng.h"
// #include "backend/minillama/gpu/gpu_file_utils.h"
// #include "backend/minillama/gpu/gpu_fss_utils.h"
// #include "backend/minillama/gpu/gpu_comms.h"
// #include "backend/minillama/gpu/gpu_mem.h"
// #include <cassert>
// #include <cstdint>
// #include <chrono>
// #include "backend/llama_extended.h"
// #include "backend/llama.h"
// #include "backend/minillama/gpu/layer.h"
// #include <fcntl.h>
// #include <errno.h>
// #include <filesystem>
// #include "backend/minillama/gpu/helper_cuda.h"
// #include "./cifar10.hpp"
// #undef I

// extern int errno;

// // refactoring caused this
// extern "C" void initAESContext(AESGlobalContext* g);

// int main(int argc, char *argv[]) {
//     prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
//     initCPURandomness();
//     AESGlobalContext g;
//     initAESContext(&g);
//     int bin = 64;
//     int bout = 64;
//     int N = atoi(argv[3]);//128;
//     printf("Batch size: %d\n", N);
//     // automatically truncates by scale
//     int party = atoi(argv[1]);
//     LlamaConfig::party = party + 1 + atoi(argv[4]);
//     printf("party: %d\n", LlamaConfig::party);
//     LlamaConfig::stochasticT = true;
//     LlamaConfig::stochasticRT = true;
//     LlamaExtended<u64>::init(/*"0.0.0.0"*/ argv[2], true);
    
//     auto layer0 = FCLayer(bin, bout, N, 400, 500000, TruncateType::LocalLRS, TruncateType::StochasticTruncate, false);
//     size_t insize = N * 500000;
//     size_t outsize = N * 400;
//     size_t sizes[] = {insize, outsize};

//     Layer* model[] = {&layer0};
//     int numLayers = 1;

//     int numIterations = atoi(argv[5]);
//     if(atoi(argv[4]) == 0) {
//         GPUGroupElement* mask[numLayers + 1];

//         for(int i = 0; i < numLayers; ++i) {
//             mask[i] = (GPUGroupElement*) cpuMalloc(sizes[i] * sizeof(GPUGroupElement));
//         }

//         Tensor4D<u64> mask_output(N, outsize / N, 1, 1);
//         std::ofstream f1("chemistry_key1.dat"), f2("chemistry_key2.dat"); 
        
//         char* zeros;
//         size_t padding;
//         for(int j = 0; j < numIterations; j++) {
//             initRandomInPlace(mask[0], insize, bin);
//             mask[numLayers] = mask_output.data;
//             for(int i = 0; i < numLayers; i++) {
//                 model[i]->genForwardKey(f1, f2, mask[i], mask[i+1]);
//             }
//             LlamaExtended<u64>::output(mask_output);
//             if(j == 0) {
//                 assert(sizeof(std::ofstream::pos_type) == 16);
//                 size_t keySize = f1.tellp();
//                 padding = 4096 - (keySize % 4096);
//                 zeros = new char[padding]; 
//                 memset(zeros, 0, padding);
//             }
//             f1.write(zeros, padding);
//             f2.write(zeros, padding);
//         }
//         f1.close();
//         f2.close();
//         delete [] zeros;
//         LlamaExtended<u64>::finalize();

//     } else {
//         Peer* peer = LlamaConfig::peer;
//         size_t dataSize;

//         for(int i = 0; i < numLayers; i++) {
//             model[i]->initWeights(peer, party);
//         }
//         Tensor4D<u64> h_output(N, outsize / N, 1, 1);
//         Stats softmaxStats;
//         // int numIterations = 5;
//         string filename("chemistry_key" + std::to_string(party+1) + ".dat");
//         size_t fileSize = std::filesystem::file_size(filename);
//         size_t keySize = fileSize / numIterations;
//         // need to ensure that the key is aligned to 4096 bytes
//         assert(keySize % 4096 == 0);
        
//         int fd = open(filename.data(), O_RDONLY | O_DIRECT | O_LARGEFILE);
//         if (fd == -1) assert(0 && "fopen");
//         lseek(fd, 0, SEEK_SET);

//         uint8_t *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf; 
//         int err = posix_memalign((void**) &keyBuf1, 4096, keySize);
//         printf("err no: %d\n", err);
//         err = posix_memalign((void**) &keyBuf2, 4096, keySize);
//         printf("err no: %d\n", err);
//         checkCudaErrors(cudaHostRegister(keyBuf1, keySize, cudaHostRegisterDefault));
//         checkCudaErrors(cudaHostRegister(keyBuf2, keySize, cudaHostRegisterDefault));
//         readKey(fd, keySize, keyBuf1);
//         // lseek(fd, 0, SEEK_SET);
//         curKeyBuf = keyBuf1;
//         nextKeyBuf = keyBuf2;
//         auto start = std::chrono::high_resolution_clock::now();
//         GPUGroupElement *data = new GPUGroupElement[insize];
//         memset(data, 1LL<<scale, insize);
//         for(int j = 0; j < numIterations; j++) {
//             #pragma omp parallel 
//             {
//                 #pragma omp sections 
//                 {
//                     #pragma omp section 
//                     {
//                         if(j < numIterations - 1)
//                             readKey(fd, keySize, nextKeyBuf);
//                     }
//                     #pragma omp section 
//                     {
//                         auto start2 = std::chrono::high_resolution_clock::now();
//                         for(int i = 0; i < numLayers; i++) {
//                             model[i]->readForwardKey(&curKeyBuf);
//                         }
//                         // for(int i = numLayers - 1; i >= 0; i--) {
//                         //     model[i]->readBackwardKey(&curKeyBuf);
//                         // }
//                         // do batches better
//                         auto res = maskInput(insize, bin, party, peer, data, layer0.matmulKey.A, NULL);
//                         auto d_I = res.first;
//                         auto d_mask_I = res.second;
//                         // conv1.d_mask_I = d_mask_I;
//                         // layer0.mask_dX = d_mask_I;
//                         for(int i = 0; i < numLayers; i++) d_I = model[i]->forward(peer, party, d_I, &g);
//                         size_t size_in_bytes = outsize * sizeof(GPUGroupElement);
//                         moveIntoCPUMem((uint8_t *) h_output.data, (uint8_t *) d_I, size_in_bytes, &softmaxStats);
//                         LlamaExtended<u64>::output(h_output);
//                         for(int i = 0; i < outsize; i++) printf("%lu ", h_output.data[i]);
//                         printf("\n");
//                         gpuFree(d_I);
//                         auto end2 = std::chrono::high_resolution_clock::now();
//                         auto elapsed2 = end2 - start2;
//                         std::cout << "Time for compute: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed2).count() << std::endl;
//                     }
//                 }
//             }
//             curKeyBuf = curKeyBuf == keyBuf1 ? keyBuf2 : keyBuf1;
//             nextKeyBuf = curKeyBuf == keyBuf1 ? keyBuf1 : keyBuf2;
//         }
//         auto end = std::chrono::high_resolution_clock::now();
//         auto elapsed = end - start;
//         std::cout << "Time for compute: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
//     }
//     return 0;
// }

#include "backend/minillama/gpu/fc_layer.h"
#include "backend/minillama/gpu/gpu_data_types.h"
// #include "gpu_truncate.h"
#include "backend/minillama/gpu/gpu_sgd.h"
#include "backend/minillama/input_prng.h"
#include "backend/minillama/gpu/gpu_file_utils.h"
#include "backend/minillama/gpu/gpu_fss_utils.h"
#include "backend/minillama/gpu/gpu_comms.h"
#include "backend/minillama/gpu/gpu_mem.h"
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
