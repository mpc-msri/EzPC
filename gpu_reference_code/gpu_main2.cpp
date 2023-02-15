
#include "backend/minillama/gpu/conv2d_layer.h"
#include "softmax.h"
#include "backend/minillama/gpu/maxpool_layer.h"
#include "backend/minillama/gpu/relu_sign_extend_layer.h"
#include "backend/minillama/gpu/fc_layer.h"
#include "backend/minillama/gpu/gpu_data_types.h"
#include "backend/minillama/gpu/gpu_truncate.h"
#include "backend/minillama/gpu/gpu_sgd.h"
#include "backend/minillama/input_prng.h"
#include "backend/minillama/gpu/gpu_file_utils.h"
#include "backend/minillama/gpu/gpu_fss_utils.h"
#include "backend/minillama/gpu/gpu_comms.h"
#include "backend/minillama/gpu/gpu_mem.h"
#include "backend/llama_extended.h"
#include "backend/llama.h"
#include "backend/minillama/gpu/layer.h"
#include "backend/minillama/gpu/helper_cuda.h"
#include "./cifar10.hpp"
#include <cassert>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <errno.h>
#include <filesystem>
#undef I
extern int errno;

extern "C" void initAESContext(AESGlobalContext* g);

int main(int argc, char *argv[]) {
    prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    initCPURandomness();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64;
    int bout = 64;
    int N = atoi(argv[3]);//128;
    printf("Batch size: %d\n", N);
    // automatically truncates by scale
    int party = atoi(argv[1]);
    LlamaConfig::party = party + 1 + atoi(argv[4]);
    printf("party: %d\n", LlamaConfig::party);
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaExtended<u64>::init(argv[2], false);
    auto layer0 = Conv2DLayer(bin, bout, N, 32, 32, 3, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, false);
    auto layer1 = MaxPool2DLayer(bout - scale, bout - scale, N, 30, 30, 64, 3, 3, 2, 2, 0, 0, 0, 0);
    auto layer2 = ReluSignExtendLayer(bout - scale, bout, N * 12544);
    auto layer3 = Conv2DLayer(bin, bout, N, 14, 14, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    auto layer4 = MaxPool2DLayer(bout - scale, bout - scale, N, 12, 12, 64, 3, 3, 2, 2, 0, 0, 0, 0);
    auto layer5 = ReluSignExtendLayer(bout - scale, bout, N * 1600);
    auto layer6 = Conv2DLayer(bin, bout, N, 5, 5, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    auto layer7 = MaxPool2DLayer(bout - scale, bout - scale, N, 3, 3, 64, 3, 3, 2, 2, 0, 0, 0, 0);
    auto layer8 = ReluSignExtendLayer(bout - scale, bout, N * 64);
    auto layer9 = FCLayer(bin, bout, N, 10, 64, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    Layer* model[] = {&layer0, &layer1, &layer2, &layer3, &layer4, &layer5, &layer6, &layer7, &layer8, &layer9, };
    int numLayers = 10;
    int numIterations = atoi(argv[5]);
    if (atoi(argv[4]) == 0) {
        GPUGroupElement* mask[numLayers + 1];
        mask[0] = (GPUGroupElement*) cpuMalloc(N * 3072 * sizeof(GPUGroupElement));
        mask[1] = (GPUGroupElement*) cpuMalloc(N * 57600 * sizeof(GPUGroupElement));
        mask[2] = (GPUGroupElement*) cpuMalloc(N * 12544 * sizeof(GPUGroupElement));
        mask[3] = (GPUGroupElement*) cpuMalloc(N * 12544 * sizeof(GPUGroupElement));
        mask[4] = (GPUGroupElement*) cpuMalloc(N * 9216 * sizeof(GPUGroupElement));
        mask[5] = (GPUGroupElement*) cpuMalloc(N * 1600 * sizeof(GPUGroupElement));
        mask[6] = (GPUGroupElement*) cpuMalloc(N * 1600 * sizeof(GPUGroupElement));
        mask[7] = (GPUGroupElement*) cpuMalloc(N * 576 * sizeof(GPUGroupElement));
        mask[8] = (GPUGroupElement*) cpuMalloc(N * 64 * sizeof(GPUGroupElement));
        mask[9] = (GPUGroupElement*) cpuMalloc(N * 64 * sizeof(GPUGroupElement));
        Tensor4D<u64> mask_output(N, 10, 1, 1), mask_softmax_O(N, 10, 1, 1);
        std::ofstream f1("key1.dat"), f2("key2.dat");
        char* zeros;
        size_t padding;
        for(int j = 0; j < numIterations; j++) {
            initRandomInPlace(mask[0], N * 3072, bin);
            mask[10] = mask_output.data;
            for(int i = 0; i < 10; i++) {
                model[i]->genForwardKey(f1, f2, mask[i], mask[i+1]);
            }
            softmax_secfloat(mask_output, mask_softmax_O, scale, party + 1);
            // LlamaExtended<u64>::output(mask_output);
            mask[10] = mask_softmax_O.data;
            for(int i = numLayers; i >= 1; i--) {
                model[i-1]->genBackwardKey(f1, f2, mask[i], mask[i-1]);
            }
            if(j == 0) {
                assert(sizeof(std::ofstream::pos_type) == 16);
                size_t keySize = f1.tellp();
                padding = 4096 - (keySize % 4096);
                zeros = new char[padding];
                memset(zeros, 0, padding);
            }
            f1.write(zeros, padding);
            f2.write(zeros, padding);
        }
        f1.close();
        f2.close();
        delete [] zeros;
        LlamaExtended<u64>::finalize();
    } else {
        Peer* peer = LlamaConfig::peer;
        int numImages = 50000;//dataset.training_images.size();
        int H = 32;
        int W = 32;
        int C = 3;
        secfloat_init(party + 1, argv[2]);
        size_t dataSize;
        auto data = (GPUGroupElement*) readFile("data_share" + std::to_string(party+1) + ".dat", &dataSize);
        for(int i = 0; i < numLayers; i++) {
            model[i]->initWeights(peer, party);
        }
        Tensor4D<u64> h_output(N, 10, 1, 1), h_softmax_O(N, 10, 1, 1);
        Stats softmaxStats;
        string filename("key" + std::to_string(party+1) + ".dat");
        size_t fileSize = std::filesystem::file_size(filename);
        size_t keySize = fileSize / numIterations;
        assert(keySize % 4096 == 0);

        int fd = open(filename.data(), O_RDONLY | O_DIRECT | O_LARGEFILE);
        if (fd == -1) assert(0 && "fopen");
        lseek(fd, 0, SEEK_SET);

        uint8_t *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf; 
        int err = posix_memalign((void**) &keyBuf1, 4096, keySize);
        printf("err no: %d\n", err);
        err = posix_memalign((void**) &keyBuf2, 4096, keySize);
        printf("err no: %d\n", err);
        checkCudaErrors(cudaHostRegister(keyBuf1, keySize, cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(keyBuf2, keySize, cudaHostRegisterDefault));
        readKey(fd, keySize, keyBuf1);
        // lseek(fd, 0, SEEK_SET);
        curKeyBuf = keyBuf1;
        nextKeyBuf = keyBuf2;
        auto start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < numIterations; j++) {
            #pragma omp parallel 
            {
                #pragma omp sections 
                {
                    #pragma omp section 
                    {
                        if(j < numIterations - 1)
                            readKey(fd, keySize, nextKeyBuf);
                    }
                    #pragma omp section 
                    {
                        auto start2 = std::chrono::high_resolution_clock::now();
                        for(int i = 0; i < numLayers; i++) {
                            model[i]->readForwardKey(&curKeyBuf);
                        }
                        for(int i = numLayers - 1; i >= 0; i--) {
                            model[i]->readBackwardKey(&curKeyBuf);
                        }
                        // do batches better
                        auto res = maskInput(layer0.p.size_I, bin, party, peer, &data[(j * N) % (numImages / N)], layer0.convKey.I, NULL);
                        auto d_I = res.first;
                        auto d_mask_I = res.second;
                        layer0.d_mask_I = d_mask_I;
                        for(int i = 0; i < numLayers; i++) d_I = model[i]->forward(peer, party, d_I, &g);
                        size_t size_in_bytes = N * 10 * sizeof(GPUGroupElement);
                        moveIntoCPUMem((uint8_t *) h_output.data, (uint8_t *) d_I, size_in_bytes, &softmaxStats);
                        gpuFree(d_I);
                        softmax_secfloat(h_output, h_softmax_O, scale, LlamaConfig::party);
                        // LlamaExtended<u64>::output(h_output);
                        h_output.print();
                        d_I = (GPUGroupElement*) moveToGPU((uint8_t *) h_output.data, size_in_bytes, &softmaxStats);
                        for(int i = numLayers - 1; i >= 0; i--) d_I = model[i]->backward(peer, party, d_I, &g);
                        auto end2 = std::chrono::high_resolution_clock::now();
                        auto elapsed2 = end2 - start2;
                        std::cout << "Time for compute: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed2).count() << std::endl;
                    }
                }
            }
            curKeyBuf = curKeyBuf == keyBuf1 ? keyBuf2 : keyBuf1;
            nextKeyBuf = curKeyBuf == keyBuf1 ? keyBuf1 : keyBuf2;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "Time for compute: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    }
    return 0;
}
