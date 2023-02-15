#include "conv2d_layer.h"
#include "ext/fptraining/softmax.h"
#include "maxpool_layer.h"
#include "relu_sign_extend_layer.h"
#include "fc_layer.h"
#include "gpu_data_types.h"
#include "cpu_fss.h"
#include "gpu_truncate.h"
#include "gpu_sgd.h"
#include <minillama/input_prng.h>
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include <cassert>
#include <cstdint>
#include <chrono>
#include "ext/fptraining/backend/llama_extended.h"
#include "ext/fptraining/backend/llama.h"
// #include <fptraining/random.h>
#include "layer.h"
#include <fcntl.h>
#include <errno.h>
#include <filesystem>
#include "helper_cuda.h"
#include "ext/fptraining/cifar10.hpp"
#undef I

extern int errno;

// refactoring caused this
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
    LlamaExtended<u64>::init(/*"0.0.0.0"*/ argv[2], true);

    auto conv1 = Conv2DLayer(bin, bout, N, 32, 32, 3, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, false);
    auto maxpool1 = MaxPool2DLayer(bout - scale, bout - scale, N, conv1.p.OH, conv1.p.OW, conv1.p.CO, 3, 3, 2, 2, 0, 0, 0, 0);
    auto relu1 = ReluSignExtendLayer(bout - scale, bout, N * maxpool1.p.H * maxpool1.p.W * maxpool1.p.C);

    auto conv2 = Conv2DLayer(bout, bout, N, maxpool1.p.H, maxpool1.p.W, maxpool1.p.C, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    auto maxpool2 = MaxPool2DLayer(bout - scale, bout - scale, N, conv2.p.OH, conv2.p.OW, conv2.p.CO, 3, 3, 2, 2, 0, 0, 0, 0);
    auto relu2 = ReluSignExtendLayer(bout - scale, bout, N * maxpool2.p.H * maxpool2.p.W * maxpool2.p.C);

    auto conv3 = Conv2DLayer(bout, bout, N, maxpool2.p.H, maxpool2.p.W, maxpool2.p.C, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    auto maxpool3 = MaxPool2DLayer(bout - scale, bout - scale, N, conv3.p.OH, conv3.p.OW, conv3.p.CO, 3, 3, 2, 2, 0, 0, 0, 0);
    auto relu3 = ReluSignExtendLayer(bout - scale, bout, N * maxpool3.p.H * maxpool3.p.W * maxpool3.p.C);
    
    auto fc4 = FCLayer(bin, bout, 100, 10, 64, TruncateType::StochasticTruncate, TruncateType::StochasticTruncate, true);

    Layer* model[] = {&conv1, &maxpool1, &relu1, &conv2, &maxpool2, &relu2, &conv3, &maxpool3, &relu3, &fc4};
    int numLayers = 10;
    int numIterations = atoi(argv[5]);
    if(atoi(argv[4]) == 0) {
        GPUGroupElement* mask[numLayers + 1];

        mask[0] = (GPUGroupElement*) cpuMalloc(conv1.p.size_I * sizeof(GPUGroupElement));
        
        mask[1] = (GPUGroupElement*) cpuMalloc(conv1.p.size_O * sizeof(GPUGroupElement));
        mask[2] = (GPUGroupElement*) cpuMalloc(relu1.numRelus * sizeof(GPUGroupElement));
        mask[3] = (GPUGroupElement*) cpuMalloc(relu1.numRelus * sizeof(GPUGroupElement));

        mask[4] = (GPUGroupElement*) cpuMalloc(conv2.p.size_O * sizeof(GPUGroupElement));
        mask[5] = (GPUGroupElement*) cpuMalloc(relu2.numRelus * sizeof(GPUGroupElement));
        mask[6] = (GPUGroupElement*) cpuMalloc(relu2.numRelus * sizeof(GPUGroupElement));
        
        mask[7] = (GPUGroupElement*) cpuMalloc(conv3.p.size_O * sizeof(GPUGroupElement));
        mask[8] = (GPUGroupElement*) cpuMalloc(relu3.numRelus * sizeof(GPUGroupElement));
        mask[9] = (GPUGroupElement*) cpuMalloc(relu3.numRelus * sizeof(GPUGroupElement));

        Tensor4D<u64> mask_fc4_O(N, 10, 1, 1), mask_softmax_O(N, 10, 1, 1);
        std::ofstream f1("3layer_network_key1.dat"), f2("3layer_network_key2.dat"); 
        
        char* zeros;
        size_t padding;
        for(int j = 0; j < numIterations; j++) {
            initRandomInPlace(mask[0], conv1.p.size_I, bin);
            mask[10] = mask_fc4_O.data;
            for(int i = 0; i < 10; i++) {
                model[i]->genForwardKey(f1, f2, mask[i], mask[i+1]);
            }
            softmax_secfloat(mask_fc4_O, mask_softmax_O, scale, party + 1);
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
        secfloat_init(party + 1, /*"0.0.0.0"*/ argv[2]);
        size_t dataSize;
        auto data = (GPUGroupElement*) readFile("cifar10_share" + std::to_string(party+1) + ".dat", &dataSize);

        for(int i = 0; i < numLayers; i++) {
            model[i]->initWeights(peer, party);
        }
        Tensor4D<u64> h_fc4_O(N, 10, 1, 1), h_softmax_O(N, 10, 1, 1);
        Stats softmaxStats;
        // int numIterations = 5;
        string filename("3layer_network_key" + std::to_string(party+1) + ".dat");
        size_t fileSize = std::filesystem::file_size(filename);
        size_t keySize = fileSize / numIterations;
        // need to ensure that the key is aligned to 4096 bytes
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
                        auto res = maskInput(conv1.p.size_I, bin, party, peer, &data[(j * N) % (numImages / N)], conv1.convKey.I, NULL);
                        auto d_I = res.first;
                        auto d_mask_I = res.second;
                        conv1.d_mask_I = d_mask_I;
                        for(int i = 0; i < numLayers; i++) d_I = model[i]->forward(peer, party, d_I, &g);
                        size_t size_in_bytes = N * 10 * sizeof(GPUGroupElement);
                        moveIntoCPUMem((uint8_t *) h_fc4_O.data, (uint8_t *) d_I, size_in_bytes, &softmaxStats);
                        gpuFree(d_I);
                        softmax_secfloat(h_fc4_O, h_softmax_O, scale, LlamaConfig::party);
                        d_I = (GPUGroupElement*) moveToGPU((uint8_t *) h_fc4_O.data, size_in_bytes, &softmaxStats);
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

        // todo: finish accelerating maxpool keygen (openmp pargma parallel for collapse below fh and fw) and other keygens with openmp
        // finish the maxpool optimization you proposed: communication still isn't optimized

        // int fadviseerr = posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        // if(fadviseerr != 0) {
            // printf("fadvise err: %d\n", fadviseerr);
            // assert(0 && "fadvise");
        // }
        // errExit("open");

        // uint8_t* key_as_bytes = readFile("3layer_network_key" + std::to_string(party+1) + ".dat", &file_size);
        // printf("file size: %lu bytes\n", file_size);
