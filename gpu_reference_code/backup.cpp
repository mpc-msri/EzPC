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
#include <cassert>
#include <cstdint>
#include <chrono>
#include "backend/llama_extended.h"
#include "backend/llama.h"
#include "backend/minillama/gpu/layer.h"
#include <fcntl.h>
#include <errno.h>
#include <filesystem>
#include "backend/minillama/gpu/helper_cuda.h"
#include "./cifar10.hpp"
#undef I

extern int errno;

struct {
    int N = 100;
    std::string ip = "0.0.0.0";
    int party = 0;
    int numIterations = 1;
} config;

// refactoring caused this
extern "C" void initAESContext(AESGlobalContext* g);

void parseArgs(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "need atleast party = 1/2/3" << std::endl;
        std::cerr << "usage: " << argv[0] << " <party> [ip] [batchsize] [iterations]" << std::endl;
    }
    config.party = atoi(argv[1]);
    if (argc > 2) {
        config.ip = argv[2];
    }
    if (argc > 3) {
        config.N = atoi(argv[3]);
    }
    if (argc > 4) {
        config.numIterations = atoi(argv[4]);
    }
}

int main(int argc, char *argv[]) {
    parseArgs(argc, argv);
    prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    initCPURandomness();
    AESGlobalContext g;
    initAESContext(&g);

    int bl = 64;
    std::cerr << "> Batch size: " << config.N << std::endl;
    LlamaConfig::party = config.party;
    std::cerr << "> Party: " << LlamaConfig::party << std::endl;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    LlamaExtended<u64>::init(config.ip, true);

    auto layer0 = Conv2DLayer(bl, bl, config.N, 32, 32, 3, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, false);
    auto layer1 = MaxPool2DLayer(bl - scale, bl - scale, config.N, conv1.p.OH, conv1.p.OW, conv1.p.CO, 3, 3, 2, 2, 0, 0, 0, 0);
    auto layer2 = ReluSignExtendLayer(bl - scale, bl, config.N * maxpool1.p.H * maxpool1.p.W * maxpool1.p.C);

    size_t insize = conv1.p.size_I;
    size_t outsize = relu1.numRelus;
    size_t sizes[] = {insize, conv1.p.size_O, relu1.numRelus, outsize};

    Layer* model[] = {&layer0, &layer1, &layer2};
    int numLayers = 3;

    int numIterations = atoi(argv[5]);
    if(atoi(argv[4]) == 0) {
        GPUGroupElement* mask[numLayers + 1];

        for(int i = 0; i < numLayers; ++i) {
            mask[i] = (GPUGroupElement*) cpuMalloc(sizes[i] * sizeof(GPUGroupElement));
        }

        Tensor4D<u64> mask_output(N, outsize / N, 1, 1);
        std::ofstream f1("3layer_network_key1.dat"), f2("3layer_network_key2.dat"); 
        
        char* zeros;
        size_t padding;
        for(int j = 0; j < numIterations; j++) {
            initRandomInPlace(mask[0], insize, bin);
            mask[numLayers] = mask_output.data;
            for(int i = 0; i < numLayers; i++) {
                model[i]->genForwardKey(f1, f2, mask[i], mask[i+1]);
            }
            LlamaExtended<u64>::output(mask_output);
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
        int H = 32;
        int W = 32;
        int C = 3;
        size_t dataSize;

        for(int i = 0; i < numLayers; i++) {
            model[i]->initWeights(peer, party);
        }
        Tensor4D<u64> h_output(N, outsize / N, 1, 1);
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
        GPUGroupElement *data = new GPUGroupElement[insize];
        memset(data, 1, insize);
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
                        // for(int i = numLayers - 1; i >= 0; i--) {
                        //     model[i]->readBackwardKey(&curKeyBuf);
                        // }
                        // do batches better
                        auto res = maskInput(insize, bin, party, peer, data, conv1.convKey.I, NULL);
                        auto d_I = res.first;
                        auto d_mask_I = res.second;
                        conv1.d_mask_I = d_mask_I;
                        for(int i = 0; i < numLayers; i++) d_I = model[i]->forward(peer, party, d_I, &g);
                        size_t size_in_bytes = outsize * sizeof(GPUGroupElement);
                        moveIntoCPUMem((uint8_t *) h_output.data, (uint8_t *) d_I, size_in_bytes, &softmaxStats);
                        LlamaExtended<u64>::output(h_output);
                        for(int i = 0; i < outsize; i++) printf("%lu ", h_output.data[i]);
                        printf("\n");
                        gpuFree(d_I);
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
