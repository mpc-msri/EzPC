
#include "../backend/minillama/gpu/conv2d_layer.h"
#include "../backend/minillama/gpu/maxpool_layer.h"
#include "../backend/minillama/gpu/relu_sign_extend_layer.h"
#include "../backend/minillama/gpu/fc_layer.h"
#include "../backend/minillama/gpu/gpu_data_types.h"
#include "../backend/minillama/gpu/gpu_truncate.h"
#include "../backend/minillama/gpu/gpu_sgd.h"
#include "../backend/minillama/gpu/gpu_file_utils.h"
#include "../backend/minillama/gpu/gpu_fss_utils.h"
#include "../backend/minillama/gpu/gpu_comms.h"
#include "../backend/minillama/gpu/gpu_mem.h"
#include "../backend/minillama/gpu/layer.h"
#include "../backend/minillama/gpu/helper_cuda.h"
#include "../backend/minillama/input_prng.h"
#include "../backend/llama_base.h"
#include "../softmax.h"
#include "../cifar10.hpp"
#include <cassert>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <errno.h>
#include <filesystem>
#undef I

extern "C" void initAESContext(AESGlobalContext* g);

struct {
    u64 N = 100;
    std::string ip = "0.0.0.0";
    int party = 0;
    int numIterations = 1;
} config;

// refactoring caused this

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
    // prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    initCPURandomness();
    AESGlobalContext g;
    initAESContext(&g);

    int bl = 64;
    std::cerr << "> Batch size: " << config.N << std::endl;
    LlamaConfig::party = config.party;
    std::cerr << "> Party: " << LlamaConfig::party << std::endl;
    LlamaConfig::stochasticT = true;
    LlamaConfig::stochasticRT = true;
    auto llama = new LlamaBase<u64>();
    llama->init(config.ip, true);
    auto layer0 = Conv2DLayer(bl, bl, config.N, 32, 32, 3, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, false);
    auto layer1 = MaxPool2DLayer(bl - scale, bl - scale, config.N, 30, 30, 64, 3, 3, 2, 2, 0, 0, 0, 0);
    auto layer2 = ReluSignExtendLayer(bl - scale, bl, config.N * 12544);
    auto layer3 = Conv2DLayer(bl, bl, config.N, 14, 14, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    auto layer4 = MaxPool2DLayer(bl - scale, bl - scale, config.N, 12, 12, 64, 3, 3, 2, 2, 0, 0, 0, 0);
    auto layer5 = ReluSignExtendLayer(bl - scale, bl, config.N * 1600);
    auto layer6 = Conv2DLayer(bl, bl, config.N, 5, 5, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    auto layer7 = MaxPool2DLayer(bl - scale, bl - scale, config.N, 3, 3, 64, 3, 3, 2, 2, 0, 0, 0, 0);
    auto layer8 = ReluSignExtendLayer(bl - scale, bl, config.N * 64);
    auto layer9 = FCLayer(bl, bl, config.N, 10, 64, TruncateType::LocalLRS, TruncateType::StochasticTruncate, true);
    u64 sizes[] = {
        config.N * 3072, 
        config.N * 57600, 
        config.N * 12544, 
        config.N * 12544, 
        config.N * 9216, 
        config.N * 1600, 
        config.N * 1600, 
        config.N * 576, 
        config.N * 64, 
        config.N * 64, 
        config.N * 10, 
    };
    u64 insize = sizes[0];
    u64 outsize = sizes[10];
    std::vector<Layer *> model = {
        &layer0, 
        &layer1, 
        &layer2, 
        &layer3, 
        &layer4, 
        &layer5, 
        &layer6, 
        &layer7, 
        &layer8, 
        &layer9, 
    };
    std::string modelName = "3Layer";
    std::cerr << "> Model: " << modelName << std::endl;

    int numLayers = model.size();
    if(config.party == 1) {
        GPUGroupElement* mask[numLayers + 1];

        for(int i = 0; i < numLayers; ++i) {
            mask[i] = (GPUGroupElement*) cpuMalloc(sizes[i] * sizeof(GPUGroupElement));
        }

        Tensor4D<u64> output_mask(config.N, outsize / config.N, 1, 1);
        Tensor4D<u64> softmax_output_mask(config.N, outsize / config.N, 1, 1);
        std::ofstream f1(modelName + "_key1.dat"), f2(modelName + "_key2.dat"); 
        GroupElement *labels = new GroupElement[50000 * 10];
        llama->ss2m(labels, 50000 * 10);

        char* zeros;
        size_t padding;
        for(int j = 0; j < config.numIterations; j++) {
            initRandomInPlace(mask[0], insize, bl);
            mask[numLayers] = output_mask.data;
            for(int i = 0; i < numLayers; i++) {
                model[i]->genForwardKey(f1, f2, mask[i], mask[i+1]);
            }
            softmax_secfloat(output_mask, softmax_output_mask, scale, 1);
            // mask[numLayers] = softmax_output_mask.data;
            // for(int i = numLayers; i >= 1; i--) {
            //     model[i-1]->genBackwardKey(f1, f2, mask[i], mask[i-1]);
            // }
            for(int img = 0; img < config.N; img++) {
                for(int c = 0; c < 10; c++) {
                    softmax_output_mask(img, c, 0, 0) -= (labels[10 * (j * config.N + img) + c] * (((1LL<<scale))/config.N));
                }
            }
            llama->output(softmax_output_mask); // just for debugging, remove later
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
    } else {
        Peer* peer = LlamaConfig::peer;
        Stats softmaxStats;
        secfloat_init(config.party - 1, config.ip);

        for(int i = 0; i < numLayers; i++) {
            model[i]->initWeights(peer, config.party - 2);
        }

        string filename(modelName + "_key" + std::to_string(config.party-1) + ".dat");
        size_t fileSize = std::filesystem::file_size(filename);
        size_t keySizePerIteration = fileSize / config.numIterations;
        // need to ensure that the key is aligned to 4096 bytes
        assert(keySizePerIteration % 4096 == 0);
        
        int fd = open(filename.data(), O_RDONLY | O_DIRECT | O_LARGEFILE);
        if (fd == -1) assert(0 && "fopen");
        lseek(fd, 0, SEEK_SET);

        // Set up key buffers. we need two buffers to pipeline key read with evaluation
        uint8_t *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf; 
        assert(0 == posix_memalign((void**) &keyBuf1, 4096, keySizePerIteration));
        assert(0 == posix_memalign((void**) &keyBuf2, 4096, keySizePerIteration));
        checkCudaErrors(cudaHostRegister(keyBuf1, keySizePerIteration, cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(keyBuf2, keySizePerIteration, cudaHostRegisterDefault));
        readKey(fd, keySizePerIteration, keyBuf1);
        curKeyBuf = keyBuf1;
        nextKeyBuf = keyBuf2;

        auto start = std::chrono::high_resolution_clock::now();

        // read data
        size_t dataSize;
        auto data = (GPUGroupElement*) readFile("cifar10_share" + std::to_string(config.party-1) + ".dat", &dataSize);
        size_t labelSize;
        auto labels = (GPUGroupElement*) readFile("cifar10_labels" + std::to_string(config.party-1) + ".dat", &labelSize);
        always_assert(labelSize == 50000 * 10 * sizeof(GPUGroupElement));
        llama->ss2m(labels, 50000 * 10);

        Tensor4D<u64> model_output(config.N, outsize / config.N, 1, 1);
        Tensor4D<u64> softmax_output(config.N, outsize / config.N, 1, 1);
        for(int j = 0; j < config.numIterations; j++) {
            #pragma omp parallel 
            {
                #pragma omp sections 
                {
                    #pragma omp section 
                    {
                        if(j < config.numIterations - 1)
                            readKey(fd, keySizePerIteration, nextKeyBuf);
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
                        auto res = maskInput(insize, bl, config.party-2, peer, data + j * insize, layer0.convKey.I, NULL);
                        auto d_I = res.first;

                        for(int i = 0; i < numLayers; i++) {
                            d_I = model[i]->forward(peer, config.party-2, d_I, &g);
                        }
                        size_t size_in_bytes = outsize * sizeof(GPUGroupElement);
                        moveIntoCPUMem((uint8_t *) model_output.data, (uint8_t *) d_I, size_in_bytes, &softmaxStats);
                        // gpuFree(d_I);

                        softmax_secfloat(model_output, softmax_output, scale, config.party);
                        // d_I = (GPUGroupElement*) moveToGPU((uint8_t *) softmax_output.data, size_in_bytes, &softmaxStats);
                        // for(int i = numLayers - 1; i >= 0; i--) {
                        //     d_I = model[i]->backward(peer, config.party-2, d_I, &g);
                        // }
                        for(int img = 0; img < config.N; img++) {
                            for(int c = 0; c < 10; c++) {
                                softmax_output(img, c, 0, 0) -= (labels[10 * (j * config.N + img) + c] * (((1LL<<scale))/config.N));
                            }
                        }

                        llama->output(softmax_output);
                        // llama->output(layer0.F, layer0.p.size_F);
                        printf("softmax output: ");
                        for(int i = 0; i < outsize; i++) {
                            std::cout << ((int64_t)softmax_output.data[i]) << " ";
                        }
                        printf("\n");

                        gpuFree(d_I);
                        auto end2 = std::chrono::high_resolution_clock::now();
                        auto elapsed2 = end2 - start2;
                        std::cout << "Time for iteration "<< j <<": " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed2).count() << std::endl;
                    }
                }
            }
            curKeyBuf = curKeyBuf == keyBuf1 ? keyBuf2 : keyBuf1;
            nextKeyBuf = curKeyBuf == keyBuf1 ? keyBuf1 : keyBuf2;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        std::cout << "Total Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;
    }
    llama->finalize();
    return 0;
}

