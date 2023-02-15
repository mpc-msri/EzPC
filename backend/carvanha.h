#include "../layers.h"
#include <iostream>
#include <string>
#include <vector>

const std::string gpu_includes = R""""(
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
)"""";

const std::string inference_args = R""""(
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
)"""";

const std::string inference_main_init = R""""(
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
    auto llama = new LlamaBase<u64>();
    llama->init(config.ip, true);
)"""";

const std::string main_init = R""""(
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
    LlamaExtended<u64>::init(argv[2], true);
)"""";

const std::string inference_end = R""""(
    int numLayers = model.size();
    if(config.party == 1) {
        GPUGroupElement* mask[numLayers + 1];

        for(int i = 0; i < numLayers; ++i) {
            mask[i] = (GPUGroupElement*) cpuMalloc(sizes[i] * sizeof(GPUGroupElement));
        }

        Tensor4D<u64> output_mask(config.N, outsize / config.N, 1, 1);
        std::ofstream f1(modelName + "_key1.dat"), f2(modelName + "_key2.dat"); 

        char* zeros;
        size_t padding;
        for(int j = 0; j < config.numIterations; j++) {
            initRandomInPlace(mask[0], insize, bl);
            mask[numLayers] = output_mask.data;
            for(int i = 0; i < numLayers; i++) {
                model[i]->genForwardKey(f1, f2, mask[i], mask[i+1]);
            }
            LlamaExtended<u64>::output(output_mask); // just for debugging, remove later
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
        GPUGroupElement *data = new GPUGroupElement[insize];
        for(int i = 0; i < insize; ++i) {
            data[i] = 1; // TODO: proper data reading
        }

        Tensor4D<u64> model_output(config.N, outsize / config.N, 1, 1);
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
                        auto res = maskInput(insize, bl, config.party-2, peer, data, layer0.%s, NULL);
                        auto d_I = res.first;

                        for(int i = 0; i < numLayers; i++) {
                            d_I = model[i]->forward(peer, config.party-2, d_I, &g);
                        }
                        size_t size_in_bytes = outsize * sizeof(GPUGroupElement);
                        moveIntoCPUMem((uint8_t *) model_output.data, (uint8_t *) d_I, size_in_bytes, &softmaxStats);

                        LlamaExtended<u64>::output(model_output);
                        for(int i = 0; i < outsize; i++) {
                            std::cout << model_output.data[i] << " ";
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
    LlamaExtended<u64>::finalize();
    return 0;
}

)"""";

const std::string training_end = R""""(
    int numLayers = model.size();
    if(config.party == 1) {
        GPUGroupElement* mask[numLayers + 1];

        for(int i = 0; i < numLayers; ++i) {
            mask[i] = (GPUGroupElement*) cpuMalloc(sizes[i] * sizeof(GPUGroupElement));
        }

        Tensor4D<u64> output_mask(config.N, outsize / config.N, 1, 1);
        Tensor4D<u64> softmax_output_mask(config.N, outsize / config.N, 1, 1);
        std::ofstream f1(modelName + "_key1.dat"), f2(modelName + "_key2.dat"); 

        char* zeros;
        size_t padding;
        for(int j = 0; j < config.numIterations; j++) {
            initRandomInPlace(mask[0], insize, bl);
            mask[numLayers] = output_mask.data;
            for(int i = 0; i < numLayers; i++) {
                model[i]->genForwardKey(f1, f2, mask[i], mask[i+1]);
            }
            softmax_secfloat(output_mask, softmax_output_mask, scale, 1);
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

                        softmax_secfloat(model_output, softmax_output, scale, config.party);

                        llama->output(softmax_output);
                        printf("softmax output: ");
                        for(int i = 0; i < outsize; i++) {
                            std::cout << softmax_output.data[i] << " ";
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

)"""";

void dump_common_gpu(Sequential<float> &model, u64 h, u64 w, u64 c, std::string name = "model") {
    always_assert((model.layers[0]->name == "Conv2D") || (model.layers[0]->name == "FC"));
    std::cout << gpu_includes;
    std::cout << inference_args;
    std::cout << inference_main_init;

    // filter flattens
    std::vector<Layer<float> *> filteredLayers;
    std::copy_if (model.layers.begin(), model.layers.end(), std::back_inserter(filteredLayers), 
        [](Layer<float> *i){return i->name != "Flatten";} );

    int numLayers = filteredLayers.size();
    struct layer_dims dims[numLayers+1];
    dims[0] = {1, h, w, c};
    
    for(int i = 0; i < filteredLayers.size(); ++i) {
        dims[i+1] = filteredLayers[i]->get_output_dims(dims[i]);
    }

    // output model description
    for(int i = 0; i < filteredLayers.size(); ++i) {
        if (filteredLayers[i]->name == "Conv2D") {
            Conv2D<float> *layer = (Conv2D<float> *) filteredLayers[i];
            std::cout << "    auto layer" << i << " = Conv2DLayer(bl, bl, config.N, " << dims[i].h << ", " << dims[i].w << ", " << dims[i].c << ", " << layer->ks << ", " << layer->ks << ", " << layer->co << ", " << layer->padding << ", " << layer->padding << ", " << layer->padding << ", " << layer->padding << ", 1, 1, true, TruncateType::LocalLRS, TruncateType::StochasticTruncate, " << (i == 0 ? "false" : "true") << ");\n";
        }
        else if (filteredLayers[i]->name == "FC") {
            FC<float> *layer = (FC<float> *) filteredLayers[i];
            std::cout << "    auto layer" << i << " = FCLayer(bl, bl, config.N, " << layer->out << ", " << layer->in << ", TruncateType::LocalLRS, TruncateType::StochasticTruncate, " << (i == 0 ? "false" : "true") << ");\n";
        }
        else if (filteredLayers[i]->name == "MaxPool2D") {
            MaxPool2D<float> *layer = (MaxPool2D<float> *) filteredLayers[i];
            std::cout << "    auto layer" << i << " = MaxPool2DLayer(bl - scale, bl - scale, config.N, " << dims[i].h << ", " << dims[i].w << ", " << dims[i].c <<  ", " << layer->ks << ", " << layer->ks << ", " << layer->stride << ", " << layer->stride <<  ", " << layer->padding << ", " << layer->padding << ", " << layer->padding << ", " << layer->padding << ");\n";
        }
        else if (filteredLayers[i]->name == "ReLU") {
            std::string outlen = "bl";
            if ((i == filteredLayers.size() - 1) || ((filteredLayers[i+1]->name != "Conv2D") && (filteredLayers[i+1]->name != "FC"))) {
                outlen = "bl - scale";
            }
            std::cout << "    auto layer" << i << " = ReluSignExtendLayer(bl - scale, " << outlen << ", config.N * " << dims[i].size() << ");\n";
        }
    }

    // output sizes
    std::cout << "    u64 sizes[] = {\n";
    for(int i = 0; i < (numLayers + 1); ++i) {
        std::cout << "        config.N * " << dims[i].size() << ", \n";
    }
    std::cout << "    };\n";
    std::cout << "    u64 insize = sizes[0];\n";
    std::cout << "    u64 outsize = sizes[" << numLayers << "];\n";

    // dump model array
    std::cout << "    std::vector<Layer *> model = {\n";
    for(int i = 0; i < filteredLayers.size(); ++i) {
        std::cout << "        &layer" << i << ", \n";
    }
    std::cout << "    };\n";

    // model name
    std::cout << "    std::string modelName = \"" << name <<  "\";\n";
    std::cout << "    std::cerr << \"> Model: \" << modelName << std::endl;\n";
}

void dump_inference_gpu_code(Sequential<float> &model, u64 h, u64 w, u64 c, std::string name = "model") {
    dump_common_gpu(model, h, w, c, name);
    if (model.layers[0]->name == "Conv2D")
        std::printf(inference_end.data(), "convKey.I");
    else
        std::printf(inference_end.data(), "matmulKey.A");
}

void dump_training_gpu_code(Sequential<float> &model, u64 h, u64 w, u64 c, std::string name = "model") {
    dump_common_gpu(model, h, w, c, name);
    if (model.layers[0]->name == "Conv2D")
        std::printf(training_end.data(), "convKey.I");
    else
        std::printf(training_end.data(), "matmulKey.A");
}
