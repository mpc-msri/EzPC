// 
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <omp.h>
#include <unistd.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"

#include "cnn.h"

#include <sytorch/backend/llama_base.h>
#include <sytorch/softmax.h>

u64 *gpuGenSoftmaxKey(int batchSz, int numClasses, u64 *d_mask_I, bool secfloat, LlamaBase<u64> *llama)
{
    Tensor4D<u64> inpMask(batchSz, numClasses, 1, 1);
    Tensor4D<u64> softmaxOpMask(batchSz, numClasses, 1, 1);
    size_t memSz = batchSz * numClasses * sizeof(u64);
    moveIntoCPUMem((u8 *)inpMask.data, (u8 *)d_mask_I, memSz, NULL);
    gpuFree(d_mask_I);
    if (secfloat)
    {
        softmax_secfloat(inpMask, softmaxOpMask, dcf::orca::global::scale, 1);
    }
    else
    {
        pirhana_softmax(inpMask, softmaxOpMask, dcf::orca::global::scale);
    }
    d_mask_I = (u64 *)moveToGPU((u8 *)softmaxOpMask.data, memSz, NULL);
    return d_mask_I;
}

void genModelKey(dcf::orca::GPUModel<u64> *m, u8 **bufPtr, int party, AESGlobalContext *g, bool secfloat, LlamaBase<u64> *llama, int epoch)
{
    auto d_mask_I = randomGEOnGpu<u64>(m->inpSz, dcf::orca::global::bw);
    u64 *d_mask_O = NULL;
    for (int i = 0; i < m->layers.size(); i++)
    {
        // printf("Layer=%s\n", m->layers[i]->name.data());
        d_mask_O = m->layers[i]->genForwardKey(bufPtr, party, d_mask_I, g);
        assert(d_mask_O != d_mask_I);
        gpuFree(d_mask_I);
        d_mask_I = d_mask_O;
    }
    d_mask_I = gpuGenSoftmaxKey(m->batchSz, m->classes, d_mask_I, secfloat, llama);
    for (int i = m->layers.size() - 1; i >= 0; i--)
    {
        d_mask_I = m->layers[i]->genBackwardKey(bufPtr, party, d_mask_I, g, epoch);
    }
}

void writeKeySz(std::string dir, std::string modelName, u64 keySz)
{
    makeDir(dir);
    std::ofstream keySzFile(dir + modelName + ".txt");
    keySzFile << keySz;
    keySzFile.close();
}

void dealerE2E(std::string modelName, int party, int epochs, int blocks, int blockSz, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir, int sleepInt, std::string weightsMask = "")
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    initGPUMemPool();
    sytorch_init();
    assert(epochs < 6);

    auto expName = modelName + "-" + std::to_string(epochs) + "e-" + std::to_string(blocks) + "b";
    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto lossDir = trainingDir + "loss/" + expName + "/";
    auto keySzDir = trainingDir + "keysize/";
    auto weightsDir = lossDir + "weights/";

    // assumes output/P0/training exists
    makeDir(trainingDir + "loss/");
    makeDir(lossDir);
    makeDir(weightsDir);
    makeDir(keySzDir);

    auto dealerFifoName = "/tmp/dealerFifo" + std::to_string(party);
    auto evalFifoName = "/tmp/evaluatorFifo" + std::to_string(party);

    int dealerFifo, evalFifo;
    bool sync = (epochs * blocks > 1);

    char one = 1;
    char two = 2;

    // load the model
    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);
    m->initWeights(weightsMask, false);

    char *zeros;
    size_t padding, bufSize = 20 * OneGB;
    u8 *startPtr, *curPtr, *tmpPtr1, *tmpPtr2;
    getAlignedBuf(&startPtr, bufSize);

    // initialize llama
    LlamaConfig::party = DEALER;
    auto llama = new LlamaBase<u64>();
    tmpPtr1 = (u8 *)malloc(OneGB);
    bool isServer = party + 2 == SERVER;
    llama->initDealer((char **)(isServer ? &curPtr : &tmpPtr2), (char **)(isServer ? &tmpPtr2 : &curPtr));

    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party) + ".dat";

    for (int l = 0; l < epochs; l++)
    {
        for (int k = 0; k < blocks; k++)
        {
            std::cout << keyFile << std::endl;
            int fd = openForWriting(keyFile);
            printf("Iteration=%u\n", l * blocks * blockSz + k * blockSz);
            for (int j = 0; j < blockSz; j++)
            {
                curPtr = startPtr;
                tmpPtr2 = tmpPtr1;
                genModelKey(m, &curPtr, party, &g, secfloat, (LlamaBase<u64> *)llama, l);
                if (l == 0 && k == 0 && j == 0)
                {
                    size_t keySz = curPtr - startPtr;
                    padding = 4096 - (keySz % 4096);
                    keySz += padding;
                    zeros = new char[padding];
                    memset(zeros, 0, padding);
                    writeKeySz(keySzDir, modelName, keySz);
                    if (sync)
                    {
                        mkfifo(dealerFifoName.c_str(), 0666);
                        mkfifo(evalFifoName.c_str(), 0666);
                        // dealer writes to this fifo
                        printf("Opening dealer fifo=%s\n", dealerFifoName.c_str());
                        dealerFifo = open(dealerFifoName.c_str(), O_WRONLY);
                        // dealer reads from this fifo
                        printf("Opening evaluator fifo=%s\n", evalFifoName.c_str());
                        evalFifo = open(evalFifoName.c_str(), O_RDONLY);
                    }
                }
                memcpy(curPtr, zeros, padding);
                curPtr += padding;
                writeKeyBuf(fd, curPtr - startPtr, startPtr);
            }
            assert(0 == fsync(fd) && "sync error!");
            close(fd);
            if (party == SERVER0)
                m->dumpWeights(weightsDir + "weights_mask_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat");
            if (sync)
            {
                printf("Sleeping for %d seconds.\n", sleepInt);
                sleep(sleepInt);
                // dealer writes to one and evaluator writes to two
                assert(1 == write(dealerFifo, &one, sizeof(char)));
                assert(1 == read(evalFifo, &two, sizeof(char)));
                assert(two == 2);
            }
        }
    }
    delete[] zeros;
    destroyGPURandomness();
    if (sync)
    {
        close(dealerFifo);
        close(evalFifo);
    }
}

void dealerPerf(std::string modelName, int party, int iterations, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir, int sleepInt)
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    initGPUMemPool();
    sytorch_init();

    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto keySzDir = trainingDir + "keysize/";
    makeDir(keySzDir);

    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);

    char *zeros;
    // Neha: remember to change this later
    size_t padding, bufSize = 20 * OneGB;
    u8 *startPtr, *curPtr, *tmpPtr1, *tmpPtr2;
    getAlignedBuf(&startPtr, bufSize);

    // initialize llama
    LlamaConfig::party = DEALER;
    auto llama = new LlamaBase<u64>();
    tmpPtr1 = (u8 *)malloc(OneGB);
    bool isServer = party + 2 == SERVER;
    llama->initDealer((char **)(isServer ? &curPtr : &tmpPtr2), (char **)(isServer ? &tmpPtr2 : &curPtr));

    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party) + ".dat";

    std::cout << keyFile << std::endl;
    int fd = openForWriting(keyFile);

    for (int j = 0; j < iterations; j++)
    {
        curPtr = startPtr;
        tmpPtr2 = tmpPtr1;
        genModelKey(m, &curPtr, party, &g, secfloat, (LlamaBase<u64> *)llama, 0);
        if (j == 0)
        {
            size_t keySz = curPtr - startPtr;
            padding = 4096 - (keySz % 4096);
            zeros = new char[padding];
            memset(zeros, 0, padding);
            keySz += padding;
            writeKeySz(keySzDir, modelName, keySz);
        }
        memcpy(curPtr, zeros, padding);
        curPtr += padding;
        writeKeyBuf(fd, curPtr - startPtr, startPtr);
    }
    assert(0 == fsync(fd) && "sync error!");
    close(fd);
    printf("Sleeping for %d seconds.\n", sleepInt);
    // sleep for a minute
    sleep(sleepInt);
    delete[] zeros;
    destroyGPURandomness();
}

int main(int argc, char *argv[])
{
    int party = atoi(argv[1]);
    auto experiment = std::string(argv[2]);
    auto keyDir = std::string(argv[3]);

    omp_set_num_threads(32);

    if (experiment.compare("CNN2") == 0)
    {
        int epochs = 1;
        int blocks = 1;
        int blockSz = 600; // 600
        int batchSz = 100;
        dealerE2E("CNN2", party, epochs, blocks, blockSz, batchSz, 28, 28, 1, true, true, keyDir, 300);
    }
    else if (experiment.compare("CNN3-5e") == 0)
    {
        int epochs = 5;
        int blocks = 20;
        int blockSz = 25;
        int batchSz = 100;
        dealerE2E("CNN3", party, epochs, blocks, blockSz, batchSz, 32, 32, 3, true, true, keyDir, 300);
    }
    else if (experiment.compare("CNN3-2e") == 0)
    {
        int epochs = 2;   // 5;
        int blocks = 20;  // 10;
        int blockSz = 25; // 50;
        int batchSz = 100;
        dealerE2E("CNN3", party, epochs, blocks, blockSz, batchSz, 32, 32, 3, true, true, keyDir, 300);
    }
    else if (experiment.compare("CNN2-loss") == 0)
    {
        int epochs = 1;
        int blocks = 60;
        int blockSz = 10;
        int batchSz = 100;
        dealerE2E("CNN2", party, epochs, blocks, blockSz, batchSz, 28, 28, 1, true, true, keyDir, 5);
    }
    else if (experiment.compare("CNN3-2e-loss") == 0)
    {
        int epochs = 2;
        int blocks = 50;
        int blockSz = 10;
        int batchSz = 100;
        dealerE2E("CNN3", party, epochs, blocks, blockSz, batchSz, 32, 32, 3, true, true, keyDir, 5);
    }
    else if (experiment.compare("P-VGG16") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        dealerPerf("P-VGG16", party, iterations, batchSz, 32, 32, 3, false, false, keyDir, 300);
    }
    else if (experiment.compare("P-AlexNet") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        dealerPerf("P-AlexNet", party, iterations, batchSz, 32, 32, 3, false, false, keyDir, 300);
    }
    else if (experiment.compare("P-LeNet") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        dealerPerf("P-LeNet", party, iterations, batchSz, 28, 28, 1, false, false, keyDir, 60);
    }
    else if (experiment.compare("P-SecureML") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        dealerPerf("P-SecureML", party, iterations, batchSz, 28, 28, 1, false, false, keyDir, 60);
    }
    else if (experiment.compare("AlexNet") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        dealerPerf("AlexNet", party, iterations, batchSz, 32, 32, 3, true, true, keyDir, 300);
    }
    else if (experiment.compare("ModelB") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        dealerPerf("ModelB", party, iterations, batchSz, 28, 28, 1, true, true, keyDir, 60);
    }
    else if (experiment.compare("CNN2-perf") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        dealerPerf("CNN2", party, iterations, batchSz, 28, 28, 1, true, true, keyDir, 60);
    }
    else if (experiment.compare("CNN3-perf") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        dealerPerf("CNN3", party, iterations, batchSz, 32, 32, 3, true, true, keyDir, 300);
    }
    else
    {
        assert(0 && "unknown experiment");
    }
    return 0;
}
