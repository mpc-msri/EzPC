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
#include <cstdio>
#include <fcntl.h>
#include <filesystem>
#include <omp.h>
#include <string>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include "datasets/gpu_data.h"

#include "nn/orca/gpu_layer.h"
#include "nn/orca/gpu_model.h"

#include "cnn.h"
#include "model_accuracy.h"

#include <sytorch/softmax.h>
#include <sytorch/backend/llama_base.h>

u64 *gpuSoftmax(int batchSz, int numClasses, int party, SigmaPeer *peer, u64 *d_I, u64 *labels, bool secfloat, LlamaBase<u64> *llama)
{
    Tensor4D<u64> inp(batchSz, numClasses, 1, 1);
    Tensor4D<u64> softmaxOp(batchSz, numClasses, 1, 1);

    size_t memSz = batchSz * numClasses * sizeof(u64);
    moveIntoCPUMem((u8 *)inp.data, (u8 *)d_I, memSz, NULL);
    gpuFree(d_I);
    if (secfloat)
    {
        softmax_secfloat(inp, softmaxOp, dcf::orca::global::scale, LlamaConfig::party);
    }
    else
    {
        pirhana_softmax(inp, softmaxOp, dcf::orca::global::scale);
    }
    for (int img = 0; img < batchSz; img++)
    {
        for (int c = 0; c < numClasses; c++)
        {
            softmaxOp(img, c, 0, 0) -= (labels[numClasses * img + c] * (((1LL << dcf::orca::global::scale)) / batchSz));
        }
    }
    reconstruct(inp.d1 * inp.d2, softmaxOp.data, 64);
    d_I = (u64 *)moveToGPU((u8 *)softmaxOp.data, memSz, NULL);
    return d_I;
}

void trainModel(dcf::orca::GPUModel<u64> *m, u8 **keyBuf, int party, SigmaPeer *peer, u64 *data, u64 *labels, AESGlobalContext *g, bool secfloat, LlamaBase<u64> *llama, int epoch)
{
    auto start = std::chrono::high_resolution_clock::now();
    size_t inpMemSz = m->inpSz * sizeof(u64);
    auto d_I = (u64 *)moveToGPU((u8 *)data, inpMemSz, &(m->layers[0]->s));
    u64 *d_O;
    for (int i = 0; i < m->layers.size(); i++)
    {
        m->layers[i]->readForwardKey(keyBuf);
        d_O = m->layers[i]->forward(peer, party, d_I, g);
        if (d_O != d_I)
            gpuFree(d_I);
        d_I = d_O;
    }
    checkCudaErrors(cudaDeviceSynchronize());
    d_I = gpuSoftmax(m->batchSz, m->classes, party, peer, d_I, labels, secfloat, llama);
    for (int i = m->layers.size() - 1; i >= 0; i--)
    {
        m->layers[i]->readBackwardKey(keyBuf, epoch);
        d_I = m->layers[i]->backward(peer, party, d_I, g, epoch);
    }
}

u64 getKeySz(std::string dir, std::string modelName)
{
    std::ifstream kFile(dir + modelName + ".txt");
    u64 keySz;
    kFile >> keySz;
    return keySz;
}

void rmWeights(std::string lossDir, int party, int l, int k)
{
    assert(std::filesystem::remove(lossDir + "weights_mask_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat"));
    assert(std::filesystem::remove(lossDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat"));
}

void evaluatorE2E(std::string modelName, std::string dataset, int party, std::string ip, std::string weightsFile, bool floatWeights, int epochs, int blocks, int blockSz, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir)
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPUMemPool();
    initGPURandomness();
    initCPURandomness();
    assert(epochs < 6);

    omp_set_num_threads(2);

    auto dealerFifoName = "/tmp/dealerFifo" + std::to_string(party);
    auto evalFifoName = "/tmp/evaluatorFifo" + std::to_string(party);

    int dealerFifo, evalFifo;
    bool sync = ((epochs * blocks) > 1);
    if (sync)
    {
        mkfifo(dealerFifoName.c_str(), 0666);
        mkfifo(evalFifoName.c_str(), 0666);
        // evaluator reads from this fifo
        printf("Opening dealer fifo=%s\n", dealerFifoName.data());
        dealerFifo = open(dealerFifoName.c_str(), O_RDONLY);
        // evalutor writes to this fifo
        printf("Opening evaluator fifo=%s\n", evalFifoName.data());
        evalFifo = open(evalFifoName.c_str(), O_WRONLY);
    }
    char one = 1;
    char two = 2;

    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto expName = modelName + "-" + std::to_string(epochs) + "e-" + std::to_string(blocks) + "b";
    auto lossDir = trainingDir + "loss/" + expName + "/";
    auto weightsDir = lossDir + "weights/";
    auto keySzDir = trainingDir + "keysize/";
    std::ofstream lossFile(lossDir + "loss.txt");
    std::ofstream accFile(lossDir + "accuracy.txt");

    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);
    m->initWeights(weightsFile, floatWeights);
    Dataset d = readDataset(dataset, party);

    u8 *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf;
    u64 keySz = getKeySz(keySzDir, modelName);
    getAlignedBuf(&keyBuf1, keySz);
    getAlignedBuf(&keyBuf2, keySz);
    int curBuf = 0;
    curKeyBuf = keyBuf1;
    nextKeyBuf = keyBuf2;

    SigmaPeer *peer = new GpuPeer(false);
    LlamaBase<u64> *llama = nullptr;

    // automatically truncates by scale
    LlamaConfig::party = party + 2;
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&curKeyBuf);
    else
        llama->initClient(ip, (char **)&curKeyBuf);
    peer->peer = LlamaConfig::peer;

    if (secfloat)
        secfloat_init(party + 1, ip);

    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party) + ".dat";
    dropOSPageCache();
    std::chrono::duration<int64_t, std::milli> onlineTime = std::chrono::duration<int64_t, std::milli>::zero();
    std::chrono::duration<int64_t, std::milli> computeTime = std::chrono::duration<int64_t, std::milli>::zero();
    uint64_t keyReadTime = 0;
    size_t commBytes = 0;
    for (int l = 0; l < epochs; l++)
    {
        for (int k = 0; k < blocks; k++)
        {
            if (sync)
            {
                // evaluator reads from one and write to two
                assert(1 == read(dealerFifo, &one, sizeof(char)));
                assert(one == 1);
            }
            int fd = openForReading(keyFile);
            printf("Iteration=%u\n", l * blocks * blockSz + k * blockSz);
            // uncomment for end to end run
            peer->sync();
            auto startComm = peer->bytesSent() + peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            readKey(fd, keySz, curKeyBuf, &keyReadTime);
            for (int j = 0; j < blockSz; j++)
            {
#pragma omp parallel num_threads(2)
                {
#pragma omp sections
                    {
#pragma omp section
                        {
                            if (j < blockSz - 1)
                                readKey(fd, keySz, nextKeyBuf, &keyReadTime);
                        }
#pragma omp section
                        {
                            peer->sync();
                            auto computeStart = std::chrono::high_resolution_clock::now();
                            auto labelsIdx = (k * blockSz + j) * batchSz * d.classes;
                            int dataIdx = (k * blockSz + j) * d.H * d.W * d.C * batchSz;
                            trainModel(m, &curKeyBuf, party, peer, &(d.data[dataIdx]), &(d.labels[labelsIdx]), &g, secfloat, llama, l);
                            auto computeEnd = std::chrono::high_resolution_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);
                            computeTime += elapsed;
                        }
                    }
                }
                curKeyBuf = curBuf == 0 ? keyBuf2 : keyBuf1;
                nextKeyBuf = curBuf == 0 ? keyBuf1 : keyBuf2;
                curBuf = (curBuf + 1) % 2;
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            onlineTime += elapsed;
            printf("Online time (ms): %lu\n", elapsed.count());

            auto endComm = peer->bytesSent() + peer->bytesReceived();
            commBytes += (endComm - startComm);
            close(fd);
            if (sync)
            {
                // evaluator writes to two
                assert(1 == write(evalFifo, &two, sizeof(char)));
            }
            if (party == SERVER0)
            {
                m->dumpWeights(weightsDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat");
                std::pair<double, double> res;
                if (dataset == "mnist")
                {
                    res = getLossMNIST<i64>(modelName, (u64)dcf::orca::global::scale, weightsDir, party, l, k);
                }
                else
                {
                    res = getLossCIFAR10<i64>(modelName, (u64)dcf::orca::global::scale, weightsDir, party, l, k);
                }
                auto accuracy = res.first;
                auto loss = res.second;
                printf("Accuracy=%lf, Loss=%lf\n", accuracy, loss);
                lossFile << loss << std::endl;
                accFile << accuracy << std::endl;
                rmWeights(weightsDir, party, l, k);
            }
        }
    }
    LlamaConfig::peer->close();
    int iterations = epochs * blocks * blockSz;
    commBytes += secFloatComm;
    std::ofstream stats(trainingDir + expName + ".txt");
    auto statsString = "Total time taken (ms): " + std::to_string(onlineTime.count()) + "\nTotal bytes communicated: " + std::to_string(commBytes) + "\nSecfloat softmax bytes: " + std::to_string(secFloatComm);

    auto avgKeyReadTime = (double)keyReadTime / (double)iterations;
    auto avgComputeTime = (double)computeTime.count() / (double)iterations;

    double commPerIt = (double)commBytes / (double)iterations;
    statsString += "\nAvg key read time (ms): " + std::to_string(avgKeyReadTime) + "\nAvg compute time (ms): " + std::to_string(avgComputeTime);
    statsString += "\nComm per iteration (B): " + std::to_string(commPerIt);
    stats << statsString;
    stats.close();
    std::cout << statsString << std::endl;
    if (sync)
    {
        close(dealerFifo);
        close(evalFifo);
    }
    lossFile.close();
    accFile.close();
    destroyCPURandomness();
    destroyGPURandomness();
}

void evaluatorPerf(std::string modelName, std::string dataset, int party, std::string ip, int iterations, int batchSz, int H, int W, int C, bool secfloat, bool momentum, std::string keyDir)
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPUMemPool();
    initGPURandomness();
    initCPURandomness();

    omp_set_num_threads(2);

    dcf::orca::GPUModel<u64> *m = getGPUModel<u64>(modelName, Tensor<u64>(nullptr, {(u64)batchSz, (u64)H, (u64)W, (u64)C}));
    m->setTrain(momentum);
    size_t inpMemSz = m->inpSz * sizeof(u64);
    auto inp = (u64 *)cpuMalloc(inpMemSz);
    memset(inp, 0, inpMemSz);
    size_t opMemSz = m->batchSz * m->classes * sizeof(u64);
    auto labels = (u64 *)cpuMalloc(opMemSz);
    memset(labels, 0, opMemSz);

    u8 *keyBuf1, *keyBuf2, *curKeyBuf, *nextKeyBuf;
    auto trainingDir = "output/P" + std::to_string(party) + "/training/";
    auto keySzDir = trainingDir + "keysize/";
    u64 keySz = getKeySz(keySzDir, modelName);
    getAlignedBuf(&keyBuf1, keySz);
    getAlignedBuf(&keyBuf2, keySz);
    int curBuf = 0;
    curKeyBuf = keyBuf1;
    nextKeyBuf = keyBuf2;

    SigmaPeer *peer = new GpuPeer(false);
    LlamaBase<u64> *llama = nullptr;

    LlamaConfig::party = party + 2;
    llama = new LlamaBase<u64>();
    if (LlamaConfig::party == SERVER)
        llama->initServer(ip, (char **)&curKeyBuf);
    else
        llama->initClient(ip, (char **)&curKeyBuf);
    peer->peer = LlamaConfig::peer;

    if (secfloat)
        secfloat_init(party + 1, ip);

    std::string keyFile = keyDir + modelName + "_training_key" + std::to_string(party) + ".dat";
    dropOSPageCache();
    std::chrono::duration<int64_t, std::milli> onlineTime = std::chrono::duration<int64_t, std::milli>::zero();
    std::chrono::duration<int64_t, std::milli> computeTime = std::chrono::duration<int64_t, std::milli>::zero();
    uint64_t keyReadTime = 0;
    size_t commBytes = 0;
    int fd = openForReading(keyFile);
    auto startComm = peer->bytesSent() + peer->bytesReceived();
    readKey(fd, keySz, curKeyBuf, &keyReadTime);
    peer->sync();
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < iterations; j++)
    {
#pragma omp parallel num_threads(2)
        {
#pragma omp sections
            {
#pragma omp section
                {
                    if (j < iterations - 1)
                        readKey(fd, keySz, nextKeyBuf, &keyReadTime);
                }
#pragma omp section
                {
                    peer->sync();
                    auto computeStart = std::chrono::high_resolution_clock::now();
                    trainModel(m, &curKeyBuf, party, peer, inp, labels, &g, secfloat, llama, 0);
                    auto computeEnd = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(computeEnd - computeStart);
                    computeTime += elapsed;
                }
            }
        }
        if (j == iterations - 2)
        {
            auto end = std::chrono::high_resolution_clock::now();
            onlineTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            printf("Online time (ms): %lu\n", onlineTime.count());
        }
        curKeyBuf = curBuf == 0 ? keyBuf2 : keyBuf1;
        nextKeyBuf = curBuf == 0 ? keyBuf1 : keyBuf2;
        curBuf = (curBuf + 1) % 2;
    }
    auto endComm = peer->bytesSent() + peer->bytesReceived();
    commBytes += (endComm - startComm);
    close(fd);
    commBytes += secFloatComm;
    LlamaConfig::peer->close();
    std::ofstream stats(trainingDir + modelName + ".txt");
    auto statsString = "\n" + modelName + "\n";
    statsString += "Total time taken (ms): " + std::to_string(onlineTime.count()) + "\nTotal bytes communicated: " + std::to_string(commBytes) + "\nSecfloat softmax bytes: " + std::to_string(secFloatComm);
    statsString += "\nIterations: " + std::to_string(iterations) + "\n";
    auto totTimeByIt = (double)onlineTime.count() / (double)(iterations - 1);
    auto avgKeyReadTime = (double)keyReadTime / (double)iterations;
    auto avgComputeTIme = (double)computeTime.count() / (double)iterations;

    double commPerIt = (double)commBytes / (double)iterations;
    statsString += "\nTotal time / iterations (ms): " + std::to_string(totTimeByIt) + "\nAvg key read time (ms): " + std::to_string(avgKeyReadTime) + "\nAvg compute time (ms): " + std::to_string(avgComputeTIme);
    statsString += "\nComm per iteration (B): " + std::to_string(commPerIt) + "\n";
    stats << statsString;
    stats.close();
    std::cout << statsString << std::endl;
    destroyCPURandomness();
    destroyGPURandomness();
}

int main(int argc, char *argv[])
{
    sytorch_init();
    int party = atoi(argv[1]);
    auto ip = argv[2];
    auto experiment = std::string(argv[3]);
    auto keyDir = std::string(argv[4]);
    using T = u64;
    // Neha: need to fix this later
    if (experiment.compare("CNN2") == 0)
    {
        int epochs = 1;
        int blocks = 1;
        int blockSz = 600;
        int batchSz = 100;
        evaluatorE2E("CNN2", "mnist", party, ip, "weights/CNN2.dat", false, epochs, blocks, blockSz, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("CNN3-2e") == 0)
    {
        int epochs = 2;   // 2
        int blocks = 20;  // 10
        int blockSz = 25; // 50
        int batchSz = 100;
        evaluatorE2E("CNN3", "cifar10", party, ip, "weights/CNN3-2e.dat", true, epochs, blocks, blockSz, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("CNN3-5e") == 0)
    {
        int epochs = 5;
        int blocks = 20;
        int blockSz = 25;
        int batchSz = 100;
        evaluatorE2E("CNN3", "cifar10", party, ip, "weights/CNN3-2e.dat", true, epochs, blocks, blockSz, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("CNN2-loss") == 0)
    {
        int epochs = 1;
        int blocks = 60;
        int blockSz = 10;
        int batchSz = 100;
        evaluatorE2E("CNN2", "mnist", party, ip, "weights/CNN2.dat", false, epochs, blocks, blockSz, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("CNN3-2e-loss") == 0)
    {
        int epochs = 2;
        int blocks = 50;
        int blockSz = 10;
        int batchSz = 100;
        evaluatorE2E("CNN3", "cifar10", party, ip, "weights/CNN3.dat", false, epochs, blocks, blockSz, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("CNN2-perf") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        evaluatorPerf("CNN2", "mnist", party, ip, iterations, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else if (experiment.compare("CNN3-perf") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        evaluatorPerf("CNN3", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("P-VGG16") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-VGG16", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, false, false, keyDir);
    }
    else if (experiment.compare("P-AlexNet") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-AlexNet", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, false, false, keyDir);
    }
    else if (experiment.compare("P-LeNet") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-LeNet", "mnist", party, ip, iterations, batchSz, 28, 28, 1, false, false, keyDir);
    }
    else if (experiment.compare("P-SecureML") == 0)
    {
        int iterations = 11;
        int batchSz = 128;
        evaluatorPerf("P-SecureML", "mnist", party, ip, iterations, batchSz, 28, 28, 1, false, false, keyDir);
    }
    else if (experiment.compare("AlexNet") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        evaluatorPerf("AlexNet", "cifar10", party, ip, iterations, batchSz, 32, 32, 3, true, true, keyDir);
    }
    else if (experiment.compare("ModelB") == 0)
    {
        int iterations = 11;
        int batchSz = 100;
        evaluatorPerf("ModelB", "mnist", party, ip, iterations, batchSz, 28, 28, 1, true, true, keyDir);
    }
    else
    {
        assert(0 && "unknown experiment");
    }

    return 0;
}
