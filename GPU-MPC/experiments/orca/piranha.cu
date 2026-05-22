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

#include "cnn.h"
#include "backend/piranha.h"

int main(int argc, char *argv[])
{
    prngWeights.SetSeed(osuCrypto::toBlock(0, 0));
    using T = u64;
    auto modelName = std::string(argv[1]);
    printf("Model name=%s\n", modelName.data());
    auto model = getCNN<T>(modelName);
    int bw = 64;
    u64 scale = 24;
    u64 bs = 128;
    std::vector<u64> inpShape;
    if (modelName.compare("P-VGG16") == 0 || modelName.compare("P-AlexNet") == 0)
    {
        u64 shape[4] = {bs, 32, 32, 3};
        inpShape.insert(inpShape.end(), shape, shape + 4);
    }
    else if (modelName.compare("P-LeNet") == 0)
    {
        u64 shape[4] = {bs, 28, 28, 1};
        inpShape.insert(inpShape.end(), shape, shape + 4);
    }
    else
    {
        u64 shape[2] = {bs, 784};
        inpShape.insert(inpShape.end(), shape, shape + 2);
    }
    Tensor<T> inp(inpShape);
    inp.zero();
    model->init(scale, inp);
    model->zero();
    int role = atoi(argv[2]);
    int party = atoi(argv[3]);
    auto keyDir = argv[4];
    auto keyFileName = keyDir + modelName;
    printf("Key file=%s\n", keyFileName.data());
    // 0 is dealer, 1 is evaluator
    if (role == 0)
    {
        auto piranha = new PiranhaKeygen<T>(party, bw, scale, keyFileName);
        model->setBackend(piranha);
        model->optimize();
        inp.d_data = (T *)moveToGPU((u8 *)inp.data, inp.size() * sizeof(T), (Stats *)NULL);
        auto &activation = model->forward(inp);
        piranha->output(activation);
        piranha->close();
    }
    else
    {
        auto ip = argv[5];
        auto piranha = new Piranha<T>(party, ip, bw, (int)scale, keyFileName);
        model->setBackend(piranha);
        model->optimize();
        std::vector<u64> time;
        u64 commBytes;
        lseek(piranha->fd, 0, SEEK_SET);
        readKey(piranha->fd, piranha->keySize, piranha->startPtr, NULL);
        for (int i = 0; i < 11; i++)
        {
            piranha->keyBuf = piranha->startPtr;
            piranha->s.reset();
            piranha->peer->sync();
            auto commStart = piranha->peer->bytesSent() + piranha->peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            inp.d_data = (T *)moveToGPU((u8 *)inp.data, inp.size() * sizeof(T), &(piranha->s));
            auto &activation = model->forward(inp);
            piranha->output(activation);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            if (i > 0)
                time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
            auto commEnd = piranha->peer->bytesSent() + piranha->peer->bytesReceived();
            if (i == 0)
                commBytes = commEnd - commStart;
        }
        piranha->close();
        auto avgTime = std::reduce(time.begin(), time.end()) / (float)time.size();
        printf("Average time taken (microseconds)=%f\n", avgTime);
        printf("Comm (B)=%lu\n", commBytes);
        // assumes output/P0/inference exists
        auto inferenceDir = "output/P" + std::to_string(party) + "/inference/";
        std::ofstream statsFile(inferenceDir + modelName + ".txt");
        statsFile << "Average time taken (microseconds)=" << avgTime << std::endl;
        statsFile << "Comm (B)=" << commBytes << std::endl;
        statsFile.close();
    }
    return 0;
}
