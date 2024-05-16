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

#include "backend/orca.h"
#include "cnn.h"

#ifndef InfType
#define InfType u64
#endif

int main(int argc, char *argv[])
{
    sytorch_init();
    auto modelName = std::string(argv[1]);
    auto model = getCNN<InfType>(modelName);
    int bw = atoi(argv[2]);
    u64 scale = strtoul(argv[3], 0, 10);
    assert(bw <= 8 * sizeof(InfType));
    assert(scale < bw);
    std::vector<u64> inpShape;
    if (modelName.compare("VGG16") == 0 || modelName.compare("ResNet50") == 0 || modelName.compare("ResNet18") == 0)
    {
        u64 shape[4] = {1, 224, 224, 3};
        inpShape.insert(inpShape.end(), shape, shape + 4);
    }
    else if (modelName.compare("CNN3") == 0 || modelName.compare("AlexNet") == 0)
    {
        u64 shape[4] = {100, 32, 32, 3};
        inpShape.insert(inpShape.end(), shape, shape + 4);
    }
    else
    {
        u64 shape[4] = {100, 28, 28, 1};
        inpShape.insert(inpShape.end(), shape, shape + 4);
    }
    Tensor<InfType> inp(inpShape);
    inp.zero();
    model->init(scale, inp);
    model->zero();
    int role = atoi(argv[4]);
    int party = atoi(argv[5]);
    auto keyDir = argv[6];
    auto expName = modelName + "_" + std::to_string(bw) + "_" + std::to_string(scale);
    auto keyFileName = keyDir + expName;
    // 0 is dealer, 1 is evaluator
    if (role == 0)
    {
        auto fss = new OrcaKeygen<InfType>(party, bw, scale, keyFileName);
        model->setBackend(fss);
        model->optimize();
        inp.d_data = (InfType *)moveToGPU((u8 *)inp.data, inp.size() * sizeof(InfType), (Stats *)NULL);
        auto &activation = model->forward(inp);
        fss->output(activation);
        fss->close();
    }
    else
    {
        auto ip = argv[7];
        auto fss = new Orca<InfType>(party, ip, bw, (int)scale, keyFileName);
        model->setBackend(fss);
        model->optimize();
        std::vector<u64> time;
        u64 commBytes;
        lseek(fss->fd, 0, SEEK_SET);
        readKey(fss->fd, fss->keySize, fss->startPtr, NULL);
        for (int i = 0; i < 11; i++)
        {
            fss->keyBuf = fss->startPtr;
            fss->s.reset();
            fss->peer->sync();
            auto commStart = fss->peer->bytesSent() + fss->peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();
            inp.d_data = (InfType *)moveToGPU((u8 *)inp.data, inp.size() * sizeof(InfType), &(fss->s));
            auto &activation = model->forward(inp);
            fss->output(activation);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            if (i > 0)
                time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
            auto commEnd = fss->peer->bytesSent() + fss->peer->bytesReceived();
            if (i == 0)
                commBytes = commEnd - commStart;
        }
        fss->close();
        auto avgTime = std::reduce(time.begin(), time.end()) / (float)time.size();
        printf("Average time taken (microseconds)=%f\n", avgTime);
        printf("Comm (B)=%lu\n", commBytes);
        // assumes output/P0/inference exists
        auto inferenceDir = "output/P" + std::to_string(party) + "/inference/";
        std::ofstream statsFile(inferenceDir + expName + ".txt");
        statsFile << "Average time taken (microseconds)=" << avgTime << std::endl;
        statsFile << "Comm (B)=" << commBytes << std::endl;
        statsFile.close();
    }
    return 0;
}
