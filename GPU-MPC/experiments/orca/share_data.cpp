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

#include <sytorch/tensor.h>
#include <sytorch/backend/llama_base.h>
#include "datasets/cifar10.h"
#include "datasets/mnist.h"
#include "utils/gpu_file_utils.h"

void writeSharesCpu(std::ostream &f1, std::ostream &f2, int bw, int N, u64 *A)
{
    u64 *A0 = new u64[N];
    u64 *A1 = new u64[N];
    for (int i = 0; i < N; i++)
    {
        auto shares = splitShare(A[i], bw);
        A0[i] = shares.first;
        A1[i] = shares.second;
    }

    f1.write((char *)A0, N * sizeof(u64));
    f2.write((char *)A1, N * sizeof(u64));
    delete[] A0;
    delete[] A1;
}

std::pair<u64 *, int> readCifar10Labels()
{
    int N = 50000;
    u64 *data = new u64[N * 10];
    auto rawData = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        uint64_t label = rawData.training_labels[i];
        data[10 * i + label] = 1;
        for (int j = 0; j < 10; j++)
        {
            if (j != label)
            {
                data[10 * i + j] = 0;
            }
        }
    }
    return std::make_pair(data, N * 10);
}

std::pair<u64 *, int> readCifar10(int scale)
{
    int N = 50000, H = 32, W = 32, C = 3;
    u64 *data = new u64[N * H * W * C];
    auto rawData = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                for (int l = 0; l < C; l++)
                {
                    data[i * H * W * C + j * W * C + k * C + l] = (rawData.training_images[i][l * H * W + j * W + k] / 255.0) * (1LL << (scale));
                }
            }
        }
    }
    return std::make_pair(data, N * H * W * C);
}

void shareCifar10Labels(int bw)
{
    auto res = readCifar10Labels();
    auto data = res.first;
    auto N = res.second;
    std::ofstream f1("./datasets/shares/cifar10/cifar10_labels1.dat"), f2("./datasets/shares/cifar10/cifar10_labels2.dat");
    writeSharesCpu(f1, f2, bw, N, data);
}

void shareCifar10Data(int bw)
{
    auto res = readCifar10(24);
    auto data = res.first;
    auto N = res.second;
    std::ofstream f1("./datasets/shares/cifar10/cifar10_share1.dat"), f2("./datasets/shares/cifar10/cifar10_share2.dat");
    writeSharesCpu(f1, f2, bw, N, data);
}

std::pair<u64 *, int> readMnist(int scale)
{
    int N = 60000, H = 28, W = 28, C = 1;
    u64 *data = new u64[N * H * W * C];
    load_mnist();
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int k = 0; k < W; k++)
            {
                for (int l = 0; l < C; l++)
                {
                    data[i * H * W * C + j * W * C + k * C + l] = train_image[i][j * 28 + k] * (1ULL << scale);
                }
            }
        }
    }
    return std::make_pair(data, N * H * W * C);
}

std::pair<u64 *, int> readMnistLabels()
{
    int N = 60000;
    u64 *data = new u64[N * 10];
#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        int label = train_label[i];
        data[10 * i + label] = 1;
        for (int j = 0; j < 10; j++)
        {
            if (j != label)
            {
                data[10 * i + j] = 0;
            }
        }
    }
    return std::make_pair(data, N * 10);
}

void shareMnistLabels(int bw)
{
    auto res = readMnistLabels();
    auto data = res.first;
    auto N = res.second;
    std::ofstream f1("./datasets/shares/mnist/mnist_labels1.dat"), f2("./datasets/shares/mnist/mnist_labels2.dat");
    writeSharesCpu(f1, f2, bw, N, data);
}

void shareMnistData(int bw)
{
    auto res = readMnist(24);
    auto data = res.first;
    auto N = res.second;
    std::ofstream f1("./datasets/shares/mnist/mnist_share1.dat"), f2("./datasets/shares/mnist/mnist_share2.dat");
    writeSharesCpu(f1, f2, bw, N, data);
}

int main()
{
    omp_set_num_threads(64);
    
    LlamaConfig::party = DEALER;
    auto llama = new LlamaBase<u64>();
    llama->initPrngs();

    makeDir("./datasets/shares");
    makeDir("./datasets/shares/mnist");
    makeDir("./datasets/shares/cifar10");
    shareMnistData(64);
    shareMnistLabels(64);
    shareCifar10Data(64);
    shareCifar10Labels(64);
    return 0;
}