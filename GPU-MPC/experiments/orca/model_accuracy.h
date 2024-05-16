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

#include "datasets/cifar10.h"
#include "datasets/mnist.h"
#include "utils/gpu_file_utils.h"
#include "cnn.h"
#include "nn/orca_opt.h"

template <typename T>
class LossBackend : public ClearText<T>
{
public:

    LossBackend() : ClearText<T>()
    {
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { orcaOpt(n, r); });
    }
};

template <typename T>
void softmax(u64 scale, const Tensor2D<T> &in, Tensor2D<double> &out)
{
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(std::is_integral<T>::value || (scale == 0));

    auto batchSz = in.d1;
    auto numClasses = in.d2;
    for (int b = 0; b < batchSz; ++b)
    {
        T max = in(b, 0);
        for (u64 j = 1; j < numClasses; ++j)
        {
            if (in(b, j) > max)
            {
                max = in(b, j);
            }
        }
        double den = 0.0;
        double exps[numClasses];
        for (u64 j = 0; j < numClasses; ++j)
        {
            double x = in(b, j) - max;
            exps[j] = std::exp(x / (1ULL << scale));
            den += exps[j];
        }

        for (u64 j = 0; j < numClasses; ++j)
        {
            out(b, j) = exps[j] / den;
        }
    }
}

template <typename T>
void readWeights(SytorchModule<T> *m, std::string lossDir, int party, int l, int k)
{
    size_t weightsSize;
    auto maskedWFile = lossDir + "masked_weights_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat";
    auto masked_W = (u64 *)readFile(maskedWFile, &weightsSize);
    u64 *mask_W = NULL;
    auto wMaskFile = lossDir + "weights_mask_" + std::to_string(party) + "_" + std::to_string(l) + "_" + std::to_string(k) + ".dat";
    mask_W = (u64 *)readFile(wMaskFile, &weightsSize);
    printf("Weights file=%s, %s\n", maskedWFile.data(), wMaskFile.data());
    int N = weightsSize / sizeof(u64);
    // printf("%d\n", N);
    auto W = new u64[N];
    for (int i = 0; i < N; i++)
    {
        W[i] = masked_W[i] - (mask_W ? mask_W[i] : 0);
        // if (i < 10)
        //     printf("%ld, %ld, %ld\n", masked_W[i], mask_W[i], W[i]);
    }
    int wIdx = 0;
    std::vector<Layer<T> *> layers;
    for (int i = 0; i < m->allNodesInExecutionOrder.size(); i++)
        layers.push_back(m->allNodesInExecutionOrder[i]->layer);
    for (int i = 0; i < layers.size(); i++)
    {
        if (layers[i]->name.find("Conv2D") != std::string::npos || layers[i]->name.find("FC") != std::string::npos)
        {
            auto weights = layers[i]->getweights();
            memcpy(weights.data, &W[wIdx], weights.size * sizeof(T));
            wIdx += weights.size;
            auto bias = layers[i]->getbias();
            memcpy(bias.data, &W[wIdx], bias.size * sizeof(T));
            wIdx += bias.size;
        }
    }
    assert(wIdx == N);
}

template <typename T>
std::pair<double, double> getLossCIFAR10(std::string modelName, u64 scale, std::string lossDir, int party, int l, int k)
{
    auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();
    const u64 testLen = dataset.test_images.size();
    int batchSz = 100;

    auto m = getCNN<T>(modelName);
    Tensor<T> testSet({(u64)batchSz, 32, 32, 3});
    m->init(scale, testSet);
    m->train();
    auto dummy = new LossBackend<T>();
    dummy->bw = 8 * sizeof(T);
    dummy->probablistic = true;
    dummy->localTruncationEmulation = false;
    m->setBackend(dummy);
    m->optimize();

    Tensor2D<double> e(batchSz, 10);
    auto testSet_4d = testSet.as_4d();

    readWeights(m, lossDir, party, l, k);

    // printf("Test len=%d\n", testLen);
    double loss = 0.0;
    u64 correct = 0;
    for (u64 i = 0; i < testLen; i += batchSz)
    {
        for (u64 b = 0; b < batchSz; ++b)
        {
            for (u64 j = 0; j < 32; ++j)
            {
                for (u64 k = 0; k < 32; ++k)
                {
                    for (u64 l = 0; l < 3; ++l)
                    {
                        testSet_4d(b, j, k, l) = (T)((dataset.test_images[i + b][j * 32 + k + l * 32 * 32] / 255.0) * (1LL << (scale)));
                    }
                }
            }
        }
        auto &activation = m->forward(testSet);
        auto act_2d = activation.as_2d();
        softmax<T>(scale, act_2d, e);
        for (u64 b = 0; b < batchSz; b++)
        {
            assert(act_2d.argmax(b) == e.argmax(b));
            if (e.argmax(b) == dataset.test_labels[i + b])
            {
                correct++;
            }
            loss += (double)std::log(e(b, dataset.test_labels[i + b]));
        }
        // printf("correct %d: %d\n", i, correct);
    }
    auto accuracy = (correct * 100.0) / testLen;
    loss = -loss / testLen;
    // printf("accuracy: %lf, loss: %lf\n", accuracy, loss);
    return std::make_pair(accuracy, loss);
}

template <typename T>
std::pair<double, double> getLossMNIST(std::string modelName, u64 scale, std::string lossDir, int party, int l, int k)
{
    load_mnist();
    const u64 testLen = 10000;
    auto m = getCNN<T>(modelName);
    Tensor<T> testSet({testLen, 28, 28, 1});
    m->init(scale, testSet);
    m->train();
    auto dummy = new LossBackend<T>();
    dummy->bw = 8 * sizeof(T);
    dummy->probablistic = true;
    dummy->localTruncationEmulation = false;
    m->setBackend(dummy);
    m->optimize();
    auto testSet_4d = testSet.as_4d();

    Tensor2D<double> e(testLen, 10);
    for (u64 i = 0; i < testLen; ++i)
    {
        for (u64 j = 0; j < 28; ++j)
        {
            for (u64 k = 0; k < 28; ++k)
            {
                testSet_4d(i, j, k, 0) = (T)(test_image[i][j * 28 + k] * (1LL << scale));
            }
        }
    }

    double loss = 0.0;
    readWeights(m, lossDir, party, l, k);
    auto &activation = m->forward(testSet);
    auto act_2d = activation.as_2d();
    softmax<T>(scale, act_2d, e);
    u64 correct = 0;
    for (u64 i = 0; i < testLen; i++)
    {
        assert(act_2d.argmax(i) == e.argmax(i));
        // printf("%d, %d\n", m->activation.argmax(i), test_label[i]);
        if (e.argmax(i) == test_label[i])
        {
            correct++;
        }
        loss += (double)std::log(e(i, test_label[i]));
        // if (i % 100 == 0)
        //     printf("correct: %d\n", correct);
    }
    auto accuracy = (correct * 100.0) / testLen;
    loss = -loss / testLen;
    return std::make_pair(accuracy, loss);
}