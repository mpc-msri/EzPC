// Authors: Kanav Gupta, Neha Jawalkar
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

#pragma once

#include <sytorch/tensor.h>
#include <sytorch/layers/layers.h>
#include <sytorch/backend/backend.h>
#include <fstream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <sytorch/backend/default.h>

template <typename T>
class SytorchModule
{
public:
    Tensor<T> activation;
    Backend<T> *backend = nullptr;
    LayerGraphNode<T> *root = nullptr;
    bool debug = false;
    u64 scale;

    std::vector<LayerGraphNode<T> *> allNodesInExecutionOrder;
    const std::vector<std::string> functionalLayers = {"Add", "Concat", "GeLU", "SoftMax", "Split", "View", "Transpose", "_MatMul", "_ScalarMul", "ReLU", "AttentionMask", "LocalAttentionMask", "_Tanh", "Unsqueeze", "SoftMaxTriangular", "_MatMulTriangular", "AttentionTriangular", "_Mul", "SiLU", "_ScalarDiv", "RotaryEmbedding"};
    static std::map<std::string, LayerGraphNode<T> *> functionalLayerMap;

public:
    virtual Tensor<T> &_forward(Tensor<T> &input) = 0;

    SytorchModule() : activation({}), allNodesInExecutionOrder(0)
    {
        backend = defaultBackend<T>();
    }

    void generateFunctionalLayerMap()
    {
        // functionalLayerMap.clear();
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         {
                             if (std::find(functionalLayers.begin(), functionalLayers.end(), node->layer->name) != functionalLayers.end())
                             {
                                 std::string id = node->layer->name;
                                 for (auto &parent : node->parents)
                                 {
                                     id += "|" + std::to_string((uint64_t)(parent));
                                 }
                                 id = id + "|" + node->layer->paramstring;
                                 // make sure it already doesn't exist
                                 always_assert(functionalLayerMap.find(id) == functionalLayerMap.end());
                                 functionalLayerMap[id] = node;
                             } });
    }

    template <typename... Args>
    LayerGraphNode<T> *getFunctionalNode(const std::string &layerName, std::vector<Tensor<T> *> ips, Args... args)
    {
        std::string id = layerName;
        for (auto &ip : ips)
        {
            id += "|" + std::to_string((uint64_t)(ip->graphNode));
        }
        id = id + "|" + paramstring(args...);
        if (functionalLayerMap.find(id) == functionalLayerMap.end())
        {
            std::cerr << "Layer not found = \"" << id << "\"" << std::endl;
            exit(1);
        }
        return functionalLayerMap[id];
    }

    template <typename LayerType, typename... Args>
    Tensor<T> &functionalGraphGen(std::vector<Tensor<T> *> arr, Args... args)
    {
        for (auto &a : arr)
        {
            always_assert(a->graphGenMode);
        }
        auto layer = new LayerType(args...);
        layer->paramstring = paramstring(args...);
        return layer->forward(arr);
    }

    void genGraphAndExecutionOrder(Tensor<T> &ip)
    {
        // Tensor<T> ip({});
        ip.graphGenMode = true;
        ip.graphNode = new LayerGraphNode<T>();
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        // ip.graphNode->currTensor = &ip;
        ip.graphNode->allNodesInExecutionOrderRef = &allNodesInExecutionOrder;
        auto &res = this->_forward(ip);
        std::vector<Tensor<T> *> a = {&res};
        if (shapesOkay(a))
            this->activation.resize(res.shape);
        ip.graphGenMode = false;
        root = ip.graphNode;
        ip.graphNode->currTensor = &ip;
    }

    void init(u64 scale, Tensor<T> &ip)
    {
        genGraphAndExecutionOrder(ip);
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         { node->layer->initScale(scale); });

        this->scale = scale;
        generateFunctionalLayerMap();
    }

    void init(u64 scale)
    {
        Tensor<T> ip({});
        init(scale, ip);
    }

    void randomize()
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         {
                             node->layer->getweights().randomize();
                             node->layer->getbias().randomize(); });
    }

    void zero()
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         {
                             node->layer->getweights().zero();
                             node->layer->getbias().zero(); });
    }

    void setBackend(Backend<T> *b)
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         { node->layer->setBackend(b); });
        backend = b;
    }

    Tensor<T> &forward(Tensor<T> &input)
    {
        // printf("%d, %lx, %d\n", input.graphGenMode, input.graphNode, debug);
        if (input.graphGenMode)
        {
            return this->_forward(input);
        }
        // input.print();
        if (input.graphNode == nullptr)
        { // when the module is a top level module
            topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                             { node->numUsages = 0; });
            input.graphNode = root;
            input.graphNode->currTensor = &input;
        }
        if (debug)
        {
            auto &res = this->_forward(input);
            this->activation.resize(res.shape);
            this->activation.copy(res);
            return this->activation;
        }
        else
        {
            int i = 0;
            // printf("Nodes in execution order=%d\n", allNodesInExecutionOrder.size());
            for (auto &n : allNodesInExecutionOrder)
            {
                // printf("%d=%s\n", i++, n->layer->name.data());
                std::vector<Tensor<T> *> a;
                for (auto p : n->parents)
                {
                    // printf("%lx\n\n\n", p->currTensor);
                    a.push_back(p->currTensor);
                }
                // printf("Input %s=%ld, %ld\n", n->layer->name.data(), a[0]->data[0], a[0]->data[a[0]->size() - 1]);
                n->layer->forward(a);
                n->currTensor->d_data = n->layer->activation.d_data;
                // printf("Output=%ld, %ld\n", n->layer->activation.data[0], n->layer->activation.data[n->layer->activation.size() - 1]);
            }
            auto l = allNodesInExecutionOrder.size();
            auto res = allNodesInExecutionOrder[l - 1]->currTensor; // todo: calculate using the generated graph
            this->activation.resize(res->shape);
            this->activation.copy(*res);
            this->activation.d_data = res->d_data;
            return this->activation;
        }
    }

    void optimize()
    {
        backend->optimize(root);
    }

    void load(const std::string weightsFile)
    {
        size_t size_in_bytes = std::filesystem::file_size(weightsFile);
        always_assert(size_in_bytes % 4 == 0); // as it's float
        size_t numParameters = size_in_bytes / 4;
        float *floatWeights = new float[numParameters];

        std::ifstream file(weightsFile, std::ios::binary);
        file.read((char *)floatWeights, size_in_bytes);
        file.close();
        u64 scale = this->scale;

        size_t wIdx = 0;
        int i = 0;
        for (auto &node : allNodesInExecutionOrder)
        {
            auto layer = node->layer;
            // printf("Loading weights layer %s\n", layer->name.data());
            if (layer->name == "_MHADummy")
            {
                auto mha = (_MHADummy<T> *)layer;
                if (mha->qkvLayout == "qkvconcat")
                {
                    for (u64 j = 0; j < mha->wQKV.size(); j++)
                    {
                        mha->wQKV.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                    }
                    wIdx += mha->wQKV.size();
                    for (u64 j = 0; j < mha->bQKV.size(); ++j)
                    {
                        mha->bQKV.data[j] = T(floatWeights[wIdx + j] * (1LL << (2 * scale)));
                    }
                    wIdx += mha->bQKV.size();
                }
                else if (mha->qkvLayout.find("sep") != std::string::npos)
                {
                    assert(mha->wQKV.d1 == mha->wQKV.d2 / 3);
                    // need this in qkvconcat format
                    Tensor2D<T> wK(mha->wQKV.d1, mha->wQKV.d2 / 3);
                    Tensor2D<T> wV(mha->wQKV.d1, mha->wQKV.d2 / 3);
                    Tensor2D<T> wQ(mha->wQKV.d1, mha->wQKV.d2 / 3);
                    if (mha->qkvLayout == "kvqsep")
                    {
                        for (u64 j = 0; j < wK.size(); j++)
                        {
                            wK.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                        }
                        wIdx += wK.size();

                        for (u64 j = 0; j < wV.size(); j++)
                        {
                            wV.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                        }
                        wIdx += wV.size();

                        for (u64 j = 0; j < wQ.size(); j++)
                        {
                            wQ.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                        }
                        wIdx += wQ.size();
                    }
                    else if (mha->qkvLayout == "qkvsep")
                    {
                        for (u64 j = 0; j < wQ.size(); j++)
                        {
                            wQ.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                        }
                        wIdx += wQ.size();

                        for (u64 j = 0; j < wK.size(); j++)
                        {
                            wK.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                        }
                        wIdx += wK.size();

                        for (u64 j = 0; j < wV.size(); j++)
                        {
                            wV.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                        }
                        wIdx += wV.size();
                    }
                    else
                    {
                        assert(0);
                    }
                    for (u64 j = 0; j < mha->wQKV.d1; j++)
                    {
                        for (u64 k = 0; k < mha->wQKV.d2 / 3; k++)
                        {
                            mha->wQKV(j, k) = wQ(j, k);
                            mha->wQKV(j, mha->wQKV.d2 / 3 + k) = wK(j, k);
                            mha->wQKV(j, 2 * mha->wQKV.d2 / 3 + k) = wV(j, k);
                        }
                    }
                    mha->bQKV.as_nd().zero();
                }
                for (u64 j = 0; j < mha->wProj.size(); j++)
                {
                    mha->wProj.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                }
                wIdx += mha->wProj.size();
                if (mha->qkvLayout == "qkvsep")
                {
                    mha->bProj.as_nd().zero();
                }
                else
                {
                    for (u64 j = 0; j < mha->bProj.size(); ++j)
                    {
                        mha->bProj.data[j] = T(floatWeights[wIdx + j] * (1LL << (2 * scale)));
                    }
                    wIdx += mha->bProj.size();
                }
            }
            else if (layer->name == "BatchNormInference")
            {
                auto bn = (BatchNormInference<T> *)layer;
                auto channel = bn->A.d1;
                auto gammaPtr = floatWeights + wIdx;
                auto betaPtr = floatWeights + wIdx + channel;
                auto meanPtr = floatWeights + wIdx + 2 * channel;
                auto varPtr = floatWeights + wIdx + 3 * channel;
                for (int j = 0; j < channel; ++j)
                {
                    bn->A(j) = T((gammaPtr[j] / std::sqrt(varPtr[j])) * (1LL << scale));
                    bn->B(j) = T((betaPtr[j] - gammaPtr[j] * meanPtr[j] / std::sqrt(varPtr[j])) * (1LL << (2 * scale)));
                }
                wIdx += 4 * channel;
            }
            else
            {
                auto weights = layer->getweights();
                for (u64 j = 0; j < weights.size; j++)
                {
                    weights.data[j] = T(floatWeights[wIdx + j] * (1LL << scale));
                }
                wIdx += weights.size;

                auto bias = layer->getbias();
                if (layer->useBias)
                {

                    for (u64 j = 0; j < bias.size; ++j)
                    {
                        bias.data[j] = T(floatWeights[wIdx + j] * (1LL << (2 * scale)));
                    }

                    wIdx += bias.size;
                }
                else
                {
                    bias.zero();
                }
            }
            i++;
        }
        printf("wIdx=%lu, numParameters=%lu\n", wIdx, numParameters);
        // always_assert(wIdx == numParameters);
        delete[] floatWeights;
    }

    void dumpi64(const std::string weightsFile)
    {
        std::ofstream file(weightsFile, std::ios::binary);
        u64 scale = this->scale;

        for (auto &node : allNodesInExecutionOrder)
        {
            auto layer = node->layer;
            if (layer->name == "BatchNormInference")
            {
                auto bn = (BatchNormInference<T> *)layer;
                auto channel = bn->A.d1;

                for (int j = 0; j < channel; ++j)
                {
                    i64 v = bn->A(j);
                    file.write((char *)(&v), sizeof(i64));
                }
                for (int j = 0; j < channel; ++j)
                {
                    i64 v = bn->B(j);
                    file.write((char *)(&v), sizeof(i64));
                }
                for (int j = 0; j < channel; ++j)
                {
                    i64 v = 0;
                    file.write((char *)(&v), sizeof(i64));
                }
                for (int j = 0; j < channel; ++j)
                {
                    i64 v = (1LL << scale);
                    file.write((char *)(&v), sizeof(i64));
                }
            }
            else
            {
                auto weights = layer->getweights();
                for (u64 j = 0; j < weights.size; j++)
                {
                    i64 v = weights.data[j];
                    file.write((char *)(&v), sizeof(i64));
                }

                auto bias = layer->getbias();
                if (layer->useBias)
                {

                    for (u64 j = 0; j < bias.size; ++j)
                    {
                        i64 v = bias.data[j];
                        file.write((char *)(&v), sizeof(i64));
                    }
                }
            }
        }
    }

    Tensor<T> &add(std::vector<Tensor<T> *> &arr)
    {
        if (arr[0]->graphGenMode)
        {
            auto &c = functionalGraphGen<Add<T>>(arr);
            return c;
        }

        auto cNode = getFunctionalNode("Add", arr);
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    template <typename... Args>
    Tensor<T> &add(Args &...args)
    {
        auto res = collect(args...);
        return add(res);
    }

    Tensor<T> &concat(std::vector<Tensor<T> *> &arr)
    {
        if (arr[0]->graphGenMode)
        {
            auto &c = functionalGraphGen<Concat<T>>(arr);
            return c;
        }

        auto cNode = getFunctionalNode("Concat", arr);
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    template <typename... Args>
    Tensor<T> &concat(Args &...args)
    {
        auto res = collect(args...);
        return concat(res);
    }

    Tensor<T> &gelu(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<GeLU<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("GeLU", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &silu(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<SiLU<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("SiLU", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &relu(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<ReLU<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("ReLU", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &unsqueeze(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<Unsqueeze<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("Unsqueeze", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &tanh(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<_Tanh<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("_Tanh", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &softmax(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<SoftMax<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("SoftMax", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &split(Tensor<T> &a, u64 n_splits)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<Split<T>>({&a}, n_splits);
            return c;
        }

        auto cNode = getFunctionalNode("Split", {&a}, n_splits);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &view(Tensor<T> &a, i64 idx)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<View<T>>({&a}, idx);
            return c;
        }

        auto cNode = getFunctionalNode("View", {&a}, idx);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &transpose(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<Transpose<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("Transpose", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &matmul(Tensor<T> &a, Tensor<T> &b)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<_MatMul<T>>({&a, &b});
            return c;
        }

        auto cNode = getFunctionalNode("_MatMul", {&a, &b});
        std::vector<Tensor<T> *> arr = {&a, &b};
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    Tensor<T> &matmul_triangular(Tensor<T> &a, Tensor<T> &b)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<_MatMulTriangular<T>>({&a, &b});
            return c;
        }

        auto cNode = getFunctionalNode("_MatMulTriangular", {&a, &b});
        std::vector<Tensor<T> *> arr = {&a, &b};
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    Tensor<T> &mul(Tensor<T> &a, Tensor<T> &b)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<_Mul<T>>({&a, &b});
            return c;
        }

        auto cNode = getFunctionalNode("_Mul", {&a, &b});
        std::vector<Tensor<T> *> arr = {&a, &b};
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    Tensor<T> &scalarmul(Tensor<T> &a, double scalar)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<_ScalarMul<T>>({&a}, scalar);
            return c;
        }

        auto cNode = getFunctionalNode("_ScalarMul", {&a}, scalar);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &scalardiv(Tensor<T> &a, double scalar)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<_ScalarDiv<T>>({&a}, scalar);
            return c;
        }

        auto cNode = getFunctionalNode("_ScalarDiv", {&a}, scalar);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &attention_mask(Tensor<T> &a, double scalar)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<AttentionMask<T>>({&a}, scalar);
            return c;
        }

        auto cNode = getFunctionalNode("AttentionMask", {&a}, scalar);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &local_attention_mask(Tensor<T> &a, double scalar)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<LocalAttentionMask<T>>({&a}, scalar);
            return c;
        }

        auto cNode = getFunctionalNode("LocalAttentionMask", {&a}, scalar);
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &softmax_triangular(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<SoftMaxTriangular<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("SoftMaxTriangular", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    Tensor<T> &attention_triangular(Tensor<T> &q, Tensor<T> &k, Tensor<T> &v, u64 n_heads)
    {
        if (q.graphGenMode)
        {
            auto &c = functionalGraphGen<AttentionTriangular<T>>({&q, &k, &v}, n_heads);
            return c;
        }

        auto cNode = getFunctionalNode("AttentionTriangular", {&q, &k, &v}, n_heads);
        std::vector<Tensor<T> *> arr = {&q, &k, &v};
        auto &c = cNode->layer->forward(arr);
        return c;
    }

    Tensor<T> &rotary_embedding(Tensor<T> &a)
    {
        if (a.graphGenMode)
        {
            auto &c = functionalGraphGen<RotaryEmbedding<T>>({&a});
            return c;
        }

        auto cNode = getFunctionalNode("RotaryEmbedding", {&a});
        auto &c = cNode->layer->forward(a);
        return c;
    }

    T invsqrt(double x)
    {
        double t = 1 / sqrt(x);
        return T(t * (1LL << scale));
    }

    void train()
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         { node->layer->train(); });
    }

    void eval()
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         { node->layer->eval(); });
    }
};

template <typename T>
std::map<std::string, LayerGraphNode<T> *> SytorchModule<T>::functionalLayerMap = std::map<std::string, LayerGraphNode<T> *>();
