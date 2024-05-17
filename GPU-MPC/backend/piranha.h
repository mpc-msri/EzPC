// Author: Neha Jawalkar
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

#include "utils/gpu_random.h"
#include "utils/gpu_mem.h"

#include "orca.h"
// pin all the weights and activations in cpu memory

template <typename T>
void piranhaOpt(LayerGraphNode<T> *n, LayerGraphNode<T> *r)
{
    if (!n->layer->isTrainingMode && n->children.size() == 0)
    {
        n->layer->doTruncationForward = false;
    }
    if (n->layer->name == "AvgPool2D" && n->children[0]->layer->name == "ReLU")
    {
        auto avgPool = static_cast<AvgPool2D<T> *>(n->layer);
        auto relu = n->children[0]->layer;
        relu->mode = (int)std::log2(avgPool->ks * avgPool->ks);
    }
    else if (n->layer->name == "Flatten")
    {
        // delete flatten and add a flag to FC instead
        assert(n->parents.size() == 1 && n->children.size() == 1);
        auto parent = n->parents[0];
        // printf("%s\n", parent->layer->name.data());
        auto child = n->children[0];
        assert(parent->children.size() == 1);
        assert(child->parents.size() == 1);
        parent->children[0] = child;
        child->parents[0] = parent;
        always_assert(parent->currTensor->shape.size() == 4);
        always_assert(child->layer->name == "FC");
        auto fc = static_cast<FC<T> *>(child->layer);
        // // todo: free the memory used up by flatten
        auto batchSz = parent->currTensor->shape[0];
        auto h = parent->currTensor->shape[1];
        auto w = parent->currTensor->shape[2];
        auto c = parent->currTensor->shape[3];
        int m = fc->out;
        assert(h * w * c == fc->in);
        parent->currTensor = new Tensor<T>(parent->layer->activation.data, parent->layer->activation.d_data, {batchSz, h * w * c});
        // printf("New tensor=%lx\n", parent->currTensor);
        parent->currTensor->graphNode = parent;
        int i;
        for (i = 0; i < n->allNodesInExecutionOrderRef->size(); i++)
        {
            if (n->allNodesInExecutionOrderRef->at(i) == n)
            {
                break;
            }
        }
        n->allNodesInExecutionOrderRef->erase(n->allNodesInExecutionOrderRef->begin() + i);
    }
}

template <typename T>
class Piranha : public Orca<T>
{
public:
    Piranha() : Orca<T>() {}

    Piranha(int party, std::string ip, int bw, int scale, std::string keyFile = "") : Orca<T>(party, ip, bw, scale, keyFile)
    {
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode)
    {
        // assert(mode == 2);
        auto start = std::chrono::high_resolution_clock::now();
        auto k = dcf::readTwoRoundReluKey<T>(&(this->keyBuf));
        auto d_temp = dcf::gpuTwoRoundRelu(this->peer, this->party, k, in.d_data, &(this->g), &(this->s));
        auto d_drelu = d_temp.first;
        gpuFree(d_drelu);
        out.d_data = d_temp.second;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        this->s.relu_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto d_inp = in.d_data;
        GPUTruncateKey<T> k;
        in.d_data = gpuTruncate<T, T>(this->bw, this->bw, TruncateType::LocalARS, k, (int)shift, this->peer, (int)this->party, (int)in.size(), (T *)in.d_data, &(this->g), &(this->s));
        gpuFree(d_inp);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        this->s.truncate_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { piranhaOpt<T>(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
    }
};

template <typename T>
class PiranhaKeygen : public OrcaKeygen<T>
{
public:
    PiranhaKeygen(int party, int bw, int scale, std::string keyFile) : OrcaKeygen<T>(party, bw, scale, keyFile)
    {
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode)
    {
        assert(in.is_same_shape(out));
        assert(in.is_same_shape(drelu));
        // assert(mode == 2);
        int tmpBw = this->bw - scale - mode;
        // printf("Inp=%lx, mode=%d, N=%lu\n", in.d_data, mode, in.size());
        auto d_tempMask = dcf::gpuGenTwoRoundReluKey(&(this->keyBuf), this->party, tmpBw, this->bw, in.size(), in.d_data, &(this->g));
        auto d_dreluMask = d_tempMask.first;
        gpuFree(d_dreluMask);
        auto d_reluMask = d_tempMask.second;
        out.d_data = d_reluMask;
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        auto d_inp = in.d_data;
        in.d_data = genGPUTruncateKey<T, T>(&(this->keyBuf), this->party, TruncateType::LocalARS, this->bw, this->bw, shift, in.size(), in.d_data, &(this->g));
        gpuFree(d_inp);
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { piranhaOpt<T>(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
    }
};