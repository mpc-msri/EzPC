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

#include <sytorch/tensor.h>
#include <sytorch/graph.h>
#include "utils/helper_cuda.h"

template <typename T>
void orcaOpt(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
{
    // ReLU-MaxPool optimization
    if (node->layer->name == "ReLU")
    {
        auto child = node->children[0];
        auto cLayer = node->children[0]->layer;
        if (node->children.size() == 1 && child->parents.size() == 1 && cLayer->name == "MaxPool2D")
        {
            child->layer = node->layer;
            node->layer = cLayer;

            cLayer->node = node;
            child->layer->node = child;

            node->currTensor = &(node->layer->activation);
            child->layer->resize({node->layer->activation.shape});
            child->currTensor = &(child->layer->activation);
        }
    }

    // in LlamaImproved, mode takes the value according to the following rule:
    // 0: the layer takes as input \ell bits and outputs \ell bits
    // 1: the layer takes as input \ell bits and outputs \ell - scale bits
    // 2: the layer takes as input \ell - scale bits and outputs \ell bits
    // 3: the layer takes as input \ell - scale bits and outputs \ell - scale bits

    // std::cerr << "Visiting " << node->layer->name << std::endl;
    if (node->layer->name == "Conv2D" || node->layer->name == "FC" || node->layer->name == "BatchNorm2dInference" || node->layer->name == "AvgPool2D" || node->layer->name == "GlobalAvgPool2D")
    {
        // only one parent
        auto parent = node->parents[0];
        if (parent->layer->mode == 1 || parent->layer->mode == 3)
        {
            // std::cerr << "    Found parent " << parent->layer->name << " with mode " << parent->layer->mode << std::endl;
            node->layer->doPreSignExtension = true;
        }
        // printf("Num children=%d\n", node->children.size());
        if (!node->layer->isTrainingMode && node->children.size() == 0)
        {
            // this is the last node
            node->layer->mode = 0;
            node->layer->doTruncationForward = false;
        }
        else
        {
            node->layer->mode = 1;
            node->layer->forwardTruncationMode = 1;
        }
    }
    else if (node->layer->name == "Add" || node->layer->name == "Concat")
    {
        int m = 0;
        for (auto &parent : node->parents)
        {
            if ((parent->layer->mode % 2) == 1)
            {
                m = 3;
                break;
            }
        }
        node->layer->mode = m;
    }
    else if (node->layer->name == "Flatten")
    {
        // delete flatten and add a flag to FC instead
        assert(node->parents.size() == 1 && node->children.size() == 1);
        auto parent = node->parents[0];
        // printf("%s\n", parent->layer->name.data());
        auto child = node->children[0];
        assert(parent->children.size() == 1);
        assert(child->parents.size() == 1);
        parent->children[0] = child;
        child->parents[0] = parent;
        always_assert(parent->currTensor->shape.size() == 4);
        always_assert(child->layer->name == "FC");
        auto fc = static_cast<FC<T> *>(child->layer);
        // // todo: free the memory used up by flatten
        // printf("%d, %d, %d, %d\n", parent->layer->activation.shape[0], parent->layer->activation.shape[1], parent->layer->activation.shape[2], parent->layer->activation.shape[3]);
        auto batchSz = parent->currTensor->shape[0];
        auto h = parent->currTensor->shape[1];
        auto w = parent->currTensor->shape[2];
        auto c = parent->currTensor->shape[3];
        int m = fc->out;
        assert(h * w * c == fc->in);
        parent->currTensor = new Tensor<T>(parent->layer->activation.data, parent->layer->activation.d_data, {batchSz, h * w * c});
        // printf("New tensor=%lx\n", parent->currTensor);
        parent->currTensor->graphNode = parent;
        auto temp = Tensor(fc->weight.data, {fc->weight.d1, fc->weight.d2});
        temp.copy(fc->weight.as_nd(), false);
        auto temp_as_2d = temp.as_2d();
        // printf("%d, %d, %d, %d, %d, %d, %d, %d\n", temp_as_2d.d1, temp_as_2d.d2, fc->weight.d1, fc->weight.d2, h, w, c, batchSz);
        for (int l = 0; l < m; l++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    for (int k = 0; k < c; k++)
                    {
                        fc->weight(i * w * c + j * c + k, l) = temp_as_2d(k * h * w + i * w + j, l);
                    }
                }
            }
        }
        int i;
        for (i = 0; i < node->allNodesInExecutionOrderRef->size(); i++)
        {
            if (node->allNodesInExecutionOrderRef->at(i) == node)
            {
                // printf("Found\n");
                break;
            }
        }
        node->allNodesInExecutionOrderRef->erase(node->allNodesInExecutionOrderRef->begin() + i);
    }
    else if (node->layer->name == "MaxPool2D")
    {
        auto parentMode = node->parents[0]->layer->mode;
        if (parentMode == 1 || parentMode == 3)
        {
            node->layer->mode = 3;
        }
        else
        {
            node->layer->mode = 0;
        }
    }
    else if (node->layer->name == "ReLU")
    {
        auto parentMode = node->parents[0]->layer->mode;
        if (parentMode == 0 || parentMode == 2)
        {
            node->layer->mode = 0;
        }
        else
        {
            bool oneChildLinear = false;
            for (auto &child : node->children)
            {
                if (child->layer->name == "Conv2D" || child->layer->name == "FC" || child->layer->name == "BatchNorm2dInference" || child->layer->name == "GlobalAvgPool2D" || child->layer->name == "AvgPool2D" || child->layer->name == "Flatten")
                {
                    oneChildLinear = true;
                    break;
                }
            }
            if (oneChildLinear)
            {
                node->layer->mode = 2;
            }
            else
            {
                node->layer->mode = 3;
            }
        }
    }
    else if (node->layer->name == "Input")
    {
        // do nothing
    }
    else
    {
        throw std::runtime_error("Unknown layer type: " + node->layer->name);
    }
}

template <typename T>
void pinCpuMem(LayerGraphNode<T> *n, LayerGraphNode<T> *r)
{
    // printf("Layer=%s, %lx, %lu\n", n->layer->name.data(), n->currTensor->data, n->currTensor->size());
    if (n->currTensor->data)
        checkCudaErrors(cudaHostRegister(n->currTensor->data, n->currTensor->size() * sizeof(T), cudaHostRegisterDefault));
    auto w = n->layer->getweights().data;
    auto b = n->layer->getbias().data;
    auto wSz = n->layer->getweights().size;
    auto bSz = n->layer->getbias().size;
    if (w)
        checkCudaErrors(cudaHostRegister(w, wSz * sizeof(T), cudaHostRegisterDefault));
    if (b)
        checkCudaErrors(cudaHostRegister(b, bSz * sizeof(T), cudaHostRegisterDefault));
    if (n->layer->name == "_MHADummy")
    {
        auto mha = static_cast<_MHADummy<T> *>(n->layer);
        checkCudaErrors(cudaHostRegister(mha->wQKV.data, mha->wQKV.size() * sizeof(T), cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(mha->bQKV.data, mha->bQKV.size() * sizeof(T), cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(mha->wProj.data, mha->wProj.size() * sizeof(T), cudaHostRegisterDefault));
        checkCudaErrors(cudaHostRegister(mha->bProj.data, mha->bProj.size() * sizeof(T), cudaHostRegisterDefault));
    }
}