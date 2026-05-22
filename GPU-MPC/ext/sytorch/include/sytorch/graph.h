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

#include <set>
#include <fstream>
#include <queue>


template <typename T>
class Layer;

template <typename T>
class Tensor;

template <typename T>
struct LayerGraphNode
{
    Layer<T> *layer;
    std::vector<LayerGraphNode<T> *> parents;
    std::vector<LayerGraphNode<T> *> children;
    int numUsages = 0;
    Tensor<T> *currTensor = nullptr;
    bool mark = false;
    std::vector<LayerGraphNode<T> *> *allNodesInExecutionOrderRef = nullptr;
    bool onGpu = false;

    bool incrementAndGc()
    {
        if (layer->name == "Input")
        {
            return false;
        }
        numUsages++; // todo: make it atomic
        if (numUsages == children.size())
        {
            // printf("Freeing gpu mem\n");
            assert(layer->activation.data == currTensor->data);
            layer->activation.freeGpu();
            return true;
        }
        return false;
    }
};

template <typename T, typename Functor>
void topologicalVisit(std::set<LayerGraphNode<T> *> &visited, LayerGraphNode<T> *node, LayerGraphNode<T> *root, Functor visitFn)
{
    if (visited.find(node) != visited.end())
    {
        return;
    }
    visited.insert(node);
    for (auto parent : node->parents)
    {
        topologicalVisit(visited, parent, root, visitFn);
    }

    visitFn(node, root);

    for (auto child : node->children)
    {
        topologicalVisit(visited, child, root, visitFn);
    }
}

template <typename T, typename Functor>
void topologicalApply(LayerGraphNode<T> *root, Functor visitFn)
{
    std::set<LayerGraphNode<T> *> visited;
    topologicalVisit(visited, root, root, visitFn);
}

template <typename T>
void print_dot_graph(LayerGraphNode<T> *root)
{
    std::ofstream dotfile("graph.dot");
    dotfile << "digraph G {" << std::endl;

    topologicalApply(root, [&dotfile](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                     {
        if (node->layer != nullptr) {
            // std::string label = node->layer->name + "-" + std::to_string(node->layer->mode) + "-" + (node->layer->doPreSignExtension ? "true" : "false");
            std::string label = node->layer->name;
            if (node->layer->paramstring != "") {
                std::string args = node->layer->paramstring;
                std::replace(args.begin(), args.end(), '|', ',');
                // remove last comma if exists
                if (args.back() == ',') {
                    args.pop_back();
                }
                label += "(" + args + ")";
            }
            dotfile << node->layer->name + std::to_string((uint64_t)(node->layer)) << " [label=\"" << label << "\"" + (node->mark ? std::string(" color=\"red\"") : std::string("")) + "];" << std::endl;
            for (auto &child : node->children) {
                dotfile << node->layer->name + std::to_string((uint64_t)(node->layer)) << " -> " << child->layer->name + std::to_string((uint64_t)(child->layer)) << ";" << std::endl;
            }
        } });

    dotfile << "}" << std::endl;
    dotfile.close();
}
