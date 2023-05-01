#pragma once
#include <set>
#include <fstream>
#include <queue>

template <typename T>
class Layer;

template <typename T>
class Tensor;

template <typename T>
struct LayerGraphNode {
    Layer<T> *layer;
    std::vector<LayerGraphNode<T> *> parents;
    std::vector<LayerGraphNode<T> *> children;
    int numUsages = 0;
    Tensor<T> *currTensor = nullptr;
    bool mark = false;
    std::vector<LayerGraphNode<T> *> *allNodesInExecutionOrderRef = nullptr;

    bool incrementAndGc()
    {
        if (layer->name == "Input") {
            return false;
        }
        numUsages++; // todo: make it atomic
        if (numUsages == children.size()) {
            currTensor->free();
            return true;
        }
        return false;
    }
};

template <typename T, typename Functor>
void topologicalVisit(std::set<LayerGraphNode<T> *> &visited, LayerGraphNode<T> *node, LayerGraphNode<T> *root, Functor visitFn)
{
    if (visited.find(node) != visited.end()) {
        return;
    }
    visited.insert(node);
    for(auto parent : node->parents) {
        topologicalVisit(visited, parent, root, visitFn);
    }

    visitFn(node, root);

    for(auto child : node->children) {
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

    topologicalApply(root, [&dotfile](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
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
        }
    });

    dotfile << "}" << std::endl;
    dotfile.close();
}
