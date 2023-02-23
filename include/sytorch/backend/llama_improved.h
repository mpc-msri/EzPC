#pragma once
#include <sytorch/backend/llama_base.h>
#include <llama/stats.h>

template <typename T>
class Sequential;

// in LlamaImproved, mode takes the value according to the following rule:
// 0: the layer takes as input \ell bits and outputs \ell bits
// 1: the layer takes as input \ell bits and outputs \ell - scale bits
// 2: the layer takes as input \ell - scale bits and outputs \ell bits
// 3: the layer takes as input \ell - scale bits and outputs \ell - scale bits

template <typename T>
class LlamaImproved : public LlamaBase<T> {
public:

    void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale, int mode) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        int sz = in.d1 * in.d2 * in.d3 * in.d4;
        if (mode == 0) {
            Relu2Round(sz, in.data, in.data, out.data, out.data, drelu.data, LlamaConfig::bitlength);
        }
        else if (mode == 3) {
            for(int i = 0; i < sz; i++) {
                in.data[i] = in.data[i] % (1ULL << (LlamaConfig::bitlength - scale));
            }
            LlamaConfig::bitlength -= scale;
            Relu2Round(sz, in.data, in.data, out.data, out.data, drelu.data, LlamaConfig::bitlength);
            LlamaConfig::bitlength += scale;
        }
        else if (mode == 2) {
            for(int i = 0; i < sz; i++) {
                in.data[i] = in.data[i] % (1ULL << (LlamaConfig::bitlength - scale));
            }
            ReluExtend(sz, LlamaConfig::bitlength - scale, LlamaConfig::bitlength, in.data, out.data, drelu.data);
        }
        else {
            std::runtime_error("this should not have happened");
        }
    }

    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode) {
        for(u64 i = 0; i < size; i++) {
            out[i] = ((u64)in[i]) >> shift;
        }
        if (mode == 0) {
            SignExtend2(size, LlamaConfig::bitlength - shift, LlamaConfig::bitlength, out, out);
        }
        else if (mode == 1) {
            // do nothing
        }
        else {
            throw std::runtime_error("this should not have happened");
        }
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        Tensor<T> maxBit((ks * ks - 1) * out.d1 * out.d2 * out.d3 * out.d4);
        maxIdx.resize(ks * ks * out.d1, out.d2, out.d3, out.d4);

        if (mode == 0) {
            MaxPoolDouble(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxBit.data);
        }
        else if (mode == 3) {
            LlamaConfig::bitlength -= scale; // why is this hack not working? new intel - it works
            MaxPoolDouble(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxBit.data);
            LlamaConfig::bitlength += scale;
        }
        else if (mode == 2) {
            // TODO: doPostSignExt should be set to true for this
            LlamaConfig::bitlength -= scale; // why is this hack not working? new intel - it works
            MaxPoolDouble(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxBit.data);
            LlamaConfig::bitlength += scale;
        }
        else {
            throw std::runtime_error("this should not have happened");
        }
        MaxPoolOneHot(out.d1, out.d2, out.d3, out.d4, ks, ks, maxBit.data, maxIdx.data);
    }

    void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        //throw std::runtime_error("Not implemented");
        MaxPoolBackward(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxIdx.data);
    }

    void signext(Tensor4D<T> &x, u64 scale)
    {
        SignExtend2(x.d1 * x.d2 * x.d3 * x.d4, LlamaConfig::bitlength - scale, LlamaConfig::bitlength, x.data, x.data);
    }

    void optimize(Sequential<T> &model) {

    }

    void doOptimize(LayerTreeNode<T> *node, LayerTreeNode<T> *root)
    {
        // in LlamaImproved, mode takes the value according to the following rule:
        // 0: the layer takes as input \ell bits and outputs \ell bits
        // 1: the layer takes as input \ell bits and outputs \ell - scale bits
        // 2: the layer takes as input \ell - scale bits and outputs \ell bits
        // 3: the layer takes as input \ell - scale bits and outputs \ell - scale bits

        // std::cerr << "Visiting " << node->curr->name << std::endl;
        if (node->curr->name == "Conv2D" || node->curr->name == "FC" || node->curr->name == "BatchNorm2dInference")
        {
            // only one parent
            auto parent = node->parents[0];
            if (parent->curr->mode == 1 || parent->curr->mode == 3) {
                // std::cerr << "    Found parent " << parent->curr->name << " with mode " << parent->curr->mode << std::endl;
                node->curr->doPreSignExtension = true;
            }
            node->curr->mode = 1;
            node->curr->forwardTruncationMode = 1;
        }
        else if (node->curr->name == "Add")
        {
            auto parentMode = node->parents[0]->curr->mode;
            for(auto &parent : node->parents) {
                if ((parent->curr->mode % 2) != (parentMode % 2)) {
                    throw std::runtime_error("Add layer has parents with different modes");
                }
            }
            node->curr->mode = 3 * (parentMode % 2);
        }
        else if (node->curr->name == "Flatten")
        {
            auto parentMode = node->parents[0]->curr->mode;
            node->curr->mode = 3 * (parentMode % 2);
        }
        else if (node->curr->name == "MaxPool2D")
        {
            auto parentMode = node->parents[0]->curr->mode;
            if (parentMode == 1 || parentMode == 3) {
                node->curr->mode = 3;
                bool atleastOneChildAdd = false;
                for(auto &child : node->children) {
                    if (child->curr->name == "Add") {
                        atleastOneChildAdd = true;
                        break;
                    }
                }
                if (atleastOneChildAdd) {
                    node->curr->mode = 2;
                    node->curr->doPostSignExtension = true;
                }
            }
            else {
                node->curr->mode = 0;
            }
        }
        else if (node->curr->name == "ReLU")
        {
            auto parentMode = node->parents[0]->curr->mode;
            if (parentMode == 0 || parentMode == 2)
            {
                node->curr->mode = 0;
            }
            else {
                bool allChildMaxpools = true;
                for(auto &child : node->children) {
                    if (child->curr->name != "MaxPool2D") {
                        allChildMaxpools = false;
                        break;
                    }
                }
                if (allChildMaxpools) {
                    node->curr->mode = 3;
                }
                else {
                    node->curr->mode = 2;
                }
            }
        }

    }

    void topologicalVisit(std::set<LayerTreeNode<T> *> &visited, LayerTreeNode<T> *node, LayerTreeNode<T> *root)
    {
        if (visited.find(node) != visited.end()) {
            return;
        }
        visited.insert(node);
        for(auto parent : node->parents) {
            topologicalVisit(visited, parent, root);
        }

        doOptimize(node, root);

        for(auto child : node->children) {
            topologicalVisit(visited, child, root);
        }
    }

    void optimize(LayerTreeNode<T> *root)
    {   
        std::set<LayerTreeNode<T> *> visited;
        topologicalVisit(visited, root, root);
    }

};
