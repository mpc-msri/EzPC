#pragma once
#include <sytorch/backend/llama_base.h>

template <typename T>
class Sequential;

template <typename T>
class LlamaExtended : public LlamaBase<T> {
public:

    void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        int sz = in.d1 * in.d2 * in.d3 * in.d4;
        Backend<T>::truncate(in, out, shift);
        relu(out, out, drelu, 0, 0);
    }

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
        Relu(sz, in.data, in.data, out.data, out.data, drelu.data);
    }

    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode) {
        if (this->useLocalTruncation) {
            for(u64 i = 0; i < size; i++) {
                out[i] = in[i] >> shift;
            }
        }
        else {
            ARS(size, in, in, out, out, shift);
        }
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        Tensor<T> maxBit((ks * ks - 1) * out.d1 * out.d2 * out.d3 * out.d4);
        maxIdx.resize(ks * ks * out.d1, out.d2, out.d3, out.d4);
        MaxPool(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxBit.data);
        MaxPoolOneHot(out.d1, out.d2, out.d3, out.d4, ks, ks, maxBit.data, maxIdx.data);
        // maxBit.template print<1>();
        // maxIdx.template print<1>();
    }

    void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        //throw std::runtime_error("Not implemented");
        MaxPoolBackward(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxIdx.data);
    }

    void doOptimize(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward) {
            if (node->children.size() == 1) {
                // std::cout << "yeah.." << std::endl;
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->doTruncationForward) {
                    // no optimization possible
                    // this is set to true for FC, Conv2D and BatchNorm2dInference
                }
                else {
                    if (child->layer->name == "MaxPool2D" || child->layer->name == "ReLU") {
                        // optimize
                        node->layer->doTruncationForward = false;
                        child->layer->doTruncationForward = true;
                    }
                }
            }
        }
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r) {
            doOptimize(n, r);
        });
    }

};
