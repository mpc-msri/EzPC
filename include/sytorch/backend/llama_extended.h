#pragma once
#include "llama_base.h"

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
        relu(out, out, drelu, 0);
    }

    void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale) {
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

    void truncate(T *in, T *out, u64 shift, u64 size) {
        if (this->useLocalTruncation) {
            for(u64 i = 0; i < size; i++) {
                out[i] = in[i] >> shift;
            }
        }
        else {
            ARS(size, in, in, out, out, shift);
        }
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale) {
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

    void batchNorm2dInference(const Tensor<T> &A, const Tensor<T> &B, const Tensor4D<T> &x, Tensor4D<T> &y, u64 scale)
    {
        assert(A.size == B.size);
        assert(A.size == x.d4);
        assert(x.d4 == y.d4);
        assert(x.d1 == y.d1);
        assert(x.d2 == y.d2);
        assert(x.d3 == y.d3);
        // replicate A and B
        Tensor4D<T> A2(x.d1, x.d2, x.d3, x.d4);
        for (int i = 0; i < x.d1; ++i) {
            for(int j = 0; j < x.d2; ++j) {
                for(int k = 0; k < x.d3; ++k) {
                    for(int l = 0; l < x.d4; ++l) {
                        A2(i, j, k, l) = A(l);
                    }
                }
            }
        }

        ElemWiseSecretSharedVectorMult(x.d1 * x.d2 * x.d3 * x.d4, x.data, x.data, A2.data, A2.data, y.data, y.data);

        for(int i = 0; i < x.d1; ++i) {
            for(int j = 0; j < x.d2; ++j) {
                for(int k = 0; k < x.d3; ++k) {
                    for(int l = 0; l < x.d4; ++l) {
                        y(i, j, k, l) += B(l);
                    }
                }
            }
        }

        this->truncateForward(y, scale);

    }

    void optimize(Sequential<T> &model) {
        // push truncations forward
        int sz = model.layers.size();
        for (int i = 0; i < sz; ++i) {
            Layer<T> *layer = model.layers[i];
            if (layer->doTruncationForward) {
                if (i != (sz - 1)) {
                    if (model.layers[i + 1]->doTruncationForward) {
                        // no optimization possible
                        // this is set to true for FC and Conv2D
                    }
                    else {
                        // optimize
                        layer->doTruncationForward = false;
                        model.layers[i + 1]->doTruncationForward = true;
                    }
                }
            }
        }

    }

};
