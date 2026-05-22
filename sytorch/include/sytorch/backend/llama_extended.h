#pragma once
#include <sytorch/backend/llama_base.h>

template <typename T>
class Sequential;

template <typename T>
class LlamaExtended : public LlamaBase<T> {
public:

    void relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) {
        assert(in.is_same_shape(out));
        assert(in.is_same_shape(drelu));
        int sz = in.size();
        Relu(sz, in.data, in.data, out.data, out.data, drelu.data);
    }

    void leakyRelu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode, T alpha)
    {
        assert(in.is_same_shape(out));
        assert(in.is_same_shape(drelu));
        int sz = in.size();
        std::vector<u64> shape = in.shape;

        T minus_one = type_cast<T>(-1 * (1LL << scale));
        auto ct = new ClearText<T>;
        // leakyrelu = relu(x) - alpha * relu(-x)

        // relu(x)
        Tensor<T> relu_x(shape);
        Relu(sz, in.data, in.data, relu_x.data, relu_x.data, drelu.data);

        // -x
        Tensor<T> minus_x(shape);
        ct->fastfor(sz, [&](u64 i)
                    { minus_x.data[i] = minus_one * in.data[i]; });
        Backend<T>::truncate(minus_x, scale);

        // relu(-x)
        Tensor<T> relu_minus_x(shape);
        Relu(sz, minus_x.data, minus_x.data, relu_minus_x.data, relu_minus_x.data, drelu.data);

        // alpha * relu(-x)
        Tensor<T> alpha_relu_minus_x(shape);
        ct->fastfor(sz, [&](u64 i)
                    { alpha_relu_minus_x.data[i] = alpha * relu_minus_x.data[i]; });
        Backend<T>::truncate(alpha_relu_minus_x, scale);

        // relu(x) - alpha * relu(-x)
        ct->fastfor(sz, [&](u64 i)
                    { out.data[i] = relu_x.data[i] - alpha_relu_minus_x.data[i]; });
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
        Tensor1D<T> maxBit((ks * ks - 1) * out.d1 * out.d2 * out.d3 * out.d4);
        maxIdx.resize(ks * ks * out.d1, out.d2, out.d3, out.d4);
        MaxPool(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxBit.data);
        MaxPoolOneHot(out.d1, out.d2, out.d3, out.d4, ks, ks, maxBit.data, maxIdx.data);
        // maxBit.template print<1>();
        // maxIdx.template print<1>();
    }

    void gelu(const Tensor<T> &in, const Tensor<T> &out, u64 scale)
    {
        u64 sz = in.size();
        always_assert(sz == out.size());
        T t1 = (T) (sqrt(2.0 / M_PI) * (1LL << scale));
        T t2 = (T) (0.044715 * (1LL << scale));
        auto ct = new ClearText<T>;
        // t = x^2
        ElemWiseActModelVectorMult(sz, in.data, in.data, in.data, in.data, out.data, out.data);
        Backend<T>::truncate(out, scale);
        
        // t = x^3
        ElemWiseActModelVectorMult(sz, out.data, out.data, in.data, in.data, out.data, out.data);
        Backend<T>::truncate(out, scale);
        
        // t = x^3 * 0.044715
        ct->fastfor(sz, [&](u64 i) {
            out.data[i] = out.data[i] * t2;
        });
        Backend<T>::truncate(out, scale);

        // t = x^3 * 0.044715 + x
        ct->fastfor(sz, [&](u64 i) {
            out.data[i] = out.data[i] + in.data[i];
        });

        // t = (x^3 * 0.044715 + x )* sqrt(2/pi)
        ct->fastfor(sz, [&](u64 i) {
            out.data[i] = out.data[i] * t1;
        });
        Backend<T>::truncate(out, scale);

        // t = tanh((x^3 * 0.044715 + x) * sqrt(2/pi))
        // TODO: teehee

        // t = 1 + tanh((x^3 * 0.044715 + x) * sqrt(2/pi))
        ct->fastfor(sz, [&](u64 i) {
            out.data[i] = out.data[i] + (1LL << scale);
        });

        // t = x * (1 + tanh((x^3 * 0.044715 + x) * sqrt(2/pi))) / 2
        ElemWiseActModelVectorMult(sz, out.data, out.data, in.data, in.data, out.data, out.data);
        Backend<T>::truncate(out, scale + 1);

        delete ct;
    }

    void softmax(Tensor<T> &in, Tensor<T> &out, u64 scale)
    {
        // TODO: teehee
        out.copy(in, false);
    }

    void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        always_assert(A.d1 == B.d1);
        always_assert(A.d1 == x.shape.back());
        always_assert(x.is_same_shape(y));
        
        u64 channels = x.shape.back();

        auto ct = new ClearText<T>;
        auto shape2 = x.shape;
        shape2.pop_back();

        Tensor<T> tmp(x.shape);
        Tensor<T> mean(shape2);
        Tensor<T> var(shape2);

        ct->fastfor(x.size() / channels, [&](u64 i) {
            mean.data[i] = 0;
            for (u64 j = 0; j < channels; j++) {
                mean.data[i] += x.data[i * channels + j];
            }
        });

        LlamaBase<T>::div(mean, channels, scale);

        ct->fastfor(x.size() / channels, [&](u64 i) {
            for (u64 j = 0; j < channels; j++) {
                tmp.data[i * channels + j] = x.data[i * channels + j] - mean.data[i];
            }
        });

        ElemWiseActModelVectorMult(tmp.size(), tmp.data, tmp.data, tmp.data, tmp.data, tmp.data, tmp.data);

        ct->fastfor(x.size() / channels, [&](u64 i) {
            var.data[i] = 0;
            for (u64 j = 0; j < channels; j++) {
                var.data[i] += tmp.data[i * channels + j];
            }
        });

        Backend<T>::truncate(var, scale);
        LlamaBase<T>::div(var, channels, scale);

        // TODO: invvar = invsqrt(var)
        auto &invvar = var;

        ct->fastfor(x.size() / channels, [&](u64 i) {
            for (u64 j = 0; j < channels; j++) {
                tmp.data[i * channels + j] = x.data[i * channels + j] - mean.data[i];
                y.data[i * channels + j] = invvar.data[i];
            }
        });

        ElemWiseActModelVectorMult(tmp.size(), tmp.data, tmp.data, y.data, y.data, y.data, y.data);
        Backend<T>::truncate(y, scale);

        auto &Aexpand = tmp;
        ct->fastfor(x.size() / channels, [&](u64 i) {
            for (u64 j = 0; j < channels; j++) {
                Aexpand.data[i * channels + j] = A(j);
            }
        });
        ElemWiseActModelVectorMult(tmp.size(), Aexpand.data, Aexpand.data, y.data, y.data, y.data, y.data);

        ct->fastfor(x.size() / channels, [&](u64 i) {
            for (u64 j = 0; j < channels; j++) {
                y.data[i * channels + j] += B(j);
            }
        });

        delete ct;
    }


    void doOptimize(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward) {
            if (node->children.size() == 1) {
                // std::cout << "yeah.." << "\n";
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->doTruncationForward) {
                    // no optimization possible
                    // this is set to true for FC, Conv2D and BatchNormInference
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
