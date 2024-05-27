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
#include <sytorch/backend/llama_base.h>

template <typename T>
class LlamaTransformer : public LlamaBase<T>
{
public:
    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode)
    {
        // ARS(size, in, in, out, out, shift);
        // SlothARS(size, in, out, shift);
        if (mode == 0)
        {
            SlothFaithfulARS(size, LlamaConfig::bitlength, in, out, shift, "Linear::");
        }
        else if (mode == 1)
        {
            SlothARS(size, in, out, shift);
        }
        else
        {
            assert(0 && "Unknown truncate type");
        }
    }

    void gelu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        u64 sz = in.size();
        always_assert(sz == out.size());
        if (mode == 0)
        {
            SlothGelu(sz, LlamaConfig::bitlength, in.data, out.data, scale);
        }
        else if (mode == 1)
        {
            SlothGelu(sz, LlamaConfig::bitlength - scale, in.data, out.data, scale);
        }
    }

    void silu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        u64 sz = in.size();
        always_assert(sz == out.size());
        if (mode == 0)
        {
            SlothSilu(sz, LlamaConfig::bitlength, in.data, out.data, scale);
        }
        else if (mode == 1)
        {
            SlothSilu(sz, LlamaConfig::bitlength - scale, in.data, out.data, scale);
        }
    }

    void softmax(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode)
    {
        in.is_same_shape(out);
        if (mode == 0)
            Softmax(in.shape[0], in.shape[1], LlamaConfig::bitlength, in.data, out.data, scale);
        else if (mode == 1)
            Softmax(in.shape[0], in.shape[1], LlamaConfig::bitlength - scale + 1, in.data, out.data, scale);
    }

    void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        always_assert(A.d1 == B.d1);
        always_assert(A.d1 == x.shape.back());
        always_assert(x.is_same_shape(y));
        u64 s2 = x.shape.back();
        u64 s1 = x.size() / s2;
        SlothLayerNorm(s1, s2, x.data, A.data, B.data, y.data, scale);
    }

    void rmsnorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        always_assert(A.d1 == B.d1);
        always_assert(A.d1 == x.shape.back());
        always_assert(x.is_same_shape(y));
        u64 s2 = x.shape.back();
        u64 s1 = x.size() / s2;
        SlothRMSNorm(s1, s2, x.data, A.data, B.data, y.data, scale);
    }

    void attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y)
    {
        always_assert(x.is_same_shape(y));
        always_assert(x.shape.size() == 2);
        always_assert(x.shape[0] == x.shape[1]);

        if (LlamaConfig::party == DEALER)
        {
            y.copy(x, false);
        }
        else
        {
            u64 n_seq = x.shape[0];
            auto y_2d = y.as_2d();
            auto x_2d = x.as_2d();

            for (u64 j = 0; j < n_seq; ++j)
            {
                for (u64 k = 0; k < j + 1; ++k)
                {
                    y_2d(j, k) = x_2d(j, k);
                }
                for (u64 k = j + 1; k < n_seq; ++k)
                {
                    y_2d(j, k) = x_2d(j, k) - scalar;
                }
            }
        }
    }

    void softmax_triangular(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode)
    {
        in.is_same_shape(out);
        if (mode == 0)
            SoftmaxTriangular(in.shape[0], in.shape[1], LlamaConfig::bitlength, in.data, out.data, scale);
        else if (mode == 1)
            SoftmaxTriangular(in.shape[0], in.shape[1], LlamaConfig::bitlength - scale, in.data, out.data, scale);
    }

    void tanh(const Tensor<T> &in, const Tensor<T> &out, u64 scale)
    {
        Tanh(in.size(), in.data, out.data, scale);
    }

    void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
    {
        always_assert(a.is_same_shape(b));
        always_assert(a.is_same_shape(out));
        // Mul(a.size(), a.data, b.data, out.data);
        ElemWiseMul(a.size(), a.data, b.data, out.data);
    }

    void doOptimizeGelu(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward)
        {
            if (node->children.size() == 1)
            {
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->name == "GeLU")
                {
                    child->layer->mode = 1;
                }
            }
        }
    }

    void doOptimizeDiv(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward)
        {
            if (node->children.size() == 1)
            {
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->name == "_ScalarDiv")
                {
                    auto layer_sd = (_ScalarDiv<T> *)child->layer;
                    T d = T(double(1LL << (layer_sd->scale)) / layer_sd->scalar);
                    // if d is power of two
                    if ((d & (d - 1)) == 0)
                    {
                        // seems very hacky
                        node->layer->scale += (layer_sd->scale - log2(d));
                        child->layer->mode = 1;
                    }
                }
            }
        }
    }

    void attention_triangular(Tensor2D<T> &q, Tensor2D<T> &k, Tensor2D<T> &v, Tensor2D<T> &out, u64 scale, u64 n_heads)
    {
        u64 n_seq = q.d1;
        u64 n_embd = q.d2;
        SlothAttentionTriangular(n_seq, n_embd, n_heads, q.data, k.data, v.data, out.data, scale);
    }

    void doOptimizeSoftmax(LayerGraphNode<T> *node, LayerGraphNode<T> *root)
    {
        if (node->layer->doTruncationForward || node->layer->name == "_ScalarDiv")
        {
            if (node->children.size() == 1)
            {
                LayerGraphNode<T> *child = node->children[0];
                if (child->layer->name == "SoftMax" || child->layer->name == "SoftMaxTriangular")
                {
                    child->layer->mode = 1;
                }
            }
        }
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { doOptimizeGelu(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { doOptimizeSoftmax(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { doOptimizeDiv(n, r); });
    }

    void scalardiv(Tensor<T> &x, double scalar, Tensor<T> &y, u64 scale, u64 mode)
    {
        if (mode == 1)
        {
            y.copy(x, false);
        }
        else
        {
            T d = T(double(1LL << (scale)) / scalar);
            if ((d & (d - 1)) == 0)
            {
                SlothFaithfulARS(x.size(), LlamaConfig::bitlength, x.data, y.data, scale - log2(d), "Linear::");
            }
            else
            {
                this->scalarmul(x, d, y);
                SlothFaithfulARS(y.size(), LlamaConfig::bitlength, y.data, y.data, scale, "Linear::");
            }
        }
    }
};
