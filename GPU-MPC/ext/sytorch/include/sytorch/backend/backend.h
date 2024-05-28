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
#include <llama/assert.h>
#include <omp.h>

#define NOT_IMPLEMENTED                                                                  \
    {                                                                                    \
        throw std::runtime_error(std::string("not implemented ") + __PRETTY_FUNCTION__); \
    }

template <typename T>
class Backend
{
public:
    // truncation API
    virtual void truncate(T *in, T *out, u64 shift, u64 size, u8 mode = 0) NOT_IMPLEMENTED;

    void truncate(const Tensor<T> &in, const Tensor<T> &out, u64 shift, u8 mode = 0)
    {
        always_assert(in.is_same_shape(out));
        truncate(in.data, out.data, shift, in.size(), mode);
    }

    void truncate(const Tensor4D<T> &in, u64 shift, u8 mode = 0)
    {
        truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4, mode);
    }

    void truncate(const Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        truncate(in.data, in.data, shift, in.size(), mode);
    }

    virtual void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        truncate(in.data, in.data, shift, in.size(), mode);
    }

    void truncate(const Tensor2D<T> &in, u64 shift, u8 mode = 0)
    {
        truncate(in.data, in.data, shift, in.d1 * in.d2, mode);
    }

    void truncate(const Tensor1D<T> &in, u64 shift, u8 mode = 0)
    {
        truncate(in.data, in.data, shift, in.d1, mode);
    }

    // matmul API
    virtual void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED;
    virtual void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c, bool useBias, Tensor1D<T> &d, bool isFirst)
    {
        // printf("inp=%lx\n", a.d_data);
        matmul(a, b, c);
        // printf("input[0]=%ld, output[0]=%ld\n", i64(a.data[0]), i64(this->activation.data[0]));
        if (useBias)
        {
            auto c_as_nd = c.as_nd();
            addbias(c_as_nd, d);
        }
    }
    virtual void matmul_triangular(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED;

    // conv API
    virtual void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst) NOT_IMPLEMENTED;
    virtual void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, bool useBias, const Tensor1D<T> &bias, Tensor4D<T> &output, bool isFirst)
    {
        conv2D(fh, fw, padding, stride, ci, co, input, filter, output, isFirst);
        if (useBias)
        {
            auto output_as_nd = output.as_nd();
            addbias(output_as_nd, bias);
        }
    }
    virtual void conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output) NOT_IMPLEMENTED;
    virtual void convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output) NOT_IMPLEMENTED;

    // relu API
    virtual void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) NOT_IMPLEMENTED;
    virtual void select(const Tensor<T> &in, const Tensor<T> &drelu, const Tensor<T> &out) NOT_IMPLEMENTED;

    // avgpool API
    virtual void div(Tensor<T> &in, T divisor, u64 scale) NOT_IMPLEMENTED;
    virtual void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out)
    {
        // assert(in.d1 == out.d1);
        // assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2 * padding - ks) / stride + 1;
        u64 newW = (in.d3 + 2 * padding - ks) / stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);

#pragma omp parallel for collapse(4)
        for (int i = 0; i < in.d1; i++)
        {
            for (int j = 0; j < newH; j++)
            {
                for (int k = 0; k < newW; k++)
                {
                    for (int l = 0; l < in.d4; l++)
                    {
                        T sum = 0;
                        for (int m = 0; m < ks; m++)
                        {
                            for (int n = 0; n < ks; n++)
                            {
                                sum += in(i, j * stride + m, k * stride + n, l);
                            }
                        }
                        out(i, j, k, l) = sum;
                    }
                }
            }
        }
    }

    virtual void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) NOT_IMPLEMENTED;
    virtual void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, u64 scale) NOT_IMPLEMENTED;

    // maxpool API
    virtual void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) NOT_IMPLEMENTED;

    virtual void batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale) NOT_IMPLEMENTED;
    virtual void signext(Tensor<T> &x, u64 scale) NOT_IMPLEMENTED;

    // add API
    virtual void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out) NOT_IMPLEMENTED;

    // softmax API
    virtual void softmaxGrad(Tensor<T> &in) NOT_IMPLEMENTED;

    virtual void gelu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0) NOT_IMPLEMENTED;
    virtual void silu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0) NOT_IMPLEMENTED;
    virtual void tanh(const Tensor<T> &in, const Tensor<T> &out, u64 scale) NOT_IMPLEMENTED;
    virtual void softmax(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0) NOT_IMPLEMENTED;
    virtual void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale) NOT_IMPLEMENTED;
    virtual void rmsnorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale) NOT_IMPLEMENTED;
    virtual void addbias(Tensor<T> &x, const Tensor1D<T> &bias)
    {
        always_assert(x.shape.back() == bias.d1);

#pragma omp parallel for
        for (u64 i = 0; i < x.size(); ++i)
        {
            x.data[i] += bias(i % bias.d1);
        }
    }

    virtual void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y) NOT_IMPLEMENTED;
    virtual void scalardiv(Tensor<T> &x, double scalar, Tensor<T> &y, u64 scale, u64 mode = 0) NOT_IMPLEMENTED;
    virtual void attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y) NOT_IMPLEMENTED;
    virtual void local_attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y) NOT_IMPLEMENTED;
    virtual void softmax_triangular(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0) NOT_IMPLEMENTED;
    virtual void attention_triangular(Tensor2D<T> &q, Tensor2D<T> &k, Tensor2D<T> &v, Tensor2D<T> &out, u64 scale, u64 n_heads) NOT_IMPLEMENTED;
    virtual void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out) NOT_IMPLEMENTED;

    // MHA
    virtual void mha(int n_heads, int n_embed, int dim_W, bool selfAttn, bool doNormQKt, bool doRotEmb, const Tensor2D<T> &wQKV, const Tensor1D<T> &bQKV, const Tensor2D<T> &wProj, const Tensor1D<T> &bProj, const Tensor2D<T> &X, Tensor2D<T> &Y)
    {
        assert(0 && "not implemented");
    }

    virtual void rotary_embedding(Tensor<T> &x, Tensor<T> &y, u64 scale, u64 max_position_embeddings = 2048, u64 base = 10000)
    {
        u64 n_seq = x.shape[0];
        u64 dim = x.shape[1];
        // printf("dims=%d, %lu, %ld, %ld\n", x.shape.size(), x.size(), x.data[0], y.data[0]);
        auto y_2d = y.as_2d();
        auto x_2d = x.as_2d();

        for (u64 i = 0; i < n_seq; ++i)
        {
            for (u64 j = 0; j < dim; j++)
            {
                double scalar = 1.0 / (std::pow(base, (double)((2 * j) % dim) / dim));
                T scalarInt = (i * scalar) * std::pow(2, scale);
                T sinx = std::sin(scalarInt / (float)std::pow(2, scale)) * std::pow(2, scale - 3);
                T cosx = std::cos(scalarInt / (float)std::pow(2, scale)) * std::pow(2, scale - 3);

                if (sinx == (1ULL << (scale - 3)))
                    sinx -= 1;
                if (cosx == (1ULL << (scale - 3)))
                    cosx -= 1;
                u64 k = (j + dim / 2) % dim;
                T mul = 2 * (j >= dim / 2) - 1;
                T z = cosx * x_2d(i, j) + sinx * mul * x_2d(i, k);
                y_2d(i, j) = z;
            }
        }
        this->truncate(y_2d, scale - 3, 1);
    }

    virtual void optimize(LayerGraphNode<T> *root)
    {
    }

    virtual void output(Tensor<T> &a)
    {
    }

    virtual void close() {}
};
