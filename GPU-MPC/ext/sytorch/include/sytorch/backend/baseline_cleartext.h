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
#include "backend.h"
#include <sytorch/utils.h>
#include <thread>

template <typename T>
class BaselineClearText : public Backend<T> {
private:
public:
    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode);
    static const bool probablistic = true;
    static const bool localTruncationEmulation = false;
    // static const u64 bw = sizeof(T) * 8;
    static const u64 bw = 50;

    template <typename Functor>
    void fastfor(u64 size, Functor f)
    {
        #pragma omp parallel for
        for (u64 i = 0; i < size; i++) {
            f(i);
        }
    }

    void modbw(T* x, u64 size)
    {
        if constexpr (std::is_floating_point<T>::value) {
            return;
        }
        else if /*constexpr*/ (bw == sizeof(T) * 8) {
            return;
        }
        else {
            i64 mask = (1LL << (bw - 1));
            fastfor(size, [&](u64 i) {
                i64 val = (x[i] + mask) % (1LL << bw);
                val -= mask;
                x[i] = val;
            });
        }
    }
    
    void modbw(T &x)
    {
        if constexpr (std::is_floating_point<T>::value) {
            return;
        }
        else if constexpr (bw == sizeof(T) * 8) {
            return;
        }
        else {
            i64 val = (x + (1LL << (bw - 1))) % (1LL << bw);
            val -= (1LL << (bw - 1));
            x = val;
        }
    }
    
    void modbw(Tensor<T> &x) { modbw(x.data, x.size()); }
    void modbw(Tensor1D<T> &x) { modbw(x.data, x.size()); }
    void modbw(Tensor2D<T> &x) { modbw(x.data, x.size()); }
    void modbw(Tensor4D<T> &x) { modbw(x.data, x.size()); }
    void modbw(Tensor5D<T> &x) { modbw(x.data, x.size()); }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmul_triangular(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst);
    void conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output);
    void convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output);

    void relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode);
    void truncate(T &in, u64 shift);
    void div(Tensor<T> &in, T divisor, u64 scale);
    void div(T &in, T divisor, u64 scale);
    u64 log2(u64 x);
    void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out);
    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale);
    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode);

    void batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale);
    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out);
    virtual void gelu(const Tensor<T> &in, const Tensor<T> &out, u64 scale, u64 mode = 0);
    virtual void tanh(const Tensor<T> &in, const Tensor<T> &out, u64 scale);
    virtual void softmax(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0);
    virtual void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale);
    void addbias(Tensor<T> &x, const Tensor1D<T> &bias);
    void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y);
    void scalardiv(Tensor<T> &x, double scalar, Tensor<T> &y, u64 scale, u64 mode);
    void attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y);
    void local_attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y);
    virtual void softmax_triangular(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0);
};
