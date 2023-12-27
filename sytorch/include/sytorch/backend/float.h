#pragma once
#include "backend.h"
#include <sytorch/utils.h>
#include <thread>

template <typename T>
class FloatClearText : public Backend<T>
{
private:
public:
    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode);

    template <typename Functor>
    void fastfor(u64 size, Functor f)
    {
//#pragma omp parallel for
        for (u64 i = 0; i < size; i++)
        {
            f(i);
        }
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmul_triangular(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output);
    void conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output);
    void convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output);
    void convTranspose2D(u64 fh, u64 fw, u64 ph, u64 pw, u64 sh, u64 sw, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output);

    void relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode);
    void leakyRelu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode, T alpha);

    void truncate(T &in, u64 shift);
    void div(Tensor<T> &in, T divisor, u64 scale);
    void div(T &in, T divisor, u64 scale);
    void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out);
    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale);
    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode);

    void batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale);
    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out);
    void gelu(const Tensor<T> &in, const Tensor<T> &out, u64 scale, u64 mode = 0);
    void tanh(const Tensor<T> &in, const Tensor<T> &out, u64 scale);
    void softmax(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0);
    void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale);
    void addbias(Tensor<T> &x, const Tensor1D<T> &bias);
    void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y);
    void scalardiv(Tensor<T> &x, double scalar, Tensor<T> &y, u64 scale, u64 mode);
    void attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y);
    void softmax_triangular(Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0);
};