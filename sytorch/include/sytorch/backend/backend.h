#pragma once

#include <sytorch/tensor.h>
#include <llama/api.h>
#include <llama/assert.h>

#define NOT_IMPLEMENTED { \
        throw std::runtime_error("not implemented");\
}

template <typename T>
class Backend {
public:
    // truncation API
    virtual void truncate(T *in, T *out, u64 shift, u64 size, u8 mode = 0) NOT_IMPLEMENTED;
    
    void truncate(const Tensor<T> &in, const Tensor<T> &out, u64 shift, u8 mode = 0) {
        always_assert(in.is_same_shape(out));
        truncate(in.data, out.data, shift, in.size(), mode);
    }
    
    void truncate(const Tensor4D<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4, mode);
    }

    void truncate(const Tensor<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.size(), mode);
    }

    virtual void truncateForward(const Tensor<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.size(), mode);
    }
    
    void truncate(const Tensor2D<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.d1 * in.d2, mode);
    }
    
    void truncate(const Tensor1D<T> &in, u64 shift, u8 mode = 0) {
        truncate(in.data, in.data, shift, in.d1, mode);
    }

    void truncate(T &in, u64 shift, u8 mode = 0) {
        truncate(&in, &in, shift, 1, mode);
    }

    // matmul API
    virtual void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED;

    // conv API
    virtual void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output) NOT_IMPLEMENTED;
    virtual void conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output) NOT_IMPLEMENTED;
    virtual void convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output) NOT_IMPLEMENTED;
    virtual void convTranspose2D(u64 fh, u64 fw, u64 ph, u64 pw, u64 sh, u64 sw, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output) NOT_IMPLEMENTED;

    // relu API
    virtual void relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) NOT_IMPLEMENTED;

    // leakyrelu API
    virtual void leakyRelu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode, T alpha) NOT_IMPLEMENTED;

    // avgpool API
    virtual void div(Tensor<T> &in, T divisor, u64 scale) NOT_IMPLEMENTED;
    virtual void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) NOT_IMPLEMENTED;
    virtual void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) NOT_IMPLEMENTED;
    
    // maxpool API
    virtual void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) NOT_IMPLEMENTED;

    virtual void batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale) NOT_IMPLEMENTED;
    virtual void signext(Tensor<T> &x, u64 scale) NOT_IMPLEMENTED;

    // add API
    virtual void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out) NOT_IMPLEMENTED;

    virtual void gelu(const Tensor<T> &in, const Tensor<T> &out, u64 scale) NOT_IMPLEMENTED;
    virtual void softmax(Tensor<T> &in, Tensor<T> &out, u64 scale) NOT_IMPLEMENTED;
    virtual void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale) NOT_IMPLEMENTED;
    virtual void addbias(Tensor<T> &x, const Tensor1D<T> &bias) NOT_IMPLEMENTED;
    virtual void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y) NOT_IMPLEMENTED;

    virtual void optimize(LayerGraphNode<T> *root)
    {
        
    }

};
