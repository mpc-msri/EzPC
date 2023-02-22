#pragma once

#include <sytorch/tensor.h>
#include <llama/api.h>

#define NOT_IMPLEMENTED { \
        throw std::runtime_error("not implemented");\
}

template <typename T>
class Backend {
public:
    // truncation API
    virtual void truncate(T *in, T *out, u64 shift, u64 size) NOT_IMPLEMENTED;
    
    void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
        always_assert(in.d1 == out.d1);
        always_assert(in.d2 == out.d2);
        always_assert(in.d3 == out.d3);
        always_assert(in.d4 == out.d4);
        truncate(in.data, out.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
    }
    
    void truncate(const Tensor4D<T> &in, u64 shift) {
        truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
    }

    virtual void truncateForward(const Tensor4D<T> &in, u64 shift) {
        truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
    }
    
    void truncate(const Tensor2D<T> &in, u64 shift) {
        truncate(in.data, in.data, shift, in.d1 * in.d2);
    }
    
    void truncate(const Tensor<T> &in, u64 shift) {
        truncate(in.data, in.data, shift, in.size);
    }

    void truncate(T &in, u64 shift) {
        truncate(&in, &in, shift, 1);
    }

    // matmul API
    virtual void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED;
    virtual void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) NOT_IMPLEMENTED;
    virtual void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED;
    virtual void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) NOT_IMPLEMENTED;
    // virtual void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) NOT_IMPLEMENTED;

    // conv API
    virtual void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output) NOT_IMPLEMENTED;
    virtual void conv2DFilterGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, Tensor2D<T> &filter, const Tensor4D<T> &output) NOT_IMPLEMENTED;
    virtual void conv2DInputGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, Tensor4D<T> &input, const Tensor2D<T> &filter, const Tensor4D<T> &output) NOT_IMPLEMENTED;
    virtual void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad) NOT_IMPLEMENTED;
    
    virtual void updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, Tensor2D<T> &Vw, u64 scale) NOT_IMPLEMENTED;
    virtual void updateBias(Tensor<T> &bias, const Tensor4D<T> &e, const Tensor<T> &Vb, u64 scale) NOT_IMPLEMENTED;
    virtual void updateBias(Tensor<T> &bias, const Tensor<T> &grad, Tensor<T> &Vb, u64 scale) NOT_IMPLEMENTED;

    // relu API
    virtual void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) NOT_IMPLEMENTED;
    virtual void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale) NOT_IMPLEMENTED;
    virtual void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) NOT_IMPLEMENTED;

    // avgpool API
    virtual void div(const Tensor4D<T> &in, T divisor) NOT_IMPLEMENTED;
    virtual void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) NOT_IMPLEMENTED;
    virtual void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) NOT_IMPLEMENTED;
    virtual void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) NOT_IMPLEMENTED;
    virtual void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, u64 scale) NOT_IMPLEMENTED;
    
    // maxpool API
    virtual void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale) NOT_IMPLEMENTED;;
    virtual void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) NOT_IMPLEMENTED;;

    // batchnorm API
    virtual void batchNorm2dForwardTrain(const Tensor4D<T> &in, Tensor4D<T> &out, const Tensor<T> &running_mean, const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta, Tensor4D<T> &x_normalized, Tensor<T> &invstd, u64 scale) NOT_IMPLEMENTED;
    virtual void batchNorm2dForwardTest(const Tensor4D<T> &in, Tensor4D<T> &out, const Tensor<T> &running_mean, const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta, u64 scale) NOT_IMPLEMENTED;
    virtual void batchNorm2dBackward(Tensor4D<T> &din, const Tensor4D<T> &dout, Tensor<T> &dgamma, Tensor<T> &dbeta, const Tensor4D<T> &normalized, const Tensor<T> &gamma, const Tensor<T> &invstd, u64 scale) NOT_IMPLEMENTED;
    virtual void updateWeight(Tensor<T> &weight, const Tensor<T> &e, Tensor<T> &Vw, u64 scale) NOT_IMPLEMENTED; // needed for batchNorm

    virtual void batchNorm2dInference(const Tensor<T> &A, const Tensor<T> &B, const Tensor4D<T> &x, Tensor4D<T> &y, u64 scale) NOT_IMPLEMENTED;

    virtual void optimize(LayerTreeNode<T> *root)
    {
        
    }

};
