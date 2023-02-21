#pragma once
#include "backend.h"
#include <sytorch/utils.h>
#include <thread>

template <typename T>
class ClearText : public Backend<T> {
private:
public:
    void truncate(T *in, T *out, u64 shift, u64 size);
    static const u64 lr_fp = 1;
    static const u64 lr_scale = 6;
    static const u64 mom_fp = 29;
    static const u64 mom_scale = 5;
    static const bool probablistic = true;
    static const bool localTruncationEmulation = false;
    static const bool numThreads = 120;

    template <typename Functor>
    void fastfor(u64 size, Functor f)
    {
        if (numThreads == 1) {
            for (u64 i = 0; i < size; i++) {
                f(i);
            }
        }
        else {
            std::thread threads[numThreads];
            u64 chunkSize = size / numThreads;
            for (u64 i = 0; i < numThreads - 1; i++) {
                threads[i] = std::thread([=, &f]() {
                    for (u64 j = i * chunkSize; j < (i + 1) * chunkSize; j++) {
                        f(j);
                    }
                });
            }
            threads[numThreads-1] = std::thread([=, &f]() {
                for (u64 j = (numThreads - 1) * chunkSize; j < size; j++) {
                    f(j);
                }
            });
        }
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c);
    void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c);
    void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c);
    void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output);
    void conv2DFilterGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, Tensor2D<T> &filter, const Tensor4D<T> &output);
    void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad);
    void conv2DInputGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, Tensor4D<T> &input, const Tensor2D<T> &filter, const Tensor4D<T> &output);

    void updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, Tensor2D<T> &Vw, u64 scale);
    void updateWeight(Tensor<T> &weight, const Tensor<T> &e, Tensor<T> &Vw, u64 scale);
    void updateBias(Tensor<T> &bias, const Tensor4D<T> &e, const Tensor<T> &Vb, u64 scale);
    void updateBias(Tensor<T> &bias, const Tensor<T> &grad, Tensor<T> &Vb, u64 scale);

    void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift);
    void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale);
    void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out);
    void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift);
    void truncate(const Tensor4D<T> &in, u64 shift);
    void truncate(const Tensor2D<T> &in, u64 shift);
    void truncate(const Tensor<T> &in, u64 shift);
    void truncate(T &in, u64 shift);
    void div(const Tensor4D<T> &in, T divisor);
    u64 log2(u64 x);
    void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out);
    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale);
    void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out);
    void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, u64 scale);
    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale);
    void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const      Tensor4D<u64> &maxIdx);
    Tensor2D<T> channelReshape(const Tensor4D<T> &x);
    void batchNorm2dForwardTrain(const Tensor4D<T> &in, Tensor4D<T> &out, const Tensor<T> &running_mean, const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta, Tensor4D<T> &x_normalized, Tensor<T> &invstd, u64 scale);
    void batchNorm2dForwardTest(const Tensor4D<T> &in, Tensor4D<T> &out, const Tensor<T> &running_mean, const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta, u64 scale);
    void batchNorm2dBackward(Tensor4D<T> &din, const Tensor4D<T> &dout, Tensor<T> &dgamma, Tensor<T> &dbeta, const Tensor4D<T> &normalized, const Tensor<T> &gamma, const Tensor<T> &invstd, u64 scale);

    void batchNorm2dInference(const Tensor<T> &A, const Tensor<T> &B, const Tensor4D<T> &x, Tensor4D<T> &y, u64 scale);
};
