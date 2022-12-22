#pragma once
#include "../utils.h"
#include <thread>

template <typename T>
class ClearText {
public:
    static const u64 lr_fp = 1;
    static const u64 lr_scale = 6;
    static const u64 mom_fp = 29;
    static const u64 mom_scale = 5;
    static const bool probablistic = true;
    static const bool numThreads = 1;

    template <typename Functor>
    static void fastfor(u64 size, Functor f)
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

    static void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    static void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c);
    static void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c);
    static void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c);
    static void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);

    static void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output);
    static void conv2DFilterGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, Tensor2D<T> &filter, const Tensor4D<T> &output);
    static void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad);
    static void conv2DInputGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, Tensor4D<T> &input, const Tensor2D<T> &filter, const Tensor4D<T> &output);

    static void updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, Tensor2D<T> &Vw, u64 scale);
    static void updateWeight(Tensor<T> &weight, const Tensor<T> &e, Tensor<T> &Vw, u64 scale);
    static void updateBias(Tensor<T> &bias, const Tensor4D<T> &e, const Tensor<T> &Vb, u64 scale);
    static void updateBias(Tensor<T> &bias, const Tensor<T> &grad, Tensor<T> &Vb, u64 scale);

    static void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift);
    static void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu);
    static void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out);
    static void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift);
    static void truncate(const Tensor4D<T> &in, u64 shift);
    static void truncate(const Tensor2D<T> &in, u64 shift);
    static void truncate(const Tensor<T> &in, u64 shift);
    static void truncate(T &in, u64 shift);
    static void div(const Tensor4D<T> &in, T divisor);
    static u64 log2(u64 x);
    static void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out);
    static void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale);
    static void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out);
    static void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, u64 scale);
    static void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx);
    static void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const      Tensor4D<u64> &maxIdx);
    static Tensor2D<T> channelReshape(const Tensor4D<T> &x);
    static void batchNorm2dForwardTrain(const Tensor4D<T> &in, Tensor4D<T> &out, const Tensor<T> &running_mean, const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta, Tensor4D<T> &x_normalized, Tensor<T> &invstd, u64 scale);
    static void batchNorm2dForwardTest(const Tensor4D<T> &in, Tensor4D<T> &out, const Tensor<T> &running_mean, const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta, u64 scale);
    static void batchNorm2dBackward(Tensor4D<T> &din, const Tensor4D<T> &dout, Tensor<T> &dgamma, Tensor<T> &dbeta, const Tensor4D<T> &normalized, const Tensor<T> &gamma, const Tensor<T> &invstd, u64 scale);
};
