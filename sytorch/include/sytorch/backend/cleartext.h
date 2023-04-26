#pragma once
#include "backend.h"
#include <sytorch/utils.h>
#include <thread>

template <typename T>
class ClearText : public Backend<T> {
private:
public:
    void truncate(T *in, T *out, u64 shift, u64 size, u8 mode);
    static const bool probablistic = false;
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
    void matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);
    void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c);

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output);
    void conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output);
    void convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output);

    void relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode);
    // void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift);
    // void truncate(const Tensor4D<T> &in, u64 shift);
    // void truncate(const Tensor2D<T> &in, u64 shift);
    // void truncate(const Tensor1D<T> &in, u64 shift);
    void truncate(T &in, u64 shift);
    void div(const Tensor<T> &in, T divisor, u64 scale);
    u64 log2(u64 x);
    void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out);
    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale);
    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode);

    void batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale);
    void add(const std::vector<Tensor<T> *> &in, const Tensor<T> &out);
};
