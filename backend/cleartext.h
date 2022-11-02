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
    static const bool probablistic = false;
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

public:

    static void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }

    static void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d1);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        assert(c.d3 == 1);
        assert(c.d4 == 1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }

    static void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c) {
        assert(a.d1 == b.d1);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(b.d3 == 1);
        assert(b.d4 == 1);
        assert(c.d1 == a.d2);
        assert(c.d2 == b.d2);
    //    c.zero();
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eA(a.data, a.d2, a.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }

    static void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d2);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        assert(c.d3 == 1);
        assert(c.d4 == 1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }

    static void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d2);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }


    static void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output)
    {
        assert(input.d4 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fh * fw * ci);
        u64 newH = (((input.d2 + 2*padding - fh)/stride) + 1);
        u64 newW = (((input.d3 + 2*padding - fw)/stride) + 1);
        assert(output.d1 == input.d1);
        assert(output.d2 == newH);
        assert(output.d3 == newW);
        assert(output.d4 == co);

        Tensor2D<T> reshapedInput = reshapeInputTransposed<T>(input, padding, stride, fh, fw);
        Tensor2D<T> tempOutput(filter.d1, reshapedInput.d1);
        matmulTransposeB(filter, reshapedInput, tempOutput);
        reshapeOutput<T>(tempOutput, input.d1, (((input.d2 + 2*padding - fh)/stride) + 1), (((input.d3 + 2*padding - fw)/stride) + 1), co, output);
    }

    static void conv2DFilterGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, Tensor2D<T> &filter, const Tensor4D<T> &output)
    {
        assert(input.d4 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fh * fw * ci);
        u64 newH = (((input.d2 + 2*padding - fh)/stride) + 1);
        u64 newW = (((input.d3 + 2*padding - fw)/stride) + 1);
        assert(output.d1 == input.d1);
        assert(output.d2 == newH);
        assert(output.d3 == newW);
        assert(output.d4 == co);
        
        Tensor2D<T> tempOutput(co, input.d1 * newH * newW);
        reshapeOutputReversed<T>(tempOutput, input.d1, (((input.d2 + 2*padding - fh)/stride) + 1), (((input.d3 + 2*padding - fw)/stride) + 1), co, output);
        Tensor2D<T> reshapedInput = reshapeInputTransposed(input, padding, stride, fh, fw);
        matmul(tempOutput, reshapedInput, filter);
    }

    static void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad)
    {
        biasGrad.fill(0);
        assert(e.d4 == biasGrad.size);
        for(int i = 0; i < e.d1; i++) {
            for(int j = 0; j < e.d2; j++) {
                for(int k = 0; k < e.d3; k++) {
                    for(int l = 0; l < e.d4; l++) {
                        biasGrad(l) += e(i, j, k, l);
                    }
                }
            }
        }
    }

    static void updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, Tensor2D<T> &Vw, u64 scale) {
        assert(weight.d1 == e.d1);
        assert(weight.d2 == e.d2);
        if (mom_fp == 0) {
            for(int i = 0; i < weight.d1; i++) {
                for(int j = 0; j < weight.d2; j++) {
                    e(i, j) = e(i, j) * lr_fp;
                }
            }
            truncate(e, scale+lr_scale);
            for(u64 i = 0; i < weight.d1; i++) {
                for(u64 j = 0; j < weight.d2; j++) {
                    weight.data[i * weight.d2 + j] -= e(i, j);
                }
            }
        }
        else {
            assert(weight.d1 = Vw.d1);
            assert(weight.d2 = Vw.d2);
            for(int i = 0; i < weight.d1; i++) {
                for(int j = 0; j < weight.d2; j++) {
                    Vw(i, j) = mom_fp * Vw(i, j) + (1ULL<<mom_scale) * e(i, j);
                }
            }
            truncate(Vw, mom_scale);
            for(int i = 0; i < weight.d1; i++) {
                for(int j = 0; j < weight.d2; j++) {
                    weight(i, j) = (1ULL<<(lr_scale+scale)) * weight(i, j) - lr_fp * Vw(i, j);
                }
            }
            truncate(weight, lr_scale+scale);
        }
    }

    static void updateBias(Tensor<T> &bias, const Tensor4D<T> &e, const Tensor<T> &Vb, u64 scale) {
        // assert(e.d1 == 1);
        assert(e.d2 == bias.size);
        assert(e.d3 == 1);
        assert(e.d4 == 1);
        if (mom_fp == 0) {
            for (u64 i = 0; i < bias.size; i++) {
                T sum = 0;
                for(u64 j = 0; j < e.d1; ++j) {
                    sum = sum + e(j, i, 0, 0);
                }
                if (scale > lr_scale) {
                    bias.data[i] -= lr_fp * sum * (1ULL << (scale-lr_scale));
                }
                else {
                    bias.data[i] -= lr_fp * sum / (1ULL << (lr_scale-scale));
                }
            }
        }
        else {
            assert(bias.size == Vb.size);
            if (scale == 0) {
                assert(std::is_floating_point<T>::value);
                for (u64 i = 0; i < bias.size; i++) {
                    T sum = 0;
                    for(u64 j = 0; j < e.d1; ++j) {
                        sum = sum + e(j, i, 0, 0);
                    }
                    Vb(i) = ((mom_fp * Vb(i)) / (1ULL << mom_scale)) + sum;
                    bias(i) = bias(i) - (lr_fp * Vb(i)) / (1UL << lr_scale);
                }
            }
            else {
                for (u64 i = 0; i < bias.size; i++) {
                    T sum = 0;
                    for(u64 j = 0; j < e.d1; ++j) {
                        sum = sum + e(j, i, 0, 0);
                    }
                    Vb(i) = mom_fp * Vb(i) + (1ULL << (scale + mom_scale - lr_scale)) * sum;
                }
                truncate(Vb, mom_scale);
                fastfor(bias.size, [&](u64 i) {
                    bias(i) = bias(i) - lr_fp * Vb(i);
                });
            }
        }
    }

    static void updateBias(Tensor<T> &bias, const Tensor<T> &grad, Tensor<T> &Vb, u64 scale) {
        assert(grad.size == bias.size);
        if (mom_fp == 0) {
            #pragma omp parallel for
            for (u64 i = 0; i < bias.size; i++) {
                if (scale > lr_scale) {

                    bias.data[i] -= lr_fp * grad.data[i] * (1ULL << (scale-lr_scale));
                }
                else {
                    assert(std::is_floating_point<T>::value);
                    bias.data[i] -= lr_fp * grad.data[i] / (1ULL << (lr_scale-scale));
                }
                // bias.data[i] -= lr_fp * grad(i) * (1ULL << (scale-lr_scale));
            }
        }
        else {
            // Scale of Vb would be 2 * scale - lr_scale
            assert(Vb.size == bias.size);
            if (scale == 0) {
                assert(std::is_floating_point<T>::value);
                fastfor(bias.size, [&](u64 i) {
                    Vb(i) = (mom_fp * Vb(i) / (1ULL << mom_scale)) + grad(i);
                    bias(i) = bias(i) - (lr_fp * Vb(i)) / (1UL << lr_scale);
                });
            }
            else {
                fastfor(bias.size, [&](u64 i) {
                    Vb(i) = mom_fp * Vb(i) + (1ULL << (scale + mom_scale - lr_scale)) * grad(i);
                });
                truncate(Vb, mom_scale);
                fastfor(bias.size, [&](u64 i) {
                    bias(i) = bias(i) - lr_fp * Vb(i);
                });
            }
        }
    }

    static void conv2DInputGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, Tensor4D<T> &input, const Tensor2D<T> &filter, const Tensor4D<T> &output)
    {
        if (stride != 1) {
            std::cerr << "Stride not supported in backward pass yet!" << std::endl;
        }
        assert(input.d4 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fh * fw * ci);
        u64 newH = (((input.d2 + 2*padding - fh)/stride) + 1);
        u64 newW = (((input.d3 + 2*padding - fw)/stride) + 1);
        assert(output.d1 == input.d1);
        assert(output.d2 == newH);
        assert(output.d3 == newW);
        assert(output.d4 == co);
        
        Tensor2D<T> transposedFilter(ci, fh * fw * co);
        transposeFilter<T>(fh, fw, ci, co, filter, transposedFilter);
        conv2D(fh, fw, fh-padding-1, stride, co, ci, output, transposedFilter, input);
    }

    static void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        fastfor(in.d1, [&] (u64 i) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? (in(i, j, k, l) >> shift) : 0;
                        if (probablistic) {
                            u64 r = rand() % (1ULL << shift);
                            u64 x0 = ((u64)in(i, j, k, l)) % (1ULL << shift);
                            out(i, j, k, l) += (x0 < r ? 0 : 1); 
                        }
                    }
                }
            }
        });
    }

    static void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        fastfor(in.d1, [&] (u64 i) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                        assert(drelu(i, j, k, l) == 1 || drelu(i, j, k, l) == 0);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                    }
                }
            }
        });
    }

    static void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        fastfor(in.d1, [&] (u64 i) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        assert(drelu(i, j, k, l) == 0 || drelu(i, j, k, l) == 1);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                    }
                }
            }
        });
    }

    static void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        fastfor(in.d1, [&] (u64 i) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        if constexpr (std::is_floating_point<T>::value) {
                            out(i, j, k, l) = in(i, j, k, l) / (1 << shift);
                        } else {
                            out(i, j, k, l) = in(i, j, k, l) >> shift;
                            if (probablistic) {
                                u64 r = rand() % (1ULL << shift);
                                u64 x0 = ((u64)in(i, j, k, l)) % (1ULL << shift);
                                out(i, j, k, l) += (x0 < r ? 0 : 1); 
                            }
                        }
                    }
                }
            }
        });
    }

    static void truncate(const Tensor4D<T> &in, u64 shift) {
        // Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2 * in.d3 * in.d4);
        // eA = eA / ((T)(1LL << shift));
        truncate(in, in, shift);
    }

    static void truncate(const Tensor2D<T> &in, u64 shift) {
    //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
    //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
        fastfor(in.d1, [&] (u64 i) {
            for (u64 j = 0; j < in.d2; j++) {
                if constexpr (std::is_floating_point<T>::value) {
                    in(i, j) = in(i, j) / ((T)(1ULL << shift));
                } else {
                    u64 x0 = ((u64)in(i, j)) % (1ULL << shift);
                    in(i, j) = in(i, j) >> shift;
                    if (probablistic) {
                        u64 r = rand() % (1ULL << shift);
                        in(i, j) += (x0 < r ? 0 : 1); 
                    }
                }
            }
        });
    }

    static void truncate(const Tensor<T> &in, u64 shift) {
    //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
    //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
        fastfor(in.size, [&] (u64 i) {
            if constexpr (std::is_floating_point<T>::value) {
                in(i) = in(i) / ((T)(1ULL << shift));
            } else {
                u64 x0 = ((u64)in(i)) % (1ULL << shift);
                in(i) = in(i) >> shift;
                if (probablistic) {
                    u64 r = rand() % (1ULL << shift);
                    in(i) += (x0 < r ? 0 : 1); 
                }
            }
        });
    }

    static void div(const Tensor4D<T> &in, T divisor) {
        fastfor(in.d1, [&] (u64 i) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        in(i, j, k, l) = in(i, j, k, l) / divisor;
                    }
                }
            }
        });
    }

    static void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        fastfor(in.d1, [&] (int i) {
            for(int j = 0; j < newH; j++) {
                for(int k = 0; k < newW; k++) {
                    for(int l = 0; l < in.d4; l++) {
                        T sum = 0;
                        for(int m = 0; m < ks; m++) {
                            for(int n = 0; n < ks; n++) {
                                sum += in(i, j*stride+m, k*stride+n, l);
                            }
                        }
                        out(i, j, k, l) = sum;
                    }
                }
            }
        });
    }

    static void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) {
        sumPool2D(ks, padding, stride, in, out);
        div(out, (T)(ks*ks));
    }

    static u64 log2(u64 x) {
        u64 y = 0;
        while (x >>= 1) y++;
        return y;
    }

    static void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        in.zero();
        fastfor(in.d1, [&] (int i) {
            for(int j = 0; j < newH; j++) {
                for(int k = 0; k < newW; k++) {
                    for(int l = 0; l < in.d4; l++) {
                        for(int m = 0; m < ks; m++) {
                            for(int n = 0; n < ks; n++) {
                                in(i, j*stride+m, k*stride+n, l) += out(i, j, k, l);
                            }
                        }
                    }
                }
            }
        });
        // hack for piranha
        truncate(in, log2(ks * ks));
    }

    static void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, u64 scale) {
        sumPool2DInputGrad(ks, padding, stride, in, out);
        div(in, (T)(ks*ks));
    }

    static void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        fastfor(in.d1, [&](int i) {
            for(int j = 0; j < newH; j++) {
                for(int k = 0; k < newW; k++) {
                    for(int l = 0; l < in.d4; l++) {
                        T max = std::numeric_limits<T>::lowest();
                        u64 maxIdxI = 0;
                        u64 maxIdxJ = 0;
                        for(int m = 0; m < ks; m++) {
                            for(int n = 0; n < ks; n++) {
                                T val = in(i, j*stride+m, k*stride+n, l);
                                if(val > max) {
                                    max = val;
                                    maxIdxI = m;
                                    maxIdxJ = n;
                                }
                            }
                        }
                        out(i, j, k, l) = max;
                        maxIdx(i, j, k, l) = maxIdxI * ks + maxIdxJ;
                    }
                }
            }
        });
    }

    static void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        in.zero();
        fastfor(in.d1, [&](int i) {
            for(int j = 0; j < newH; j++) {
                for(int k = 0; k < newW; k++) {
                    for(int l = 0; l < in.d4; l++) {
                        u64 maxIdxI = maxIdx(i, j, k, l) / ks;
                        u64 maxIdxJ = maxIdx(i, j, k, l) % ks;
                        in(i, j*stride+maxIdxI, k*stride+maxIdxJ, l) += out(i, j, k, l);
                    }
                }
            }
        });
    }

};
