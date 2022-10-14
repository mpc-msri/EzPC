#pragma once
#include "../utils.h"

template <typename T>
class ClearText {

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

        Tensor2D<T> reshapedInput = reshapeInput<T>(input, padding, stride, fh, fw);
        Tensor2D<T> tempOutput(filter.d1, reshapedInput.d2);
        matmul(filter, reshapedInput, tempOutput);
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
        Tensor2D<T> reshapedInput = reshapeInput(input, padding, stride, fh, fw);
        matmulTransposeB(tempOutput, reshapedInput, filter);
    }

    static void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad)
    {
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
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? (in(i, j, k, l) >> shift) : 0;
                    }
                }
            }
        }
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
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                        assert(drelu(i, j, k, l) == 1 || drelu(i, j, k, l) == 0);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                    }
                }
            }
        }
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
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        assert(drelu(i, j, k, l) == 0 || drelu(i, j, k, l) == 1);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                    }
                }
            }
        }
    }

    static void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        if constexpr (std::is_floating_point<T>::value) {
                            out(i, j, k, l) = in(i, j, k, l) / (1 << shift);
                        } else {
                            out(i, j, k, l) = in(i, j, k, l) >> shift;
                        }
                    }
                }
            }
        }
    }

    static void truncate(const Tensor4D<T> &in, u64 shift) {
        // Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2 * in.d3 * in.d4);
        // eA = eA / ((T)(1LL << shift));
        truncate(in, in, shift);
    }

    static void truncate(const Tensor2D<T> &in, u64 shift) {
    //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
    //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                if constexpr (std::is_floating_point<T>::value) {
                    in(i, j) = in(i, j) / ((T)(1ULL << shift));
                } else {
                    in(i, j) = in(i, j) >> shift;
                }
            }
        }
    }

    static void div(const Tensor4D<T> &in, T divisor) {
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        in(i, j, k, l) = in(i, j, k, l) / divisor;
                    }
                }
            }
        }
    }

    static void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        for(int i = 0; i < in.d1; i++) {
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
        }
    }

    static void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
        sumPool2D(ks, padding, stride, in, out);
        div(out, (T)(ks*ks));
    }

    static void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        in.zero();
        for(int i = 0; i < in.d1; i++) {
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
        }
    }

    static void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) {
        sumPool2DInputGrad(ks, padding, stride, in, out);
        div(in, (T)(ks*ks));
    }

};
