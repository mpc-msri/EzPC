
#include "cleartext.h"

template <typename T>
void ClearText<T>::matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void ClearText<T>::matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
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

template <typename T>
void ClearText<T>::matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c) {
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

template <typename T>
void ClearText<T>::matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
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

template <typename T>
void ClearText<T>::matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d2);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
}

template <typename T>
void ClearText<T>::conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output)
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

template <typename T>
void ClearText<T>::conv2DFilterGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, Tensor2D<T> &filter, const Tensor4D<T> &output)
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

template <typename T>
void ClearText<T>::conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad)
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

template <typename T>
void ClearText<T>::updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, Tensor2D<T> &Vw, u64 scale) {
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

template <typename T>
void ClearText<T>::updateWeight(Tensor<T> &weight, const Tensor<T> &e, Tensor<T> &Vw, u64 scale) {
    assert(weight.size == e.size);
    if (mom_fp == 0) {
        for(int i = 0; i < weight.size; i++) {
            e(i) = e(i) * lr_fp;
        }
        truncate(e, scale+lr_scale);
        for(u64 i = 0; i < weight.size; i++) {
            weight(i) -= e(i);
        }
    }
    else {
        assert(weight.size = Vw.size);
        for(int i = 0; i < weight.size; i++) {
            Vw(i) = mom_fp * Vw(i) + (1ULL<<mom_scale) * e(i);
        }
        truncate(Vw, mom_scale);
        for(int i = 0; i < weight.size; i++) {
            weight(i) = (1ULL<<(lr_scale+scale)) * weight(i) - lr_fp * Vw(i);
        }
        truncate(weight, lr_scale+scale);
    }
}

template <typename T>
void ClearText<T>::updateBias(Tensor<T> &bias, const Tensor4D<T> &e, const Tensor<T> &Vb, u64 scale) {
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

template <typename T>
void ClearText<T>::updateBias(Tensor<T> &bias, const Tensor<T> &grad, Tensor<T> &Vb, u64 scale) {
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

template <typename T>
void expandOutputConv(const Tensor4D<T> &output, u64 stride, Tensor4D<T> &outputExpanded) {
    // put elements with a gap of stride-1 in between
    assert(outputExpanded.d1 == output.d1);
    assert(outputExpanded.d2 == output.d2 + (output.d2 - 1) * (stride - 1));
    assert(outputExpanded.d3 == output.d3 + (output.d3 - 1) * (stride - 1));
    assert(outputExpanded.d4 == output.d4);

    outputExpanded.fill(0);
    for (u64 i = 0; i < output.d1; i++) {
        for (u64 j = 0; j < output.d2; j++) {
            for (u64 k = 0; k < output.d3; k++) {
                for (u64 l = 0; l < output.d4; l++) {
                    outputExpanded(i, j*stride, k*stride, l) = output(i, j, k, l);
                }
            }
        }
    }
}

template <typename T>
void ClearText<T>::conv2DInputGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, Tensor4D<T> &input, const Tensor2D<T> &filter, const Tensor4D<T> &output)
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
    if (stride == 1) {
        conv2D(fh, fw, fh-padding-1, 1, co, ci, output, transposedFilter, input);
    }
    else {
        Tensor4D<T> outputExpanded(input.d1, input.d2 + (input.d2 - 1) * (stride - 1), input.d3 + (input.d3 - 1) * (stride - 1), co);
        expandOutputConv(output, stride, outputExpanded);
        conv2D(fh, fw, fh-padding-1, 1, co, ci, outputExpanded, transposedFilter, input);
    }
}

template <typename T>
void ClearText<T>::relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) {
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
                    if(localTruncationEmulation) {
                        if (drelu(i, j, k, l) == 0) {
                            out(i, j, k, l) = 0;
                        }
                        else {
                            u64 a = prngStr.get<T>();
                            u64 b = ((u64)in(i, j, k, l)) - a;
                            a = a >> shift;
                            b = -((-b) >> shift);
                            out(i, j, k, l) = a + b;
                        }
                        continue;
                    }
                    out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? (in(i, j, k, l) / (1LL << shift)) : 0;
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

template <typename T>
void ClearText<T>::relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale) {
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

template <typename T>
void ClearText<T>::select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) {
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

template <typename T>
void ClearText<T>::truncate(T *in, T *out, u64 shift, u64 size) {
    fastfor(size, [&] (u64 i) {
        if constexpr (std::is_floating_point<T>::value) {
            out[i] = in[i] / ((T)(1ULL << shift));
        } else {
            if(localTruncationEmulation) {
                u64 a = prngStr.get<u64>();
                u64 b = ((u64)in[i]) - a;
                a = a >> shift;
                b = -((-b) >> shift);
                out[i] = a + b;
                return;
            }
            u64 x0 = ((u64)in[i]) % (1ULL << shift);
            in[i] = in[i] >> shift;
            if (probablistic) {
                u64 r = rand() % (1ULL << shift);
                out[i] += (x0 < r ? 0 : 1); 
            }
        }
    });
}

template <typename T>
void ClearText<T>::truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    truncate(in.data, out.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
}

template <typename T>
void ClearText<T>::truncate(const Tensor4D<T> &in, u64 shift) {
    // Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2 * in.d3 * in.d4);
    // eA = eA / ((T)(1LL << shift));
    truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
}

template <typename T>
void ClearText<T>::truncate(const Tensor2D<T> &in, u64 shift) {
//    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
//    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
    truncate(in.data, in.data, shift, in.d1 * in.d2);
}

template <typename T>
void ClearText<T>::truncate(const Tensor<T> &in, u64 shift) {
//    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
//    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
    truncate(in.data, in.data, shift, in.size);
}

template <typename T>
void ClearText<T>::truncate(T &in, u64 shift) {
    if constexpr (std::is_floating_point<T>::value) {
        in = in / ((T)(1ULL << shift));
    } else {
        if(localTruncationEmulation) {
            u64 a = prngStr.get<T>();
            u64 b = ((u64)in) - a;
            a = a >> shift;
            b = -((-b) >> shift);
            in = a + b;
            return;
        }
        u64 x0 = ((u64)in) % (1ULL << shift);
        in = in >> shift;
        if (probablistic) {
            u64 r = rand() % (1ULL << shift);
            in += (x0 < r ? 0 : 1); 
        }
    }
}

template <typename T>
void ClearText<T>::div(const Tensor4D<T> &in, T divisor) {
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

template <typename T>
void ClearText<T>::sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
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

template <typename T>
void ClearText<T>::avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) {
    sumPool2D(ks, padding, stride, in, out);
    div(out, (T)(ks*ks));
}

template <typename T>
u64 ClearText<T>::log2(u64 x) {
    u64 y = 0;
    while (x >>= 1) y++;
    return y;
}

template <typename T>
void ClearText<T>::sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) {
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

template <typename T>
void ClearText<T>::avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, u64 scale) {
    sumPool2DInputGrad(ks, padding, stride, in, out);
    div(in, (T)(ks*ks));
}

template <typename T>
void ClearText<T>::maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale) {
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

template <typename T>
void ClearText<T>::maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) {
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

template <typename T>
void ClearText<T>::batchNorm2dForwardTrain(const Tensor4D<T> &in, Tensor4D<T> &out, 
    const Tensor<T> &running_mean, const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta,
    Tensor4D<T> &x_normalized, Tensor<T> &invstd, u64 scale) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d4 == gamma.size);
    assert(in.d4 == beta.size);
    assert(in.d4 == running_mean.size);
    assert(in.d4 == running_var.size);
    assert(!((scale == 0) ^ std::is_floating_point<T>::value));

    Tensor<T> mean(in.d4); mean.fill(0);
    Tensor<T> var(in.d4); var.fill(0);
    i64 m = in.d1 * in.d2 * in.d3;
    
    fastfor(in.d4, [&](int l) {
        for(int i = 0; i < in.d1; i++) {
            for(int j = 0; j < in.d2; j++) {
                for(int k = 0; k < in.d3; k++) {
                    mean(l) += in(i, j, k, l);
                }
            }
        }
        mean(l) /= m;

        for(int i = 0; i < in.d1; i++) {
            for(int j = 0; j < in.d2; j++) {
                for(int k = 0; k < in.d3; k++) {
                    var(l) += (in(i, j, k, l) - mean(l)) * (in(i, j, k, l) - mean(l));
                }
            }
        }

        truncate(var(l), scale);

        if constexpr (!std::is_floating_point<T>::value) {
            double var_d = (var(l) / m);
            var_d /= (1LL << scale); // fix2float
            invstd(l) = (1LL << scale) / sqrt(var_d + 1e-5);
        } else {
            invstd(l) = 1 / sqrt((var(l) / m) + 1e-5);
        }

        running_mean(l) = 29 * running_mean(l) + 3 * mean(l);
        running_mean(l) /= 32;
        running_var(l) = 29 * running_var(l) + 3 * (var(l) / (m - 1));
        running_var(l) /= 32;


        for(int i = 0; i < in.d1; i++) {
            for(int j = 0; j < in.d2; j++) {
                for(int k = 0; k < in.d3; k++) {
                    x_normalized(i, j, k, l) = (in(i, j, k, l) - mean(l)) * invstd(l);
                    truncate(x_normalized(i, j, k, l), scale);
                    out(i, j, k, l) = gamma(l) * x_normalized(i, j, k, l) + beta(l);
                    truncate(out(i, j, k, l), scale);
                }
            }
        }
    });
}

template <typename T>
void ClearText<T>::batchNorm2dForwardTest(const Tensor4D<T> &in, Tensor4D<T> &out, const Tensor<T> &running_mean, 
    const Tensor<T> &running_var, const Tensor<T> &gamma, const Tensor<T> &beta, u64 scale) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d4 == gamma.size);
    assert(in.d4 == beta.size);
    assert(in.d4 == running_mean.size);
    assert(in.d4 == running_var.size);

    fastfor(in.d4, [&](int l) {
        T invstd;
        if constexpr (!std::is_floating_point<T>::value) {
            double var_d = running_var(l);
            var_d /= (1LL << scale); // fix2float
            invstd = (1LL << scale) / sqrt(var_d + 1e-5);
        } else {
            invstd = 1 / sqrt(running_var(l) + 1e-5);
        }
        for(int i = 0; i < in.d1; i++) {
            for(int j = 0; j < in.d2; j++) {
                for(int k = 0; k < in.d3; k++) {
                    out(i, j, k, l) = gamma(l) * (in(i, j, k, l) - running_mean(l));
                    if constexpr (!std::is_floating_point<T>::value)
                        out(i, j, k, l) /= (1LL << scale); // due to multiplication
                    out(i, j, k, l) *= invstd;
                    out(i, j, j, l) += beta(l);
                    if constexpr (!std::is_floating_point<T>::value)
                        out(i, j, k, l) /= (1LL << scale); // due to multiplication
                }
            }
        }
    });
}

template <typename T>
Tensor2D<T> ClearText<T>::channelReshape(const Tensor4D<T> &x) {
    Tensor2D<T> res(x.d4, x.d1 * x.d2 * x.d3);
    for(int i = 0; i < x.d1; i++) {
        for(int j = 0; j < x.d2; j++) {
            for(int k = 0; k < x.d3; k++) {
                for(int l = 0; l < x.d4; l++) {
                    res(l, i*x.d2*x.d3 + j*x.d3 + k) = x(i, j, k, l);
                }
            }
        }
    }
    return res;
}

template <typename T>
void ClearText<T>::batchNorm2dBackward(Tensor4D<T> &din, const Tensor4D<T> &dout, Tensor<T> &dgamma, Tensor<T> &dbeta,
    const Tensor4D<T> &normalized, const Tensor<T> &gamma, const Tensor<T> &invstd, u64 scale) {
    const i64 M = din.d1 * din.d2 * din.d3;
    const u64 C = din.d4;
    assert(din.d1 == dout.d1);
    assert(din.d2 == dout.d2);
    assert(din.d3 == dout.d3);
    assert(din.d4 == dout.d4);
    assert(C == dgamma.size);
    assert(C == dbeta.size);
    Tensor2D<T> dinReshaped(C, M);
    Tensor2D<T> doutReshaped = channelReshape(dout);
    Tensor2D<T> xcap = channelReshape(normalized);
    Tensor2D<T> dxcap(C, M);
    dgamma.fill(0);
    dbeta.fill(0);

    fastfor(C, [&](int c) {
        for(int i = 0; i < M; ++i) {
            dgamma(c) += doutReshaped(c, i) * xcap(c, i);
            dbeta(c) += doutReshaped(c, i);
            dxcap(c, i) = doutReshaped(c, i) * gamma(c);
            truncate(dxcap(c, i), scale);
        }

        T tmp1 = 0;
        T tmp2 = 0;
        for(int i = 0; i < M; ++i) {
            tmp1 += dxcap(c, i);
            tmp2 += dxcap(c, i) * xcap(c, i);
        }
        truncate(tmp2, scale);

        for(int i = 0; i < M; ++i) {
            dinReshaped(c, i) = - xcap(c, i) * tmp2;
            truncate(dinReshaped(c, i), scale);
            dinReshaped(c, i) += M * dxcap(c, i) - tmp1;
            dinReshaped(c, i) = (dinReshaped(c, i) * invstd(c)) / M;
            truncate(dinReshaped(c, i), scale);
        }
    });

    for(int i = 0; i < din.d1; i++) {
        for(int j = 0; j < din.d2; j++) {
            for(int k = 0; k < din.d3; k++) {
                for(int l = 0; l < din.d4; l++) {
                    din(i, j, k, l) = dinReshaped(l, i*din.d2*din.d3 + j*din.d3 + k);
                }
            }
        }
    }
}

template class ClearText<i64>;
template class ClearText<i32>;
template class ClearText<u64>;
template class ClearText<double>;
template class ClearText<float>;
