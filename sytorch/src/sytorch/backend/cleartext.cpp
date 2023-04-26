
#include <sytorch/backend/cleartext.h>
#include <Eigen/Dense>

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
void ClearText<T>::matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d1 == b.d1);
    assert(c.d1 == a.d2);
    assert(c.d2 == b.d2);
//    c.zero();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eA(a.data, a.d2, a.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
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
void ClearText<T>::conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
{
    assert(input.d5 == ci);
    assert(filter.d1 == co);
    assert(filter.d2 == fd * fh * fw * ci);
    always_assert(dd == 1);
    always_assert(dh == 1);
    always_assert(dw == 1);
    u64 newD = (((input.d2 + 2*pd - fd - (fd-1)*(dd-1))/sd) + 1);
    u64 newH = (((input.d3 + 2*ph - fh - (fh-1)*(dh-1))/sh) + 1);
    u64 newW = (((input.d4 + 2*pw - fw - (fw-1)*(dw-1))/sw) + 1);
    assert(output.d1 == input.d1);
    assert(output.d2 == newD);
    assert(output.d3 == newH);
    assert(output.d4 == newW);
    assert(output.d5 == co);

    Tensor2D<T> reshapedInput = reshapeInputTransposed3d<T>(input, pd, ph, pw, sd, sh, sw, fd, fh, fw);
    Tensor2D<T> tempOutput(filter.d1, reshapedInput.d1);
    matmulTransposeB(filter, reshapedInput, tempOutput);
    reshapeOutput3d<T>(tempOutput, input.d1, newD, newH, newW, co, output);
}

template <typename T>
void ClearText<T>::convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
    {
        assert(input.d5 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fd * fh * fw * ci);
        u64 newD = (((input.d2 - 1)*sd + fd - 2*pd));
        u64 newH = (((input.d3 - 1)*sh + fh - 2*ph));
        u64 newW = (((input.d4 - 1)*sw + fw - 2*pw));
        assert(output.d1 == input.d1);
        assert(output.d2 == newD);
        assert(output.d3 == newH);
        assert(output.d4 == newW);
        assert(output.d5 == co);

        convTranspose3dLoop<T>(input.d1, input.d2, input.d3, input.d4, input.d5, fd, fh, fw, co, 
            pd, pd, ph, ph, pw, pw, sd, sh, sw,
            output.d2, output.d3, output.d4, input.data, filter.data, output.data);
    }

template <typename T>
void ClearText<T>::relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) {
    assert(in.is_same_shape(out));
    assert(in.is_same_shape(drelu));
    fastfor(in.size(), [&] (u64 i) {
        drelu.data[i] = (T)(in.data[i] > 0);
        assert(drelu.data[i] == 1 || drelu.data[i] == 0);
        out.data[i] = (drelu.data[i] == 1) ? in.data[i] : 0;
    });
}

template <typename T>
void ClearText<T>::truncate(T *in, T *out, u64 shift, u64 size, u8 mode) {
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

// template <typename T>
// void ClearText<T>::truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
//     assert(in.d1 == out.d1);
//     assert(in.d2 == out.d2);
//     assert(in.d3 == out.d3);
//     assert(in.d4 == out.d4);
//     truncate(in.data, out.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
// }

// template <typename T>
// void ClearText<T>::truncate(const Tensor4D<T> &in, u64 shift) {
//     // Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2 * in.d3 * in.d4);
//     // eA = eA / ((T)(1LL << shift));
//     truncate(in.data, in.data, shift, in.d1 * in.d2 * in.d3 * in.d4);
// }

// template <typename T>
// void ClearText<T>::truncate(const Tensor2D<T> &in, u64 shift) {
// //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
// //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
//     truncate(in.data, in.data, shift, in.d1 * in.d2);
// }

// template <typename T>
// void ClearText<T>::truncate(const Tensor1D<T> &in, u64 shift) {
// //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
// //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
//     truncate(in.data, in.data, shift, in.size);
// }

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
void ClearText<T>::div(const Tensor<T> &in, T divisor, u64 scale) {
    // fastfor(in.size(), [&] (u64 i) {
    //     in.data[i] = in.data[i] / divisor;
    // });

    T divfp = (1LL << scale) / divisor;
    u64 sz = in.size();
    fastfor(in.size(), [&] (u64 i) {
        in.data[i] *= divfp;
    });
    Backend<T>::truncate(in, scale, 3);
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
    div(out.as_nd(), (T)(ks*ks), scale);
}

template <typename T>
u64 ClearText<T>::log2(u64 x) {
    u64 y = 0;
    while (x >>= 1) y++;
    return y;
}

template <typename T>
void ClearText<T>::maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) {
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
                            auto h2 = j*stride+m-padding;
                            auto w2 = k*stride+n-padding;
                            T val = 0;
                            if (h2 < in.d2 && w2 < in.d3 && h2 >= 0 && w2 >= 0)
                                val = in(i, h2, w2, l);
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
void ClearText<T>::batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
{
    assert(A.size == B.size);
    assert(A.size == x.shape.back());
    assert(x.is_same_shape(y));
    u64 channels = x.shape.back();

    fastfor(x.size(), [&](u64 i) {
        y.data[i] = x.data[i] * A(i % channels) + B(i % channels);
    });
}

template <typename T>
void ClearText<T>::add(const std::vector<Tensor<T> *> &in, const Tensor<T> &out)
{
    always_assert(in.size() > 0);
    always_assert(out.size() == in[0]->size());
    for (int i = 0; i < in.size(); i++) {
        always_assert(out.size() == in[i]->size());
    }
    fastfor(out.size(), [&](int i) {
        T sum = 0;
        for (int j = 0; j < in.size(); j++) {
            sum += in[j]->data[i];
        }
        out.data[i] = sum;
    });
}

template class ClearText<i64>;
template class ClearText<i32>;
template class ClearText<u64>;
template class ClearText<double>;
template class ClearText<float>;
