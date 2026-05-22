// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <sytorch/backend/baseline_cleartext.h>
#include <Eigen/Dense>

template <typename T>
void BaselineClearText<T>::matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
    modbw(c);
}

template <typename T>
void BaselineClearText<T>::matmul_triangular(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d1);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = (eA * eB).template triangularView<Eigen::Lower>();
    modbw(c);
}

template <typename T>
void BaselineClearText<T>::matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d1 == b.d1);
    assert(c.d1 == a.d2);
    assert(c.d2 == b.d2);
//    c.zero();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eA(a.data, a.d2, a.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
    modbw(c);
}

template <typename T>
void BaselineClearText<T>::matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
    assert(a.d2 == b.d2);
    assert(c.d1 == a.d1);
    assert(c.d2 == b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
    eC = eA * eB;
    modbw(c);
}

template <typename T>
void BaselineClearText<T>::conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst)
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
    modbw(output);
}

template <typename T>
void BaselineClearText<T>::conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
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
    modbw(output);
}

template <typename T>
void BaselineClearText<T>::convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
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
        modbw(output);
    }

template <typename T>
void BaselineClearText<T>::relu(const Tensor<T> &in, const Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode) {
    assert(in.is_same_shape(out));
    assert(in.is_same_shape(drelu));
    fastfor(in.size(), [&] (u64 i) {
        drelu.data[i] = (T)(in.data[i] > 0);
        assert(drelu.data[i] == 1 || drelu.data[i] == 0);
        out.data[i] = (drelu.data[i] == 1) ? in.data[i] : 0;
    });
}

template <int bw>
void _modbw(u64& x) {
    if /*constexpr*/ (bw == 64) {
        return;
    }
    else {
        x = x % (1ULL << bw);
    }
}

template <typename T>
void BaselineClearText<T>::truncate(T *in, T *out, u64 shift, u64 size, u8 mode) {
    fastfor(size, [&] (u64 i) {
        if constexpr (localTruncationEmulation) {
            u64 a = prngStr.get<u64>();
            _modbw<bw>(a);
            u64 b = a - ((u64)in[i]);
            _modbw<bw>(b);
            a = a >> shift;
            b = b >> shift;
            out[i] = a - b;
            modbw(out[i]);
            return;
        }
        out[i] = in[i] >> shift;
        if constexpr (probablistic) {
            u64 x0 = ((u64)in[i]) % (1ULL << shift);
            u64 r = rand() % (1ULL << shift);
            out[i] += (x0 < r ? 0 : 1); 
        }
    });
}

template <typename T>
void BaselineClearText<T>::truncate(T &in, u64 shift) {
    if constexpr (localTruncationEmulation) {
        u64 a = prngStr.get<u64>();
        _modbw<bw>(a);
        u64 b = a - ((u64)in);
        _modbw<bw>(b);
        a = a >> shift;
        b = b >> shift;
        in = a - b;
        modbw(in);
        return;
    }
    u64 x0 = ((u64)in) % (1ULL << shift);
    in = in >> shift;
    if constexpr (probablistic) {
        u64 r = rand() % (1ULL << shift);
        in += (x0 < r ? 0 : 1); 
    }
    modbw(in);
}

template <typename T>
void BaselineClearText<T>::div(Tensor<T> &in, T divisor, u64 scale) {
    // fastfor(in.size(), [&] (u64 i) {
    //     in.data[i] = in.data[i] / divisor;
    // });

    T divfp = (1LL << scale) / divisor;
    u64 sz = in.size();
    fastfor(in.size(), [&] (u64 i) {
        in.data[i] *= divfp;
    });
    modbw(in);
    Backend<T>::truncate(in, scale, 3);
}

template <typename T>
void BaselineClearText<T>::div(T &in, T divisor, u64 scale) {
    // in = in / divisor;

    T divfp = (1LL << scale) / divisor;
    in *= divfp;
    modbw(in);
    BaselineClearText<T>::truncate(in, scale);
}

template <typename T>
void BaselineClearText<T>::sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
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
void BaselineClearText<T>::avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) {
    sumPool2D(ks, padding, stride, in, out);
    auto out_nd = out.as_nd();
    div(out_nd, (T)(ks*ks), scale);
}

template <typename T>
u64 BaselineClearText<T>::log2(u64 x) {
    u64 y = 0;
    while (x >>= 1) y++;
    return y;
}

template <typename T>
void BaselineClearText<T>::maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode) {
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
void BaselineClearText<T>::batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
{
    assert(A.d1 == B.d1);
    assert(A.d1 == x.shape.back());
    assert(x.is_same_shape(y));
    u64 channels = x.shape.back();

    fastfor(x.size(), [&](u64 i) {
        y.data[i] = x.data[i] * A(i % channels) + B(i % channels);
    });
}

template <typename T>
void BaselineClearText<T>::add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
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
    modbw(out);
}

double gelu(double x)
{
    return 0.5 * x * (1 + erf(x / sqrt(2.0)));
}

template <typename T>
void BaselineClearText<T>::gelu(const Tensor<T> &in, const Tensor<T> &out, u64 scale, u64 mode)
{
    fastfor(in.size(), [&](u64 i) {
        out.data[i] = ::gelu(double(in.data[i]) / (1LL << scale)) * (1LL << scale);
    });
}

template <typename T>
void BaselineClearText<T>::softmax(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
{
    always_assert(_in.shape.size() == 2);
    always_assert(_out.shape.size() == 2);
    always_assert(_in.shape[0] == _out.shape[0]);
    always_assert(_in.shape[1] == _out.shape[1]);

    auto in = _in.as_2d();
    auto out = _out.as_2d();

    auto batchSize = in.d1;
    auto numClasses = in.d2;
    T exps[numClasses];
    for(int b = 0; b < batchSize; ++b) {
        T max = in(b, 0);
        for(u64 j = 1; j < numClasses; ++j) {
            if(in(b, j) > max) {
                max = in(b, j);
            }
        }

        T den = 0;
        for(u64 j = 0; j < numClasses; ++j) {
            T x = max - in(b, j);
            exps[j] = T(std::exp(-x / double(1LL<<scale)) * (1LL<<scale));
            modbw(exps[j]);
            den += exps[j];
        }

        for(u64 j = 0; j < numClasses; ++j) {
            out(b, j) = exps[j] * (1LL<<scale) / den;
        }
    }
}

template <typename T>
void BaselineClearText<T>::softmax_triangular(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
{
    always_assert(_in.shape.size() == 2);
    always_assert(_out.shape.size() == 2);
    always_assert(_in.shape[0] == _out.shape[0]);
    always_assert(_in.shape[1] == _out.shape[1]);
    always_assert(_in.shape[0] == _in.shape[1]); // should be a square matrix

    auto in = _in.as_2d();
    auto out = _out.as_2d();

    auto batchSize = in.d1;
    auto numClasses = in.d2;
    T exps[numClasses];

    out.zero();
    out(0, 0) = T(1LL << (scale));
    
    for(int b = 1; b < batchSize; ++b) {
        T max = in(b, 0);
        for(u64 j = 1; j < b + 1; ++j) {
            if(in(b, j) > max) {
                max = in(b, j);
            }
        }

        T den = 0;
        for(u64 j = 0; j < b + 1; ++j) {
            T x = max - in(b, j);
            exps[j] = T(std::exp(-x / double(1LL<<scale)) * (1LL<<scale));
            modbw(exps[j]);
            den += exps[j];
        }

        for(u64 j = 0; j < b + 1; ++j) {
            out(b, j) = exps[j] * (1LL<<scale) / den;
        }
    }
}

template <typename T>
void BaselineClearText<T>::layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
{
    always_assert(A.d1 == B.d1);
    always_assert(A.d1 == x.shape.back());
    always_assert(x.is_same_shape(y));
    
    u64 channels = x.shape.back();

    fastfor(x.size() / channels, [&](u64 i) {
        T mean = 0;
        T var = 0;
        for (u64 j = 0; j < channels; j++) {
            mean += x.data[i * channels + j];
        }
        modbw(mean);
        // mean = mean / T(channels);
        div(mean, channels, scale);
        for (u64 j = 0; j < channels; j++) {
            var += (x.data[i * channels + j] - mean) * (x.data[i * channels + j] - mean);
        }
        modbw(var);
        truncate(var, scale);
        div(var, channels, scale);
        double d = double(var) / ((1LL << (scale)));
        // var = var / T(channels);
        // div(var, channels, scale);
        // truncate(var, scale);
        var = T((1LL << scale) / std::sqrt(d));
        for (u64 j = 0; j < channels; j++) {
            y.data[i * channels + j] = (x.data[i * channels + j] - mean) * var;
        }
    });
    modbw(y);

    Backend<T>::truncate(y, scale);

    fastfor(x.size(), [&](u64 i) {
        y.data[i] = y.data[i] * A(i % channels) + B(i % channels);
    });

    Backend<T>::truncate(y, scale);
}

template <typename T>
void BaselineClearText<T>::addbias(Tensor<T> &x, const Tensor1D<T> &bias)
{
    always_assert(x.shape.back() == bias.d1);
    fastfor(x.size(), [&](u64 i) {
        x.data[i] += bias(i % bias.d1);
    });
    modbw(x);
}

template <typename T>
void BaselineClearText<T>::scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y)
{
    always_assert(x.is_same_shape(y));
    fastfor(x.size(), [&](u64 i) {
        y.data[i] = x.data[i] * scalar;
    });
    modbw(y);
}

template <typename T>
void BaselineClearText<T>::attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y)
{
    always_assert(x.is_same_shape(y));
    always_assert(x.shape.size() == 2);
    always_assert(x.shape[0] == x.shape[1]);

    u64 n_seq = x.shape[0];
    auto y_2d = y.as_2d();
    auto x_2d = x.as_2d();

    for (u64 j = 0; j < n_seq; ++j) {
        for (u64 k = 0; k < j + 1; ++k) {
            y_2d(j, k) = x_2d(j, k);
        }
        for (u64 k = j + 1; k < n_seq; ++k) {
            y_2d(j, k) = x_2d(j, k) - scalar;
        }
    }

    modbw(y);
}

template <typename T>
void BaselineClearText<T>::local_attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y)
{
    always_assert(x.is_same_shape(y));
    always_assert(x.shape.size() == 2);
    always_assert(x.shape[0] == x.shape[1]);

    u64 n_seq = x.shape[0];
    auto y_2d = y.as_2d();
    auto x_2d = x.as_2d();
    u64 window_size = 256;

    for (u64 j = 1; j <= n_seq - 1 ; ++j) {
        for (u64 k = 0; k < n_seq - j; ++k) {
            y_2d(k, k+j) = x_2d(k, k+j) - scalar;
        }
    }

    for (u64 j = 0; j <= window_size -1 ; ++j) {
        for (u64 k = j; k < n_seq; ++k) {
            y_2d(k, k-j) = x_2d(k, k-j);
        }
    }

    for (u64 j = window_size; j <= n_seq -1 ; ++j) {
        for (u64 k = j; k < n_seq; ++k) {
            y_2d(k, k-j) = x_2d(k, k-j) - scalar;
        }
    }

    modbw(y);
}

template <typename T>
void BaselineClearText<T>::tanh(const Tensor<T> &in, const Tensor<T> &out, u64 scale)
{

    fastfor(in.size(), [&](u64 i) {
        double x = double(in.data[i]) / double(1LL << scale);
        out.data[i] = T(std::tanh(x) * (1LL << scale));
    });
}

template <typename T>
void BaselineClearText<T>::scalardiv(Tensor<T> &x, double scalar, Tensor<T> &y, u64 scale, u64 mode)
{
    T d = T(double(1LL << (scale)) / scalar);
    if ((d & (d - 1)) == 0) 
    {
        Backend<T>::truncate(x, y, scale - log2(d), 0);
    }
    else
    {
        this->scalarmul(x, d, y);
        Backend<T>::truncate(y, y, scale, 0);
    }
}

template class BaselineClearText<i64>;
template class BaselineClearText<i32>;
