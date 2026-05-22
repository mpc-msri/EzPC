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


#include <sytorch/backend/cleartext.h>
#include <Eigen/Dense>

template <typename T>
void ClearText<T>::matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
{
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
void ClearText<T>::matmul_triangular(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
{
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
void ClearText<T>::matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
{
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
void ClearText<T>::matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
{
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
void ClearText<T>::conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst)
{
    assert(input.d4 == ci);
    assert(filter.d1 == co);
    assert(filter.d2 == fh * fw * ci);
    u64 newH = (((input.d2 + 2 * padding - fh) / stride) + 1);
    u64 newW = (((input.d3 + 2 * padding - fw) / stride) + 1);
    assert(output.d1 == input.d1);
    assert(output.d2 == newH);
    assert(output.d3 == newW);
    assert(output.d4 == co);

    Tensor2D<T> reshapedInput = reshapeInputTransposed<T>(input, padding, stride, fh, fw);
    Tensor2D<T> tempOutput(filter.d1, reshapedInput.d1);
    matmulTransposeB(filter, reshapedInput, tempOutput);
    reshapeOutput<T>(tempOutput, input.d1, (((input.d2 + 2 * padding - fh) / stride) + 1), (((input.d3 + 2 * padding - fw) / stride) + 1), co, output);
    modbw(output);
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
    u64 newD = (((input.d2 + 2 * pd - fd - (fd - 1) * (dd - 1)) / sd) + 1);
    u64 newH = (((input.d3 + 2 * ph - fh - (fh - 1) * (dh - 1)) / sh) + 1);
    u64 newW = (((input.d4 + 2 * pw - fw - (fw - 1) * (dw - 1)) / sw) + 1);
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
void ClearText<T>::convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
{
    assert(input.d5 == ci);
    assert(filter.d1 == co);
    assert(filter.d2 == fd * fh * fw * ci);
    u64 newD = (((input.d2 - 1) * sd + fd - 2 * pd));
    u64 newH = (((input.d3 - 1) * sh + fh - 2 * ph));
    u64 newW = (((input.d4 - 1) * sw + fw - 2 * pw));
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
void ClearText<T>::relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode)
{
    assert(in.is_same_shape(out));
    assert(in.is_same_shape(drelu));
    fastfor(in.size(), [&](u64 i)
            {
                drelu.data[i] = (T)(in.data[i] > 0);
                assert(drelu.data[i] == 1 || drelu.data[i] == 0);
                out.data[i] = (drelu.data[i] == 1) ? in.data[i] : 0; });
}

template <typename T>
void ClearText<T>::truncate(T *in, T *out, u64 shift, u64 size, u8 mode)
{
    fastfor(size, [&](u64 i)
            {
                if /*constexpr*/ (localTruncationEmulation)
                {
                    u64 a = prngStr.get<u64>();
                    u64 b = ((u64)in[i]) - a;
                    a = a >> shift;
                    b = -((-b) >> shift);
                    out[i] = a + b;
                    return;
                }
                out[i] = in[i] >> shift;
                if /*constexpr*/ (probablistic)
                {
                    u64 x0 = ((u64)in[i]) % (1ULL << shift);
                    u64 r = rand() % (1ULL << shift);
                    out[i] += (x0 < r ? 0 : 1);
                } });
}

template <typename T>
void ClearText<T>::truncate(T &in, u64 shift)
{
    if /*constexpr*/ (localTruncationEmulation)
    {
        u64 a = prngStr.get<T>();
        u64 b = ((u64)in) - a;
        a = a >> shift;
        b = -((-b) >> shift);
        in = a + b;
        return;
    }
    u64 x0 = ((u64)in) % (1ULL << shift);
    in = in >> shift;
    if /*constexpr*/ (probablistic)
    {
        u64 r = rand() % (1ULL << shift);
        in += (x0 < r ? 0 : 1);
    }
    modbw(in);
}

template <typename T>
void ClearText<T>::div(Tensor<T> &in, T divisor, u64 scale)
{
    // fastfor(in.size(), [&] (u64 i) {
    //     in.data[i] = in.data[i] / divisor;
    // });

    T divfp = (1LL << scale) / divisor;
    u64 sz = in.size();
    fastfor(in.size(), [&](u64 i)
            { in.data[i] *= divfp; });
    modbw(in);
    Backend<T>::truncate(in, scale, 3);
}

template <typename T>
void ClearText<T>::div(T &in, T divisor, u64 scale)
{
    // in = in / divisor;

    T divfp = (1LL << scale) / divisor;
    in *= divfp;
    modbw(in);
    ClearText<T>::truncate(in, scale);
}

template <typename T>
void ClearText<T>::sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out)
{
    assert(in.d1 == out.d1);
    assert(in.d4 == out.d4);
    u64 newH = (in.d2 + 2 * padding - ks) / stride + 1;
    u64 newW = (in.d3 + 2 * padding - ks) / stride + 1;
    assert(out.d2 == newH);
    assert(out.d3 == newW);
    fastfor(in.d1, [&](int i)
            {
                for (int j = 0; j < newH; j++)
                {
                    for (int k = 0; k < newW; k++)
                    {
                        for (int l = 0; l < in.d4; l++)
                        {
                            T sum = 0;
                            for (int m = 0; m < ks; m++)
                            {
                                for (int n = 0; n < ks; n++)
                                {
                                    sum += in(i, j * stride + m, k * stride + n, l);
                                }
                            }
                            out(i, j, k, l) = sum;
                        }
                    }
                } });
}

template <typename T>
void divPartial(const Tensor4D<T> &in, T divisor, u64 scale)
{
    T divfp = (1LL << scale) / divisor;
    u64 sz = in.d1 * in.d2 * in.d3 * in.d4;
#pragma omp parallel for
    for (u64 i = 0; i < sz; i++)
    {
        in.data[i] *= divfp;
    }
}

template <typename T>
void ClearText<T>::avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale)
{
    sumPool2D(ks, padding, stride, in, out);
    divPartial(out, (T)(ks * ks), scale);
}


template <typename T>
u64 ClearText<T>::log2(u64 x)
{
    u64 y = 0;
    while (x >>= 1)
        y++;
    return y;
}

template <typename T>
void ClearText<T>::maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode)
{
    assert(in.d1 == out.d1);
    assert(in.d4 == out.d4);
    u64 newH = (in.d2 + 2 * padding - ks) / stride + 1;
    u64 newW = (in.d3 + 2 * padding - ks) / stride + 1;
    assert(out.d2 == newH);
    assert(out.d3 == newW);
    fastfor(in.d1, [&](int i)
            {
                for (int j = 0; j < newH; j++)
                {
                    for (int k = 0; k < newW; k++)
                    {
                        for (int l = 0; l < in.d4; l++)
                        {
                            T max = std::numeric_limits<T>::lowest();
                            u64 maxIdxI = 0;
                            u64 maxIdxJ = 0;
                            for (int m = 0; m < ks; m++)
                            {
                                for (int n = 0; n < ks; n++)
                                {
                                    auto h2 = j * stride + m - padding;
                                    auto w2 = k * stride + n - padding;
                                    T val = 0;
                                    if (h2 < in.d2 && w2 < in.d3 && h2 >= 0 && w2 >= 0)
                                        val = in(i, h2, w2, l);
                                    if (val > max)
                                    {
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
                } });
}


template <typename T>
void ClearText<T>::batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
{
    assert(A.d1 == B.d1);
    assert(A.d1 == x.shape.back());
    assert(x.is_same_shape(y));
    u64 channels = x.shape.back();

    fastfor(x.size(), [&](u64 i)
            { y.data[i] = x.data[i] * A(i % channels) + B(i % channels); });
}

template <typename T>
void ClearText<T>::add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
{
    always_assert(in.size() > 0);
    always_assert(out.size() == in[0]->size());
    for (int i = 0; i < in.size(); i++)
    {
        always_assert(out.size() == in[i]->size());
    }
    fastfor(out.size(), [&](int i)
            {
                T sum = 0;
                for (int j = 0; j < in.size(); j++)
                {
                    sum += in[j]->data[i];
                }
                out.data[i] = sum; });
    modbw(out);
}

template <typename T>
T tanh(T x, u64 scale)
{
    double d = ((double)x) / (1LL << scale);
    return (T)(tanh(d) * (1LL << scale));
}

i64 clip(i64 x, u64 l)
{
    assert(x >= 0);
    if (x >= (1LL << l))
    {
        x = (1LL << l) - 1;
    }
    return x;
}

template <typename T>
T tanh_lut(T x, u64 scale)
{
    u64 idx = x < 0 ? (-x) : x;
    idx = clip(idx, 14);
    float xf = double(idx) / double(1LL << scale);
    xf = std::tanh(xf);
    i64 res = i64(xf * (1LL << scale));
    ;
    if (x < 0)
    {
        res = -res;
    }
    return res;
}

double gelu_sub_relu(double x)
{
    // double g = 0.5 * x * (1 + std::tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
    double g = 0.5 * x * (1 + erf(x / sqrt(2.0)));
    return g - std::max(0.0, x);
}

template <typename T>
T gelu_sub_relu(T x, u64 scale_in, u64 scale_out)
{
    return (T)(gelu_sub_relu((double)x / (1LL << scale_in)) * (1LL << scale_out));
}

// double relu_sub_gelu_double(double x)
// {
//     // double g = 0.5 * x * (1 + std::tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
//     double g = 0.5 * x * (1 + erf(x / sqrt(2.0)));
//     return std::max(0.0, x) - g;
// }

// template <typename T>
// T ClearText<T>::relu_sub_gelu(T x, u64 scale_in, u64 scale_out)
// {
//     return (T) (relu_sub_gelu_double((double) x / (1LL << scale_in)) * (1LL << scale_out));
// }

template <typename T>
void ClearText<T>::gelu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode)
{
    // gen_tanh_table(scale, scale);
    // always_assert(in.size() == out.size());
    // T t1 = (T) (sqrt(2.0 / M_PI) * (1LL << scale));
    // T t2 = (T) (0.044715 * (1LL << scale));
    // fastfor(in.size(), [&](u64 i) {
    //     T ini = in.data[i];
    //     T t = ini * ini;
    //     modbw(t);
    //     truncate(t, scale);
    //     t = t * ini;
    //     modbw(t);
    //     truncate(t, scale);
    //     t = t * t2;
    //     modbw(t);
    //     truncate(t, scale);
    //     t = t + ini;
    //     t = t * t1;
    //     modbw(t);
    //     truncate(t, scale);
    //     t = tanh_lut(t, scale);
    //     t = t + (1LL << scale);
    //     t = t * ini;
    //     modbw(t);
    //     truncate(t, scale+1);
    //     out.data[i] = t;
    // });
    // printf("In here!\n");
    // always_assert(scale == 12);
    fastfor(in.size(), [&](u64 i)
            {
                // T r = in.data[i] > 0 ? in.data[i] : 0;
                // T t = 2 * r - in.data[i];
                // t = clip(t, scale + 2);
                // // t = t / (1LL << (scale - 6)); // bw = 8, scale = 6
                // ClearText<T>::truncate(t, scale - 6);
                // t = gelu_sub_relu(t, 6, scale);
                // out.data[i] = t + r;

                auto inpSmall = in.data[i];
                ClearText<T>::truncate(inpSmall, scale - 6);
                T r = in.data[i] > 0 ? in.data[i] : 0;
                T rSmall = in.data[i] > 0 ? inpSmall : 0;
                T t = 2 * rSmall - inpSmall;
                t = clip(t, scale + 2);
                t = gelu_sub_relu(t, 6, scale);
                out.data[i] = r + t; });
}

double silu_sub_relu(double x)
{
    double g = x / (1 + exp(-x));
    return g - std::max(0.0, x);
}

template <typename T>
T silu_sub_relu(T x, u64 scale_in, u64 scale_out)
{
    return (T)(silu_sub_relu(((double)x) / (1LL << scale_in)) * (1LL << scale_out));
}

template <typename T>
void ClearText<T>::silu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode)
{
    // always_assert(scale == 12);
    fastfor(in.size(), [&](u64 i)
            {
                auto inpSmall = in.data[i];
                ClearText<T>::truncate(inpSmall, scale - 6);
                T r = in.data[i] > 0 ? in.data[i] : 0;
                T rSmall = in.data[i] > 0 ? inpSmall : 0;
                T t = 2 * rSmall - inpSmall;
                t = clip(t, scale + 4);
                t = silu_sub_relu(t, 6, scale);
                out.data[i] = r + t;
                // ClearText<T>::truncate(t, scale - 6);
                // T r = in.data[i] > 0 ? in.data[i] : 0;
                // T t = 2 * r - in.data[i];
                // t = clip(t, scale + 4);
                // t = silu_sub_relu(t, 6, scale);
                // out.data[i] = t + r;
            });
}

std::vector<i64> exp_tab(0);

void gen_exp_table(u64 l1, u64 s1, u64 l2, u64 s2)
{
    // table for exp(-x), where x is negative
    if (exp_tab.size() == (1LL << l1))
    {
        return;
    }

    exp_tab.resize(1LL << l1);
    for (u64 i = 0; i < (1LL << l1); ++i)
    {
        double x = -double(i) / double(1LL << s1);
        double ex = std::exp(x);
        exp_tab[i] = i64(ex * (1LL << s2));
    }
}

std::vector<i64> inv_tab(0);

void gen_inv_table(u64 l1, u64 s1, u64 l2, u64 s2)
{
    // table for inv(x), where x is negative
    if (inv_tab.size() == (1LL << l1))
    {
        return;
    }

    inv_tab.resize(1LL << l1);
    for (u64 i = 1; i < (1LL << l1); ++i)
    {
        double x = double(i) / double(1LL << s1);
        double iv = 1.0 / x;
        inv_tab[i] = i64(iv * (1LL << s2));
    }
}

template <typename T>
void ClearText<T>::softmax_table(Tensor2D<T> &in, Tensor2D<T> &out, u64 scale)
{
    always_assert(scale == 12);
    auto batchSize = in.d1;
    auto numClasses = in.d2;
    // hardcoded params for now
    gen_exp_table(16, 12, 22, 12);
    gen_inv_table(16, 6, 13, 12);
    T exps[numClasses];

    for (int b = 0; b < batchSize; ++b)
    {
        T max = in(b, 0);
        for (u64 j = 1; j < numClasses; ++j)
        {
            if (in(b, j) > max)
            {
                max = in(b, j);
            }
        }
        T den = 0;
        for (u64 j = 0; j < numClasses; ++j)
        {
            T x = max - in(b, j);
            x = clip(x, 16);
            exps[j] = exp_tab[x];
            den += exps[j];
        }

        den = den / (1LL << 6);
        T iden = inv_tab[den];
        for (u64 j = 0; j < numClasses; ++j)
        {
            out(b, j) = exps[j] * iden;
        }
    }
    Backend<T>::truncate(out, 12);
}

template <typename T>
void ClearText<T>::softmax_sirnn_2part_lut(Tensor2D<T> &in, Tensor2D<T> &out, u64 scale)
{
    always_assert(scale >= 12);
    auto batchSize = in.d1;
    auto numClasses = in.d2;
    // hardcoded params for now
    T exps[numClasses];

    for (int b = 0; b < batchSize; ++b)
    {
        T max = in(b, 0);
        for (u64 j = 1; j < numClasses; ++j)
        {
            if (in(b, j) > max)
            {
                max = in(b, j);
            }
        }
        // printf("Max %d=%ld\n", b, max);
        for (u64 j = 0; j < numClasses; ++j)
        {
            T x = max - in(b, j);
            x = clip(x, scale + 4);
            x = x / (1LL << (scale - 12)); // x is now in bw=16, scale=12
            T x0 = x % (1LL << 8);
            T x1 = x / (1LL << 8);
            T e0 = T(std::exp(-x0 / double(1LL << 12)) * (1LL << scale));
            T e1 = T(std::exp(-x1 / double(1LL << 4)) * (1LL << scale));
            exps[j] = e0 * e1;
            modbw(exps[j]);
            truncate(exps[j], scale);
        }

        T den = 0;
        for (u64 j = 0; j < numClasses; ++j)
        {
            den += exps[j];
        }

        den = den / (1LL << (scale - 6)); // bw=16, scale=6 (as max 1024 classes supported)
        T iden = T((1.0 / double(den / double(1LL << 6))) * (1LL << scale));
        for (u64 j = 0; j < numClasses; ++j)
        {
            out(b, j) = exps[j] * iden;
        }
    }
    Backend<T>::truncate(out, scale);
}

template <typename T>
void ClearText<T>::softmax_sirnn_2part_lut_triangular(Tensor2D<T> &in, Tensor2D<T> &out, u64 scale)
{
    always_assert(scale >= 12);
    auto batchSize = in.d1;
    auto numClasses = in.d2;
    always_assert(batchSize == numClasses);
    T exps[numClasses];

    out.zero();
    out(0, 0) = T(1LL << (2 * scale));
    for (int b = 1; b < batchSize; ++b)
    {
        // in.as_nd().print();
        T max = in(b, 0);
        // printf("Max=%ld\n", max);
        for (u64 j = 1; j < b + 1; ++j)
        {
            if (in(b, j) > max)
            {
                max = in(b, j);
            }
        }
        // printf("Max %d=%ld\n", b, max);
        for (u64 j = 0; j < b + 1; ++j)
        {
            T x = max - in(b, j);
            // printf("b=%d, %ld\n", j, x);
            x = clip(x, scale + 4);
            x = x / (1LL << (scale - 12)); // x is now in bw=16, scale=12
            T x0 = x % (1LL << 8);
            T x1 = x / (1LL << 8);
            T e0 = T(std::exp(-x0 / double(1LL << 12)) * (1LL << scale));
            T e1 = T(std::exp(-x1 / double(1LL << 4)) * (1LL << scale));
            exps[j] = e0 * e1;
            modbw(exps[j]);
            truncate(exps[j], scale);
        }

        T den = 0;
        for (u64 j = 0; j < b + 1; ++j)
        {
            // printf("j=%d, exp=%ld\n", b, exps[j]);
            den += exps[j];
        }
        // printf("Den=%ld\n", den);
        den = den / (1LL << (scale - 6)); // bw=16, scale=6 (as max 1024 classes supported)
        T iden = T((1.0 / double(den / double(1LL << 6))) * (1LL << scale));
        // printf("iDen=%ld\n", iden);
        for (u64 j = 0; j < b + 1; ++j)
        {
            // printf("j=%d, exp=%ld\n", j, exps[j]);
            out(b, j) = exps[j] * iden;
        }
    }
    Backend<T>::truncate(out, scale);
}

template <typename T>
void ClearText<T>::softmax(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
{
    always_assert(_in.shape.size() == 2);
    always_assert(_out.shape.size() == 2);
    always_assert(_in.shape[0] == _out.shape[0]);
    always_assert(_in.shape[1] == _out.shape[1]);

    auto in = _in.as_2d();
    auto out = _out.as_2d();
    if (mode == 1)
        return softmax_table(in, out, scale);

    if (mode == 2)
        return softmax_polynomial(in, out, scale);

    if (mode == 0)
        return softmax_sirnn_2part_lut(in, out, scale);

    auto batchSize = in.d1;
    auto numClasses = in.d2;
    for (int b = 0; b < batchSize; ++b)
    {
        T max = in(b, 0);
        for (u64 j = 1; j < numClasses; ++j)
        {
            if (in(b, j) > max)
            {
                max = in(b, j);
            }
        }
        // std::cout << "max: " << max << std::endl;
        double den = 0.0;
        double exps[numClasses];
        for (u64 j = 0; j < numClasses; ++j)
        {
            double x = in(b, j) - max;
            // std::cout << "out[" << j << "]: " << T(-x) << std::endl;
            if (scale == 0)
            {
                exps[j] = std::exp(x);
            }
            else
            {
                exps[j] = std::exp(x / (1LL << scale));
            }
            // std::cout << "exps[" << j << "]: " << T(exps[j] * (1LL << scale)) << std::endl;
            den += exps[j];
        }

        for (u64 j = 0; j < numClasses; ++j)
        {
            if (scale == 0)
            {
                out(b, j) = exps[j] / den;
            }
            else
            {
                auto t = (exps[j] / den) * (1LL << scale);
                out(b, j) = (T)(t);
            }
        }
    }
}

template <typename T>
void ClearText<T>::softmax_triangular(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
{
    always_assert(_in.shape.size() == 2);
    always_assert(_out.shape.size() == 2);
    always_assert(_in.shape[0] == _out.shape[0]);
    always_assert(_in.shape[1] == _out.shape[1]);
    always_assert(_in.shape[0] == _in.shape[1]); // should be a square matrix

    auto in = _in.as_2d();
    auto out = _out.as_2d();
    // printf("Printing out the mode=%d\n", mode);
    if (mode == 0)
        return softmax_sirnn_2part_lut_triangular(in, out, scale);

    auto batchSize = in.d1;
    auto numClasses = in.d2;
    out(0, 0) = 1LL << scale;

    for (int b = 1; b < batchSize; ++b)
    {
        T max = in(b, 0);
        for (u64 j = 1; j < b + 1; ++j)
        {
            if (in(b, j) > max)
            {
                max = in(b, j);
            }
        }
        // std::cout << "max: " << max << std::endl;
        double den = 0.0;
        double exps[b + 1];
        for (u64 j = 0; j < b + 1; ++j)
        {
            double x = in(b, j) - max;
            exps[j] = std::exp(x / (1LL << scale));
            den += exps[j];
        }

        for (u64 j = 0; j < b + 1; ++j)
        {
            auto t = (exps[j] / den) * (1LL << scale);
            out(b, j) = (T)(t);
        }
    }
}

template <typename T>
T ClearText<T>::invsqrt_lut_2(T x, u64 scale, u64 additional_div, u64 n)
{
    i64 k = 0;
    u64 unsigned_X = (u64)x & ((1ULL << n) - 1);
    while (unsigned_X >= (1ULL << k) && k < bw)
    {
        k++;
    }
    k = k - 1;
    u64 m = (unsigned_X << (n - k - 1));
    m >>= (n - 8);
    // ClearText<T>::truncate(m, n - 8);
    // m = m - 128;
    double val = double(m) * std::pow(2.0, k - 7);
    return (T)(double(1LL << (2 * scale)) / sqrt(val / additional_div));
}

template <typename T>
void ClearText<T>::layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
{
    always_assert(A.d1 == B.d1);
    always_assert(A.d1 == x.shape.back());
    always_assert(x.is_same_shape(y));
    u64 channels = x.shape.back();
    fastfor(x.size() / channels, [&](u64 i)
            {
                T mean = 0;
                T var = 0;
                for (u64 j = 0; j < channels; j++)
                {
                    mean += x.data[i * channels + j];
                }
                modbw(mean);
                div(mean, channels, scale);
                for (u64 j = 0; j < channels; j++)
                {
                    var += (x.data[i * channels + j] - mean) * (x.data[i * channels + j] - mean);
                }
                modbw(var);
                var = invsqrt_lut_2(var, scale, channels, bw);
                for (u64 j = 0; j < channels; j++)
                {
                    y.data[i * channels + j] = (x.data[i * channels + j] - mean) * var;
                } });
    modbw(y);
    Backend<T>::truncate(y, scale);
    fastfor(x.size(), [&](u64 i)
            { y.data[i] = y.data[i] * A(i % channels) + B(i % channels); });

    Backend<T>::truncate(y, scale);
}

template <typename T>
void ClearText<T>::rmsnorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
{
    always_assert(A.d1 == B.d1);
    always_assert(A.d1 == x.shape.back());
    always_assert(x.is_same_shape(y));

    u64 channels = x.shape.back();

    fastfor(x.size() / channels, [&](u64 i)
            {
                T var = 0; // not exactly variance as no mean subtraction
                for (u64 j = 0; j < channels; j++)
                {
                    var += x.data[i * channels + j] * x.data[i * channels + j];
                }
                modbw(var);
                var = invsqrt_lut_2(var, scale, channels, bw);
                for (u64 j = 0; j < channels; j++)
                {
                    y.data[i * channels + j] = x.data[i * channels + j] * var;
                } });
    modbw(y);

    Backend<T>::truncate(y, scale);

    fastfor(x.size(), [&](u64 i)
            { y.data[i] = y.data[i] * A(i % channels) + B(i % channels); });
}

template <typename T>
void ClearText<T>::addbias(Tensor<T> &x, const Tensor1D<T> &bias)
{
    always_assert(x.shape.back() == bias.d1);
    fastfor(x.size(), [&](u64 i)
            { x.data[i] += bias(i % bias.d1); });
    modbw(x);
}

template <typename T>
void ClearText<T>::scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y)
{
    always_assert(x.is_same_shape(y));
    fastfor(x.size(), [&](u64 i)
            { y.data[i] = x.data[i] * scalar; });
    modbw(y);
}

template <typename T>
void ClearText<T>::polyeval(const Tensor<T> &x, Tensor<T> &y, const std::vector<double> &coefficients, u64 scale)
{
    always_assert(x.is_same_shape(y));
    T c0 = (T)(coefficients[0] * (1LL << scale));
    T c1 = (T)(coefficients[1] * (1LL << (2 * scale)));
    // std::cout << c0 << std::endl;
    // std::cout << c1 / (1LL << scale) << std::endl;
    Tensor<T> tmp(x.shape);

    fastfor(x.size(), [&](u64 i)
            { tmp.data[i] = c0 * x.data[i] + c1; });

    Backend<T>::truncate(tmp, scale);

    for (u64 e = 2; e < coefficients.size(); ++e)
    {
        T c = (T)(coefficients[e] * (1LL << (2 * scale)));
        // std::cout << c / (1LL << scale) << std::endl;

        fastfor(x.size(), [&](u64 i)
                { tmp.data[i] = tmp.data[i] * x.data[i] + c; });

        Backend<T>::truncate(tmp, scale);
    }

    fastfor(x.size(), [&](u64 i)
            { y.data[i] = tmp.data[i]; });
}

template <typename T>
void ClearText<T>::softmax_polynomial(Tensor2D<T> &in, Tensor2D<T> &out, u64 scale)
{
    always_assert(scale == 12);
    auto batchSize = in.d1;
    auto numClasses = in.d2;
    // hardcoded params for now
    gen_inv_table(16, 6, 13, 12);

    for (int b = 0; b < batchSize; ++b)
    {
        T max = in(b, 0);
        for (u64 j = 1; j < numClasses; ++j)
        {
            if (in(b, j) > max)
            {
                max = in(b, j);
            }
        }
        // std::cout << "max: " << max << std::endl;
        for (u64 j = 0; j < numClasses; ++j)
        {
            T x = max - in(b, j);
            x = clip(x, 14);
            out(b, j) = x;
            // std::cout << "out[" << j << "]: " << out(b, j) << std::endl;
        }
        // std::cout << std::endl;
    }

    u64 max_deg = 8;
    std::vector<double> coeffs(max_deg);
    coeffs[max_deg - 1] = 1.0;
    for (u64 i = 1; i < max_deg; i++)
    {
        coeffs[max_deg - 1 - i] = coeffs[max_deg - i] * (-1.0 / i);
    }
    // for (u64 i = 0; i < 8; i++) {
    //     std::cout << coeffs[i] << " ";
    // }
    // std::cout << std::endl;

    auto out_2d = out.as_nd();
    polyeval(out_2d, out_2d, coeffs, scale);

    for (int b = 0; b < batchSize; ++b)
    {
        T den = 0;
        for (u64 j = 0; j < numClasses; ++j)
        {
            // std::cout << "exps[" << j << "]: " << out(b, j) << std::endl;
            den += out(b, j);
        }
        // std::cout << std::endl;

        den = den / (1LL << 6);
        // std::cout << "den: " << den << std::endl;
        T iden = inv_tab[den];
        for (u64 j = 0; j < numClasses; ++j)
        {
            out(b, j) = out(b, j) * iden;
        }
    }
    Backend<T>::truncate(out, 12);
}

template <typename T>
void ClearText<T>::attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y)
{
    always_assert(x.is_same_shape(y));
    always_assert(x.shape.size() == 2);
    always_assert(x.shape[0] == x.shape[1]);

    u64 n_seq = x.shape[0];
    auto y_2d = y.as_2d();
    auto x_2d = x.as_2d();

    for (u64 j = 0; j < n_seq; ++j)
    {
        for (u64 k = 0; k < j + 1; ++k)
        {
            y_2d(j, k) = x_2d(j, k);
        }
        for (u64 k = j + 1; k < n_seq; ++k)
        {
            y_2d(j, k) = x_2d(j, k) - scalar;
        }
    }

    modbw(y);
}

template <typename T>
void ClearText<T>::local_attention_mask(Tensor<T> &x, T scalar, Tensor<T> &y)
{
    always_assert(x.is_same_shape(y));
    always_assert(x.shape.size() == 2);
    always_assert(x.shape[0] == x.shape[1]);

    u64 n_seq = x.shape[0];
    auto y_2d = y.as_2d();
    auto x_2d = x.as_2d();
    u64 window_size = 256;

    for (u64 j = 1; j <= n_seq - 1; ++j)
    {
        for (u64 k = 0; k < n_seq - j; ++k)
        {
            y_2d(k, k + j) = x_2d(k, k + j) - scalar;
        }
    }

    for (u64 j = 0; j <= window_size - 1; ++j)
    {
        for (u64 k = j; k < n_seq; ++k)
        {
            y_2d(k, k - j) = x_2d(k, k - j);
        }
    }

    for (u64 j = window_size; j <= n_seq - 1; ++j)
    {
        for (u64 k = j; k < n_seq; ++k)
        {
            y_2d(k, k - j) = x_2d(k, k - j) - scalar;
        }
    }

    modbw(y);
}

template <typename T>
void ClearText<T>::tanh(const Tensor<T> &in, const Tensor<T> &out, u64 scale)
{
    fastfor(in.size(), [&](u64 i)
            { out.data[i] = tanh_lut(in.data[i], scale); });
}

template <typename T>
void ClearText<T>::mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
{
    always_assert(a.is_same_shape(b));
    always_assert(a.is_same_shape(out));

    fastfor(a.size(), [&](u64 i)
            { out.data[i] = a.data[i] * b.data[i]; });
}

template <typename T>
void ClearText<T>::scalardiv(Tensor<T> &x, double scalar, Tensor<T> &y, u64 scale, u64 mode)
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


template class ClearText<i64>;
template class ClearText<i32>;
template class ClearText<u64>;
template class ClearText<u32>;
// template class ClearText<double>;
// template class ClearText<float>;
