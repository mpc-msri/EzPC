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

#pragma once


#include <sytorch/tensor.h>
#include <llama/stats.h>
#include <llama/api.h>

template <typename T, u64 scale>
void softmax(const Tensor4D<T> &in, Tensor4D<T> &out)
{
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == 1);
    assert(in.d4 == 1);
    assert(out.d3 == 1);
    assert(out.d4 == 1);
    assert(std::is_integral<T>::value || (scale == 0));

    auto batchSize = in.d1;
    auto numClasses = in.d2;
    for(int b = 0; b < batchSize; ++b) {
        T max = in(b, 0, 0, 0);
        for(u64 j = 1; j < numClasses; ++j) {
            if(in(b, j, 0, 0) > max) {
                max = in(b, j, 0, 0);
            }
        }
        double den = 0.0;
        double exps[numClasses];
        for(u64 j = 0; j < numClasses; ++j) {
            double x = in(b, j, 0, 0) - max;
            if constexpr (scale == 0) {
                exps[j] = std::exp(x);
            } else {
                exps[j] = std::exp(x / (1ULL << scale));
            }
            den += exps[j];
        }
        den = den * batchSize;

        for(u64 j = 0; j < numClasses; ++j) {
            if constexpr (scale == 0) {
                out(b, j, 0, 0) = exps[j] / den;
            } else {
                auto t = (exps[j] / den) * (1ULL << scale);
                t += rand_float();
                out(b, j, 0, 0) = (T)(t);
            }
        }
    }
}

inline void pirhana_softmax(const Tensor4D<u64> &in, Tensor4D<u64> &out, u64 scale)
{
    PiranhaSoftmax(in.d1, in.d2, in.data, in.data, out.data, out.data, scale);
}

inline i64 pirhana_inverse(i64 x, u64 scale)
{
    i64 alpha = 0;
    while((1LL << alpha) < x) {
        alpha++;
    }
    // inv(x/2^scale) * 2^scale = [inv(x/2^alpha) * 2^alpha] * 2^(scale - alpha) * 2^(scale - alpha)
    i64 a_alpha = 2.63 * (1LL << alpha);
    i64 b_alpha = -5.857 * (1LL << alpha);
    i64 c_alpha = 4.245 * (1LL << alpha);

    i64 res = a_alpha * x + b_alpha * (1LL << alpha);
    res >>= alpha;
    res = res * x + c_alpha * (1LL << alpha);
    res >>= alpha;
    if (scale > alpha) {
        res = res * (1LL<<(scale-alpha));
        res = res * (1LL<<(scale-alpha));
    }
    else {
        res = res / (1LL<<(alpha-scale));
        res = res / (1LL<<(alpha-scale));
    }
    return res;
}

inline void pirhana_softmax_ct(const Tensor4D<i64> &in, Tensor4D<i64> &out, u64 scale)
{
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == 1);
    assert(in.d4 == 1);
    assert(out.d3 == 1);
    assert(out.d4 == 1);

    auto batchSize = in.d1;
    auto numClasses = in.d2;
    for(int b = 0; b < batchSize; ++b) {
        i64 max = in(b, 0, 0, 0);
        for(u64 j = 1; j < numClasses; ++j) {
            if(in(b, j, 0, 0) - max > 0) {
                max = in(b, j, 0, 0);
            }
        }
        i64 den = 0;
        i64 exps[numClasses];
        for(u64 j = 0; j < numClasses; ++j) {
            i64 x = in(b, j, 0, 0) - max;
            exps[j] = (x > (-2*(1LL << scale))) ? ((x + 2*(1LL << scale)) / 2) : 0;
            den += exps[j];
        }
        // den = den * batchSize;
        den = pirhana_inverse(den, scale);

        for(u64 j = 0; j < numClasses; ++j) {
            auto t = exps[j] * den;
            t >>= scale;
            t /= batchSize;
            out(b, j, 0, 0) = (i64)(t);
        }
    }
}

inline void reconstrunct_and_print_float(Tensor4D<u64> &in, int llamaParty) {
    reconstruct(in.d1 * in.d2 * 4, in.data, 64);
    for(int i = 0; i < in.d1; ++i) {
        for(int j = 0; j < in.d2; ++j) {
            int m = in(i, j, 0, 0) % (1LL << 24);
            int e = in(i, j, 1, 0) % 256;
            int z = in(i, j, 2, 0) % 2;
            int s = in(i, j, 3, 0) % 2;
            std::cout << m << " " << e << " " << s << " " << z << std::endl;
            double x = m;
            x = x / (1LL << 23);
            if (s == 1) {
                x = -x;
            }
            e = e - 127;
            if (e < 0) {
                x = x / (1LL << (-e));
            } else {
                x = x * (1LL << e);
            }
            if (z == 1) {
                x = 0;
            }
            std::cout << x << std::endl;
        }
    }
    if (llamaParty == 3) {
        in.fill(0);
    }
}

void secfloat_init(int secfloatParty, std::string secfloatAddr);
void softmax_secfloat(Tensor4D<u64> &in, Tensor4D<u64> &out, u64 scale, int llamaParty);