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
#include <sytorch/backend/baseline_cleartext.h>

template <typename T>
class SecureMLClearText : public BaselineClearText<T> {
public:
    void softmax(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
    {
        always_assert(_in.shape.size() == 2);
        always_assert(_out.shape.size() == 2);
        always_assert(_in.shape[0] == _out.shape[0]);
        always_assert(_in.shape[1] == _out.shape[1]);
        T twofix = 2 * (1LL << scale);

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
                T x = in(b, j);
                exps[j] = (x > 0 ? x : 0);
                den += exps[j];
            }

            if (den == 0)
                for(u64 j = 0; j < numClasses; ++j) {
                    out(b, j) = (1LL<<scale) / numClasses;
                }
            else
                for(u64 j = 0; j < numClasses; ++j) {
                    out(b, j) = exps[j] * (1LL<<scale) / den;
                }
        }
    }

    void softmax_triangular(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
    {
        always_assert(_in.shape.size() == 2);
        always_assert(_out.shape.size() == 2);
        always_assert(_in.shape[0] == _out.shape[0]);
        always_assert(_in.shape[1] == _out.shape[1]);
        always_assert(_in.shape[0] == _in.shape[1]); // should be a square matrix
        T twofix = 2 * (1LL << scale);

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
                T x = in(b, j);
                exps[j] = (x > 0 ? x : 0);
                den += exps[j];
            }

            if (den == 0)
                for(u64 j = 0; j < b + 1; ++j) {
                    out(b, j) = (1LL<<scale) / (b + 1);
                }
            else
                for(u64 j = 0; j < b + 1; ++j) {
                    out(b, j) = exps[j] * (1LL<<scale) / den;
                }
        }
    }

};
