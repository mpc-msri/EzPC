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
