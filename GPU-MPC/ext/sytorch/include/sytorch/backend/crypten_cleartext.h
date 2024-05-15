#pragma once
#include <sytorch/backend/baseline_cleartext.h>

template <typename T>
class CryptenClearText : public BaselineClearText<T> {
public:
    T crypten_inverse(T x, u64 scale)
    {
        T result = (1LL << scale) - 2 * x;
        BaselineClearText<T>::modbw(result);
        result = 3 * crypten_exp(result, scale) + T(0.003 * (1LL << scale));
        BaselineClearText<T>::modbw(result);
        u64 iters = 10;
        for(u64 i = 0; i < iters; ++i) {
            T tmp = x * result;
            BaselineClearText<T>::modbw(tmp);
            BaselineClearText<T>::truncate(tmp, scale);
            result = result * (2 * (1LL << scale) - tmp);
            BaselineClearText<T>::modbw(result);
            BaselineClearText<T>::truncate(result, scale);
        }
        return result;
    }

    T crypten_exp(T x, u64 scale)
    {
        u64 iters = 8;
        T y = x;
        BaselineClearText<T>::truncate(y, iters);
        y = y + (1LL << scale);
        BaselineClearText<T>::modbw(y);
        for(u64 i = 0; i < iters; ++i) {
            y = y * y;
            BaselineClearText<T>::modbw(y);
            BaselineClearText<T>::truncate(y, scale);
        }
        return y;
    }
    
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
                T x = in(b, j) - max;
                exps[j] = crypten_exp(x, scale);
                den += exps[j];
            }

            BaselineClearText<T>::modbw(den);
            T inv_den = crypten_inverse(den, scale);
            for(u64 j = 0; j < numClasses; ++j) {
                out(b, j) = exps[j] * inv_den;
            }
        }
        Backend<T>::truncate(out, scale);
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
        out(0, 0) = T(1LL << (2*scale));
        
        for(int b = 1; b < batchSize; ++b) {
            T max = in(b, 0);
            for(u64 j = 1; j < b + 1; ++j) {
                if(in(b, j) > max) {
                    max = in(b, j);
                }
            }

            T den = 0;
            for(u64 j = 0; j < b + 1; ++j) {
                T x = in(b, j) - max;
                exps[j] = crypten_exp(x, scale);
                den += exps[j];
            }

            BaselineClearText<T>::modbw(den);
            T inv_den = crypten_inverse(den, scale);
            for(u64 j = 0; j < b + 1; ++j) {
                out(b, j) = exps[j] * inv_den;
            }
        }
        Backend<T>::truncate(out, scale);
    }

};
