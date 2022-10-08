#include "tensor.h"

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
                out(b, j, 0, 0) = (T)((exps[j] / den) * (1ULL << scale));
            }
        }
    }
}
