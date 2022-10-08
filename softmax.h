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

    auto batchSize = in.d1;
    for(int b = 0; b < batchSize; ++b) {
        T max = in(b, 0, 0, 0);
        for(u64 j = 1; j < 10; ++j) {
            if(in(b, j, 0, 0) > max) {
                max = in(b, j, 0, 0);
            }
        }
        double den = 0.0;
        for(u64 j = 0; j < 10; ++j) {
            out(b, j, 0, 0) = in(b, j, 0, 0) - max;
            den += exp(((double)out(b, j, 0, 0)) / (1ULL << scale));
        }

        for(u64 j = 0; j < 10; ++j) {
            double x = e(b, j, 0, 0);
            e(b, j, 0, 0) = exp(x) / den;
        }
        // e(b, train_label[i+b], 0, 0) /= batchSize;
    }
}
