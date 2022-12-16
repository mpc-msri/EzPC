#pragma once
#include "tensor.h"
#include "backend/sci/src/library_float.h"
#include "backend/minillama/stats.h"

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
            if(in(b, j, 0, 0) > max) {
                max = in(b, j, 0, 0);
            }
        }
        i64 den = 0;
        i64 exps[numClasses];
        for(u64 j = 0; j < numClasses; ++j) {
            i64 x = in(b, j, 0, 0) - max;
            exps[j] = (x > (-2*(1LL << scale))) ? ((x + 2*(1LL << scale)) / 2) : 0;
            den += exps[j];
            // std::cout << x << std::endl;
        }
        den = den * batchSize;
        den = pirhana_inverse(den, scale);

        for(u64 j = 0; j < numClasses; ++j) {
            auto t = exps[j] * den;
            t >>= scale;
            out(b, j, 0, 0) = (i64)(t);
        }
    }
}

inline void secfloat_init(int secfloatParty, std::string secfloatAddr)
{
    __party = secfloatParty;
    __address = secfloatAddr;
    __init(0, nullptr);
}

inline void softmax_secfloat(const Tensor4D<u64> &in, Tensor4D<u64> &out, u64 scale, int llamaParty)
{
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == 1);
    assert(in.d4 == 1);
    assert(out.d3 == 1);
    assert(out.d4 == 1);

    Tensor4D<u64> inFloat(in.d1, in.d2, 4, 1);
    FixToFloat(in.d1 * in.d2, in.data, inFloat.data, scale);
    Tensor4D<u64> outFloat(in.d1, in.d2, 4, 1);
    if (llamaParty != 1) {
        vector < vector < FPArray > > inpFloatSecfloat = make_vector_float(llamaParty-1, in.d1, in.d2);
        for(int i = 0; i < in.d1; ++i) {
            for(int j = 0; j < in.d2; ++j) {
                inpFloatSecfloat[i][j].m[0] = inFloat(i, j, 0, 0);
                inpFloatSecfloat[i][j].e[0] = inFloat(i, j, 1, 0);
                inpFloatSecfloat[i][j].z[0] = inFloat(i, j, 2, 0);
                inpFloatSecfloat[i][j].s[0] = inFloat(i, j, 3, 0);
            }
        }
        vector < vector < FPArray > > outFloatSecfloat = make_vector_float(llamaParty-1, in.d1, in.d2);

        std::cerr << ">> Softmax (SecFloat) - Start" << std::endl;
        auto secfloat_start = std::chrono::high_resolution_clock::now();
        auto secfloat_comm_start = __get_comm();
        auto secfloat_round_start = __iopack->get_rounds();
        Softmax2(in.d1, in.d2, inpFloatSecfloat, outFloatSecfloat);
        auto secfloat_round_end = __iopack->get_rounds();
        auto secfloat_comm_end = __get_comm();
        auto secfloat_end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(secfloat_end - secfloat_start).count();
        evalMicroseconds += eval_time;
        secFloatComm += (uint64_t)(secfloat_comm_end - secfloat_comm_start);
        numRounds += (secfloat_round_end - secfloat_round_start);
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << ">> Softmax (SecFloat) - End" << std::endl;

        for(int i = 0; i < in.d1; ++i) {
            for(int j = 0; j < in.d2; ++j) {
                outFloat(i, j, 0, 0) = outFloatSecfloat[i][j].m[0];
                outFloat(i, j, 1, 0) = outFloatSecfloat[i][j].e[0];
                outFloat(i, j, 2, 0) = outFloatSecfloat[i][j].z[0];
                outFloat(i, j, 3, 0) = outFloatSecfloat[i][j].s[0];
            }
        }
    }

    FloatToFix(in.d1 * in.d2, outFloat.data, out.data, scale);
}
