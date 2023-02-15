#pragma once
#include "llama_base.h"

template <typename T>
class Sequential;

template <typename T>
class LlamaImproved : public LlamaBase<T> {
public:

    void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 scale) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        int sz = in.d1 * in.d2 * in.d3 * in.d4;
        // Relu2Round(sz, in.data, in.data, out.data, out.data, drelu.data, LlamaConfig::bitlength);
        for(int i = 0; i < sz; i++) {
            in.data[i] = in.data[i] % (1ULL << (LlamaConfig::bitlength - scale));
        }
        ReluExtend(sz, LlamaConfig::bitlength - scale, LlamaConfig::bitlength, in.data, out.data, drelu.data);
    }

    void truncate(T *in, T *out, u64 shift, u64 size) {
        for(u64 i = 0; i < size; i++) {
            out[i] = ((u64)in[i]) >> shift;
        }
        if (!(this->useLocalTruncation)) {
            SignExtend2(size, LlamaConfig::bitlength - shift, LlamaConfig::bitlength, in, out);
        }
    }

    virtual void truncateForward(const Tensor4D<T> &in, u64 shift) {
        std::cerr << ">> Stochastic Truncate Reduce (Local) - Start" << std::endl;
        int sz = in.d1 * in.d2 * in.d3 * in.d4;
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for(u64 i = 0; i < sz; i++) {
            in.data[i] = ((u64)(in.data[i])) >> shift;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto microsecs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "   Online Time = " << microsecs / 1000.0 << " milliseconds" << std::endl;
        evalMicroseconds += microsecs;
        arsEvalMicroseconds += microsecs;
        std::cerr << ">> Stochastic Truncate Reduce (Local) - End" << std::endl;
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        Tensor<T> maxBit((ks * ks - 1) * out.d1 * out.d2 * out.d3 * out.d4);
        maxIdx.resize(ks * ks * out.d1, out.d2, out.d3, out.d4);
        // int sz = in.d1 * in.d2 * in.d3 * in.d4;
        // u64 modulo = (1ULL << (LlamaConfig::bitlength - scale));
        // for(int i = 0; i < sz; i++) {
        //     in.data[i] = in.data[i] % modulo;
        // }
        LlamaConfig::bitlength -= scale; // why is this hack not working? new intel - it works
        MaxPoolDouble(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxBit.data);
        LlamaConfig::bitlength += scale;
        MaxPoolOneHot(out.d1, out.d2, out.d3, out.d4, ks, ks, maxBit.data, maxIdx.data);
    }

    void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        //throw std::runtime_error("Not implemented");
        MaxPoolBackward(out.d1, out.d2, out.d3, out.d4, ks, ks, padding, padding, padding, padding, stride, stride, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxIdx.data);
    }

    void optimize(Sequential<T> &model) {

    }

};
