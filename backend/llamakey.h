#pragma once
#include "../utils.h"
#include "cleartext.h"

template <typename T>
class LlamaKey {
public:
    static u64 serverkeysize; // in bits
    static u64 clientkeysize; // in bits
    static u64 bw;
    static bool probablistic;
    static constexpr double gb = 1024ULL * 1024ULL * 1024ULL * 8ULL;
    static const u64 lr_fp = 1;
    static const u64 lr_scale = 7;
    static bool verbose;

private:
    static u64 dcfcost(u64 bin, u64 bout) {
        return bin * (130 + bout) + 128 + bout;
    }

    static u64 selectcost(u64 bin) {
        return 5 * bin;
    }

    static u64 signextcost(u64 bin, u64 bout) {
        // return dcfcost(bin, 1) + selectcost(bout);
        return dcfcost(bin, bout);
    }

    static u64 truncatereducecost(u64 bin, u64 s) {
        return probablistic ? 0 : dcfcost(s, bin);
    }

    static u64 truncatecost(u64 bin, u64 s) {
        if (probablistic) {
            return signextcost(bin-s, bin);
        }
        return truncatereducecost(bin, s) + signextcost(bin-s, bin);
        // return dcfcost(s, bin) + dcfcost(bin - 1, 2*bin); // this is what we currently do
    }

    static u64 relutruncatecost(u64 bin, u64 s) {
        if (probablistic) {
            return dcfcost(bin, 1) + selectcost(bin);
        }
        else {
            // return dcfcost(bin, bin) + dcfcost(s, bin) + selectcost(bin) + bin;
            return dcfcost(bin, s) + dcfcost(s, bin) + selectcost(bin) + bin; // after neha's fix
        }
    }

    static u64 drelucost(u64 bin) {
        return dcfcost(bin, 1) + 1;
    }

    static u64 relucost(u64 bin) {
        return drelucost(bin) + selectcost(bin);
    }

    static u64 maxcost(u64 bin) {
        return relucost(bin);
    }

public:

    static void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        if (verbose) std::cout << "matmul key size (server) = " << bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2) / gb << std::endl;
        serverkeysize += bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2);
    }

    static void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d1);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        assert(c.d3 == 1);
        assert(c.d4 == 1);
        if (verbose) std::cout << "matmul key size (server) = " << bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2) / gb << std::endl;
        serverkeysize += bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2);
    }

    static void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c) {
        assert(a.d1 == b.d1);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(b.d3 == 1);
        assert(b.d4 == 1);
        assert(c.d1 == a.d2);
        assert(c.d2 == b.d2);
    //    c.zero();
        if (verbose) std::cout << "matmul key size (server) = " << bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2) / gb << std::endl;
        serverkeysize += bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2);
    }

    static void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d2);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        assert(c.d3 == 1);
        assert(c.d4 == 1);
        if (verbose) std::cout << "matmul key size (server) = " << bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2) / gb << std::endl;
        serverkeysize += bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2);
    }

    static void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d2);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        if (verbose) std::cout << "matmul key size (server) = " << bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2) / gb << std::endl;
        serverkeysize += bw * (a.d1 * a.d2 + b.d1 * b.d2 + c.d1 * c.d2);
    }


    static void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output)
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

        if (verbose) std::cout << "conv2D key size (server) = " << bw * (input.d1 * input.d2 * input.d3 * input.d4 + filter.d1 * filter.d2 + output.d1 * output.d2 * output.d3 * output.d4) / gb << std::endl;
        serverkeysize += bw * (input.d1 * input.d2 * input.d3 * input.d4 + filter.d1 * filter.d2 + output.d1 * output.d2 * output.d3 * output.d4);
    }

    static void conv2DFilterGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, Tensor2D<T> &filter, const Tensor4D<T> &output)
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
        
        if (verbose) std::cout << "conv2D key size (server) = " << bw * (input.d1 * input.d2 * input.d3 * input.d4 + filter.d1 * filter.d2 + output.d1 * output.d2 * output.d3 * output.d4) / gb << std::endl;
        serverkeysize += 64 * (input.d1 * input.d2 * input.d3 * input.d4 + filter.d1 * filter.d2 + output.d1 * output.d2 * output.d3 * output.d4);
    }

    static void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad)
    {
        assert(e.d4 == biasGrad.size);
    }

    static void updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, Tensor2D<T> &Vw, u64 scale) {
        assert(weight.d1 == e.d1);
        assert(weight.d2 == e.d2);
        // step 1 : convert lr to fixed point (free)
        // step 2 : delta = e * lr (free)
        // step 3 : truncate(delta)
        // 2 * as momentum needs two truncations
        u64 cost = 2 * weight.d1 * weight.d2 * truncatecost(bw, scale+lr_scale);
        if (verbose) std::cout << "updateWeight key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
        // step 4 : weight = weight - delta (free)

    }

    static void updateBias(Tensor<T> &bias, const Tensor4D<T> &e, Tensor<T> &Vb, u64 scale) {
        // assert(e.d1 == 1);
        assert(e.d2 == bias.size);
        assert(e.d3 == 1);
        assert(e.d4 == 1);
        // step 1 : convert lr to fixed point (free)
        // step 2 : delta = e * lr (free)
        // step 4 : bias = bias - delta (free) (bias is already with scale 2s)
    }

    static void updateBias(Tensor<T> &bias, const Tensor<T> &grad, Tensor<T> &Vb, u64 scale) {
        assert(grad.size == bias.size);
        // step 1 : convert lr to fixed point (free)
        // step 2 : delta = e * lr (free)
        // step 4 : bias = bias - delta (free) (bias is already with scale 2s)
    }

    static void conv2DInputGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, Tensor4D<T> &input, const Tensor2D<T> &filter, const Tensor4D<T> &output)
    {
        if (stride != 1) {
            std::cerr << "Stride not supported in backward pass yet!" << std::endl;
        }
        assert(input.d4 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fh * fw * ci);
        u64 newH = (((input.d2 + 2*padding - fh)/stride) + 1);
        u64 newW = (((input.d3 + 2*padding - fw)/stride) + 1);
        assert(output.d1 == input.d1);
        assert(output.d2 == newH);
        assert(output.d3 == newW);
        assert(output.d4 == co);
        
        Tensor2D<T> transposedFilter(ci, fh * fw * co);
        // transposeFilter<T>(fh, fw, ci, co, filter, transposedFilter);
        conv2D(fh, fw, fh-padding-1, stride, co, ci, output, transposedFilter, input);
    }

    static void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        u64 cost = in.d1 * in.d2 * in.d3 * in.d4 * relutruncatecost(bw, shift);
        if (verbose) std::cout << "relutruncate key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
    }

    static void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        u64 cost = in.d1 * in.d2 * in.d3 * in.d4 * relucost(bw);
        if (verbose) std::cout << "relu key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
    }

    static void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        u64 cost = in.d1 * in.d2 * in.d3 * in.d4 * selectcost(bw);
        if (verbose) std::cout << "select key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
    }

    static void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        u64 cost = in.d1 * in.d2 * in.d3 * in.d4 * truncatecost(bw, shift);
        if (verbose) std::cout << "truncate key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
    }

    static void truncate(const Tensor4D<T> &in, u64 shift) {
        truncate(in, in, shift);
    }

    static void truncate(const Tensor2D<T> &in, u64 shift) {
        u64 cost = in.d1 * in.d2 * truncatecost(bw, shift);
        if (verbose) std::cout << "truncate key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
    }

    static void div(const Tensor4D<T> &in, T divisor, u64 scale) {
        if (__builtin_popcount(divisor) == 1) {
            // if divisor is a power of 2, we can increase the shift of a later truncate layer and we get it for free (if we use probablistic truncation)
            if (verbose) std::cout << "div key size = " << 0 << std::endl;
        }
        else {
            u64 cost = in.d1 * in.d2 * in.d3 * in.d4 * truncatecost(bw, scale);
            if (verbose) std::cout << "div key size = " << cost / gb << std::endl;
            serverkeysize += cost;
            clientkeysize += cost;
        }
    }

    static void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
    }

    static void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) {
        sumPool2D(ks, padding, stride, in, out);
        div(out, (T)(ks*ks), scale);
    }

    static void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
    }

    static void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, u64 scale) {
        sumPool2DInputGrad(ks, padding, stride, in, out);
        div(out, (T)(ks*ks), scale);
    }

    static void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        u64 cost = out.d1 * out.d2 * out.d3 * out.d4 * (ks * ks - 1) * maxcost(bw);
        cost += out.d1 * out.d2 * out.d3 * out.d4 * (2 * ks * ks - 3) * 4;
        if (verbose) std::cout << "maxpool key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
    }

    static void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        u64 cost = out.d1 * out.d2 * out.d3 * out.d4 * (ks * ks - 1) * selectcost(bw);
        if (verbose) std::cout << "maxpool key size = " << cost / gb << std::endl;
        serverkeysize += cost;
        clientkeysize += cost;
    }

};

template<typename T>
u64 LlamaKey<T>::serverkeysize = 0;

template<typename T>
u64 LlamaKey<T>::clientkeysize = 0;

template<typename T>
bool LlamaKey<T>::probablistic = true;

template<typename T>
bool LlamaKey<T>::verbose = false;

template<typename T>
u64 LlamaKey<T>::bw = 64;
