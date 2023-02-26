#pragma once
#include <sytorch/utils.h>
#include "cleartext.h"
#include <llama/config.h>
#include <llama/input_prng.h>
#include <llama/comms.h>
#include <llama/api.h>
#include "backend.h"

template <typename T>
class Sequential;

template <typename T>
class LlamaBase : public Backend<T> {
public:
    const u64 lr_fp = 1;
    const u64 lr_scale = 6;
    const u64 mom_fp = 29;
    const u64 mom_scale = 5;
    const bool useLocalTruncation = false;

    void init(std::string ip, bool ramdisk = false)
    {
        u64 seedKey = 0xdeadbeefbadc0ffe;
        for(int i = 0; i < 256; ++i) {
            LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(i, seedKey));
        }
        if (LlamaConfig::party == 1) {
            LlamaConfig::server = new Peer("server.dat");
            LlamaConfig::client = new Peer("client.dat");
        }
        else if (LlamaConfig::party == 2) {
            LlamaConfig::dealer = new Dealer("server.dat", ramdisk);
            LlamaConfig::client = waitForPeer(42002);
            LlamaConfig::peer = LlamaConfig::client;
        }
        else if (LlamaConfig::party == 3) {
            LlamaConfig::dealer = new Dealer("client.dat", ramdisk);
            LlamaConfig::server = new Peer(ip, 42002);
            LlamaConfig::peer = LlamaConfig::server;
        }
        else {
            throw std::runtime_error("Invalid party");
        }
        input_prng_init();
    }

    void finalize()
    {
        switch (LlamaConfig::party)
		{
		case 1:
				LlamaConfig::server->close();
				LlamaConfig::client->close();
				break;
		case 2:
				LlamaConfig::dealer->close();
				LlamaConfig::client->close();
				break;
		case 3:
				LlamaConfig::dealer->close();
				LlamaConfig::server->close();
		}
    }

    void initializeData(Tensor4D<T> &data, u64 numImagesWithServer)
    {
        u64 b1 = numImagesWithServer * data.d2 * data.d3 * data.d4;
        u64 b2 = (data.d1 - numImagesWithServer) * data.d2 * data.d3 * data.d4;
        if (LlamaConfig::party == 1) {
            input_layer(nullptr, data.data, b1, 2);
            input_layer(nullptr, data.data + b1, b2, 3);
        }
        else {
            if (LlamaConfig::party == 2) {
                Tensor4D<T> tmp(numImagesWithServer, data.d2, data.d3, data.d4);
                input_layer(data.data, tmp.data, b1, 2);
                input_layer(data.data + b1, nullptr, b2, 3);
            }
            else {
                Tensor4D<T> tmp(data.d1 - numImagesWithServer, data.d2, data.d3, data.d4);
                input_layer(data.data, nullptr, b1, 2);
                input_layer(data.data + b1, tmp.data, b2, 3);
            }
        }
    }

    void inputA(Tensor4D<T> &data)
    {
        u64 b1 = data.d1 * data.d2 * data.d3 * data.d4;
        if (LlamaConfig::party == 1) {
            input_layer(nullptr, data.data, b1, 2);
        }
        else {
            if (LlamaConfig::party == 2) {
                Tensor4D<T> tmp(data.d1, data.d2, data.d3, data.d4);
                input_layer(data.data, tmp.data, b1, 2);
            }
            else {
                input_layer(data.data, nullptr, b1, 2);
            }
        }
    }

    void initializeWeights(Sequential<T> &model)
    {
        // DEALER selects the inital weights and sends them to parties as keys
        for(int i = 0; i < model.layers.size(); ++i)
        {
            if (model.layers[i]->name == "Conv2D" || model.layers[i]->name == "FC")
            {
                auto &weights = model.layers[i]->getweights();
                auto &bias = model.layers[i]->getbias();
                if (LlamaConfig::party == 1)
                {
                    // weights.fill(1);
                    // bias.fill(1);
                    LlamaConfig::server->send_ge_array(weights.data, weights.d1 * weights.d2);
                    LlamaConfig::server->send_ge_array(bias.data, bias.size);
                    LlamaConfig::client->send_ge_array(weights.data, weights.d1 * weights.d2);
                    LlamaConfig::client->send_ge_array(bias.data, bias.size);
                    weights.fill(0);
                    bias.fill(0);
                }
                else
                {
                    LlamaConfig::dealer->recv_ge_array(weights.data, weights.d1 * weights.d2);
                    LlamaConfig::dealer->recv_ge_array(bias.data, bias.size);
                }
            }
        }
    }

    void output(Tensor4D<T> &a) {
        u64 sz = a.d1 * a.d2 * a.d3 * a.d4;
        if (LlamaConfig::party == 1) {
            for (int i = 0; i < sz; i++){
                LlamaConfig::client->send_mask(a.data[i]);
                LlamaConfig::server->send_mask(a.data[i]);
                a.data[i] = 0;
            }
        }
        else {
            for (int i = 0; i < sz; i++){
                auto mask = LlamaConfig::dealer->recv_mask();
                a.data[i] = a.data[i] - mask;
            }
        }
    }

    void output(Tensor2D<T> &a) {
        u64 sz = a.d1 * a.d2;
        if (LlamaConfig::party == 1) {
            for (int i = 0; i < sz; i++){
                LlamaConfig::client->send_mask(a.data[i]);
                LlamaConfig::server->send_mask(a.data[i]);
                a.data[i] = 0;
            }
        }
        else {
            for (int i = 0; i < sz; i++){
                auto mask = LlamaConfig::dealer->recv_mask();
                a.data[i] = a.data[i] - mask;
            }
        }
    }

    void output(Tensor<T> &a) {
        u64 sz = a.size;
        if (LlamaConfig::party == 1) {
            for (int i = 0; i < sz; i++){
                LlamaConfig::client->send_mask(a.data[i]);
                LlamaConfig::server->send_mask(a.data[i]);
                a.data[i] = 0;
            }
        }
        else {
            for (int i = 0; i < sz; i++){
                auto mask = LlamaConfig::dealer->recv_mask();
                a.data[i] = a.data[i] - mask;
            }
        }
    }

    void output(T *a, u64 sz) {
        if (LlamaConfig::party == 1) {
            for (int i = 0; i < sz; i++){
                LlamaConfig::client->send_mask(a[i]);
                LlamaConfig::server->send_mask(a[i]);
                a[i] = 0;
            }
        }
        else {
            for (int i = 0; i < sz; i++){
                auto mask = LlamaConfig::dealer->recv_mask();
                a[i] = a[i] - mask;
            }
        }
    }

    void ss2m(T *data, u64 size)
    {
        std::cerr << ">> SS2M - Start" << std::endl;
        if (LlamaConfig::party == 1) {
            for (int i = 0; i < size; i++){
                data[i] = random_ge(64);
                auto p = splitShare(data[i], 64);
                LlamaConfig::client->send_mask(p.first);
                LlamaConfig::server->send_mask(p.second);
            }
        }
        else {
            for (int i = 0; i < size; i++){
                auto mask = LlamaConfig::dealer->recv_mask();
                data[i] = data[i] + mask;
            }
            reconstruct(size, data, 64);
        }
        std::cerr << ">> SS2M - End" << std::endl;
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        MatMul2D(a.d1, a.d2, b.d2, a.data, a.data, b.data, b.data, c.data, c.data, true);
    }

    void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d1);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        assert(c.d3 == 1);
        assert(c.d4 == 1);

        MatMul2D(a.d1, a.d2, b.d2, a.data, a.data, b.data, b.data, c.data, c.data, true);
    }

    void matmulTransposeA(const Tensor4D<T> &a, const Tensor4D<T> &b, Tensor2D<T> &c) {
        assert(a.d1 == b.d1);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(b.d3 == 1);
        assert(b.d4 == 1);
        assert(c.d1 == a.d2);
        assert(c.d2 == b.d2);

        Tensor2D<T> aTranspose(a.d2, a.d1);
        for(int i = 0; i < a.d1; ++i)
            for(int j = 0; j < a.d2; ++j)
                aTranspose(j, i) = a(i, j, 0, 0);
        MatMul2D(aTranspose.d1, aTranspose.d2, b.d2, aTranspose.data, aTranspose.data, b.data, b.data, c.data, c.data, true);
    }

    void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d2);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        assert(c.d3 == 1);
        assert(c.d4 == 1);
        Tensor2D<T> bTranspose(b.d2, b.d1);
        for(int i = 0; i < b.d1; ++i)
            for(int j = 0; j < b.d2; ++j)
                bTranspose(j, i) = b(i, j);
        matmul(a, bTranspose, c);
    }

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output)
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

        Conv2DWrapper(input.d1, input.d2, input.d3, input.d4, fh, fw, co, 
            padding, padding, padding, padding, stride, stride, 
            input.data, input.data, filter.data, filter.data, output.data, output.data);
    }

    void conv2DFilterGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, Tensor2D<T> &filter, const Tensor4D<T> &output)
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
        
        Tensor2D<T> tempOutput(co, input.d1 * newH * newW);
        reshapeOutputReversed<T>(tempOutput, input.d1, (((input.d2 + 2*padding - fh)/stride) + 1), (((input.d3 + 2*padding - fw)/stride) + 1), co, output);
        Tensor2D<T> reshapedInput = reshapeInputTransposed(input, padding, stride, fh, fw);
        matmul(tempOutput, reshapedInput, filter);
    }

    void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad)
    {
        assert(e.d4 == biasGrad.size);
        biasGrad.fill(0);
        for(int i = 0; i < e.d1; i++) {
            for(int j = 0; j < e.d2; j++) {
                for(int k = 0; k < e.d3; k++) {
                    for(int l = 0; l < e.d4; l++) {
                        biasGrad(l) += e(i, j, k, l);
                    }
                }
            }
        }
    }

    void updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, Tensor2D<T> &Vw, u64 scale) {
        assert(weight.d1 == e.d1);
        assert(weight.d2 == e.d2);
        if (mom_fp == 0) {
            for(int i = 0; i < weight.d1; i++) {
                for(int j = 0; j < weight.d2; j++) {
                    e(i, j) = e(i, j) * lr_fp;
                }
            }
            Backend<T>::truncate(e, scale+lr_scale);
            for(u64 i = 0; i < weight.d1; i++) {
                for(u64 j = 0; j < weight.d2; j++) {
                    weight.data[i * weight.d2 + j] -= e(i, j);
                }
            }
        }
        else {
            assert(weight.d1 = Vw.d1);
            assert(weight.d2 = Vw.d2);
            for(int i = 0; i < weight.d1; i++) {
                for(int j = 0; j < weight.d2; j++) {
                    Vw(i, j) = mom_fp * Vw(i, j) + (1ULL<<mom_scale) * e(i, j);
                }
            }
            Backend<T>::truncate(Vw, mom_scale);
            for(int i = 0; i < weight.d1; i++) {
                for(int j = 0; j < weight.d2; j++) {
                    weight(i, j) = (1ULL<<(lr_scale+scale)) * weight(i, j) - lr_fp * Vw(i, j);
                }
            }
            Backend<T>::truncate(weight, lr_scale+scale);
        }
    }

    void updateBias(Tensor<T> &bias, const Tensor4D<T> &e, const Tensor<T> &Vb, u64 scale) {
        // assert(e.d1 == 1);
        assert(e.d2 == bias.size);
        assert(e.d3 == 1);
        assert(e.d4 == 1);
        if (mom_fp == 0) {
            for (u64 i = 0; i < bias.size; i++) {
                T sum = 0;
                for(u64 j = 0; j < e.d1; ++j) {
                    sum = sum + e(j, i, 0, 0);
                }
                if (scale > lr_scale) {
                    bias.data[i] -= lr_fp * sum * (1ULL << (scale-lr_scale));
                }
                else {
                    throw std::runtime_error("scale should be greater than lr_scale");
                }
            }
        }
        else {
            assert(bias.size == Vb.size);
            
            for (u64 i = 0; i < bias.size; i++) {
                T sum = 0;
                for(u64 j = 0; j < e.d1; ++j) {
                    sum = sum + e(j, i, 0, 0);
                }
                Vb(i) = mom_fp * Vb(i) + (1ULL << (scale + mom_scale - lr_scale)) * sum;
            }
            Backend<T>::truncate(Vb, mom_scale);
            for (u64 i = 0; i < bias.size; i++) {
                bias(i) = bias(i) - lr_fp * Vb(i);
            }
            
        }
    }

    void updateBias(Tensor<T> &bias, const Tensor<T> &grad, Tensor<T> &Vb, u64 scale) {
        assert(grad.size == bias.size);
        if (mom_fp == 0) {
            for (u64 i = 0; i < bias.size; i++) {
                if (scale > lr_scale) {
                    bias.data[i] -= lr_fp * grad.data[i] * (1ULL << (scale-lr_scale));
                }
                else {
                    throw std::runtime_error("scale should be greater than lr_scale");
                }
            }
        }
        else {
            // Scale of Vb would be 2 * scale - lr_scale
            for (u64 i = 0; i < bias.size; i++) {
                Vb(i) = mom_fp * Vb(i) + (1ULL << (scale + mom_scale - lr_scale)) * grad(i);
            }
            Backend<T>::truncate(Vb, mom_scale);
            for (u64 i = 0; i < bias.size; i++) {
                bias(i) = bias(i) - lr_fp * Vb(i);
            }
        }
    }

    void conv2DInputGrad(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, Tensor4D<T> &input, const Tensor2D<T> &filter, const Tensor4D<T> &output)
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
        transposeFilter<T>(fh, fw, ci, co, filter, transposedFilter);
        conv2D(fh, fw, fh-padding-1, stride, co, ci, output, transposedFilter, input);
    }

    void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        for(int i = 0; i < in.d1; i++) {
            for(int j = 0; j < newH; j++) {
                for(int k = 0; k < newW; k++) {
                    for(int l = 0; l < in.d4; l++) {
                        T sum = 0;
                        for(int m = 0; m < ks; m++) {
                            for(int n = 0; n < ks; n++) {
                                sum += in(i, j*stride+m, k*stride+n, l);
                            }
                        }
                        out(i, j, k, l) = sum;
                    }
                }
            }
        }
    }

    void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2*padding - ks)/stride + 1;
        u64 newW = (in.d3 + 2*padding - ks)/stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);
        in.zero();
        for(int i = 0; i < in.d1; i++) {
            for(int j = 0; j < newH; j++) {
                for(int k = 0; k < newW; k++) {
                    for(int l = 0; l < in.d4; l++) {
                        for(int m = 0; m < ks; m++) {
                            for(int n = 0; n < ks; n++) {
                                in(i, j*stride+m, k*stride+n, l) += out(i, j, k, l);
                            }
                        }
                    }
                }
            }
        }
        // Hack for Pirhana
        // assert(std::popcount(ks * ks) == 1);
        // Backend<T>::truncate(in, log2(ks * ks));
    }

    void div(const Tensor4D<T> &in, T divisor) {
        throw std::runtime_error("Not implemented");
    }

    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) {
        sumPool2D(ks, padding, stride, in, out);
        div(out, (T)(ks*ks));
    }

    u64 log2(u64 x) {
        u64 y = 0;
        while (x >>= 1) y++;
        return y;
    }

    void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out,  u64 scale) {
        sumPool2DInputGrad(ks, padding, stride, in, out);
        div(in, (T)(ks*ks));
    }

    void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        assert(in.d1 == drelu.d1);
        assert(in.d2 == drelu.d2);
        assert(in.d3 == drelu.d3);
        assert(in.d4 == drelu.d4);
        Select(in.d1 * in.d2 * in.d3 * in.d4, drelu.data, in.data, out.data);
    }

    void batchNorm2dInference(const Tensor<T> &A, const Tensor<T> &B, const Tensor4D<T> &x, Tensor4D<T> &y, u64 scale)
    {
        assert(A.size == B.size);
        assert(A.size == x.d4);
        assert(x.d4 == y.d4);
        assert(x.d1 == y.d1);
        assert(x.d2 == y.d2);
        assert(x.d3 == y.d3);
        // replicate A and B
        Tensor4D<T> A2(x.d1, x.d2, x.d3, x.d4);
        for (int i = 0; i < x.d1; ++i) {
            for(int j = 0; j < x.d2; ++j) {
                for(int k = 0; k < x.d3; ++k) {
                    for(int l = 0; l < x.d4; ++l) {
                        A2(i, j, k, l) = A(l);
                    }
                }
            }
        }

        ElemWiseSecretSharedVectorMult(x.d1 * x.d2 * x.d3 * x.d4, x.data, x.data, A2.data, A2.data, y.data, y.data);

        for(int i = 0; i < x.d1; ++i) {
            for(int j = 0; j < x.d2; ++j) {
                for(int k = 0; k < x.d3; ++k) {
                    for(int l = 0; l < x.d4; ++l) {
                        y(i, j, k, l) += B(l);
                    }
                }
            }
        }

        // this->truncateForward(y, scale);

    }
};