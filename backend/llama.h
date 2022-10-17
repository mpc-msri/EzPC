#pragma once
#include "../utils.h"
#include "cleartext.h"
#include "minillama/config.h"
#include "minillama/input_prng.h"
#include "minillama/comms.h"
#include "minillama/api.h"

template <typename T>
class Sequential;

template <typename T>
class Llama {
    static const u64 lr_fp = 1;
    static const u64 lr_scale = 7;

public:

    static void init()
    {
        prng.SetSeed(toBlock(0, time(NULL)));
        if (LlamaConfig::party == 1) {
            LlamaConfig::server = new Peer("server.dat");
            LlamaConfig::client = new Peer("client.dat");
        }
        else if (LlamaConfig::party == 2) {
            LlamaConfig::dealer = new Dealer("server.dat");
            LlamaConfig::client = waitForPeer(42002);
            LlamaConfig::peer = LlamaConfig::client;
        }
        else if (LlamaConfig::party == 3) {
            LlamaConfig::dealer = new Dealer("client.dat");
            LlamaConfig::server = new Peer("127.0.0.1", 42002);
            LlamaConfig::peer = LlamaConfig::server;
        }
        else {
            throw std::runtime_error("Invalid party");
        }
        input_prng_init();
    }

    static void finalize()
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

    static void initializeData(Tensor4D<T> &data, u64 numImagesWithServer)
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

    static void initializeWeights(Sequential<T> &model)
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

    static void output(Tensor4D<T> &a) {
        u64 sz = a.d1 * a.d2 * a.d3 * a.d4;
        if (LlamaConfig::party == 1) {
            for (int i = 0; i < sz; i++){
                LlamaConfig::client->send_mask(a.data[i]);
                LlamaConfig::server->send_mask(a.data[i]);
            }
        }
        else {
            for (int i = 0; i < sz; i++){
                auto mask = LlamaConfig::dealer->recv_mask();
                a.data[i] = a.data[i] - mask;
            }
        }
    }

    static void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }

    static void matmul(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d1);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        assert(c.d3 == 1);
        assert(c.d4 == 1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
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
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eA(a.data, a.d2, a.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(b.data, b.d1, b.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }

    static void matmulTransposeB(const Tensor4D<T> &a, const Tensor2D<T> &b, Tensor4D<T> &c) {
        assert(a.d2 == b.d2);
        assert(a.d3 == 1);
        assert(a.d4 == 1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        assert(c.d3 == 1);
        assert(c.d4 == 1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
    }

    static void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d2);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(a.data, a.d1, a.d2);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> eB(b.data, b.d2, b.d1);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(c.data, c.d1, c.d2);
        eC = eA * eB;
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

        Conv2DWrapper(input.d1, input.d2, input.d3, input.d4, fh, fw, co, 
            padding, padding, padding, padding, stride, stride, 
            input.data, input.data, filter.data, filter.data, output.data, output.data);
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
        
        Tensor2D<T> tempOutput(co, input.d1 * newH * newW);
        reshapeOutputReversed<T>(tempOutput, input.d1, (((input.d2 + 2*padding - fh)/stride) + 1), (((input.d3 + 2*padding - fw)/stride) + 1), co, output);
        Tensor2D<T> reshapedInput = reshapeInput(input, padding, stride, fh, fw);
        matmulTransposeB(tempOutput, reshapedInput, filter);
    }

    static void conv2DBiasGrad(const Tensor4D<T> &e, Tensor<T> &biasGrad)
    {
        assert(e.d4 == biasGrad.size);
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

    static void updateWeight(Tensor2D<T> &weight, const Tensor2D<T> &e, u64 scale) {
        assert(weight.d1 == e.d1);
        assert(weight.d2 == e.d2);
        for(int i = 0; i < weight.d1; i++) {
            for(int j = 0; j < weight.d2; j++) {
                e(i, j) = e(i, j) * lr_fp;
            }
        }
        truncate(e, scale+lr_scale);
        for(u64 i = 0; i < weight.d1; i++) {
            for(u64 j = 0; j < weight.d2; j++) {
                weight.data[i * weight.d2 + j] -= e(i, j);
            }
        }
    }

    static void updateBias(Tensor<T> &bias, const Tensor4D<T> &e, u64 scale) {
        // assert(e.d1 == 1);
        assert(e.d2 == bias.size);
        assert(e.d3 == 1);
        assert(e.d4 == 1);
        for (u64 i = 0; i < bias.size; i++) {
            T sum = 0;
            for(u64 j = 0; j < e.d1; ++j) {
                sum = sum + e(j, i, 0, 0);
            }
            bias.data[i] -= lr_fp * sum * (1ULL << (scale-lr_scale));
        }
    }

    static void updateBias(Tensor<T> &bias, const Tensor<T> &grad, u64 scale) {
        assert(grad.size == bias.size);
        for (u64 i = 0; i < bias.size; i++) {
            bias.data[i] -= lr_fp * grad(i) * (1ULL << (scale-lr_scale));
        }
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
        transposeFilter<T>(fh, fw, ci, co, filter, transposedFilter);
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
        int sz = in.d1 * in.d2 * in.d3 * in.d4;
        ReluTruncate(sz, in.data, in.data, out.data, out.data, shift);
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
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                        assert(drelu(i, j, k, l) == 1 || drelu(i, j, k, l) == 0);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                    }
                }
            }
        }
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
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        assert(drelu(i, j, k, l) == 0 || drelu(i, j, k, l) == 1);
                        out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                    }
                }
            }
        }
    }

    static void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
        assert(in.d1 == out.d1);
        assert(in.d2 == out.d2);
        assert(in.d3 == out.d3);
        assert(in.d4 == out.d4);
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        if constexpr (std::is_floating_point<T>::value) {
                            out(i, j, k, l) = in(i, j, k, l) / (1 << shift);
                        } else {
                            out(i, j, k, l) = in(i, j, k, l) >> shift;
                        }
                    }
                }
            }
        }
    }

    static void truncate(const Tensor4D<T> &in, u64 shift) {
        // Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2 * in.d3 * in.d4);
        // eA = eA / ((T)(1LL << shift));
        truncate(in, in, shift);
    }

    static void truncate(const Tensor2D<T> &in, u64 shift) {
    //    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
    //    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                if constexpr (std::is_floating_point<T>::value) {
                    in(i, j) = in(i, j) / ((T)(1ULL << shift));
                } else {
                    in(i, j) = in(i, j) >> shift;
                }
            }
        }
    }

    static void div(const Tensor4D<T> &in, T divisor) {
        for (u64 i = 0; i < in.d1; i++) {
            for (u64 j = 0; j < in.d2; j++) {
                for (u64 k = 0; k < in.d3; k++) {
                    for (u64 l = 0; l < in.d4; l++) {
                        in(i, j, k, l) = in(i, j, k, l) / divisor;
                    }
                }
            }
        }
    }

    static void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out) {
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

    static void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) {
        sumPool2D(ks, padding, stride, in, out);
        div(out, (T)(ks*ks));
    }

    static void sumPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out) {
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
    }

    static void avgPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out,  u64 scale) {
        sumPool2DInputGrad(ks, padding, stride, in, out);
        div(in, (T)(ks*ks));
    }

    static void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx) {
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
                        T max = std::numeric_limits<T>::lowest();
                        u64 maxIdxI = 0;
                        u64 maxIdxJ = 0;
                        for(int m = 0; m < ks; m++) {
                            for(int n = 0; n < ks; n++) {
                                T val = in(i, j*stride+m, k*stride+n, l);
                                if(val > max) {
                                    max = val;
                                    maxIdxI = m;
                                    maxIdxJ = n;
                                }
                            }
                        }
                        out(i, j, k, l) = max;
                        maxIdx(i, j, k, l) = maxIdxI * ks + maxIdxJ;
                    }
                }
            }
        }
    }

    static void maxPool2DInputGrad(u64 ks, u64 padding, u64 stride, Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<u64> &maxIdx) {
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
                        u64 maxIdxI = maxIdx(i, j, k, l) / ks;
                        u64 maxIdxJ = maxIdx(i, j, k, l) % ks;
                        in(i, j*stride+maxIdxI, k*stride+maxIdxJ, l) += out(i, j, k, l);
                    }
                }
            }
        }
    }

};
