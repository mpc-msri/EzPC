#pragma once
#include <sytorch/utils.h>
#include "cleartext.h"
#include <llama/config.h>
#include <llama/input_prng.h>
#include <llama/comms.h>
#include <llama/api.h>
#include "backend.h"
#include <sytorch/layers/layers.h>

template <typename T>
class Sequential;

template <typename T>
class LlamaBase : public Backend<T> {
public:
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

    void initializeInferencePartyB(Tensor<T>&data){
        u64 size = data.size();
        if(LlamaConfig::party == 1){
            input_layer(nullptr,data.data, size, 3);
        }
        else{
            Tensor<T> tmp(data.shape);
            input_layer(data.data, tmp.data, size, 3);
        }
    }

    void initializeInferencePartyA(LayerGraphNode<T> *root) {
         topologicalApply(root, [&](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            auto layer = node->layer;
            if(layer->name == "Conv2D" || layer->name == "FC" || layer->name == "Conv3D" || layer->name == "ConvTranspose3D") {
                auto& weights = layer->getweights();
                auto& bias = layer->getbias();
                if(LlamaConfig::party == 1){
                    input_layer(nullptr, weights.data, weights.d1 * weights.d2, 2);
                    if (layer->useBias) {
                        input_layer(nullptr, bias.data, bias.size, 2);
                    }
                }
                else{
                    Tensor2D<T> tmp(weights.d1, weights.d2);
                    input_layer(weights.data, tmp.data, weights.d1 * weights.d2, 2);
                    if(layer->useBias){
                        Tensor1D<T> tmp2(bias.size);
                        input_layer(bias.data, tmp2.data, bias.size, 2);
                    }
                }
            }
            else if (layer->name.find("BatchNorm2dInference") != std::string::npos) {
                auto bn = (BatchNorm2dInference<T>*) layer;
                auto channel = bn->A.size;
                if(LlamaConfig::party == 1){
                    input_layer(nullptr, bn->A.data, channel, 2);
                    input_layer(nullptr, bn->B.data, channel, 2);
                }
                else{
                    Tensor1D<T> tmp(channel);
                    input_layer(bn->A.data, tmp.data, channel, 2);
                    input_layer(bn->B.data, tmp.data, channel, 2);
                }
            }
        });
    }

    void inputA(Tensor<T> &data)
    {
        u64 b1 = data.size();
        if (LlamaConfig::party == 1) {
            input_layer(nullptr, data.data, b1, 2);
        }
        else {
            if (LlamaConfig::party == 2) {
                Tensor<T> tmp(data.shape);
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
            if (model.layers[i]->name == "Conv2D" || model.layers[i]->name == "FC" || model.layers[i]->name == "Conv3D" || model.layers[i]->name == "ConvTranspose3D")
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

     void outputA(Tensor<T> &a) {
        outputA(a.data, a.size());
    }

    void output(Tensor<T> &a) {
        output(a.data, a.size());
    }

    void outputA(Tensor2D<T> &a) {
        outputA(a.data, a.d1 * a.d2);
    }

    void output(Tensor2D<T> &a) {
        output(a.data, a.d1 * a.d2);
    }

    void outputA(Tensor1D<T> &a) {
        outputA(a.data, a.size);
    }

    void output(Tensor1D<T> &a) {
        output(a.data, a.size);
    }

    void outputA(T *a, u64 sz) {
        if (LlamaConfig::party == 1) {
            for (int i = 0; i < sz; i++){
                LlamaConfig::client->send_mask(a[i]);
                a[i] = 0;
            }
        }
        else if(LlamaConfig::party ==3) {
            for (int i = 0; i < sz; i++){
                auto mask = LlamaConfig::dealer->recv_mask();
                a[i] = a[i] - mask;
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

    void matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d1 == b.d1);
        assert(c.d1 == a.d2);
        assert(c.d2 == b.d2);

        Tensor2D<T> aTranspose(a.d2, a.d1);
        for(int i = 0; i < a.d1; ++i)
            for(int j = 0; j < a.d2; ++j)
                aTranspose(j, i) = a(i, j);
        MatMul2D(aTranspose.d1, aTranspose.d2, b.d2, aTranspose.data, aTranspose.data, b.data, b.data, c.data, c.data, true);
    }

    void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c) {
        assert(a.d2 == b.d2);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
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

    void conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 stride, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
    {
        assert(input.d5 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fd * fh * fw * ci);
        u64 newD = (((input.d2 + 2*pd - fd)/stride) + 1);
        u64 newH = (((input.d3 + 2*ph - fh)/stride) + 1);
        u64 newW = (((input.d4 + 2*pw - fw)/stride) + 1);
        assert(output.d1 == input.d1);
        assert(output.d2 == newD);
        assert(output.d3 == newH);
        assert(output.d4 == newW);
        assert(output.d5 == co);

        Conv3DWrapper(input.d1, input.d2, input.d3, input.d4, input.d5, fd, fh, fw, co, 
            pd, pd, ph, ph, pw, pw, stride, stride, stride, 
            input.data, filter.data, output.data);
    }

    void convTranspose3D(u64 fd, u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
    {
        assert(input.d5 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fd * fh * fw * ci);
        u64 newD = (((input.d2 - 1)*stride + fd - 2*padding));
        u64 newH = (((input.d3 - 1)*stride + fh - 2*padding));
        u64 newW = (((input.d4 - 1)*stride + fw - 2*padding));
        assert(output.d1 == input.d1);
        assert(output.d2 == newD);
        assert(output.d3 == newH);
        assert(output.d4 == newW);
        assert(output.d5 == co);

        ConvTranspose3DWrapper(input.d1, input.d2, input.d3, input.d4, input.d5, fd, fh, fw, co, 
            padding, padding, padding, padding, padding, padding, stride, stride, stride, 
            output.d2, output.d3, output.d4, input.data, filter.data, output.data);
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

    void div(const Tensor<T> &in, T divisor, u64 scale) {
        if (!(divisor & (divisor - 1))) {
            Backend<T>::truncate(in, log2(divisor), 3);
        }
        else {
            T divfp = (1LL << scale) / divisor;
            u64 sz = in.size();
            for (u64 i = 0; i < sz; i++) {
                in.data[i] *= divfp;
            }
            Backend<T>::truncate(in, scale, 3);
        }
    }

    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale) {
        sumPool2D(ks, padding, stride, in, out);
        div(out.as_nd(), (T)(ks*ks), scale);
    }

    u64 log2(u64 x) {
        u64 y = 0;
        while (x >>= 1) y++;
        return y;
    }

    void batchNorm2dInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor4D<T> &x, Tensor4D<T> &y, u64 scale)
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

    void add(const std::vector<Tensor<T> *> &in, const Tensor<T> &out) {
        auto ct = new ClearText<T>;
        ct->add(in, out);
        delete ct;
    }
};