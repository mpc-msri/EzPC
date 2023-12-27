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

    void init(std::string ip, bool ramdisk = true,bool ramdisk_path=false)
    {
        u64 seedKey = 0xdeadbeefbadc0ffe;
        for(int i = 0; i < 256; ++i) {
            LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(i, seedKey));
        }
        if (LlamaConfig::party == 1) {
            std::cerr<<ramdisk<<ramdisk_path<<"\n";
            if (ramdisk && ramdisk_path)
            {
            LlamaConfig::server = new Peer("/tmp/ramdisk/server.dat");
            LlamaConfig::client = new Peer("/tmp/ramdisk/client.dat");
            }
            else
            {
            LlamaConfig::server = new Peer("server.dat");
            LlamaConfig::client = new Peer("client.dat");
            }
        }
        else if (LlamaConfig::party == 2) {
            if(ramdisk && ramdisk_path)
            {
            LlamaConfig::dealer = new Dealer("/tmp/ramdisk/server.dat", ramdisk,ramdisk_path);
            }
            else
            {
            LlamaConfig::dealer = new Dealer("server.dat", ramdisk,ramdisk_path);
            }
            LlamaConfig::client = waitForPeer(42005);
            LlamaConfig::peer = LlamaConfig::client;
        }
        else if (LlamaConfig::party == 3) {
            if(ramdisk && ramdisk_path)
            {
            LlamaConfig::dealer = new Dealer("/tmp/ramdisk/client.dat", ramdisk,ramdisk_path);
            }
            else
            {
            LlamaConfig::dealer = new Dealer("client.dat", ramdisk,ramdisk_path);
            }
            LlamaConfig::server = new Peer(ip, 42005);
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
#ifdef Do_Masking
                input_no_prng_with_frontend(nullptr, data.data, size, 3);
#else
                input_layer(nullptr, data.data, size, 3);
#endif        
        }
        else{
            Tensor<T> tmp(data.shape);
#ifdef Do_Masking
            input_no_prng_with_frontend(data.data, tmp.data, size, 3);
#else
            input_layer(data.data, tmp.data, size, 3);
#endif
        }
    }

    void initializeInferencePartyA(LayerGraphNode<T> *root) {
         topologicalApply(root, [&](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            auto layer = node->layer;
            auto weights = layer->getweights();
            auto bias = layer->getbias();
            if(LlamaConfig::party == 1){
                input_layer(nullptr, weights.data, weights.size, 2);
                if (layer->useBias) {
                    input_layer(nullptr, bias.data, bias.size, 2);
                }
            }
            else{
                Tensor1D<T> tmp(weights.size);
                input_layer(weights.data, tmp.data, weights.size, 2);
                if(layer->useBias){
                    Tensor1D<T> tmp2(bias.size);
                    input_layer(bias.data, tmp2.data, bias.size, 2);
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
        outputA(a.data, a.d1);
    }

    void output(Tensor1D<T> &a) {
        output(a.data, a.d1);
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
        std::cerr << ">> SS2M - Start" << "\n";
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
        std::cerr << ">> SS2M - End" << "\n";
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

    void conv3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 dd, u64 dh, u64 dw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
    {
        assert(input.d5 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fd * fh * fw * ci);
        always_assert(dd == 1);
        always_assert(dh == 1);
        always_assert(dw == 1);
        u64 newD = (((input.d2 + 2*pd - fd - (fd-1)*(dd-1))/sd) + 1);
        u64 newH = (((input.d3 + 2*ph - fh - (fh-1)*(dh-1))/sh) + 1);
        u64 newW = (((input.d4 + 2*pw - fw - (fw-1)*(dw-1))/sw) + 1);
        assert(output.d1 == input.d1);
        assert(output.d2 == newD);
        assert(output.d3 == newH);
        assert(output.d4 == newW);
        assert(output.d5 == co);

        Conv3DWrapper(input.d1, input.d2, input.d3, input.d4, input.d5, fd, fh, fw, co, 
            pd, pd, ph, ph, pw, pw, sd, sh, sw,
            input.data, filter.data, output.data);
    }

    void convTranspose3D(u64 fd, u64 fh, u64 fw, u64 pd, u64 ph, u64 pw, u64 sd, u64 sh, u64 sw, u64 ci, u64 co, const Tensor5D<T> &input, const Tensor2D<T> &filter, Tensor5D<T> &output)
    {
        assert(input.d5 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fd * fh * fw * ci);
        u64 newD = (((input.d2 - 1)*sd + fd - 2*pd));
        u64 newH = (((input.d3 - 1)*sh + fh - 2*ph));
        u64 newW = (((input.d4 - 1)*sw + fw - 2*pw));
        assert(output.d1 == input.d1);
        assert(output.d2 == newD);
        assert(output.d3 == newH);
        assert(output.d4 == newW);
        assert(output.d5 == co);

        ConvTranspose3DWrapper(input.d1, input.d2, input.d3, input.d4, input.d5, fd, fh, fw, co, 
            pd, pd, ph, ph, pw, pw, sd, sh, sw, 
            output.d2, output.d3, output.d4, input.data, filter.data, output.data);
    }

    void convTranspose2D(u64 fh, u64 fw, u64 ph, u64 pw, u64 sh, u64 sw, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output)
    {
        assert(input.d4 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fh * fw * ci);
        u64 newH = (((input.d2 - 1) * sh + fh - 2 * ph));
        u64 newW = (((input.d3 - 1) * sw + fw - 2 * pw));
        assert(output.d1 == input.d1);
        assert(output.d2 == newH);
        assert(output.d3 == newW);
        assert(output.d4 == co);

        ConvTranspose2DWrapper(input.d1, input.d2, input.d3, input.d4, fh, fw, co,
                               ph, ph, pw, pw, sh, sw, output.d2, output.d3, input.data, filter.data, output.data);
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

    void batchNormInference(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        assert(A.d1 == B.d1);
        assert(A.d1 == x.shape.back());
        assert(x.is_same_shape(y));
        u64 channels = x.shape.back();
        // replicate A
        Tensor<T> A2(x.shape);

        for (u64 i = 0; i < x.size(); ++i)
        {
            A2.data[i] = A.data[i % channels];
        }

        ElemWiseSecretSharedVectorMult(x.size(), x.data, x.data, A2.data, A2.data, y.data, y.data);

        for (u64 i = 0; i < x.size(); ++i)
        {
            y.data[i] += B.data[i % channels];
        }

    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out) {
        auto ct = new ClearText<T>;
        ct->add(in, out);
        delete ct;
    }

    void addbias(Tensor<T> &x, const Tensor1D<T> &bias) {
        auto ct = new ClearText<T>;
        ct->addbias(x, bias);
        delete ct;
    }

    void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y) {
        auto ct = new ClearText<T>;
        ct->scalarmul(x, scalar, y);
        delete ct;
    }
};