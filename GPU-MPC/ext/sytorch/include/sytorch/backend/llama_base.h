// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
//
// Copyright (c) 2024 Microsoft Research
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <sytorch/utils.h>
#include <llama/config.h>
#include <llama/input_prng.h>
#include <llama/comms.h>
#include <llama/api.h>
#include "backend.h"
#include <sytorch/layers/layers.h>
#include <omp.h>
#include <filesystem>

char *readFile(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);
    size_t fileSz = std::filesystem::file_size(filename);
    char *mem_bytes = (char *)malloc(fileSz);
    file.read((char *)mem_bytes, fileSz);
    file.close();
    return mem_bytes;
}

template <typename T>
class LlamaBase : public Backend<T>
{
public:
    const bool useLocalTruncation = false;
    char *_kBuf;

    void initPrngs()
    {
        u64 seedKey = 0xdeadbeefbadc0ffe;
        for (int i = 0; i < 256; ++i)
        {
            LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(i, seedKey));
        }
    }

    void initDealer(char **sBuf, char **cBuf)
    {
        assert(LlamaConfig::party == 1);
        initPrngs();
        LlamaConfig::server = new Peer(sBuf);
        LlamaConfig::client = new Peer(cBuf);
        // input_prng_init();
    }

    void initServer(std::string ip, char **kBuf)
    {
        assert(LlamaConfig::party == 2);
        initPrngs();
        LlamaConfig::dealer = new Dealer(kBuf);
        LlamaConfig::client = waitForPeer(42002);
        LlamaConfig::peer = LlamaConfig::client;
        // input_prng_init();
    }

    void initClient(std::string ip, char **kBuf)
    {
        assert(LlamaConfig::party == 3);
        initPrngs();
        LlamaConfig::dealer = new Dealer(kBuf);
        LlamaConfig::server = new Peer(ip, 42002);
        LlamaConfig::peer = LlamaConfig::server;
        // input_prng_init();
    }

    void init(std::string ip, bool memBuf = false)
    {
        if (LlamaConfig::party == 1)
        {
            initPrngs();
            LlamaConfig::server = new Peer("server.dat");
            LlamaConfig::client = new Peer("client.dat");
        }
        else if (LlamaConfig::party == 2)
        {
            if (memBuf)
            {
                _kBuf = readFile("server.dat");
                initServer(ip, &_kBuf);
            }
            else
            {
                initPrngs();
                LlamaConfig::dealer = new Dealer("server.dat");
                LlamaConfig::client = waitForPeer(42005);
                LlamaConfig::peer = LlamaConfig::client;
            }
        }
        else if (LlamaConfig::party == 3)
        {
            if (memBuf)
            {
                _kBuf = readFile("client.dat");
                initClient(ip, &_kBuf);
            }
            else
            {
                initPrngs();
                LlamaConfig::dealer = new Dealer("client.dat");
                LlamaConfig::server = new Peer(ip, 42005);
                LlamaConfig::peer = LlamaConfig::server;
            }
        }
        else
        {
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

    void initializeInferencePartyB(Tensor<T> &data)
    {
        u64 size = data.size();
        if (LlamaConfig::party == 1)
        {
            input_layer(nullptr, data.data, size, 3);
        }
        else
        {
            Tensor<T> tmp(data.shape);
            input_layer(data.data, tmp.data, size, 3);
        }
    }

    void initializeInferencePartyA(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *node, LayerGraphNode<T> *_root)
                         {
                             auto layer = node->layer;
                             if (layer->name == "_MHADummy")
                             {
                                 auto mha = (_MHADummy<T> *)layer;
                                 if (LlamaConfig::party == 1)
                                 {
                                     // populate with randomness
                                     input_layer(nullptr, mha->wQKV.data, mha->wQKV.size(), 2);
                                     input_layer(nullptr, mha->bQKV.data, mha->bQKV.size(), 2);
                                     input_layer(nullptr, mha->wProj.data, mha->wProj.size(), 2);
                                     input_layer(nullptr, mha->bProj.data, mha->bProj.size(), 2);
                                 }
                                 else
                                 {
                                     Tensor1D<T> tmp1(mha->wQKV.size());
                                     printf("Weight=%ld\n", mha->wQKV.data[0]);
                                     input_layer(mha->wQKV.data, tmp1.data, mha->wQKV.size(), 2);
                                     printf("Masked weight=%ld\n", mha->wQKV.data[0]);
                                     Tensor1D<T> tmp2(mha->bQKV.size());
                                     input_layer(mha->bQKV.data, tmp2.data, mha->bQKV.size(), 2);
                                     Tensor1D<T> tmp3(mha->wProj.size());
                                     input_layer(mha->wProj.data, tmp3.data, mha->wProj.size(), 2);
                                     Tensor1D<T> tmp4(mha->bProj.size());
                                     input_layer(mha->bProj.data, tmp4.data, mha->bProj.size(), 2);
                                 }
                             }
                             else
                             {
                                 auto weights = layer->getweights();
                                 auto bias = layer->getbias();
                                 if (LlamaConfig::party == 1)
                                 {
                                     input_layer(nullptr, weights.data, weights.size, 2);
                                     if (layer->useBias)
                                     {
                                         input_layer(nullptr, bias.data, bias.size, 2);
                                     }
                                 }
                                 else
                                 {
                                     Tensor1D<T> tmp(weights.size);
                                     input_layer(weights.data, tmp.data, weights.size, 2);
                                     if (layer->useBias)
                                     {
                                         Tensor1D<T> tmp2(bias.size);
                                         input_layer(bias.data, tmp2.data, bias.size, 2);
                                     }
                                 }
                             } });
    }

    void inputA(Tensor<T> &data)
    {
        u64 b1 = data.size();
        if (LlamaConfig::party == 1)
        {
            input_layer(nullptr, data.data, b1, 2);
        }
        else
        {
            if (LlamaConfig::party == 2)
            {
                Tensor<T> tmp(data.shape);
                input_layer(data.data, tmp.data, b1, 2);
            }
            else
            {
                input_layer(data.data, nullptr, b1, 2);
            }
        }
    }

    void outputA(Tensor<T> &a)
    {
        outputA(a.data, a.size());
    }

    void output(Tensor<T> &a)
    {
        output(a.data, a.size());
    }

    void outputA(Tensor2D<T> &a)
    {
        outputA(a.data, a.d1 * a.d2);
    }

    void output(Tensor2D<T> &a)
    {
        output(a.data, a.d1 * a.d2);
    }

    void outputA(Tensor1D<T> &a)
    {
        outputA(a.data, a.d1);
    }

    void output(Tensor1D<T> &a)
    {
        output(a.data, a.d1);
    }

    void outputA(T *a, u64 sz)
    {
        if (LlamaConfig::party == 1)
        {
            for (int i = 0; i < sz; i++)
            {
                LlamaConfig::client->send_mask(a[i]);
                a[i] = 0;
            }
        }
        else if (LlamaConfig::party == 3)
        {
            for (int i = 0; i < sz; i++)
            {
                auto mask = LlamaConfig::dealer->recv_mask();
                a[i] = a[i] - mask;
                mod(a[i], LlamaConfig::bitlength);
                a[i] -= ((a[i] >> (LlamaConfig::bitlength - 1) << LlamaConfig::bitlength));
            }
        }
    }

    void output(T *a, u64 sz)
    {
        if (LlamaConfig::party == 1)
        {
            for (int i = 0; i < sz; i++)
            {
                LlamaConfig::client->send_mask(a[i]);
                LlamaConfig::server->send_mask(a[i]);
                a[i] = 0;
            }
        }
        else
        {
            for (int i = 0; i < sz; i++)
            {
                auto mask = LlamaConfig::dealer->recv_mask();
                a[i] = a[i] - mask;
            }
        }
    }

    void ss2m(T *data, u64 size)
    {
        std::cerr << ">> SS2M - Start" << std::endl;
        if (LlamaConfig::party == 1)
        {
            for (int i = 0; i < size; i++)
            {
                data[i] = random_ge(64);
                auto p = splitShare(data[i], 64);
                LlamaConfig::client->send_mask(p.first);
                LlamaConfig::server->send_mask(p.second);
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                auto mask = LlamaConfig::dealer->recv_mask();
                data[i] = data[i] + mask;
            }
            reconstruct(size, data, 64);
        }
        std::cerr << ">> SS2M - End" << std::endl;
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        assert(a.d2 == b.d1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        MatMul2D(a.d1, a.d2, b.d2, a.data, a.data, b.data, b.data, c.data, c.data, true);
    }

    void matmul_triangular(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        assert(a.d2 == b.d1);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d2);
        MatMul2DTriangular(a.d1, a.d2, b.d2, a.data, a.data, b.data, b.data, c.data, c.data, true);
    }

    void matmulTransposeA(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        assert(a.d1 == b.d1);
        assert(c.d1 == a.d2);
        assert(c.d2 == b.d2);

        Tensor2D<T> aTranspose(a.d2, a.d1);
        for (int i = 0; i < a.d1; ++i)
            for (int j = 0; j < a.d2; ++j)
                aTranspose(j, i) = a(i, j);
        MatMul2D(aTranspose.d1, aTranspose.d2, b.d2, aTranspose.data, aTranspose.data, b.data, b.data, c.data, c.data, true);
    }

    void matmulTransposeB(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        assert(a.d2 == b.d2);
        assert(c.d1 == a.d1);
        assert(c.d2 == b.d1);
        Tensor2D<T> bTranspose(b.d2, b.d1);
        for (int i = 0; i < b.d1; ++i)
            for (int j = 0; j < b.d2; ++j)
                bTranspose(j, i) = b(i, j);
        matmul(a, bTranspose, c);
    }

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst)
    {
        assert(input.d4 == ci);
        assert(filter.d1 == co);
        assert(filter.d2 == fh * fw * ci);
        u64 newH = (((input.d2 + 2 * padding - fh) / stride) + 1);
        u64 newW = (((input.d3 + 2 * padding - fw) / stride) + 1);
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
        u64 newD = (((input.d2 + 2 * pd - fd - (fd - 1) * (dd - 1)) / sd) + 1);
        u64 newH = (((input.d3 + 2 * ph - fh - (fh - 1) * (dh - 1)) / sh) + 1);
        u64 newW = (((input.d4 + 2 * pw - fw - (fw - 1) * (dw - 1)) / sw) + 1);
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
        u64 newD = (((input.d2 - 1) * sd + fd - 2 * pd));
        u64 newH = (((input.d3 - 1) * sh + fh - 2 * ph));
        u64 newW = (((input.d4 - 1) * sw + fw - 2 * pw));
        assert(output.d1 == input.d1);
        assert(output.d2 == newD);
        assert(output.d3 == newH);
        assert(output.d4 == newW);
        assert(output.d5 == co);

        ConvTranspose3DWrapper(input.d1, input.d2, input.d3, input.d4, input.d5, fd, fh, fw, co,
                               pd, pd, ph, ph, pw, pw, sd, sh, sw,
                               output.d2, output.d3, output.d4, input.data, filter.data, output.data);
    }

    void sumPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out)
    {
        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        u64 newH = (in.d2 + 2 * padding - ks) / stride + 1;
        u64 newW = (in.d3 + 2 * padding - ks) / stride + 1;
        assert(out.d2 == newH);
        assert(out.d3 == newW);

#pragma omp parallel for collapse(4)
        for (int i = 0; i < in.d1; i++)
        {
            for (int j = 0; j < newH; j++)
            {
                for (int k = 0; k < newW; k++)
                {
                    for (int l = 0; l < in.d4; l++)
                    {
                        T sum = 0;
                        for (int m = 0; m < ks; m++)
                        {
                            for (int n = 0; n < ks; n++)
                            {
                                sum += in(i, j * stride + m, k * stride + n, l);
                            }
                        }
                        out(i, j, k, l) = sum;
                    }
                }
            }
        }
    }

    void div(const Tensor<T> &in, T divisor, u64 scale)
    {
        if (!(divisor & (divisor - 1)))
        {
            Backend<T>::truncate(in, log2(divisor), 3);
        }
        else
        {
            T divfp = (1LL << scale) / divisor;
            u64 sz = in.size();
            for (u64 i = 0; i < sz; i++)
            {
                in.data[i] *= divfp;
            }
            Backend<T>::truncate(in, scale, 3);
        }
    }

    void divPartial(const Tensor4D<T> &in, T divisor, u64 scale)
    {
        T divfp = (1LL << scale) / divisor;
        u64 sz = in.d1 * in.d2 * in.d3 * in.d4;
#pragma omp parallel for
        for (u64 i = 0; i < sz; i++)
        {
            in.data[i] *= divfp;
        }
    }

    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale)
    {
        sumPool2D(ks, padding, stride, in, out);
        divPartial(out, (T)(ks * ks), scale);
    }

    u64 log2(u64 x)
    {
        u64 y = 0;
        while (x >>= 1)
            y++;
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

        ElemWiseMul(x.size(), x.data, A2.data, y.data);

        for (u64 i = 0; i < x.size(); ++i)
        {
            y.data[i] += B.data[i % channels];
        }
    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
    {
        always_assert(in.size() > 0);
        always_assert(out.size() == in[0]->size());
        for (int i = 0; i < in.size(); i++)
        {
            always_assert(out.size() == in[i]->size());
        }

#pragma omp parallel for
        for (u64 i = 0; i < out.size(); ++i)
        {
            T sum = 0;
            for (int j = 0; j < in.size(); j++)
            {
                sum += in[j]->data[i];
            }
            out.data[i] = sum;
        }
    }

    void addbias(Tensor<T> &x, const Tensor1D<T> &bias)
    {
        always_assert(x.shape.back() == bias.d1);

#pragma omp parallel for
        for (u64 i = 0; i < x.size(); ++i)
        {
            x.data[i] += bias(i % bias.d1);
        }
    }

    void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y)
    {
        always_assert(x.is_same_shape(y));

#pragma omp parallel for
        for (u64 i = 0; i < x.size(); ++i)
        {
            y.data[i] = x.data[i] * scalar;
        }
    }
};