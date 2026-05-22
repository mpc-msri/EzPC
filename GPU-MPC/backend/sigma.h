// Author: Neha Jawalkar
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

#include <omp.h>

#include <sytorch/backend/backend.h>
#include <sytorch/backend/llama_transformer.h>
#include <llama/comms.h>
#include <llama/api.h>

#include "nn/orca_opt.h"

#include "utils/gpu_random.h"
#include "utils/gpu_mem.h"

#include "fss/gpu_matmul.h"
#include "fss/gpu_gelu.h"
#include "fss/gpu_layernorm.h"
#include "fss/gpu_mha.h"
#include "fss/gpu_add.h"

template <typename T>
void noTruncateAfterRmsnorm(LayerGraphNode<T> *n, LayerGraphNode<T> *r)
{
    if (n->layer->name == "RMSNorm")
    {
        n->layer->doTruncationForward = false;
    }
}

template <typename T>
class SIGMA : public Backend<T>
{
public:
    u8 *startPtr = NULL;
    u8 *keyBuf = NULL;
    size_t keySize = 0;
    // int fd = -1;
    GpuPeer *peer = NULL;
    int party = -1;
    Stats s;
    int bw = 0, scale = 0, n_seq = 0;
    AESGlobalContext g;
    MHATables<T> d_mhaTab;
    T *d_geluTab, *d_siluTab;
    std::vector<GroupElement> *invSqrtTab;
    LlamaTransformer<T> *llama;

    SIGMA(int party, std::string ip, std::string keyFile, int bw, int scale, int n_seq, int n_embed, int numThreads, bool gpuMemPool = true) : party(party), bw(bw), scale(scale), n_seq(n_seq)
    {
        initAESContext(&g);
        if (gpuMemPool)
            initGPUMemPool();
        // initCommBufs(true);

        d_geluTab = genLUT<T, reluSubGelu<T>>(8, 6, scale);
        d_siluTab = genLUT<T, reluSubSilu<T>>(10, 6, scale);
        d_mhaTab = initMHATables<T>(n_seq, scale);

        omp_set_num_threads(numThreads);

        invSqrtTab = new std::vector<GroupElement>(1LL << 13);
#pragma omp parallel for
        for (int i = 0; i < (1LL << 13); ++i)
        {
            GroupElement k = i % (1LL << 6);
            GroupElement m = i >> 6;
            double val = double(m + 128) * std::pow(2.0, k - 7);
            (*invSqrtTab)[i] = GroupElement(double(1LL << (2 * scale)) / sqrt(val / n_embed));
        }
        if (keyFile.compare("") != 0)
        {
            auto filename = keyFile + "_" + std::to_string(party) + ".dat";
            keySize = std::filesystem::file_size(filename);
            int fd = openForReading(filename);
            printf("%s, %d\n", filename.data(), fd);
            getAlignedBuf(&keyBuf, keySize);
            readKey(fd, keySize, keyBuf, NULL);
            startPtr = keyBuf;
            closeFile(fd);
        }

        LlamaConfig::bitlength = bw;
        LlamaConfig::party = party + 2;
        LlamaConfig::stochasticT = false;
        LlamaConfig::stochasticRT = false;

        llama = new LlamaTransformer<T>();
        if (party == SERVER0)
            llama->initServer(ip, (char **)&keyBuf);
        else
            llama->initClient(ip, (char **)&keyBuf);

        peer = new GpuPeer(true);
        peer->peer = LlamaConfig::peer;
    }

    void close()
    {
        peer->close();
        // printf("Key read=%lu\n", keyBuf - startPtr);
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c, bool useBias, Tensor1D<T> &d, bool isFirst)
    {
        auto start = std::chrono::high_resolution_clock::now();

        MatmulParams p;
        p.M = a.d1;
        p.K = a.d2;
        p.N = b.d2;
        p.batchSz = 1;
        stdInit(p, bw, 0);

        auto k = readGPUMatmulKey<T>(p, TruncateType::None, &keyBuf);
        c.d_data = gpuMatmul(peer, party, p, k, a.d_data, b.data, useBias ? d.data : (T *)NULL, TruncateType::None, &g, &s, false);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.matmul_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void gelu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        u64 b0 = peer->bytesSent() + peer->bytesReceived();
        auto start = std::chrono::high_resolution_clock::now();

        auto k = readGpuGeluKey<T, u8>(&keyBuf);
        out.d_data = gpuGelu<T, u8, 8>(peer, party, k, bw, bw - scale, (int)scale, in.size(), in.d_data, d_geluTab, &g, &s);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.gelu_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        u64 b1 = peer->bytesSent() + peer->bytesReceived();
        s.gelu_comm_bytes += (b1 - b0);
    }

    void silu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        u64 b0 = peer->bytesSent() + peer->bytesReceived();
        auto start = std::chrono::high_resolution_clock::now();

        auto k = readGpuGeluKey<T, u16>(&keyBuf);
        out.d_data = gpuGelu<T, u16, 10>(peer, party, k, bw, bw - scale, (int)scale, in.size(), in.d_data, d_siluTab, &g, &s);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.gelu_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        u64 b1 = peer->bytesSent() + peer->bytesReceived();
        s.gelu_comm_bytes += (b1 - b0);
    }

    void SIGMALayernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale, bool computeMu)
    {
        u64 b0 = peer->bytesSent() + peer->bytesReceived();
        auto start = std::chrono::high_resolution_clock::now();

        AvgPoolParams p = {bw, bw, scale, 0, 0, 1, x.shape[0], x.shape[1], 1, 1, x.shape[1], 1, x.shape[1], 0, 0, 0, 0};
        initPoolParams(p);
        auto k = readGPULayerNormKey<T>(p, &keyBuf, computeMu);
        // assert(d_invSqrtTab);
        auto d_A = (T *)moveToGPU((u8 *)A.data, A.size() * sizeof(T), &s);
        auto d_B = (T *)moveToGPU((u8 *)B.data, B.size() * sizeof(T), &s);
        y.d_data = gpuLayerNorm(peer, party, p, k, d_A, d_B, x.d_data, /*(std::vector<GroupElement> *)*/ invSqrtTab, &g, &s, computeMu);
        gpuFree(d_A);
        gpuFree(d_B);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.layernorm_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        u64 b1 = peer->bytesSent() + peer->bytesReceived();
        s.layernorm_comm_bytes += (b1 - b0);
    }

    void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        SIGMALayernorm(A, B, x, y, scale, true);
    }

    void rmsnorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        SIGMALayernorm(A, B, x, y, scale, false);
    }

    void mha(int n_heads, int n_embed, int dim_W, bool selfAttn, bool doNormQKt, bool doRotEmb, const Tensor2D<T> &wQKV, const Tensor1D<T> &bQKV, const Tensor2D<T> &wProj, const Tensor1D<T> &bProj, const Tensor2D<T> &X, Tensor2D<T> &Y)
    {
        auto start = std::chrono::high_resolution_clock::now();

        MHAParams pMHA = {n_seq, n_embed, n_heads, dim_W, selfAttn, doNormQKt, doRotEmb};
        MHAMulParams pMHAMul = initMHAMulParams(pMHA, bw, scale);
        auto k = readGPUMHAKey<T>(pMHA, pMHAMul, &keyBuf);
        Y.d_data = gpuMHA(peer, party, bw, scale, pMHA, pMHAMul, k, wQKV.data, bQKV.data, wProj.data, bProj.data, X.d_data, d_mhaTab, &g, &s);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.mha_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        auto start = std::chrono::high_resolution_clock::now();

        TruncateType t = TruncateType::TrFloor;
        auto k = readGPUTruncateKey<T>(t, &keyBuf);
        in.d_data = gpuTruncate<T, T>(k.bin, k.bout, t, k, k.shift, peer, party, k.N, in.d_data, &g, &s);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.truncate_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
    {
        u64 N = a.size();
        auto k = readGPUMulKey<T>(&keyBuf, N, N, N, TruncateType::None);
        out.d_data = gpuMul(peer, party, bw, scale, N, k, a.d_data, b.d_data, TruncateType::None, &g, &s);
    }

    void output(Tensor<T> &a)
    {
        int N = a.size();
        unmaskValues(bw, N, a.d_data, (T *)keyBuf, &s);
        moveIntoCPUMem((u8 *)a.data, (u8 *)a.d_data, N * sizeof(T), &s);
    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
    {
        int N = in[0]->size();
        std::vector<T *> gpuInp;
        for (int i = 0; i < in.size(); i++)
        {
            gpuInp.push_back(in[i]->d_data);
        }
        out.d_data = gpuAdd(bw, N, gpuInp);
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { noTruncateAfterRmsnorm(n, r); });
    }
};

template <typename T>
class SIGMAKeygen : public Backend<T>
{
public:
    u8 *startPtr;
    u8 *keyBuf = NULL;
    size_t keyBufSize = 0;
    int party = -1;
    std::string keyFile;
    size_t keySize = 0;
    int scale;
    int bw;
    AESGlobalContext g;
    LlamaTransformer<T> *llama;
    u8 *llamaBuf1, *llamaBuf2;
    u8 *dummyBuf1, *dummyBuf2;

    SIGMAKeygen(int party, int bw, int scale, std::string keyFile, size_t keyBufSize) : party(party), bw(bw), scale(scale), keyFile(keyFile), keyBufSize(keyBufSize)
    {
        initAESContext(&g);
        initGPURandomness();
        initGPUMemPool();
        // keyBufSize = 20 * OneGB;
        keyBuf = cpuMalloc(keyBufSize);
        startPtr = keyBuf;

        LlamaConfig::bitlength = bw;
        LlamaConfig::party = DEALER;
        LlamaConfig::stochasticT = false;
        LlamaConfig::stochasticRT = false;

        llama = new LlamaTransformer<T>();
        llamaBuf1 = (u8 *)cpuMalloc(OneGB);
        dummyBuf1 = (u8 *)cpuMalloc(OneGB);
        llamaBuf2 = llamaBuf1;
        dummyBuf2 = dummyBuf1;
        llama->initDealer((char **)(party == SERVER0 ? &llamaBuf2 : &dummyBuf2), (char **)(party == SERVER1 ? &llamaBuf2 : &dummyBuf2));
    }

    void close()
    {
        /*size_t*/ keySize = keyBuf - startPtr;
        size_t padding = 4096 - (keySize % 4096);
        char *zeros = new char[padding];
        memset(zeros, 0, padding);
        memcpy(keyBuf, zeros, padding);
        keyBuf += padding;
        keySize += padding;
        assert(keySize < keyBufSize);
        if (keyFile.compare("") != 0)
        {
            std::ofstream f(keyFile + "_" + std::to_string(party) + ".dat");
            f.write((char *)startPtr, keySize);
            f.close();
            cpuFree(startPtr);
        }
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        MatmulParams p;
        p.M = a.d1;
        p.K = a.d2;
        p.N = b.d2;
        p.batchSz = 1;
        stdInit(p, bw, 0);
        c.d_data = gpuKeygenMatmul<T>(&keyBuf, party, p, a.d_data, b.data, (T *)NULL, TruncateType::None, &g, false);
    }

    void gelu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        out.d_data = gpuKeyGenGelu<T, u8, 8>(&keyBuf, party, bw, bw - scale, (int)scale, in.size(), in.d_data, &g);
    }

    void silu(const Tensor<T> &in, Tensor<T> &out, u64 scale, u64 mode = 0)
    {
        out.d_data = gpuKeyGenGelu<T, u16, 10>(&keyBuf, party, bw, bw - scale, (int)scale, in.size(), in.d_data, &g);
    }

    void SIGMALayernormKeygen(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale, bool computeMu)
    {
        AvgPoolParams p = {bw, bw, scale, 0, 0, 1, x.shape[0], x.shape[1], 1, 1, x.shape[1], 1, x.shape[1], 0, 0, 0, 0};
        initPoolParams(p);
        auto d_mask_A = (T *)moveToGPU((u8 *)A.data, A.size() * sizeof(T), (Stats *)NULL);
        auto d_mask_B = (T *)moveToGPU((u8 *)B.data, B.size() * sizeof(T), (Stats *)NULL);
        y.d_data = gpuKeygenLayerNorm(&keyBuf, party, p, d_mask_A, d_mask_B, x.d_data, &g, computeMu);
        size_t llamaKeySz = llamaBuf2 - llamaBuf1;
        memcpy(keyBuf, llamaBuf1, llamaKeySz);
        keyBuf += llamaKeySz;
        llamaBuf2 = llamaBuf1;
        gpuFree(d_mask_A);
        gpuFree(d_mask_B);
    }

    void layernorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        SIGMALayernormKeygen(A, B, x, y, scale, true);
    }

    void rmsnorm(const Tensor1D<T> &A, const Tensor1D<T> &B, const Tensor<T> &x, Tensor<T> &y, u64 scale)
    {
        SIGMALayernormKeygen(A, B, x, y, scale, false);
    }

    void mha(int n_heads, int n_embed, int dim_W, bool selfAttn, bool doNormQKt, bool doRotEmb, const Tensor2D<T> &wQKV, const Tensor1D<T> &bQKV, const Tensor2D<T> &wProj, const Tensor1D<T> &bProj, const Tensor2D<T> &X, Tensor2D<T> &Y)
    {
        MHAParams pMHA = {X.d1, n_embed, n_heads, dim_W, selfAttn, doNormQKt, doRotEmb};
        MHAMulParams pMHAMul = initMHAMulParams(pMHA, bw, scale);
        Y.d_data = gpuKeygenMHA(&keyBuf, party, bw, scale, pMHA, pMHAMul, wQKV.data, bQKV.data, wProj.data, bProj.data, X.d_data, &g);
    }

    void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
    {
        out.d_data = gpuKeygenMul(&keyBuf, party, bw, scale, a.size(), a.d_data, b.d_data, TruncateType::None, &g);
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        TruncateType t = TruncateType::TrFloor;
        in.d_data = genGPUTruncateKey<T, T>(&keyBuf, party, t, bw, bw, shift, in.size(), in.d_data, &g);
    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
    {
        int N = in[0]->size();
        std::vector<T *> gpuInp;
        for (int i = 0; i < in.size(); i++)
        {
            gpuInp.push_back(in[i]->d_data);
        }
        out.d_data = gpuAdd(bw, N, gpuInp);
    }

    void addbias(Tensor<T> &x, const Tensor1D<T> &bias)
    {
        gpuAddBias(1, x.size() / bias.d1, bias.d1, bw, x.d_data, bias.data, NULL);
    }

    void output(Tensor<T> &a)
    {
        int N = a.size();
        size_t memSz = N * sizeof(T);
        moveIntoCPUMem((u8 *)keyBuf, (u8 *)a.d_data, memSz, (Stats *)NULL);
        keyBuf += memSz;
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { noTruncateAfterRmsnorm(n, r); });
    }
};
