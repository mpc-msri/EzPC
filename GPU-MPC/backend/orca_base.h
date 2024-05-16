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
#include <sytorch/backend/llama_base.h>
#include <llama/comms.h>
#include <llama/api.h>

#include "nn/orca_opt.h"

#include "utils/gpu_random.h"
#include "utils/gpu_mem.h"

#include "fss/gpu_matmul.h"
#include "fss/gpu_conv2d.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_maxpool.h"
#include "fss/gpu_avgpool.h"
#include "fss/gpu_add.h"

template <typename T>
class OrcaBase : public Backend<T>
{
public:
    u8 *startPtr = NULL;
    u8 *keyBuf = NULL;
    size_t keySize = 0;
    int fd = -1;
    GpuPeer *peer = NULL;
    int party = -1;
    Stats s;
    int bw;
    int scale;
    AESGlobalContext g;

    OrcaBase() {}

    OrcaBase(int party, std::string ip, int bw, int scale, std::string keyFile = "", bool compress = true) : party(party), bw(bw), scale(scale)
    {
        initAESContext(&g);
        initGPUMemPool();
        // omp_set_num_threads(2);
        if (keyFile.compare("") != 0)
        {
            auto filename = keyFile + "_inference_key" + std::to_string(party) + ".dat";
            keySize = std::filesystem::file_size(filename);
            fd = openForReading(filename);
            // printf("%s, %d\n", filename.data(), fd);
            getAlignedBuf(&keyBuf, keySize);
            startPtr = keyBuf;
        }
        peer = new GpuPeer(compress);
        peer->connect(party, ip);
    }

    void close()
    {
        peer->close();
        // printf("Key read=%lu\n", keyBuf - startPtr);
    }

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, bool useBias, const Tensor1D<T> &bias, Tensor4D<T> &output, bool isFirst)
    {
        auto comm_start = s.comm_time;
        auto start = std::chrono::high_resolution_clock::now();
        GPUConv2DKey<T> k;
        k.p = {
            bw, bw, (int)input.d1, (int)input.d2, (int)input.d3, (int)ci,
            (int)fh, (int)fw, (int)co, (int)padding, (int)padding, (int)padding, (int)padding,
            (int)stride, (int)stride, 0, 0, 0, 0, 0};
        fillConv2DParams(&(k.p));
        k.mem_size_I = k.p.size_I * sizeof(T);
        k.mem_size_F = k.p.size_F * sizeof(T);
        k.mem_size_O = k.p.size_O * sizeof(T);

        k.I = (T *)keyBuf;
        keyBuf += k.mem_size_I;
        k.F = (T *)keyBuf;
        keyBuf += k.mem_size_F;
        k.O = (T *)keyBuf;
        keyBuf += k.mem_size_O;

        auto d_mask_I = (T *)moveToGPU((u8 *)k.I, k.mem_size_I, &s);
        if (isFirst)
        {
            gpuLinearComb(bw, k.p.size_I, input.d_data, T(1), input.d_data, T(1), d_mask_I);
            peer->reconstructInPlace(input.d_data, bw, k.p.size_I, &s);
        }
        // printf("Input=%lx\n", input.d_data);
        auto d_F = (T *)moveToGPU((u8 *)filter.data, k.mem_size_F, &s);
        // printf("filter=%lu\n", filter.data[k.p.size_F - 1]);
        auto d_mask_F = (T *)moveToGPU((u8 *)k.F, k.mem_size_F, &s);
        auto d_C = gpuConv2DBeaver<T>(k, party, input.d_data, d_F, d_mask_I, d_mask_F, useBias && party == SERVER0 ? bias.data : (T *)NULL, &s, 0);

        gpuFree(d_F);
        gpuFree(d_mask_I);
        gpuFree(d_mask_F);
        // printf("size O=%lu\n", k.p.size_O);
        peer->reconstructInPlace(d_C, k.p.bout, k.p.size_O, &s);
        output.d_data = d_C;

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.conv_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        auto comm_end = s.comm_time;
        s.conv_comm_time += (comm_end - comm_start);
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c, bool useBias, Tensor1D<T> &d, bool isFirst)
    {
        // auto h_data = (T*) moveToCPU((u8*) a.d_data, a.size() * sizeof(T), NULL);
        // printf("Matmul input=%ld, %ld\n", h_data[0], h_data[1]);
        // for(int i = 0; i < a.size(); i++) printf("Matmul input=%ld\n", h_data[i]);
        auto comm_start = s.comm_time;
        auto start = std::chrono::high_resolution_clock::now();

        MatmulParams p;
        p.M = a.d1;
        p.K = a.d2;
        p.N = b.d2;
        p.batchSz = 1;
        stdInit(p, bw, 0);
        auto k = readGPUMatmulKey<T>(p, TruncateType::None, &keyBuf);

        auto d_mask_A = (T *)moveToGPU((u8 *)k.A, k.mem_size_A, &s);
        if (isFirst)
        {
            gpuLinearComb(bw, p.size_A, a.d_data, T(1), a.d_data, T(1), d_mask_A);
            peer->reconstructInPlace(a.d_data, bw, p.size_A, &s);
        }
        c.d_data = gpuMatmul(peer, party, p, k, a.d_data, b.data, useBias ? d.data : (T *)NULL, TruncateType::None, &g, &s, false, d_mask_A);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.matmul_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        auto comm_end = s.comm_time;
        s.matmul_comm_time += (comm_end - comm_start);
    }

    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale)
    {
        AvgPoolParams p = {
            bw, bw, (int)scale, (int)scale, 0, (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks, (int)stride, (int)stride, (int)padding, (int)padding, (int)padding, (int)padding, 0, 0, false};
        initPoolParams(p);
        out.d_data = gpuAddPool(p, in.d_data, &s);
    }

    void output(Tensor<T> &a)
    {
        // int tmpBw = bw - scale;
        int N = a.size();
        unmaskValues(/*tmpBw*/ bw, N, a.d_data, (T *)keyBuf, &s);
        gpuLocalTr<T, T, ars>(party, bw, scale, N, a.d_data, true);
        moveIntoCPUMem((u8 *)a.data, (u8 *)a.d_data, N * sizeof(T), &s);
    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
    {
        int tmpBw = bw - scale;
        int N = in[0]->size();
        std::vector<T *> gpuInp;
        for (int i = 0; i < in.size(); i++)
        {
            gpuInp.push_back(in[i]->d_data);
        }
        out.d_data = gpuAdd(tmpBw, N, gpuInp);
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { orcaOpt<T>(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
    }
};

template <typename T>
class OrcaBaseKeygen : public Backend<T>
{
public:
    u8 *startPtr;
    u8 *keyBuf = NULL;
    size_t keyBufSize = 0;
    int party = -1;
    std::string keyFile;
    int scale;
    int bw;
    AESGlobalContext g;

    OrcaBaseKeygen(int party, int bw, int scale, std::string keyFile) : party(party), bw(bw), scale(scale), keyFile(keyFile)
    {
        initAESContext(&g);
        initGPURandomness();
        initCPURandomness();
        initGPUMemPool();
        keyBufSize = 20 * OneGB;
        getAlignedBuf(&keyBuf, keyBufSize, true);
        startPtr = keyBuf;
    }

    void close()
    {
        size_t keySize = keyBuf - startPtr;
        size_t padding = 4096 - (keySize % 4096);
        char *zeros = new char[padding];
        memset(zeros, 0, padding);
        memcpy(keyBuf, zeros, padding);
        keyBuf += padding;
        keySize += padding;
        assert(keySize < keyBufSize);
        int fd = openForWriting(keyFile + "_inference_key" + std::to_string(party) + ".dat");
        writeKeyBuf(fd, keySize, startPtr);
        assert(0 == fsync(fd) && "sync error!");
        closeFile(fd);
        cpuFree(startPtr, true);
        destroyGPURandomness();
        destroyCPURandomness();
    }

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst)
    {
        GPUConv2DKey<T> k;
        k.p = {
            bw, bw, (int)input.d1, (int)input.d2, (int)input.d3, (int)ci,
            (int)fh, (int)fw, (int)co, (int)padding, (int)padding, (int)padding, (int)padding,
            (int)stride, (int)stride, 0, 0, 0, 0, 0};
        fillConv2DParams(&(k.p));
        k.mem_size_I = k.p.size_I * sizeof(T);
        k.mem_size_F = k.p.size_F * sizeof(T);
        k.mem_size_O = k.p.size_O * sizeof(T);
        output.d_data = gpuKeygenConv2D<T>(&keyBuf, party, k, input.d_data, filter.data, true);
    }

    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        MatmulParams p;
        p.M = a.d1;
        p.K = a.d2;
        p.N = b.d2;
        p.batchSz = 1;
        stdInit(p, bw, 0);
        // printf("####### X=%lu\n", a.size());
        // auto h_temp = (u8*) moveToCPU((u8*) a.d_data, a.size() * sizeof(T), (Stats*) NULL);
        c.d_data = gpuKeygenMatmul<T>(&keyBuf, party, p, a.d_data, b.data, (T *)NULL, TruncateType::None, &g, false);
    }

    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale)
    {
        AvgPoolParams p = {
            bw, bw, (int)scale, (int)scale, 0, (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks, (int)stride, (int)stride, (int)padding, (int)padding, (int)padding, (int)padding, 0, 0, false};
        initPoolParams(p);
        out.d_data = gpuAddPool(p, in.d_data, (Stats *)NULL);
    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
    {
        int tmpBw = this->bw - this->scale;
        int N = in[0]->size();
        std::vector<T *> gpuInp;
        for (int i = 0; i < in.size(); i++)
        {
            gpuInp.push_back(in[i]->d_data);
        }
        out.d_data = gpuAdd(tmpBw, N, gpuInp);
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
                         { orcaOpt<T>(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
    }
};
