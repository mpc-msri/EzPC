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

#include "utils/gpu_random.h"
#include "utils/gpu_mem.h"

#include "orca_base.h"

#include "fss/dcf/gpu_relu.h"
#include "fss/dcf/gpu_truncate.h"
#include "fss/dcf/gpu_maxpool.h"
// pin all the weights and activations in cpu memory

template <typename T>
class Orca : public OrcaBase<T>
{
public:
    Orca() : OrcaBase<T>() {}

    Orca(int party, std::string ip, int bw, int scale, std::string keyFile = "") : OrcaBase<T>(party, ip, bw, scale, keyFile, false)
    {
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode)
    {
        if (mode == 2)
        {
            // auto h_inp = (T*) moveToCPU((u8*) in.d_data, in.size() * sizeof(T), NULL);
            // printf("Relu input=%ld, %ld, %ld\n", h_inp[0], h_inp[1], h_inp[2]);

            auto start = std::chrono::high_resolution_clock::now();

            auto k = dcf::readGPUReluExtendKey<T>(&(this->keyBuf));
            auto d_temp = dcf::gpuReluExtend(this->peer, this->party, k, in.d_data, &(this->g), &(this->s));
            auto d_drelu = d_temp.first;
            gpuFree(d_drelu);
            out.d_data = d_temp.second;
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            this->s.reluext_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

            // printf("Num relus=%d, %lx, %lu\n", out.size(), in.d_data, out.size() * sizeof(T));
            // auto h_data = (T*) moveToCPU((u8*) out.d_data, out.size() * sizeof(T), NULL);
            // printf("Relu output=%lu, %lu, %ld\n", h_data[0], h_data[1], h_data[2]);
        }
        else
        {
            auto start = std::chrono::high_resolution_clock::now();

            auto k = dcf::readTwoRoundReluKey<T>(&(this->keyBuf));
            auto d_temp = dcf::gpuTwoRoundRelu(this->peer, this->party, k, in.d_data, &(this->g), &(this->s));
            auto d_drelu = d_temp.first;
            gpuFree(d_drelu);
            out.d_data = d_temp.second;
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            this->s.relu_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        }
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        // printf("Truncate=%lu, %lu, %lu\n", mode, shift, size);
        auto start = std::chrono::high_resolution_clock::now();
        if (mode == 0)
        {
            auto k = dcf::readGPUTrStochasticKey<T>(&(this->keyBuf));
            dcf::gpuStochasticTruncate(k, this->party, this->peer, in.d_data, &(this->g), &(this->s));
        }
        else if (mode == 1)
        {
            auto k = dcf::readGPUStTRKey<T>(&(this->keyBuf));
            dcf::gpuStochasticTR(k, this->party, this->peer, in.d_data, &(this->g), &(this->s));
        }
        else
        {
            assert(0);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        this->s.truncate_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void signext(Tensor<T> &x, u64 scale)
    {
        // printf("Sign ext=%lu\n", x.size());
        auto start = std::chrono::high_resolution_clock::now();
        auto k = dcf::readGPUSignExtendKey<T>(&(this->keyBuf));
        dcf::gpuSignExtend(k, this->party, this->peer, x.d_data, &(this->g), &(this->s));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        this->s.signext_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode)
    {
        auto start = std::chrono::high_resolution_clock::now();

        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        int tmpBw = this->bw;
        // Neha: ugly hack
        if (mode == 3)
            tmpBw -= scale;
        MaxpoolParams p = {
            tmpBw, tmpBw, 0, 0, this->bw,
            (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks,
            (int)stride, (int)stride,
            (int)padding, (int)padding,
            (int)padding, (int)padding,
            0, 0, false};
        initPoolParams(p);
        auto k = dcf::readGPUMaxpoolKey<T>(p, &(this->keyBuf));
        out.d_data = dcf::gpuMaxPool(this->peer, this->party, p, k, in.d_data, (u32 *)NULL, &(this->g), &(this->s));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        this->s.maxpool_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }
};

template <typename T>
class OrcaKeygen : public OrcaBaseKeygen<T>
{
public:
    OrcaKeygen(int party, int bw, int scale, std::string keyFile) : OrcaBaseKeygen<T>(party, bw, scale, keyFile)
    {
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode)
    {
        assert(in.is_same_shape(out));
        assert(in.is_same_shape(drelu));
        // printf("Keygen relu mode=%d\n", mode);
        if (mode == 2)
        {
            // auto h_inp = (T*) moveToCPU((u8*) in.d_data, in.size() * sizeof(T), NULL);
            // printf("Relu inp mask=%ld, %ld\n", h_inp[0], h_inp[1]);
            // printf("Addr=%lx\n", in.d_data);
            auto d_tempMask = dcf::gpuKeygenReluExtend<T>(&(this->keyBuf), this->party, this->bw - scale, this->bw, in.size(), in.d_data, &(this->g));
            auto d_dreluMask = d_tempMask.first;
            gpuFree(d_dreluMask);
            auto d_reluMask = d_tempMask.second;
            out.d_data = d_reluMask;
            // auto h_out = (T*) moveToCPU((u8*) out.d_data, in.size() * sizeof(T), NULL);
            // printf("Relu out mask=%ld, %ld\n", h_out[0], h_out[1]);
        }
        else
        {
            int tmpBw = this->bw;
            if (mode == 3)
                tmpBw -= scale;
            auto d_tempMask = dcf::gpuGenTwoRoundReluKey(&(this->keyBuf), this->party, tmpBw, tmpBw, in.size(), in.d_data, &(this->g));
            auto d_dreluMask = d_tempMask.first;
            gpuFree(d_dreluMask);
            auto d_reluMask = d_tempMask.second;
            out.d_data = d_reluMask;
        }
        // printf("Done keygen relu\n");
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        if (mode == 0)
        {
            in.d_data = dcf::genGPUStochasticTruncateKey(&(this->keyBuf), this->party, this->bw, this->bw, shift, in.size(), in.d_data, &(this->g));
        }
        else if (mode == 1)
        {
            in.d_data = dcf::genGPUStTRKey(&(this->keyBuf), this->party, this->bw, this->bw - shift, shift, in.size(), in.d_data, &(this->g));
        }
        else
        {
            assert(0);
        }
    }

    void
    signext(Tensor<T> &x, u64 scale)
    {
        // printf("Signext inp mask %lx\n", x.d_data);

        int bin = this->bw - scale;
        int bout = this->bw;
        x.d_data = dcf::genSignExtendKey(&(this->keyBuf), this->party, bin, bout, x.size(), x.d_data, &(this->g));

        // auto h_mask = (T*) moveToCPU((u8*) x.d_data, x.size() * sizeof(T), NULL);
        // printf("Signext out mask %lx=%ld, %ld\n", x.d_data, h_mask[0], h_mask[1]);
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode)
    {
        int tmpBw = this->bw;
        // Neha: ugly hack
        if (mode == 3)
            tmpBw -= scale;
        MaxpoolParams p = {
            tmpBw, tmpBw, 0, 0, this->bw,
            (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks,
            (int)stride, (int)stride,
            (int)padding, (int)padding,
            (int)padding, (int)padding,
            0, 0, false};
        initPoolParams(p);
        out.d_data = dcf::gpuKeygenMaxpool(&(this->keyBuf), this->party, p, in.d_data, (u8 *)NULL, &(this->g));
        // printf("done with keygen maxpool=%lx\n", out.d_data);
    }
};

template <typename T>
class OrcaDummy : public Orca<T>
{
public:
    OrcaDummy()
    {
    }
};