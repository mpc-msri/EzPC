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

#include "sigma_comms.h"
#include "gpu_data_types.h"
#include "misc_utils.h"

template <typename T>
__global__ void compressKernel(int bw, int modBw, u64 threads, int N, T *d_A, u8 *d_compressedA)
{
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < threads)
    {
        // this thread is responsible for packing 64 bits
        // figure out the index of the starting element
        u64 b = (64 * i) / bw;
        int elems = 63 / bw + 2;
        if (b + elems > N)
            elems = N - b;
        u64 temp = 0;
        int offset = (64 * i) % bw;
        gpuMod(d_A[b], modBw);
        temp = u64(d_A[b] >> offset);
        offset = bw - offset;
        for (int j = 1; j < elems; j++)
        {
            gpuMod(d_A[b + j], modBw);
            temp += (u64(d_A[b + j]) << offset);
            offset += bw;
            if (offset >= 64)
                break;
        }
        ((u64 *)d_compressedA)[i] = temp;
    }
}

template <typename T>
__global__ void expandKernel(int bw, int N, u8 *d_compressedA, T *d_A)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        assert(bw <= 64);
        u64 b = bw * (u64)i / 64;
        AESBlock temp;
        ((u64 *)&temp)[0] = ((u64 *)d_compressedA)[b];
        ((u64 *)&temp)[1] = ((u64 *)d_compressedA)[b + 1];
        int offset = (bw * (u64)i) % 64;
        auto elem = T(temp >> offset);
        gpuMod(elem, bw);
        d_A[i] = elem;
    }
}

__global__ void addMod4(int numInts, u32 *A, u32 *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < numInts)
    {
        u32 x = A[j];
        u32 y = B[j];
        u32 z = 0;
        for (int i = 0; i < 32; i += 2)
        {
            u32 a = (x >> i) & 3;
            u32 b = (y >> i) & 3;
            u32 c = ((a + b) & 3) << i;
            z |= c;
        }
        A[j] = z;
    }
}

class GpuPeer : public SigmaPeer
{
private:
    template <typename T>
    u8 *compressMem(int bw, int modBw, int N, T *d_A0, size_t &memSz, size_t &numInts, Stats *s, bool returnNew = false)
    {
        assert(modBw == bw);
        u8 *d_compressedA0;
        // printf("######## compressing=%d, bw=%d\n", this->compress, bw);
        if (this->compress && (bw > 2 /*|| compressBwLt3*/) && bw < 8 * sizeof(T))
        {
            // size in bytes
            // printf("^^^^^^^^^^^^^^ here\n");
            memSz = size_t(((u64)N * bw - 1) / 64 + 1) * 8;
            d_compressedA0 = (u8 *)gpuMalloc(memSz);
            u64 threads = memSz / 8; //(memSz - 1) / 8 + 1;
            // printf("%lu\n", threads);
            compressKernel<<<(threads - 1) / 128 + 1, 128>>>(bw, modBw, threads, N, d_A0, d_compressedA0);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else
        {
            // printf("Not compressing memory\n");
            assert(modBw == bw);
            memSz = 0;
            this->getMemSz<T>(bw, N, memSz, numInts);
            d_compressedA0 = (u8 *)d_A0;
            if (returnNew)
            {
                assert(memSz > 0);
                d_compressedA0 = (u8 *)gpuMalloc(memSz);
                checkCudaErrors(cudaMemcpy(d_compressedA0, d_A0, memSz, cudaMemcpyDeviceToDevice));
            }
        }
        return d_compressedA0;
    }

    template <typename T>
    T *expandMem(int bw, int N, u8 *h_compressedA, size_t memSz, size_t numInts, Stats *s)
    {
        T *d_A;
        if (this->compress && bw > 2 && bw < 8 * sizeof(T))
        {
            auto d_compressedA = (u8 *)moveToGPU(h_compressedA, memSz, s);
            // size in bytes
            memSz = size_t(N * sizeof(T));
            d_A = (T *)gpuMalloc(memSz);
            expandKernel<<<(N - 1) / 128 + 1, 128>>>(bw, N, d_compressedA, (T *)d_A);
            checkCudaErrors(cudaDeviceSynchronize());
            gpuFree(d_compressedA);
        }
        else
        {
            // don't need this here
            // getMemSz<T>(bw, N, memSz, numInts);
            // printf("Moving to gpu\n");
            d_A = (T *)moveToGPU((u8 *)h_compressedA, memSz, s);
        }
        return d_A;
    }

    void gpuAddMod4(u32 *d_A, u32 *d_B, int N)
    {
        const int thread_blk_size = 128;
        int numInts = (2 * N - 1) / 32 + 1;
        // printf("numInts: %d\n", numInts);
        addMod4<<<(numInts - 1) / thread_blk_size + 1, thread_blk_size>>>(numInts, d_A, d_B);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    template <typename T>
    void reconstructHelper(int bw, int N, u64 memSz, int numInts, T *d_A0, Stats *s, T *d_A1 = NULL)
    {
        if (!d_A1)
            d_A1 = (T *)moveToGPU((u8 *)h_bufA1, memSz, s);
        if (bw == 1)
            gpuXor((u32 *)d_A0, (u32 *)d_A1, numInts, s);
        else if (bw == 2)
            gpuAddMod4((u32 *)d_A0, (u32 *)d_A1, N);
        else
            gpuLinearComb(bw, N, d_A0, T(1), d_A0, T(1), d_A1);
        gpuFree(d_A1);
    }

public:
    GpuPeer(bool compress) : SigmaPeer(true, compress)
    {
    }

    void connect(int party, std::string addr, int port = 42003)
    {
        SigmaPeer::connect(party, addr, port);
        printf("Not setting socket priority!\n");
        // int optval = 7; // valid values are in the range [1,7]
        //                 // 1- low priority, 7 - high priority
        // auto sendsocket = static_cast<SocketBuf *>(this->peer->keyBuf)->sendsocket;
        // int prio = -1;
        // socklen_t len;
        // if (getsockopt(sendsocket, SOL_SOCKET, SO_PRIORITY, &prio, &len) < 0)
        // {
        //     assert(0 && "setsockopt error");
        // }
        // printf("Current prio=%d, %lu\n", prio, len);
        // if (setsockopt(sendsocket, SOL_SOCKET, SO_PRIORITY, &optval, sizeof(optval)) < 0)
        // {
        //     printf("errno: %d, %s\n", errno, strerror(errno));
        //     assert(0 && "setsockopt error");
        // }
        // if (getsockopt(sendsocket, SOL_SOCKET, SO_PRIORITY, &prio, &len) < 0)
        // {
        //     assert(0 && "setsockopt error");
        // }
        // printf("New prio=%d\n", prio);
    }

    template <typename T>
    void _reconstructInPlace(T *d_A0, int bw, int N, Stats *s)
    {
        // printf("%d, %d\n", bw, N);
        size_t memSz = 0, numInts = 0;
        auto d_compressedA0 = compressMem(bw, bw, N, d_A0, memSz, numInts, s);
        moveIntoCPUMem(h_bufA0, (u8 *)d_compressedA0 /*d_A0*/, memSz, s);
        if (d_compressedA0 != (u8 *)d_A0)
            gpuFree(d_compressedA0);
        this->exchangeShares((u8 *)h_bufA0, memSz, s);
        auto d_A1 = expandMem<T>(bw, N, h_bufA1, memSz, numInts, s);
        reconstructHelper(bw, N, memSz, numInts, d_A0, s, d_A1);
    }

    template <typename T>
    void _send(T *d_A0, int bw, int N, Stats *s)
    {
        size_t memSz = 0, numInts = 0;
        auto d_compressedA0 = compressMem(bw, bw, N, d_A0, memSz, numInts, s);
        moveIntoCPUMem(h_bufA0, (u8 *)d_compressedA0 /*d_A0*/, memSz, s);
        if (d_compressedA0 != (u8 *)d_A0)
            gpuFree(d_compressedA0);
        auto start = std::chrono::high_resolution_clock::now();
        this->sendBytes((u8 *)h_bufA0, memSz);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        if (s)
            s->comm_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    template <typename T>
    T *_recv(int bw, int N, Stats *s)
    {
        size_t memSz = 0, numInts = 0;
        this->getMemSz<T>(bw, N, memSz, numInts);
        auto start = std::chrono::high_resolution_clock::now();
        this->recvBytes((u8 *)h_bufA0, memSz);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        if (s)
            s->comm_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

        auto d_A1 = expandMem<T>(bw, N, h_bufA0, memSz, numInts, s);
        return d_A1;
    }

    void Send(u64 *h_A0, int bw, u64 N, Stats *s)
    {
        _send<u64>(h_A0, bw, N, s);
    }

    void Send(u32 *h_A0, int bw, u64 N, Stats *s)
    {
        _send<u32>(h_A0, bw, N, s);
    }

    void Send(u8 *h_A0, int bw, u64 N, Stats *s)
    {
        _send<u8>(h_A0, bw, N, s);
    }

    u8 *Recv(int bw, u64 N, Stats *s)
    {
        return _recv<u8>(bw, N, s);
    }

    void reconstructInPlace(u64 *A0, int bw, u64 N, Stats *s)
    {
        _reconstructInPlace<u64>(A0, bw, N, s);
    }

    void reconstructInPlace(u32 *A0, int bw, u64 N, Stats *s)
    {
        _reconstructInPlace<u32>(A0, bw, N, s);
    }

    void reconstructInPlace(u16 *A0, int bw, u64 N, Stats *s)
    {
        _reconstructInPlace<u16>(A0, bw, N, s);
    }

    void reconstructInPlace(u8 *A0, int bw, u64 N, Stats *s)
    {
        _reconstructInPlace<u8>(A0, bw, N, s);
    }

    template <typename T>
    T *_addAndReconstruct(int bw, u64 N, T *d_A0, T *h_B0, Stats *s)

    {
        auto d_B0 = (T *)moveToGPU((u8 *)h_B0, N * sizeof(T), s);
        gpuLinearComb(bw, N, d_A0, T(1), d_A0, T(1), d_B0);
        gpuFree(d_B0);
        _reconstructInPlace(d_A0, bw, N, s);
        return d_A0;
    }

    u64 *addAndReconstruct(int bw, u64 N, u64 *d_A0, u64 *h_B0, Stats *s, bool inPlace = true)
    {
        assert(inPlace == true);
        return _addAndReconstruct(bw, N, d_A0, h_B0, s);
    }

    u32 *addAndReconstruct(int bw, u64 N, u32 *A0, u32 *B0, Stats *s, bool inPlace)
    {
        assert(0);
    }
};

// T *reconstruct(T *h_A0, int bw, int N, Stats *s)
// {
//     size_t memSz, numInts = 0;
//     getMemSz(bw, N, memSz, numInts);
//     exchangeShares(peer, (u8 *)h_A0, h_bufA1, memSz, party, s);
//     auto d_A0 = (T *)moveToGPU((u8 *)h_A0, memSz, s);
//     reconstructHelper(bw, N, memSz, numInts, d_A0, s);
//     return d_A0;
// }