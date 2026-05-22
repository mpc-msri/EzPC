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
#include <sys/socket.h>

class CpuPeer : public SigmaPeer
{
private:
    // compression also takes time
    template <typename T>
    u8 *cpuCompressMem(int bw, int modBw, int N, T *h_A0, size_t &memSz, size_t &numInts, Stats *s)
    {
        assert(modBw == bw);
        memSz = size_t((N * bw - 1) / 64 + 1) * 8;
        u8 *h_compressedA0 = (u8 *)malloc(memSz);
        if (bw < 8 * sizeof(T))
        {
#pragma omp parallel for
            for (int i = 0; i < memSz / 8; i++)
            {
                int b = (64 * i) / bw;
                int elems = 63 / bw + 2;
                if (b + elems > N)
                    elems = N - b;
                u64 temp = 0;
                int offset = (64 * i) % bw;
                u64 elem = h_A0[b];
                cpuMod(elem, modBw);
                // printf("%d=%lu\n", b, h_A0[b]);
                temp = u64(/*h_A0[b]*/ elem >> offset);
                offset = bw - offset;
                for (int j = 1; j < elems; j++)
                {
                    u64 elem = h_A0[b + j];
                    cpuMod(elem, modBw);
                    // printf("%d=%lu\n", b + j, h_A0[b + j]);
                    temp += (elem << offset);
                    offset += bw;
                    if (offset >= 64)
                        break;
                }
                ((u64 *)h_compressedA0)[i] = temp;
            }
        }
        else
        {
            assert(modBw == bw);
            memcpy(h_compressedA0, h_A0, memSz);
        }
        return h_compressedA0;
    }

    void cpuXor(u64 N, u32 *x, u32 *y)
    {
#pragma omp parallel for
        for (u64 i = 0; i < N; i++)
        {
            x[i] ^= y[i];
        }
    }

    template <typename T>
    T *cpuAdd(int bw, u64 N, const T *x, const T *y)
    {
        T *z = (T *)malloc(N * sizeof(T));
#pragma omp parallel for
        for (u64 i = 0; i < N; i++)
        {
            z[i] = x[i] + y[i];
            cpuMod(z[i], bw);
        }
        return z;
    }

    template <typename T>
    void cpuAddInPlace(int bw, u64 N, T *x, T *y)
    {
#pragma omp parallel for
        for (u64 i = 0; i < N; i++)
        {
            x[i] += y[i];
            cpuMod(x[i], bw);
        }
    }

public:
    CpuPeer() : SigmaPeer(false, false)
    {
    }

    void connect(int party, std::string addr, int port = 42003)
    {
        SigmaPeer::connect(party, addr, port);
        // int optval = 4; // valid values are in the range [1,7]
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
    void _send(T *h_A0, int bw, u64 N, Stats *s)
    {
        size_t memSz = 0, numInts = 0;
        this->getMemSz<T>(bw, N, memSz, numInts);
        // printf("Getting mem size=%lu\n", memSz);
        auto start = std::chrono::high_resolution_clock::now();
        this->sendBytes((u8 *)h_A0, memSz);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        if (s)
            s->comm_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    template <typename T>
    T *_recv(int bw, u64 N, Stats *s)
    {
        size_t memSz = 0, numInts = 0;
        this->getMemSz<T>(bw, N, memSz, numInts);
        // if (!h_A)
        auto h_A = (T *)cpuMalloc(memSz, false);
        auto start = std::chrono::high_resolution_clock::now();
        this->recvBytes((u8 *)h_A, memSz);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        if (s)
            s->comm_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        return h_A;
    }

    template <typename T>
    void _reconstructInPlace(T *A0, int bw, u64 N, Stats *s)
    {
        size_t memSz = 0, numInts = 0;
        this->getMemSz<T>(bw, N, memSz, numInts);
        this->exchangeShares((u8 *)A0, memSz, s);
        if (bw == 1)
            cpuXor(numInts, (u32 *)A0, (u32 *)h_bufA1);
        else
            cpuAddInPlace(bw, N, A0, (T *)h_bufA1);
    }

    template <typename T>
    T *_addAndReconstruct(int bw, u64 N, T *A0, T *B0, Stats *s, bool inPlace)
    {
        auto C0 = A0;
        if (inPlace)
            cpuAddInPlace(bw, N, A0, B0);
        else
            C0 = cpuAdd(bw, N, A0, B0);
        this->reconstructInPlace(C0, bw, N, s);
        return C0;
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

    u64 *addAndReconstruct(int bw, u64 N, u64 *A0, u64 *B0, Stats *s, bool inPlace = false)
    {
        return _addAndReconstruct<u64>(bw, N, A0, B0, s, inPlace);
    }

    u32 *addAndReconstruct(int bw, u64 N, u32 *A0, u32 *B0, Stats *s, bool inPlace)
    {
        return _addAndReconstruct<u32>(bw, N, A0, B0, s, inPlace);
    }
};