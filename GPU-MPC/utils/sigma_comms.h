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

#include <sytorch/tensor.h>
#include <llama/comms.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "gpu_stats.h"

extern size_t OneGB;
// extern u8 *h_bufA0, *h_bufA1;
// extern size_t commBufSize;
class SigmaPeer
{
public:
    size_t commBufSize = 5 * OneGB;
    u8 *h_bufA0, *h_bufA1;
    u8 *sendBuf = nullptr;
    size_t sendSz;
    Peer *peer;
    int party;
    bool compress;
    std::mutex mtx;
    std::condition_variable cv;
    bool terminate = false;
    std::atomic<bool> sendHasWork;
    std::thread sendThread;

    SigmaPeer(bool pinMem, bool compress);
    void wait();
    void initCommBufs(bool pinMem);
    void freeCommBufs(bool pinMem);
    void sendBytes(const u8 *data, size_t size);
    void recvBytes(u8 *data, size_t size);
    /*virtual*/ void exchangeShares(u8 *to_send, size_t bytes, Stats *s);
    void connect(int party, std::string addr, int port);
    inline void sync()
    {
        peer->sync();
    }

    inline u64 bytesSent()
    {
        return peer->bytesSent();
    }

    inline u64 bytesReceived()
    {
        return peer->bytesReceived();
    }

    inline void close()
    {
        {
            std::unique_lock<std::mutex> lock(mtx);
            terminate = true;
        }
        cv.notify_one();
        sendThread.join();
        peer->close();
    }

    template <typename T>
    void getMemSz(int bw, u64 N, size_t &memSz, size_t &numInts);
    virtual void Send(u64 *h_A0, int bw, u64 N, Stats *s) = 0;
    virtual void Send(u32 *h_A0, int bw, u64 N, Stats *s) = 0;
    virtual void Send(u8 *h_A0, int bw, u64 N, Stats *s) = 0;
    virtual u8 *Recv(int bw, u64 N, Stats *s) = 0;
    virtual void reconstructInPlace(u64 *A0, int bw, u64 N, Stats *s) = 0;
    virtual void reconstructInPlace(u32 *A0, int bw, u64 N, Stats *s) = 0;
    virtual void reconstructInPlace(u16 *A0, int bw, u64 N, Stats *s) = 0;
    virtual void reconstructInPlace(u8 *A0, int bw, u64 N, Stats *s) = 0;
    virtual u64 *addAndReconstruct(int bw, u64 N, u64 *A0, u64 *B0, Stats *s, bool inPlace) = 0;
    virtual u32 *addAndReconstruct(int bw, u64 N, u32 *A0, u32 *B0, Stats *s, bool inPlace) = 0;
};
