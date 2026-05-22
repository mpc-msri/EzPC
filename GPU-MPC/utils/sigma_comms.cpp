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

#include "sigma_comms.h"
#include "gpu_file_utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "helper_cuda.h"
#include "gpu_mem.h"
#include <chrono>
#include <sys/socket.h>
#include <future>

size_t OneGB = 1024 * 1024 * 1024;

SigmaPeer::SigmaPeer(bool pinMem, bool compress) : compress(compress), sendThread{}, mtx{}, cv{}
{
    initCommBufs(pinMem);
    sendHasWork = false;
    sendThread = std::thread(&SigmaPeer::wait, this);
    // sendThread.detach();
}

void SigmaPeer::wait()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]()
                {return terminate || sendHasWork; });
        if (sendHasWork)
        {
            sendBytes(sendBuf, sendSz);
            sendHasWork = false;
        }
        if (terminate)
        {
            return;
        }
        lock.unlock();
    }
}

void SigmaPeer::initCommBufs(bool pinMem)
{
    printf("Allocating %lu bytes of memory for comm bufs\n", commBufSize);
    h_bufA0 = cpuMalloc(commBufSize, pinMem);
    h_bufA1 = cpuMalloc(commBufSize, pinMem);
}

void SigmaPeer::freeCommBufs(bool pinMem)
{
    cpuFree(h_bufA0, pinMem);
    cpuFree(h_bufA1, pinMem);
}

void SigmaPeer::sendBytes(const u8 *data, size_t size)
{
    int sendsocket = static_cast<SocketBuf *>(peer->keyBuf)->sendsocket;
    writeKeyBuf(sendsocket, size, data);
    peer->keyBuf->bytesSent += size;
}

void SigmaPeer::recvBytes(u8 *data, size_t size)
{
    int recvsocket = static_cast<SocketBuf *>(peer->keyBuf)->recvsocket;
    size_t chunkSize = (1ULL << 30);
    size_t bytesRead = 0;
    while (bytesRead < size)
    {
        size_t toRead = std::min(chunkSize, size - bytesRead);
        ssize_t numRead = recv(recvsocket, data + bytesRead, toRead, MSG_WAITALL);

        if (numRead == -1)
        {
            printf("errno: %d, %s\n", errno, strerror(errno));
            assert(0 && "read");
        }
        assert(numRead == toRead);
        bytesRead += numRead;
    }
    // assert(size == recv(peer->recvsocket, data, size, MSG_WAITALL));
    peer->keyBuf->bytesReceived += size;
}

void SigmaPeer::exchangeShares(u8 *to_send, size_t bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel /*sections*/ num_threads(2)
    //     {
    //         // printf("thread %d entering\n", omp_get_thread_num());
    // #pragma omp sections
    //         {
    // #pragma omp section
    //             {
    //                 // printf("send %d %d %lu\n", omp_get_thread_num(), omp_get_num_threads(), bytes);
    //                 sendBytes(to_send, bytes);
    //             }
    // #pragma omp section
    //             {
    //                 // printf("recv %d %d %lu\n", omp_get_thread_num(), omp_get_num_threads(), bytes);
    //                 recvBytes(h_bufA1, bytes);
    //             }
    //         }
    //         // printf("thread %d exiting, %d\n", omp_get_thread_num(), omp_get_active_level());
    //     }
    // std::future<void> res = std::async(&SigmaPeer::sendBytes, this, to_send, bytes);
    // std::thread send_thread(&SigmaPeer::sendBytes, this, to_send, bytes);
    // std::thread recv_thread(&recvBytes, peer, to_recv, bytes);
    {
        std::lock_guard<std::mutex> lock(mtx);
        sendBuf = to_send;
        sendSz = bytes;
        sendHasWork = true;
    }
    cv.notify_one();
    recvBytes(h_bufA1, bytes);
    while (sendHasWork)
    {
    }
    // res.get();
    // send_thread.join();
    // recv_thread.join();
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->comm_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    // std::cout << "Time to exchange shares in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " " << s->comm_time << std::endl;
    // return h_bufA1;
}

void SigmaPeer::connect(int party, std::string addr, int port)
{
    this->party = party;
    // Peer* peer;
    if (party == SERVER0)
        peer = waitForPeer(/*42003*/ port);
    else
        peer = new Peer(addr, /*42003*/ port);
    peer->sync();
    // return peer;
}

template <typename T>
void SigmaPeer::getMemSz(int bw, u64 N, size_t &memSz, size_t &numInts)
{
    if (bw > 2)
        memSz = N * sizeof(T);
    else
    {
        assert(bw == 1 || bw == 2);
        numInts = ((bw * N - 1) / 32 + 1);
        memSz = numInts * sizeof(u32);
    }
    assert(memSz < commBufSize);
}

template void SigmaPeer::getMemSz<u64>(int bw, u64 N, size_t &memSz, size_t &numInts);
template void SigmaPeer::getMemSz<u32>(int bw, u64 N, size_t &memSz, size_t &numInts);
template void SigmaPeer::getMemSz<u16>(int bw, u64 N, size_t &memSz, size_t &numInts);
template void SigmaPeer::getMemSz<u8>(int bw, u64 N, size_t &memSz, size_t &numInts);