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

/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <llama/comms.h>
#include <llama/utils.h>
#include <llama/array.h>
#include <llama/config.h>
#include <llama/stats.h>
#include <llama/assert.h>
#include <llama/freekey.h>
#include <llama/api.h>
#include <llama/conv.h>

#include "and.h"
#include "mult.h"
#include "pubdiv.h"
#include "relu.h"
#include "signextend.h"
#include "clip.h"
#include <llama/dcf.h>
#include "lut.h"
#include "select.h"
#include "fixtobfloat16.h"
#include "wrap.h"
#include <llama/dpf.h>
#include "taylor.h"
#include "float.h"

#include <cassert>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <thread>
#include <Eigen/Dense>
#include <bitpack/bitpack.h>

template <typename T>
using pair = std::pair<T, T>;

bool localTruncation = false;

using namespace LlamaConfig;

template <typename Functor>
uint64_t time_this_block(Functor f)
{
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

template <typename Functor>
auto time_comm_this_block(Functor f)
{
    uint64_t comm_start = peer->bytesReceived() + peer->bytesSent();
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    uint64_t comm_end = peer->bytesReceived() + peer->bytesSent();
    return std::make_pair((uint64_t)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()), comm_end - comm_start);
}

void llama::start()
{
    Llama::stats.clear();
    // std::cerr << "=== COMPUTATION START ===\n\n";
    if (party != DEALER)
        peer->sync();

    startTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    if (party != DEALER)
    {
        if (party == SERVER)
        {
            inputOfflineComm = peer->bytesSent();
            inputOnlineComm = peer->bytesReceived();
        }
        else
        {
            inputOfflineComm = peer->bytesReceived();
            inputOnlineComm = peer->bytesSent();
        }
        peer->zeroBytesSent();
        peer->zeroBytesReceived();
    }
    else
    {
        // always_assert(server->bytesSent() == 16);
        // always_assert(server->bytesSent() == 16);
        server->zeroBytesSent();
        client->zeroBytesSent();
    }

    if (party == DEALER)
    {
        osuCrypto::AES aesSeed(prngs[0].get<osuCrypto::block>());
        auto commonSeed = aesSeed.ecbEncBlock(osuCrypto::ZeroBlock);
        server->send_block(commonSeed);
        prngShared.SetSeed(commonSeed);
    }
    else if (party == SERVER)
    {
        auto commonSeed = dealer->recv_block();
        prngShared.SetSeed(commonSeed);
    }
    sendTime = 0;
    recvTime = 0;
    packTime = 0;
    unpackTime = 0;
}

void llama::end()
{
    // std::cerr << "\n=== COMPUTATION END ===\n\n";
    if (party != DEALER)
    {
        uint64_t agg_time = 0;
        uint64_t recons_time = 0;
        uint64_t keyread_time = 0;
        for (auto &func : Llama::stats)
        {
            uint64_t online_time = func.second.compute_time + func.second.reconstruct_time;
            agg_time += online_time;
            recons_time += func.second.reconstruct_time;
            keyread_time += func.second.keyread_time;
        }
        std::cerr << "Offline Communication = " << inputOfflineComm << " bytes\n";
        std::cerr << "Offline Time = " << accumulatedInputTimeOffline / 1000.0 << " milliseconds\n";
        std::cerr << "Online Rounds = " << numRounds << "\n";
        std::cerr << "Online Communication = " << (peer->bytesSent() + peer->bytesReceived() + inputOnlineComm + secFloatComm) /*/ (1024.0 * 1024.0)*/ << " B\n";
        std::cerr << "Input Online Communication = " << (inputOnlineComm) << " B\n";
        std::cerr << "Secfloat Online Communication = " << (secFloatComm) /*/ (1024.0 * 1024.0)*/ << " B\n";

        std::cerr << "Online Time = " << (evalMicroseconds + accumulatedInputTimeOnline + agg_time) / 1000.0 << " milliseconds\n";
        std::cerr << "Key Read Time = " << keyread_time / 1000.0 << " milliseconds\n";
        std::cerr << "Total Eigen Time = " << eigenMicroseconds / 1000.0 << " milliseconds\n";
        auto endTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::cerr << "Key Read Time = " << keyread_time / 1000.0 << " milliseconds\n";
        std::cerr << "Total Time (including Key Read) = " << (endTime - startTime) / 1000000.0 << " milliseconds\n";

        std::cerr << "packTime = " << packTime / 1000.0 << " miliseconds\n";
        std::cerr << "sendTime = " << sendTime / 1000.0 << " miliseconds\n";
        std::cerr << "recvTime = " << recvTime / 1000.0 << " miliseconds\n";
        std::cerr << "unpackTime = " << unpackTime / 1000.0 << " miliseconds\n";
        std::cerr << "reconsTime = " << recons_time / 1000.0 << " miliseconds\n";
        std::cerr << "accumulatedInputTimeOnline = " << accumulatedInputTimeOnline / 1000.0 << " miliseconds\n";

        if (convEvalMicroseconds > 0)
            std::cerr << "Conv Time = " << convEvalMicroseconds / 1000.0 << " milliseconds\n";
        if (arsEvalMicroseconds > 0)
            std::cerr << "ARS Time = " << arsEvalMicroseconds / 1000.0 << " milliseconds\n";

        if (convOnlineComm > 0)
            std::cerr << "Conv Online Communication = " << convOnlineComm << " bytes\n";
        if (arsOnlineComm > 0)
            std::cerr << "ARS Online Communication = " << arsOnlineComm << " bytes\n";

        Llama::dump_stats_csv("llama" + std::to_string(party) + ".csv");
    }

    std::cerr << "=========\n";
}

const bool parallel_reconstruct = true;
const bool doPack = false;

inline void pack_wrapper(GroupElement *dst, const GroupElement *src, std::size_t n, int bw)
{
    auto start = std::chrono::high_resolution_clock::now();
    bitpack::pack(dst, src, n, bw);
    auto end = std::chrono::high_resolution_clock::now();
    packTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

inline void unpack_wrapper(GroupElement *dst, const GroupElement *src, std::size_t n, int bw)
{
    auto start = std::chrono::high_resolution_clock::now();
    bitpack::unpack(dst, src, n, bw);
    auto end = std::chrono::high_resolution_clock::now();
    unpackTime += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void packed_reconstruct(int32_t size, GroupElement *arr, int bw)
{
    auto psize = bitpack::packed_size(size, bw);
    GroupElement *packedArr = new GroupElement[psize];
    GroupElement *packedTmp = new GroupElement[psize];
    pack_wrapper(packedArr, arr, size, bw);

    if (parallel_reconstruct)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                peer->send_batched_input(packedArr, psize, 64);
            }

#pragma omp section
            {
                peer->recv_batched_input(packedTmp, psize, 64);
            }
        }
    }
    else
    {
        peer->send_batched_input(packedArr, psize, 64);
        peer->recv_batched_input(packedTmp, psize, 64);
    }

    GroupElement *tmp = new GroupElement[size];
    unpack_wrapper(tmp, packedTmp, size, bw);

    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] + tmp[i];
    }
    delete[] tmp;
    delete[] packedArr;
    delete[] packedTmp;
    numRounds += 1;
}

void reconstruct(int32_t size, GroupElement *arr, int bw)
{
    if (doPack)
    {
        return packed_reconstruct(size, arr, bw);
    }
    uint64_t *tmp = new uint64_t[size];

    if (parallel_reconstruct)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                peer->send_batched_input(arr, size, bw);
            }

#pragma omp section
            {
                peer->recv_batched_input(tmp, size, bw);
            }
        }
    }
    else
    {
        peer->send_batched_input(arr, size, bw);
        peer->recv_batched_input(tmp, size, bw);
    }
    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] + tmp[i];
    }
    delete[] tmp;
    numRounds += 1;
}

void serverReconstruct(int32_t size, GroupElement *arr, int bw)
{
    // TODO: do packing
    if (party == CLIENT)
    {
        peer->send_batched_input(arr, size, bw);
    }
    else
    {
        uint64_t *tmp = new uint64_t[size];
        peer->recv_batched_input(tmp, size, bw);
        for (int i = 0; i < size; i++)
        {
            arr[i] = arr[i] + tmp[i];
        }
        delete[] tmp;
    }
    numRounds += 1;
}

void serverToClient(int32_t size, GroupElement *arr, int bw)
{
    // TODO: do packing
    if (party == SERVER)
    {
        peer->send_batched_input(arr, size, bw);
    }
    else
    {
        peer->recv_batched_input(arr, size, bw);
    }
    numRounds += 1;
}

void reconstructRT(int32_t size, GroupElement *arr, int bw)
{
    int bitarraySize = size % 8 == 0 ? size / 8 : size / 8 + 1;

    uint8_t *tmp2 = new uint8_t[bitarraySize];
    uint8_t *tmp3 = new uint8_t[bitarraySize];

    packBitArray(arr + size, size, tmp2);

    uint64_t *tmp = new uint64_t[size];
    GroupElement *sendArr, *recvArr;
    int sendSize;
    if (doPack)
    {
        auto psize = bitpack::packed_size(size, bw);
        sendArr = new GroupElement[psize];
        recvArr = new GroupElement[psize];
        pack_wrapper(sendArr, arr, size, bw);
        sendSize = psize;
    }
    else
    {
        sendArr = arr;
        recvArr = tmp;
        sendSize = size;
    }

    if (parallel_reconstruct)
    {
#pragma omp parallel sections
        {
#pragma omp section
            {
                peer->send_batched_input(sendArr, sendSize, (doPack ? 64 : bw));
                peer->send_uint8_array(tmp2, bitarraySize);
            }

#pragma omp section
            {
                peer->recv_batched_input(recvArr, sendSize, (doPack ? 64 : bw));
                peer->recv_uint8_array(tmp3, bitarraySize);
            }
        }
    }
    else
    {
        peer->send_batched_input(sendArr, sendSize, (doPack ? 64 : bw));
        peer->send_uint8_array(tmp2, bitarraySize);
        peer->recv_batched_input(recvArr, sendSize, (doPack ? 64 : bw));
        peer->recv_uint8_array(tmp3, bitarraySize);
    }

    if (doPack)
    {
        unpack_wrapper(tmp, recvArr, size, bw);
        delete[] sendArr;
        delete[] recvArr;
    }

    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] + tmp[i];
        arr[i + size] = arr[i + size] + ((tmp3[i / 8] >> (i % 8)) & 1);
    }

    delete[] tmp;
    delete[] tmp2;
    delete[] tmp3;
    numRounds += 1;
}

inline std::pair<int32_t, int32_t> get_start_end(int32_t size, int32_t thread_idx)
{
    int32_t chunk_size = size / num_threads;
    if (thread_idx == num_threads - 1)
    {
        return std::make_pair(thread_idx * chunk_size, size);
    }
    else
    {
        return std::make_pair(thread_idx * chunk_size, (thread_idx + 1) * chunk_size);
    }
}

void Conv2DWrapper(int32_t N, int32_t H, int32_t W,
                   int32_t CI, int32_t FH, int32_t FW,
                   int32_t CO, int32_t zPadHLeft,
                   int32_t zPadHRight, int32_t zPadWLeft,
                   int32_t zPadWRight, int32_t strideH,
                   int32_t strideW, MASK_PAIR(GroupElement *inputArr), MASK_PAIR(GroupElement *filterArr),
                   MASK_PAIR(GroupElement *outArr))
{
    std::cerr << ">> Conv2D - Start" << std::endl;
    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;

    if (party == DEALER)
    {
        auto local_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < d0; ++i)
        {
            for (int j = 0; j < d1; ++j)
            {
                for (int k = 0; k < d2; ++k)
                {
                    for (int l = 0; l < d3; ++l)
                    {
                        Arr4DIdx(outArr_mask, d0, d1, d2, d3, i, j, k, l) = random_ge(bitlength);
                    }
                }
            }
        }

        auto keys = KeyGenConv2D(bitlength, bitlength, N, H, W, CI, FH, FW, CO,
                                 zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW,
                                 inputArr_mask, filterArr_mask, outArr_mask);

        auto local_end = std::chrono::high_resolution_clock::now();

        // server->send_conv2d_key(keys.first);
        freeConv2dKey(keys.first);
        client->send_conv2d_key(keys.second);
        freeConv2dKey(keys.second);
        auto local_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                                                      local_start)
                                    .count();
        dealerMicroseconds += local_time_taken;
        std::cerr << "   Dealer Time = " << local_time_taken / 1000.0 << " milliseconds\n";
    }
    else
    {

        auto keyread_start = std::chrono::high_resolution_clock::now();
        auto key = dealer->recv_conv2d_key(bitlength, bitlength, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW);
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        peer->sync();
        uint64_t eigen_start = eigenMicroseconds;
        auto local_start = std::chrono::high_resolution_clock::now();
        EvalConv2D(party, key, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr);
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        reconstruct(d0 * d1 * d2 * d3, outArr, bitlength);
        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        convOnlineComm += (onlineComm1 - onlineComm0);
        auto local_end = std::chrono::high_resolution_clock::now();

        freeConv2dKey(key);
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                                                  local_start)
                                .count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                                                      t1)
                                    .count();
        convEvalMicroseconds += (reconstruct_time + compute_time);
        evalMicroseconds += (reconstruct_time + compute_time);
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "      Eigen Time = " << (eigenMicroseconds - eigen_start) / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";
    }

    std::cerr << ">> Conv2D - End" << std::endl;
}

void Conv3DWrapper(int32_t N, int32_t D, int32_t H, int32_t W,
                   int32_t CI, int32_t FD, int32_t FH, int32_t FW,
                   int32_t CO, int32_t zPadDLeft, int32_t zPadDRight, int32_t zPadHLeft,
                   int32_t zPadHRight, int32_t zPadWLeft,
                   int32_t zPadWRight, int32_t strideD, int32_t strideH,
                   int32_t strideW, GroupElement *inputArr, GroupElement *filterArr,
                   GroupElement *outArr)
{
    std::cerr << ">> Conv3D - Start" << std::endl;
    int d0 = N;
    int d1 = ((D - FD + (zPadDLeft + zPadDRight)) / strideD) + 1;
    int d2 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d3 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d4 = CO;

    if (party == DEALER)
    {
        auto local_start = std::chrono::high_resolution_clock::now();

        // not good for in place operations
        for (int i = 0; i < d0 * d1 * d2 * d3 * d4; ++i)
        {
            outArr[i] = random_ge(bitlength);
        }

        auto keys = KeyGenConv3D(bitlength, bitlength, N, D, H, W, CI, FD, FH, FW, CO,
                                 zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW,
                                 inputArr, filterArr, outArr);

        auto local_end = std::chrono::high_resolution_clock::now();

        // server->send_conv2d_key(keys.first);
        freeConv3dKey(keys.first);
        client->send_conv3d_key(keys.second);
        freeConv3dKey(keys.second);
        auto local_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                                                      local_start)
                                    .count();
        dealerMicroseconds += local_time_taken;
        std::cerr << "   Dealer Time = " << local_time_taken / 1000.0 << " milliseconds\n";
    }
    else
    {

        auto keyread_start = std::chrono::high_resolution_clock::now();
        auto key = dealer->recv_conv3d_key(bitlength, bitlength, N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW);
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end - keyread_start).count();

        peer->sync();
        uint64_t eigen_start = eigenMicroseconds;
        auto local_start = std::chrono::high_resolution_clock::now();
        EvalConv3D(party, key, N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, inputArr, filterArr, outArr);
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        reconstruct(d0 * d1 * d2 * d3 * d4, outArr, bitlength);
        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        convOnlineComm += (onlineComm1 - onlineComm0);
        auto local_end = std::chrono::high_resolution_clock::now();

        freeConv3dKey(key);
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                                                  local_start)
                                .count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                                                      t1)
                                    .count();
        convEvalMicroseconds += (reconstruct_time + compute_time);
        evalMicroseconds += (reconstruct_time + compute_time);
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "      Eigen Time = " << (eigenMicroseconds - eigen_start) / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";
    }

    std::cerr << ">> Conv3D - End" << std::endl;
}

void Conv2DGroupWrapper(int64_t N, int64_t H, int64_t W,
                        int64_t CI, int64_t FH, int64_t FW,
                        int64_t CO, int64_t zPadHLeft,
                        int64_t zPadHRight, int64_t zPadWLeft,
                        int64_t zPadWRight, int64_t strideH,
                        int64_t strideW, int64_t G,
                        MASK_PAIR(GroupElement *inputArr), MASK_PAIR(GroupElement *filterArr), MASK_PAIR(GroupElement *outArr))
{
    if (G == 1)
    {
        Conv2DWrapper(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, inputArr_mask, filterArr, filterArr_mask, outArr, outArr_mask);
    }
    else
    {
        // TODO
        assert(false && "Conv2DGroup not implemented");
    }
}

void ScaleUp(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf)
{
    if (party == DEALER)
    {
        for (int i = 0; i < size; ++i)
        {
            inArr_mask[i] = inArr_mask[i] << sf;
        }
    }
    else
    {
        for (int i = 0; i < size; ++i)
        {
            inArr[i] = inArr[i] << sf;
        }
    }
}

void ars_threads_helper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *outArr, ARSKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        outArr[i] = evalARS(party - 2, inArr[i], keys[i].shift, keys[i]);
        freeARSKeyPack(keys[i]);
    }
}

/*
        auto keyread_start = std::chrono::high_resolution_clock::now();
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();
        auto start = std::chrono::high_resolution_clock::now();
        auto mid = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        lolEvalMicroseconds += (reconstruct_time + compute_time);
*/
void ARS(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t shift)
{
    std::cerr << ">> Truncate" << (LlamaConfig::stochasticT ? " (stochastic)" : "") << " - Start" << std::endl;
    if (party == DEALER)
    {
        pair<ARSKeyPack> *keys = new pair<ARSKeyPack>[size];
        auto dealer_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            GroupElement rout = random_ge(bitlength);
            keys[i] = keyGenARS(bitlength, bitlength, shift, inArr_mask[i], rout);
            outArr_mask[i] = rout;
        }
        auto dealer_end = std::chrono::high_resolution_clock::now();
        auto dealer_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(dealer_end -
                                                                                       dealer_start)
                                     .count();

        for (int i = 0; i < size; i++)
        {
            server->send_ars_key(keys[i].first);
            client->send_ars_key(keys[i].second);
            freeARSKeyPackPair(keys[i]);
        }
        dealerMicroseconds += dealer_time_taken;
        delete[] keys;
        std::cerr << "   Dealer Time = " << dealer_time_taken / 1000.0 << " milliseconds\n";
    }
    else
    {
        ARSKeyPack *keys = new ARSKeyPack[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
        {
            keys[i] = dealer->recv_ars_key(bitlength, bitlength, shift);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(ars_threads_helper, i, size, inArr, outArr, keys);
        }

        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }
        auto mid = std::chrono::high_resolution_clock::now();

        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        reconstruct(size, outArr, bitlength);
        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        arsOnlineComm += (onlineComm1 - onlineComm0);

        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        arsEvalMicroseconds += (reconstruct_time + compute_time);
        delete[] keys;
    }
    std::cerr << ">> Truncate - End" << std::endl;
}

void ScaleDown(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf)
{
    std::cerr << ">> ScaleDown - Start " << std::endl;

    if (localTruncation)
    {
        uint64_t m = ((1L << sf) - 1) << (bitlength - sf);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
        {
            if (party == DEALER)
            {
                auto x_msb = msb(inArr_mask[i], bitlength);
                inArr_mask[i] = x_msb ? (inArr_mask[i] >> sf) | m : inArr_mask[i] >> sf;
                mod(inArr_mask[i], bitlength);
            }
            else
            {
                auto x_msb = msb(inArr[i], bitlength);
                inArr[i] = x_msb ? (inArr[i] >> sf) | m : inArr[i] >> sf;
                mod(inArr[i], bitlength);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto timeMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (party == DEALER)
        {
            dealerMicroseconds += timeMicroseconds;
        }
        else
        {
            evalMicroseconds += timeMicroseconds;
            arsEvalMicroseconds += timeMicroseconds;
            std::cerr << "   Eval Time = " << timeMicroseconds / 1000.0 << " milliseconds\n";
        }
    }
    else
    {
        ARS(size, inArr, inArr_mask, inArr, inArr_mask, sf);
    }
    std::cerr << ">> ScaleDown - End " << std::endl;
}

inline void matmul2d_server_helper(int thread_idx, int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C, GroupElement *a, GroupElement *b, GroupElement *c)
{
    auto p = get_start_end(s1 * s3, thread_idx);
    for (int ik = p.first; ik < p.second; ik += 1)
    {
        int i = ik / s3;
        int k = ik % s3;
        Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(c, s1, s3, i, k);
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(C, s1, s3, i, k) - Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(b, s2, s3, j, k) - Arr2DIdx(a, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k) + Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k);
        }
        // mod(Arr2DIdx(C, s1, s3, i, k));
    }
}

inline void matmul2d_client_helper(int thread_idx, int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C, GroupElement *a, GroupElement *b, GroupElement *c)
{
    auto p = get_start_end(s1 * s3, thread_idx);
    for (int ik = p.first; ik < p.second; ik += 1)
    {
        int i = ik / s3;
        int k = ik % s3;
        Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(c, s1, s3, i, k);
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(C, s1, s3, i, k) - Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(b, s2, s3, j, k) - Arr2DIdx(a, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k);
        }
        // mod(Arr2DIdx(C, s1, s3, i, k));
    }
}

void MatMul2D(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
              MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA)
{
    if (party == DEALER)
    {

        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < s3; ++j)
            {
                Arr2DIdx(C_mask, s1, s3, i, j) = random_ge(bitlength);
            }
        }

        auto keys = KeyGenMatMul(bitlength, bitlength, s1, s2, s3, A_mask, B_mask, C_mask);

        // server->send_matmul_key(keys.first);
        freeMatMulKey(keys.first);
        client->send_matmul_key(keys.second);
        freeMatMulKey(keys.second);
    }
    else
    {
        MatMulKey key;
        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            { key = dealer->recv_matmul_key(bitlength, bitlength, s1, s2, s3); });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            { matmul_eval_helper(party, s1, s2, s3, A, B, C, key.a, key.b, key.c); });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(s1 * s3, C, bitlength); });

        Llama::stat_t stat = {"Linear::MatMul", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        freeMatMulKey(key);
    }
}

void ArgMax(int32_t rows, int32_t cols, MASK_PAIR(GroupElement *inp), MASK_PAIR(GroupElement *out))
{
    // inp is a vector of size rows*columns and max (resp. maxidx) is caclulated for every
    // column chunk of elements. Result maxidx is stored in out (size: rows)

    std::cerr << ">> ArgMax - Start" << std::endl;
    always_assert(rows == 1);
    if (party == DEALER)
    {
        int32_t curCols = cols;
        int32_t round = 0;

        GroupElement *tmpMax_mask = make_array<GroupElement>(rows, cols);
        GroupElement *tmpIdx_mask = make_array<GroupElement>(rows, cols);
        GroupElement *drelu_mask = make_array<GroupElement>(rows, cols / 2);
        GroupElement *mult_res_mask = make_array<GroupElement>(2 * rows, cols / 2);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                Arr2DIdx(tmpMax_mask, rows, cols, i, j) = Arr2DIdx(inp_mask, rows, cols, i, j);
                Arr2DIdx(tmpIdx_mask, rows, cols, i, j) = 0;
            }
        }

        while (curCols > 1)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {
                    Arr2DIdx(drelu_mask, rows, curCols / 2, row, j) = random_ge(bitlength);
                    auto scmpKeys = keyGenSCMP(bitlength, bitlength, Arr2DIdx(tmpMax_mask, rows, curCols, row, 2 * j), Arr2DIdx(tmpMax_mask, rows, curCols, row, 2 * j + 1), Arr2DIdx(drelu_mask, rows, curCols / 2, row, j));
                    server->send_scmp_keypack(scmpKeys.first);
                    client->send_scmp_keypack(scmpKeys.second);
                }
            }

            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {

                    Arr2DIdx(mult_res_mask, 2 * rows, curCols / 2, row, j) = random_ge(bitlength);
                    auto multKeys1 = MultGen(Arr2DIdx(drelu_mask, rows, curCols / 2, row, j), Arr2DIdx(tmpMax_mask, rows, curCols, row, 2 * j) - Arr2DIdx(tmpMax_mask, rows, curCols, row, 2 * j + 1), Arr2DIdx(mult_res_mask, 2 * rows, curCols / 2, row, j));

                    server->send_mult_key(multKeys1.first);
                    client->send_mult_key(multKeys1.second);

                    Arr2DIdx(mult_res_mask, 2 * rows, curCols / 2, rows + row, j) = random_ge(bitlength);
                    auto multKeys2 = MultGen(Arr2DIdx(drelu_mask, rows, curCols / 2, row, j), Arr2DIdx(tmpIdx_mask, rows, curCols, row, 2 * j) - Arr2DIdx(tmpIdx_mask, rows, curCols, row, 2 * j + 1), Arr2DIdx(mult_res_mask, 2 * rows, curCols / 2, rows + row, j));

                    server->send_mult_key(multKeys2.first);
                    client->send_mult_key(multKeys2.second);
                }
            }

            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {
                    Arr2DIdx(tmpMax_mask, rows, curCols / 2, row, j) = Arr2DIdx(mult_res_mask, 2 * rows, curCols / 2, row, j) + Arr2DIdx(tmpMax_mask, rows, curCols, row, 2 * j + 1);
                    Arr2DIdx(tmpIdx_mask, rows, curCols / 2, row, j) = Arr2DIdx(mult_res_mask, 2 * rows, curCols / 2, rows + row, j) + Arr2DIdx(tmpIdx_mask, rows, curCols, row, 2 * j + 1);
                }
                if (curCols % 2 == 1)
                {
                    Arr2DIdx(tmpMax_mask, rows, curCols / 2, row, curCols / 2) = Arr2DIdx(tmpMax_mask, 2 * rows, curCols, row, curCols - 1);
                    Arr2DIdx(tmpIdx_mask, rows, curCols / 2, row, curCols / 2) = Arr2DIdx(tmpIdx_mask, 2 * rows, curCols, row, curCols - 1);
                }
            }

            curCols = (curCols + 1) / 2;
            round += 1;
        }

        for (int row = 0; row < rows; row++)
        {
            out_mask[row] = Arr2DIdx(tmpIdx_mask, rows, 1, row, 0);
        }
        auto end = std::chrono::high_resolution_clock::now();
        dealerMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        delete[] tmpMax_mask;
        delete[] tmpIdx_mask;
        delete[] drelu_mask;
        delete[] mult_res_mask;
    }
    else
    {

        ScmpKeyPack keys[(cols - 1) * rows];
        MultKey mult_keys1[(cols - 1) * rows];
        MultKey mult_keys2[(cols - 1) * rows];
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;

        int32_t curCols = cols;
        while (curCols > 1)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {
                    keys[k1++] = dealer->recv_scmp_keypack(bitlength, bitlength);
                }
            }

            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {
                    mult_keys1[k2++] = dealer->recv_mult_key();
                    mult_keys2[k3++] = dealer->recv_mult_key();
                }
            }
            curCols = (curCols + 1) / 2;
        }

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        curCols = cols;
        k1 = 0;
        k2 = 0;
        k3 = 0;

        GroupElement *tmpMax = make_array<GroupElement>(rows, cols);
        GroupElement *tmpIdx = make_array<GroupElement>(rows, cols);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                Arr2DIdx(tmpMax, rows, cols, i, j) = Arr2DIdx(inp, rows, cols, i, j);
                Arr2DIdx(tmpIdx, rows, cols, i, j) = j;
            }
        }

        GroupElement *drelu = make_array<GroupElement>(rows, cols / 2);
        GroupElement *mult_res = make_array<GroupElement>(2 * rows, cols / 2);

        while (curCols > 1)
        {
            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {
                    Arr2DIdx(drelu, rows, curCols / 2, row, j) = evalSCMP(party - 2, keys[k1++], Arr2DIdx(tmpMax, rows, curCols, row, 2 * j), Arr2DIdx(tmpMax, rows, curCols, row, 2 * j + 1));
                }
            }

            reconstruct(rows * (curCols / 2), drelu, bitlength);

            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {

                    Arr2DIdx(mult_res, 2 * rows, curCols / 2, row, j) = MultEval(party - 2, mult_keys1[k2++], Arr2DIdx(drelu, rows, curCols / 2, row, j), Arr2DIdx(tmpMax, rows, curCols, row, 2 * j) - Arr2DIdx(tmpMax, rows, curCols, row, 2 * j + 1));

                    Arr2DIdx(mult_res, 2 * rows, curCols / 2, rows + row, j) = MultEval(party - 2, mult_keys2[k3++],
                                                                                        Arr2DIdx(drelu, rows, curCols / 2, row, j),
                                                                                        Arr2DIdx(tmpIdx, rows, curCols, row, 2 * j) - Arr2DIdx(tmpIdx, rows, curCols, row, 2 * j + 1));
                }
            }

            reconstruct((2 * rows) * (curCols / 2), mult_res, bitlength);

            for (int row = 0; row < rows; row++)
            {
                for (int j = 0; j < curCols / 2; ++j)
                {
                    Arr2DIdx(tmpMax, rows, curCols / 2, row, j) = Arr2DIdx(mult_res, 2 * rows, curCols / 2, row, j) + Arr2DIdx(tmpMax, rows, curCols, row, 2 * j + 1);
                    Arr2DIdx(tmpIdx, rows, curCols / 2, row, j) = Arr2DIdx(mult_res, 2 * rows, curCols / 2, rows + row, j) + Arr2DIdx(tmpIdx, rows, curCols, row, 2 * j + 1);
                }
                if (curCols % 2 == 1)
                {
                    Arr2DIdx(tmpMax, rows, curCols / 2, row, curCols / 2) = Arr2DIdx(tmpMax, 2 * rows, curCols, row, curCols - 1);
                    Arr2DIdx(tmpIdx, rows, curCols / 2, row, curCols / 2) = Arr2DIdx(tmpIdx, 2 * rows, curCols, row, curCols - 1);
                }
            }

            curCols = (curCols + 1) / 2;
        }

        for (int row = 0; row < rows; row++)
        {
            out[row] = Arr2DIdx(tmpIdx, rows, 1, row, 0);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        argmaxEvalMicroseconds += eval_time;
        evalMicroseconds += eval_time;
        std::cerr << "   Eval time: " << eval_time / 1000.0 << " milliseconds" << std::endl;
        delete[] tmpMax;
        delete[] tmpIdx;
        delete[] drelu;
        delete[] mult_res;
    }
    std::cerr << ">> ArgMax - End" << std::endl;
}

void AvgPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr))
{
    // taken from the equivalent function in Porthos/src/EzPCFunctionalities.cpp
    std::cerr << ">> AvgPool - Start" << std::endl;
    int rows = N * H * W * C;
    std::vector<GroupElement> filterAvg(rows, 0);
    std::vector<GroupElement> filterAvg_mask(rows, 0);
    std::vector<GroupElement> outp(rows), outp_mask(rows);

    auto common_start = std::chrono::high_resolution_clock::now();
    int rowIdx = 0;
    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            int32_t leftTopCornerH = -zPadHLeft;
            int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
            while ((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH)
            {
                int32_t leftTopCornerW = -zPadWLeft;
                int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
                while ((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW)
                {

                    GroupElement curFilterSum = 0, curFilterSum_mask = 0;
                    for (int fh = 0; fh < ksizeH; fh++)
                    {
                        for (int fw = 0; fw < ksizeW; fw++)
                        {
                            int32_t curPosH = leftTopCornerH + fh;
                            int32_t curPosW = leftTopCornerW + fw;

                            GroupElement temp = 0, temp_mask = 0;
                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                            {
                                temp = 0;
                                temp_mask = 0;
                            }
                            else
                            {
                                temp = Arr4DIdx(inArr, N, imgH, imgW, C, n, curPosH, curPosW, c);
                                temp_mask = Arr4DIdx(inArr_mask, N, imgH, imgW, C, n, curPosH, curPosW, c);
                            }

                            curFilterSum = curFilterSum + temp;
                            curFilterSum_mask = curFilterSum_mask + temp_mask;
                        }
                    }

                    // todo: put if-else everywhere when tasks are specific to dealer
                    filterAvg_mask[rowIdx] = curFilterSum_mask;
                    filterAvg[rowIdx] = curFilterSum;

                    rowIdx += 1;
                    leftTopCornerW = leftTopCornerW + strideW;
                }

                leftTopCornerH = leftTopCornerH + strideH;
            }
        }
    }
    auto common_end = std::chrono::high_resolution_clock::now();
    auto common_time = std::chrono::duration_cast<std::chrono::microseconds>(common_end - common_start).count();
    if (party == DEALER)
    {
        dealerMicroseconds += common_time;
        std::cerr << "   Dealer Time (without PubDiv) = " << common_time / 1000.0 << " miliseconds" << std::endl;
    }
    else
    {
        avgpoolEvalMicroseconds += common_time;
        evalMicroseconds += common_time;
        std::cerr << "   Eval Time (without PubDiv) = " << common_time / 1000.0 << " miliseconds" << std::endl;
    }

    // The division should always be signed division.
    // Local division will introduce error

    bool doLocal = false;

    if (doLocal)
    {
        // following what porthos does: convert to signed and then back to unsigned
        // todo: check why this double negative trick works to output signed division
        // todo: assuming 64 bitlen here
        for (int rowIdx = 0; rowIdx < rows; rowIdx++)
        {
            if (party == DEALER)
            {
                filterAvg_mask[rowIdx] = static_cast<uint64_t>((static_cast<int64_t>(filterAvg_mask[rowIdx])) / (ksizeH * ksizeW));
            }
            else
            {
                filterAvg[rowIdx] = -static_cast<uint64_t>((static_cast<int64_t>(-filterAvg[rowIdx])) / (ksizeH * ksizeW));
            }
        }
    }
    else
    {
        // call fss protocol for division
        // todo: the divisor ksizeH * ksizeW is 32 bits long when passed as param, but ezpc cleartext explicitly converts to 64 bit value
        // will this be an issue in the future?
        // ElemWiseVectorPublicDiv(rows, filterAvg.data(), filterAvg_mask.data(), ksizeH * ksizeW, outp.data(), outp_mask.data());
        std::cerr << "Error Error Error" << std::endl;
        exit(1);
    }

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                {
                    int iidx = n * C * H * W + c * H * W + h * W + w;
                    if (party == DEALER)
                    {
                        Arr4DIdx(outArr_mask, N, H, W, C, n, h, w, c) = outp_mask[iidx];
                    }
                    else
                    {
                        Arr4DIdx(outArr, N, H, W, C, n, h, w, c) = outp[iidx];
                    }
                }
            }
        }
    }
    std::cerr << ">> AvgPool - End" << std::endl;
}

void ElemWiseMul(int32_t size, GroupElement *inArr, GroupElement *multArrVec, GroupElement *outputArr, std::string prefix)
{
    if (party == DEALER)
    {
        pair<MultKey> *keys = new pair<MultKey>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            auto rout = random_ge(bitlength);
            keys[i] = MultGen(inArr[i], multArrVec[i], rout);
            outputArr[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_mult_key(keys[i].first);
            client->send_mult_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        MultKey *keys = new MultKey[size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for(int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_mult_key();
            } });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            {
#pragma omp parallel for
            for(int i = 0; i < size; ++i) {
                outputArr[i] = MultEval(party - SERVER, keys[i], inArr[i], multArrVec[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, outputArr, bitlength); });

        Llama::stat_t stat = {prefix + "ElemWiseMul", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot, std::string prefix)
{
    std::cerr << ">> MaxPool - Start" << std::endl;
    int d1 = ((imgH - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((imgW - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    always_assert(d1 == H);
    always_assert(d2 == W);
    always_assert(N1 == N);
    always_assert(C1 == C);

    GroupElement *maxUntilNow = outArr;
    GroupElement *maxUntilNow_mask = outArr_mask;

    if (party == DEALER)
    {
        uint64_t dealer_file_read_time = 0;
        auto dealer_start = std::chrono::high_resolution_clock::now();
        for (int fh = 0; fh < FH; fh++)
        {
            for (int fw = 0; fw < FW; fw++)
            {
                for (int n = 0; n < N; n++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        for (int ctH = 0; ctH < H; ctH++)
                        {
                            for (int ctW = 0; ctW < W; ctW++)
                            {
                                int leftTopCornerH = ctH * strideH - zPadHLeft;
                                int leftTopCornerW = ctW * strideW - zPadWLeft;

                                if (fh == 0 && fw == 0)
                                {
                                    if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW)
                                    {
                                        Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = GroupElement(0);
                                    }
                                    else
                                    {
                                        Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = Arr4DIdx(inArr_mask, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                                    }
                                }
                                else
                                {
                                    int curPosH = leftTopCornerH + fh;
                                    int curPosW = leftTopCornerW + fw;

                                    GroupElement maxi_mask = Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c);
                                    GroupElement temp_mask;
                                    if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                                    {
                                        temp_mask = GroupElement(0);
                                    }
                                    else
                                    {
                                        temp_mask = Arr4DIdx(inArr_mask, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
                                    }
                                    GroupElement rout = random_ge(bitlength);
                                    GroupElement routBit = random_ge(1);
                                    auto keys = keyGenMaxpool(bitlength, bitlength, maxi_mask, temp_mask, rout, routBit);
                                    Arr5DIdx(oneHot, FH * FW - 1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c) = routBit;
                                    Arr4DIdx(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = rout;

                                    auto read_start = std::chrono::high_resolution_clock::now();
                                    server->send_maxpool_key(keys.first);
                                    client->send_maxpool_key(keys.second);
                                    freeMaxpoolKeyPackPair(keys);
                                    auto read_end = std::chrono::high_resolution_clock::now();
                                    auto read_time = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
                                    dealer_file_read_time += read_time;
                                }
                            }
                        }
                    }
                }
            }
        }
        auto dealer_end = std::chrono::high_resolution_clock::now();
        auto dealer_time = std::chrono::duration_cast<std::chrono::microseconds>(dealer_end - dealer_start).count() - dealer_file_read_time;
        dealerMicroseconds += dealer_time;
        std::cerr << "   Dealer time: " << dealer_time / 1000.0 << " milliseconds" << std::endl;
    }
    else
    {
        MaxpoolKeyPack *keys = new MaxpoolKeyPack[(FH * FW - 1) * N * C * H * W];
        int kidx = 0;
        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int fh = 0; fh < FH; fh++)
        {
            for (int fw = 0; fw < FW; fw++)
            {
                if (fh == 0 && fw == 0)
                {
                    continue;
                }
                for (int n = 0; n < N; n++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        for (int ctH = 0; ctH < H; ctH++)
                        {
                            for (int ctW = 0; ctW < W; ctW++)
                            {
                                keys[kidx] = dealer->recv_maxpool_key(bitlength, bitlength);
                                kidx++;
                            }
                        }
                    }
                }
            }
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        uint64_t keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();

        peer->sync();
        uint64_t timeCompute = 0;
        uint64_t timeReconstruct = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                for (int ctH = 0; ctH < H; ctH++)
                {
                    for (int ctW = 0; ctW < W; ctW++)
                    {
                        int leftTopCornerH = ctH * strideH - zPadHLeft;
                        int leftTopCornerW = ctW * strideW - zPadWLeft;
                        if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW)
                        {
                            Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = 0;
                        }
                        else
                        {
                            Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = Arr4DIdx(inArr, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                        }
                    }
                }
            }
        }
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        auto t0 = std::chrono::high_resolution_clock::now();
        timeCompute += std::chrono::duration_cast<std::chrono::microseconds>(t0 - start).count();

        for (int fh = 0; fh < FH; fh++)
        {
            for (int fw = 0; fw < FW; fw++)
            {
                if (fh == 0 && fw == 0)
                {
                    continue;
                }

                auto t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
                for (int i = 0; i < N * C * H * W; i += 1)
                {
                    int curr = i;
                    int ctW = curr % W;
                    curr = curr / W;
                    int ctH = curr % H;
                    curr = curr / H;
                    int c = curr % C;
                    curr = curr / C;
                    int n = curr % N;
                    curr = curr / N;

                    int leftTopCornerH = ctH * strideH - zPadHLeft;
                    int leftTopCornerW = ctW * strideW - zPadWLeft;
                    int curPosH = leftTopCornerH + fh;
                    int curPosW = leftTopCornerW + fw;

                    GroupElement maxi = Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c);
                    GroupElement temp;
                    if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                    {
                        temp = GroupElement(0);
                    }
                    else
                    {
                        temp = Arr4DIdx(inArr, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
                    }
                    int kidx = (fh * FW + fw - 1) * (N * C * H * W) + i;
                    Arr4DIdx(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = evalMaxpool(party - 2, maxi, temp, keys[kidx], Arr5DIdx(oneHot, FH * FW - 1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c));
                    freeMaxpoolKeyPack(keys[kidx]);
                }

                auto t2 = std::chrono::high_resolution_clock::now();

                if (!(fh == 0 && fw == 0))
                {
                    reconstruct(N * C * H * W, maxUntilNow, bitlength);
                }
                auto t3 = std::chrono::high_resolution_clock::now();
                timeCompute += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
                timeReconstruct += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        reconstruct(N * C * H * W * (FH * FW - 1), oneHot, 1);
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        timeReconstruct += std::chrono::duration_cast<std::chrono::microseconds>(end - t4).count();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        delete[] keys;

        Llama::stat_t stat = {
            prefix + "MaxPool",
            keyread_time,
            timeCompute,
            timeReconstruct,
            onlineComm1 - onlineComm0,
            dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);
    }

    std::cerr << ">> MaxPool - End" << std::endl;
}

void reluHelper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *outArr, GroupElement *drelu, ReluKeyPack *keys)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        outArr[i] = evalRelu(party - 2, inArr[i], keys[i], &drelu[i]);
        freeReluKeyPack(keys[i]);
    }
    auto thread_end = std::chrono::high_resolution_clock::now();
}

void relu_dealer_threads_helper(int thread_idx, int32_t size, GroupElement *inArr_mask, GroupElement *outArr_mask, GroupElement *drelu, std::pair<ReluKeyPack, ReluKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        auto rout = random_ge(bitlength); // prng inside multithreads, need some locking
        drelu[i] = random_ge(1);
        keys[i] = keyGenRelu(bitlength, bitlength, inArr_mask[i], rout, drelu[i]);
        outArr_mask[i] = rout;
    }
}

void Relu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu, std::string prefix)
{
    if (party == DEALER)
    {

        pair<ReluKeyPack> *keys = new pair<ReluKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; i += 1)
        {
            auto rout = random_ge(bitlength);
            drelu[i] = random_ge(1);
            keys[i] = keyGenRelu(bitlength, bitlength, inArr_mask[i], rout, drelu[i]);
            outArr_mask[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_relu_key(keys[i].first);
            client->send_relu_key(keys[i].second);
            freeReluKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        ReluKeyPack *keys = new ReluKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for(int i = 0; i < size; i++){
                keys[i] = dealer->recv_relu_key(bitlength, bitlength);
            } });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            {
#pragma omp parallel for
            for(int i = 0; i < size; i++)
            {
                outArr[i] = evalRelu(party - 2, inArr[i], keys[i], &drelu[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         {
            reconstruct(size, outArr, bitlength);
            reconstruct(size, drelu, 1); });

        Llama::stat_t stat = {prefix + "ReLU-Spline", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

#pragma omp parallel for
        for (int i = 0; i < size; i++)
        {
            freeReluKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

#define BIG_LOOPY(e)                        \
    for (int n = 0; n < N; ++n)             \
    {                                       \
        for (int h = 0; h < H; ++h)         \
        {                                   \
            for (int w = 0; w < W; ++w)     \
            {                               \
                for (int c = 0; c < C; ++c) \
                {                           \
                    e;                      \
                }                           \
            }                               \
        }                                   \
    }

void maxpool_onehot_threads_helper(int thread_idx, int f, int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
                                   int32_t FW, GroupElement *maxBits, GroupElement *curr, GroupElement *oneHot, BitwiseAndKeyPack *keys)
{
    auto p = get_start_end(N * H * W * C, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        int curridx = i;
        int c = curridx % C;
        curridx = curridx / C;
        int w = curridx % W;
        curridx = curridx / W;
        int h = curridx % H;
        curridx = curridx / H;
        int n = curridx % N;
        curridx = curridx / N;

        auto max = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
        auto c1 = Arr4DIdx(curr, N, H, W, C, n, h, w, c);
        auto key = keys[(FH * FW - 2 - f) * N * H * W * C + n * H * W * C + h * W * C + w * C + c];
        Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = evalAnd(party - 2, max, 1 ^ c1, key);
        mod(Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c), 1);
    }
}

// maxBits contains all the comparison bits from maxpool and converts them to one hot
// For eg - in a filter of size 5, if the numbers where 3, 2, 5, 4, 7  MaxPool would have set the maxBits array to be 0, 1, 0, 1
// this functionality converts this to 0, 0, 0, 1 (retains the last 1 and makes the rest 0)
// This is compatible with both MaxPool and MaxPoolDouble
void MaxPoolOneHot(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH, int32_t FW, GroupElement *maxBits, GroupElement *oneHot)
{
    std::cerr << ">> MaxPoolOneHot - Start" << std::endl;
    GroupElement *curr = make_array<GroupElement>(N * H * W * C);
    if (party == DEALER)
    {
        BIG_LOOPY(
            auto m = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, FH * FW - 2, n, h, w, c);
            Arr4DIdx(curr, N, H, W, C, n, h, w, c) = m;
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, FH * FW - 1, n, h, w, c) = m;)

        for (int f = FH * FW - 2; f >= 1; --f)
        {
            // out[f] = max[f - 1] ^ !curr
            BIG_LOOPY(
                auto max = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
                auto c1 = Arr4DIdx(curr, N, H, W, C, n, h, w, c);
                auto rout = random_ge(1);
                auto keys = keyGenBitwiseAnd(max, c1, rout);
                server->send_bitwise_and_key(keys.first);
                client->send_bitwise_and_key(keys.second);
                Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = rout;)

            BIG_LOOPY(
                Arr4DIdx(curr, N, H, W, C, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c) ^ Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c);)
        }

        BIG_LOOPY(
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, 0, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c);)
    }
    else
    {
        BitwiseAndKeyPack *keys = new BitwiseAndKeyPack[(FH * FW - 2) * N * H * W * C];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < (FH * FW - 2) * N * H * W * C; ++i)
        {
            keys[i] = dealer->recv_bitwise_and_key();
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        BIG_LOOPY(
            auto m = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, FH * FW - 2, n, h, w, c);
            Arr4DIdx(curr, N, H, W, C, n, h, w, c) = m;
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, FH * FW - 1, n, h, w, c) = m;)

        for (int f = FH * FW - 2; f >= 1; --f)
        {

            // out[f] = max[f - 1] ^ !curr
            BIG_LOOPY(
                auto max = Arr5DIdx(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
                auto c1 = Arr4DIdx(curr, N, H, W, C, n, h, w, c);
                auto key = keys[(FH * FW - 2 - f) * N * H * W * C + n * H * W * C + h * W * C + w * C + c];
                Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = evalAnd(party - 2, max, 1 ^ c1, key);
                mod(Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c), 1);)

            // std::thread thread_pool[num_threads];
            // for(int i = 0; i < num_threads; ++i) {
            //     thread_pool[i] = std::thread(maxpool_onehot_threads_helper, i, f, N, H, W, C, FH, FW, maxBits, curr, oneHot, keys);
            // }

            // for(int i = 0; i < num_threads; ++i) {
            //     thread_pool[i].join();
            // }

            reconstruct(N * H * W * C, oneHot + f * N * H * W * C, 1);

            BIG_LOOPY(
                Arr4DIdx(curr, N, H, W, C, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c) ^ Arr5DIdx(oneHot, FH * FW, N, H, W, C, f, n, h, w, c);)
        }

        BIG_LOOPY(
            Arr5DIdx(oneHot, FH * FW, N, H, W, C, 0, n, h, w, c) = Arr4DIdx(curr, N, H, W, C, n, h, w, c) ^ 1;)
        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        evalMicroseconds += eval_time;
        selectEvalMicroseconds += eval_time;
        std::cerr << "   Key Read Time = " << keyread_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;
        delete[] keys;
    }
    delete[] curr;
    std::cerr << ">> MaxPoolOneHot - End" << std::endl;
}

void ConvTranspose3DWrapper(int64_t N,
                            int64_t D,
                            int64_t H,
                            int64_t W,
                            int64_t CI,
                            int64_t FD,
                            int64_t FH,
                            int64_t FW,
                            int64_t CO,
                            int64_t zPadDLeft,
                            int64_t zPadDRight,
                            int64_t zPadHLeft,
                            int64_t zPadHRight,
                            int64_t zPadWLeft,
                            int64_t zPadWRight,
                            int64_t strideD,
                            int64_t strideH,
                            int64_t strideW,
                            int64_t outD,
                            int64_t outH,
                            int64_t outW,
                            GroupElement *inputArr,
                            GroupElement *filterArr,
                            GroupElement *outArr)
{
    std::cerr << ">> ConvTranspose3D - Start" << std::endl;
    always_assert(outD == (D - 1) * strideD - zPadDLeft - zPadDRight + FD);
    always_assert(outH == (H - 1) * strideH - zPadHLeft - zPadHRight + FH);
    always_assert(outW == (W - 1) * strideW - zPadWLeft - zPadWRight + FW);

    if (party == DEALER)
    {
        auto local_start = std::chrono::high_resolution_clock::now();

        // not good for in place operations
        for (int i = 0; i < N * outD * outH * outW * CO; ++i)
        {
            outArr[i] = random_ge(bitlength);
        }

        auto keys = KeyGenConvTranspose3D(bitlength, N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, inputArr, filterArr, outArr);

        auto local_end = std::chrono::high_resolution_clock::now();

        client->send_triple_key(keys.second);
        freeTripleKey(keys.second);
        auto local_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                                                      local_start)
                                    .count();
        dealerMicroseconds += local_time_taken;
        std::cerr << "   Dealer Time = " << local_time_taken / 1000.0 << " milliseconds\n";
    }
    else
    {

        auto keyread_start = std::chrono::high_resolution_clock::now();
        auto key = dealer->recv_triple_key(bitlength, N * D * H * W * CI, CI * FD * FH * FW * CO, N * outD * outH * outW * CO);
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end - keyread_start).count();

        peer->sync();

        auto local_start = std::chrono::high_resolution_clock::now();

        EvalConvTranspose3D(party, key, N, D, H, W, CI, FD, FH, FW, CO,
                            zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight,
                            strideD, strideH, strideW, outD, outH, outW, inputArr, filterArr, outArr);

        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();

        reconstruct(N * outD * outH * outW * CO, outArr, bitlength);

        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        convOnlineComm += (onlineComm1 - onlineComm0);
        auto local_end = std::chrono::high_resolution_clock::now();

        freeTripleKey(key);
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - local_start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(local_end - t1).count();

        convEvalMicroseconds += (reconstruct_time + compute_time);
        evalMicroseconds += (reconstruct_time + compute_time);
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";
    }

    std::cerr << ">> ConvTranspose3D - End" << std::endl;
}

void sign_extend2_eval_threads_helper(int thread_idx, int32_t size, int bin, int bout, GroupElement *x, GroupElement *wrap, SignExtend2KeyPack *keys)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        GroupElement y = x[i] + (1ULL << (bin - 1));
        mod(y, bin);
        evalDCF(party - 2, &wrap[i], y, keys[i].dcfKey);
        wrap[i] = wrap[i] + keys[i].rw;
        mod(wrap[i], 1);
    }
    auto thread_end = std::chrono::high_resolution_clock::now();
}

void SignExtend2(int size, int bin, int bout, GroupElement *x, GroupElement *y)
{
    for (int i = 0; i < size; i += 1)
    {
        mod(x[i], bin);
    }
    std::cerr << ">> SignExtend2 - Start" << std::endl;
    if (party == DEALER)
    {
        uint64_t dealer_total_time = 0;
        std::pair<SignExtend2KeyPack, SignExtend2KeyPack> *keys = new std::pair<SignExtend2KeyPack, SignExtend2KeyPack>[size];
        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (int i = 0; i < size; i += 1)
        {
            auto rout = random_ge(bout); // prng inside multithreads, need some locking
            keys[i] = keyGenSignExtend2(bin, bout, x[i], rout);
            y[i] = rout;
        }
        auto end = std::chrono::high_resolution_clock::now();
        dealer_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for (int i = 0; i < size; ++i)
        {
            server->send_sign_extend2_key(keys[i].first, bin, bout);
            client->send_sign_extend2_key(keys[i].second, bin, bout);
            freeSignExtend2KeyPackPair(keys[i]);
        }
        delete[] keys;
        dealerMicroseconds += dealer_total_time;
        std::cerr << "   Dealer time = " << dealer_total_time / 1000.0 << " milliseconds" << std::endl;
    }
    else
    {
        // Step 1: Preprocessing Keys from Dealer
        SignExtend2KeyPack *keys = new SignExtend2KeyPack[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
        {
            keys[i] = dealer->recv_sign_extend2_key(bin, bout);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        peer->sync();
        GroupElement *wrap = new GroupElement[size];
        auto start = std::chrono::high_resolution_clock::now();
        if (num_threads == 1)
        {
            sign_extend2_eval_threads_helper(0, size, bin, bout, x, wrap, keys);
        }
        else
        {
            std::thread thread_pool[num_threads];
            for (int thread_idx = 0; thread_idx < num_threads; thread_idx++)
            {
                thread_pool[thread_idx] = std::thread(sign_extend2_eval_threads_helper, thread_idx, size, bin, bout, x, wrap, keys);
            }

            for (int thread_idx = 0; thread_idx < num_threads; thread_idx++)
            {
                thread_pool[thread_idx].join();
            }
        }

        auto mid = std::chrono::high_resolution_clock::now();
        // Step 3: Online Communication
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        reconstruct(size, wrap, 1);

        auto mid2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
        {
            GroupElement out = keys[i].p[wrap[i] % 2];
            if (party == 2)
            {
                GroupElement z = x[i] + (1ULL << (bin - 1));
                mod(z, bin);
                out = out + z;
            }
            y[i] = out;
            freeSignExtend2KeyPack(keys[i]);
        }
        auto mid3 = std::chrono::high_resolution_clock::now();

        reconstruct(size, y, bout);

        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        // reluOnlineComm += (onlineComm1 - onlineComm0);
        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count() + std::chrono::duration_cast<std::chrono::microseconds>(mid3 - mid2).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(mid2 - mid).count() + std::chrono::duration_cast<std::chrono::microseconds>(end - mid3).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        // reluEvalMicroseconds += (reconstruct_time + compute_time);
        delete[] keys;
        delete[] wrap;
    }
    std::cerr << ">> SignExtend2 - End" << std::endl;
}

// Masked version of Protocol of Probablistic Truncation over ZZ_{2^k} from ePrint 2020/338
// Here, we set \ell = k - 1
void EdabitsPrTrunc(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    if (party == DEALER)
    {
        pair<EdabitsPrTruncKeyPack> *keys = new pair<EdabitsPrTruncKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; i += 1)
        {
            auto rout = random_ge(bitlength);
            keys[i] = keyGenEdabitsPrTrunc(bitlength, scale, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_edabits_prtrunc_key(keys[i].first, bitlength);
            client->send_edabits_prtrunc_key(keys[i].second, bitlength);
        }
        delete[] keys;
    }
    else
    {
        EdabitsPrTruncKeyPack *keys = new EdabitsPrTruncKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for(int i = 0; i < size; i++){
                keys[i] = dealer->recv_edabits_prtrunc_key(bitlength);
            } });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            {
            if (party == SERVER)
            {
#pragma omp parallel for
                for(int i = 0; i < size; i++) {
                    GroupElement vb = keys[i].a;
                    GroupElement ip = x[i] + (1ULL << (bitlength - 2));
                    GroupElement msb = (ip >> (bitlength - 1)) % 2;
                    if (msb) {
                        vb = 1 - vb;
                    }
                    GroupElement t = keys[i].b + (1ULL << (bitlength - scale - 1)) * vb;
                    y[i] = t + ((ip % (1ULL << (bitlength - 1))) >> scale) - (1ULL << (bitlength - scale - 2));
                }
            }
            else
            {
#pragma omp parallel for
                for(int i = 0; i < size; i++) {
                    GroupElement vb = keys[i].a;
                    GroupElement ip = x[i] + (1ULL << (bitlength - 2));
                    GroupElement msb = (ip >> (bitlength - 1)) % 2;
                    if (msb) {
                        vb = -vb;
                    }
                    GroupElement t = keys[i].b + (1ULL << (bitlength - scale - 1)) * vb;
                    y[i] = t;
                }
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bitlength); });

        Llama::stat_t stat = {prefix + "EdabitsPrTrunc", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

void LUT_ss(int size, int bin, int bout, const std::vector<GroupElement> &tab, GroupElement *x, GroupElement *y, std::string prefix)
{
    always_assert(bin == 8);
    always_assert(tab.size() == (1LL << bin));

    if (party == DEALER)
    {
        pair<LUTSSKeyPack> *keys = new pair<LUTSSKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(bout);
            keys[i] = keyGenLUTSS(bin, bout, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_lutss_key(keys[i].first);
            client->send_lutss_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        LUTSSKeyPack *keys = new LUTSSKeyPack[size];
        GroupElement *tmp = new GroupElement[2 * size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_lutss_key(bin, bout);
            } });

        peer->sync();

        auto compute_time_1 = time_this_block([&]()
                                              {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                auto res = evalLUTSS_1(party - 2, x[i], tab, keys[i]);
                tmp[2*i] = res.first;
                tmp[2*i+1] = res.second;
            } });

        auto reconstruction_stats_1 = time_comm_this_block([&]()
                                                           { reconstruct(2 * size, tmp, bout); });

        auto compute_time_2 = time_this_block([&]()
                                              {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalLUTSS_2(party - 2, tmp[2*i], tmp[2*i+1], keys[i]);
            } });

        auto reconstruction_stats_2 = time_comm_this_block([&]()
                                                           { reconstruct(size, y, bout); });

        Llama::stat_t stat = {prefix + "LUTSS", keyread_time, compute_time_1 + compute_time_2, reconstruction_stats_1.first + reconstruction_stats_2.first, reconstruction_stats_1.second + reconstruction_stats_2.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
        delete[] tmp;
    }
}

void LUT_dpf(int size, int bin, int bout, const std::vector<GroupElement> &tab, GroupElement *x, GroupElement *y, std::string prefix, bool doReconstruct)
{
    if (bin >= 8)
    {
        return LUT_dfpet(size, bin, bout, tab, x, y, prefix, doReconstruct);
    }

    always_assert(false);

    if (bin == 8)
    {
        return LUT_ss(size, bin, bout, tab, x, y, prefix);
    }

    always_assert(tab.size() == (1LL << bin));

    if (party == DEALER)
    {
        pair<LUTKeyPack> *keys = new pair<LUTKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            mod(x[i], bin);
            GroupElement rout = random_ge(bout);
            keys[i] = keyGenLUT(bin, bout, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_lut_key(keys[i].first);
            client->send_lut_key(keys[i].second);
            freeLUTKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        LUTKeyPack *keys = new LUTKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_lut_key(bin, bout);
            } });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalAll_reduce(party - 2, keys[i].dpfKey, x[i], tab) + keys[i].rout;
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bout); });

        Llama::stat_t stat = {prefix + "LUT", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        for (int i = 0; i < size; i++)
        {
            freeLUTKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

void LUT_dfpet(int size, int bin, int bout, const std::vector<GroupElement> &tab, GroupElement *x, GroupElement *y, std::string prefix, bool doReconstruct)
{
    always_assert(bin >= 8);
    always_assert(tab.size() == (1LL << bin));
    GroupElement *tmp = new GroupElement[2 * size];
    GroupElement *res = tmp;
    GroupElement *corr = tmp + size;

    if (party == DEALER)
    {
        pair<LUTDPFETKeyPack> *keys = new pair<LUTDPFETKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement routRes = random_ge(bout);
            GroupElement routCorr = random_ge(1);
            keys[i] = keyGenLUTDPFET(bin, bout, x[i], routRes, routCorr);
            res[i] = routRes;
            corr[i] = routCorr;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_lutdpfet_key(keys[i].first);
            client->send_lutdpfet_key(keys[i].second);
            freeLUTDPFETKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        LUTDPFETKeyPack *keys = new LUTDPFETKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_lutdpfet_key(bin, bout);
            } });

        peer->sync();

        auto compute_time_1 = time_this_block([&]()
                                              {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                auto res_corr = evalLUTDPFET_1(party - 2, x[i], tab, keys[i]);
                res[i] = res_corr.first;
                corr[i] = res_corr.second;
            } });

        auto reconstruction_stats_1 = time_comm_this_block([&]()
                                                           {
            // reconstruct(size, res, bout);
            // reconstruct(size, corr, 1);
            reconstructRT(size, res, bout); });

        Llama::stat_t stat = {prefix + "LUT::FirstRound", keyread_time, compute_time_1, reconstruction_stats_1.first, reconstruction_stats_1.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            freeLUTDPFETKeyPack(keys[i]);
        }

        delete[] keys;
    }

    Select(size, bout, corr, res, y, prefix + "LUT::", doReconstruct);

    auto t = time_this_block([&]()
                             {
    if (doReconstruct || party == DEALER || party == SERVER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            y[i] = 2 * y[i] - res[i];
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            y[i] = 2 * y[i];
        }
    } });

    if (party != DEALER)
    {
        Llama::stat_t stat = {prefix + "LUT::Misc", 0, t, 0, 0, 0};
        stat.print();
        Llama::push_stats(stat);
    }

    delete[] tmp;
}

void nExp_SIRNN(int size, int bin, GroupElement *x, GroupElement *y, int scale)
{
    SlothClip(size, bin, 16, bitlength, x, y, "nExp::");

    GroupElement *x0 = new GroupElement[size];
    GroupElement *x1 = new GroupElement[size];

    // ARS(size, y, y, x1, x1, 8);
    TruncateReduce(size, 16, y, x1, 8, "nExp::");

    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        x0[i] = y[i] % (1LL<<8);
        mod(x1[i], 8);
    } });

    std::vector<GroupElement> lut_0(1LL << 8);
    std::vector<GroupElement> lut_1(1LL << 8);
    for (int i = 0; i < (1LL << 8); ++i)
    {
        lut_0[i] = GroupElement(std::exp(-i / double(1LL << scale)) * (1LL << scale));
        lut_1[i] = GroupElement(std::exp(-i / double(1LL << (scale - 8))) * (1LL << scale));
    }

    LUT_dpf(size, 8, LlamaConfig::bitlength, lut_0, x0, x0, "nExp::");
    LUT_dpf(size, 8, LlamaConfig::bitlength, lut_1, x1, x1, "nExp::");

    ElemWiseMul(size, x0, x1, y, "nExp::");
    SlothARS(size, y, y, scale, "nExp::");

    if (party != DEALER)
    {
        Llama::push_stats({"nExp::Misc", 0, t1, 0, 0, 0});
    }

    delete[] x0;
    delete[] x1;
}

void nExp(int size, int bin, GroupElement *x, GroupElement *y, int scale)
{
    return nExp_SIRNN(size, bin, x, y, scale);
    int lutBw = 16;
    always_assert(bin >= lutBw);

    std::vector<GroupElement> lut(1LL << lutBw);
    for (int i = 0; i < (1LL << lutBw); ++i)
    {
        lut[i] = GroupElement(std::exp(-i / double(1LL << scale)) * (1LL << scale));
    }

    SlothClip(size, bin, lutBw, bitlength, x, y);
    LUT_dpf(size, lutBw, LlamaConfig::bitlength, lut, y, y, "nExp::");
}

void Tanh(int size, GroupElement *x, GroupElement *y, int scale)
{
    int lutBw = 14;
    always_assert(LlamaConfig::bitlength >= lutBw);

    GroupElement *abs = new GroupElement[size];
    GroupElement *drelu = new GroupElement[size];
    // Relu2Round(size, x, x, abs, abs, drelu, LlamaConfig::bitlength);
    Relu(size, x, x, abs, abs, drelu, "Tanh::");

#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        abs[i] = abs[i] * 2 - x[i];
        mod(abs[i], LlamaConfig::bitlength);
    }

    std::vector<GroupElement> lut(1LL << lutBw);
    for (int i = 0; i < (1LL << lutBw); ++i)
    {
        lut[i] = GroupElement(std::tanh(i / double(1LL << scale)) * (1LL << scale));
    }

    SlothClip(size, bitlength, lutBw, bitlength, abs, abs);
    LUT_dpf(size, lutBw, LlamaConfig::bitlength, lut, abs, abs, "Tanh::");
    Select(size, drelu, abs, y);

#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        y[i] = y[i] * 2 - abs[i];
    }

    delete[] abs;
}

// unused
void Clip(int size, int maxbw, GroupElement *x, GroupElement *y, std::string prefix)
{
    if (party == DEALER)
    {
        pair<ClipKeyPack> *keys = new pair<ClipKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(bitlength);
            keys[i] = keyGenClip(bitlength, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_clip_key(keys[i].first);
            client->send_clip_key(keys[i].second);
            freeClipKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        ClipKeyPack *keys = new ClipKeyPack[size];
        GroupElement *tmp = new GroupElement[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_clip_key(bitlength);
            } });

        peer->sync();

        uint64_t compute_time_1 = time_this_block([&]()
                                                  {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                tmp[i] = evalClip_1(party - 2, maxbw, x[i], keys[i]);
            } });

        auto reconstruction_stats_1 = time_comm_this_block([&]()
                                                           { reconstruct(size, tmp, 1); });

        uint64_t compute_time_2 = time_this_block([&]()
                                                  {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalClip_2(party - 2, maxbw, tmp[i], x[i], keys[i]);
            } });

        auto reconstruction_stats_2 = time_comm_this_block([&]()
                                                           { reconstruct(size, y, bitlength); });

        Llama::stat_t stat = {
            prefix + "Clip",
            keyread_time,
            compute_time_1 + compute_time_2,
            reconstruction_stats_1.first + reconstruction_stats_2.first,
            reconstruction_stats_1.second + reconstruction_stats_2.second,
            dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        for (int i = 0; i < size; ++i)
        {
            freeClipKeyPack(keys[i]);
        }
        delete[] keys;
        delete[] tmp;
    }
}

void Select(int32_t size, int bin, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix, bool doReconstruct)
{

    if (party == DEALER)
    {
        pair<SelectKeyPack> *keys = new pair<SelectKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            auto rout = random_ge(bin);
            keys[i] = keyGenSelect(bin, s[i], x[i], rout);
            out[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_select_key(keys[i].first);
            client->send_select_key(keys[i].second);
        }
    }
    else
    {
        SelectKeyPack *keys = new SelectKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for(int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_select_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for(int i = 0; i < size; ++i) {
                out[i] = evalSelect(party - 2, s[i], x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         {
            if (doReconstruct)
                reconstruct(size, out, bin); });

        Llama::stat_t stat = {
            prefix + "Select",
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

void Select(int32_t size, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix, bool doReconstruct)
{
    Select(size, bitlength, s, x, out, prefix, doReconstruct);
}

void InverseLUT(int size, GroupElement *x, GroupElement *y, int scale, int bw, std::string prefix = "")
{

    // for (int i = 0; i < size; ++i)
    // {
    //     y[i] = x[i] >> (bw - 16);
    // }
    // ARS(size, x, x, y, y, bw - 16);
    TruncateReduce(size, bw, x, y, bw - 16, prefix + "Inverse::");

    std::vector<GroupElement> lut(1LL << 16);
    for (int i = 1; i < (1LL << 16); ++i)
    {
        lut[i] = GroupElement(double(1LL << (scale + scale - bw + 16)) / i);
    }

    LUT_dpf(size, 16, LlamaConfig::bitlength, lut, y, y, prefix + "Inverse::");
}

void Softmax(int32_t s1, int32_t s2, int bin, GroupElement *x, GroupElement *y, int32_t scale)
{
    // s1 = batch size
    // s2 = number of classes

    GroupElement *max = make_array<GroupElement>(s1);
    // step 1 - calculate max for each image in batch
    // GroupElement *oneHot = make_array<GroupElement>(s1 * (s2 - 1));
    // MaxPool(s1, 1, 1, 1, s2, 1, 0, 0, 0, 0, 1, 1, s1, s2, 1, 1, x, x, max, max, oneHot);
    // delete[] oneHot; // TODO: support passing oneHot as nullptr

    SlothMaxpool(s1, s2, bin, x, max, "Softmax::");

    // step 2 - subtract max from each element in each image in batch
    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for(int i = 0; i < s1; ++i) {
        for(int j = 0; j < s2; ++j) {
            Arr2DIdx(y, s1, s2, i, j) = max[i] - Arr2DIdx(x, s1, s2, i, j);
        }
    } });

    // step 3 - exponentiate each element in each image in batch
    nExp(s1 * s2, std::min(bin + 1, bitlength), y, y, scale);

    GroupElement *denominators = max; // reuse the array
    // // step 4 - calculate sum of exponentiated elements for each image in batch
    auto t2 = time_this_block([&]()
                              {
#pragma omp parallel for
    for(int i = 0; i < s1; ++i) {
        denominators[i] = 0;
        for(int j = 0; j < s2; ++j) {
            denominators[i] = denominators[i] + Arr2DIdx(y, s1, s2, i, j);
        }
    } });

    // step 5 - calculate inverse of all the denominators
    InverseLUT(s1, denominators, denominators, scale, scale + 10, "Softmax::"); // only works if numClasses <= 1024

    // step 6 - multiply each element in each image in batch by the inverse of the denominator
    GroupElement *expandedDenominator = make_array<GroupElement>(s1 * s2);
    auto t3 = time_this_block([&]()
                              {
#pragma omp parallel for
    for(int i = 0; i < s1; ++i) {
        for(int j = 0; j < s2; ++j) {
            Arr2DIdx(expandedDenominator, s1, s2, i, j) = denominators[i];
        }
    } });
    delete[] max;

    ElemWiseMul(s1 * s2, expandedDenominator, y, y, "Softmax::");
    SlothARS(s1 * s2, y, y, scale, "Softmax::");

    if (party != DEALER)
        Llama::push_stats({"Softmax::Misc", 0, t1 + t2 + t3, 0, 0, 0});

    delete[] expandedDenominator;
}

void F2BF16(int size, GroupElement *x, GroupElement *y, std::string prefix)
{
    // assuming z = 0 and s = 0
    // y is 13 bit

    if (party == DEALER)
    {
        pair<F2BF16KeyPack> *keys = new pair<F2BF16KeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(13);
            keys[i] = keyGenF2BF16(bitlength, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_f2bf16_key(keys[i].first);
            client->send_f2bf16_key(keys[i].second);
            freeF2BF16KeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        F2BF16KeyPack *keys = new F2BF16KeyPack[size];
        GroupElement *tmp = new GroupElement[2 * size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_f2bf16_key(bitlength);
            } });

        peer->sync();

        uint64_t compute_time_1 = time_this_block([&]()
                                                  {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                auto p = evalF2BF16_1(party - 2, x[i], keys[i]);
                tmp[2*i] = p.first;
                tmp[2*i + 1] = p.second;
            } });

        auto reconstruction_stats_1 = time_comm_this_block([&]()
                                                           { reconstruct(2 * size, tmp, bitlength); });

        uint64_t compute_time_2 = time_this_block([&]()
                                                  {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalF2BF16_2(party - 2, x[i], tmp[2*i], tmp[2*i+1], keys[i]);
            } });

        auto reconstruction_stats_2 = time_comm_this_block([&]()
                                                           { reconstruct(size, y, bitlength); });

        uint64_t compute_time_3 = time_this_block([&]()
                                                  {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalF2BF16_3(party - 2, tmp[2*i], y[i], keys[i]);
            } });

        auto reconstruction_stats_3 = time_comm_this_block([&]()
                                                           { reconstruct(size, y, 13); });

        Llama::stat_t stat = {
            prefix + "F2BF16",
            keyread_time,
            compute_time_1 + compute_time_2 + compute_time_3,
            reconstruction_stats_1.first + reconstruction_stats_2.first + reconstruction_stats_3.first,
            reconstruction_stats_1.second + reconstruction_stats_2.second + reconstruction_stats_3.second,
            dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        for (int i = 0; i < size; ++i)
        {
            freeF2BF16KeyPack(keys[i]);
        }
        delete[] keys;
        delete[] tmp;
    }
}

void Rsqrt(int size, GroupElement *x, GroupElement *y, GroupElement extradiv, int scale, std::string prefix, std::vector<GroupElement> *lut)
{
    // printf("Inp=%lu\n", x[0]);
    // assumes x is in precision = 2*scale and have to output in precision = scale
    F2BF16(size, x, y, prefix + "Rsqrt::");

    // std::vector<GroupElement> lut(1LL<<13);
    if (!lut)
    {
        lut = new std::vector<GroupElement>(1LL << 13);
        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
    for(int i = 0; i < (1LL<<13); ++i)
    {
        GroupElement k = i % (1LL << 6);
        GroupElement m = i >> 6;
        double val = double(m+128) * std::pow(2.0, k-7);
        (*lut)[i] = GroupElement(double(1LL<<(2*scale)) / sqrt(val / extradiv));
    } });
    }
    // if (party != DEALER)
    // {
    //     // Llama::stat_t stat = { "Rsqrt::LutGen", 0, t, 0, 0, 0 };
    //     // stat.print();
    //     // Llama::push_stats(stat);
    // }

    LUT_dpf(size, 13, bitlength, *lut, y, y, prefix + "Rsqrt::");
    // printf("Op=%lu\n", y[0]);
}

inline double relu_sub_gelu(double x)
{
    double g = 0.5 * x * (1 + erf(x / sqrt(2.0)));
    return std::max(0.0, x) - g;
}

inline GroupElement relu_sub_gelu(GroupElement x, int scale_in, int scale_out)
{
    return (GroupElement)(relu_sub_gelu((double)x / (1LL << scale_in)) * (1LL << scale_out));
}

void Gelu(int size, int bin, GroupElement *x, GroupElement *y, int scale)
{
    GroupElement *drelu = new GroupElement[size];
    // Relu(size, x, x, y, y, drelu, "GeLU::");
    SlothRelu(size, bin, x, y, "GeLU::");

    GroupElement *abs = drelu;

#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        abs[i] = 2 * y[i] - x[i];
        mod(abs[i], bitlength);
    }
    SlothClip(size, bin, scale + 2, bitlength, abs, abs, "GeLU::");

    SlothTR(size, scale + 2, abs, abs, scale - 6, "GeLU::");

    std::vector<GroupElement> lut(1LL << 8);
    for (int i = 0; i < (1LL << 8); ++i)
    {
        lut[i] = relu_sub_gelu(i, 6, scale);
    }
    LUT_dpf(size, 8, bitlength, lut, abs, abs, "GeLU::");

#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        y[i] = y[i] - abs[i];
    }

    delete[] drelu;
}

void SlothGelu(int size, int bin, GroupElement *x, GroupElement *out, int scale)
{
    always_assert(scale == 12);

    GroupElement *y = new GroupElement[size];
    GroupElement *d = new GroupElement[size];
    GroupElement *rp = new GroupElement[size];
    GroupElement *abs = new GroupElement[size];
    GroupElement *r = new GroupElement[size];

    SlothTR(size, bin, x, y, 6, "GeLU::");
    SlothDrelu(size, bin - 6, y, d, "GeLU::");

    Select(size, bin - 6, d, y, rp, "GeLU::");

    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        abs[i] = 2 * rp[i] - y[i];
        mod(abs[i], bin - 6);
    } });

    SlothClip(size, bin - 6, 8, 8, abs, abs, "GeLU::");

    std::vector<GroupElement> lut(1LL << 8);
    auto t2 = time_this_block([&]()
                              {
    for(int i = 0; i < (1LL<<8); ++i)
    {
        lut[i] = relu_sub_gelu(i, 6, scale);
    } });

    LUT_dpf(size, 8, bitlength, lut, abs, abs, "GeLU::", false);

    Select(size, bitlength, d, x, r, "GeLU::", false);

    auto t3 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        out[i] = r[i] - abs[i];
    } });

    if (party != DEALER)
    {
        Llama::push_stats({"GeLU::Misc", 0, t1 + t3, 0, 0, 0});
        // Llama::push_stats({ "GeLU::LutGen", 0, t2, 0, 0, 0 });
        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, out, bitlength); });

        Llama::stat_t stat = {
            "GeLU::Reconstruct",
            0,
            0,
            reconstruction_stats.first,
            reconstruction_stats.second,
            0,
        };
        stat.print();
        Llama::push_stats(stat);
    }
}

inline double relu_sub_silu(double x)
{
    double g = x / (1 + exp(-x));
    return std::max(0.0, x) - g;
}

inline GroupElement relu_sub_silu(GroupElement x, int scale_in, int scale_out)
{
    return (GroupElement)(relu_sub_silu((double)x / (1LL << scale_in)) * (1LL << scale_out));
}

void SlothSilu(int size, int bin, GroupElement *x, GroupElement *out, int scale)
{
    always_assert(scale == 12);

    GroupElement *y = new GroupElement[size];
    GroupElement *d = new GroupElement[size];
    GroupElement *rp = new GroupElement[size];
    GroupElement *abs = new GroupElement[size];
    GroupElement *r = new GroupElement[size];

    SlothTR(size, bin, x, y, 6, "SiLU::");
    SlothDrelu(size, bin - 6, y, d, "SiLU::");

    Select(size, bin - 6, d, y, rp, "SiLU::");

    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        abs[i] = 2 * rp[i] - y[i];
        mod(abs[i], bin - 6);
    } });

    SlothClip(size, bin - 6, 10, 10, abs, abs, "SiLU::");

    std::vector<GroupElement> lut(1LL << 10);
    auto t2 = time_this_block([&]()
                              {
    for(int i = 0; i < (1LL<<10); ++i)
    {
        lut[i] = relu_sub_silu(i, 6, scale);
    } });

    LUT_dpf(size, 10, bitlength, lut, abs, abs, "SiLU::", false);

    Select(size, bitlength, d, x, r, "SiLU::", false);

    auto t3 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        out[i] = r[i] - abs[i];
    } });

    if (party != DEALER)
    {
        Llama::push_stats({"SiLU::Misc", 0, t1 + t3, 0, 0, 0});
        // Llama::push_stats({ "GeLU::LutGen", 0, t2, 0, 0, 0 });
        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, out, bitlength); });

        Llama::stat_t stat = {
            "SiLU::Reconstruct",
            0,
            0,
            reconstruction_stats.first,
            reconstruction_stats.second,
            0,
        };
        stat.print();
        Llama::push_stats(stat);
    }
}

void TruncateReduce(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    if (party == DEALER)
    {
        pair<TruncateReduceKeyPack> *keys = new pair<TruncateReduceKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            mod(x[i], bin);
            GroupElement rout = random_ge(bin - scale);
            keys[i] = keyGenTruncateReduce(bin, scale, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_truncate_reduce_key(keys[i].first);
            client->send_truncate_reduce_key(keys[i].second);
            freeTruncateReduceKeyPackPair(keys[i]);
        }
    }
    else
    {
        TruncateReduceKeyPack *keys = new TruncateReduceKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_truncate_reduce_key(bin, scale);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                mod(x[i], bin);
                y[i] = evalTruncateReduce(party - 2, x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bin - scale); });

        Llama::stat_t stat = {
            prefix + "TruncateReduce",
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        Llama::push_stats(stat);

        for (int i = 0; i < size; ++i)
        {
            freeTruncateReduceKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

void SlothDrelu(int size, int bin, GroupElement *x, GroupElement *y, std::string prefix)
{
    if (party == DEALER)
    {
        pair<SlothDreluKeyPack> *keys = new pair<SlothDreluKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenSlothDrelu(bin, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_drelu_key(keys[i].first);
            client->send_sloth_drelu_key(keys[i].second);
            freeSlothDreluKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        SlothDreluKeyPack *keys = new SlothDreluKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_slothdrelu_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothDrelu(party - 2, x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, 1); });

        Llama::stat_t stat = {
            prefix + "Drelu",
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        Llama::push_stats(stat);

        for (int i = 0; i < size; ++i)
        {
            freeSlothDreluKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

void SlothRelu(int size, int bin, GroupElement *x, GroupElement *y, std::string prefix)
{
    GroupElement *drelu = new GroupElement[size];
    SlothDrelu(size, bin, x, drelu, prefix + "Relu::");
    Select(size, drelu, x, y, prefix + "Relu::");
    delete[] drelu;
}

// Guaranteed to work only when x is positive
void SlothClip(int size, int bin, int maxbw, int bout, GroupElement *x, GroupElement *y, std::string prefix)
{
    GroupElement *tmp = new GroupElement[size];
    GroupElement *drelu = new GroupElement[size];

    auto t1 = time_this_block([&]()
                              {
    if (party != DEALER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            tmp[i] = x[i] - (1LL << maxbw);
        }
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            tmp[i] = x[i];
        }
    } });
    SlothDrelu(size, bin, tmp, drelu, prefix + "Clip::");

    auto t2 = time_this_block([&]()
                              {
    if (party != DEALER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            drelu[i] ^= 1;
            tmp[i] = tmp[i] + 1;
        }
    } });

    Select(size, bout, drelu, tmp, y, prefix + "Clip::");

    auto t3 = time_this_block([&]()
                              {
    if (party != DEALER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            y[i] = y[i] + (1LL << maxbw) - 1;
        }
    } });

    delete[] tmp;
    delete[] drelu;
    if (party != DEALER)
        Llama::push_stats({prefix + "Clip::Misc", 0, t1 + t2 + t3, 0, 0, 0});
}

void SlothMax(int size, int bin, GroupElement *x, GroupElement *y, GroupElement *out, std::string prefix)
{
    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        out[i] = x[i] - y[i];
    } });
    SlothRelu(size, bin, out, out, prefix + "Max::");
    auto t2 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        out[i] = out[i] + y[i];
    } });

    if (party != DEALER)
        Llama::push_stats({prefix + "Max::Misc", 0, t1 + t2, 0, 0, 0});
}

// x is [s1 x s2]
// y is [s1]
void SlothMaxpool(int s1, int s2, int bin, GroupElement *x, GroupElement *y, std::string prefix)
{
    GroupElement *left = new GroupElement[s1 * s2];  // more elements than required but whatever
    GroupElement *right = new GroupElement[s1 * s2]; // more elements than required but whatever
    GroupElement *res = new GroupElement[s1 * s2];
    GroupElement *tmp = new GroupElement[s1];

    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < s1 * s2; ++i)
    {
        res[i] = x[i];
    } });

    // do in log rounds
    int curr = s2;
    while (curr != 1)
    {
        int curr2 = curr / 2;

        auto t2 = time_this_block([&]()
                                  {
#pragma omp parallel for
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < curr2; ++j)
            {
                Arr2DIdx(left, s1, curr2, i, j) = Arr2DIdx(res, s1, curr, i, 2 * j);
                Arr2DIdx(right, s1, curr2, i, j) = Arr2DIdx(res, s1, curr, i, 2 * j + 1);
            }
        } });

        SlothMax(s1 * curr2, bin, left, right, left, prefix + "Maxpool::");

        int currNext;
        auto t3 = time_this_block([&]()
                                  {
        if ((curr % 2) == 0)
        {
            currNext = curr / 2;
        }
        else
        {
            currNext = curr / 2 + 1;
#pragma omp parallel for
            for (int i = 0; i < s1; ++i)
            {
                tmp[i] = Arr2DIdx(res, s1, curr, i, curr - 1);
            }
#pragma omp parallel for
            for (int i = 0; i < s1; ++i)
            {
                Arr2DIdx(res, s1, currNext, i, currNext - 1) = tmp[i];
            }
        }

#pragma omp parallel for
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < curr2; ++j)
            {
                Arr2DIdx(res, s1, currNext, i, j) = Arr2DIdx(left, s1, curr2, i, j);
            }
        }
        curr = currNext; });

        if (party != DEALER)
            Llama::push_stats({prefix + "Maxpool::Misc", 0, t2 + t3, 0, 0, 0});
    }

    auto t4 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < s1; ++i)
    {
        y[i] = Arr2DIdx(res, s1, 1, i, 0);
    } });

    if (party != DEALER)
        Llama::push_stats({prefix + "Maxpool::Misc", 0, t1 + t4, 0, 0, 0});

    delete[] left;
    delete[] right;
    delete[] res;
    delete[] tmp;
}

void SlothMaxpoolTriangular(int s1, int s2, int bin, GroupElement *x, GroupElement *y, std::string prefix)
{
    always_assert(s1 == s2);
    GroupElement *left = new GroupElement[s1 * s2];  // more elements than required but whatever
    GroupElement *right = new GroupElement[s1 * s2]; // more elements than required but whatever
    GroupElement *res = new GroupElement[s1 * s2];
    int curr[s1];
    int currNext[s1];

    auto t1 = time_this_block([&]()
                              {
    int idx = 0;
    for (int i = 0; i < s1; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            res[idx] = x[s1*i + j];
            ++idx;
        }
    }

    // do in log rounds
    for (int i = 0; i < s1; ++i)
    {
        curr[i] = i + 1;
    } });

    while (curr[s1 - 1] != 1)
    {
        int idx = 0;
        int offset = 0;
        auto t2 = time_this_block([&]()
                                  {
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < curr[i] / 2; ++j)
            {
                left[idx] =  res[offset + 2 * j];
                right[idx] = res[offset + 2 * j + 1];
                ++idx;
            }
            offset += curr[i];
        } });

        SlothMax(idx, bin, left, right, left, prefix + "Maxpool::");

        int offsetOld = 0;
        auto t3 = time_this_block([&]()
                                  {
        idx = 0;
        offset = 0;
        for (int i = 0; i < s1; ++i)
        {
            if ((curr[i] % 2) == 0)
            {
                currNext[i] = curr[i] / 2;
                for (int j = 0; j < curr[i] / 2; ++j)
                {
                    right[offset + j] = left[idx];
                    ++idx;
                }
            }
            else
            {
                currNext[i] = curr[i] / 2 + 1;
                for (int j = 0; j < curr[i] / 2; ++j)
                {
                    right[offset + j] = left[idx];
                    ++idx;
                }
                right[offset + currNext[i] - 1] = res[offsetOld + curr[i] - 1];

            }
            offset += currNext[i];
            offsetOld += curr[i];
        }

        idx = 0;
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < currNext[i]; ++j)
            {
                res[idx] = right[idx];
                ++idx;
            }
        }

        for (int i = 0; i < s1; ++i)
        {
            curr[i] = currNext[i];
        } });

        if (party != DEALER)
            Llama::push_stats({prefix + "Maxpool::Misc", 0, t2 + t3, 0, 0, 0});
    }

    auto t4 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < s1; ++i)
    {
        y[i] = res[i];
    } });

    if (party != DEALER)
        Llama::push_stats({prefix + "Maxpool::Misc", 0, t1 + t4, 0, 0, 0});

    delete[] left;
    delete[] right;
    delete[] res;
}

void SumOfSquare(int s1, int s2, GroupElement *x, GroupElement *y, std::string prefix)
{
    int size = s1 * s2;
    GroupElement *squares = new GroupElement[size];
    if (party == DEALER)
    {
        pair<SquareKey> *keys = new pair<SquareKey>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            auto rout = random_ge(bitlength);
            keys[i] = keyGenSquare(x[i], rout);
            squares[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_square_key(keys[i].first);
            client->send_square_key(keys[i].second);
        }

        delete[] keys;

#pragma omp parallel for
        for (int i = 0; i < s1; ++i)
        {
            y[i] = 0;
            for (int j = 0; j < s2; ++j)
            {
                y[i] = y[i] + squares[i * s2 + j];
            }
        }
    }
    else
    {
        SquareKey *keys = new SquareKey[size];

        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            {
            for(int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_square_key();
            } });

        peer->sync();

        auto compute_time = time_this_block([&]()
                                            {
#pragma omp parallel for
            for(int i = 0; i < size; ++i) {
                squares[i] = evalSquare(party - SERVER, x[i], keys[i]);
            }

#pragma omp parallel for
            for (int i = 0; i < s1; ++i)
            {
                y[i] = 0;
                for (int j = 0; j < s2; ++j)
                {
                    y[i] = y[i] + squares[i * s2 + j];
                }
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(s1, y, bitlength); });

        Llama::stat_t stat = {prefix + "SumOfSquare", keyread_time, compute_time, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

GroupElement log2(GroupElement x)
{
    GroupElement y = 0;
    while (x >>= 1)
        y++;
    return y;
}

void SlothDiv(int size, GroupElement *x, GroupElement *y, GroupElement divisor, int scale, std::string prefix = "")
{
    if (!(divisor & (divisor - 1)))
    {
        SlothARS(size, x, y, log2(divisor), prefix + "Div::");
    }
    else
    {
        GroupElement divfp = (1LL << scale) / divisor;

        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            y[i] = x[i] * divfp;
        } });

        Llama::stat_t stat = {prefix + "Div::Misc", 0, t, 0, 0, 0};
        stat.print();
        Llama::push_stats(stat);

        SlothARS(size, y, y, scale, prefix + "Div::");
    }
}

void SlothDivFaithful(int size, GroupElement *x, GroupElement *y, GroupElement divisor, int scale, std::string prefix = "")
{
    if (!(divisor & (divisor - 1)))
    {
        SlothFaithfulARS(size, LlamaConfig::bitlength, x, y, log2(divisor), prefix + "Div::");
    }
    else
    {
        GroupElement divfp = (1LL << scale) / divisor;

        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; i++) {
            y[i] = x[i] * divfp;
        } });

        Llama::stat_t stat = {prefix + "Div::Misc", 0, t, 0, 0, 0};
        stat.print();
        Llama::push_stats(stat);

        SlothFaithfulARS(size, LlamaConfig::bitlength, y, y, scale, prefix + "Div::");
    }
}

void SlothLayerNorm(int s1, int s2, GroupElement *x, GroupElement *A, GroupElement *B, GroupElement *y, int scale)
{
    GroupElement *mean = new GroupElement[s1];

    auto t1 = time_this_block([&]()
                              {
#pragma omp parallel for
    for (int i = 0; i < s1; ++i) {
        mean[i] = 0;
        for (int j = 0; j < s2; j++) {
            mean[i] += x[i * s2 + j];
        }
    } });

    SlothDivFaithful(s1, mean, mean, s2, scale, "LayerNorm::");

    GroupElement *tmp = new GroupElement[s1 * s2];

    auto t2 = time_this_block([&]()
                              {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s2; j++) {
            tmp[i * s2 + j] = x[i * s2 + j] - mean[i];
        }
    } });

    GroupElement *var = new GroupElement[s1];
    SumOfSquare(s1, s2, tmp, var, "LayerNorm::");

    Rsqrt(s1, var, var, s2, scale, "LayerNorm::");

    auto t3 = time_this_block([&]()
                              {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s2; j++) {
            y[i * s2 + j] = var[i];
        }
    } });

    ElemWiseMul(s1 * s2, tmp, y, y, "LayerNorm::");
    SlothARS(s1 * s2, y, y, scale, "LayerNorm::");

    GroupElement *Aexpand = tmp;
    auto t4 = time_this_block([&]()
                              {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s2; j++) {
            Aexpand[i * s2 + j] = A[j];
        }
    } });

    ElemWiseMul(s1 * s2, Aexpand, y, y, "LayerNorm::");

    auto t5 = time_this_block([&]()
                              {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s2; j++) {
            y[i * s2 + j] += B[j];
        }
    } });

    SlothARS(s1 * s2, y, y, scale, "LayerNorm::");

    Llama::stat_t stat = {"LayerNorm::Misc", 0, t1 + t2 + t3 + t4 + t5, 0, 0, 0};
    stat.print();
    Llama::push_stats(stat);

    delete[] mean;
    delete[] tmp;
    delete[] var;
}

///////////////////////////////////////////////////////////////////////////////////

void SlothRMSNorm(int s1, int s2, GroupElement *x, GroupElement *A, GroupElement *B, GroupElement *y, int scale)
// void SlothRMSNorm(int s1, int s2, GroupElement *x, GroupElement *A, GroupElement *y, int scale)
{
    GroupElement *tmp = new GroupElement[s1 * s2];

    auto t1 = time_this_block([&]() {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s2; j++) {
            tmp[i * s2 + j] = x[i * s2 + j];
        }
    }
    });

    GroupElement *var = new GroupElement[s1];
    SumOfSquare(s1, s2, tmp, var, "LayerNorm::");

    Rsqrt(s1, var, var, s2, scale, "LayerNorm::");

    auto t2 = time_this_block([&]() { 
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s2; j++) {
            y[i * s2 + j] = var[i];
        }
    }
    });

    ElemWiseMul(s1 * s2, tmp, y, y, "LayerNorm::");
    SlothARS(s1 * s2, y, y, scale, "LayerNorm::");

    GroupElement *Aexpand = tmp;
    auto t3 = time_this_block([&]() { 
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < s1; ++i) {
        for (int j = 0; j < s2; j++) {
            Aexpand[i * s2 + j] = A[j];
        }
    }
    });
    
    ElemWiseMul(s1 * s2, Aexpand, y, y, "LayerNorm::");

    // auto t5 = time_this_block([&]() { 
    // #pragma omp parallel for collapse(2)
    // for (int i = 0; i < s1; ++i) {
    //     for (int j = 0; j < s2; j++) {
    //         y[i * s2 + j] += B[j];
    //     }
    // }
    // });

    // SlothARS(s1 * s2, y, y, scale, "LayerNorm::");

    Llama::stat_t stat = {"LayerNorm::Misc", 0, t1 + t2 + t3, 0, 0, 0};
    stat.print();
    Llama::push_stats(stat);

    delete[] tmp;
    delete[] var;


}


///////////////////////////////////////////////////////////////////////////////////

// unused
void SlothGemm(int s1, int s2, int s3, GroupElement *x, GroupElement *A, GroupElement *y, int scale)
{
    if (party == DEALER)
    {
        // part 1 : MatMul
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < s3; ++j)
            {
                Arr2DIdx(y, s1, s3, i, j) = random_ge(bitlength);
            }
        }

        auto keys = KeyGenMatMul(bitlength, bitlength, s1, s2, s3, x, A, y);

        freeMatMulKey(keys.first);
        client->send_matmul_key(keys.second);
        freeMatMulKey(keys.second);

        // part 2: truncation
        pair<EdabitsPrTruncKeyPack> *keysTrunc = new pair<EdabitsPrTruncKeyPack>[s1 * s3];

#pragma omp parallel for
        for (int i = 0; i < s1 * s3; i += 1)
        {
            auto rout = random_ge(bitlength);
            keysTrunc[i] = keyGenEdabitsPrTrunc(bitlength, scale, y[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < s1 * s3; ++i)
        {
            server->send_edabits_prtrunc_key(keysTrunc[i].first, bitlength);
            client->send_edabits_prtrunc_key(keysTrunc[i].second, bitlength);
        }
        delete[] keysTrunc;
    }
    else
    {
        // part 1 : MatMul
        MatMulKey key;
        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time_1 = time_this_block([&]()
                                              { key = dealer->recv_matmul_key(bitlength, bitlength, s1, s2, s3); });

        peer->sync();

        auto compute_time_1 = time_this_block([&]()
                                              { matmul_eval_helper(party, s1, s2, s3, x, A, y, key.a, key.b, key.c); });

        auto reconstruction_stats_1 = time_comm_this_block([&]()
                                                           {
                                                               serverReconstruct(s1 * s3, y, bitlength); // this is where we get gainz, we only send a one way message of `bitlength` bits
                                                           });

        GroupElement *msb = new GroupElement[s1 * s3];

        if (party == SERVER)
        {
#pragma omp parallel for
            for (int i = 0; i < s1 * s3; ++i)
            {
                y[i] = y[i] + (1ULL << (bitlength - 2));
                msb[i] = (y[i] >> (bitlength - 1)) & 1;
            }
        }

        auto msb_stats = time_comm_this_block([&]()
                                              { serverToClient(s1 * s3, msb, 1); });

        freeMatMulKey(key);

        // part 2: truncation
        EdabitsPrTruncKeyPack *keys = new EdabitsPrTruncKeyPack[s1 * s3];

        auto keyread_time_2 = time_this_block([&]()
                                              {
            for(int i = 0; i < s1 * s3; i++){
                keys[i] = dealer->recv_edabits_prtrunc_key(bitlength);
            } });

        peer->sync();

        auto compute_time_2 = time_this_block([&]()
                                              {
            if (party == SERVER)
            {
#pragma omp parallel for
                for(int i = 0; i < s1 * s3; i++) {
                    GroupElement vb = keys[i].a;
                    GroupElement ip = y[i];
                    if (msb[i]) {
                        vb = 1 - vb;
                    }
                    GroupElement t = keys[i].b + (1ULL << (bitlength - scale - 1)) * vb;
                    y[i] = t + ((ip % (1ULL << (bitlength - 1))) >> scale) - (1ULL << (bitlength - scale - 2));
                }
            }
            else
            {
#pragma omp parallel for
                for(int i = 0; i < s1 * s3; i++) {
                    GroupElement vb = keys[i].a;
                    if (msb[i]) {
                        vb = -vb;
                    }
                    GroupElement t = keys[i].b + (1ULL << (bitlength - scale - 1)) * vb;
                    y[i] = t;
                }
            } });

        auto reconstruction_stats_2 = time_comm_this_block([&]()
                                                           { reconstruct(s1 * s3, y, bitlength); });

        Llama::stat_t stat = {"Gemm", keyread_time_1 + keyread_time_2, compute_time_1 + compute_time_2, reconstruction_stats_1.first + reconstruction_stats_2.first + msb_stats.first, reconstruction_stats_1.second + reconstruction_stats_2.second + msb_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
        delete[] msb;
    }
}

void SoftmaxTriangular(int32_t s1, int32_t s2, int bin, GroupElement *x, GroupElement *y, int32_t scale)
{
    // s1 = batch size
    // s2 = number of classes
    always_assert(s1 == s2);
    GroupElement *max = make_array<GroupElement>(s1);
    // step 1 - calculate max for each image in batch
    SlothMaxpoolTriangular(s1, s2, bin, x, max, "Softmax::");

    // step 2 - subtract max from each element in each image in batch
    int idx = 0;
    auto t1 = time_this_block([&]()
                              {
    for(int i = 1; i < s1; ++i) {
        for(int j = 0; j < i + 1; ++j) {
            y[idx] = max[i] - Arr2DIdx(x, s1, s2, i, j);
            idx++;
        }
    } });
    // step 3 - exponentiate each element in each image in batch
    nExp(idx, std::min(bin + 1, bitlength), y, y, scale);

    GroupElement *denominators = max; // reuse the array
    // // step 4 - calculate sum of exponentiated elements for each image in batch
    auto t2 = time_this_block([&]()
                              {
    idx = 0;
    for(int i = 1; i < s1; ++i) {
        denominators[i] = 0;
        for(int j = 0; j < i + 1; ++j) {
            denominators[i] = denominators[i] + y[idx];
            idx++;
        }
    } });

    // step 5 - calculate inverse of all the denominators
    InverseLUT(s1 - 1, denominators + 1, denominators + 1, scale, scale + 10, "Softmax::"); // only works if numClasses <= 1024

    // step 6 - multiply each element in each image in batch by the inverse of the denominator
    GroupElement *expandedDenominator = make_array<GroupElement>(s1 * s2);
    auto t3 = time_this_block([&]()
                              {
    idx = 0;
    for(int i = 1; i < s1; ++i) {
        for(int j = 0; j < i + 1; ++j) {
            expandedDenominator[idx] = denominators[i];
            idx++;
        }
    } });
    delete[] max;

    ElemWiseMul(idx, expandedDenominator, y, y, "Softmax::");
    SlothARS(idx, y, y, scale, "Softmax::");

    GroupElement *yRes = expandedDenominator; // reuse the array
    auto t4 = time_this_block([&]()
                              {
    idx = 0;

    Arr2DIdx(yRes, s1, s2, 0, 0) = (party == DEALER ? 0 : (1LL << scale));
    for(int j = 1; j < s2; ++j) {
        Arr2DIdx(yRes, s1, s2, 0, j) = 0;
    }
    for(int i = 1; i < s1; ++i) {
        for(int j = 0; j < i + 1; ++j) {
            Arr2DIdx(yRes, s1, s2, i, j) = y[idx];
            idx++;
        }
        for(int j = i + 1; j < s2; ++j) {
            Arr2DIdx(yRes, s1, s2, i, j) = 0;
        }
    }

    for (int i = 0; i < s1 * s2; ++i)
    {
        y[i] = yRes[i];
    } });

    if (party != DEALER)
        Llama::push_stats(Llama::stat_t{"Softmax::Misc", 0, t1 + t2 + t3 + t4, 0, 0, 0});

    delete[] expandedDenominator;
}

void MatMul2DTriangular(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
                        MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA)
{
    always_assert(s1 == s3);
    if (party == DEALER)
    {

        // TODO: dealer can generate less key material
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < s3; ++j)
            {
                Arr2DIdx(C_mask, s1, s3, i, j) = random_ge(bitlength);
            }
        }

        auto keys = KeyGenMatMul(bitlength, bitlength, s1, s2, s3, A_mask, B_mask, C_mask);

        // server->send_matmul_key(keys.first);
        freeMatMulKey(keys.first);
        client->send_matmul_key(keys.second);
        freeMatMulKey(keys.second);
    }
    else
    {
        MatMulKey key;
        uint64_t keysize_start = dealer->bytesReceived();
        auto keyread_time = time_this_block([&]()
                                            { key = dealer->recv_matmul_key(bitlength, bitlength, s1, s2, s3); });
        GroupElement *C_compress = make_array<GroupElement>(s1 * s3);

        peer->sync();

        int idx = 0;
        auto compute_time = time_this_block([&]()
                                            {
            matmul_eval_helper_triangular(party, s1, s2, s3, A, B, C, key.a, key.b, key.c);
            for (int i = 0; i < s1; ++i)
            {
                for (int j = 0; j < i + 1; ++j)
                {
                    C_compress[idx] = C[i * s3 + j];
                    idx++;
                }
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(idx, C_compress, bitlength); });

        auto t1 = time_this_block([&]()
                                  {
        idx = 0;
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < i + 1; ++j)
            {
                C[i * s3 + j] = C_compress[idx];
                idx++;
            }
        } });

        Llama::stat_t stat = {"Linear::MatMul", keyread_time, compute_time + t1, reconstruction_stats.first, reconstruction_stats.second, dealer->bytesReceived() - keysize_start};
        stat.print();
        Llama::push_stats(stat);

        freeMatMulKey(key);
    }
}

// unused
void SlothAttentionTriangular(int n_seq, int n_embd, int n_heads, GroupElement *q, GroupElement *k, GroupElement *v, GroupElement *out, int scale)
{
    GroupElement *kt = new GroupElement[n_seq * n_embd];
    for (int i = 0; i < n_seq; ++i)
    {
        for (int j = 0; j < n_embd; ++j)
        {
            Arr2DIdx(kt, n_embd, n_seq, j, i) = Arr2DIdx(k, n_seq, n_embd, i, j);
        }
    }

    GroupElement *qkt = new GroupElement[n_seq * n_seq];

    MatMul2DTriangular(n_seq, n_embd, n_seq, q, q, kt, kt, qkt, qkt, true);

    GroupElement *qkt_compressed = new GroupElement[n_seq * n_seq];
    int idx = 0;
    for (int i = 0; i < n_seq; ++i)
    {
        for (int j = 0; j < i + 1; ++j)
        {
            qkt_compressed[idx] = Arr2DIdx(qkt, n_seq, n_seq, i, j);
            idx++;
        }
    }

    GroupElement invdiv = (1LL << scale) / sqrt(double(n_embd) / double(n_heads));

    // if indiv is power of 2
    if ((invdiv & (invdiv - 1)) == 0)
    {
        int s2 = log2(invdiv);
        if (2 * scale > s2)
        {
            SlothARS(idx, qkt_compressed, qkt_compressed, 2 * scale - s2, "Attention::");
        }
        else
        {
            for (int i = 0; i < idx; ++i)
            {
                qkt_compressed[i] = qkt_compressed[i] * (1LL << (s2 - 2 * scale));
            }
        }
    }
    else
    {
        SlothARS(idx, qkt_compressed, qkt_compressed, scale, "Attention::");
        for (int i = 0; i < idx; ++i)
        {
            qkt_compressed[i] = qkt_compressed[i] * invdiv;
        }
        SlothARS(idx, qkt_compressed, qkt_compressed, scale, "Attention::");
    }

    GroupElement *qkt_res = qkt;
    idx = 0;
    for (int i = 0; i < n_seq; ++i)
    {
        for (int j = 0; j < i + 1; ++j)
        {
            Arr2DIdx(qkt_res, n_seq, n_seq, i, j) = qkt_compressed[idx];
            idx++;
        }
    }

    SoftmaxTriangular(n_seq, n_seq, bitlength - scale, qkt_res, qkt_res, scale);

    MatMul2D(n_seq, n_seq, n_embd, qkt_res, qkt_res, v, v, out, out, true);
    SlothARS(n_seq * n_embd, out, out, scale, "Attention::");
}

void SlothWrap_ss(int size, int bin, GroupElement *x, GroupElement *y, std::string parent)
{
    if (party == DEALER)
    {
        pair<WrapSSKeyPack> *keys = new pair<WrapSSKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenWrapSS(bin, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_wrap_ss_key(keys[i].first);
            client->send_wrap_ss_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        WrapSSKeyPack *keys = new WrapSSKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_wrap_ss_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalWrapSS(party - 2, x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, 1); });

        Llama::stat_t stat = {
            parent,
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

void SlothWrap_dpf(int size, int bin, GroupElement *x, GroupElement *y, std::string parent)
{
    if (party == DEALER)
    {
        pair<WrapDPFKeyPack> *keys = new pair<WrapDPFKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenWrapDPF(bin, x[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_wrap_dpf_key(keys[i].first);
            client->send_wrap_dpf_key(keys[i].second);
            freeWrapDPFKeyPackPair(keys[i]);
        }

        delete[] keys;
    }
    else
    {
        WrapDPFKeyPack *keys = new WrapDPFKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_wrap_dpf_key(bin);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalWrapDPF(party - 2, x[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, 1); });

        Llama::stat_t stat = {
            parent,
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        Llama::push_stats(stat);

        for (int i = 0; i < size; ++i)
        {
            freeWrapDPFKeyPack(keys[i]);
        }
        delete[] keys;
    }
}

void SlothWrap(int size, int bin, GroupElement *x, GroupElement *w, std::string parent)
{
    if (bin <= 7)
    {
        SlothWrap_ss(size, bin, x, w, parent);
    }
    else
    {
        SlothWrap_dpf(size, bin, x, w, parent);
    }
}

void SlothLRSfromWrap(int size, GroupElement *x, GroupElement *w, GroupElement *y, int scale, std::string parent)
{
    if (party == DEALER)
    {
        pair<SlothLRSKeyPack> *keys = new pair<SlothLRSKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenSlothLRS(bitlength, scale, x[i], w[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_lrs_key(keys[i].first);
            client->send_sloth_lrs_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        SlothLRSKeyPack *keys = new SlothLRSKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_sloth_lrs_key(bitlength, scale);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothLRS(party - 2, x[i], w[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bitlength); });

        Llama::stat_t stat = {
            parent,
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

void SlothLRS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *w = new GroupElement[size];
    GroupElement *x0 = w;

    auto t = time_this_block([&]()
                             {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        x0[i] = x[i];
        mod(x0[i], scale);
    } });

    SlothWrap(size, scale, x0, w, prefix + "Truncation");
    SlothLRSfromWrap(size, x, w, y, scale, prefix + "Truncation");

    if (party != DEALER)
        Llama::push_stats({prefix + "Truncation::Misc", 0, t, 0, 0, 0});

    delete[] w;
}

void SlothARS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *z = new GroupElement[size];

    if (party == DEALER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            z[i] = x[i];
        }
    }
    else
    {
        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            z[i] = x[i] + (1LL << (bitlength - 2));
        } });
        Llama::stat_t stat = {prefix + "Truncation::Misc", 0, t, 0, 0, 0};
        stat.print();
        Llama::push_stats(stat);
    }

    SlothLRS(size, z, z, scale, prefix);

    if (party == DEALER)
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            y[i] = z[i];
        }
    }
    else
    {
        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            y[i] = z[i] - (1LL << (bitlength - scale - 2));
        } });
        Llama::stat_t stat = {prefix + "Truncation::Misc", 0, t, 0, 0, 0};
        stat.print();
        Llama::push_stats(stat);
    }
}

void SlothTRfromWrap(int size, int bin, GroupElement *x, GroupElement *w, GroupElement *y, int scale, std::string parent)
{
    if (party == DEALER)
    {
        pair<SlothLRSKeyPack> *keys = new pair<SlothLRSKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(1);
            keys[i] = keyGenSlothLRS(bin, scale, x[i], w[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_lrs_key(keys[i].first);
            client->send_sloth_lrs_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        SlothLRSKeyPack *keys = new SlothLRSKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_sloth_lrs_key(bin, scale);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothLRS(party - 2, x[i], w[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bin - scale); });

        Llama::stat_t stat = {
            parent,
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

void SlothTR(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *w = new GroupElement[size];
    GroupElement *x0 = w;

    auto t = time_this_block([&]()
                             {
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        x0[i] = x[i];
        mod(x0[i], scale);
    } });

    SlothWrap(size, scale, x0, w, prefix + "TruncateReduce");
    SlothTRfromWrap(size, bin, x, w, y, scale, prefix + "TruncateReduce");

    delete[] w;

    if (party != DEALER)
        Llama::push_stats({prefix + "TruncateReduce::Misc", 0, t, 0, 0, 0});
}

void SlothSignExtendFromWrap(int size, int bin, int bout, GroupElement *x, GroupElement *w, GroupElement *y, std::string parent)
{
    if (party == DEALER)
    {
        pair<SlothSignExtendKeyPack> *keys = new pair<SlothSignExtendKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            GroupElement rout = random_ge(bout);
            keys[i] = keyGenSlothSignExtend(bin, bout, x[i], w[i], rout);
            y[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_sloth_sign_extend_key(keys[i].first);
            client->send_sloth_sign_extend_key(keys[i].second);
        }

        delete[] keys;
    }
    else
    {
        SlothSignExtendKeyPack *keys = new SlothSignExtendKeyPack[size];

        uint64_t keysize_start = dealer->bytesReceived();
        uint64_t keyread_time = time_this_block([&]()
                                                {
            for (int i = 0; i < size; ++i) {
                keys[i] = dealer->recv_sloth_sign_extend_key(bin, bout);
            } });

        peer->sync();

        uint64_t compute_time = time_this_block([&]()
                                                {
#pragma omp parallel for
            for (int i = 0; i < size; ++i) {
                y[i] = evalSlothSignExtend(party - 2, x[i], w[i], keys[i]);
            } });

        auto reconstruction_stats = time_comm_this_block([&]()
                                                         { reconstruct(size, y, bout); });

        Llama::stat_t stat = {
            parent,
            keyread_time,
            compute_time,
            reconstruction_stats.first,
            reconstruction_stats.second,
            dealer->bytesReceived() - keysize_start};

        stat.print();
        Llama::push_stats(stat);

        delete[] keys;
    }
}

void SlothFaithfulARS(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix)
{
    GroupElement *w = new GroupElement[size];

    SlothTR(size, bin, x, y, scale, prefix + "FaithfulARS::");

    if (party != DEALER)
    {
        auto t = time_this_block([&]()
                                 {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            y[i] = y[i] + (1LL << (bin-scale-1));
            mod(y[i], bin-scale);
        } });

        Llama::push_stats({prefix + "FaithfulARS::Misc", 0, t, 0, 0, 0});
    }
    else
    {
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            mod(y[i], bin - scale);
        }
    }

    SlothWrap(size, bin - scale, y, w, prefix + "FaithfulARS");
    SlothSignExtendFromWrap(size, bin - scale, bin, y, w, y, prefix + "FaithfulARS");

    delete[] w;
}

// this API is not exposed directly, so we used single array for values and masks
// all APIs should do the same in principal
void InsecureInverse(int32_t size, GroupElement *A, GroupElement *invA, int32_t sf, int32_t upper)
{
    // KG: make sure this is inplace secure (i.e can accept invA = A)
    uint64_t logk = osuCrypto::log2ceil(upper);
    uint64_t m = logk + 1;
    std::cerr << ">> InsecureInverse - start" << std::endl;

    if (party == DEALER)
    {
        for (int i = 0; i < size; ++i)
        {
            auto rout = random_ge(bitlength);
            auto keys = keyGenTaylor(bitlength, bitlength, 2.630, -5.857, 4.245, A[i], rout, sf, logk);
            server->send_taylor_key(keys.first, bitlength, m);
            client->send_taylor_key(keys.second, bitlength, m);
            invA[i] = rout;
            // TODO: delete keys[i].first and keys[i].second
        }
    }
    else
    {
        TaylorKeyPack *keys = new TaylorKeyPack[size];
        for (int i = 0; i < size; ++i)
        {
            keys[i] = dealer->recv_taylor_key(bitlength, m, sf);
        }

        peer->sync();

        GroupElement *tmp = new GroupElement[2 * size];
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; ++i)
        {
            auto tup = evalTaylor_round1(party - SERVER, bitlength, bitlength, 2.630, -5.857, 4.245, A[i], keys[i], sf, logk);
            tmp[i] = tup.first;
            tmp[i + size] = tup.second;
        }
        reconstruct(2 * size, tmp, bitlength);

        for (int i = 0; i < size; ++i)
        {
            auto tup = evalTaylor_round2(party - SERVER, bitlength, bitlength, 2.630, -5.857, 4.245, A[i], keys[i], sf, logk, tmp[i], tmp[i + size]);
            tmp[i + size] = tup.first + tup.second;
        }
        reconstruct(size, tmp + size, bitlength);

        for (int i = 0; i < size; ++i)
        {
            auto tup = evalTaylor_round3(party - SERVER, bitlength, bitlength, 2.630, -5.857, 4.245, A[i], keys[i], sf, logk, tmp[i], tmp[i + size], tmp[i + size]);
            tmp[i + size] = tup;
        }
        reconstruct(size, tmp + size, bitlength);

        for (int i = 0; i < size; ++i)
        {
            invA[i] = tmp[i + size];
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " milliseconds" << std::endl;
        evalMicroseconds += eval_time;
        delete[] tmp;
    }
    std::cerr << ">> InsecureInverse - end" << std::endl;
}

void r2_threads_helper_1(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *tmp, Relu2RoundKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        tmp[i] = evalRelu2_drelu(party - 2, inArr[i], keys[i]);
    }
}

void r2_threads_helper_2(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *tmp, GroupElement *outArr, Relu2RoundKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        outArr[i] = evalRelu2_mult(party - 2, tmp[i], inArr[i], keys[i]);
    }
}

void Relu2Round(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu_cache, int effectiveInputBw)
{
    std::cerr << ">> Relu2Round - Start" << std::endl;
    GroupElement *tmp = make_array<GroupElement>(size);
    if (party == DEALER)
    {
        GroupElement *drelu_mask = tmp;
        pair<Relu2RoundKeyPack> *keys = new pair<Relu2RoundKeyPack>[size];
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            drelu_mask[i] = random_ge(1);
            if (drelu_cache != nullptr)
                drelu_cache[i] = drelu_mask[i];
            GroupElement rout = random_ge(bitlength);
            keys[i] = keyGenRelu2Round(effectiveInputBw, bitlength, inArr_mask[i], drelu_mask[i], rout);
            outArr_mask[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_relu_2round_key(keys[i].first);
            client->send_relu_2round_key(keys[i].second);
            freeRelu2RoundKeyPackPair(keys[i]);
        }
        delete[] keys;
    }
    else
    {
        auto keyread_start = std::chrono::high_resolution_clock::now();
        Relu2RoundKeyPack *keys = new Relu2RoundKeyPack[size];
        for (int i = 0; i < size; ++i)
        {
            keys[i] = dealer->recv_relu_2round_key(effectiveInputBw, bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        {
            std::thread thread_pool[num_threads];
            for (int i = 0; i < num_threads; ++i)
            {
                thread_pool[i] = std::thread(r2_threads_helper_1, i, size, inArr, tmp, keys);
            }

            for (int i = 0; i < num_threads; ++i)
            {
                thread_pool[i].join();
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t onlineComm0 = peer->bytesReceived() + peer->bytesSent();
        reconstruct(size, tmp, 1);
        if (drelu_cache != nullptr)
            for (int i = 0; i < size; ++i)
            {
                drelu_cache[i] = tmp[i];
            }
        auto t2 = std::chrono::high_resolution_clock::now();
        {
            std::thread thread_pool[num_threads];
            for (int i = 0; i < num_threads; ++i)
            {
                thread_pool[i] = std::thread(r2_threads_helper_2, i, size, inArr, tmp, outArr, keys);
            }

            for (int i = 0; i < num_threads; ++i)
            {
                thread_pool[i].join();
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        reconstruct(size, outArr, bitlength);
        uint64_t onlineComm1 = peer->bytesReceived() + peer->bytesSent();
        reluOnlineComm += (onlineComm1 - onlineComm0);
        auto end = std::chrono::high_resolution_clock::now();

        uint64_t time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        reluEvalMicroseconds += time_taken;
        evalMicroseconds += time_taken;

        uint64_t time1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();
        uint64_t time2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        uint64_t time3 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        uint64_t time4 = std::chrono::duration_cast<std::chrono::microseconds>(end - t3).count();
        // std::cerr << "   Compute time 1: " << time1 / 1000.0 << " milliseconds" << std::endl;
        // std::cerr << "   Reconstruct time 1: " << time2 / 1000.0 << " milliseconds" << std::endl;
        // std::cerr << "   Compute time 2: " << time3 / 1000.0 << " milliseconds" << std::endl;
        // std::cerr << "   Reconstruct time 2: " << time4 / 1000.0 << " milliseconds" << std::endl;
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds" << std::endl;
        std::cerr << "   Compute Time = " << (time1 + time3) / 1000.0 << " milliseconds" << std::endl;
        std::cerr << "   Reconstruct Time = " << (time2 + time4) / 1000.0 << " milliseconds" << std::endl;
        std::cerr << "   Online Time = " << time_taken / 1000.0 << " milliseconds" << std::endl;
        std::cerr << "   Online Comm = " << (onlineComm1 - onlineComm0) << " bytes\n";

        for (int i = 0; i < size; ++i)
        {
            freeRelu2RoundKeyPack(keys[i]);
        }
        delete[] keys;
    }

    delete[] tmp;
    std::cerr << ">> Relu2Round - End" << std::endl;
}

void fixtofloat_threads_helper(int thread_idx, int32_t size, int scale, GroupElement *inp, GroupElement *out, GroupElement *pl, GroupElement *q,
                               GroupElement *pow, GroupElement *sm, FixToFloatKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        evalFixToFloat_1(party - 2, bitlength, scale, inp[i], keys[i], pl, q,
                         out[i * 4 + 0], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3], pow[i], sm[i]);
    }
}

void FixToFloat(int size, GroupElement *inp, GroupElement *out, int scale)
{
    // std::cerr << ">> FixToFloat - Start" << std::endl;
    GroupElement *p = new GroupElement[2 * bitlength];
    GroupElement *q = new GroupElement[2 * bitlength];
    fill_pq(p, q, bitlength);

    if (party == DEALER)
    {
        pair<FixToFloatKeyPack> *keys = new pair<FixToFloatKeyPack>[size];
#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            keys[i] = keyGenFixToFloat(bitlength, scale, inp[i], p, q);
            out[4 * i] = 0;
            out[4 * i + 1] = 0;
            out[4 * i + 2] = 0;
            out[4 * i + 3] = 0;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_fix_to_float_key(keys[i].first, bitlength);
            client->send_fix_to_float_key(keys[i].second, bitlength);
            freeFixToFloatKeyPackPair(keys[i]);
        }
        delete[] keys;
    }
    else
    {
        auto keyread_start = std::chrono::high_resolution_clock::now();
        FixToFloatKeyPack *keys = new FixToFloatKeyPack[size];

        for (int i = 0; i < size; ++i)
        {
            keys[i] = dealer->recv_fix_to_float_key(bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();
        GroupElement *pow = new GroupElement[size];
        GroupElement *sm = new GroupElement[size];
        GroupElement *ym = new GroupElement[size];

        peer->sync();
        auto eval_start = std::chrono::high_resolution_clock::now();

        std::thread thread_pool[num_threads];
        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(fixtofloat_threads_helper, i, size, scale, inp, out, p, q, pow, sm, keys);
        }

        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }

        reconstruct(size, sm, 1);
        reconstruct(size, pow, bitlength);

        for (int i = 0; i < size; ++i)
        {
            ym[i] = 2 * evalSelect(party - 2, 1 ^ sm[i], inp[i], keys[i].selectKey);
            if (party == 2)
            {
                ym[i] = ym[i] - inp[i];
            }
        }

        reconstruct(size, ym, bitlength);

        for (int i = 0; i < size; ++i)
        {
            out[i * 4 + 0] = -keys[i].ry * pow[i] - keys[i].rpow * ym[i] + keys[i].rm;
            if (party == 2)
            {
                out[i * 4 + 0] = out[i * 4 + 0] + ym[i] * pow[i];
                out[i * 4 + 0] = -((-out[i * 4 + 0]) >> (bitlength - scale));
            }
            else
            {
                out[i * 4 + 0] = out[i * 4 + 0] >> (bitlength - scale);
            }
        }

        auto eval_end = std::chrono::high_resolution_clock::now();
        auto eval_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(eval_end -
                                                                                     eval_start)
                                   .count();
        // std::cerr << "   Key Read Time = " << keyread_time_taken << " miliseconds" << std::endl;
        // std::cerr << "   Online Time = " << eval_time_taken / 1000.0 << " miliseconds" << std::endl;
        evalMicroseconds += eval_time_taken;
        delete[] sm;
        delete[] pow;
        delete[] ym;
        for (int i = 0; i < size; ++i)
        {
            freeFixToFloatKeyPack(keys[i]);
        }
        delete[] keys;
    }
    // std::cerr << ">> FixToFloat - End" << std::endl;
}

void FloatToFixCt(int size, GroupElement *inp, GroupElement *out, int scale)
{
    if (party == DEALER)
    {
        memset(out, 0, size * sizeof(GroupElement));
    }
    else
    {
        GroupElement *m = new GroupElement[2 * size];
        GroupElement *e = m + size;

        for (int i = 0; i < size; ++i)
        {
            m[i] = inp[4 * i + 0];
            e[i] = inp[4 * i + 1];
            // if (party == 2)
            // {
            //     e[i] += scale;
            //     e[i] -= 127; // fp32 bias
            // }
        }
        // now have m and e in the clear
        // reconstruct(2 * size, m, 64);
        for (int i = 0; i < size; ++i)
        {
            mod(m[i], 24);
            mod(e[i], 10);
            assert(e[i] < 256);

            // int eAsInt = e[i] < 512 ? e[i] : -1 * (1024 - e[i]);
            // assert(eAsInt <= 126 && eAsInt >= -127);
            // if(i < 10) printf("%d=%ld, %ld\n", i, m[i], e[i]);
            int ePrime = e[i] - 127 + scale;
            // if(i < 10) printf("%d=%ld, %ld, %d\n", i, m[i], e[i], ePrime);
            GroupElement x = 0;
            if (ePrime >= 0 && ePrime <= scale)
            {
                x = m[i] * (1ULL << ePrime);
                assert(x < (1ULL << 63));
                x >>= 23;
                // auto xf = x;
                // mod(xf, scale);
                // auto s = random_ge(scale);
                // if(s < xf) x += 1;
                // if(i < 10) printf("%d=%ld, %ld, %ld\n", i, m[i], ePrime, x);
            }
            out[i] = x;
        }
        delete[] m;
    }
}

void FloatToFix(int size, GroupElement *inp, GroupElement *out, int scale)
{
    // std::cerr << ">> FloatToFix - Start" << std::endl;

    if (party == DEALER)
    {
        pair<FloatToFixKeyPack> *keys = new pair<FloatToFixKeyPack>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            auto rout = random_ge(bitlength);
            keys[i] = keyGenFloatToFix(bitlength, scale, rout);
            out[i] = rout;
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_float_to_fix_key(keys[i].first, bitlength);
            client->send_float_to_fix_key(keys[i].second, bitlength);
            freeFloatToFixKeyPackPair(keys[i]);
        }
        delete[] keys;
    }
    else
    {
        auto keyread_start = std::chrono::high_resolution_clock::now();
        FloatToFixKeyPack *keys = new FloatToFixKeyPack[size];
        for (int i = 0; i < size; ++i)
        {
            keys[i] = dealer->recv_float_to_fix_key(bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        GroupElement *m = new GroupElement[2 * size];
        GroupElement *e = m + size;
        GroupElement *w = new GroupElement[2 * size];
        GroupElement *t = new GroupElement[size];
        GroupElement *h = w + size;
        GroupElement *d = new GroupElement[size];

        peer->sync();
        auto eval_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; ++i)
        {
            m[i] = inp[4 * i + 0] + keys[i].rm;
            e[i] = inp[4 * i + 1] + keys[i].re;
            if (party == 2)
            {
                e[i] += (scale);
                e[i] -= 127; // fp32 bias
            }
        }

        // m and e are in a single array. m is the first half and e is the second half
        reconstruct(2 * size, m, 24);

        for (int i = 0; i < size; ++i)
        {
            mod(m[i], 24);
            evalDCF(party - 2, &w[i], m[i], keys[i].dcfKey);
            w[i] = w[i] + keys[i].rw;
        }

        for (int i = 0; i < size; i++)
        {
            mod(e[i], 10);
            d[i] = 0;
            for (int j = 0; j < 1024; j++)
            {
                d[i] = d[i] + (pow_helper(scale, j) * keys[i].p[(j - e[i]) % 1024]);
            }
            h[i] = keys[i].rh + (pow((GroupElement)2, 24) * d[i]);
        }

        // w and h are in a single array w. w is the first half and h is the second half
        reconstruct(2 * size, w, bitlength);

        for (int i = 0; i < size; ++i)
        {
            t[i] = evalSelect(party - 2, w[i], h[i], keys[i].selectKey);
            t[i] = t[i] + keys[i].q[e[i]];
            t[i] = t[i] + (m[i] * d[i]);
        }

        reconstruct(size, t, bitlength);

        for (int i = 0; i < size; ++i)
        {
            out[i] = evalARS(party - 2, t[i], 23, keys[i].arsKey);
        }

        // reconstruct(size, out, bitlength);

        auto eval_end = std::chrono::high_resolution_clock::now();
        auto eval_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(eval_end -
                                                                                     eval_start)
                                   .count();
        // std::cerr << "   Key Read Time = " << keyread_time_taken << " miliseconds" << std::endl;
        // std::cerr << "   Online Time = " << eval_time_taken / 1000.0 << " miliseconds" << std::endl;
        evalMicroseconds += eval_time_taken;
        delete[] m;
        delete[] w;
        delete[] t;
        for (int i = 0; i < size; ++i)
        {
            freeFloatToFixKeyPack(keys[i]);
        }
        delete[] keys;
    }
    // std::cerr << ">> FloatToFix - End" << std::endl;
}

void mult_threads_helper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *multArrVec, GroupElement *outArr, MultKey *keys)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    auto p = get_start_end(size, thread_idx);
    for (int i = p.first; i < p.second; i += 1)
    {
        outArr[i] = MultEval(party - SERVER, keys[i], inArr[i], multArrVec[i]);
    }
    auto thread_end = std::chrono::high_resolution_clock::now();
}

void ElemWiseSecretSharedVectorMult(int32_t size, MASK_PAIR(GroupElement *inArr),
                                    MASK_PAIR(GroupElement *multArrVec), MASK_PAIR(GroupElement *outputArr))
{
    std::cerr << ">> ElemWise Mult - start" << std::endl;
    if (party == DEALER)
    {
        uint64_t dealer_toal_time = 0;
        pair<MultKey> *keys = new pair<MultKey>[size];

#pragma omp parallel for
        for (int i = 0; i < size; ++i)
        {
            auto dealer_start = std::chrono::high_resolution_clock::now();
            auto rout = random_ge(bitlength);
            keys[i] = MultGen(inArr_mask[i], multArrVec_mask[i], rout);
            outputArr_mask[i] = rout;
            auto dealer_end = std::chrono::high_resolution_clock::now();
            dealer_toal_time += std::chrono::duration_cast<std::chrono::microseconds>(dealer_end - dealer_start).count();
        }

        for (int i = 0; i < size; ++i)
        {
            server->send_mult_key(keys[i].first);
            client->send_mult_key(keys[i].second);
        }
        dealerMicroseconds = dealerMicroseconds + dealer_toal_time;
        delete[] keys;
    }
    else
    {
        MultKey *keys = new MultKey[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; ++i)
        {
            keys[i] = dealer->recv_mult_key();
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                                                        keyread_start)
                                      .count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(mult_threads_helper, i, size, inArr, multArrVec, outputArr, keys);
        }

        for (int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }
        auto mid = std::chrono::high_resolution_clock::now();
        // reconstruct(size, outputArr, bitlength);
        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        multEvalMicroseconds += (reconstruct_time + compute_time);
        delete[] keys;
    }
    std::cerr << ">> ElemWise Mult - end" << std::endl;
}

void PiranhaSoftmax(int32_t s1, int32_t s2, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t sf)
{
    // s1 = batch size
    // s2 = number of classes

    std::cerr << ">> Softmax - start" << std::endl;
    GroupElement *max = make_array<GroupElement>(s1);
    // step 1 - calculate max for each image in batch
    GroupElement *oneHot = make_array<GroupElement>(s1 * (s2 - 1));
    MaxPool(s1, 1, 1, 1, s2, 1, 0, 0, 0, 0, 1, 1, s1, s2, 1, 1, MASK_PAIR(inArr), max, max, oneHot);
    delete[] oneHot; // TODO: support passing oneHot as nullptr

    // step 2 - subtract max from each element in each image in batch and add 2
    if (party == DEALER)
    {
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < s2; ++j)
            {
                Arr2DIdx(outArr_mask, s1, s2, i, j) = Arr2DIdx(inArr_mask, s1, s2, i, j) - max[i];
            }
        }
    }
    else
    {
        for (int i = 0; i < s1; ++i)
        {
            for (int j = 0; j < s2; ++j)
            {
                Arr2DIdx(outArr, s1, s2, i, j) = Arr2DIdx(inArr, s1, s2, i, j) - max[i] + (1 << (sf + 1));
            }
        }
    }

    // step 3 - exponentiate each element in each image in batch
    // e^x = RT((x+2), 1) for negative x
    // ReluTruncate(s1 * s2, MASK_PAIR(outArr), MASK_PAIR(outArr), 1, nullptr); // Q: can we do this in place? can be a source of bug in future
    Relu2Round(s1 * s2, MASK_PAIR(outArr), MASK_PAIR(outArr), nullptr, 64);
    for (int i = 0; i < s1 * s2; ++i)
    {
        if (party == DEALER)
        {
            outArr_mask[i] = outArr_mask[i] / 2;
        }
        else
        {
            outArr[i] = outArr[i] / 2;
        }
    }

    GroupElement *denominators = max; // reuse the array
    // // step 4 - calculate sum of exponentiated elements for each image in batch
    if (party == DEALER)
    {
        for (int i = 0; i < s1; ++i)
        {
            denominators[i] = 0;
            for (int j = 0; j < s2; ++j)
            {
                denominators[i] = denominators[i] + Arr2DIdx(outArr_mask, s1, s2, i, j);
            }
            // denominators[i] = denominators[i] * s1;
        }
    }
    else
    {
        for (int i = 0; i < s1; ++i)
        {
            denominators[i] = 0;
            for (int j = 0; j < s2; ++j)
            {
                denominators[i] = denominators[i] + Arr2DIdx(outArr, s1, s2, i, j);
            }
            // denominators[i] = denominators[i] * s1;
        }
    }
    // step 5 - calculate inverse of all the denominators
    InsecureInverse(s1, denominators, denominators, sf, s2 * s1);

    // step 6 - multiply each element in each image in batch by the inverse of the denominator
    GroupElement *expandedDenominator = make_array<GroupElement>(s1 * s2);
    for (int i = 0; i < s1; ++i)
    {
        for (int j = 0; j < s2; ++j)
        {
            Arr2DIdx(expandedDenominator, s1, s2, i, j) = denominators[i];
        }
    }
    delete[] max;

    ElemWiseSecretSharedVectorMult(s1 * s2, expandedDenominator, expandedDenominator, MASK_PAIR(outArr), MASK_PAIR(outArr));
    // ScaleDown(s1 * s2, MASK_PAIR(outArr), sf);

    always_assert((s1 & (s1 - 1)) == 0);
    auto logs1 = osuCrypto::log2ceil(s1);
    for (int i = 0; i < s1 * s2; ++i)
    {
        if (party == DEALER)
        {
            outArr_mask[i] = outArr_mask[i] >> (sf + logs1);
        }
        else
        {
            outArr[i] = outArr[i] >> (sf + logs1);
        }
    }
    std::cerr << ">> Softmax - end" << std::endl;

    delete[] expandedDenominator;
}
