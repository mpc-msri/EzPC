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

#include "api.h"
#include "and.h"
#include "comms.h"
#include "utils.h"
#include "array.h"
#include "conv.h"
#include "mult.h"
#include "pubdiv.h"
#include "config.h"
#include "stats.h"
#include "relutruncate.h"
#include "taylor.h"
#include "relu.h"
#include "select.h"
#include "float.h"
#include <cassert>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <thread>
#include <Eigen/Dense>

template <typename T> using pair = std::pair<T,T>;

bool localTruncation = false;

using namespace LlamaConfig;

void StartComputation()
{
    std::cerr << "=== COMPUTATION START ===\n\n";
    startTime = std::chrono::system_clock::now().time_since_epoch().count();

    if (party != DEALER)
        peer->sync();
    
    if (party != DEALER) {
        if (party == SERVER) {
            inputOfflineComm = peer->bytesSent;
            inputOnlineComm = peer->bytesReceived;
        }
        else {
            inputOfflineComm = peer->bytesReceived;
            inputOnlineComm = peer->bytesSent;
        }
        peer->bytesSent = 0;
        peer->bytesReceived = 0;
    }
    else {
        // always_assert(server->bytesSent == 16);
        // always_assert(server->bytesSent == 16);
        server->bytesSent = 0;
        client->bytesSent = 0;
    }

    if (party == DEALER) {
        osuCrypto::AES aesSeed(osuCrypto::toBlock(0, time(NULL)));
        auto commonSeed = aesSeed.ecbEncBlock(osuCrypto::ZeroBlock);
        server->send_block(commonSeed);
        prngShared.SetSeed(commonSeed);
    }
    else if (party == SERVER) {
        auto commonSeed = dealer->recv_block();
        prngShared.SetSeed(commonSeed);
    }
}

void EndComputation()
{
    std::cerr << "\n=== COMPUTATION END ===\n\n";
    if (party != DEALER) {
        std::cerr << "Offline Communication = " << inputOfflineComm << " bytes\n";
        std::cerr << "Offline Time = " << accumulatedInputTimeOffline / 1000.0 << " milliseconds\n";
        std::cerr << "Online Rounds = " << numRounds << "\n";
        std::cerr << "Online Communication = " << peer->bytesSent + peer->bytesReceived + inputOnlineComm << " bytes\n";
        std::cerr << "Online Time = " << (evalMicroseconds + accumulatedInputTimeOnline) / 1000.0 << " milliseconds\n";
        std::cerr << "Total Eigen Time = " << eigenMicroseconds / 1000.0 << " milliseconds\n\n";
        std::cerr << "Conv + Matmul Time = " << (convEvalMicroseconds + matmulEvalMicroseconds) / 1000.0 << " milliseconds\n";
        std::cerr << "Relu time = " << reluEvalMicroseconds / 1000.0 << " milliseconds\n";
        std::cerr << "ReluTruncate time = " << reluTruncateEvalMicroseconds / 1000.0 << " milliseconds\n";
        std::cerr << "MaxPool time = " << maxpoolEvalMicroseconds / 1000.0 << " milliseconds\n";
        std::cerr << "Select/Bit operations Time = " << selectEvalMicroseconds / 1000.0 << " milliseconds\n";
        std::cerr << "Truncate time = " << arsEvalMicroseconds / 1000.0 << " milliseconds\n";
        auto endTime = std::chrono::system_clock::now().time_since_epoch().count();
        std::cerr << "Total Time (including Key Read) = " << (endTime - startTime) / 1000000.0 << " milliseconds\n";
    }
    else {
        std::cerr << "Offline Communication = " << server->bytesSent + client->bytesSent << " bytes\n";
        std::cerr << "Offline Time = " << (dealerMicroseconds + accumulatedInputTimeOffline) / 1000.0 << " milliseconds\n";
    }
    std::cerr << "=========\n";
}

const bool parallel_reconstruct = true;
void reconstruct(int32_t size, GroupElement *arr, int bw)
{
    uint64_t *tmp = new uint64_t[size];
    if (parallel_reconstruct)
    {
        std::thread send_thread(&Peer::send_batched_input, peer, arr, size, bw);
        std::thread recv_thread(&Peer::recv_batched_input, peer, tmp, size, bw);
        send_thread.join();
        recv_thread.join();
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

void reconstructRT(int32_t size, GroupElement *arr, int bw)
{
    uint64_t *tmp = new uint64_t[size];
    int bitarraySize = size % 8 == 0 ? size / 8 : size / 8 + 1;
    // std::cerr << "bitarraySize = " << bitarraySize << std::endl;
    uint8_t *tmp2 = new uint8_t[bitarraySize];
    uint8_t *tmp3 = new uint8_t[bitarraySize];
    // std::cerr << "bits = ";
    // for(int i = 0; i < size; ++i)
    // {
    //     std::cerr << arr[i + size] << "  ";
    // }
    // std::cerr << std::endl;
    packBitArray(arr + size, size, tmp2);
    // std::cerr << "encoded = " << (int)tmp2[0] << std::endl;
    if (parallel_reconstruct)
    {
        std::thread send_thread(&Peer::send_batched_input, peer, arr, size, bw);
        std::thread recv_thread(&Peer::recv_batched_input, peer, tmp, size, bw);
        send_thread.join();
        recv_thread.join();
        std::thread send_thread2(&Peer::send_uint8_array, peer, tmp2, bitarraySize);
        std::thread recv_thread2(&Peer::recv_uint8_array, peer, tmp3, bitarraySize);
        send_thread2.join();
        recv_thread2.join();
    }
    else
    {
        peer->send_batched_input(arr, size, bw);
        peer->send_uint8_array(tmp2, bitarraySize);
        peer->recv_batched_input(tmp, size, bw);
        peer->recv_uint8_array(tmp3, bitarraySize);
    }
    // std::cerr << "bits = ";
    for (int i = 0; i < size; i++)
    {
        arr[i] = arr[i] + tmp[i];
        // std::cerr << ((tmp3[i / 8] >> (i % 8)) & 1) << "  ";
        arr[i + size] = arr[i + size] + ((tmp3[i / 8] >> (i % 8)) & 1);
    }
    // std::cerr << std::endl;
    delete[] tmp;
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

    if (party == DEALER) {
        auto local_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < d0; ++i) {
            for(int j = 0; j < d1; ++j) {
                for(int k = 0; k < d2; ++k) {
                    for(int l = 0; l < d3; ++l) {
                        Arr4DIdxRowM(outArr_mask, d0, d1, d2, d3, i, j, k, l) = random_ge(bitlength);
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
                                                            local_start).count();
        dealerMicroseconds += local_time_taken;
        std::cerr << "   Dealer Time = " << local_time_taken / 1000.0 << " milliseconds\n";
    }
    else {

        auto keyread_start = std::chrono::high_resolution_clock::now();
        auto key = dealer->recv_conv2d_key(bitlength, bitlength, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW);
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();

        peer->sync();
        uint64_t eigen_start = eigenMicroseconds;
        auto local_start = std::chrono::high_resolution_clock::now();
        EvalConv2D(party, key, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr);
        auto t1 = std::chrono::high_resolution_clock::now();
        reconstruct(d0 * d1 * d2 * d3, outArr, bitlength);
        auto local_end = std::chrono::high_resolution_clock::now();
        
        freeConv2dKey(key);
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                            local_start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                            t1).count();
        convEvalMicroseconds += (reconstruct_time + compute_time);
        evalMicroseconds += (reconstruct_time + compute_time);
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "      Eigen Time = " << (eigenMicroseconds - eigen_start) / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
    }

    std::cerr << ">> Conv2D - End" << std::endl;

}

void Conv2DGroupWrapper(int64_t N, int64_t H, int64_t W,
                        int64_t CI, int64_t FH, int64_t FW,
                        int64_t CO, int64_t zPadHLeft,
                        int64_t zPadHRight, int64_t zPadWLeft,
                        int64_t zPadWRight, int64_t strideH,
                        int64_t strideW, int64_t G,
                        MASK_PAIR(GroupElement *inputArr), MASK_PAIR(GroupElement *filterArr), MASK_PAIR(GroupElement *outArr))
{
    if (G == 1) {
        Conv2DWrapper(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, inputArr_mask, filterArr, filterArr_mask, outArr, outArr_mask);
    }
    else {
        // TODO
        assert(false && "Conv2DGroup not implemented");
    }
}

void ScaleUp(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf)
{
    if (party == DEALER) {
        for(int i = 0; i < size; ++i) {
            inArr_mask[i] = inArr_mask[i] << sf;
        }
    }
    else {
        for(int i = 0; i < size; ++i) {
            inArr[i] = inArr[i] << sf;
        }
    }
}

void ars_threads_helper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *outArr, ARSKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalARS(party - 2, inArr[i], keys[i].shift, keys[i]);
        freeARSKeyPack(keys[i]);
    }
}

void ars_dealer_helper(int thread_idx, int size, int shift, GroupElement *inArr_mask, GroupElement *outArr_mask, pair<ARSKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr_mask[i] = random_ge(bitlength);
        keys[i] = keyGenARS(bitlength, bitlength, shift, inArr_mask[i], outArr_mask[i]);
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
    if (party == DEALER) {
        uint64_t dealer_time_taken = 0;
        for (int i = 0; i < size; i++) {
            auto dealer_start = std::chrono::high_resolution_clock::now();
            GroupElement rout = random_ge(bitlength);
            auto keys = keyGenARS(bitlength, bitlength, shift, inArr_mask[i], rout);
            outArr_mask[i] = rout;
            auto dealer_end = std::chrono::high_resolution_clock::now();
            dealer_time_taken += std::chrono::duration_cast<std::chrono::microseconds>(dealer_end -
                                                                dealer_start)
              .count();
            server->send_ars_key(keys.first);
            client->send_ars_key(keys.second);
            freeARSKeyPackPair(keys);
        }
        dealerMicroseconds += dealer_time_taken;
        std::cerr << "   Dealer Time = " << dealer_time_taken / 1000.0 << " milliseconds\n";
    }
    else {
        ARSKeyPack *keys = new ARSKeyPack[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++) {
            keys[i] = dealer->recv_ars_key(bitlength, bitlength, shift);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(ars_threads_helper, i, size, inArr, outArr, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto mid = std::chrono::high_resolution_clock::now();

        reconstruct(size, outArr, bitlength);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        arsEvalMicroseconds += (reconstruct_time + compute_time);
        delete[] keys;
    }
    std::cerr << ">> Truncate - End" << std::endl;
}

void ScaleDown(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf)
{
    std::cerr << ">> ScaleDown - Start " << std::endl;

    if (localTruncation) {
        uint64_t m = ((1L << sf) - 1) << (bitlength - sf);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++) {
            if (party == DEALER) {
                auto x_msb = msb(inArr_mask[i], bitlength);
                inArr_mask[i] = x_msb ? (inArr_mask[i] >> sf) | m : inArr_mask[i] >> sf;
                mod(inArr_mask[i], bitlength);
            }
            else {
                auto x_msb = msb(inArr[i], bitlength);
                inArr[i] = x_msb ? (inArr[i] >> sf) | m : inArr[i] >> sf;
                mod(inArr[i], bitlength);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto timeMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (party == DEALER) {
            dealerMicroseconds += timeMicroseconds;
        }
        else {
            evalMicroseconds += timeMicroseconds;
            arsEvalMicroseconds += timeMicroseconds;
            std::cerr << "   Eval Time = " << timeMicroseconds / 1000.0 << " milliseconds\n";
        }
    }
    else {
        ARS(size, inArr, inArr_mask, inArr, inArr_mask, sf);
    }
    std::cerr << ">> ScaleDown - End " << std::endl;
}

inline void matmul2d_server_helper(int thread_idx, int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C, GroupElement *a, GroupElement *b, GroupElement *c)
{
    auto p = get_start_end(s1 * s3, thread_idx);
    for(int ik = p.first; ik < p.second; ik += 1){
        int i = ik / s3;
        int k = ik % s3;
        Arr2DIdxRowM(C, s1, s3, i, k) = Arr2DIdxRowM(c, s1, s3, i, k);
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdxRowM(C, s1, s3, i, k) = Arr2DIdxRowM(C, s1, s3, i, k) - Arr2DIdxRowM(A, s1, s2, i, j) * Arr2DIdxRowM(b, s2, s3, j, k) - Arr2DIdxRowM(a, s1, s2, i, j) * Arr2DIdxRowM(B, s2, s3, j, k) + Arr2DIdxRowM(A, s1, s2, i, j) * Arr2DIdxRowM(B, s2, s3, j, k);
        }
        // mod(Arr2DIdxRowM(C, s1, s3, i, k));
    }

}

inline void matmul2d_client_helper(int thread_idx, int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C, GroupElement *a, GroupElement *b, GroupElement *c)
{
    auto p = get_start_end(s1 * s3, thread_idx);
    for(int ik = p.first; ik < p.second; ik += 1){
        int i = ik / s3;
        int k = ik % s3;
        Arr2DIdxRowM(C, s1, s3, i, k) = Arr2DIdxRowM(c, s1, s3, i, k);
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdxRowM(C, s1, s3, i, k) = Arr2DIdxRowM(C, s1, s3, i, k) - Arr2DIdxRowM(A, s1, s2, i, j) * Arr2DIdxRowM(b, s2, s3, j, k) - Arr2DIdxRowM(a, s1, s2, i, j) * Arr2DIdxRowM(B, s2, s3, j, k);
        }
        // mod(Arr2DIdxRowM(C, s1, s3, i, k));
    }

}

void MatMul2D(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA)
{
    std::cerr << ">> MatMul2D - Start" << std::endl;
    if (party == DEALER) {

        auto dealer_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < s1; ++i) {
            for(int j = 0; j < s3; ++j) {
                Arr2DIdxRowM(C_mask, s1, s3, i, j) = random_ge(bitlength);
            }
        }

        auto keys = KeyGenMatMul(bitlength, bitlength, s1, s2, s3, A_mask, B_mask, C_mask);
        auto dealer_end = std::chrono::high_resolution_clock::now();

        // server->send_matmul_key(keys.first);
        freeMatMulKey(keys.first);
        client->send_matmul_key(keys.second);
        freeMatMulKey(keys.second);
        dealerMicroseconds += std::chrono::duration_cast<std::chrono::microseconds>(dealer_end - dealer_start).count();
        std::cerr << "   Dealer Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(dealer_end - dealer_start).count() << " milliseconds" << std::endl;
    }
    else {

        auto keyread_start = std::chrono::high_resolution_clock::now();
        auto key = dealer->recv_matmul_key(bitlength, bitlength, s1, s2, s3);
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();

        peer->sync();
        uint64_t eigen_start = eigenMicroseconds;
        auto start = std::chrono::high_resolution_clock::now();
        matmul_eval_helper(party, s1, s2, s3, A, B, C, key.a, key.b, key.c);
        auto mid = std::chrono::high_resolution_clock::now();
        reconstruct(s1 * s3, C, bitlength);
        auto end = std::chrono::high_resolution_clock::now();

        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        evalMicroseconds += (compute_time + reconstruct_time);
        matmulEvalMicroseconds += (compute_time + reconstruct_time);
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "      Eigen Time = " << (eigenMicroseconds - eigen_start) / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        
        freeMatMulKey(key);
    }

    std::cerr << ">> MatMul2D - End" << std::endl;
}

void ElemWiseActModelVectorMult(int32_t size, MASK_PAIR(GroupElement *inArr),
                                MASK_PAIR(GroupElement *multArrVec), MASK_PAIR(GroupElement *outputArr))
{
    ElemWiseSecretSharedVectorMult(size, inArr, inArr_mask, multArrVec, multArrVec_mask, outputArr, outputArr_mask);
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
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                Arr2DIdxRowM(tmpMax_mask, rows, cols, i, j) = Arr2DIdxRowM(inp_mask, rows, cols, i, j);
                Arr2DIdxRowM(tmpIdx_mask, rows, cols, i, j) = 0;
            }
        }

        while(curCols > 1) {
            for(int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    Arr2DIdxRowM(drelu_mask, rows, curCols / 2, row, j) = random_ge(bitlength);
                    auto scmpKeys = keyGenSCMP(bitlength, bitlength, Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j), Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j + 1), Arr2DIdxRowM(drelu_mask, rows, curCols / 2, row, j));
                    server->send_scmp_keypack(scmpKeys.first);
                    client->send_scmp_keypack(scmpKeys.second);
                }
            }

            for (int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    
                    Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, row, j) = random_ge(bitlength);
                    auto multKeys1 = MultGen(Arr2DIdxRowM(drelu_mask, rows, curCols / 2, row, j), Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j) - Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j + 1), Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, row, j));
                    
                    server->send_mult_key(multKeys1.first);
                    client->send_mult_key(multKeys1.second);
                    
                    Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, rows + row, j) = random_ge(bitlength);
                    auto multKeys2 = MultGen(Arr2DIdxRowM(drelu_mask, rows, curCols / 2, row, j), Arr2DIdxRowM(tmpIdx_mask, rows, curCols, row, 2*j) - Arr2DIdxRowM(tmpIdx_mask, rows, curCols, row, 2*j + 1), Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, rows + row, j));
                    
                    server->send_mult_key(multKeys2.first);
                    client->send_mult_key(multKeys2.second);
                }
            }

            for (int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    Arr2DIdxRowM(tmpMax_mask, rows, curCols / 2, row, j) = Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, row, j) + Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j + 1);
                    Arr2DIdxRowM(tmpIdx_mask, rows, curCols / 2, row, j) = Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, rows + row, j) + Arr2DIdxRowM(tmpIdx_mask, rows, curCols, row, 2*j + 1);
                }
                if (curCols % 2 == 1) {
                    Arr2DIdxRowM(tmpMax_mask, rows, curCols / 2, row, curCols / 2) = Arr2DIdxRowM(tmpMax_mask, 2 * rows, curCols, row, curCols - 1);
                    Arr2DIdxRowM(tmpIdx_mask, rows, curCols / 2, row, curCols / 2) = Arr2DIdxRowM(tmpIdx_mask, 2 * rows, curCols, row, curCols - 1);
                }
            }

            curCols = (curCols + 1) / 2;
            round += 1;
        }

        for(int row = 0; row < rows; row++) {
            out_mask[row] = Arr2DIdxRowM(tmpIdx_mask, rows, 1, row, 0);
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

        ScmpKeyPack keys[(cols-1) * rows];
        MultKey mult_keys1[(cols-1) * rows];
        MultKey mult_keys2[(cols-1) * rows];
        int k1 = 0; int k2 = 0; int k3 = 0;

        int32_t curCols = cols;
        while(curCols > 1) {
            for(int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    keys[k1++] = dealer->recv_scmp_keypack(bitlength, bitlength);
                }
            }

            for (int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    mult_keys1[k2++] = dealer->recv_mult_key();
                    mult_keys2[k3++] = dealer->recv_mult_key();
                }
            }
            curCols = (curCols + 1) / 2;
        }

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        curCols = cols;
        k1 = 0; k2 = 0; k3 = 0;

        GroupElement *tmpMax = make_array<GroupElement>(rows, cols);
        GroupElement *tmpIdx = make_array<GroupElement>(rows, cols);
        
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                Arr2DIdxRowM(tmpMax, rows, cols, i, j) = Arr2DIdxRowM(inp, rows, cols, i, j);
                Arr2DIdxRowM(tmpIdx, rows, cols, i, j) = j;
            }
        }
        
        GroupElement *drelu = make_array<GroupElement>(rows, cols / 2);
        GroupElement *mult_res = make_array<GroupElement>(2 * rows, cols / 2);

        while(curCols > 1) {
            for(int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    Arr2DIdxRowM(drelu, rows, curCols / 2, row, j) = evalSCMP(party - 2, keys[k1++], Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j), Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j + 1));
                }
            }

            reconstruct(rows * (curCols / 2), drelu, bitlength);

            for (int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    
                    Arr2DIdxRowM(mult_res, 2 * rows, curCols / 2, row, j) = MultEval(party - 2, mult_keys1[k2++], Arr2DIdxRowM(drelu, rows, curCols / 2, row, j), Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j) - Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j + 1));
                    
                    Arr2DIdxRowM(mult_res, 2 * rows, curCols / 2, rows + row, j) = MultEval(party - 2, mult_keys2[k3++], 
                        Arr2DIdxRowM(drelu, rows, curCols / 2, row, j), 
                        Arr2DIdxRowM(tmpIdx, rows, curCols, row, 2*j) - Arr2DIdxRowM(tmpIdx, rows, curCols, row, 2*j + 1));
                }
            }

            reconstruct((2*rows) * (curCols / 2), mult_res, bitlength);

            for (int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    Arr2DIdxRowM(tmpMax, rows, curCols / 2, row, j) = Arr2DIdxRowM(mult_res, 2 * rows, curCols / 2, row, j) + Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j + 1);
                    Arr2DIdxRowM(tmpIdx, rows, curCols / 2, row, j) = Arr2DIdxRowM(mult_res, 2 * rows, curCols / 2, rows + row, j) + Arr2DIdxRowM(tmpIdx, rows, curCols, row, 2*j + 1);
                }
                if (curCols % 2 == 1) {
                    Arr2DIdxRowM(tmpMax, rows, curCols / 2, row, curCols / 2) = Arr2DIdxRowM(tmpMax, 2 * rows, curCols, row, curCols - 1);
                    Arr2DIdxRowM(tmpIdx, rows, curCols / 2, row, curCols / 2) = Arr2DIdxRowM(tmpIdx, 2 * rows, curCols, row, curCols - 1);
                }
            }

            curCols = (curCols + 1) / 2;
        }

        for(int row = 0; row < rows; row++) {
            out[row] = Arr2DIdxRowM(tmpIdx, rows, 1, row, 0);
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

void rt_threads_helper_1(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *tmp, ReluTruncateKeyPack *keys)
{
    GroupElement cache = 0;
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        tmp[i] = evalRT_lrs(party - 2, inArr[i], keys[i], cache);
        tmp[i+size] = evalRT_drelu(party - 2, inArr[i], keys[i], cache);
    }
}

void rt_threads_helper_2(int thread_idx, int32_t size, GroupElement *tmp, GroupElement *outArr, ReluTruncateKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalRT_mult(party - 2, tmp[i+size], tmp[i], keys[i]);
    }
}

void rt_dealer_threads_helper(int thread_idx, int32_t size, int sf, GroupElement *inArr_mask, GroupElement *drelu_mask, GroupElement *lrs_mask, GroupElement *outArr_mask, std::pair<ReluTruncateKeyPack, ReluTruncateKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        drelu_mask[i] = random_ge(1);
        lrs_mask[i] = random_ge(bitlength);
        outArr_mask[i] = random_ge(bitlength);
        keys[i] = keyGenReluTruncate(bitlength, bitlength, sf, inArr_mask[i], lrs_mask[i], drelu_mask[i], outArr_mask[i]);
    }
}

void ReluTruncate(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int sf, GroupElement *drelu_cache)
{
    std::cerr << ">> ReluTruncate - Start" << std::endl;
    GroupElement *tmp = make_array<GroupElement>(2*size);
    if (party == DEALER) {
        GroupElement *lrs_mask = tmp;
        GroupElement *drelu_mask = tmp + size;
        for(int i = 0; i < size; ++i) {
            drelu_mask[i] = random_ge(1);
            if (drelu_cache != nullptr)
                drelu_cache[i] = drelu_mask[i];
            lrs_mask[i] = random_ge(bitlength);
            GroupElement rout = random_ge(bitlength);
            auto keys = keyGenReluTruncate(bitlength, bitlength, sf, inArr_mask[i], lrs_mask[i], drelu_mask[i], rout);
            outArr_mask[i] = rout;
            server->send_relu_truncate_key(keys.first);
            client->send_relu_truncate_key(keys.second);
            freeReluTruncateKeyPackPair(keys);
        }
    }
    else {
        auto keyread_start = std::chrono::high_resolution_clock::now();
        ReluTruncateKeyPack *keys = new ReluTruncateKeyPack[size];
        for(int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_relu_truncate_key(bitlength, bitlength, sf);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        {
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i] = std::thread(rt_threads_helper_1, i, size, inArr, tmp, keys);
            }

            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i].join();
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        reconstructRT(size, tmp, bitlength);
        if (drelu_cache != nullptr)
            for(int i = 0; i < size; ++i) {
                drelu_cache[i] = tmp[i+size];
            }
        auto t2 = std::chrono::high_resolution_clock::now();
        {
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i] = std::thread(rt_threads_helper_2, i, size, tmp, outArr, keys);
            }

            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i].join();
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        reconstruct(size, outArr, bitlength);
        auto end = std::chrono::high_resolution_clock::now();
        
        uint64_t time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        reluTruncateEvalMicroseconds += time_taken;
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
        std::cerr << "   Online Time = " << (time1 + time2 + time3 + time4) / 1000.0 << " milliseconds" << std::endl;


        for(int i = 0; i < size; ++i) {
            freeReluTruncateKeyPack(keys[i]);
        }
        delete[] keys;
    }

    delete[] tmp;
    std::cerr << ">> ReluTruncate - End" << std::endl;
}

void r2_threads_helper_1(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *tmp, Relu2RoundKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        tmp[i] = evalRelu2_drelu(party - 2, inArr[i], keys[i]);
    }
}

void r2_threads_helper_2(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *tmp, GroupElement *outArr, Relu2RoundKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalRelu2_mult(party - 2, tmp[i], inArr[i], keys[i]);
    }
}

void Relu2Round(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu_cache, int effectiveInputBw)
{
    std::cerr << ">> Relu2Round - Start" << std::endl;
    GroupElement *tmp = make_array<GroupElement>(size);
    if (party == DEALER) {
        GroupElement *drelu_mask = tmp;
        for(int i = 0; i < size; ++i) {
            drelu_mask[i] = random_ge(1);
            if (drelu_cache != nullptr)
                drelu_cache[i] = drelu_mask[i];
            GroupElement rout = random_ge(bitlength);
            auto keys = keyGenRelu2Round(effectiveInputBw, bitlength, inArr_mask[i], drelu_mask[i], rout);
            outArr_mask[i] = rout;
            server->send_relu_2round_key(keys.first);
            client->send_relu_2round_key(keys.second);
            freeRelu2RoundKeyPackPair(keys);
        }
    }
    else {
        auto keyread_start = std::chrono::high_resolution_clock::now();
        Relu2RoundKeyPack *keys = new Relu2RoundKeyPack[size];
        for(int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_relu_2round_key(effectiveInputBw, bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        {
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i] = std::thread(r2_threads_helper_1, i, size, inArr, tmp, keys);
            }

            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i].join();
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        reconstruct(size, tmp, 1);
        if (drelu_cache != nullptr)
            for(int i = 0; i < size; ++i) {
                drelu_cache[i] = tmp[i];
            }
        auto t2 = std::chrono::high_resolution_clock::now();
        {
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i] = std::thread(r2_threads_helper_2, i, size, inArr, tmp, outArr, keys);
            }

            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i].join();
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        reconstruct(size, outArr, bitlength);
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


        for(int i = 0; i < size; ++i) {
            freeRelu2RoundKeyPack(keys[i]);
        }
        delete[] keys;
    }

    delete[] tmp;
    std::cerr << ">> Relu2Round - End" << std::endl;
}

void AvgPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr)) 
{
    // taken from the equivalent function in Porthos/src/EzPCFunctionalities.cpp
    std::cerr << ">> AvgPool - Start" << std::endl;
    int rows = N*H*W*C;
	std::vector<GroupElement> filterAvg(rows, 0);
    std::vector<GroupElement> filterAvg_mask(rows, 0);
    std::vector<GroupElement> outp(rows), outp_mask(rows);

    auto common_start = std::chrono::high_resolution_clock::now();
	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					GroupElement curFilterSum = 0, curFilterSum_mask = 0;
					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							GroupElement temp = 0, temp_mask = 0;
							if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
								temp = 0;
                                temp_mask = 0;
							}
							else{
								temp = Arr4DIdxRowM(inArr, N, imgH, imgW, C, n, curPosH, curPosW, c);
                                temp_mask = Arr4DIdxRowM(inArr_mask, N, imgH, imgW, C, n, curPosH, curPosW, c);
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
    if (party == DEALER) {
        dealerMicroseconds += common_time;
        std::cerr << "   Dealer Time (without PubDiv) = " << common_time / 1000.0 << " miliseconds" << std::endl;
    }
    else {
        avgpoolEvalMicroseconds += common_time;
        evalMicroseconds += common_time;
        std::cerr << "   Eval Time (without PubDiv) = " << common_time / 1000.0 << " miliseconds" << std::endl;
    }
    


    // The division should always be signed division.
    // Local division will introduce error
    
    bool doLocal = false;

    if (doLocal) {
        // following what porthos does: convert to signed and then back to unsigned
        // todo: check why this double negative trick works to output signed division
        // todo: assuming 64 bitlen here
        for (int rowIdx = 0; rowIdx < rows; rowIdx++) {
            if (party == DEALER) {
                filterAvg_mask[rowIdx] = static_cast<uint64_t>((static_cast<int64_t>(filterAvg_mask[rowIdx]))/(ksizeH*ksizeW));
            }
            else {
                filterAvg[rowIdx] = -static_cast<uint64_t>((static_cast<int64_t>(-filterAvg[rowIdx]))/(ksizeH*ksizeW));
            } 
        }                 	
    }
    else {
        // call fss protocol for division
        // todo: the divisor ksizeH * ksizeW is 32 bits long when passed as param, but ezpc cleartext explicitly converts to 64 bit value
        // will this be an issue in the future?
        // ElemWiseVectorPublicDiv(rows, filterAvg.data(), filterAvg_mask.data(), ksizeH * ksizeW, outp.data(), outp_mask.data());
        std::cerr << "Error Error Error" << std::endl;
        exit(1);
    }

	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					int iidx = n*C*H*W + c*H*W + h*W + w;
                    if (party == DEALER) {
                        Arr4DIdxRowM(outArr_mask, N, H, W, C, n, h, w, c) = outp_mask[iidx];
                    }
                    else {
					    Arr4DIdxRowM(outArr, N, H, W, C, n, h, w, c) = outp[iidx];
                    }
				}
			}
		}
	}
    std::cerr << ">> AvgPool - End" << std::endl;
}


void mult_threads_helper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *multArrVec, GroupElement *outArr, MultKey *keys)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = MultEval(party - SERVER, keys[i], inArr[i], multArrVec[i]);
    }
    auto thread_end = std::chrono::high_resolution_clock::now();
}

void ElemWiseSecretSharedVectorMult(int32_t size, MASK_PAIR(GroupElement *inArr),
                                    MASK_PAIR(GroupElement *multArrVec), MASK_PAIR(GroupElement *outputArr))
{
    std::cerr << ">> ElemWise Mult - start" << std::endl;
    if (party == DEALER) {
        uint64_t dealer_toal_time = 0;
        for(int i = 0; i < size; ++i) {
            auto dealer_start = std::chrono::high_resolution_clock::now();
            auto rout = random_ge(bitlength);
            auto keys = MultGen(inArr_mask[i], multArrVec_mask[i], rout);
            outputArr_mask[i] = rout;
            auto dealer_end = std::chrono::high_resolution_clock::now();
            dealer_toal_time += std::chrono::duration_cast<std::chrono::microseconds>(dealer_end - dealer_start).count();
            server->send_mult_key(keys.first);
            client->send_mult_key(keys.second);
        }
        dealerMicroseconds = dealerMicroseconds + dealer_toal_time;
    }
    else {
        MultKey *keys = new MultKey[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_mult_key();
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(mult_threads_helper, i, size, inArr, multArrVec, outputArr, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto mid = std::chrono::high_resolution_clock::now();
        reconstruct(size, outputArr, bitlength);
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

void maxpool_threads_helper(int thread_idx, int fh, int fw, int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, GroupElement *inArr, GroupElement *maxUntilNow, GroupElement *oneHot, MaxpoolKeyPack *keys)
{
    auto p = get_start_end(N * C * H * W, thread_idx);
    for(int i = p.first; i < p.second; i += 1) {
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
        
        GroupElement maxi = Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c);
        GroupElement temp;
        if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))) {
            temp = GroupElement(0);
        }
        else {
            temp = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
        }
        int kidx = (fh * FW + fw - 1) * (N * C * H * W) + i;
        Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = evalMaxpool(party - 2, maxi, temp, keys[kidx], Arr5DIdxRowM(oneHot, FH*FW-1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c));
        freeMaxpoolKeyPack(keys[kidx]);
    }
}

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot) 
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
    
    if (party == DEALER) {
        uint64_t dealer_file_read_time = 0;
        auto dealer_start = std::chrono::high_resolution_clock::now();
        for (int fh = 0; fh < FH; fh++) {
            for(int fw = 0; fw < FW; fw++) {
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for(int ctH = 0; ctH < H; ctH++) {
                            for(int ctW = 0; ctW < W; ctW++) {
                                int leftTopCornerH = ctH * strideH - zPadHLeft;
                                int leftTopCornerW = ctW * strideW - zPadWLeft;

                                if (fh == 0 && fw == 0) {
                                    if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW) {
                                        Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = GroupElement(0);
                                    }
                                    else {
                                        Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = Arr4DIdxRowM(inArr_mask, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                                    }
                                }
                                else {
                                    int curPosH = leftTopCornerH + fh;
                                    int curPosW = leftTopCornerW + fw;

                                    GroupElement maxi_mask = Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c);
                                    GroupElement temp_mask;
                                    if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))) {
                                        temp_mask = GroupElement(0);
                                    }
                                    else {
                                        temp_mask = Arr4DIdxRowM(inArr_mask, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
                                    }
                                    GroupElement rout = random_ge(bitlength);
                                    GroupElement routBit = random_ge(1);
                                    auto keys = keyGenMaxpool(bitlength, bitlength, maxi_mask, temp_mask, rout, routBit);
                                    Arr5DIdxRowM(oneHot, FH * FW - 1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c) = routBit;
                                    Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = rout;

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
    else {
        MaxpoolKeyPack *keys = new MaxpoolKeyPack[(FH * FW - 1) * N * C * H * W];
        int kidx = 0;
        uint64_t keysize_start = dealer->bytesReceived;
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for (int fh = 0; fh < FH; fh++) {
            for(int fw = 0; fw < FW; fw++) {
                if (fh == 0 && fw == 0) {
                    continue;
                }
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for(int ctH = 0; ctH < H; ctH++) {
                            for(int ctW = 0; ctW < W; ctW++) {
                                keys[kidx] = dealer->recv_maxpool_key(bitlength, bitlength);
                                kidx++;
                            }
                        }
                    }
                }
            }
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();
        auto keysize = dealer->bytesReceived - keysize_start;

        peer->sync();
        uint64_t timeCompute = 0;
        uint64_t timeReconstruct = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for(int ctH = 0; ctH < H; ctH++) {
                    for(int ctW = 0; ctW < W; ctW++) {
                        int leftTopCornerH = ctH * strideH - zPadHLeft;
                        int leftTopCornerW = ctW * strideW - zPadWLeft;
                        if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW) {
                            Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = 0;
                        }
                        else {
                            Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                        }
                    }
                }
            }
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        timeCompute += std::chrono::duration_cast<std::chrono::microseconds>(t0 - start).count();

        for (int fh = 0; fh < FH; fh++) {
            for(int fw = 0; fw < FW; fw++) {
                if (fh == 0 && fw == 0) {
                    continue;
                }

                auto t1 = std::chrono::high_resolution_clock::now();
                std::thread thread_pool[num_threads];
                
                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i] = std::thread(maxpool_threads_helper, i, fh, fw, 
                                        N, H, W, C, FH,
                                        FW, zPadHLeft, zPadHRight,
                                        zPadWLeft, zPadWRight, strideH,
                                        strideW, N1, imgH, imgW,
                                        C1,inArr, maxUntilNow, oneHot, keys);
                }

                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i].join();
                }

                auto t2 = std::chrono::high_resolution_clock::now();

                if (!(fh == 0 && fw == 0)) {
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
        timeReconstruct += std::chrono::duration_cast<std::chrono::microseconds>(end - t4).count();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        evalMicroseconds += eval_time;
        maxpoolEvalMicroseconds += eval_time;
        delete[] keys;
        std::cerr << "   Key Read Time = " << keyread_time / 1000.0 << " milliseconds" << std::endl;
        std::cerr << "   Key Size = " << keysize / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cerr << "   Compute Time = " << timeCompute / 1000.0 << " milliseconds" << std::endl;
        std::cerr << "   Reconstruct Time = " << timeReconstruct / 1000.0 << " milliseconds" << std::endl;
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;
    }

    std::cerr << ">> MaxPool - End" << std::endl;
}

// this API is not exposed directly, so we used single array for values and masks
// all APIs should do the same in principal
void InsecureInverse(int32_t size, GroupElement *A, GroupElement *invA, int32_t sf, int32_t upper) 
{
    // KG: make sure this is inplace secure (i.e can accept invA = A)
    uint64_t logk = osuCrypto::log2ceil(upper);
    uint64_t m = logk + 1;
    std::cerr << ">> InsecureInverse - start" << std::endl;

    if (party == DEALER) {
        for(int i = 0; i < size; ++i) {
            auto rout = random_ge(bitlength);
            auto keys = keyGenTaylor(bitlength, bitlength, 2.630, -5.857, 4.245, A[i], rout, sf, logk);
            server->send_taylor_key(keys.first, bitlength, m);
            client->send_taylor_key(keys.second, bitlength, m);
            invA[i] = rout;
        }
    }
    else {
        TaylorKeyPack *keys = new TaylorKeyPack[size];
        for(int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_taylor_key(bitlength, m, sf);
        }

        peer->sync();

        GroupElement *tmp = new GroupElement[2*size];
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < size; ++i) {
            auto tup = evalTaylor_round1(party - SERVER, bitlength, bitlength, 2.630, -5.857, 4.245, A[i], keys[i], sf, logk);
            tmp[i] = tup.first;
            tmp[i+size] = tup.second;
        }
        reconstruct(2*size, tmp, bitlength);


        for(int i = 0; i < size; ++i) {
            auto tup = evalTaylor_round2(party - SERVER, bitlength, bitlength, 2.630, -5.857, 4.245, A[i], keys[i], sf, logk, tmp[i], tmp[i+size]);
            tmp[i+size] = tup.first + tup.second;
        }
        reconstruct(size, tmp+size, bitlength);

        for(int i = 0; i < size; ++i) {
            auto tup = evalTaylor_round3(party - SERVER, bitlength, bitlength, 2.630, -5.857, 4.245, A[i], keys[i], sf, logk, tmp[i], tmp[i+size], tmp[i+size]);
            tmp[i+size] = tup;
        }
        reconstruct(size, tmp+size, bitlength);

        for(int i = 0; i < size; ++i) {
            invA[i] = tmp[i+size];
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " milliseconds" << std::endl;
        evalMicroseconds += eval_time;
        delete[] tmp;
    }
    std::cerr << ">> InsecureInverse - end" << std::endl;
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
    if (party == DEALER) {
        for(int i = 0; i < s1; ++i) {
            for(int j = 0; j < s2; ++j) {
                Arr2DIdxRowM(outArr_mask, s1, s2, i, j) = Arr2DIdxRowM(inArr_mask, s1, s2, i, j) - max[i];
            }
        }
    }
    else {
        for(int i = 0; i < s1; ++i) {
            for(int j = 0; j < s2; ++j) {
                Arr2DIdxRowM(outArr, s1, s2, i, j) = Arr2DIdxRowM(inArr, s1, s2, i, j) - max[i] + (1<<(sf + 1));
            }
        }
    }

    // step 3 - exponentiate each element in each image in batch 
    // e^x = RT((x+2), 1) for negative x
    ReluTruncate(s1 * s2, MASK_PAIR(outArr), MASK_PAIR(outArr), 1, nullptr); // Q: can we do this in place? can be a source of bug in future

    GroupElement *denominators = max; // reuse the array
    // // step 4 - calculate sum of exponentiated elements for each image in batch
    if (party == DEALER) {
        for(int i = 0; i < s1; ++i) {
            denominators[i] = 0;
            for(int j = 0; j < s2; ++j) {
                denominators[i] = denominators[i] + Arr2DIdxRowM(outArr_mask, s1, s2, i, j);
            }
            denominators[i] = denominators[i] * s1;
        }
    }
    else {
        for(int i = 0; i < s1; ++i) {
            denominators[i] = 0;
            for(int j = 0; j < s2; ++j) {
                denominators[i] = denominators[i] + Arr2DIdxRowM(outArr, s1, s2, i, j);
            }
            denominators[i] = denominators[i] * s1;
        }
    }

    // step 5 - calculate inverse of all the denominators
    InsecureInverse(s1, denominators, denominators, sf, s2 * s1);

    // step 6 - multiply each element in each image in batch by the inverse of the denominator
    GroupElement *expandedDenominator = make_array<GroupElement>(s1 * s2);
    for(int i = 0; i < s1; ++i) {
        for(int j = 0; j < s2; ++j) {
            Arr2DIdxRowM(expandedDenominator, s1, s2, i, j) = denominators[i];
        }
    }
    delete[] max;

    ElemWiseSecretSharedVectorMult(s1 * s2, expandedDenominator, expandedDenominator, MASK_PAIR(outArr), MASK_PAIR(outArr));
    ScaleDown(s1 * s2, MASK_PAIR(outArr), sf);
    std::cerr << ">> Softmax - end" << std::endl;

    delete[] expandedDenominator;
}

void select_threads_helper(int thread_idx, int32_t size, GroupElement *s, GroupElement *x, GroupElement *out, SelectKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        out[i] = evalSelect(party - 2, s[i], x[i], keys[i]);
    }
}

void Select(int32_t size, GroupElement *s, GroupElement *x, GroupElement *out) {
    std::cerr << ">> Select - start" << std::endl;
    if (party == DEALER) {
        for(int i = 0; i < size; ++i) {
            auto rout = random_ge(bitlength);
            auto keys = keyGenSelect(bitlength, s[i], x[i], rout);
            out[i] = rout;
            server->send_select_key(keys.first);
            client->send_select_key(keys.second);
        }
    }
    else {
        SelectKeyPack *keys = new SelectKeyPack[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_select_key(bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(select_threads_helper, i, size, s, x, out, keys);
        }
        
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto mid = std::chrono::high_resolution_clock::now();
        reconstruct(size, out, bitlength);

        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        selectEvalMicroseconds += (reconstruct_time + compute_time);
        delete[] keys;
    }
    std::cerr << ">> Select - end" << std::endl;
}

void reluHelper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *outArr, GroupElement *drelu, ReluKeyPack *keys)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalRelu(party - 2, inArr[i], keys[i], &drelu[i]);
        freeReluKeyPack(keys[i]);
    }
    auto thread_end = std::chrono::high_resolution_clock::now();
}

void relu_dealer_threads_helper(int thread_idx, int32_t size, GroupElement *inArr_mask, GroupElement *outArr_mask, GroupElement *drelu, std::pair<ReluKeyPack, ReluKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        auto rout = random_ge(bitlength); // prng inside multithreads, need some locking
        drelu[i] = random_ge(1);
        keys[i] = keyGenRelu(bitlength, bitlength, inArr_mask[i], rout, drelu[i]);
        outArr_mask[i] = rout;
    }
}

void Relu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu)
{
    std::cerr << ">> Relu (Spline) - Start" << std::endl;
    // todo: handle doTruncation param
    if (party == DEALER) {
        uint64_t dealer_total_time = 0;
        std::pair<ReluKeyPack, ReluKeyPack> *keys = new std::pair<ReluKeyPack, ReluKeyPack>[size];
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(relu_dealer_threads_helper, i, size, inArr_mask, outArr_mask, drelu, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        dealer_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for(int i = 0; i < size; ++i) {
            server->send_relu_key(keys[i].first);
            client->send_relu_key(keys[i].second);
            freeReluKeyPackPair(keys[i]);
        }
        delete[] keys;
        dealerMicroseconds += dealer_total_time;
        std::cerr << "   Dealer time = " << dealer_total_time / 1000.0 << " milliseconds" << std::endl;
    }
    else {
        // Step 1: Preprocessing Keys from Dealer
        ReluKeyPack *keys = new ReluKeyPack[size];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < size; i++){
            keys[i] = dealer->recv_relu_key(bitlength, bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(keyread_end -
                                                            keyread_start).count();
        // Step 2: Online Local ReLU Eval
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        if (num_threads == 1) {
            reluHelper(0, size, inArr, outArr, drelu, keys);
        }
        else {
            std::thread thread_pool[num_threads];
            for(int thread_idx = 0; thread_idx < num_threads; thread_idx++)
            {
                thread_pool[thread_idx] = std::thread(reluHelper, thread_idx, size, inArr, outArr, drelu, keys);
            }

            for(int thread_idx = 0; thread_idx < num_threads; thread_idx++)
            {
                thread_pool[thread_idx].join();
            }
        }

        auto mid = std::chrono::high_resolution_clock::now();
        // Step 3: Online Communication
        reconstruct(size, outArr, bitlength);
        reconstruct(size, drelu, 1);
        auto end = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        std::cerr << "   Key Read Time = " << keyread_time_taken << " milliseconds\n";
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " milliseconds\n";
        std::cerr << "   Online Time = " << (reconstruct_time + compute_time) / 1000.0 << " milliseconds\n";
        evalMicroseconds += (reconstruct_time + compute_time);
        reluEvalMicroseconds += (reconstruct_time + compute_time);
        delete[] keys;
    }
    std::cerr << ">> Relu (Spline) - End " << std::endl;
}

void maxpool_double_threads_helper_1(int thread_idx, int fh, int fw, int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, GroupElement *inArr, GroupElement *maxUntilNow, GroupElement *oneHot, MaxpoolDoubleKeyPack *keys)
{
    auto p = get_start_end(N * C * H * W, thread_idx);
    for(int i = p.first; i < p.second; i += 1) {
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
        
        GroupElement maxi = Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c);
        GroupElement temp;
        if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))) {
            temp = GroupElement(0);
        }
        else {
            temp = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
        }
        int kidx = (fh * FW + fw - 1) * (N * C * H * W) + i;
        Arr5DIdxRowM(oneHot, FH*FW-1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c) = evalMaxpoolDouble_1(party - 2, maxi, temp, keys[kidx]);
        // freeMaxpoolKeyPack(keys[kidx]);
    }
}

void maxpool_double_threads_helper_2(int thread_idx, int fh, int fw, int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, GroupElement *inArr, GroupElement *maxUntilNow, GroupElement *oneHot, MaxpoolDoubleKeyPack *keys)
{
    auto p = get_start_end(N * C * H * W, thread_idx);
    for(int i = p.first; i < p.second; i += 1) {
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
        
        GroupElement maxi = Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c);
        GroupElement temp;
        if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))) {
            temp = GroupElement(0);
        }
        else {
            temp = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
        }
        int kidx = (fh * FW + fw - 1) * (N * C * H * W) + i;
        // Arr5DIdxRowM(oneHot, FH*FW-1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c) = evalMaxpoolDouble_1(party - 2, maxi, temp, keys[kidx]);
        GroupElement s = Arr5DIdxRowM(oneHot, FH*FW-1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c);
        Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = evalMaxpoolDouble_2(party - 2, maxi, temp, s, keys[kidx]);
        freeMaxpoolDoubleKeyPack(keys[kidx]);
    }
}

void MaxPoolDouble(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot) 
{
    std::cerr << ">> MaxPoolDouble - Start" << std::endl;
    int d1 = ((imgH - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((imgW - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    always_assert(d1 == H);
    always_assert(d2 == W);
    always_assert(N1 == N);
    always_assert(C1 == C);

    GroupElement *maxUntilNow = outArr;
    GroupElement *maxUntilNow_mask = outArr_mask;
    
    if (party == DEALER) {
        uint64_t dealer_file_read_time = 0;
        auto dealer_start = std::chrono::high_resolution_clock::now();
        for (int fh = 0; fh < FH; fh++) {
            for(int fw = 0; fw < FW; fw++) {
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for(int ctH = 0; ctH < H; ctH++) {
                            for(int ctW = 0; ctW < W; ctW++) {
                                int leftTopCornerH = ctH * strideH - zPadHLeft;
                                int leftTopCornerW = ctW * strideW - zPadWLeft;

                                if (fh == 0 && fw == 0) {
                                    if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW) {
                                        Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = GroupElement(0);
                                    }
                                    else {
                                        Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = Arr4DIdxRowM(inArr_mask, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                                    }
                                }
                                else {
                                    int curPosH = leftTopCornerH + fh;
                                    int curPosW = leftTopCornerW + fw;

                                    GroupElement maxi_mask = Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c);
                                    GroupElement temp_mask;
                                    if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))) {
                                        temp_mask = GroupElement(0);
                                    }
                                    else {
                                        temp_mask = Arr4DIdxRowM(inArr_mask, N1, imgH, imgW, C1, n, curPosH, curPosW, c);
                                    }
                                    GroupElement rout = random_ge(bitlength);
                                    GroupElement routBit = random_ge(1);
                                    auto keys = keyGenMaxpoolDouble(bitlength, bitlength, maxi_mask, temp_mask, routBit, rout);
                                    Arr4DIdxRowM(maxUntilNow_mask, N, H, W, C, n, ctH, ctW, c) = rout;
                                    Arr5DIdxRowM(oneHot, FH * FW - 1, N, H, W, C, fh * FW + fw - 1, n, ctH, ctW, c) = routBit;

                                    auto read_start = std::chrono::high_resolution_clock::now();
                                    server->send_maxpool_double_key(keys.first);
                                    client->send_maxpool_double_key(keys.second);
                                    freeMaxpoolDoubleKeyPackPair(keys);
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
    else {
        MaxpoolDoubleKeyPack *keys = new MaxpoolDoubleKeyPack[(FH * FW - 1) * N * C * H * W];
        int kidx = 0;
        for (int fh = 0; fh < FH; fh++) {
            for(int fw = 0; fw < FW; fw++) {
                if (fh == 0 && fw == 0) {
                    continue;
                }
                for (int n = 0; n < N; n++) {
                    for (int c = 0; c < C; c++) {
                        for(int ctH = 0; ctH < H; ctH++) {
                            for(int ctW = 0; ctW < W; ctW++) {
                                keys[kidx] = dealer->recv_maxpool_double_key(bitlength, bitlength);
                                kidx++;
                            }
                        }
                    }
                }
            }
        }

        peer->sync();
        auto start_eval = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for(int ctH = 0; ctH < H; ctH++) {
                    for(int ctW = 0; ctW < W; ctW++) {
                        int leftTopCornerH = ctH * strideH - zPadHLeft;
                        int leftTopCornerW = ctW * strideW - zPadWLeft;
                        if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW) {
                            Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = 0;
                        }
                        else {
                            Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c);
                        }
                    }
                }
            }
        }

        for (int fh = 0; fh < FH; fh++) {
            for(int fw = 0; fw < FW; fw++) {
                if (fh == 0 && fw == 0) {
                    continue;
                }

                std::thread thread_pool[num_threads];
                
                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i] = std::thread(maxpool_double_threads_helper_1, i, fh, fw, 
                                        N, H, W, C, FH,
                                        FW, zPadHLeft, zPadHRight,
                                        zPadWLeft, zPadWRight, strideH,
                                        strideW, N1, imgH, imgW,
                                        C1,inArr, maxUntilNow, oneHot, keys);
                }

                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i].join();
                }

                reconstruct(N * C * H * W, oneHot + (fh * FW + fw - 1) * N * C * H * W, 1);

                std::thread thread_pool2[num_threads];

                for(int i = 0; i < num_threads; ++i) {
                    thread_pool2[i] = std::thread(maxpool_double_threads_helper_2, i, fh, fw, 
                                        N, H, W, C, FH,
                                        FW, zPadHLeft, zPadHRight,
                                        zPadWLeft, zPadWRight, strideH,
                                        strideW, N1, imgH, imgW,
                                        C1,inArr, maxUntilNow, oneHot, keys);
                }

                for(int i = 0; i < num_threads; ++i) {
                    thread_pool2[i].join();
                }

                reconstruct(N * C * H * W, maxUntilNow, bitlength);
            }
        }
        auto end_eval = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end_eval - start_eval).count();
        evalMicroseconds += eval_time;
        maxpoolEvalMicroseconds += eval_time;
        delete[] keys;
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;
    }

    std::cerr << ">> MaxPoolDouble - End" << std::endl;
}

#define BIG_LOOPY(e) for(int n = 0; n < N; ++n) {\
        for(int h = 0; h < H; ++h) {\
            for(int w = 0; w < W; ++w) {\
                for(int c = 0; c < C; ++c) {\
                    e;\
                }\
            }\
        }\
    }


void maxpool_onehot_threads_helper(int thread_idx, int f, int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, GroupElement *maxBits, GroupElement *curr, GroupElement *oneHot, BitwiseAndKeyPack *keys)
{
    auto p = get_start_end(N * H * W * C, thread_idx);
    for(int i = p.first; i < p.second; i += 1) {
        int curridx = i;
        int c = curridx % C;
        curridx = curridx / C;
        int w = curridx % W;
        curridx = curridx / W;
        int h = curridx % H;
        curridx = curridx / H;
        int n = curridx % N;
        curridx = curridx / N;

        auto max = Arr5DIdxRowM(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
        auto c1 = Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c);
        auto key = keys[(FH * FW - 2 - f) * N * H * W * C + n * H * W * C + h * W * C + w * C + c];
        Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = evalAnd(party - 2, max, 1 ^ c1, key);
        mod(Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, f, n, h, w, c), 1);
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
    if (party == DEALER) {
        BIG_LOOPY(
            auto m = Arr5DIdxRowM(maxBits, FH * FW - 1, N, H, W, C, FH * FW - 2, n, h, w, c);
            Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c) = m;
            Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, FH * FW - 1, n, h, w, c) = m;
        )

        for(int f = FH * FW - 2; f >= 1; --f) {
            // out[f] = max[f - 1] ^ !curr
            BIG_LOOPY(
                auto max = Arr5DIdxRowM(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
                auto c1 = Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c);
                auto rout = random_ge(1);
                auto keys = keyGenBitwiseAnd(max, c1, rout);
                server->send_bitwise_and_key(keys.first);
                client->send_bitwise_and_key(keys.second);
                Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = rout;
            )
            
            BIG_LOOPY(
                Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c) = Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c) ^ Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, f, n, h, w, c);
            )
        }

        BIG_LOOPY(
            Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, 0, n, h, w, c) = Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c);
        )
    }
    else {
        BitwiseAndKeyPack *keys = new BitwiseAndKeyPack[(FH * FW - 2) * N * H * W * C];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < (FH * FW - 2) * N * H * W * C; ++i) {
            keys[i] = dealer->recv_bitwise_and_key();
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        BIG_LOOPY(
            auto m = Arr5DIdxRowM(maxBits, FH * FW - 1, N, H, W, C, FH * FW - 2, n, h, w, c);
            Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c) = m;
            Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, FH * FW - 1, n, h, w, c) = m;
        )

        for(int f = FH * FW - 2; f >= 1; --f) {
            
            // out[f] = max[f - 1] ^ !curr
            BIG_LOOPY(
                auto max = Arr5DIdxRowM(maxBits, FH * FW - 1, N, H, W, C, f - 1, n, h, w, c);
                auto c1 = Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c);
                auto key = keys[(FH * FW - 2 - f) * N * H * W * C + n * H * W * C + h * W * C + w * C + c];
                Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, f, n, h, w, c) = evalAnd(party - 2, max, 1 ^ c1, key);
                mod(Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, f, n, h, w, c), 1);
            )

            // std::thread thread_pool[num_threads];
            // for(int i = 0; i < num_threads; ++i) {
            //     thread_pool[i] = std::thread(maxpool_onehot_threads_helper, i, f, N, H, W, C, FH, FW, maxBits, curr, oneHot, keys);
            // }

            // for(int i = 0; i < num_threads; ++i) {
            //     thread_pool[i].join();
            // }

            reconstruct(N * H * W * C, oneHot + f * N * H * W * C, 1);
            
            BIG_LOOPY(
                Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c) = Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c) ^ Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, f, n, h, w, c);
            )
        }

        BIG_LOOPY(
            Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, 0, n, h, w, c) = Arr4DIdxRowM(curr, N, H, W, C, n, h, w, c) ^ 1;
        )
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

void MaxPoolBackward(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot)
{
    std::cerr << ">> MaxPoolBackward - Start" << std::endl;
    // currently not very confident about maxpool with padding, hence this assert
    always_assert((zPadHLeft == 0) && (zPadHRight == 0) && (zPadWLeft == 0) && (zPadWRight == 0));

    if (party == DEALER)
    {
        for(int n = 0; n < N; ++n) {
            for(int h = 0; h < imgH; ++h) {
                for(int w = 0; w < imgW; ++w) {
                    for(int c = 0; c < C; ++c) {
                        Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, h, w, c) = 0;
                    }
                }
            }
        }

        BIG_LOOPY(
            int leftTopCornerH = h * strideH - zPadHLeft;
            int leftTopCornerW = w * strideW - zPadWLeft;
            auto src = Arr4DIdxRowM(outArr, N, H, W, C, n, h, w, c);
            for(int fh = 0; fh < FH; ++fh) {
                for(int fw = 0; fw < FW; ++fw) {
                    if ((leftTopCornerH + fh >= 0) && (leftTopCornerH + fh < imgH) && (leftTopCornerW + fw >= 0) && (leftTopCornerW + fw < imgW)) {
                        auto s = Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, fh * FW + fw, n, h, w, c);
                        auto dst = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c);
                        // dst = dst + select(s, src)
                        auto rout = random_ge(bitlength);
                        auto keys = keyGenSelect(bitlength, s, src, rout);
                        Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c) = dst + rout;
                        server->send_select_key(keys.first);
                        client->send_select_key(keys.second);
                    }
                }
            }
        )
    }
    else 
    {
        SelectKeyPack *keys = new SelectKeyPack[FH * FW * N * H * W * C];
        auto keyread_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < FH * FW * N * H * W * C; ++i) {
            keys[i] = dealer->recv_select_key(bitlength);
        }
        auto keyread_end = std::chrono::high_resolution_clock::now();
        auto keyread_time = std::chrono::duration_cast<std::chrono::microseconds>(keyread_end - keyread_start).count();

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        for(int n = 0; n < N; ++n) {
            for(int h = 0; h < imgH; ++h) {
                for(int w = 0; w < imgW; ++w) {
                    for(int c = 0; c < C; ++c) {
                        Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, h, w, c) = 0;
                    }
                }
            }

            for(int h = 0; h < H; ++h) {
                for(int w = 0; w < W; ++w) {
                    for(int c = 0; c < C; ++c) {
                        int leftTopCornerH = h * strideH - zPadHLeft;
                        int leftTopCornerW = w * strideW - zPadWLeft;
                        auto src = Arr4DIdxRowM(outArr, N, H, W, C, n, h, w, c);
                        for(int fh = 0; fh < FH; ++fh) {
                            for(int fw = 0; fw < FW; ++fw) {
                                auto s = Arr5DIdxRowM(oneHot, FH * FW, N, H, W, C, fh * FW + fw, n, h, w, c);
                                auto dst = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c);
                                // dst = dst + select(s, src)
                                auto key = keys[n * H * W * C * FH * FW + h * W * C * FH * FW + w * C * FH * FW + c * FH * FW + fh * FW + fw];
                                // auto key = dealer->recv_select_key(bitlength);
                                Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, leftTopCornerH + fh, leftTopCornerW + fw, c) = dst + evalSelect(party - 2, s, src, key);
                            }
                        }
                    }
                }
            }
        }

        auto mid = std::chrono::high_resolution_clock::now();
        reconstruct(N1 * imgH * imgW * C1, inArr, bitlength);
        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        auto reconstruct_time = std::chrono::duration_cast<std::chrono::microseconds>(end - mid).count();
        auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(mid - start).count();
        evalMicroseconds += eval_time;
        selectEvalMicroseconds += eval_time;
        std::cerr << "   Key Read Time = " << keyread_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Compute Time = " << compute_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Reconstruct Time = " << reconstruct_time / 1000.0 << " miliseconds" << std::endl;
        std::cerr << "   Online Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;

        delete[] keys;
    }
    std::cerr << ">> MaxPoolBackward - End" << std::endl;
}

void FixToFloat(int size, GroupElement *inp, GroupElement *out, int scale)
{
    std::cerr << ">> FixToFloat - Start" << std::endl;
    GroupElement *p = new GroupElement[2*bitlength];
    GroupElement *q = new GroupElement[2*bitlength];
    fill_pq(p, q, bitlength);
    // std::cout << "intervals = {" << std::endl;
    // for(int i = 0; i < 2*bitlength; ++i) {
    //     std::cout << "  " << i + 1 << "= [" << p[i] << ", " << q[i] << "]" << std::endl;
    // }
    // std::cout << "}" << std::endl;

    if (party == DEALER) {
        for(int i = 0; i < size; ++i) {
            auto keys = keyGenFixToFloat(bitlength, scale, inp[i], p, q);
            out[4*i] = 0;
            out[4*i+1] = 0;
            out[4*i+2] = 0;
            out[4*i+3] = 0;
            server->send_fix_to_float_key(keys.first, bitlength);
            client->send_fix_to_float_key(keys.second, bitlength);
        }
    }
    else {
        FixToFloatKeyPack *keys = new FixToFloatKeyPack[size];
        for(int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_fix_to_float_key(bitlength);
        }
        GroupElement *pow = new GroupElement[size];
        GroupElement *sm = new GroupElement[size];
        GroupElement *ym = new GroupElement[size];

        peer->sync();
        for(int i = 0; i < size; ++i) {
            evalFixToFloat_1(party - 2, bitlength, scale, inp[i], keys[i], p, q, out[i*4 + 0], out[i*4 + 1], out[i*4 + 2], out[i*4 + 3], pow[i], sm[i]);
        }

        delete[] keys;

        reconstruct(size, sm, 1);
        reconstruct(size, pow, bitlength);

        for(int i = 0; i < size; ++i) {
            ym[i] = 2 * evalSelect(party - 2, 1 ^ sm[i], inp[i], keys[i].selectKey);
            if (party == 2) {
                ym[i] = ym[i] - inp[i];
            }
        }

        reconstruct(size, ym, bitlength);

        for(int i = 0; i < size; ++i) {
            out[i*4 + 0] = -keys[i].ry * pow[i] - keys[i].rpow * ym[i] + keys[i].rm; 
            if (party == 2) {
                out[i*4 + 0] = out[i*4 + 0] + ym[i] * pow[i];
                out[i*4 + 0] = -((-out[i*4+0]) >> (bitlength - scale));
            }
            else {
                out[i*4 + 0] = out[i*4 + 0] >> (bitlength - scale);
            }
        }

        reconstruct(4*size, out, bitlength);
    }
    std::cerr << ">> FixToFloat - End" << std::endl;
}
