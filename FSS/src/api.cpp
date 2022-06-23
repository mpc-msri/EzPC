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
#include "comms.h"
#include "utils.h"
#include "array.h"
#include "conv.h"
#include "spline.h"
#include "mult.h"
#include "pubdiv.h"
#include "dcf.h"
#include "mini_aes.h"
#include "input_prng.h"
#include <cassert>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <thread>
#include <Eigen/Dense>

template <typename T> using pair = std::pair<T,T>;

uint64_t evalMicroseconds = 0;
uint64_t reconstructMicroseconds = 0;
uint64_t arsEvalMicroseconds = 0;
uint64_t matmulEvalMicroseconds = 0;
uint64_t reluEvalMicroseconds = 0;
uint64_t convEvalMicroseconds = 0;
uint64_t maxpoolEvalMicroseconds = 0;
uint64_t avgpoolEvalMicroseconds = 0;
uint64_t pubdivEvalMicroseconds = 0;
uint64_t argmaxEvalMicroseconds = 0;
uint64_t multEvalMicroseconds = 0;

uint64_t dealerMicroseconds = 0;
uint64_t inputOfflineComm = 0;
uint64_t inputOnlineComm = 0;

bool localTruncation = false;

void StartComputation()
{
    std::cerr << "=== COMPUTATION START ===\n\n";
    std::cerr << "bitlength = " << bitlength << std::endl;
    std::cerr << "local truncation = " << (localTruncation ? "yes" : "no") << std::endl << std::endl;
    aes_init();

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
        always_assert(server->bytesSent == 16);
        always_assert(server->bytesSent == 16);
        server->bytesSent = 0;
        client->bytesSent = 0;
    }

    if (party == DEALER) {
        auto commonSeed = aes_enc(toBlock(0, time(NULL)), 0);
        server->send_block(commonSeed);
        prngShared.SetSeed(commonSeed);
    }
    else if (party == SERVER) {
        auto commonSeed = dealer->recv_block();
        prngShared.SetSeed(commonSeed);
    }
}

int32_t numRounds = 0;
void EndComputation()
{
    std::cerr << "\n=== COMPUTATION END ===\n\n";
    if (party != DEALER) {
        std::cerr << "Offline Communication = " << inputOfflineComm << " bytes\n";
        std::cerr << "Offline Time = " << accumulatedInputTimeOffline / 1000.0 << " milliseconds\n";
        std::cerr << "Online Rounds = " << numRounds << "\n";
        std::cerr << "Online Communication = " << peer->bytesSent + peer->bytesReceived + inputOnlineComm << " bytes\n";
        std::cerr << "Online Time = " << (evalMicroseconds + accumulatedInputTimeOnline) / 1000.0 << " milliseconds\n\n";
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

extern int num_threads;

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

        auto key = dealer->recv_conv2d_key(bitlength, bitlength, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW);

        peer->sync();
        auto local_start = std::chrono::high_resolution_clock::now();
        EvalConv2D(party, key, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr);
        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();
        reconstruct(d0 * d1 * d2 * d3, outArr, bitlength);
        auto local_end = std::chrono::high_resolution_clock::now();
        
        freeConv2dKey(key);
        auto local_time_taken = std::chrono::duration_cast<std::chrono::microseconds>(local_end -
                                                            t2).count();
        local_time_taken += std::chrono::duration_cast<std::chrono::microseconds>(t1 -
                                                            local_start).count();
        convEvalMicroseconds += local_time_taken;
        evalMicroseconds += local_time_taken;
        std::cerr << "   Eval Time = " << local_time_taken / 1000.0 << " milliseconds\n";
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

void ARS(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t shift)
{
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
        for (int i = 0; i < size; i++) {
            keys[i] = dealer->recv_ars_key(bitlength, bitlength, shift);
        }

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(ars_threads_helper, i, size, inArr, outArr, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();

        reconstruct(size, outArr, bitlength);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                            t2).count();
        time_taken += std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();
        std::cerr << "   Eval Time: " << time_taken / 1000.0 << " milliseconds" << std::endl;
        evalMicroseconds += time_taken;
        arsEvalMicroseconds += time_taken;
        delete[] keys;
    }
}

void ScaleDown(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf)
{
    std::cerr << ">> ScaleDown - Start " << std::endl;

    if (localTruncation) {
        uint64_t m = ((1L << sf) - 1) << (bitlength - sf);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++) {
            if (party == DEALER) {
                auto msb = inArr_mask[i][0];
                inArr_mask[i].value = msb ? (inArr_mask[i].value >> sf) | m : inArr_mask[i].value >> sf;
                mod(inArr_mask[i]);
            }
            else {
                auto msb = inArr[i][0];
                inArr[i].value = msb ? (inArr[i].value >> sf) | m : inArr[i].value >> sf;
                mod(inArr[i]);
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
        Arr2DIdxRowM(C, s1, s3, i, k).value = Arr2DIdxRowM(c, s1, s3, i, k).value;
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdxRowM(C, s1, s3, i, k).value = Arr2DIdxRowM(C, s1, s3, i, k).value - Arr2DIdxRowM(A, s1, s2, i, j).value * Arr2DIdxRowM(b, s2, s3, j, k).value - Arr2DIdxRowM(a, s1, s2, i, j).value * Arr2DIdxRowM(B, s2, s3, j, k).value + Arr2DIdxRowM(A, s1, s2, i, j).value * Arr2DIdxRowM(B, s2, s3, j, k).value;
        }
        mod(Arr2DIdxRowM(C, s1, s3, i, k));
    }

}

inline void matmul2d_client_helper(int thread_idx, int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C, GroupElement *a, GroupElement *b, GroupElement *c)
{
    auto p = get_start_end(s1 * s3, thread_idx);
    for(int ik = p.first; ik < p.second; ik += 1){
        int i = ik / s3;
        int k = ik % s3;
        Arr2DIdxRowM(C, s1, s3, i, k).value = Arr2DIdxRowM(c, s1, s3, i, k).value;
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdxRowM(C, s1, s3, i, k).value = Arr2DIdxRowM(C, s1, s3, i, k).value - Arr2DIdxRowM(A, s1, s2, i, j).value * Arr2DIdxRowM(b, s2, s3, j, k).value - Arr2DIdxRowM(a, s1, s2, i, j).value * Arr2DIdxRowM(B, s2, s3, j, k).value;
        }
        mod(Arr2DIdxRowM(C, s1, s3, i, k));
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

        auto key = dealer->recv_matmul_key(bitlength, bitlength, s1, s2, s3);

        peer->sync();
        auto start_eval = std::chrono::high_resolution_clock::now();
        matmul_eval_helper(s1, s2, s3, A, B, C, key.a, key.b, key.c);

        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();

        reconstruct(s1 * s3, C, bitlength);

        auto end_eval = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(t1 - start_eval).count();
        eval_time += std::chrono::duration_cast<std::chrono::microseconds>(end_eval - t2).count();
        evalMicroseconds += eval_time;
        matmulEvalMicroseconds += eval_time;
        std::cerr << "   Eval time: " << eval_time / 1000.0 << " milliseconds" << std::endl;
        
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
                Arr2DIdxRowM(tmpIdx_mask, rows, cols, i, j).value = 0;
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
                Arr2DIdxRowM(tmpIdx, rows, cols, i, j).value = j;
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

void reluHelper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *outArr, ReluKeyPack *keys)
{
    auto thread_start = std::chrono::high_resolution_clock::now();
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalRelu(party - 2, inArr[i], keys[i]);
        freeReluKeyPack(keys[i]);
    }
    auto thread_end = std::chrono::high_resolution_clock::now();
}

void relu_dealer_threads_helper(int thread_idx, int32_t size, GroupElement *inArr_mask, GroupElement *outArr_mask, std::pair<ReluKeyPack, ReluKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr_mask[i] = random_ge(bitlength); // prng inside multithreads, need some locking
        keys[i] = keyGenRelu(bitlength, bitlength, inArr_mask[i], outArr_mask[i]);
    }
}

void Relu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int sf,
          bool doTruncation)
{
    std::cerr << ">> Relu - Start" << std::endl;
    // todo: handle doTruncation param
    if (party == DEALER) {
        uint64_t dealer_total_time = 0;
#ifdef DEALER_DIRECT_SEND
        for(int i = 0; i < size; i++){
            auto dealer_time_start = std::chrono::high_resolution_clock::now();
            outArr_mask[i] = random_ge(bitlength);
            auto keys = keyGenRelu(bitlength, bitlength, inArr_mask[i], outArr_mask[i]);
            auto dealer_time_end = std::chrono::high_resolution_clock::now();
            dealer_total_time += std::chrono::duration_cast<std::chrono::microseconds>(dealer_time_end - dealer_time_start).count();
            server->send_relu_key(keys.first);
            client->send_relu_key(keys.second);
            freeReluKeyPackPair(keys);
        }
#else
        std::pair<ReluKeyPack, ReluKeyPack> *keys = new std::pair<ReluKeyPack, ReluKeyPack>[size];
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(relu_dealer_threads_helper, i, size, inArr_mask, outArr_mask, keys);
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
#endif
        dealerMicroseconds += dealer_total_time;
        std::cerr << "   Dealer time = " << dealer_total_time / 1000.0 << " milliseconds" << std::endl;
    }
    else {
        // Step 1: Preprocessing Keys from Dealer
        ReluKeyPack *keys = new ReluKeyPack[size];
        for(int i = 0; i < size; i++){
            keys[i] = dealer->recv_relu_key(bitlength, bitlength);
        }
        // Step 2: Online Local ReLU Eval
        peer->sync();
        auto start_eval = std::chrono::high_resolution_clock::now();
        if (num_threads == 1) {
            reluHelper(0, size, inArr, outArr, keys);
        }
        else {
            std::thread thread_pool[num_threads];
            for(int thread_idx = 0; thread_idx < num_threads; thread_idx++)
            {
                thread_pool[thread_idx] = std::thread(reluHelper, thread_idx, size, inArr, outArr, keys);
            }

            for(int thread_idx = 0; thread_idx < num_threads; thread_idx++)
            {
                thread_pool[thread_idx].join();
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();

        // Step 3: Online Communication
        reconstruct(size, outArr, bitlength);
        
        auto end_eval = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end_eval - t2).count();
        eval_time += std::chrono::duration_cast<std::chrono::microseconds>(t1 - start_eval).count();
        evalMicroseconds += eval_time;
        reluEvalMicroseconds += eval_time;
        std::cerr << "   Eval time: " << eval_time / 1000.0 << " milliseconds" << std::endl;

        delete[] keys;
    }
    std::cerr << ">> Relu - End " << std::endl;
    
    if (doTruncation) {
        ScaleDown(size, outArr, outArr_mask, sf);
    }
}

void maxpool_threads_helper(int thread_idx, int fh, int fw, int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, GroupElement *inArr, GroupElement *maxUntilNow, MaxpoolKeyPack *keys)
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
        Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c) = evalMaxpool(party - 2, maxi, temp, keys[kidx]);
        freeMaxpoolKeyPack(keys[kidx]);
    }
}

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr)) 
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
                                    auto keys = keyGenMaxpool(bitlength, bitlength, maxi_mask, temp_mask, rout);
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

        peer->sync();
        auto start_eval = std::chrono::high_resolution_clock::now();
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for(int ctH = 0; ctH < H; ctH++) {
                    for(int ctW = 0; ctW < W; ctW++) {
                        int leftTopCornerH = ctH * strideH - zPadHLeft;
                        int leftTopCornerW = ctW * strideW - zPadWLeft;
                        if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= imgH || leftTopCornerW >= imgW) {
                            Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c).value = 0;
                        }
                        else {
                            Arr4DIdxRowM(maxUntilNow, N, H, W, C, n, ctH, ctW, c).value = Arr4DIdxRowM(inArr, N1, imgH, imgW, C1, n, leftTopCornerH, leftTopCornerW, c).value;
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
                    thread_pool[i] = std::thread(maxpool_threads_helper, i, fh, fw, 
                                        N, H, W, C, FH,
                                        FW, zPadHLeft, zPadHRight,
                                        zPadWLeft, zPadWRight, strideH,
                                        strideW, N1, imgH, imgW,
                                        C1,inArr, maxUntilNow, keys);
                }

                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i].join();
                }

                if (!(fh == 0 && fw == 0)) {
                    reconstruct(N * C * H * W, maxUntilNow, bitlength);
                }
            }
        }
        auto end_eval = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end_eval - start_eval).count();
        evalMicroseconds += eval_time;
        maxpoolEvalMicroseconds += eval_time;
        delete[] keys;
        std::cerr << "   Eval Time = " << eval_time / 1000.0 << " miliseconds" << std::endl;
    }

    std::cerr << ">> MaxPool - End" << std::endl;
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
                filterAvg_mask[rowIdx] = static_cast<uint64_t>((static_cast<int64_t>(filterAvg_mask[rowIdx].value))/(ksizeH*ksizeW));
            }
            else {
                filterAvg[rowIdx] = -static_cast<uint64_t>((static_cast<int64_t>(-filterAvg[rowIdx].value))/(ksizeH*ksizeW));
            } 
        }                 	
    }
    else {
        // call fss protocol for division
        // todo: the divisor ksizeH * ksizeW is 32 bits long when passed as param, but ezpc cleartext explicitly converts to 64 bit value
        // will this be an issue in the future?
        ElemWiseVectorPublicDiv(rows, filterAvg.data(), filterAvg_mask.data(), ksizeH * ksizeW, outp.data(), outp_mask.data());
        
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

void div_first_threads_helper(int thread_idx, int32_t size, int32_t divisor, GroupElement *inArr, GroupElement *outArr, GroupElement *w_share, GroupElement *publicICresult_share, SignedPublicDivKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalSignedPublicDiv_First(party - 2, keys[i], inArr[i], w_share[i], publicICresult_share[i]);
    }
}

void div_second_threads_helper(int thread_idx, int32_t size, int32_t divisor, GroupElement *inArr, GroupElement *outArr, GroupElement *w_share, GroupElement *publicICresult_share, SignedPublicDivKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalSignedPublicDiv_Second(party - 2, keys[i], inArr[i], outArr[i], w_share[i], publicICresult_share[i]);
    }
}

void ElemWiseVectorPublicDiv(int32_t s1, MASK_PAIR(GroupElement *arr1), int32_t divisor,
                             MASK_PAIR(GroupElement *outArr)) 
{
    std::cerr << ">> ElemWiseVectorPublicDiv - Start" << std::endl;
    // if divisor is a power of 2, call fss truncate else call fss pubdiv

    assert(divisor > 0);
    bool is_powerof2 = ((divisor & (divisor - 1)) == 0);

    if (is_powerof2) {

        int shift = 0;
        while (divisor != 1) {
            divisor = divisor >> 1;
            shift++;
        }
        std::cerr << "   Divisor is power of 2 - using ARS" << std::endl;
        ARS(s1, arr1, arr1_mask, outArr, outArr_mask, shift);
    }
    else {
        if (party == DEALER) {
            uint64_t dealer_toal_time = 0;
            auto keys = new pair<SignedPublicDivKeyPack>[s1];

            auto dealer_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < s1; i++) {
                GroupElement rin = arr1_mask[i], rout = random_ge(bitlength), rout1 = random_ge(bitlength);
                keys[i] = keyGenSignedPublicDiv(bitlength, bitlength, rin, rout1, rout, GroupElement(divisor));
                outArr_mask[i] = rout;
            }
            auto dealer_end = std::chrono::high_resolution_clock::now();
           
            for (int i = 0; i < s1; i++) {
                server->send_signedpubdiv_key(keys[i].first);
                client->send_signedpubdiv_key(keys[i].second);
                // TODO: clear key memory
            }
            
            dealer_toal_time += std::chrono::duration_cast<std::chrono::microseconds>(dealer_end - dealer_start).count();
            dealerMicroseconds = dealerMicroseconds + dealer_toal_time;
            std::cerr << "   Dealer time: " << dealer_toal_time / 1000.0 << " milliseconds" << std::endl;
        }
        else {
            SignedPublicDivKeyPack* keys = new SignedPublicDivKeyPack[s1];
            GroupElement* w_share = new GroupElement[s1];
            GroupElement* publicICresult_share = new GroupElement[s1];

            // round 1 of fss pubdiv
            for (int i = 0; i < s1; i++) {
                keys[i] = dealer->recv_signedpubdiv_key(bitlength, bitlength);
            }

            peer->sync();
            auto start = std::chrono::high_resolution_clock::now();
            {
                std::thread thread_pool[num_threads];
                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i] = std::thread(div_first_threads_helper, i, s1, divisor, arr1, outArr, w_share, publicICresult_share, keys);
                }

                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i].join();
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            peer->sync();
            auto t2 = std::chrono::high_resolution_clock::now();
            // reconstruct
            reconstruct(s1, outArr, bitlength);

            // round 2 of fss pubdiv
            {
                std::thread thread_pool[num_threads];
                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i] = std::thread(div_second_threads_helper, i, s1, divisor, arr1, outArr, w_share, publicICresult_share, keys);
                }

                for(int i = 0; i < num_threads; ++i) {
                    thread_pool[i].join();
                }
            }

            auto t3 = std::chrono::high_resolution_clock::now();
            peer->sync();
            auto t4 = std::chrono::high_resolution_clock::now();
            // reconstruct
            reconstruct(s1, outArr, bitlength);
            auto end = std::chrono::high_resolution_clock::now();
            uint64_t eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - t4).count() + std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() + std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();
            evalMicroseconds = evalMicroseconds + eval_time;
            pubdivEvalMicroseconds = pubdivEvalMicroseconds + eval_time;
            std::cerr << "   Eval Time: " << eval_time / 1000.0 << " milliseconds" << std::endl;
            delete[] keys;
        }
    }
    std::cerr << ">> ElemWiseVectorPublicDiv - End" << std::endl;
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
            outputArr_mask[i] = random_ge(bitlength);
            auto keys = MultGen(inArr_mask[i], multArrVec_mask[i], outputArr_mask[i]);
            auto dealer_end = std::chrono::high_resolution_clock::now();
            dealer_toal_time += std::chrono::duration_cast<std::chrono::microseconds>(dealer_end - dealer_start).count();
            server->send_mult_key(keys.first);
            client->send_mult_key(keys.second);
        }
        dealerMicroseconds = dealerMicroseconds + dealer_toal_time;
    }
    else {
        MultKey *keys = new MultKey[size];

        for(int i = 0; i < size; ++i) {
            keys[i] = dealer->recv_mult_key();
        }

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(mult_threads_helper, i, size, inArr, multArrVec, outputArr, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();

        reconstruct(size, outputArr, bitlength);
        auto end = std::chrono::high_resolution_clock::now();
        auto eval_time = std::chrono::duration_cast<std::chrono::microseconds>(end - t2).count() + std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();
        std::cerr << "   Eval Time: " << eval_time / 1000.0 << " milliseconds" << std::endl;
        evalMicroseconds += eval_time;
        multEvalMicroseconds += eval_time;
        delete[] keys;

    }
    std::cerr << ">> ElemWise Mult - end" << std::endl;
}


void Floor(int32_t s1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t sf) 
{
    assert(false && "Floor not implemented");
}
