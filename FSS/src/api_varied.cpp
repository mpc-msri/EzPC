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

#include "api_varied.h"
#include "api.h"
#include "comms.h"
#include "conv.h"
#include "api.h"
#include "mult.h"
#include "dcf.h"
#include "group_element.h"
#include "mini_aes.h"
#include "pubdiv.h"
#include "spline.h"
#include "utils.h"
#include "input_prng.h"
#include <thread>

extern int num_threads;

struct {
uint64_t 
    truncateFix = 0,
    scalarMul = 0,
    matmul = 0,
    matadd = 0,
    mulcir = 0,
    matbroadcast = 0,
    shiftleft = 0,
    sigmoid = 0,
    tanh = 0,
    sqrt = 0,
    argmax = 0;
} evaluatorStats;

uint64_t matmulOfflineTime = 0;

uint64_t inputOfflineCommVaried = 0;
uint64_t inputOnlineCommVaried = 0;

template <typename T> using pair = std::pair<T,T>;

void initialize()
{
    std::cerr << "=== COMPUTATION START ===\n\n";
    aes_init();

    if (party != DEALER) {
        if (party == SERVER) {
            inputOfflineCommVaried = peer->bytesSent;
            inputOnlineCommVaried = peer->bytesReceived;
        }
        else {
            inputOfflineCommVaried = peer->bytesReceived;
            inputOnlineCommVaried = peer->bytesSent;
        }
        peer->bytesSent = 0;
        peer->bytesReceived = 0;
    }
    else {
        // std::cerr << "masks sent to server: " << server->bytesSent << std::endl;
        // std::cerr << "masks sent to client: " << client->bytesSent << std::endl;
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

// #define PRINT_LAYERWISE_STATS

void finalize()
{
    std::cerr << "\n=== COMPUTATION END ===\n\n";
#ifdef PRINT_LAYERWISE_STATS
    std::cerr << "evaluatorStats.truncateFix: " << evaluatorStats.truncateFix << std::endl;
    std::cerr << "evaluatorStats.scalarMul: " << evaluatorStats.scalarMul << std::endl;
    std::cerr << "evaluatorStats.matmul: " << evaluatorStats.matmul << std::endl;
    std::cerr << "evaluatorStats.matadd: " << evaluatorStats.matadd << std::endl;
    std::cerr << "evaluatorStats.mulcir: " << evaluatorStats.mulcir << std::endl;
    std::cerr << "evaluatorStats.matbroadcast: " << evaluatorStats.matbroadcast << std::endl;
    std::cerr << "evaluatorStats.shiftleft: " << evaluatorStats.shiftleft << std::endl;
    std::cerr << "evaluatorStats.sigmoid: " << evaluatorStats.sigmoid << std::endl;
    std::cerr << "evaluatorStats.sqrt: " << evaluatorStats.sqrt << std::endl;
    std::cerr << "evaluatorStats.tanh: " << evaluatorStats.tanh << std::endl;
    std::cerr << "evaluatorStats.argmax: " << evaluatorStats.argmax << std::endl;
    std::cerr << std::endl;
#endif
    auto totalTime = evaluatorStats.truncateFix + evaluatorStats.scalarMul + evaluatorStats.matmul + evaluatorStats.matadd + evaluatorStats.mulcir + evaluatorStats.matbroadcast + evaluatorStats.shiftleft + evaluatorStats.sigmoid + evaluatorStats.tanh + evaluatorStats.argmax + evaluatorStats.sqrt;

    if (party != DEALER) {
        std::cerr << "Offline Communication = " << inputOfflineCommVaried << " bytes\n";
        std::cerr << "Offline Time = " << (accumulatedInputTimeOffline + matmulOfflineTime) / 1000.0 << " milliseconds\n";
        std::cerr << "Online Rounds = " << numRounds << "\n";
        std::cerr << "Online Communication = " << peer->bytesSent + peer->bytesReceived + inputOnlineCommVaried << " bytes\n";
        std::cerr << "Online Time = " << (totalTime + accumulatedInputTimeOnline) / 1000.0 << " milliseconds\n\n";
    }
    else {
        std::cerr << "Offline Communication = " << server->bytesSent + client->bytesSent << " bytes\n";
        std::cerr << "Offline Time = " << (totalTime + accumulatedInputTimeOffline) / 1000.0 << " milliseconds\n";
    }
    std::cerr << "=========\n";

}

int64_t log(int64_t x)
{
   for(int i = 0; i < 64; i++)
   {
        if ((1L << i) == x)
        {
            return i;
        }
   }
   return -1;
}

int64_t ceillog(int64_t x)
{
    for(int i = 0; i < 64; i++)
    {
        if ((1L << i) >= x)
        {
            return i;
        }
    }
    return -1;
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

GroupElement* make_ge_array(int size, int bw)
{
    GroupElement* arr = new GroupElement[size];
    for(int i = 0; i < size; i++)
    {
        arr[i].bitsize = bw;
        arr[i].value = 0;
    }
    return arr;
}

void fix_bitwidth_threads_helper(int thread_idx, GroupElement* arr, int size, int bw)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i++)
    {
        arr[i].bitsize = bw;
        mod(arr[i]);
    }
}

void fix_bitwidth(GroupElement* arr, int size, int bw)
{
    std::thread thread_pool[num_threads];
    for(int i = 0; i < num_threads; i++)
    {
        thread_pool[i] = std::thread(fix_bitwidth_threads_helper, i, arr, size, bw);
    }
    for(int i = 0; i < num_threads; i++)
    {
        thread_pool[i].join();
    }
}

inline uint64_t sign_extend_clear(uint64_t x, int b1, int b2)
{
    uint64_t m1 = (1L << b1) - 1;
    uint64_t m2 = (1L << b2) - 1;

    return (((x + (1<<(b1- 1))) & m1) - (1<<(b1- 1))) & m2;
}

inline uint64_t truncate_reduce_clear(uint64_t x, int b1, int b2, int s)
{
    if (s == 0)
    {
        return sign_extend_clear(x, b1, b2);
    }
    else
    {
        uint8_t msb = x & (1L << (b1 - 1)) ? 1 : 0;
        return ((x >> s) - ((1L << (b1 - s)) * msb)) & ((1<<b2) - 1);
    }
}

void internalExtend_threads_helper(int thread_idx, int32_t size, int bin, int bout, GroupElement *inArr, GroupElement *outArr, DCFKeyPack *dcfKeys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        uint64_t xpval = inArr[i].value ^ (1L<<(bin - 1));
        GroupElement xp(xpval, bin);
        GroupElement t(0, bout);
        evalDCF(party - SERVER, &t, xp, dcfKeys[i]);
        freeDCFKeyPack(dcfKeys[i]);
        mod(t);
        outArr[i] = GroupElement((party - SERVER) * (xpval - (1L<<(bin - 1))) + t.value * (1L<<bin), bout);
        mod(outArr[i]);
    }
}

void internalReduce_threads_helper(int thread_idx, int32_t size, int bin, int bout, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr))
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1) {
        if (party == DEALER) {
            outArr_mask[i].value = sign_extend_clear(inArr_mask[i].value, bin, bout);
            outArr_mask[i].bitsize = bout;
        }
        else {
            outArr[i].value = sign_extend_clear(inArr[i].value, bin, bout);
            outArr[i].bitsize = bout;
        }
    }
}

void internalExtend_dealer_threads_helper(int threads_idx, int size, int bin, int bout, GroupElement *inArr_mask, GroupElement *outArr_mask, pair<DCFKeyPack> *keys)
{
    auto p = get_start_end(size, threads_idx);
    GroupElement one(1, bout);
    for(int i = p.first; i < p.second; i += 1) {
        keys[i] = keyGenDCF(bin, bout, 1, inArr_mask[i], &one);
        outArr_mask[i].value = inArr_mask[i].value;
    }
}

void internalExtend(int size, int bin, int bout, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), bool doReconstruct)
{
    if (bin == bout)
    {   
        auto start = std::chrono::high_resolution_clock::now();
        if (party == DEALER) {
            for(int i = 0; i < size; i++)
            {
                outArr_mask[i].value = inArr_mask[i].value;
                outArr_mask[i].bitsize = bout;
            }
        }
        else {
            for(int i = 0; i < size; i++)
            {
                outArr[i].value = inArr[i].value;
                outArr[i].bitsize = bout;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return;
    }

    if (party == DEALER)
    {
        fix_bitwidth(outArr_mask, size, bout);
        fix_bitwidth(inArr_mask, size, bin);
    }
    else
    {
        fix_bitwidth(outArr, size, bout);
        fix_bitwidth(inArr, size, bin);
    }
    
    if (bout < bin)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(internalReduce_threads_helper, i, size, bin, bout, inArr, inArr_mask, outArr, outArr_mask);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        return;
    }

    GroupElement one(1, bout);

    if (party == DEALER) {
#ifdef DEALER_DIRECT_SEND
        for (int i = 0; i < size; i++) {
            auto keys = keyGenDCF(bin, bout, 1, inArr_mask[i], &one);
            outArr_mask[i].value = inArr_mask[i].value;
            server->send_dcf_keypack(keys.first);
            client->send_dcf_keypack(keys.second);
            freeDCFKeyPackPair(keys);
        }
#else
        std::pair<DCFKeyPack, DCFKeyPack> *keys = new std::pair<DCFKeyPack, DCFKeyPack>[size];

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(internalExtend_dealer_threads_helper, i, size, bin, bout, inArr_mask, outArr_mask, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for(int i = 0; i < size; ++i) {
            server->send_dcf_keypack(keys[i].first);
            client->send_dcf_keypack(keys[i].second);
            freeDCFKeyPackPair(keys[i]);
        }

#endif
    }
    else {
        DCFKeyPack *keys = new DCFKeyPack[size];
        for (int i = 0; i < size; i++) {
            keys[i] = dealer->recv_dcf_keypack(bin, bout, 1);
        }

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(internalExtend_threads_helper, i, size, bin, bout, inArr, outArr, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }

        if (doReconstruct)
            reconstruct(size, outArr, bout);

        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        delete[] keys;
    }
}

void internalTF_threads_helper(int thread_idx, int32_t size, GroupElement *inArr, GroupElement *outArr, ARSKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        outArr[i] = evalARS(party - 2, inArr[i], keys[i].shift, keys[i]);
        freeARSKeyPack(keys[i]);
    }
}

void internalTF_dealer_threads_helper(int threads_idx, int size, int bin, int bout, int shift, GroupElement *inArr_mask, GroupElement *outArr_mask, pair<ARSKeyPack> *keys)
{
    auto p = get_start_end(size, threads_idx);
    for(int i = p.first; i < p.second; i += 1){
        GroupElement rout = random_ge(bout);
        keys[i] = keyGenARS(bin, bout, shift, inArr_mask[i], rout);
        outArr_mask[i] = rout;
    }
}

void internalTruncateAndFix(int size, int shift, int bin, int bout, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), bool doReconstruct)
{
    if (shift == 0)
    {
        internalExtend(size, bin, bout, inArr, inArr_mask, outArr, outArr_mask, doReconstruct);
        return;
    }
    
    if (party == DEALER) {
        fix_bitwidth(inArr_mask, size, bin);
        fix_bitwidth(outArr_mask, size, bout);
    }
    else {
        fix_bitwidth(inArr, size, bin);
        fix_bitwidth(outArr, size, bout);
    }

    if (party == DEALER) {
#ifdef DEALER_DIRECT_SEND
        for (int i = 0; i < size; i++) {
            GroupElement rout = random_ge(bout);
            auto keys = keyGenARS(bin, bout, shift, inArr_mask[i], rout);
            outArr_mask[i] = rout;
            server->send_ars_key(keys.first);
            client->send_ars_key(keys.second);
            freeARSKeyPackPair(keys);
        }
#else
        pair<ARSKeyPack> *keys = new pair<ARSKeyPack>[size];

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(internalTF_dealer_threads_helper, i, size, bin, bout, shift, inArr_mask, outArr_mask, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for(int i = 0; i < size; ++i) {
            server->send_ars_key(keys[i].first);
            client->send_ars_key(keys[i].second);
            freeARSKeyPackPair(keys[i]);
        }
        delete[] keys;
#endif
    }
    else {
        ARSKeyPack *keys = new ARSKeyPack[size];
        for (int i = 0; i < size; i++) {
            keys[i] = dealer->recv_ars_key(bin, bout, shift);
        }

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];
        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(internalTF_threads_helper, i, size, inArr, outArr, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }

        if (doReconstruct)
            reconstruct(size, outArr, bout);
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        delete[] keys;
    }
}

void ScalarMul_threads_helper(int thread_idx, int32_t size, uint64_t s, int bwB, int bwTemp, GroupElement *B, GroupElement *tmpC, DCFKeyPack *dcfKeys, GroupElement *r)
{
    auto p = get_start_end(size, thread_idx);
    uint64_t mB = (1 << bwB) - 1;
    for(int i = p.first; i < p.second; i += 1){
        uint64_t xp = (B[i].value + (1<<(bwB-1))) & mB;
        GroupElement t(0, bwTemp);
        evalDCF(party - SERVER, &t, GroupElement(xp, bwB), dcfKeys[i]);
        freeDCFKeyPack(dcfKeys[i]);
        tmpC[i] = (party - SERVER) * s * (xp - (1<<(bwB-1))) + r[i] + s * (1 << bwB) * t;
    }
}

void ScalarMul_dealer_threads_helper(int thread_idx, int size, uint64_t s, int bwB, int bwTemp, GroupElement *B_mask, GroupElement *tmpC_mask, pair<DCFKeyPack> *dcfkeys, pair<GroupElement> *r)
{
    GroupElement one1(1, bwTemp);
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        dcfkeys[i] = keyGenDCF(bwB, bwTemp, 1, B_mask[i], &one1);
        tmpC_mask[i] = random_ge(bwTemp);
        r[i] = splitShare(tmpC_mask[i] - s * B_mask[i].value);
    }
}

// A is public assuming mask is zero, for gods sake...
void ScalarMul(int64_t I, int64_t J, int64_t shrA, int64_t shrB, int64_t demote,
               int64_t bwA, int64_t bwB, int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement A),
               MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C))
{
    fix_bitwidth(&A, 1, bwA);
    fix_bitwidth(&A_mask, 1, bwA);
    // Convert the public s to bwTemp
    uint64_t s = sign_extend_clear(party == DEALER ? A_mask.value : A.value, bwA, bwTemp);

    fix_bitwidth(B, I*J, bwB);
    fix_bitwidth(B_mask, I*J, bwB);
    fix_bitwidth(C, I*J, bwC);
    fix_bitwidth(C_mask, I*J, bwC);
    
    GroupElement *temp = make_ge_array(I*J, bwTemp);
    GroupElement *tmpC, *tmpC_mask;
    if (party == DEALER) {
        tmpC_mask = temp;
    }
    else
    {
        tmpC = temp;
    }

    uint64_t mA = (1ULL << bwA) - 1;
    uint64_t mB = (1ULL << bwB) - 1;
    uint64_t mT = (1ULL << bwTemp) - 1;
    uint64_t mC = (1ULL << bwC) - 1;
    int shift = log(shrA) + log(shrB) + log(demote);

    if (bwTemp <= bwB)
    {
        // no need of dcf
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < I*J; ++i)
        {
            if (party == DEALER) {
                tmpC_mask[i].value = (s * B_mask[i].value) & mT;
            }
            else
            {
                tmpC[i].value = (s * sign_extend_clear(B[i].value, bwB, bwTemp)) & mT;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.scalarMul += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    else
    {
        // need dcf
        if (party == DEALER)
        {
#ifdef DEALER_DIRECT_SEND
            GroupElement one1(1, bwTemp);
            for(int i = 0; i < I*J; ++i) 
            {
                auto k = keyGenDCF(bwB, bwTemp, 1, B_mask[i], &one1);
                tmpC_mask[i] = random_ge(bwTemp);
                auto r = splitShare(tmpC_mask[i] - s * B_mask[i].value);
                server->send_dcf_keypack(k.first);
                server->send_ge(r.first, bwTemp);
                client->send_dcf_keypack(k.second);
                client->send_ge(r.second, bwTemp);
                // freeDCFKeyPackPair(k);
            }
#else
            int size = I*J;
            pair<DCFKeyPack> *dcfkeys = new pair<DCFKeyPack>[size];
            pair<GroupElement> *r = new pair<GroupElement>[size];

            auto start = std::chrono::high_resolution_clock::now();
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i] = std::thread(ScalarMul_dealer_threads_helper, i, size, s, bwB, bwTemp, B_mask, tmpC_mask, dcfkeys, r);
            }

            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i].join();
            }
            auto end = std::chrono::high_resolution_clock::now();
            evaluatorStats.scalarMul += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            for(int i = 0; i < size; ++i) {
                server->send_dcf_keypack(dcfkeys[i].first);
                server->send_ge(r[i].first, bwTemp);
                client->send_dcf_keypack(dcfkeys[i].second);
                client->send_ge(r[i].second, bwTemp);
                freeDCFKeyPackPair(dcfkeys[i]);
            }
            delete[] dcfkeys;
            delete[] r;
#endif
        }
        else
        {
            DCFKeyPack *dcfKeys = new DCFKeyPack[I*J];
            GroupElement *r = new GroupElement[I*J];

            for(int i = 0; i < I*J; ++i)
            {
                dcfKeys[i] = dealer->recv_dcf_keypack(bwB, bwTemp, 1);
                r[i] = dealer->recv_ge(bwTemp);
            }

            peer->sync();

            auto start = std::chrono::high_resolution_clock::now();
            std::thread thread_pool[num_threads];
            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i] = std::thread(ScalarMul_threads_helper, i, I*J, s, bwB, bwTemp, B, tmpC, dcfKeys, r);
            }

            for(int i = 0; i < num_threads; ++i) {
                thread_pool[i].join();
            }

            reconstruct(I*J, tmpC, bwTemp);
            auto end = std::chrono::high_resolution_clock::now();
            evaluatorStats.scalarMul += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            delete[] dcfKeys;
            delete[] r;

        }
    }

    internalTruncateAndFix(I*J, shift, bwTemp, bwC, tmpC, tmpC_mask, C, C_mask);
    delete[] temp;
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

void MatMul_internal(int bw, int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA)
{
    if (party == DEALER) {

        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < s1; ++i) {
            for(int j = 0; j < s3; ++j) {
                Arr2DIdxRowM(C_mask, s1, s3, i, j) = random_ge(bw);
            }
        }

        auto keys = KeyGenMatMul(bw, bw, s1, s2, s3, A_mask, B_mask, C_mask);
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.matmul += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // server->send_matmul_key(keys.first);
        freeMatMulKey(keys.first);
        client->send_matmul_key(keys.second);
        freeMatMulKey(keys.second);
    }
    else {

        auto offline_start = std::chrono::high_resolution_clock::now();
        auto key = dealer->recv_matmul_key(bw, bw, s1, s2, s3);
        auto offline_end = std::chrono::high_resolution_clock::now();
        if (party == SERVER) {
            matmulOfflineTime += std::chrono::duration_cast<std::chrono::microseconds>(offline_end - offline_start).count();
        }

        peer->sync();
        
        auto start = std::chrono::high_resolution_clock::now();
        matmul_eval_helper(s1, s2, s3, A, B, C, key.a, key.b, key.c);

        reconstruct(s1 * s3, C, bw);

        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.matmul += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        freeMatMulKey(key);
    }
}

void MatMulUniform(int bw, int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C))
{
    MatMul_internal(bw, s1, s2, s3, A, A_mask, B, B_mask, C, C_mask, false);
}

// #define SIRNN_STYLE_TRUNCATION_MATMUL

void MatMul(int64_t I, int64_t K, int64_t J, int64_t shrA, int64_t shrB,
            int64_t H1, int64_t H2, int64_t demote, int32_t bwA, int32_t bwB,
            int32_t bwTemp, int32_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C),
            MASK_PAIR(GroupElement *tmp), bool verbose)
{
    // we dont use bwTemp at all..
    // always_assert(bwTemp == bwA + bwB + ceillog(K));
    bwTemp = bwA + bwB + ceillog(K);
    GroupElement *memPool = make_ge_array(I*K + K*J, bwTemp);
    GroupElement *tmpA = memPool;
    GroupElement *tmpB = memPool + I*K;
    GroupElement *tmpC = make_ge_array(I*J, bwTemp);
    GroupElement *tmpA_mask, *tmpB_mask, *tmpC_mask;
    if (party == DEALER) {
        tmpA_mask = tmpA;
        tmpB_mask = tmpB;
        tmpC_mask = tmpC;
    }

    internalExtend(I*K, bwA, bwTemp, A, A_mask, tmpA, tmpA_mask, false);
    internalExtend(K*J, bwB, bwTemp, B, B_mask, tmpB, tmpB_mask, false);

    if (party != DEALER) {
        auto start = std::chrono::high_resolution_clock::now();
        if (bwTemp > bwA) {
            if (bwTemp > bwB) {
                reconstruct(I*K + K*J, memPool, bwTemp);
            }
            else {
                reconstruct(I*K, tmpA, bwTemp);
            }
        }
        else if (bwTemp > bwB) {
            reconstruct(K*J, tmpB, bwTemp);
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    MatMul_internal(bwTemp, I, K, J, tmpA, tmpA_mask, tmpB, tmpB_mask, tmpC, tmpC_mask, false);
    delete[] memPool;

    int32_t shiftA = log(shrA);
    int32_t shiftB = log(shrB);
    int32_t shiftDemote = log(demote);
    int32_t depth = ceillog(K);
#ifdef SIRNN_STYLE_TRUNCATION_MATMUL
    if (shiftA + shiftB + shiftDemote + H1 - depth > 0) {

        internalTruncateAndFix(I*J, shiftA + shiftB + shiftDemote + H1, bwTemp, bwC, tmpC, tmpC_mask, C, C_mask);
    } 
    else {
        internalTruncateAndFix(I*J, depth, bwTemp, bwC, tmpC, tmpC_mask, C, C_mask);
        AdjustScaleShl(I, J, (1L<<(depth - shiftA - shiftB - shiftDemote - H1)), C, C_mask);
    }
#else
    internalTruncateAndFix(I*J, shiftA + shiftB + shiftDemote + H1, bwTemp, bwC, tmpC, tmpC_mask, C, C_mask);
#endif
    delete[] tmpC;
}

inline bool needReconstruct(int shift, int bwA, int bwB) {
    if (shift > 0) {
        return true;
    }
    else {
        return bwB > bwA;
    }
}

void MatAdd_threads_helper(int thread_idx, int32_t size, MASK_PAIR(GroupElement *tmpA), MASK_PAIR(GroupElement *tmpB))
{
    auto p = get_start_end(size, thread_idx);
    if (party == DEALER) {
        for(int i = p.first; i < p.second; i += 1){
            tmpA_mask[i] = tmpA_mask[i] + tmpB_mask[i];
        }
    }
    else {
        for(int i = p.first; i < p.second; i += 1){
            tmpA[i] = tmpA[i] + tmpB[i];
        }
    }
}

void MatAdd(int64_t I, int64_t J, int64_t shrA, int64_t shrB, int64_t shrC,
            int64_t demote, int64_t bwA, int64_t bwB, int64_t bwTemp,
            int64_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C),
            bool verbose)
{
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);
    fix_bitwidth(B, I*J, bwB);
    fix_bitwidth(B_mask, I*J, bwB);

    int32_t shiftA = log(shrA);
    int32_t shiftB = log(shrB);
    int32_t shiftC = log(shrC);
    int32_t shift_demote = log(demote);

    GroupElement *memPool = make_ge_array(2*I*J, bwTemp);
    GroupElement *tmpA = memPool;
    GroupElement *tmpB = memPool + I*J;
    GroupElement *tmpA_mask;
    GroupElement *tmpB_mask;
    if (party == DEALER) {
        tmpA_mask = tmpA;
        tmpB_mask = tmpB;
    }

    internalTruncateAndFix(I*J, shiftA + shiftC, bwA, bwTemp, A, A_mask, tmpA, tmpA_mask, false);
    internalTruncateAndFix(I*J, shiftB + shiftC, bwB, bwTemp, B, B_mask, tmpB, tmpB_mask, false);

    if (party != DEALER) {
        auto start = std::chrono::high_resolution_clock::now();
        if (needReconstruct(shiftA + shiftC, bwA, bwTemp)) {
            if (needReconstruct(shiftB + shiftC, bwB, bwTemp)) {
                reconstruct(2*I*J, memPool, bwTemp);
            }
            else {
                reconstruct(I*J, tmpA, bwTemp);
            }
        }
        else if (needReconstruct(shiftB + shiftC, bwB, bwTemp)) {
            reconstruct(I*J, tmpB, bwTemp);
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.truncateFix += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::thread thread_pool[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_pool[i] = std::thread(MatAdd_threads_helper, i, I*J, tmpA, tmpA_mask, tmpB, tmpB_mask);
    }

    for (int i = 0; i < num_threads; i++) {
        thread_pool[i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    evaluatorStats.matadd += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    internalTruncateAndFix(I*J, shift_demote, bwTemp, bwC, tmpA, tmpA_mask, C, C_mask);
    delete[] memPool;

}

void MulCir_threads_helper(int thread_idx, int32_t size, int bwA, int bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *tmpC), MultKeyNew *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        tmpC[i].value = new_mult_signed_eval(party - SERVER, bwA, bwB, keys[i], A[i].value, B[i].value);
            mod(tmpC[i]);
        // TODO: needs clearing of dcf keys
    }
}

void MulCir_dealer_threads_helper(int thread_idx, int32_t size, int bwA, int bwB, GroupElement *A_mask, GroupElement *B_mask, GroupElement *tmpC_mask, pair<MultKeyNew> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; i += 1){
        tmpC_mask[i] = random_ge(bwA + bwB);
        keys[i] = new_mult_signed_gen(bwA, bwB, A_mask[i].value, B_mask[i].value, tmpC_mask[i].value);
    }
}

void MulCir(int64_t I, int64_t J, int64_t shrA, int64_t shrB, int64_t demote,
            int64_t bwA, int64_t bwB, int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C))
{
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);
    fix_bitwidth(B, I*J, bwB);
    fix_bitwidth(B_mask, I*J, bwB);

    int shiftA = log(shrA);
    int shiftB = log(shrB);
    int shiftDemote = log(demote);

    GroupElement *tmpC = make_ge_array(I*J, bwTemp);
    GroupElement *tmpC_mask;
    if (party == DEALER) {
        tmpC_mask = tmpC;
    }

    if (party == DEALER)
    {
#ifdef DEALER_DIRECT_SEND
        for(int i = 0; i < I*J; ++i)
        {
            tmpC_mask[i] = random_ge(bwA + bwB);
            auto k = new_mult_signed_gen(bwA, bwB, A_mask[i].value, B_mask[i].value, tmpC_mask[i].value);
            server->send_new_mult_key(k.first, bwA, bwB);
            client->send_new_mult_key(k.second, bwA, bwB);
        }
#else
        int size = I*J;
        pair<MultKeyNew> *keys = new pair<MultKeyNew>[size];

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i] = std::thread(MulCir_dealer_threads_helper, i, size, bwA, bwB, A_mask, B_mask, tmpC_mask, keys);
        }

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.mulcir += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for(int i = 0; i < size; ++i)
        {
            server->send_new_mult_key(keys[i].first, bwA, bwB);
            client->send_new_mult_key(keys[i].second, bwA, bwB);
            // TODO: needs clearing of dcf keys
        }

        delete[] keys;
#endif
    }
    else
    {
        MultKeyNew *keys = new MultKeyNew[I*J];
        for(int i = 0; i < I*J; ++i)
        {
            keys[i] = dealer->recv_new_mult_key(bwA, bwB);
        }

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i] = std::thread(MulCir_threads_helper, i, I*J, bwA, bwB, A, A_mask, B, B_mask, tmpC, tmpC_mask, keys);
        }

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i].join();
        }

        reconstruct(I*J, tmpC, bwTemp);
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.mulcir += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        delete[] keys;
    }

    internalTruncateAndFix(I*J, shiftA + shiftB + shiftDemote, bwTemp, bwC, tmpC, tmpC_mask, C, C_mask);
    delete[] tmpC;
}

void MatAddBroadCastA_threads_helper(int thread_idx, int size, uint64_t tmpA, GroupElement *tmpB)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB[i] = tmpA + tmpB[i];
    }
}

// A is public here
void MatAddBroadCastA(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement A), MASK_PAIR(GroupElement *B),
                      MASK_PAIR(GroupElement *C))
{
    fix_bitwidth(&A, 1, bwA);
    fix_bitwidth(B, I*J, bwB);
    fix_bitwidth(B_mask, I*J, bwB);

    int32_t shiftA = log(shrA);
    int32_t shiftB = log(shrB);
    int32_t shiftC = log(shrC);
    int32_t shift_demote = log(demote);

    GroupElement *tmpB = make_ge_array(I*J, bwTemp);
    GroupElement *tmpB_mask = party == DEALER ? tmpB : nullptr;

    uint64_t tmpA = truncate_reduce_clear(party == DEALER ? A_mask.value : A.value, bwA, bwTemp, shiftA + shiftC);
    internalTruncateAndFix(I*J, shiftB + shiftC, bwB, bwTemp, B, B_mask, tmpB, tmpB_mask);

    if (party != DEALER) {
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i] = std::thread(MatAddBroadCastA_threads_helper, i, I*J, tmpA, tmpB);
        }

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.matbroadcast += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    internalTruncateAndFix(I*J, shift_demote, bwTemp, bwC, tmpB, tmpB_mask, C, C_mask);
    delete[] tmpB;
}

void MatSubBroadCastA_threads_helper(int thread_idx, int size, uint64_t tmpA, GroupElement *tmpB)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB[i] = tmpA - tmpB[i];
    }
}

// A is public here
void MatSubBroadCastA(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement A), MASK_PAIR(GroupElement *B),
                      MASK_PAIR(GroupElement *C))
{
    fix_bitwidth(&A, 1, bwA);
    fix_bitwidth(B, I*J, bwB);
    fix_bitwidth(B_mask, I*J, bwB);

    int32_t shiftA = log(shrA);
    int32_t shiftB = log(shrB);
    int32_t shiftC = log(shrC);
    int32_t shift_demote = log(demote);

    GroupElement *tmpB = make_ge_array(I*J, bwTemp);
    GroupElement *tmpB_mask = party == DEALER ? tmpB : nullptr;

    uint64_t tmpA = truncate_reduce_clear(party == DEALER ? A_mask.value : A.value, bwA, bwTemp, shiftA + shiftC);
    internalTruncateAndFix(I*J, shiftB + shiftC, bwB, bwTemp, B, B_mask, tmpB, tmpB_mask);

    if (party == DEALER) {
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < I*J; ++i)
        {
            tmpB_mask[i] = -tmpB_mask[i];
            mod(tmpB_mask[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.matbroadcast += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    else {
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i] = std::thread(MatSubBroadCastA_threads_helper, i, I*J, tmpA, tmpB);
        }

        for (int i = 0; i < num_threads; i++) {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.matbroadcast += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    internalTruncateAndFix(I*J, shift_demote, bwTemp, bwC, tmpB, tmpB_mask, C, C_mask);
    delete[] tmpB;
}

void MatAddBroadCastB_threads_helper(int thread_idx, int size, GroupElement *tmpA, uint64_t tmpB)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpA[i] = tmpA[i] + tmpB;
    }
}

// B is public here
void MatAddBroadCastB(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement B),
                      MASK_PAIR(GroupElement *C))
{
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);
    fix_bitwidth(&B, 1, bwB);

    int32_t shiftA = log(shrA);
    int32_t shiftB = log(shrB);
    int32_t shiftC = log(shrC);
    int32_t shift_demote = log(demote);

    GroupElement *tmpA = make_ge_array(I*J, bwTemp);
    GroupElement *tmpA_mask = party == DEALER ? tmpA : nullptr;

    internalTruncateAndFix(I*J, shiftA + shiftC, bwA, bwTemp, A, A_mask, tmpA, tmpA_mask);
    uint64_t tmpB = truncate_reduce_clear(party == DEALER ? B_mask.value : B.value, bwB, bwTemp, shiftB + shiftC);

    if (party != DEALER){
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(MatAddBroadCastB_threads_helper, i, I*J, tmpA, tmpB);
        }

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.matbroadcast += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    internalTruncateAndFix(I*J, shift_demote, bwTemp, bwC, tmpA, tmpA_mask, C, C_mask);
    delete[] tmpA;
}

void MatSubBroadCastB_threads_helper(int thread_idx, int size, GroupElement *tmpA, uint64_t tmpB)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpA[i] = tmpA[i] - tmpB;
    }
}

// B is public here
void MatSubBroadCastB(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement B),
                      MASK_PAIR(GroupElement *C))
{
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);
    fix_bitwidth(&B, 1, bwB);

    int32_t shiftA = log(shrA);
    int32_t shiftB = log(shrB);
    int32_t shiftC = log(shrC);
    int32_t shift_demote = log(demote);

    GroupElement *tmpA = make_ge_array(I*J, bwTemp);
    GroupElement *tmpA_mask = party == DEALER ? tmpA : nullptr;

    internalTruncateAndFix(I*J, shiftA + shiftC, bwA, bwTemp, A, A_mask, tmpA, tmpA_mask);
    uint64_t tmpB = truncate_reduce_clear(party == DEALER ? B_mask.value : B.value, bwB, bwTemp, shiftB + shiftC);

    if (party != DEALER)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(MatSubBroadCastB_threads_helper, i, I*J, tmpA, tmpB);
        }

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.matbroadcast += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    internalTruncateAndFix(I*J, shift_demote, bwTemp, bwC, tmpA, tmpA_mask, C, C_mask);
    delete[] tmpA;
}

void AdjustScaleShl(int64_t I, int64_t J, int64_t scale, MASK_PAIR(GroupElement *A))
{
    if (party == DEALER) {
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < I*J; ++i) {
            A_mask[i].value = A_mask[i].value * scale;
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.shiftleft += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    else {
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < I*J; ++i) {
            A[i].value = A[i].value * scale;
        }
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.shiftleft += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
}


void ArgMax_internal(int32_t rows, int32_t cols, int bw, MASK_PAIR(GroupElement *inp), MASK_PAIR(GroupElement *out)) 
{
    // inp is a vector of size rows*columns and max (resp. maxidx) is caclulated for every
    // column chunk of elements. Result maxidx is stored in out (size: rows)

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
                    Arr2DIdxRowM(drelu_mask, rows, curCols / 2, row, j) = random_ge(bw);
                    auto scmpKeys = keyGenSCMP(bw, bw, Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j), Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j + 1), Arr2DIdxRowM(drelu_mask, rows, curCols / 2, row, j));
                    server->send_scmp_keypack(scmpKeys.first);
                    client->send_scmp_keypack(scmpKeys.second);
                }
            }

            for (int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    
                    Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, row, j) = random_ge(bw);
                    auto multKeys1 = MultGen(Arr2DIdxRowM(drelu_mask, rows, curCols / 2, row, j), Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j) - Arr2DIdxRowM(tmpMax_mask, rows, curCols, row, 2*j + 1), Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, row, j));
                    
                    server->send_mult_key(multKeys1.first);
                    client->send_mult_key(multKeys1.second);
                    
                    Arr2DIdxRowM(mult_res_mask, 2 * rows, curCols / 2, rows + row, j) = random_ge(bw);
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
        evaluatorStats.argmax += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
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
                    keys[k1++] = dealer->recv_scmp_keypack(bw, bw);
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

        GroupElement *tmpMax = make_array<GroupElement>(rows, cols);
        GroupElement *tmpIdx = make_array<GroupElement>(rows, cols);
        GroupElement *drelu = make_array<GroupElement>(rows, cols / 2);
        GroupElement *mult_res = make_array<GroupElement>(2 * rows, cols / 2);

        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        curCols = cols;
        k1 = 0; k2 = 0; k3 = 0;
        
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                Arr2DIdxRowM(tmpMax, rows, cols, i, j) = Arr2DIdxRowM(inp, rows, cols, i, j);
                Arr2DIdxRowM(tmpIdx, rows, cols, i, j).value = j;
            }
        }

        while(curCols > 1) {
            for(int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    Arr2DIdxRowM(drelu, rows, curCols / 2, row, j) = evalSCMP(party - 2, keys[k1++], Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j), Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j + 1));
                }
            }

            reconstruct(rows * (curCols / 2), drelu, bw);

            for (int row = 0; row < rows; row++) {
                for(int j = 0; j < curCols / 2; ++j) {
                    
                    Arr2DIdxRowM(mult_res, 2 * rows, curCols / 2, row, j) = MultEval(party - 2, mult_keys1[k2++], Arr2DIdxRowM(drelu, rows, curCols / 2, row, j), Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j) - Arr2DIdxRowM(tmpMax, rows, curCols, row, 2*j + 1));
                    
                    Arr2DIdxRowM(mult_res, 2 * rows, curCols / 2, rows + row, j) = MultEval(party - 2, mult_keys2[k3++], 
                        Arr2DIdxRowM(drelu, rows, curCols / 2, row, j), 
                        Arr2DIdxRowM(tmpIdx, rows, curCols, row, 2*j) - Arr2DIdxRowM(tmpIdx, rows, curCols, row, 2*j + 1));
                }
            }

            reconstruct((2*rows) * (curCols / 2), mult_res, bw);

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
        evaluatorStats.argmax += eval_time;
        delete[] tmpMax;
        delete[] tmpIdx;
        delete[] drelu;
        delete[] mult_res;

    }
}

void ArgMax(int64_t I, int64_t J, int32_t bwA, int32_t bw_index, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *index))
{
    assert(bwA == bw_index);
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);
    ArgMax_internal(1, I*J, bwA, A, A_mask, index, index_mask);
    fix_bitwidth(index, 1, bw_index);
}

void Sigmoid_threads_helper(int thread_idx, int size, GroupElement *A, GroupElement *tmpB, SplineKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB[i] = evalSigmoid_main_wrapper(party - SERVER, A[i], keys[i]);
        freeSplineKey(keys[i]);
    }
}

void Sigmoid_dealer_threads_helper(int thread_idx, int size, int ib, int ob, int shift_in, int shift_out, GroupElement *tmpA_mask, GroupElement *tmpB_mask, pair<SplineKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB_mask[i] = random_ge(ob);
        keys[i] = keyGenSigmoid_main_wrapper(ib, ob, shift_in, shift_out, tmpA_mask[i], tmpB_mask[i]);
    }
}

void Sigmoid(int64_t I, int64_t J, int64_t scale_in, int64_t scale_out,
             int64_t bwA, int64_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B))
{
    always_assert(bwA == bwB);
#ifdef SIGMOID_TANH_37
    always_assert(bwA == 37);
#else 
    always_assert(bwA == 16);
#endif
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);

    int32_t shift_in = log(scale_in);
    int32_t shift_out = log(scale_out);


#ifdef SIGMOID_TANH_37
    always_assert(shift_in == 12);
    always_assert(shift_out == 12);
    int ib = 64, ob = 64, sin = 12, scoef = 20, sout = 12, degree = 2, numPoly = 20;
#elif defined(SIGMOID_12_12)
    always_assert(shift_in == 12);
    always_assert(shift_out == 12);
    int ib = 64, ob = 64, sin = 12, scoef = 20, sout = 12, degree = 2, numPoly = 19;
#elif defined(SIGMOID_9_14)
    always_assert(shift_in == 9);
    always_assert(shift_out == 14);
    int ib = 64, ob = 64, sin = 9, scoef = 20, sout = 14, degree = 2, numPoly = 34;
#elif defined(SIGMOID_8_14)
    always_assert(shift_in == 8);
    always_assert(shift_out == 14);
    int ib = 64, ob = 64, sin = 8, scoef = 20, sout = 14, degree = 2, numPoly = 34;
#elif defined(SIGMOID_11_14)
    always_assert(shift_in == 11);
    always_assert(shift_out == 14);
    int ib = 64, ob = 64, sin = 11, scoef = 20, sout = 14, degree = 2, numPoly = 34;
#elif defined(SIGMOID_13_14)
    always_assert(shift_in == 13);
    always_assert(shift_out == 14);
    int ib = 64, ob = 64, sin = 13, scoef = 20, sout = 14, degree = 2, numPoly = 29;
#else 
    throw std::invalid_argument("no scales selected for sigmoid");
#endif

    GroupElement *tmpA = make_ge_array(I*J, ib);
    GroupElement *tmpA_mask = party == DEALER ? tmpA : nullptr;

    internalExtend(I*J, bwA, ib, A, A_mask, tmpA, tmpA_mask);

    GroupElement *tmpB = make_ge_array(I*J, ob);
    GroupElement *tmpB_mask = party == DEALER ? tmpB : nullptr;

    if (party == DEALER) {
#ifdef DEALER_DIRECT_SEND
        for(int i = 0; i < I*J; ++i) {
            tmpB_mask[i] = random_ge(ob);
            auto keys = keyGenSigmoid_main_wrapper(ib, ob, shift_in, shift_out, tmpA_mask[i], tmpB_mask[i]);
            server->send_spline_key(keys.first);
            client->send_spline_key(keys.second);
            freeSplineKeyPair(keys);
        }
#else
        int size = I*J;
        pair<SplineKeyPack> *keys = new pair<SplineKeyPack>[size];

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(Sigmoid_dealer_threads_helper, i, size, ib, ob, shift_in, shift_out, tmpA, tmpB, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.sigmoid += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for(int i = 0; i < size; ++i) {
            server->send_spline_key(keys[i].first);
            client->send_spline_key(keys[i].second);
            freeSplineKeyPair(keys[i]);
        }
        delete[] keys;
#endif
    }
    else {
        SplineKeyPack *keys = new SplineKeyPack[I*J];
        for(int i = 0; i < I*J; ++i) {
            keys[i] = dealer->recv_spline_key(ib, ob, numPoly, degree);
        }

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(Sigmoid_threads_helper, i, I*J, tmpA, tmpB, keys);
        }

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();

        reconstruct(I*J, tmpB, ob);
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.sigmoid += std::chrono::duration_cast<std::chrono::microseconds>(end - t2).count();
        evaluatorStats.sigmoid += std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();

        delete[] keys;
    }

    delete[] tmpA;
    internalTruncateAndFix(I*J, (degree * sin + scoef - sout), ob, bwB, tmpB, tmpB_mask, B, B_mask, true);
    delete[] tmpB;
}

void Tanh_dealer_threads_helper(int thread_idx, int size, int ib, int ob, int shift_in, int shift_out, GroupElement *tmpA_mask, GroupElement *tmpB_mask, pair<SplineKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB_mask[i] = random_ge(ob);
        keys[i] = keyGenTanh_main_wrapper(ib, ob, shift_in, shift_out, tmpA_mask[i], tmpB_mask[i]);
    }
}


void Tanh_threads_helper(int thread_idx, int size, GroupElement *A, GroupElement *tmpB, SplineKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB[i] = evalTanh_main_wrapper(party - SERVER, A[i], keys[i]);
        freeSplineKey(keys[i]);
    }
}

void TanH(int64_t I, int64_t J, int64_t scale_in, int64_t scale_out,
          int64_t bwA, int64_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B))
{
    always_assert(bwA == bwB);
#ifdef SIGMOID_TANH_37
    always_assert(bwA == 37);
#else 
    always_assert(bwA == 16);
#endif
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);

    int32_t shift_in = log(scale_in);
    int32_t shift_out = log(scale_out);
#if defined(TANH_12_12) || defined(SIGMOID_TANH_37)
// same spline for both cases for tanh
    always_assert(shift_in == 12);
    always_assert(shift_out == 12);
    int ib = 64, ob = 64, sin = 12, scoef = 18, sout = 12, degree = 2, numPoly = 26;
#elif defined(TANH_9_9)
    always_assert(shift_in == 9);
    always_assert(shift_out == 9);
    int ib = 64, ob = 64, sin = 9, scoef = 18, sout = 9, degree = 2, numPoly = 12;
#elif defined(TANH_8_8)
    always_assert(shift_in == 8);
    always_assert(shift_out == 8);
    int ib = 64, ob = 64, sin = 8, scoef = 18, sout = 8, degree = 2, numPoly = 10;
#elif defined(TANH_11_11)
    always_assert(shift_in == 11);
    always_assert(shift_out == 11);
    int ib = 64, ob = 64, sin = 11, scoef = 18, sout = 11, degree = 2, numPoly = 20;
#elif defined(TANH_13_13)
    always_assert(shift_in == 13);
    always_assert(shift_out == 13);
    int ib = 64, ob = 64, sin = 13, scoef = 18, sout = 13, degree = 2, numPoly = 12;
#else 
    throw std::invalid_argument("no scales selected for tanh");
#endif

    GroupElement *tmpA = make_ge_array(I*J, ib);
    GroupElement *tmpA_mask = party == DEALER ? tmpA : nullptr;

    internalExtend(I*J, bwA, ib, A, A_mask, tmpA, tmpA_mask);

    GroupElement *tmpB = make_ge_array(I*J, ob);
    GroupElement *tmpB_mask = party == DEALER ? tmpB : nullptr;

    if (party == DEALER) {
#ifdef DEALER_DIRECT_SEND
        for(int i = 0; i < I*J; ++i) {
            tmpB_mask[i] = random_ge(ob);
            auto keys = keyGenTanh_main_wrapper(ib, ob, shift_in, shift_out, tmpA_mask[i], tmpB_mask[i]);
            server->send_spline_key(keys.first);
            client->send_spline_key(keys.second);
            freeSplineKeyPair(keys);
        }
#else
        int size = I*J;
        pair<SplineKeyPack> *keys = new pair<SplineKeyPack>[size];

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(Tanh_dealer_threads_helper, i, size, ib, ob, shift_in, shift_out, tmpA, tmpB, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.tanh += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for(int i = 0; i < size; ++i) {
            server->send_spline_key(keys[i].first);
            client->send_spline_key(keys[i].second);
            freeSplineKeyPair(keys[i]);
        }
        delete[] keys;
#endif
    }
    else {
        SplineKeyPack *keys = new SplineKeyPack[I*J];
        for(int i = 0; i < I*J; ++i) {
            keys[i] = dealer->recv_spline_key(ib, ob, numPoly, degree);
        }

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(Tanh_threads_helper, i, I*J, tmpA, tmpB, keys);
        }

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();

        reconstruct(I*J, tmpB, ob);

        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.tanh += std::chrono::duration_cast<std::chrono::microseconds>(end - t2).count();
        evaluatorStats.tanh += std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();
        delete[] keys;
    }

    delete[] tmpA;
    internalTruncateAndFix(I*J, (degree * sin + scoef - sout), ob, bwB, tmpB, tmpB_mask, B, B_mask, true);
    delete[] tmpB;
}

void Invsqrt_threads_helper(int thread_idx, int size, GroupElement *A, GroupElement *tmpB, SplineKeyPack *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB[i] = evalInvsqrt_main_wrapper(party - SERVER, A[i], keys[i]);
        freeSplineKey(keys[i]);
    }
}

void Invsqrt_dealer_threads_helper(int thread_idx, int size, int ib, int ob, int shift_in, int shift_out, GroupElement *tmpA_mask, GroupElement *tmpB_mask, pair<SplineKeyPack> *keys)
{
    auto p = get_start_end(size, thread_idx);
    for(int i = p.first; i < p.second; ++i)
    {
        tmpB_mask[i] = random_ge(ob);
        keys[i] = keyGenInvsqrt_main_wrapper(ib, ob, shift_in, shift_out, tmpA_mask[i], tmpB_mask[i]);
    }
}

void Sqrt(int64_t I, int64_t J, int64_t scale_in, int64_t scale_out,
             int64_t bwA, int64_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B))
{
    always_assert(bwA == bwB);
    always_assert(bwA == 16);
    
    fix_bitwidth(A, I*J, bwA);
    fix_bitwidth(A_mask, I*J, bwA);

    int32_t shift_in = log(scale_in);
    int32_t shift_out = log(scale_out);

    

#ifdef INVSQRT_10_9
    always_assert(shift_in == 10);
    always_assert(shift_out == 9);
    int ib = 64, ob = 64, sin = 10, scoef = 13, sout = 9, degree = 2, numPoly = 10;
#elif defined(INVSQRT_12_11)
    always_assert(shift_in == 12);
    always_assert(shift_out == 11);
    int ib = 64, ob = 64, sin = 12, scoef = 13, sout = 11, degree = 2, numPoly = 10;
#else
    throw std::invalid_argument("no scales selected for invsqrt");
#endif

    GroupElement *tmpA = make_ge_array(I*J, ib);
    GroupElement *tmpA_mask = party == DEALER ? tmpA : nullptr;

    internalExtend(I*J, bwA, ib, A, A_mask, tmpA, tmpA_mask);

    GroupElement *tmpB = make_ge_array(I*J, ob);
    GroupElement *tmpB_mask = party == DEALER ? tmpB : nullptr;

    if (party == DEALER) {
#ifdef DEALER_DIRECT_SEND
        for(int i = 0; i < I*J; ++i) {
            tmpB_mask[i] = random_ge(ob);
            auto keys = keyGenInvsqrt_main_wrapper(ib, ob, shift_in, shift_out, tmpA_mask[i], tmpB_mask[i]);
            server->send_spline_key(keys.first);
            client->send_spline_key(keys.second);
            freeSplineKeyPair(keys);
        }
#else
        int size = I*J;
        pair<SplineKeyPack> *keys = new pair<SplineKeyPack>[size];

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i] = std::thread(Invsqrt_dealer_threads_helper, i, size, ib, ob, shift_in, shift_out, tmpA, tmpB, keys);
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_pool[i].join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.sqrt += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for(int i = 0; i < size; ++i) {
            server->send_spline_key(keys[i].first);
            client->send_spline_key(keys[i].second);
            freeSplineKeyPair(keys[i]);
        }
        delete[] keys;
#endif
    }
    else {
        SplineKeyPack *keys = new SplineKeyPack[I*J];
        for(int i = 0; i < I*J; ++i) {
            keys[i] = dealer->recv_spline_key(ib, ob, numPoly, degree);
        }

        peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        std::thread thread_pool[num_threads];

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i] = std::thread(Invsqrt_threads_helper, i, I*J, tmpA, tmpB, keys);
        }

        for(int i = 0; i < num_threads; ++i)
        {
            thread_pool[i].join();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        peer->sync();
        auto t2 = std::chrono::high_resolution_clock::now();

        reconstruct(I*J, tmpB, ob);
        auto end = std::chrono::high_resolution_clock::now();
        evaluatorStats.sqrt += std::chrono::duration_cast<std::chrono::microseconds>(end - t2).count();
        evaluatorStats.sqrt += std::chrono::duration_cast<std::chrono::microseconds>(t1 - start).count();

        delete[] keys;
    }

    delete[] tmpA;
    internalTruncateAndFix(I*J, (degree * sin + scoef - sout), ob, bwB, tmpB, tmpB_mask, B, B_mask, true);
    delete[] tmpB;
}

