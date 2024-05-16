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

#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"

#include "fss/gpu_aes_shm.h"

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>

#include "gpu_dcf_templates.h"
#include "gpu_sstab.h"

namespace dcf
{

    __device__ u64 getGroupElementFromAESBlock(AESBlock b, int bout, int vector_elem_pos)
    {
        /* returning the last 64 bits */
        assert((vector_elem_pos + 1) * bout <= AES_BLOCK_LEN_IN_BITS);
        u64 g = static_cast<u64>(b >> (vector_elem_pos * bout));
        gpuMod(g, bout);
        return g;
    }

    __device__ AESBlock traverseOneDCF(int bin, int bout, int party,
                                       const AESBlock s,
                                       const AESBlock cw,
                                       const u8 keep,
                                       u64 *v_share,
                                       u64 vcw,
                                       uint64_t level,
                                       AESSharedContext *c)

    {
        // /* these need to be written to constant memory */
        const AESBlock notThreeAESBlock = ~3;
        const AESBlock zeroAndAllOne[2] = {0, static_cast<AESBlock>(~0)};
        const AESBlock OneAESBlock = 1;

        AESBlock tau = 0, cur_v = 0, stcw;
        u8 t_previous = lsb(s);
        auto ss = s & notThreeAESBlock;
        applyAESPRGTwoTimes(c, (u32 *)&ss, keep, (u32 *)&tau, (u32 *)&cur_v);
        const auto scw = (cw & notThreeAESBlock);
        AESBlock ds[] = {((cw >> 1) & OneAESBlock), (cw & OneAESBlock)};
        const auto mask = zeroAndAllOne[t_previous];
        stcw = tau ^ ((scw ^ ds[keep]) & mask);

        uint64_t sign = (party == SERVER1) ? -1 : 1;
        u64 v = getGroupElementFromAESBlock(cur_v, bout, 0);
        *v_share += (sign * (v + (static_cast<u64>(mask) & vcw)));
        return stcw;
    }

    // fix the ballot sync bug
    template <typename T, int E, dcfPrologue pr, dcfEpilogue ep>
    __global__ void doDcf(int bin, int bout, int party, int N,
                          T *in,         // might want to pass a pointer to this later
                          AESBlock *scw, // k.bin + 1
                          u32 *vcw,      // k.bin * groupSize
                          AESBlock *l_g,
                          u32 *out, u64 oStride, AESGlobalContext gaes)
    {
        AESSharedContext saes;
        loadSbox(&gaes, &saes);
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < N)
        {
            scw = &scw[tid];
            auto x = u64(in[tid]);
            AESBlock s[E];
            u64 x0[E], x1[E], v_alpha[E];
            pr(party, bin, N, x, x0);
            // printf("dcf=%lu, %lu\n", x0[0], x0[1]);
            for (int e = 0; e < E; e++)
            {
                s[e] = scw[0];
                gpuMod(x0[e], bin);
                x1[e] = __brevll(x0[e]) >> (64 - bin);
                v_alpha[e] = 0;
            }
            int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / bout;
            int levelsPacked = int(ceil(log2((double)elemsPerBlock)));
            for (int i = 0; i < bin - levelsPacked - 1; ++i)
            {
                auto curVcw = getVCW(bout, vcw, N, i);
                auto curScw = scw[(i + 1) * N];
                for (int e = 0; e < E; e++)
                {
                    const u8 keep = lsb(x1[e]);
                    s[e] = traverseOneDCF(bin, bout, party, s[e],
                                          curScw, keep, &v_alpha[e], curVcw, i, &saes);
                    x1[e] >>= 1;
                }
            }
            AESBlock l[2];
            l[0] = l_g[2 * tid];
            l[1] = l_g[2 * tid + 1];
            AESBlock ct;
            for (int e = 0; e < E; e++)
            {
                int j = x1[e] & 1;
                u64 offset = x0[e];
                gpuMod(offset, levelsPacked);
                u64 t = lsb(s[e]);
                auto ss = s[e] & ~3;
                applyAESPRG(&saes, (u32 *)&ss, 2 * j + 1, (u32 *)&ct);
                u64 v = getGroupElementFromAESBlock(ct, bout, offset);
                u64 curVcw = getGroupElementFromAESBlock(l[j], bout, offset);
                u64 sign = party == SERVER1 ? -1 : 1;
                v_alpha[e] += (sign * (v + (t * curVcw)));
                gpuMod(v_alpha[e], bout);
            }
            ep(party, bin, bout, N, x, v_alpha, out, oStride);
        }
    }

    // no memory leak
    template <typename T, int E, dcfPrologue pr, dcfEpilogue ep>
    void gpuDcfTreeEval(GPUDCFTreeKey k, int party, T *d_in, u32 *d_out, u64 oStride, AESGlobalContext *g, Stats *s)
    {
        // do not change tb size it is needed to load the sbox
        const int tb_size = 256;
        int num_thread_blocks = (k.N - 1) / tb_size + 1;
        AESBlock *d_scw, *d_l;
        u32 *d_vcw;

        d_scw = (AESBlock *)moveToGPU((u8 *)k.scw, k.memSzScw, s);
        d_vcw = (u32 *)moveToGPU((u8 *)k.vcw, k.memSzVcw, s);
        d_l = (AESBlock *)moveToGPU((u8 *)k.l, k.memSzL, s);

        doDcf<T, E, pr, ep><<<num_thread_blocks, tb_size>>>(k.bin, k.bout, party, k.N, d_in, d_scw, d_vcw, d_l, d_out, oStride, *g);

        checkCudaErrors(cudaDeviceSynchronize());
        gpuFree(d_scw);
        gpuFree(d_l);
        gpuFree(d_vcw);
    }

    template <typename T, int E, dcfPrologue pr, dcfEpilogue ep>
    u32 *gpuDcf(GPUDCFKey k, int party, T *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks = NULL)
    {
        u32 *d_out;
        if (k.bin <= 8)
        {
            d_out = dcf::gpuLookupSSTable<T, E, pr, ep>(k.ssKey, party, d_in, s, h_masks);
        }
        else
        {
            d_out = moveMasks(k.memSzOut, h_masks, s);
            size_t gIntSzOut = k.memSzOut / sizeof(PACK_TYPE);
            int n = k.dcfTreeKey[0].N;
            size_t bIntSzOut = k.dcfTreeKey[0].memSzOut / sizeof(PACK_TYPE);
            for (int b = 0; b < k.B; b++)
            {
                gpuDcfTreeEval<T, E, pr, ep>(k.dcfTreeKey[b], party, d_in + b * n, d_out + b * bIntSzOut, (u64)gIntSzOut, g, s);
            }
        }
        return d_out;
    }

    // Real Endpoints
    template <typename T>
    __global__ void keyGenDCFKernel(int party, int bin, int bout, int N, T *rinArr,
                                    u64 payload, AESBlock *s0, AESBlock *s1, AESBlock *k0, u32 *v0, AESBlock *leaves, AESGlobalContext gaes, bool leq = false)
    {
        AESSharedContext saes;
        loadSbox(&gaes, &saes);
        int threadId = blockIdx.x * blockDim.x + threadIdx.x;
        if (threadId < N)
        {
            static const AESBlock notOneBlock = ~1;
            static const AESBlock notThreeBlock = ~3;
            static const AESBlock OneBlock = 1;

            AESBlock s[2];
            s[0] = s0[threadId];
            s[1] = s1[threadId];
            AESBlock si[2][2];
            AESBlock vi[2][2];

            u64 v_alpha = 0;

            s[0] = (s[0] & notOneBlock) ^ ((s[1] & OneBlock) ^ OneBlock);
            k0[threadId] = s[party == 1];
            AESBlock ct[4];

            auto rin = u64(rinArr[threadId]);
            int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / bout;
            int levelsPacked = int(ceil(log2((double)elemsPerBlock)));

            for (int i = 0; i < bin - levelsPacked - 1; ++i)
            {
                const u8 keep = static_cast<u8>(rin >> (bin - 1 - i)) & 1;
                AESBlock a = AESBlock(keep);

                auto ss0 = s[0] & notThreeBlock;
                auto ss1 = s[1] & notThreeBlock;

                applyAESPRGFourTimes(&saes, (u32 *)&ss0, (u32 *)ct, (u32 *)&ct[1], (u32 *)&ct[2], (u32 *)&ct[3]);
                si[0][0] = ct[0];
                si[0][1] = ct[1];
                vi[0][0] = ct[2];
                vi[0][1] = ct[3];
                applyAESPRGFourTimes(&saes, (u32 *)&ss1, (u32 *)ct, (u32 *)&ct[1], (u32 *)&ct[2], (u32 *)&ct[3]);
                si[1][0] = ct[0];
                si[1][1] = ct[1];
                vi[1][0] = ct[2];
                vi[1][1] = ct[3];

                auto ti0 = (u8)lsb(s[0]);
                auto ti1 = (u8)lsb(s[1]);
                u64 sign = (ti1 == 1) ? -1 : +1;

                auto vi_00_converted = getGroupElementFromAESBlock(vi[0][keep], bout, 0);
                auto vi_10_converted = getGroupElementFromAESBlock(vi[1][keep], bout, 0);
                auto vi_01_converted = getGroupElementFromAESBlock(vi[0][keep ^ 1], bout, 0);
                auto vi_11_converted = getGroupElementFromAESBlock(vi[1][keep ^ 1], bout, 0);

                auto v = sign * (-v_alpha - vi_01_converted + vi_11_converted);
                if (keep == 1)
                {
                    v = v + sign * payload;
                }
                gpuMod(v, bout);
                writeVCW(bout, v0, v, i, N);
                v_alpha = v_alpha - vi_10_converted + vi_00_converted + sign * v;

                AESBlock siXOR[] = {si[0][0] ^ si[1][0], si[0][1] ^ si[1][1]};

                // get the left and right t_CW bits
                AESBlock t[] = {
                    (OneBlock & siXOR[0]) ^ a ^ OneBlock,
                    (OneBlock & siXOR[1]) ^ a};

                // take scw to be the bits [127, 2] as scw = s0_loss ^ s1_loss
                auto scw = siXOR[keep ^ 1] & notThreeBlock; // not15Block;
                k0[(i + 1) * N + threadId] = scw            // set bits [127, 2] as scw = s0_loss ^ s1_loss
                                             ^ (t[0] << 1)  // set bit 1 as tL
                                             ^ t[1];        // set bit 0 as tR

                auto si0Keep = si[0][keep];
                auto si1Keep = si[1][keep];

                // extract the t^Keep_CW bit
                auto TKeep = t[keep];
                const AESBlock zeroAndAllOne[2] = {0, static_cast<AESBlock>(~0)};
                // set the next level of s,t
                s[0] = si0Keep ^ (zeroAndAllOne[ti0] & (scw ^ TKeep));
                s[1] = si1Keep ^ (zeroAndAllOne[ti1] & (scw ^ TKeep));
            }

            auto ti0 = (u8)lsb(s[0]);
            auto ti1 = (u8)lsb(s[1]);
            auto ss0 = s[0] & notThreeBlock;
            auto ss1 = s[1] & notThreeBlock;
            AESBlock vcw[2];
            vcw[0] = 0;
            vcw[1] = 0;
            applyAESPRGTwoTimes(&saes, (u32 *)&ss0, 1, (u32 *)&ct[0], (u32 *)&ct[1]);
            applyAESPRGTwoTimes(&saes, (u32 *)&ss1, 1, (u32 *)&ct[2], (u32 *)&ct[3]);

            u64 zeroMask = ~((1ULL << (levelsPacked + 1)) - 1);
            auto rinPrefix = rin & zeroMask;
            for (int i = 0; i < 2 * elemsPerBlock; i++)
            {

                u64 s0_converted = getGroupElementFromAESBlock(ct[i / elemsPerBlock], bout, i % elemsPerBlock);
                u64 s1_converted = getGroupElementFromAESBlock(ct[2 + i / elemsPerBlock], bout, i % elemsPerBlock);
                auto x = rinPrefix + i;
                auto g0 = s1_converted - s0_converted - v_alpha + (x < rin) * payload + leq * (x == rin) * payload;
                assert(((x < rin) * payload + leq * (x == rin) * payload) <= payload);
                if (ti1 == 1)
                {
                    g0 = g0 * -1;
                }
                gpuMod(g0, bout);
                vcw[i / elemsPerBlock] |= (AESBlock(g0) << ((i % elemsPerBlock) * bout));
            }
            leaves[2 * threadId] = vcw[0];
            leaves[2 * threadId + 1] = vcw[1];
        }
    }

    template <typename T>
    void doDcfTreeKeyGen(u8 **key_as_bytes, int party, int bin, int bout, int N,
                         T *d_rin, T payload, AESGlobalContext *gaes, bool leq = false)
    {
        writeInt(key_as_bytes, bin);
        writeInt(key_as_bytes, bout);
        writeInt(key_as_bytes, N);

        // can think about other bitlengths later
        // assert(bout == 1 || bout == 2);
        int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / bout;
        int newBin = bin - int(log2(elemsPerBlock));
        size_t memSzK = N * newBin * sizeof(AESBlock);
        size_t memSzL = 2 * N * sizeof(AESBlock);

        AESBlock *d_k0 = (AESBlock *)gpuMalloc(memSzK);
        AESBlock *d_leaves = (AESBlock *)gpuMalloc(memSzL);

        size_t memSzV = ((bout * N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (newBin - 1);
        auto d_v0 = (u32 *)gpuMalloc(memSzV);

        auto d_s0 = randomAESBlockOnGpu(N);
        auto d_s1 = randomAESBlockOnGpu(N);

        keyGenDCFKernel<<<(N - 1) / 256 + 1, 256>>>(party, bin, bout, N, d_rin,
                                                    u64(payload), d_s0, d_s1, d_k0, d_v0, d_leaves, *gaes, leq);
        checkCudaErrors(cudaDeviceSynchronize());

        moveIntoCPUMem(*key_as_bytes, (u8 *)d_k0, memSzK, NULL);
        *key_as_bytes += memSzK;
        moveIntoCPUMem(*key_as_bytes, (u8 *)d_leaves, memSzL, NULL);
        *key_as_bytes += memSzL;
        moveIntoCPUMem(*key_as_bytes, (u8 *)d_v0, memSzV, NULL);
        *key_as_bytes += memSzV;

        gpuFree(d_s0);
        gpuFree(d_s1);
        gpuFree(d_k0);
        gpuFree(d_v0);
        gpuFree(d_leaves);
    }

    template <typename T>
    void gpuKeyGenDCF(uint8_t **key_as_bytes, int party, int bin, int bout, int N,
                      T *d_rin, T payload, AESGlobalContext *gaes, bool leq = false)
    {
        if (bin <= 8)
        {
            assert(bout == 1 && payload == T(1) && leq);
            genSSTable<T, dcfShares>(key_as_bytes, party, bin, N, d_rin);
        }
        else
        {
            int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / bout;
            int newBin = bin - int(log2(elemsPerBlock));
            u64 memSzOneK = (newBin + 2) * sizeof(AESBlock);
            int m = (24 * OneGB) / memSzOneK;
            m -= (m % 32);
            int B = (N - 1) / m + 1;
            // printf("N=%d, m=%d, B=%d\n", N, m, B);
            writeInt(key_as_bytes, bin);
            writeInt(key_as_bytes, bout);
            writeInt(key_as_bytes, N);
            writeInt(key_as_bytes, B);
            for (int b = 0; b < B; b++)
                doDcfTreeKeyGen(key_as_bytes, party, bin, bout, std::min(m, N - b * m), d_rin + b * m, payload, gaes, leq);
        }
    }
}