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

#include <assert.h>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>

#include "utils/gpu_data_types.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_random.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"

#include "gpu_linear_helper.h"
#include "gpu_dpf_templates.h"
#include "gpu_dpf.h"

typedef void (*treeTraversal)(int party, int bin, int N,
                              u64 x,
                              AESBlock *scw, AESBlock *l0, AESBlock *l1,
                              u32 *tR, AESSharedContext *c, u32 *out, u64 oStride);

// only supports one bit output
// can avoid returning AESBlock to reduce copying
__device__ AESBlock expandDPFTreeNode(int bin, int party,
                                      const AESBlock s,
                                      const AESBlock cw,
                                      const AESBlock l0,
                                      const AESBlock l1,
                                      u32 tR,
                                      const u8 keep,
                                      int i,
                                      AESSharedContext *c)
{
    const AESBlock notOneAESBlock = ~1;
    const AESBlock zeroAndAllOne[2] = {0, static_cast<AESBlock>(~0)};
    const AESBlock OneAESBlock = 1;

    AESBlock tau = 0, stcw;
    u8 t_previous = lsb(s);
    /* remove the last two bits from the AES seed */
    auto ss = s & notOneAESBlock;

    /* get the seed for this level (tau) */
    // apply aes to either 0 or 2 based on keep (is what is hopefully happening)
    applyAESPRG(c, (u32 *)&ss, keep * 2, (u32 *)&tau);
    AESBlock scw = 0;
    if (i < bin - LOG_AES_BLOCK_LEN - 1)
    {
        /* zero out the last two bits of the correction word for s because
    they must contain the corrections for t0 and t1 */
        scw = (cw & notOneAESBlock);
        /* separate the correction bits for t0 and t1 and place them
    in the lsbs of two AES blocks */
        // u32 ds1 = tR_l;
        // if (evalAll)
        // tR_l = ((tR_l >> i) & 1);
        // else
        // ds1 = tR; /*getVCW(1, tR, N, i);*/
        AESBlock ds[2] = {cw & OneAESBlock, AESBlock(tR)};
        scw ^= ds[keep];
    }
    else
    {
        AESBlock ds[2] = {l0, l1};
        scw = ds[keep];
    }

    const auto mask = zeroAndAllOne[t_previous];

    /* correct the seed for the next level if necessary */
    // tau is completely pseudorandom and is being xored with (scw || 0 || tcw)*keep
    stcw = tau ^ (scw & mask);
    return stcw;
}

__device__ u8 getDPFOutput(AESBlock *s, u64 x)
{
    gpuMod(x, LOG_AES_BLOCK_LEN);
    return u8(*s >> x) & 1;
}

__device__ void doDpf(int party, int bin, int N,
                      u64 x,
                      AESBlock *scw, AESBlock *l0, AESBlock *l1,
                      u32 *tR, AESSharedContext *c, u32 *out, u64 oStride)
{
    AESBlock s = scw[0];
    auto x1 = __brevll(x) >> (64 - bin);
    for (int i = 0; i < bin - LOG_AES_BLOCK_LEN; ++i)
    {
        const u8 keep = lsb(x1);
        if (i < bin - LOG_AES_BLOCK_LEN - 1)
        {
            u32 tR_l = u32(getVCW(1, tR, N, i));
            s = expandDPFTreeNode(bin, party, s, scw[(i + 1) * N], 0, 0, tR_l, keep, i, c);
        }
        else
        {

            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            s = expandDPFTreeNode(bin, party, s, 0, l0[tid], l1[tid], 0, keep, i, c);
        }
        x1 >>= 1;
    }
    auto o = getDPFOutput(&s, x);
    writePackedOp(out, u64(o), 1, N);
}

template <int E, dpfPrologue pr, dpfEpilogue ep>
__device__ void doDcf(int party, int bin, int N,
                      u64 x,
                      AESBlock *scw, AESBlock *l0, AESBlock *l1,
                      u32 *tR,
                      AESSharedContext *c, u32 *out, u64 oStride)
{
    AESBlock s[E];
    u64 x0[E], x1[E];
    // populate the input
    pr(party, bin, N, x, x0);
    u8 p[E], oldDir[E], keep[E];
    for (int e = 0; e < E; e++)
    {
        s[e] = scw[0];
        x1[e] = __brevll(x0[e]) >> (64 - bin);
        p[e] = 0;
        oldDir[e] = 0;
        keep[e] = 0;
    }
    for (int i = 0; i < bin - LOG_AES_BLOCK_LEN; ++i)
    {
        AESBlock curScw = 0, l0_l = 0, l1_l = 0;
        u32 tR_l;
        if (i < bin - LOG_AES_BLOCK_LEN - 1)
        {
            curScw = scw[(i + 1) * N];
            tR_l = u32(getVCW(1, tR, N, i));
        }
        else
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            l0_l = l0[tid];
            l1_l = l1[tid];
        }

        for (int e = 0; e < E; e++)
        {

            keep[e] = lsb(x1[e]);
            // the direction changed
            if (oldDir[e] != keep[e])
                p[e] ^= lsb(s[e]);
            // need to keep track of all the current seeds separately
            if (i < bin - LOG_AES_BLOCK_LEN - 1)
            {
                s[e] = expandDPFTreeNode(bin, party, s[e], curScw, 0, 0, tR_l, keep[e], i, c);
            }
            else
            {
                s[e] = expandDPFTreeNode(bin, party, s[e], 0, l0_l, l1_l, 0, keep[e], i, c);
                int ub;
                int pos = x0[e] & 127;
                if (keep[e] == 1)
                {
                    // xor with the complement of the prefix substring
                    // get rid of the lower order bits
                    // Neha: need to change this later
                    // can need to xor anywhere from 127 bits to 0 bits
                    ub = 127 - pos;
                    s[e] >>= (pos + 1);
                }
                else
                {
                    ub = pos + 1; // x0[e] & 127;
                    // don't get rid of the lower order bits
                }
                for (int i = 0; i < ub; i++)
                {
                    // extract the lsb of s
                    p[e] ^= lsb(s[e]) /*((u32)(*s) & 1)*/;
                    s[e] >>= 1;
                }
            }
            oldDir[e] = keep[e];
            x1[e] >>= 1;
        }
    }
    // add loop here as well
    if (party == SERVER1)
    {
        for (int e = 0; e < E; e++)
        {
            p[e] ^= u8(1);
        }
    }
    ep(party, bin, N, x, p, out, oStride);
}

// think about when to pass pointers to large amounts of data like AESBlocks
/* out needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
template <typename T, treeTraversal t>
__global__ void dpfTreeEval(int party, int bin, int N, T *in, AESBlock *scw,
                            AESBlock *l0, AESBlock *l1, u32 *tR, u32 *out, u64 oStride, AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        scw = &scw[tid];
        auto x = u64(in[tid]);
        t(party, bin, N, x, scw, l0, l1, tR, &saes, out, oStride);
    }
}

template <typename T, treeTraversal t>
void gpuDpfTreeEval(GPUDPFTreeKey k, int party, T *d_in, AESGlobalContext *g, Stats *s, u32 *d_out, u64 oStride)
{
    // auto d_out = moveMasks(k.memSzOut, h_masks, s);
    assert(k.memSzScw % (k.bin - LOG_AES_BLOCK_LEN) == 0);

    AESBlock *d_scw = (AESBlock *)moveToGPU((u8 *)k.scw, k.memSzScw, s);
    AESBlock *d_l0 = (AESBlock *)moveToGPU((u8 *)k.l0, k.memSzL, s);
    AESBlock *d_l1 = (AESBlock *)moveToGPU((u8 *)k.l1, k.memSzL, s);
    u32 *d_tR = (u32 *)moveToGPU((u8 *)k.tR, k.memSzT, s);

    const int tbSz = 256;
    int tb = (k.N - 1) / tbSz + 1;
    // auto start = std::chrono::high_resolution_clock::now();
    // kernel launch
    dpfTreeEval<T, t><<<tb, tbSz>>>(party, k.bin, k.N, d_in, d_scw, d_l0, d_l1, d_tR, d_out, oStride, *g);
    checkCudaErrors(cudaDeviceSynchronize());
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // printf("Time taken by dpf kernel=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

    gpuFree(d_scw);
    gpuFree(d_l0);
    gpuFree(d_l1);
    gpuFree(d_tR);
}

// no memory leak
template <typename T>
u32 *gpuDpf(GPUDPFKey k, int party, T *d_in, AESGlobalContext *g, Stats *s)
{
    u32 *d_out;
    if (k.bin <= 7)
        d_out = gpuLookupSSTable<T, 1, idPrologue, idEpilogue>(k.ssKey, party, d_in, s);
    else
    {
        d_out = moveMasks(k.memSzOut, NULL, s);
        int n = k.dpfTreeKey[0].N;
        size_t gIntSzOut = k.memSzOut / sizeof(PACK_TYPE);
        size_t bIntSzOut = k.dpfTreeKey[0].memSzOut / sizeof(PACK_TYPE);
        for (int b = 0; b < k.B; b++)
        {
            gpuDpfTreeEval<T, doDpf>(k.dpfTreeKey[b], party, d_in + b * n, g, s, d_out + b * bIntSzOut, (u64)gIntSzOut);
        }
    }
    return d_out;
}

template <typename T, int E, dpfPrologue pr, dpfEpilogue ep>
u32 *gpuDcf(GPUDPFKey k, int party, T *d_in, AESGlobalContext *g, Stats *s, std::vector<u32 *> *h_masks = NULL)
{
    // printf("Started gpu dcf\n");
    u32 *d_out;
    if (k.bin <= 7)
        d_out = gpuLookupSSTable<T, E, pr, ep>(k.ssKey, party, d_in, s, h_masks);
    else
    {
        d_out = moveMasks(k.memSzOut, h_masks, s);
        size_t gIntSzOut = k.memSzOut / sizeof(PACK_TYPE);
        int n = k.dpfTreeKey[0].N;
        size_t bIntSzOut = k.dpfTreeKey[0].memSzOut / sizeof(PACK_TYPE);
        // printf("outSz=%lu\n", bIntSzOut);
        for (int b = 0; b < k.B; b++)
        {
            gpuDpfTreeEval<T, doDcf<E, pr, ep>>(k.dpfTreeKey[b], party, d_in + b * n, g, s, d_out + b * bIntSzOut, (u64)gIntSzOut);
        }
    }
    return d_out;
}

// Real Endpoints
template <typename T>
__global__ void keyGenDPFTreeKernel(int party, int bin, int N, T *rinArr, AESBlock *s0, AESBlock *s1, AESBlock *k0, AESBlock *l0, AESBlock *l1, u32 *tR, AESGlobalContext gaes, bool evalAll)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        static const AESBlock notOneBlock = ~1;
        static const AESBlock OneBlock = 1;
        const AESBlock zeroAndAllOne[2] = {0, static_cast<AESBlock>(~0)};

        AESBlock s[2];
        s[0] = s0[tid];
        s[1] = s1[tid];
        AESBlock si[2][2];
        // s || secret share of 1
        s[0] = (s[0] & notOneBlock) ^ ((s[1] & OneBlock) ^ OneBlock);
        k0[tid] = s[party == 1];
        u32 tR_l = 0;
        auto rin = u64(rinArr[tid]);
        gpuMod(rin, bin);
        int i;
        for (i = 0; i < bin - LOG_AES_BLOCK_LEN; ++i)
        {
            const u8 keep = static_cast<u8>(rin >> (bin - 1 - i)) & 1;
            AESBlock a = AESBlock(keep);
            // 127-bit AES seed
            auto ss0 = s[0] & notOneBlock;
            auto ss1 = s[1] & notOneBlock;

            // apply aes to 0 and 2
            // Neha: make sure we don't need &(si[0][1])
            applyAESPRGTwoTimes(&saes, (u32 *)&ss0, 0, (u32 *)si[0], (u32 *)&si[0][1]);
            applyAESPRGTwoTimes(&saes, (u32 *)&ss1, 0, (u32 *)si[1], (u32 *)&si[1][1]);
            // get the advice bits for this level

            // correction words for the AES seed
            AESBlock siXOR[] = {si[0][0] ^ si[1][0], si[0][1] ^ si[1][1]};
            if (i < bin - LOG_AES_BLOCK_LEN - 1)
            {
                auto ti0 = (u8)lsb(s[0]);
                auto ti1 = (u8)lsb(s[1]);
                // correction words for the advice bits
                AESBlock t[] = {
                    (OneBlock & siXOR[0]) ^ a ^ OneBlock,
                    (OneBlock & siXOR[1]) ^ a};

                auto scw = siXOR[keep ^ 1] & notOneBlock;
                k0[(i + 1) * N + tid] = scw       // set bits [127, 2] as scw = s0_loss ^ s1_loss
                                        ^ (t[0]); // set bit 0 as tL
                                                  //  ^ t[1];       // set bit 0 as tR

                if (evalAll)
                    tR_l = (tR_l << 1) ^ u32(t[1]);
                else
                    writeVCW(1, tR, u64(t[1]), i, N);
                // printf("tR %d=%u, %u\n", tid, u32(t[1]), tR_l);
                // next seeds along the special path
                auto si0Keep = si[0][keep];
                auto si1Keep = si[1][keep];

                // the advice correction bit along the special path
                auto TKeep = t[keep];
                // set the next level of s,t
                s[0] = si0Keep ^ (zeroAndAllOne[ti0] & (scw ^ TKeep));
                s[1] = si1Keep ^ (zeroAndAllOne[ti1] & (scw ^ TKeep));
            }
            else
            {
                auto rinPosInBlock = rin;
                gpuMod(rinPosInBlock, LOG_AES_BLOCK_LEN);
                // remember that in the packing code here we are assuming that 0 is the lsb (and
                // not the msb)
                auto rinBlock = OneBlock << rinPosInBlock;
                l0[tid] = siXOR[0] ^ (zeroAndAllOne[keep == 0] & rinBlock);
                l1[tid] = siXOR[1] ^ (zeroAndAllOne[keep == 1] & rinBlock);
            }
        }
        if (evalAll)
        {
            tR[tid] = __brev(tR_l) >> (32 - bin + LOG_AES_BLOCK_LEN + 1);
        }
    }
}

template <typename T>
void doDpfTreeKeyGen(u8 **key_as_bytes, int party, int bin, int N,
                     T *d_rin, AESGlobalContext *gaes, bool evalAll)
{
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, N);
    writeInt(key_as_bytes, evalAll);
    assert(bin > LOG_AES_BLOCK_LEN);

    u64 memSizeK = N * (bin - LOG_AES_BLOCK_LEN) * sizeof(AESBlock);
    AESBlock *d_k0 = (AESBlock *)gpuMalloc(memSizeK);
    u64 memSizeL = N * sizeof(AESBlock);
    AESBlock *d_l0 = (AESBlock *)gpuMalloc(memSizeL);
    AESBlock *d_l1 = (AESBlock *)gpuMalloc(memSizeL);
    u64 memSizeT;
    if (evalAll)
        memSizeT = N * sizeof(u32);
    else
        memSizeT = ((N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (bin - LOG_AES_BLOCK_LEN);
    u32 *d_tR = (u32 *)gpuMalloc(memSizeT);

    auto d_s0 = randomAESBlockOnGpu(N);
    auto d_s1 = randomAESBlockOnGpu(N);
    keyGenDPFTreeKernel<<<(N - 1) / 256 + 1, 256>>>(party, bin, N, d_rin, d_s0, d_s1, d_k0, d_l0, d_l1, d_tR, *gaes, evalAll);
    checkCudaErrors(cudaDeviceSynchronize());
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_k0, memSizeK, NULL);

    *key_as_bytes += memSizeK;
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_l0, memSizeL, NULL);
    *key_as_bytes += memSizeL;
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_l1, memSizeL, NULL);
    *key_as_bytes += memSizeL;
    moveIntoCPUMem(*key_as_bytes, (u8 *)d_tR, memSizeT, NULL);
    *key_as_bytes += memSizeT;

    gpuFree(d_s0);
    gpuFree(d_s1);
    gpuFree(d_k0);
    gpuFree(d_l0);
    gpuFree(d_l1);
    gpuFree(d_tR);
}

template <typename T>
void gpuKeyGenBatchedDPF(u8 **key_as_bytes, int party, int bin, int N,
                         T *d_rin, AESGlobalContext *gaes, bool evalAll)
{
    u64 memSzOneK = (bin - LOG_AES_BLOCK_LEN + 2) * sizeof(AESBlock);
    int m = (24 * OneGB) / memSzOneK;
    m -= (m % 32);
    int B = (N - 1) / m + 1;
    // printf("N=%d, m=%d, B=%d, evalAll=%d\n", N, m, B, evalAll);
    writeInt(key_as_bytes, bin);
    writeInt(key_as_bytes, N);
    writeInt(key_as_bytes, B);
    for (int b = 0; b < B; b++)
        doDpfTreeKeyGen(key_as_bytes, party, bin, std::min(m, N - b * m), d_rin + b * m, gaes, evalAll);
}
// only a payload of 1 is supported so far
// should i write the key here?
template <typename T>
void gpuKeyGenDPF(u8 **key_as_bytes, int party, int bin, int N,
                  T *d_rin, AESGlobalContext *gaes, bool evalAll = false)
{
    if (bin <= 7)
    {
        genSSTable<T, dpfShares>(key_as_bytes, party, bin, N, d_rin);
    }
    else
    {
        gpuKeyGenBatchedDPF(key_as_bytes, party, bin, N, d_rin, gaes, evalAll);
    }
}

template <typename T>
void gpuKeyGenDCF(u8 **key_as_bytes, int party, int bin, int N,
                  T *d_rin, AESGlobalContext *gaes)
{
    // printf("Bin inside keygenDCF=%d\n", bin);
    if (bin <= 7)
    {
        genSSTable<T, dcfShares>(key_as_bytes, party, bin, N, d_rin);
    }
    else
    {
        gpuKeyGenBatchedDPF(key_as_bytes, party, bin, N, d_rin, gaes, false);
    }
}
