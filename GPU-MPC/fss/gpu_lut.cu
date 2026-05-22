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

#include "gpu_lut.h"
#include "gpu_dpf.h"
#include "utils/gpu_comms.h"

template <typename TIn, typename TOut>
TOut *gpuKeyGenLUT(uint8_t **key_as_bytes, int party, int bin, int bout, int N,
                   TIn *d_rin, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, bout);
    gpuKeyGenDPF(key_as_bytes, party, bin, N, d_rin, gaes, true);
    auto d_maskU = randomGEOnGpu<TOut>(N, 1);
    // checkCudaErrors(cudaMemset(d_maskU, 0, N * sizeof(TOut)));
    writeShares<TOut, TOut>(key_as_bytes, party, N, d_maskU, 1);
    auto d_maskV = randomGEOnGpu<TOut>(N, bout);
    // checkCudaErrors(cudaMemset(d_maskV, 0, N * sizeof(TOut)));
    auto d_maskOut = gpuKeyGenSelect<TOut, TOut>(key_as_bytes, party, N, d_maskV, d_maskU, bout);
    gpuLinearComb(bout, N, d_maskOut, TOut(2), d_maskOut, TOut(-1), d_maskV);
    gpuFree(d_maskU);
    gpuFree(d_maskV);
    return d_maskOut;
    // return std::make_pair(d_maskU, d_maskV);
}

__device__ void storeAESBlock(AESBlock *x, int idx, AESBlock y, int N, int threadId)
{
    x[idx * N + threadId] = y;
}
// stripe the stack across all threads for the time being
__device__ AESBlock loadAESBlock(AESBlock *x, int idx, int N, int threadId)
{
    return x[idx * N + threadId];
}

template <typename TIn, typename TOut>
// striping the stack as well for now we'll see what to do later
__global__ void dpfLUT(int party, int bin, int N, TIn *X, TOut *tab, AESBlock *scw_g, AESBlock *stack_g,
                        AESBlock *l0_g, AESBlock *l1_g, uint32_t *tR_g, u32 *U, TOut *V, AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // don't need a sync here at all because there is no data sharing
    // just data reuse
    if (threadId < N)
    {
        storeAESBlock(stack_g, 0, scw_g[threadId], N, threadId);
        auto x = (u64)X[threadId];
        gpuMod(x, bin);
        auto l0_cw = l0_g[threadId];
        auto l1_cw = l1_g[threadId];
        auto tR = tR_g[threadId];
        uint32_t pathStack = 0;
        int depth = 1;
        TOut u = 0, v = 0;
        while (depth > 0)
        {
            auto seed = loadAESBlock(stack_g, depth - 1, N, threadId);
            auto bit = pathStack & 1ULL;
            if (depth == bin - LOG_AES_BLOCK_LEN)
            {
                auto lastBlock = expandDPFTreeNode(bin, party,
                                                   seed,
                                                   0,
                                                   l0_cw,
                                                   l1_cw,
                                                   0,
                                                   uint8_t(bit),
                                                   depth - 1,
                                                   &saes);
                TOut c = party == SERVER1 ? -1 : 1;
                auto lb = pathStack << LOG_AES_BLOCK_LEN;
                // do the dot product here
                for (u64 i = 0; i < AES_BLOCK_LEN_IN_BITS; i++)
                {
                    auto w = c * TOut(lastBlock & 1);
                    u += w;
                    auto lookup = x - (lb ^ i);
                    gpuMod(lookup, bin);
                    v += tab[lookup] * w;
                    lastBlock >>= 1;
                }
                // pop all the 1s from the stack
                while (pathStack & 1ULL /*&& depth > 0*/)
                {
                    pathStack >>= 1;
                    depth--;
                }
                // xor the last 0 with 1 to make it 1
                pathStack ^= 1;
            }
            else
            { 
                // manipulate the seed depending on the bit
                // aren't storing the first cw because it sees no reuse
                auto tR_l = (tR >> (depth - 1)) & 1;
                auto newSeed = expandDPFTreeNode(bin, party,
                                                 seed,
                                                 loadAESBlock(scw_g, depth, N, threadId),
                                                 //   scw[][depth - 1][],
                                                 0,
                                                 0,
                                                 tR_l,
                                                 uint8_t(bit),
                                                 depth - 1,
                                                 &saes);
                storeAESBlock(stack_g, depth, newSeed, N, threadId);
                // stack[threadIdx.x / 32][depth - 1][threadIdx.x & 31] = newSeed;
                depth++;
                // push a 0 on top of the stack
                pathStack <<= 1; // (pathStack << 1) ^ 1ULL;
            }
        }
        // Neha: might want to examine this later
        gpuMod(u, 2);
        if (party == SERVER1)
        {
            u = 2 - ((4 - u) / 2);
        }
        else
        {
            u = (u + 1) / 2;
        }
        gpuMod(u, 1);
        auto maskU = getVCW(1, U, N, 0);
        writeVCW(1, U, u64(u ^ maskU), 0, N);
        V[threadId] += v;
    }
}

template <typename TIn, typename TOut>
TOut *gpuDpfLUT(GPULUTKey<TOut> k0, SigmaPeer *peer, int party, TIn *d_X, TOut *d_tab, AESGlobalContext *g, Stats *s, bool opMasked = true)
{
    auto k = *(k0.k.dpfTreeKey);
    assert(k0.k.bin >= 8 && k0.k.B == 1);
    // Neha: need to change the key reading and writing code
    //  do not change tb size it is needed to load the sbox
    const int tbSz = 256;
    int tb = (k.N - 1) / tbSz + 1;
    AESBlock *d_scw, *d_stack, *d_l0, *d_l1;
    uint32_t *d_tR;
    // *d_out;

    assert(k.memSzScw % (k.bin - LOG_AES_BLOCK_LEN) == 0);

    d_scw = (AESBlock *)moveToGPU((uint8_t *)k.scw, k.memSzScw, s);
    d_stack = (AESBlock *)gpuMalloc(k.memSzScw);
    d_l0 = (AESBlock *)moveToGPU((uint8_t *)k.l0, k.memSzL, s);
    d_l1 = (AESBlock *)moveToGPU((uint8_t *)k.l1, k.memSzL, s);
    d_tR = (u32 *)moveToGPU((uint8_t *)k.tR, k.memSzT, s);
    auto d_U = (u32 *)moveToGPU((u8 *)k0.maskU, k.memSzOut, s); // a lot of bits packed together
    auto d_V = (TOut *)moveToGPU((u8 *)k0.s.b, k.N * sizeof(TOut), s);
    dpfLUT<TIn, TOut><<<tb, tbSz /*, shmSize*/>>>(party, k.bin, k.N, d_X, d_tab, d_scw, d_stack, d_l0, d_l1, d_tR, d_U, d_V, *g);
    checkCudaErrors(cudaDeviceSynchronize());

    gpuFree(d_scw);
    gpuFree(d_stack);
    gpuFree(d_l0);
    gpuFree(d_l1);
    gpuFree(d_tR);

    peer->reconstructInPlace(d_U, 1, k.N, s);
    peer->reconstructInPlace(d_V, k0.bout, k.N, s);
    auto d_O = gpuSelect<TOut, TOut, 0, 0>(peer, party, k0.bout, k0.s, d_U, d_V, s, opMasked);
    gpuLinearComb(k0.bout, k.N, d_O, TOut(2), d_O, TOut(-1 * (opMasked || party == SERVER1)), d_V);
    gpuFree(d_U);
    gpuFree(d_V);
    return d_O;
}
