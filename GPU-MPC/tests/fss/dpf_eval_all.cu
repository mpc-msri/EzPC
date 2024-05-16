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
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_comms.h"

#include <cassert>

#include "utils/gpu_random.h"
#include "fss/gpu_lut.h"

#include <sytorch/tensor.h>

using TIn = u64;
using TOut = u64;

template <typename TIn>
// striping the stack as well for now we'll see what to do later
__global__ void dpfEvalAll(int party, int bin, int N, TIn *X, AESBlock *scw_g, AESBlock *stack_g,
                        AESBlock *l0_g, AESBlock *l1_g, u32 *tR_g, u32* U, AESGlobalContext gaes)
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
        u32 pathStack = 0;
        int depth = 1;
        TOut u = 0;
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
                    lastBlock >>= 1;
                }
                // sum &= 1;
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
                depth++;
                pathStack <<= 1; 
            }
        }
        // Neha: might want to examine this later
        gpuMod(u, 1);
        writeVCW(1, U, u64(u), 0, N);
    }
}


template <typename TIn>
u32 *gpuDpfEvalAll(GPUDPFKey k0, int party, TIn *d_X, AESGlobalContext *g, Stats *s)
{
    auto k = *(k0.dpfTreeKey);
    assert(k0.bin >= 8 && k0.B == 1);

    const int tbSz = 256;
    int tb = (k.N - 1) / tbSz + 1;
    AESBlock *d_scw, *d_stack, *d_l0, *d_l1;
    u32 *d_tR;
    // *d_out;

    assert(k.memSzScw % (k.bin - LOG_AES_BLOCK_LEN) == 0);

    d_scw = (AESBlock *)moveToGPU((uint8_t *)k.scw, k.memSzScw, s);
    d_stack = (AESBlock *)gpuMalloc(k.memSzScw);
    d_l0 = (AESBlock *)moveToGPU((uint8_t *)k.l0, k.memSzL, s);
    d_l1 = (AESBlock *)moveToGPU((uint8_t *)k.l1, k.memSzL, s);
    d_tR = (u32 *)moveToGPU((uint8_t *)k.tR, k.memSzT, s);
    auto d_U = (u32 *) gpuMalloc(k.memSzOut); // a lot of bits packed together

    dpfEvalAll<TIn><<<tb, tbSz /*, shmSize*/>>>(party, k.bin, k.N, d_X, d_scw, d_stack, d_l0, d_l1, d_tR, d_U, *g);
    checkCudaErrors(cudaDeviceSynchronize());
    // auto end = std::chrono::high_resolution_clock::now();
    // auto elapsed = end - start;
    // printf("Time taken by dpfLUT kernel=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

    gpuFree(d_scw);
    gpuFree(d_stack);
    gpuFree(d_l0);
    gpuFree(d_l1);
    gpuFree(d_tR);

    return d_U;
}


int main(int argc, char *argv[])
{
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = atoi(argv[1]);
    
    int N = atoi(argv[2]);//1536;//16384;
    int party = 0;//atoi(argv[1]);

    uint8_t *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 40 * OneGB);

    initGPURandomness();
    auto d_rin = randomGEOnGpu<TIn>(N, bin);
    // destroyGPURandomness();

    // initGPURandomness();
    printf("Started DPF keygen\n");
    gpuKeyGenDPF(&curPtr, party, bin, N, d_rin, &g, true);
    auto k = readGPUDPFKey(&startPtr);
    Stats s;
    printf("Starting DPF eval\n");
    auto start = std::chrono::high_resolution_clock::now();
    u32 *d_O = gpuDpfEvalAll(k, party, d_rin, &g, (Stats *)&s);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Time taken for P0=%lu micros, Transfer time=%lu\n", elapsed.count(), s.transfer_time);

 
    // printf("Bin=%d, N=%d\n", bin, N);
    // for (int i = 0; i < N; i++)
    // {
    //     auto unmasked_O = (h_O[i] - h_mask_O[i]);
    //     mod(unmasked_O, bout);
    //     if (i < 10 || h_X[i] != unmasked_O)
    //         printf("%d: %ld, %ld, %lf\n", i, h_X[i], unmasked_O, asFloat(unmasked_O, bout, 12)); //double(h_X[i]) / (1ULL << 6), double(unmasked_O) / (1ULL << 12));
    //     assert(h_X[i] == unmasked_O);
    // }
    // // gpuFree(d_identity);
    return 0;
}