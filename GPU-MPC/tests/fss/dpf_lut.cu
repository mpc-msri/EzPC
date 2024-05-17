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

using TIn = u8;
using TOut = u64;

int main(int argc, char *argv[])
{
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64;
    int bout = 50;
    // auto d_inv = genLUT<T, inv<T>>(bin, 6, 12);
    auto d_tab = genLUT<TOut, identity<TOut>>(bin, 6, 12);
    int N = 10000000;//1536;//16384;
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(true);
    peer->connect(party, "0.0.0.0");

    uint8_t *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);
    // auto keyBuf1 = ptr1;
    // auto keyBuf2 = ptr2;
    // auto d_x = (T*) gpuMalloc(N * sizeof(T));
    // checkCudaErrors(cudaMemset(d_x, 0, N * sizeof(T)));

    initGPURandomness();
    auto d_mask_X = randomGEOnGpu<TIn>(N, bin);
    auto h_mask_X = (TIn *)moveToCPU((u8 *)d_mask_X, N * sizeof(TIn), NULL);
    TIn *h_X;
    auto d_masked_X = getMaskedInputOnGpu(N, bin, d_mask_X, &h_X);
    // destroyGPURandomness();

    // initGPURandomness();
    auto d_mask_O = gpuKeyGenLUT<TIn, TOut>(&curPtr, party, bin, bout, N, d_mask_X, &g);
    auto h_mask_O = (TOut *)moveToCPU((u8 *)d_mask_O, N * sizeof(TOut), NULL);
    auto k = readGPULUTKey<TOut>(&startPtr);
    auto start = std::chrono::high_resolution_clock::now();
    TOut *d_O = gpuDpfLUT<TIn, TOut>(k, peer, party, d_masked_X, d_tab, &g, (Stats *)NULL, false);
    peer->reconstructInPlace(d_O, bout, N, (Stats*) NULL);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

    auto h_O = (TOut *)moveToCPU((uint8_t *)d_O, N * sizeof(TOut), (Stats *)NULL);
    gpuFree(d_O);
    destroyGPURandomness();

    printf("Bin=%d, N=%d\n", bin, N);
    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = (h_O[i] - h_mask_O[i]);
        mod(unmasked_O, bout);
        if (i < 10 || h_X[i] != unmasked_O)
            printf("%d: %ld, %ld, %lf\n", i, h_X[i], unmasked_O, asFloat(unmasked_O, bout, 12)); //double(h_X[i]) / (1ULL << 6), double(unmasked_O) / (1ULL << 12));
        assert(h_X[i] == unmasked_O);
    }
    // gpuFree(d_identity);
    return 0;
}