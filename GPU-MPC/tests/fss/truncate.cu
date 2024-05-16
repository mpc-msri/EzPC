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
#include "utils/gpu_random.h"

#include "fss/gpu_truncate.h"

#include <cassert>
#include <sytorch/tensor.h>

using T = u64;

int main(int argc, char *argv[])
{
    // initCommBufs(true);
    // initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64;
    int bout = 40;
    int shift = 24;
    int N = atoi(argv[3]);
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);
    
    TruncateType t = TruncateType::TrWithSlack;

    uint8_t *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 10 * OneGB);
    
    initGPURandomness();
    auto d_mask_X = randomGEOnGpu<T>(N, bin);
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    T *h_X;
    auto d_masked_X = getMaskedInputOnGpu(N, bin, d_mask_X, &h_X, true, bin - 1);

    auto d_truncateMask = genGPUTruncateKey<T, T>(&curPtr, party, t, bin, bout, shift, N, d_mask_X, &g);
    printf("Key size=%lu\n", curPtr - startPtr);
    auto h_mask_O = (T *)moveToCPU((u8 *)d_truncateMask, N * sizeof(T), NULL);
    gpuFree(d_truncateMask);
    // printf("here\n");
    auto k1 = readGPUTruncateKey<T>(t, &startPtr);
    T *d_O;
    // printf("here\n");
    for (int i = 0; i < 1; i++)
    {
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        d_O = gpuTruncate<T, T>(bin, bout, t, k1, shift, peer, party, N, d_masked_X, &g, (Stats *)NULL);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    }

    auto h_O = (T *)moveToCPU((uint8_t *)d_O, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_O);
    destroyGPURandomness();

    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = (h_O[i] - h_mask_O[i]);
        cpuMod(unmasked_O, bout);
        auto o = cpuArs(h_X[i], bin, shift);
        cpuMod(o, bout);
        if (i < 10 || o != unmasked_O)
            printf("%d: %ld, %ld, %ld, %ld\n", i, h_X[i], o, unmasked_O, h_mask_X[i]);
        assert(o == unmasked_O);
    }

    return 0;
}