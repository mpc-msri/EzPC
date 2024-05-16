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

#include "../../../utils/gpu_data_types.h"
#include "../../../utils/gpu_file_utils.h"
#include "../../../utils/misc_utils.h"
#include "../../../utils/gpu_mem.h"
#include "../../../utils/gpu_random.h"
#include "../../../utils/gpu_comms.h"

#include "../../../fss/dcf/gpu_relu.h"

#include <cassert>
#include <sytorch/tensor.h>

using T = u64;

int main(int argc, char *argv[])
{
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 40;
    int bout = 64;
    int N = atoi(argv[3]); //8;
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);

    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 10 * OneGB);
    initGPURandomness();
    auto d_mask_X = randomGEOnGpu<T>(N, bin);
    // checkCudaErrors(cudaMemset(d_mask_X, 0, N * sizeof(T)));
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    T *h_X;
    auto d_masked_X = getMaskedInputOnGpu(N, bin, d_mask_X, &h_X);

    auto d_tempMask = dcf::gpuKeygenReluExtend(&curPtr, party, bin, bout, N, d_mask_X, &g);
    auto d_dreluMask = d_tempMask.first;
    gpuFree(d_dreluMask);
    auto d_reluMask = d_tempMask.second;
    printf("Key size=%lu\n", curPtr - startPtr);
    auto h_mask_O = (T *)moveToCPU((u8 *)d_reluMask, N * sizeof(T), NULL);
    // printf("here\n");
    auto k1 = dcf::readGPUReluExtendKey<T>(&startPtr);
    T *d_relu;
    // printf("here\n");
    for (int i = 0; i < 1; i++)
    {
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        auto temp = dcf::gpuReluExtend(peer, party, k1, d_masked_X, &g, (Stats *)NULL);
        auto d_drelu = temp.first;
        gpuFree(d_drelu);
        d_relu = temp.second;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    }

    auto h_relu = (T *)moveToCPU((u8 *)d_relu, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_relu);
    destroyGPURandomness();

    for (int i = 0; i < N; i++)
    {
        auto unmasked_O = (h_relu[i] - h_mask_O[i]);
        cpuMod(unmasked_O, bout);
        auto o = h_X[i] * (1 - (h_X[i] >> (bin - 1)));
        if (i < 10)
            printf("%d: %ld, %ld, %ld, %ld, %ld, %ld\n", i, h_X[i], o, unmasked_O, h_mask_X[i], h_relu[i], h_mask_O[i]);
        assert(o == unmasked_O);
    }

    return 0;
}