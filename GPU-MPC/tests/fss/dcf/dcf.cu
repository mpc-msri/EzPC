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

#include <cassert>
#include <chrono>

#include "utils/gpu_random.h"
#include "fss/dcf/gpu_dcf.h"

#include <sytorch/tensor.h>

using T = u64;

int main(int argc, char *argv[])
{
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = atoi(argv[1]);
    int bout = atoi(argv[2]);
    int N = atoi(argv[3]);

    printf("Bin=%d, Bout=%d, N=%d\n", bin, bout, N);

    u8 *ptr1, *ptr2;
    getKeyBuf(&ptr1, &ptr2, 10 * OneGB);
    auto keyBuf1 = ptr1;
    auto keyBuf2 = ptr2;
    // auto d_x = (T*) gpuMalloc(N * sizeof(T));
    // checkCudaErrors(cudaMemset(d_x, 0, N * sizeof(T)));
    printf("N=%d, memSzN=%lu\n", N, N * sizeof(T));
    initGPURandomness();
    auto d_rin = randomGEOnGpu<T>(N, bin);
    auto h_rin = (T *)moveToCPU((u8 *)d_rin, N * sizeof(T), NULL);
    auto d_X = randomGEOnGpu<T>(N, bin);
    auto h_X = (T *)moveToCPU((u8 *)d_X, N * sizeof(T), NULL);
    // printf("%ld\n", h_X[3]);
    destroyGPURandomness();

    initGPURandomness();
    dcf::gpuKeyGenDCF(&keyBuf1, 0, bin, bout, N, d_rin, T(1), &g);
    printf("Key size=%lu\n", keyBuf1 - ptr1);
    auto k1 = dcf::readGPUDCFKey(&ptr1);
    Stats s;
    auto start = std::chrono::high_resolution_clock::now();
    auto d_O1 = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(k1, 0, d_X, &g, (Stats *)&s);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;

    printf("Time taken for P0=%lu micros, transfer time=%lu\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count(), s.transfer_time);

    auto h_O1 = (u32 *)moveToCPU((u8 *)d_O1, k1.memSzOut, (Stats *)NULL);
    gpuFree(d_O1);
    destroyGPURandomness();

    initGPURandomness();
    dcf::gpuKeyGenDCF(&keyBuf2, 1, bin, bout, N, d_rin, T(1), &g);
    auto k2 = dcf::readGPUDCFKey(&ptr2);

    start = std::chrono::high_resolution_clock::now();
    auto d_O2 = dcf::gpuDcf<T, 1, dcf::idPrologue, dcf::idEpilogue>(k2, 1, d_X, &g, (Stats *)NULL);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;

    printf("Time taken for P1=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());

    auto h_O2 = (u32 *)moveToCPU((u8 *)d_O2, k2.memSzOut, NULL);
    gpuFree(d_O2);
    destroyGPURandomness();

    for (int i = 0; i < N; i++)
    {
        // auto o1 = h_O1[i];
        // auto o2 = h_O2[i];
        // auto o = (o1 + o2);
        // cpuMod(o, bout);
        // if (i < 10 || (o != (h_X[i] < h_rin[i])))
        //     printf("%d: %lu, %lu, %lu, %lu, %lu\n", i, o1, o2, o, h_X[i], h_rin[i]);
        // // assert((h_O1[i] ^ h_O2[i]) == u32(0));
        // // assert(o == (h_X[i] < h_rin[i]));
        // assert(o == (h_X[i] < h_rin[i]));

        auto o1 = (h_O1[i / 32] >> (i & 31)) & T(1);
        auto o2 = (h_O2[i / 32] >> (i & 31)) & T(1);
        auto o = (o1 + o2) & u32(1);
        if (i < 10 || (o != (h_X[i] < h_rin[i])))
            printf("%d: %u, %u, %u, %lu, %lu, %u\n", i, o1, o2, o, h_X[i], h_rin[i], i / 32);
        // assert((h_O1[i] ^ h_O2[i]) == u32(0));
        // assert(o == (h_X[i] < h_rin[i]));
        assert(o == (h_X[i] < h_rin[i]));
    }
    return 0;
}