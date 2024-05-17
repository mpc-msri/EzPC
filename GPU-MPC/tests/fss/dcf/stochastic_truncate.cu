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

#include <cassert>
#include <cstdint>

#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/dcf/gpu_truncate.h"

using T = u64;

int main(int argc, char *argv[]) {
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    // initCommBufs(true);
    int bin = 64;
    int bout = 64;
    int shift = 5;
    int N = atoi(argv[3]);
    int party = atoi(argv[1]);
    
    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);

    T *h_I;
    auto d_inputMask = randomGEOnGpu<T>(N, bin);
    // checkCudaErrors(cudaMemset(d_inputMask, 0, N * sizeof(T)));
    auto h_inputMask = (T*) moveToCPU((u8*) d_inputMask, N * sizeof(T), NULL);
    auto d_masked_I = getMaskedInputOnGpu(N, bin, d_inputMask, &h_I);

    u8 *startPtr, *curPtr;
    size_t keyBufSz = 10 * OneGB;
    getKeyBuf(&startPtr, &curPtr, keyBufSz);
    T* h_r = (T*) cpuMalloc(N * sizeof(T));
    auto d_outputMask = dcf::genGPUStochasticTruncateKey(&curPtr, party, bin, bout, shift, N, d_inputMask, &g, h_r);
    assert(curPtr - startPtr < keyBufSz);
    auto h_outputMask = (T*) moveToCPU((u8*) d_outputMask, N * sizeof(T), NULL);
    gpuFree(d_outputMask);

    curPtr = startPtr;
    auto k = dcf::readGPUTrStochasticKey<T>(&curPtr);

    dcf::gpuStochasticTruncate(k, party, peer, d_masked_I, &g, (Stats*) NULL);

    auto h_O = (T*) moveToCPU((u8*) d_masked_I, N * sizeof(T), NULL);
    checkTrStochastic(bin, bout, shift, N, h_O, h_outputMask, h_I, h_r);
}