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
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include <cassert>
#include <numeric>
#include <sytorch/backend/cleartext.h>

#include "fss/gpu_softmax.h"

using T = u64;

int main(int argc, char *argv[])
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    // initCommBufs(true);
    MaxpoolParams p;
    p.bw = 50;
    p.bin = 38;
    p.scale = 12;
    p.scaleDiv = 0;
    p.bwBackprop = 0;
    p.N = 12;
    p.imgH = atoi(argv[2]); // 128 * 12;
    p.imgW = atoi(argv[3]); // 128;
    p.C = 1;
    p.FH = 1;
    p.FW = p.imgW;
    p.strideH = 1;
    p.strideW = p.FW;
    p.zPadHLeft = 0;
    p.zPadHRight = 0;
    p.zPadWLeft = 0;
    p.zPadWRight = 0;
    p.H = ((p.imgH - p.FH + (p.zPadHLeft + p.zPadHRight)) / p.strideH) + 1;
    p.W = ((p.imgW - p.FW + (p.zPadWLeft + p.zPadWRight)) / p.strideW) + 1;
    p.isLowerTriangular = true;
    printf("Output H=%d, output W=%d\n", p.H, p.W);
    auto d_nExpMsbTab = genLUT<T, nExpMsb<T>>(8, 4, p.scale);
    auto d_nExpLsbTab = genLUT<T, nExpLsb<T>>(8, 12, p.scale);
    auto d_invTab = genLUT<T, inv<T>>(int(ceil(log2(p.FW))) + p.scale, 6 /*p.FW*/, p.scale);

    int party = atoi(argv[1]);

    auto peer = new GpuPeer(true);
    peer->connect(party, argv[4]);
    
    // printf("Here\n");
    int inSz = getInSz(p);
    int outSz = p.N * p.imgH * p.imgW * p.C;
    // printf("N=%d, %d\n", inSz, p.N * p.imgH * p.imgW * p.C);

    T *h_I;
    auto d_mask_I = randomGEOnGpu<T>(inSz, p.bin);
    // checkCudaErrors(cudaMemset(d_mask_I, 0, inSz * sizeof(T)));
    auto h_mask_I = (T *)moveToCPU((u8 *)d_mask_I, inSz * sizeof(T), NULL);
    auto d_masked_I = getMaskedInputOnGpu(inSz, p.bw, d_mask_I, &h_I, true, 15);

    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 10 * OneGB);

    auto d_mask_O = gpuKeygenSoftmax(&curPtr, party, p, d_mask_I, &g);
    auto h_mask_O = (T *)moveToCPU((u8 *)d_mask_O, outSz * sizeof(T), NULL);

    auto k = readGPUSoftMaxKey<T>(p, &startPtr);
    Stats s;
    s.compute_time = 0;
    peer->sync();
    auto startComm = peer->bytesSent() + peer->bytesReceived();
    auto start = std::chrono::high_resolution_clock::now();
    auto d_O = gpuSoftmax(peer, party, p, k, d_masked_I, d_nExpMsbTab, d_nExpLsbTab, d_invTab, &g, (Stats *)&s);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    printf("Maxpool time=%lu micros\n", s.compute_time);
    auto endComm = peer->bytesSent() + peer->bytesReceived();
    printf("Comm=%ld B\n", endComm - startComm);
    auto h_O = (T *)moveToCPU((u8 *)d_O, outSz * sizeof(T), NULL);

    for (int i = 0; i < outSz; i++)
    {
        h_O[i] = h_O[i] - h_mask_O[i];
        cpuMod(h_O[i], p.bw);
    }
    auto ct = new ClearText<i64>();
    ct->bw = p.bw;
    
    auto d_I = (T *)moveToGPU((u8 *)h_I, inSz * sizeof(T), NULL);
    T mInf = -(1ULL << (p.bin - 1));
    auto d_eI = expandLowerTriangularMatrix(p, d_I, mInf);
    auto h_eI = (T *)moveToCPU((u8 *)d_eI, outSz * sizeof(T), NULL);
    Tensor<i64> t1((i64 *)h_eI, {(u64)p.N * p.imgH, (u64)p.imgW});
    Tensor<i64> t2({(u64)p.N * p.imgH, (u64)p.imgW});
    ct->softmax(t1, t2, p.scale, 0);
    for (int i = 0; i < outSz; i++)
    {
        if (i < 10)
            printf("Index %d=%ld, %ld, %lf\n", i, t2.data[i], h_O[i], asFloat(h_eI[i], p.bw, p.scale));

        if (T(t2.data[i]) != h_O[i])
        {
            printf("Index %d=%ld, %ld, %lf\n", i, t2.data[i], h_O[i], asFloat(h_eI[i], p.bw, p.scale));
            assert(0);
        }
    }
    return 0;
}