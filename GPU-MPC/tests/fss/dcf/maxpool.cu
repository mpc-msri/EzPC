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

#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/dcf/gpu_maxpool.h"

#include <llama/array.h>

using T = u32;

template <typename T>
void maxPool2D(MaxpoolParams p, T *in, T *out, T *incomingGrad, T *outgoingGrad)
{
    memset(outgoingGrad, 0, p.N * p.imgH * p.imgW * p.C * sizeof(T));
    for (int i = 0; i < p.N; i++)
    {
        for (int j = 0; j < p.H; j++)
        {
            for (int k = 0; k < p.W; k++)
            {
                for (int l = 0; l < p.C; l++)
                {
                    u64 M = 0;
                    u64 maxIdxI = 0;
                    u64 maxIdxJ = 0;
                    int leftTopCornerH = j * p.strideH - p.zPadHLeft;
                    int leftTopCornerW = k * p.strideW - p.zPadWLeft;
                    for (int m = 0; m < p.FH; m++)
                    {
                        for (int n = 0; n < p.FW; n++)
                        {
                            u64 val = 0;
                            int posH = leftTopCornerH + m;
                            int posW = leftTopCornerW + n;
                            if (posH >= 0 && posH <= p.imgH && posW >= 0 && posW <= p.imgW)
                            {
                                // printf("%d, %d, %d, %d\n", i, posH, posW, l);
                                val = Arr4DIdx(in, p.N, p.imgH, p.imgW, p.C, i, posH, posW, l);
                            }
                            // printf("Val=%lu, %d\n", val, i * p.imgH * p.imgW * p.C + posH * p.imgW * p.C + posW * p.C + l);
                            if (m == 0 && n == 0)
                                M = val;
                            else if (((val - M) & ((T(1) << p.bin) - 1)) < (T(1) << (p.bin - 1)))
                            {
                                M = val;
                                maxIdxI = m;
                                maxIdxJ = n;
                            }
                        }
                    }
                    Arr4DIdx(out, p.N, p.H, p.W, p.C, i, j, k, l) = M;
                    auto inGrad = Arr4DIdx(incomingGrad, p.N, p.H, p.W, p.C, i, j, k, l);
                    auto gradSum = Arr4DIdx(outgoingGrad, p.N, p.imgH, p.imgW, p.C, i, j * p.strideH + maxIdxI, k * p.strideW + maxIdxJ, l);
                    gradSum = (gradSum + inGrad);
                    cpuMod(gradSum, p.bwBackprop);
                    Arr4DIdx(outgoingGrad, p.N, p.imgH, p.imgW, p.C, i, j * p.strideH + maxIdxI, k * p.strideW + maxIdxJ, l) = gradSum;
                    // printf("maxI, maxJ = %d, %d\n", maxIdxI, maxIdxJ);
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    // initCommBufs(true);
    int bin = 20;
    int bout = 20;
    int bwBackprop = 32;
    int N = 100;
    int imgH = 30;
    int imgW = 30;
    int C = 3;
    int FH = 5;
    int FW = 5;
    int strideH = 2;
    int strideW = 2;
    int zPadHLeft = 0;
    int zPadHRight = 0;
    int zPadWLeft = 0;
    int zPadWRight = 0;
    bool useMomentum = true;
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);

    MaxpoolParams p = {bin, bout, };
    int inSz = getInSz(p);
    int outSz = getMSz(p);

    T *h_I, *h_incomingGrad;
    auto d_inputMask = randomGEOnGpu<T>(inSz, bin);
    // checkCudaErrors(cudaMemset(d_inputMask, 0, inSz * sizeof(T)));
    auto h_inputMask = (T *)moveToCPU((u8 *)d_inputMask, inSz * sizeof(T), NULL);
    auto d_masked_I = getMaskedInputOnGpu(inSz, bin, d_inputMask, &h_I);

    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);

    auto d_outputMask = dcf::gpuKeygenMaxpool(&startPtr, party, p, d_inputMask, (u8*) NULL, &g);
    auto h_outputMask = (T *)moveToCPU((u8 *)d_outputMask, outSz * sizeof(T), NULL);

    curPtr = startPtr;
    auto k = dcf::readGPUMaxpoolKey<T>(p, &curPtr);

    auto d_O = dcf::gpuMaxPool(peer, party, p, k, d_masked_I, (u32*) NULL, &g, (Stats*) NULL);
    auto h_O = (T *)moveToCPU((u8 *)d_O, outSz * sizeof(T), NULL);

    T *ct_o = new T[outSz];
    maxPool2D(p, h_I, ct_o, h_incomingGrad, outgoingGradCt);
    for (int i = 0; i < outSz; i++)
    {
        auto unmasked_output = (h_O[i] - h_outputMask[i]);
        cpuMod(unmasked_output, bout);
        if (i < 10 || unmasked_output != ct_o[i])
            printf("%d=%lu %lu\n", i, unmasked_output, ct_o[i]);

        assert(unmasked_output == ct_o[i]);
    }
    return 0;
}