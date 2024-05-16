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

#include "gpu_avgpool.h"

template <typename T>
__global__ void addPoolKernel(AvgPoolParams p, T *I, T *O, int N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N)
    {
        int t = thread_id;
        int n = t / (p.H * p.W * p.C);
        t = t % (p.H * p.W * p.C);
        int h = t / (p.W * p.C);
        t = t % (p.W * p.C);
        int w = t / p.C;
        int c = t % p.C;
        int leftTopCornerH = h * p.strideH; // - p.zPadHLeft;
        int leftTopCornerW = w * p.strideW; // - p.zPadWLeft;
        T sum = 0;
        for (int fh = 0; fh < p.FH; fh++)
        {
            for (int fw = 0; fw < (p.isLowerTriangular ? h+1 : p.FW); fw++)
            {
                int posH = leftTopCornerH + fh;
                int posW = leftTopCornerW + fw;
                assert(posH >= 0 && posH <= p.imgH);
                assert(posW >= 0 && posW <= p.imgW);
                int idx = (p.isLowerTriangular? n * (p.imgH * (p.imgH + 1)) / 2 + (leftTopCornerH * (leftTopCornerH+1))/2 + posW : n * p.imgH * p.imgW * p.C + posH * p.imgW * p.C + posW * p.C + c);
                sum += I[idx];
                gpuMod(sum, p.bw);
                // if (thread_id <= 10)
                    // printf("Add pool with thread %d, %d, %d: %ld, %d, %d, %d, %d, %ld, %d\n", thread_id, fh, fw, sum, N, p.bw, p.bin, idx, I[idx], p.scaleDiv);
            }
        }
        if (p.scaleDiv)
        {
            T den = (T(1) << p.scaleDiv) / T(p.FH * p.FW);
            sum *= den;
            // printf("Multiplying the sum by a large number: %ld\n", den);
        }
        gpuMod(sum, p.bw);
        O[thread_id] = sum;
        // if (thread_id <= 10)
            // printf("Add pool %d: %ld, %d\n", thread_id, sum, p.bw);
    }
}

template <typename T>
T *gpuAddPoolBackProp(AvgPoolParams p, T *d_incomingGrad, Stats *s)
{
    int outSz = p.N * p.H * p.W * p.C * p.FH * p.FW;
    auto d_expandedGrad = (T *)gpuMalloc(outSz * sizeof(T));
    expandKernel<<<(outSz - 1) / 256 + 1, 256>>>(p, d_incomingGrad, d_expandedGrad, outSz);
    auto d_outgoingGrad = gpuCollectGradients(p, d_expandedGrad, s);
    gpuFree(d_expandedGrad);
    size_t inSz = p.N * p.imgH * p.imgW * p.C;
    T c = (T(1) << p.scaleDiv) / T(p.FH * p.FW);
    gpuLinearComb(p.bw, inSz, d_outgoingGrad, c, d_outgoingGrad);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_outgoingGrad;
}

template <typename T>
T *gpuAddPool(AvgPoolParams p, T *d_I, Stats *s)
{
    // printf("Avg pool: %d\n", p.bw);
    int outSz = getMSz(p);
    T *d_O = (T *)gpuMalloc(outSz * sizeof(T));
    addPoolKernel<<<(outSz - 1) / 256 + 1, 256>>>(p, d_I, d_O, outSz);
    checkCudaErrors(cudaDeviceSynchronize());
    return d_O;
}
