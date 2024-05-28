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

#include "gpu_softmax.h"
#include "gpu_avgpool.h"
#include "gpu_truncate.h"

template <typename T>
__global__ void expandLtMatrixKernel(int N, int imgH, int imgW, T *ltA, T *A, T c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * imgH * imgW)
    {
        // have something to write
        int t = i;
        int n = t / (imgH * imgW);
        t = t % (imgH * imgW);
        int h = t / imgW;
        int w = t % imgW;
        if (w <= h)
        {
            // add an element
            A[i] = ltA[n * (imgH * (imgH + 1)) / 2 + (h * (h + 1)) / 2 + w];
        }
        else
        {
            // add zero
            A[i] = c;
        }
    }
}

template <typename T>
T *expandLowerTriangularMatrix(MaxpoolParams p, T *d_ltA, T c = 0)
{
    p.isLowerTriangular = false;
    int sz = getInSz(p);
    auto d_A = (T *)gpuMalloc(sz * sizeof(T));
    expandLtMatrixKernel<<<(sz - 1) / 128 + 1, 128>>>(p.N, p.imgH, p.imgW, d_ltA, d_A, c);
    return d_A;
}

template <typename T>
T *gpuKeygenSoftmax(u8 **key_as_bytes, int party, MaxpoolParams p, T *d_mask_X, AESGlobalContext *gaes)
{
    int inSz = getInSz(p);
    int mSz = getMSz(p);
    int ogBw = p.bw;
    int reducedBw = p.bin + 2;
    // int bin = p.bin;
    // Neha: can change this bin later if necessary but I don't think we need to
    // do an implicit reduce to bin + 2 bits
    p.bw = reducedBw;
    // +1 slack for maxpool
    p.bin = p.bin + 1;
    // get the max in 39 bits (implicit reduce)
    // in this case the input bw and the output bw are the same
    auto d_maxMask = gpuKeygenMaxpool(key_as_bytes, party, p, d_mask_X, gaes, true);
    assert(p.strideH == p.FH && p.strideW == p.FW);
    auto d_X1Mask = windowFunc<T, xPlusM<u64(-1), u64(1)>>(party, p, d_mask_X, d_maxMask);
    gpuFree(d_maxMask);
    // +1 slack for sum, nexp also needs a +1 slack, so overall need a +2 slack
    auto d_expX1Mask = gpuKeygenNExp(key_as_bytes, party, ogBw, reducedBw, p.scale, inSz, d_X1Mask, gaes);
    gpuFree(d_X1Mask);
    p.bw = ogBw;
    auto d_sumExpX1 = gpuAddPool(p, d_expX1Mask, NULL);
    // assert(p.imgW <= 128);
    auto d_sumExpX1Inv = gpuKeygenLUTInverse(key_as_bytes, party, p.bw, p.scale + int(ceil(log2(p.FW))), p.scale, mSz, d_sumExpX1, gaes);
    gpuFree(d_sumExpX1);
    auto d_softmaxMask = keygenWindowMul(key_as_bytes, party, p, d_expX1Mask, d_sumExpX1Inv, TruncateType::TrWithSlack, gaes);
    gpuFree(d_expX1Mask);
    gpuFree(d_sumExpX1Inv);
    if (p.isLowerTriangular)
    {
        auto d_eSoftmaxMask = expandLowerTriangularMatrix(p, d_softmaxMask);
        gpuFree(d_softmaxMask);
        d_softmaxMask = d_eSoftmaxMask;
    }
    return d_softmaxMask;
}

template <typename T>
T *gpuSoftmax(SigmaPeer *peer, int party, MaxpoolParams p, GPUSoftMaxKey<T> k, T *d_X, T *d_nExpMsbTab, T *d_nExpLsbTab, T *d_invTab, AESGlobalContext *gaes, Stats *s)
{
    u64 b0 = peer->bytesSent() + peer->bytesReceived();
    // need to make sure that this works for N > 1
    int inSz = getInSz(p);
    int mSz = getMSz(p);
    int ogBw = p.bw;
    int reducedBw = p.bin + 2;
    p.bw = reducedBw;
    // need a +1 slack for maxpool
    p.bin = p.bin + 1;
    auto start = std::chrono::high_resolution_clock::now();
    // doesn't assume anything about a gap
    auto d_max = gpuMaxpool(peer, party, p, k.maxPoolKey, d_X, gaes, s);
    assert(p.strideH == p.FH && p.strideW == p.FW);
    auto d_X1 = windowFunc<T, xPlusM<u64(-1), u64(1)>>(party, p, d_X, d_max);
    gpuFree(d_max);
    // need a +1 slack for nExp for the first clip (which computes drelu(x - 2^16))
    auto d_expX1 = gpuNExp(peer, party, ogBw, reducedBw, p.scale, inSz, k.nExpKey, d_X1, d_nExpMsbTab, d_nExpLsbTab, gaes, s);
    gpuFree(d_X1);
    p.bw = ogBw;
    auto d_sumExpX1 = gpuAddPool(p, d_expX1, NULL);
    auto d_sumExpX1Inv = gpuLUTInverse(peer, party, p.bw, p.scale + int(ceil(log2(p.FW))), p.scale, mSz, k.invKey, d_sumExpX1, d_invTab, gaes, s);
    gpuFree(d_sumExpX1);
    auto d_softmax = windowMul(peer, party, p, k.wMulKey, d_expX1, d_sumExpX1Inv, TruncateType::TrWithSlack, gaes, s);
    gpuFree(d_expX1);
    gpuFree(d_sumExpX1Inv);
    if (p.isLowerTriangular)
    {
        auto d_eSoftmax = expandLowerTriangularMatrix(p, d_softmax);
        gpuFree(d_softmax);
        d_softmax = d_eSoftmax;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    s->softmax_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    u64 b1 = peer->bytesSent() + peer->bytesReceived();
    s->softmax_comm_bytes += (b1 - b0);
    return d_softmax;
}