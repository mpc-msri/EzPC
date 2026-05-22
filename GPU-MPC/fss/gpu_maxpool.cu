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
#include "utils/gpu_random.h"
#include "gpu_relu.h"

template <typename T>
__global__ void populateCurMax(MaxpoolParams p, T *curMax, T *img, int N)
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
        curMax[thread_id] = T(0);
        int leftTopCornerH = h * p.strideH - p.zPadHLeft;
        int leftTopCornerW = w * p.strideW - p.zPadWLeft;
        assert(leftTopCornerH < p.imgH);
        assert(leftTopCornerW < p.imgW);
        if (leftTopCornerH >= 0 && leftTopCornerW >= 0)
        {
            curMax[thread_id] = img[n * p.imgH * p.imgW * p.C + leftTopCornerH * p.imgW * p.C + leftTopCornerW * p.C + c];
        }
    }
}

/* out needs to be zeroed out before output is written into it. Am currently NOT adding a check for this */
template <typename T>
__global__ void diffWithCurMax(MaxpoolParams p, int fh, int fw, T *curMax, T *img, T *diff, int N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N)
    {
        // printf("CurMax %d=%ld\n", thread_id, curMax[thread_id]);
        int t = thread_id;
        int n = t / (p.H * p.W * p.C);
        t = t % (p.H * p.W * p.C);
        int h = t / (p.W * p.C);
        t = t % (p.W * p.C);
        int w = t / p.C;
        int c = t % p.C;
        int leftTopCornerH = h * p.strideH - p.zPadHLeft;
        int leftTopCornerW = w * p.strideW - p.zPadWLeft;
        int posH = leftTopCornerH + fh;
        int posW = leftTopCornerW + fw;
        // assert(posH >= 0 && posH <= p.imgH);
        // assert(posW >= 0 && posW <= p.imgW);
        diff[thread_id] = 0;
        if (posH >= 0 && posH < p.imgH && posW >= 0 && posW < p.imgW)
        {
            T toCmp1 = curMax[thread_id];
            T toCmp2 = img[n * p.imgH * p.imgW * p.C + posH * p.imgW * p.C + posW * p.C + c];
            diff[thread_id] = (toCmp2 - toCmp1);
            gpuMod(diff[thread_id], p.bw);
        }
    }
}

// no memory leak
template <typename T>
T *gpuMaxpoolLinHelper(SigmaPeer *peer, int party, MaxpoolParams p, GPUReluKey<T> k, int fh, int fw,
                       T *d_curMax, T *d_in,
                       AESGlobalContext *gaes, Stats *s)
{
    int outSz = getMSz(p);
    T *d_diff = (T *)gpuMalloc(outSz * sizeof(T));
    // int tb_size = 256;
    diffWithCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, fh, fw, d_curMax, d_in, d_diff, outSz);
    checkCudaErrors(cudaDeviceSynchronize());
    // relu(x-y)
    auto d_newMax = gpuRelu<T, T, 0, 0, false>(peer, party, k, d_diff, gaes, s);
    gpuFree(d_diff);
    // relu(x-y) + y
    gpuLinearComb(p.bw, outSz, d_newMax, T(1), d_newMax, T(1), d_curMax);
    return d_newMax;
}

template <typename T>
T *gpuMaxpoolLin(SigmaPeer *peer, int party, MaxpoolParams p, GPUMaxpoolKey<T> k, T *d_I, AESGlobalContext *gaes, Stats *s)
{
    int outSz = getMSz(p);
    T *d_curMax = (T *)gpuMalloc(outSz * sizeof(T));
    for (int i = 0; i < p.FH; i++)
    {
        for (int j = 0; j < p.FW; j++)
        {
            if (i == 0 && j == 0)
            {
                populateCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, d_curMax, d_I, outSz);
                continue;
            }
            // printf("Inside Maxpool=%d, %d\n", i, j);
            auto d_newMax = gpuMaxpoolLinHelper(peer, party, p, k.reluKey[i * p.FW + j - 1], i, j, d_curMax, d_I, gaes, s);
            // printf("Finished Maxpool=%d, %d\n", i, j);
            gpuFree(d_curMax);
            d_curMax = d_newMax;
        }
    }
    return d_curMax;
}

template <typename T>
T *gpuKeygenMaxpoolLinHelper(u8 **key_as_bytes, int party, MaxpoolParams p, int fh, int fw, T *d_curMaxMask, T *d_inputMask, AESGlobalContext *gaes)
{
    // printf("gpu addr=%lx\n", d_inputMask);
    int outSz = getMSz(p);
    // printf("############## FSS maxpool outsz=%lu\n", outSz);
    T *d_diffMask = (T *)gpuMalloc(outSz * sizeof(T));
    // int tb_size = 256;
    diffWithCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, fh, fw, d_curMaxMask, d_inputMask, d_diffMask, outSz);
    checkCudaErrors(cudaDeviceSynchronize());
    // auto d_newMaxMask = randomGEOnGpu<T>(outSz, p.bw);
    // relu(x-y)
    auto d_newMaxMask = gpuGenReluKey<T, T, 0, 0, false>(key_as_bytes, party, p.bin, p.bw, outSz, d_diffMask, gaes);
    gpuFree(d_diffMask);
    // relu(x-y) + y
    gpuLinearComb(p.bw, outSz, d_newMaxMask, T(1), d_newMaxMask, T(1), d_curMaxMask);
    return d_newMaxMask;
}

template <typename T>
T *gpuKeygenMaxpoolLin(uint8_t **key_as_bytes, int party, MaxpoolParams p, T *d_inputMask, AESGlobalContext *gaes)
{
    writeInt(key_as_bytes, p.FH * p.FW - 1);
    int outSz = getMSz(p);
    T *d_curMaxMask = (T *)gpuMalloc(outSz * sizeof(T));
    for (int i = 0; i < p.FH; i++)
    {
        for (int j = 0; j < p.FW; j++)
        {
            if (i == 0 && j == 0)
            {
                populateCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, d_curMaxMask, d_inputMask, outSz);
                continue;
            }
            auto d_newMaxMask = gpuKeygenMaxpoolLinHelper(key_as_bytes, party, p, i, j, d_curMaxMask, d_inputMask, gaes);
            gpuFree(d_curMaxMask);
            d_curMaxMask = d_newMaxMask;
        }
    }
    return d_curMaxMask;
}

template <typename T>
__global__ void sub(int bw, int N, int imgH, int imgW, int i, bool isLowerTriangular, T *d_I, T *d_O)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (isLowerTriangular)
    {
        int l = i + 1;
        int elemsPerImg = ((imgH * imgH) / (1ULL << (l + 1)) + imgH / 2);
        int oLen = N * elemsPerImg;
        if (j < oLen)
        {
            int n = j / elemsPerImg;
            int k = j % elemsPerImg;
            float twoPowl = float(1ULL << l);
            float fourPowl = float(1ULL << (2 * l));
            int twoPowlInt = (1ULL << l);
            int h1 = int(floor((-twoPowl + __fsqrt_rd(fourPowl + 8 * twoPowl * k)) / (2.0f * twoPowl)));
            int elemsInRow = h1 + 1;
            int h2 = (k - ((h1 * (h1 + 1)) / 2) * twoPowlInt) / elemsInRow;
            int h = h1 * twoPowlInt + h2;
            // printf("%d=%d, %d, %d\n", k, h1, h2, h);
            int w = k - ((h1 * (h1 + 1)) / 2) * twoPowlInt - h2 * elemsInRow;
            assert(w < elemsInRow);
            int inW = 2 * w;
            T o = 0;
            float twoPowi = float(1ULL << i);
            elemsInRow = int(ceil((h + 1) / twoPowi));
            if (inW + 1 < elemsInRow)
            {
                int twoPowiInt = (1ULL << i);
                h1 = int(floor(h / twoPowi));
                h2 = ((h1 * (h1 + 1)) / 2) * twoPowiInt;
                int h3 = (h % twoPowiInt) * elemsInRow;
                elemsPerImg = ((imgH * imgH) / (1ULL << (i + 1)) + imgH / 2);
                int idx = n * elemsPerImg + h2 + h3 + inW;
                o = d_I[idx + 1] - d_I[idx];
                gpuMod(o, bw);
                // printf("i=%d, j=%d, idx=%d, i0=%ld, i1=%ld, o=%ld\n", i, j, idx, d_I[idx], d_I[idx + 1], o);
            }
            d_O[j] = o;
        }
    }
    else
    {
        int oLen = N * imgH * (imgW / (1ULL << (i + 1)));
        int curW = (imgW / (1ULL << (i)));
        int newW = (imgW / (1ULL << (i + 1)));
        if (j < oLen)
        {
            // now you know you have an element to fill (potentially)
            int m = j / newW;
            int n = j % newW;
            int nIn = 2 * n;
            T o = 0; // d_I[m * N + nIn];
            if (nIn + 1 < curW)
            {
                // both operands are legit, do the subtraction and store the result
                o = d_I[m * curW + nIn + 1] - d_I[m * curW + nIn];
                gpuMod(o, bw);
                // printf("sub %d, %d, %d: %ld-%ld=%ld\n", i, m, nIn, d_I[m * N + nIn + 1], d_I[m * N + nIn], o);
            }
            d_O[m * newW + n] = o;
        }
    }
}

template <typename T>
__global__ void add(int bw, int N, int imgH, int imgW, int i, bool isLowerTriangular, T *d_I, T *d_O)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (isLowerTriangular)
    {
        int l = i + 1;
        int elemsPerImg = ((imgH * imgH) / (1ULL << (l + 1)) + imgH / 2);
        int oLen = N * elemsPerImg;
        if (j < oLen)
        {
            int n = j / elemsPerImg;
            int k = j % elemsPerImg;
            float twoPowl = float(1ULL << l);
            float fourPowl = float(1ULL << (2 * l));
            int twoPowlInt = (1ULL << l);
            int h1 = int(floor((-twoPowl + __fsqrt_rd(fourPowl + 8 * twoPowl * k)) / (2.0f * twoPowl)));
            int elemsInRow = h1 + 1;
            int h2 = (k - ((h1 * (h1 + 1)) / 2) * twoPowlInt) / elemsInRow;
            int h = h1 * twoPowlInt + h2;
            int w = k - ((h1 * (h1 + 1)) / 2) * twoPowlInt - h2 * elemsInRow;
            assert(w < elemsInRow);

            int inW = 2 * w;
            float twoPowi = float(1ULL << i);
            int twoPowiInt = (1ULL << i);
            h1 = int(floor(h / twoPowi));
            h2 = ((h1 * (h1 + 1)) / 2) * twoPowiInt;
            elemsInRow = int(ceil((h + 1) / twoPowi));
            int h3 = (h % twoPowiInt) * elemsInRow;
            elemsPerImg = ((imgH * imgH) / (1ULL << (i + 1)) + imgH / 2);
            d_O[j] += d_I[n * elemsPerImg + h2 + h3 + inW];
            gpuMod(d_O[j], bw);
            // if(h == 6)
            // printf("H=%d, %d, %ld, %ld\n", h, j, d_O[j], d_I[n * elemsPerImg + h2 + h3 + inW]);
        }
    }
    else
    {
        int oLen = N * imgH * (imgW / (1ULL << (i + 1)));
        if (j < oLen)
        {
            int curW = (imgW / (1ULL << (i + 1)));
            // now you know you have an element to fill (potentially)
            int m = j / curW;
            int n = j % curW;
            int nIn = 2 * n;
            d_O[m * curW + n] += d_I[m * (imgW / (1ULL << i)) + nIn];
            gpuMod(d_O[j], bw);
            // printf("New max %d=%ld\n", i, d_O[oLen + n]);
        }
    }
}

// M*N matrix
template <typename T>
T *keygenMaxpoolLogHelper(u8 **key_as_bytes, int party, MaxpoolParams p, int i, T *d_I, AESGlobalContext *gaes)
{
    // p.N is the batch size and N is the number of elems to compare in this round
    int oLen;
    if (p.isLowerTriangular)
    {
        int l = i + 1;
        int elemsPerImg = ((p.imgH * p.imgH) / (1ULL << (l + 1)) + p.imgH / 2);
        oLen = p.N * elemsPerImg;
    }
    else
    {
        oLen = p.N * p.imgH * (p.imgW / (1ULL << (i + 1)));
    }
    T *d_diff = (T *)gpuMalloc(oLen * sizeof(T));
    sub<<<(oLen - 1) / 128 + 1, 128>>>(p.bw, p.N, p.imgH, p.imgW, i, p.isLowerTriangular, d_I, d_diff);
    // assert(p.bin + 1 <= p.bw);
    auto d_mask_relu = gpuGenReluKey<T, T, 0, 0, false>(key_as_bytes, party, p.bin, p.bw, oLen, d_diff, gaes);
    add<<<(oLen - 1) / 128 + 1, 128>>>(p.bw, p.N, p.imgH, p.imgW, i, p.isLowerTriangular, d_I, d_mask_relu);
    return d_mask_relu;
}

template <typename T>
T *gpuKeygenMaxpoolLog(uint8_t **key_as_bytes, int party, MaxpoolParams p, T *d_inputMask, AESGlobalContext *gaes)
{
    assert(/*p.N == 1 &&*/ p.C == 1 && p.strideH == 1 && p.strideW == p.FW && p.strideH == p.FH);
    T *d_I = d_inputMask;
    T *d_O;
    // num elements to compare in round r
    int r = p.FH * p.FW;
    // number of rounds
    int R = int(ceil(log2(r)));
    writeInt(key_as_bytes, R);
    for (int i = 0; i < R; i++)
    {
        // compare r consecutive elements
        d_O = keygenMaxpoolLogHelper(key_as_bytes, party, p, i, d_I, gaes);
        if (i > 0)
            gpuFree(d_I);
        d_I = d_O;
        // halve the number of elements to compare
        // r = int(ceil(r / 2.0f));
    }
    return d_O;
}

// M*N matrix
template <typename T>
T *maxpoolLogHelper(SigmaPeer *peer, int party, MaxpoolParams p, int i, GPUReluKey<T> k, T *d_I, AESGlobalContext *gaes, Stats *s)
{
    int oLen;
    if (p.isLowerTriangular)
    {
        int l = i + 1;
        int elemsPerImg = ((p.imgH * p.imgH) / (1ULL << (l + 1)) + p.imgH / 2);
        oLen = p.N * elemsPerImg;
    }
    else
    {
        oLen = p.N * p.imgH * (p.imgW / (1ULL << (i + 1)));
    }
    T *d_diff = (T *)gpuMalloc(oLen * sizeof(T));
    sub<<<(oLen - 1) / 128 + 1, 128>>>(p.bw, p.N, p.imgH, p.imgW, i, p.isLowerTriangular, d_I, d_diff);
    auto d_relu = gpuRelu<T, T, 0, 0, false>(peer, party, k, d_diff, gaes, s);
    add<<<(oLen - 1) / 128 + 1, 128>>>(p.bw, p.N, p.imgH, p.imgW, i, p.isLowerTriangular, d_I, d_relu);
    return d_relu;
}

template <typename T>
T *gpuMaxpoolLog(SigmaPeer *peer, int party, MaxpoolParams p, GPUMaxpoolKey<T> k, T *d_I, AESGlobalContext *gaes, Stats *s)
{
    assert(/*p.N == 1 &&*/ p.C == 1 && p.strideH == 1 && p.strideW == p.FW && p.strideH == p.FH);
    // T *d_I = d_in;
    T *d_O;
    // num elements to compare in round r
    int r = p.FH * p.FW;
    // number of rounds
    int R = int(ceil(log2(r)));
    for (int i = 0; i < R; i++)
    {
        // printf("Round=%d, num Relus=%d\n", i, k.reluKey[i].numRelus);
        // compare r consecutive elements
        d_O = maxpoolLogHelper(peer, party, p, i, k.reluKey[i], d_I, gaes, s);
        if (i > 0)
            gpuFree(d_I);
        d_I = d_O;
        // halve the number of elements to compare
        // r = int(ceil(r / 2.0f));
    }
    return d_O;
}

template <typename T>
T *gpuKeygenMaxpool(uint8_t **key_as_bytes, int party, MaxpoolParams p, T *d_inputMask, AESGlobalContext *gaes, bool logRounds = false)
{
    T *d_mask_O;
    if (logRounds)
    {
        assert(p.zPadHLeft == 0 && p.zPadHRight == 0 && p.zPadWLeft == 0 && p.zPadWRight == 0);
        d_mask_O = gpuKeygenMaxpoolLog(key_as_bytes, party, p, d_inputMask, gaes);
    }
    else
    {
        d_mask_O = gpuKeygenMaxpoolLin(key_as_bytes, party, p, d_inputMask, gaes);
    }
    return d_mask_O;
}

template <typename T>
T *gpuMaxpool(SigmaPeer *peer, int party, MaxpoolParams p, GPUMaxpoolKey<T> k, T *d_I, AESGlobalContext *gaes, Stats *s)
{
    T *d_O;
    if (k.rounds < p.FH * p.FW - 1)
    {
        assert(p.zPadHLeft == 0 && p.zPadHRight == 0 && p.zPadWLeft == 0 && p.zPadWRight == 0);
        d_O = gpuMaxpoolLog(peer, party, p, k, d_I, gaes, s);
    }
    else
    {
        d_O = gpuMaxpoolLin(peer, party, p, k, d_I, gaes, s);
    }
    return d_O;
}

template <typename T>
__global__ void selectForMaxpoolBackpropKernel(MaxpoolParams p, uint32_t *oneHot,
                                               T *incomingGrad,
                                               T *out,
                                               T *a, T *b,
                                               T *e, T *d1,
                                               T *d2, int party, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        int laneId = threadIdx.x & 0x1f;
        int t = i;
        int n = t / (p.H * p.W * p.C * p.FH * p.FW);
        t = t % (p.H * p.W * p.C * p.FH * p.FW);
        int h = t / (p.W * p.C * p.FH * p.FW);
        t = t % (p.W * p.C * p.FH * p.FW);
        int w = t / (p.C * p.FH * p.FW);
        t = t % (p.C * p.FH * p.FW);
        int c = t / (p.FH * p.FW);

        T x = (oneHot[i / 32] >> laneId) & T(1);
        T is_zero_x = (x == 0);
        int j = n * p.H * p.W * p.C + h * p.W * p.C + w * p.C + c;
        T y = incomingGrad[j];
        out[i] = (-a[i] * y - b[i] * x + e[i] + y * is_zero_x * d1[i] + is_zero_x * d2[i] + (party == SERVER1) * (x * y));
        gpuMod(out[i], p.bwBackprop);
    }
}

__global__ void andForMaxpoolKernel(MaxpoolParams p, int pos, uint32_t *dreluBits, uint32_t *oneHotBits,
                                    uint32_t *b0Bits, uint32_t *b1Bits, uint32_t *b2Bits, int party, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < N);
    if (i < N)
    {
        int laneId = threadIdx.x & 0x1f;
        int t = i;
        int n = t / (p.H * p.W * p.C * p.FH * p.FW);
        t = t % (p.H * p.W * p.C * p.FH * p.FW);
        int h = t / (p.W * p.C * p.FH * p.FW);
        t = t % (p.W * p.C * p.FH * p.FW);
        int w = t / (p.C * p.FH * p.FW);
        t = t % (p.C * p.FH * p.FW);
        int c = t / (p.FH * p.FW);
        int q = t % (p.FH * p.FW);
        int newOneHot = 0;
        int idx = i / 32;
        if (q < pos)
        {
            // need to check this once
            int dreluIndex = n * p.H * p.W * p.C + h * p.W * p.C + w * p.C + c;
            // printf("%d: %d %d\n", i, dreluIndex / 32, idx);
            uint32_t drelu = (dreluBits[dreluIndex / 32] >> (dreluIndex % 32)) & 1;
            // int idx = (dreluIndex * p.FH * p.FW + q) / 32;
            uint32_t oneHot; // = 0;
            if (pos == 2 && q == 0)
                oneHot = 1;
            else if (pos == 2)
                oneHot = 0;
            else
                oneHot = (oneHotBits[idx] >> laneId) & 1;
            uint32_t incomingOneHot = (q == pos - 1 ? 1 : 0);
            uint32_t diff = (incomingOneHot - oneHot) & 1;
            int keyNum = dreluIndex * pos + q;
            int keyIdx = keyNum / 32;
            int keyPos = keyNum % 32;
            uint32_t b0 = (b0Bits[keyIdx] >> keyPos) & 1;
            uint32_t b1 = (b1Bits[keyIdx] >> keyPos) & 1;
            uint32_t b2 = (b2Bits[keyIdx] >> keyPos) & 1;
            // printf("drelu: %d diff: %d b0: %d b1: %d b2: %d curOneHot: %d %u %u %u\n", drelu, diff, b0, b1, b2, oneHot, b0Bits[idx], b1Bits[idx], b2Bits[idx]);
            newOneHot = (-b0 * diff - drelu * b1 + b2 + (party == SERVER1) * (drelu * diff + oneHot)) & 1;
            // printf("%d %d %d %d %d %d %d: %d %d %d %d %d %d\n", n, h, w, c, q, pos, dreluIndex, drelu, oneHot, incomingOneHot, newOneHot, keyIdx, laneId);
        }
        // printf("%d: %d\n", i, newOneHot);
        assert(newOneHot == 0 || newOneHot == 1);
        newOneHot <<= laneId;
        for (int j = 16; j >= 1; j /= 2)
            newOneHot += __shfl_down_sync(mask, newOneHot, j, 32);
        if (laneId == 0)
        {
            // printf("new one hot: %d %d\n", idx, newOneHot);
            oneHotBits[idx] = static_cast<uint32_t>(newOneHot);
        }
        // update uint32_t in tandem
    }
}

template <typename T>
__global__ void gpuCollectGradientsKernel(MaxpoolParams p, T *outgoingGradExpanded, T *outgoingGrad, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        int t = i;
        int n = t / (p.imgH * p.imgW * p.C);
        t = t % (p.imgH * p.imgW * p.C);
        int h = t / (p.imgW * p.C);
        t = t % (p.imgW * p.C);
        int w = t / (p.C);
        int c = t % (p.C);
        T sumGrads = 0;
        for (int fh = 0; fh < p.FH; fh++)
        {
            for (int fw = 0; fw < p.FW; fw++)
            {
                int leftTopCornerH = h - fh;
                int leftTopCornerW = w - fw;
                int rightTopCornerH = leftTopCornerH;
                int rightTopCornerW = leftTopCornerW + p.FW - 1;
                int leftBottomCornerH = leftTopCornerH + p.FH - 1;
                int leftBottomCornerW = leftTopCornerW;
                int rightBottomCornerH = leftTopCornerH + p.FH - 1;
                int rightBottomCornerW = leftTopCornerW + p.FW - 1;
                if (leftTopCornerH >= 0 && leftTopCornerW >= 0 &&
                    rightTopCornerH >= 0 && rightTopCornerW < p.imgW &&
                    leftBottomCornerH < p.imgH && leftBottomCornerW >= 0 &&
                    rightBottomCornerH < p.imgH && rightBottomCornerW < p.imgW &&
                    leftTopCornerH % p.strideH == 0 && leftTopCornerW % p.strideW == 0)
                {
                    int gradH = leftTopCornerH / p.strideH;
                    int gradW = leftTopCornerW / p.strideW;
                    int idx = n * p.H * p.W * p.C * p.FH * p.FW + gradH * p.W * p.C * p.FH * p.FW + gradW * p.C * p.FH * p.FW + c * p.FH * p.FW + fh * p.FW + fw;
                    // if(i == 0) printf("gpu: %lu %d\n", outgoingGradExpanded[idx], p.bwBackprop);
                    sumGrads += outgoingGradExpanded[idx];
                }
            }
        }
        outgoingGrad[i] = sumGrads;
        gpuMod(outgoingGrad[i], p.bwBackprop);
    }
}

// no memory leak
template <typename T>
T *gpuSelectForMaxpoolBackprop(MaxpoolParams p, GPUSelectKey<T> k,
                               uint32_t *d_oneHot,
                               T *d_incomingGrad,
                               int party, Stats *stats)
{
    size_t size_in_bytes = k.N * sizeof(T);

    T *d_out = (T *)gpuMalloc(size_in_bytes);
    T *d_a, *d_b, *d_c, *d_d1, *d_d2;
    d_a = (T *)moveToGPU((uint8_t *)k.a, 5 * size_in_bytes, stats);
    d_b = d_a + k.N;
    d_c = d_b + k.N;
    d_d1 = d_c + k.N;
    d_d2 = d_d1 + k.N;

    const int tb_size = 256;

    selectForMaxpoolBackpropKernel<T><<<(k.N - 1) / tb_size + 1, tb_size>>>(p, d_oneHot,
                                                                            d_incomingGrad, d_out, d_a, d_b, d_c, d_d1, d_d2, party, k.N);
    checkCudaErrors(cudaDeviceSynchronize());

    gpuFree(d_a);
    return d_out;
}

// no memory leak
void gpuAndForMaxpool(MaxpoolParams p, int pos, GPUAndKey k,
                      uint32_t *d_drelu, /* uint32_t *d_drelu2,*/
                      uint32_t *d_oneHot,
                      int party, Stats *stats)
{
    // printf("selectKey.N: %d\n", k.N);
    int num_ints = (k.N - 1) / PACKING_SIZE + 1;
    size_t size_in_bytes = num_ints * sizeof(uint32_t);

    uint32_t *d_b0, *d_b1, *d_b2;
    d_b0 = (uint32_t *)moveToGPU((uint8_t *)k.b0, 3 * size_in_bytes, stats);
    d_b1 = d_b0 + num_ints;
    d_b2 = d_b1 + num_ints;

    const int tb_size = 256;
    // printf("N for maxpool and: %d %d %d\n", k.N, num_ints, size_in_bytes);
    int numElems = p.N * p.H * p.W * p.C * p.FH * p.FW;
    andForMaxpoolKernel<<<(numElems - 1) / tb_size + 1, tb_size>>>(p, pos, d_drelu, d_oneHot,
                                                                   d_b0, d_b1, d_b2, party, numElems);
    checkCudaErrors(cudaDeviceSynchronize());

    gpuFree(d_b0);
}

template <typename T>
T *gpuCollectGradients(MaxpoolParams p, T *d_outgoingGradExpanded, Stats *s)
{
    size_t outgoingGradSize = p.N * p.imgH * p.imgW * p.C;
    size_t outgoingGradMemSize = outgoingGradSize * sizeof(T);
    T *d_outgoingGrad = (T *)gpuMalloc(outgoingGradMemSize);
    const int tbSize = 256;
    assert(p.zPadHLeft == 0 && p.zPadHRight == 0 && p.zPadWLeft == 0 && p.zPadWRight == 0);
    gpuCollectGradientsKernel<<<(outgoingGradSize - 1) / tbSize + 1, tbSize>>>(p, d_outgoingGradExpanded, d_outgoingGrad, outgoingGradSize);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    return d_outgoingGrad;
}

template <typename T>
__global__ void expandKernel(MaxpoolParams p,
                             T *incomingGradMask,
                             T *expandedIncomingGradMask, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        // int laneId = threadIdx.x & 0x1f;
        int t = i;
        int n = t / (p.H * p.W * p.C * p.FH * p.FW);
        t = t % (p.H * p.W * p.C * p.FH * p.FW);
        int h = t / (p.W * p.C * p.FH * p.FW);
        t = t % (p.W * p.C * p.FH * p.FW);
        int w = t / (p.C * p.FH * p.FW);
        t = t % (p.C * p.FH * p.FW);
        int c = t / (p.FH * p.FW);

        int j = n * p.H * p.W * p.C + h * p.W * p.C + w * p.C + c;
        expandedIncomingGradMask[i] = incomingGradMask[j];
    }
}

template <typename T>
T *keyGenMaxpoolBackProp(uint8_t **key_as_bytes, int party, MaxpoolParams p, u8 *d_oneHotMask, T *d_incomingGradMask)
{
    int outSz = p.N * p.H * p.W * p.C * p.FH * p.FW;
    auto d_expandedGradMask = (T *)gpuMalloc(outSz * sizeof(T));
    expandKernel<<<(outSz - 1) / 256 + 1, 256>>>(p, d_incomingGradMask, d_expandedGradMask, outSz);
    auto d_randomMaskOut = gpuKeyGenSelect<T, T, u8>(key_as_bytes, party, outSz, d_expandedGradMask, d_oneHotMask, p.bwBackprop);
    gpuFree(d_expandedGradMask);
    auto d_outgoingGradMask = gpuCollectGradients(p, d_randomMaskOut, NULL);
    gpuFree(d_randomMaskOut);
    return d_outgoingGradMask;
}

// need to check this
template <typename T>
__global__ void keygenAndForMaxpoolKernel(MaxpoolParams p, int N, int oneHotLen, T *dreluMask, T *oneHotMask,
                                          T *b0, T *b1, /*T* b2,*/ T *randomOutMask)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        int t = i;
        int n = t / (p.H * p.W * p.C * oneHotLen);
        t = t % (p.H * p.W * p.C * oneHotLen);
        int h = t / (p.W * p.C * oneHotLen);
        t = t % (p.W * p.C * oneHotLen);
        int w = t / (p.C * oneHotLen);
        t = t % (p.C * oneHotLen);
        int c = t / (oneHotLen);
        int q = t % (oneHotLen);
        // if(q < pos) {
        // need to check this once
        int dreluIndex = n * p.H * p.W * p.C + h * p.W * p.C + w * p.C + c;
        int oneHotIdx = dreluIndex * p.FH * p.FW + q;
        b0[i] = dreluMask[dreluIndex];
        b1[i] = oneHotMask[oneHotIdx];
        if (q == oneHotLen - 1)
            assert(b1[i] == 0);
        // check if this is okay
        oneHotMask[oneHotIdx] = randomOutMask[i];
        // printf("inside keygen and kernel %d: %lu %lu %lu %llu\n", i, b0[i], b1[i], randomOutMask[i], (randomOutMask[i] - b1[i]) & 1ULL);
        randomOutMask[i] = (b0[i] * b1[i] + randomOutMask[i] - b1[i]) & 1ULL;
    }
}

void gpuKeygenOneHotMaxpool(u8 **key_as_bytes, int party, MaxpoolParams p, int outSz, int fh, int fw, u8 *d_dreluMask, u8 *d_oneHotMask)
{
    int oneHotLen = fh * p.FW + fw + 1;
    int numAnds = outSz * oneHotLen;
    auto d_b0 = gpuMalloc(numAnds);
    auto d_b1 = gpuMalloc(numAnds);
    auto d_andOutMask = randomGEOnGpu<u8>(numAnds, 1);
    // this function copies the latest and out mask to the onehot mask
    keygenAndForMaxpoolKernel<<<(numAnds - 1) / 256 + 1, 256>>>(p, numAnds, oneHotLen, d_dreluMask, d_oneHotMask,
                                                                d_b0, d_b1, d_andOutMask);
    writeAndKey(key_as_bytes, party, numAnds, d_b0, d_b1, d_andOutMask, 1);
    gpuFree(d_b0);
    gpuFree(d_b1);
    gpuFree(d_andOutMask);
}