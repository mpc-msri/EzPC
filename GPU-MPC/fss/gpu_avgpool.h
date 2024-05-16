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

#pragma once

#include "utils/gpu_data_types.h"

struct AvgPoolParams {
    int bw, bin, scale, scaleDiv, bwBackprop;
    int N, imgH, imgW, C; 
    int FH, FW; 
    int strideH, strideW; 
    int zPadHLeft, zPadHRight; 
    int zPadWLeft, zPadWRight;
    int H, W;
    bool isLowerTriangular = false;
};

inline int getMSz(AvgPoolParams p)
{
    return p.N * p.H * p.W * p.C;
}

inline void initPoolParams(AvgPoolParams &p)
{
    p.H = ((p.imgH - p.FH + (p.zPadHLeft + p.zPadHRight)) / p.strideH) + 1;
    p.W = ((p.imgW - p.FW + (p.zPadWLeft + p.zPadWRight)) / p.strideW) + 1;
    // printf("OH=%d, OW=%d\n", p.H, p.W);
}

template <typename T>
T* gpuAddPool(AvgPoolParams p, T* d_I, Stats* s);


#include "gpu_avgpool.cu"