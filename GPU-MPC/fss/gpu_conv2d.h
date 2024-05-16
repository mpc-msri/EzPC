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

struct Conv2DParams
{
    int bin, bout, N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight,
        zPadWLeft, zPadWRight,
        strideH, strideW, OH, OW;
    size_t size_I, size_F, size_O;
};

template <typename T>
struct GPUConv2DKey
{
    Conv2DParams p;
    size_t mem_size_I, mem_size_F, mem_size_O;
    T *I, *F, *O;
};

void fillConv2DParams(Conv2DParams *p)
{
    p->OH = ((p->H - p->FH + (p->zPadHLeft + p->zPadHRight)) / p->strideH) + 1;
    p->OW = ((p->W - p->FW + (p->zPadWLeft + p->zPadWRight)) / p->strideW) + 1;
    p->size_I = p->N * p->H * p->W * p->CI;
    p->size_F = p->CO * p->FH * p->FW * p->CI;
    p->size_O = p->N * p->OH * p->OW * p->CO;
}

#include "gpu_conv2d.cu"
