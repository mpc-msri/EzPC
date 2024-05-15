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
