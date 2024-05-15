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