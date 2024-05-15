#pragma once

#include "gpu_avgpool.h"
#include "gpu_relu.h"
#include "gpu_mul.h"

using MaxpoolParams = AvgPoolParams;

inline int getInSz(MaxpoolParams p)
{
    int sz;
    if (p.isLowerTriangular)
    {
        assert(p.imgH == p.imgW);
        assert(p.C == 1);
        sz = p.N * (p.imgH * (p.imgH + 1)) / 2;
    }
    else
    {
        sz = p.N * p.imgH * p.imgW * p.C;
    }
    return sz;
}

template <typename T>
struct GPUMaxpoolKey
{
    int rounds;
    GPUReluKey<T> *reluKey;
    // GPUAndKey* andKey;
};

template <typename T>
GPUMaxpoolKey<T> readGPUMaxpoolKey(MaxpoolParams p, u8 **key_as_bytes)
{
    GPUMaxpoolKey<T> k;
    k.rounds = *((int *)*key_as_bytes);
    printf("Rounds=%d\n", k.rounds);
    *key_as_bytes += sizeof(int);
    k.reluKey = new GPUReluKey<T>[/*p.FH * p.FW*/ k.rounds];
    for (int i = 0; i < /*p.FH*/ k.rounds; i++)
    {
        // for (int j = 0; j < p.FW; j++)
        // {
        // if (i == 0 && j == 0)
        // continue;
        // printf("Reading Relu key=%d, %d\n", i, j);
        k.reluKey[i] = readReluKey<T>(key_as_bytes);
        printf("Round %d=%d relus\n", i, k.reluKey[i].numRelus);
        // if(this->train) maxpoolKey.andKey[i * p.FW + j] = readGPUAndKey(key_as_bytes);
        // }
    }
    return k;
}

#include "gpu_maxpool.cu"