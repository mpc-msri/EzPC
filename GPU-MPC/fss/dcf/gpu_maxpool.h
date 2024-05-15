#pragma once

#include "fss/gpu_maxpool.h"
#include "fss/gpu_and.h"
#include "gpu_relu.h"

namespace dcf
{

    template <typename T>
    struct GPUMaxpoolKey
    {
        GPU2RoundReLUKey<T> *reluKey;
        GPUAndKey *andKey;
    };

    template <typename T>
    GPUMaxpoolKey<T> readGPUMaxpoolKey(MaxpoolParams p, u8 **key_as_bytes)
    {
        GPUMaxpoolKey<T> k;
        int rounds = p.FH * p.FW - 1;
        // printf("Rounds=%d\n", rounds);
        k.reluKey = new GPU2RoundReLUKey<T>[rounds + 1];
        for (int i = 0; i < rounds; i++)
        {
            k.reluKey[i + 1] = readTwoRoundReluKey<T>(key_as_bytes);
            // printf("Round %d=%d relus\n", i + 1, k.reluKey[i + 1].N);
        }
        return k;
    }
}

#include "gpu_maxpool.cu"