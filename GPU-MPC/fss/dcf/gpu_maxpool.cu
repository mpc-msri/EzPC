#include "utils/gpu_random.h"
#include "gpu_dcf.h"
#include "gpu_maxpool.h"

namespace dcf
{
    template <typename T>
    T *gpuMaxpoolHelper(SigmaPeer *peer, int party, MaxpoolParams p, GPU2RoundReLUKey<T> k, GPUAndKey andKey, int i, int j, T *d_I, T *d_curMax, u32 *d_oneHot, AESGlobalContext *gaes, Stats *s)
    {
        int outSz = getMSz(p);
        T *d_diff = (T *)gpuMalloc(outSz * sizeof(T));
        diffWithCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, i, j, d_curMax, d_I, d_diff, outSz);
        checkCudaErrors(cudaDeviceSynchronize());
        auto d_res = gpuTwoRoundRelu(peer, party, k, d_diff, gaes, s);
        auto d_drelu = d_res.first;
        auto d_newMax = d_res.second;
        gpuFree(d_diff);
        // relu(x-y) + y
        gpuLinearComb(p.bw, outSz, d_newMax, T(1), d_newMax, T(1), d_curMax);
        if (d_oneHot)
        {
            gpuAndForMaxpool(p, i * p.FW + j + 1, andKey, d_drelu, d_oneHot, party, s);
            int numBits = k.selectKey.N * p.FH * p.FW;
            peer->reconstructInPlace((T *)d_oneHot, 1, numBits, s);
        }
        gpuFree(d_drelu);
        return d_newMax;
    }

    template <typename T>
    T *gpuMaxPool(SigmaPeer *peer, int party, MaxpoolParams p, GPUMaxpoolKey<T> k, T *d_I, u32 *d_oneHot,
                  AESGlobalContext *gaes, Stats *s)

    {
        int outSz = getMSz(p);
        T *d_curMax = (T *)gpuMalloc(outSz * sizeof(T));
        populateCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, d_curMax, d_I, outSz);
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < p.FH; i++)
        {
            for (int j = 0; j < p.FW; j++)
            {
                if (i == 0 && j == 0)
                    continue;
                auto d_newMax = gpuMaxpoolHelper(peer, party, p, k.reluKey[i * p.FW + j], k.andKey[i * p.FW + j], i, j, d_I, d_curMax, d_oneHot, gaes, s);
                gpuFree(d_curMax);
                d_curMax = d_newMax;
            }
        }
        return d_curMax;
    }

    template <typename T>
    T *gpuKeygenMaxpoolHelper(uint8_t **key_as_bytes, int party, MaxpoolParams p, int fh, int fw,
                              T *d_inputMask, T *d_curMaxMask, u8 *d_oneHotMask,
                              AESGlobalContext *gaes)
    {
        int outSz = getMSz(p);
        T *d_diffMask = (T *)gpuMalloc(outSz * sizeof(T));
        // d_diffMask = inputMask - curMask
        diffWithCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, fh, fw, d_curMaxMask, d_inputMask, d_diffMask, outSz);
        checkCudaErrors(cudaDeviceSynchronize());
        auto d_res = gpuGenTwoRoundReluKey(key_as_bytes, party, p.bin, p.bw, outSz, d_diffMask, gaes);
        auto d_dreluMask = d_res.first;
        auto d_newMaxMask = d_res.second;
        gpuFree(d_diffMask);
        gpuLinearComb(p.bw, outSz, d_newMaxMask, T(1), d_newMaxMask, T(1), d_curMaxMask);
        if (d_oneHotMask)
        {
            gpuKeygenOneHotMaxpool(key_as_bytes, party, p, outSz, fh, fw, d_dreluMask, d_oneHotMask);
        }
        gpuFree(d_dreluMask);
        return d_newMaxMask;
    }

    template <typename T>
    T *gpuKeygenMaxpool(uint8_t **key_as_bytes, int party, MaxpoolParams p,
                        T *d_inputMask, u8 *d_oneHotMask,
                        AESGlobalContext *gaes)
    {
        int outSz = getMSz(p);
        T *d_curMaxMask = (T *)gpuMalloc(outSz * sizeof(T));
        populateCurMax<<<(outSz - 1) / 256 + 1, 256>>>(p, d_curMaxMask, d_inputMask, outSz);
        checkCudaErrors(cudaDeviceSynchronize());
        for (int i = 0; i < p.FH; i++)
        {
            for (int j = 0; j < p.FW; j++)
            {
                if (i == 0 && j == 0)
                    continue;
                auto d_newMax = gpuKeygenMaxpoolHelper(key_as_bytes, party, p, i, j, d_inputMask, d_curMaxMask, d_oneHotMask, gaes);
                gpuFree(d_curMaxMask);
                d_curMaxMask = d_newMax;
            }
        }
        return d_curMaxMask;
    }
}