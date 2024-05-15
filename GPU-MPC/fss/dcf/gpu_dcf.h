#pragma once

#include "utils/gpu_data_types.h"
#include "fss/gpu_sstab.h"
#include <cassert>

namespace dcf
{
    struct GPUDCFTreeKey
    {
        int bin, bout, N;
        AESBlock *scw, *l;
        u32 *vcw;
        u64 memSzScw, memSzVcw, memSzL, memSzOut;
    };

    struct GPUDCFKey
    {
        int bin, bout, M, B;
        u64 memSzOut;
        GPUDCFTreeKey *dcfTreeKey;
        GPUSSTabKey ssKey;
    };

    GPUDCFTreeKey readGPUDCFTreeKey(uint8_t **key_as_bytes)
    {
        GPUDCFTreeKey k;
        std::memcpy((char *)&k, *key_as_bytes, 3 * sizeof(int));
        *key_as_bytes += 3 * sizeof(int);

        int elemsPerBlock = AES_BLOCK_LEN_IN_BITS / k.bout;
        int b2 = k.bin - int(ceil(log2(elemsPerBlock)));

        size_t szScw = k.N * b2;
        k.memSzScw = szScw * sizeof(AESBlock);
        k.memSzL = 2 * k.N * sizeof(AESBlock);
        // assert(k.bout == 1 || k.bout == 2);
        k.memSzVcw = ((k.bout * k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE) * (b2 - 1);
        k.memSzOut = ((k.bout * k.N - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);

        k.scw = (AESBlock *)*key_as_bytes;
        *key_as_bytes += k.memSzScw;
        k.l = (AESBlock *)*key_as_bytes;
        *key_as_bytes += k.memSzL;
        k.vcw = (u32 *)*key_as_bytes;
        *key_as_bytes += k.memSzVcw;
        return k;
    }

    GPUDCFKey readGPUDCFKey(uint8_t **key_as_bytes)
    {
        GPUDCFKey k;
        k.bin = *((int *)*key_as_bytes);
        if (k.bin <= 8)
        {
            k.ssKey = readGPUSSTabKey(key_as_bytes);
            k.bout = 1;
            k.M = k.ssKey.N;
            k.B = 1;
            k.memSzOut = k.ssKey.memSzOut;
        }
        else
        {
            memcpy(&k, *key_as_bytes, 4 * sizeof(int));
            *key_as_bytes += (4 * sizeof(int));

            // assert(k.bin > 8);

            k.dcfTreeKey = new GPUDCFTreeKey[k.B];
            k.memSzOut = 0;
            for (int b = 0; b < k.B; b++)
            {
                k.dcfTreeKey[b] = readGPUDCFTreeKey(key_as_bytes);
                k.memSzOut += k.dcfTreeKey[b].memSzOut;
            }
        }
        return k;
    }
}

#include "gpu_dcf.cu"
