#pragma once

#include "utils/gpu_data_types.h"
#include "utils/misc_utils.h"
#include "utils/gpu_stats.h"
#include "utils/gpu_mem.h"

#include "fss/gpu_fss_helper.h"

#include <vector>

// using namespace std;
namespace dcf
{

    typedef void (*dcfPrologue)(int party, int bin, int N,
                                u64 x,
                                u64 *o);
    typedef void (*dcfEpilogue)(int party, int bin, int bout, int N,
                                u64 x,
                                u64 *o_l, u32 *out_g, u64 oStride);

    __device__ void idPrologue(int party, int bin, int N,
                               u64 x,
                               u64 *o)
    {
        o[0] = x;
    }

    __device__ void idEpilogue(int party, int bin, int bout, int N,
                               u64 x,
                               u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = u64(*o_l);
        writePackedOp(out_g, o1, bout, N);
    }

    __device__ void maskEpilogue(int party, int bin, int bout, int N,
                                 u64 x,
                                 u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = u64(*o_l);
        auto mask = getVCW(bout, out_g, N, 0);
        // printf("Mask: %ld, output: %ld\n", mask, o);
        o1 = o1 + mask;
        gpuMod(o1, bout);
        writePackedOp(out_g, o1, bout, N);
    }

    __device__ void dReluPrologue(int party, int bin, int N,
                                  u64 x,
                                  u64 *o)
    {
        o[0] = x;
        o[1] = (x + (1ULL << (bin - 1)));
    }

    template <bool returnXLtRin>
    __device__ void dReluEpilogue(int party, int bin, int bout, int N,
                                  u64 x,
                                  u64 *o_l, u32 *out_g, u64 oStride)
    {
        auto o1 = o_l[0];
        auto o2 = o_l[1];
        auto mask = getVCW(bout, out_g, N, 0);
        auto o = o2 - o1 + mask;
        // printf("o1=%lu, o2=%lu, mask=%lu, o=%lu\n", o1, o2, mask, o);
        if (party == SERVER1)
        {
            auto x2 = (x + (1ULL << (bin - 1)));
            gpuMod(x2, bin);
            // printf("x=%lu, x2=%lu, %lu\n", x, x2, (1ULL << (bin - 1)));
            o += (x2 >= (1ULL << (bin - 1)));
            // printf("o=%ld, %d, %d, %d\n", o, (x2 >= (1ULL << (bin - 1))), bin, bout);
        }
        gpuMod(o, bout);
        writePackedOp(out_g, o, bout, N);
        // writeVCW(bout, out_g, o, 0, N);
        if (returnXLtRin)
        {
            o1 += getVCW(bout, out_g + oStride, N, 0);
            gpuMod(o1, bout);
            writePackedOp(out_g + oStride, o1, bout, N);
            // writeVCW(bout, out_g, o1, 1, N);
        }
    }

}