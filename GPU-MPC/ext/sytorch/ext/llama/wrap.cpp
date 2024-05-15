#include "wrap.h"
#include <cassert>
#include <llama/dpf.h>

std::pair<WrapSSKeyPack, WrapSSKeyPack> keyGenWrapSS(int bin, GroupElement rin, GroupElement rout)
{
    mod(rin, bin);
    mod(rout, 1);
    assert(bin <= 7);
    uint64_t b0 = 0, b1 = 0;
    WrapSSKeyPack k1, k2;
    k1.bin = bin;
    k2.bin = bin;

    if (bin == 7)
    {
        for (int i = 0; i < 64; ++i)
        {
            uint64_t bit = rout;
            if (i < rin)
            {
                bit ^= 1;
            }
            b0 |= (bit << (63 - i));
        }

        for (int i = 64; i < 128; ++i)
        {
            uint64_t bit = rout;
            if (i < rin)
            {
                bit ^= 1;
            }
            b1 |= (bit << (127 - i));
        }

        auto b0_split = splitShareXor(b0, 64);
        auto b1_split = splitShareXor(b1, 64);
        k1.b0 = b0_split.first;
        k2.b0 = b0_split.second;
        k1.b1 = b1_split.first;
        k2.b1 = b1_split.second;

    }
    else
    {
        for (int i = 0; i < (1LL << bin); ++i)
        {
            uint64_t bit = rout;
            if (i < rin)
            {
                bit ^= 1;
            }
            b0 |= (bit << (63 - i));
        }

        auto b0_split = splitShareXor(b0, 64);
        k1.b0 = b0_split.first;
        k2.b0 = b0_split.second;
    }
    
    return std::make_pair(k1, k2);
}

GroupElement evalWrapSS(int party, GroupElement x, const WrapSSKeyPack &key)
{
    mod(x, key.bin);
    if (key.bin == 7)
    {
        if (x > 63)
        {
            return (key.b1 >> (127 - x)) & 1;
        }
        else
        {
            return (key.b0 >> (63 - x)) & 1;
        }
    }
    else
    {
        return (key.b0 >> (63 - x)) & 1;
    }
}

std::pair<WrapDPFKeyPack, WrapDPFKeyPack> keyGenWrapDPF(int bin, GroupElement rin, GroupElement rout)
{
    mod(rin, bin);
    auto dpfKeys = keyGenDPFET(bin, rin);
    auto r_split = splitShare(rout, 1);

    WrapDPFKeyPack k0, k1;
    k0.bin = bin; k1.bin = bin;
    k0.dpfKey = dpfKeys.first;
    k1.dpfKey = dpfKeys.second;
    k0.r = r_split.first;
    k1.r = r_split.second;
    return std::make_pair(k0, k1);
}

GroupElement evalWrapDPF(int party, GroupElement x, const WrapDPFKeyPack &key)
{
    GroupElement u_b = evalDPFET_LT(party, key.dpfKey, x) ^ key.r;
    return u_b;
}
