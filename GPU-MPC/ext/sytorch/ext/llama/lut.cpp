#include "lut.h"
#include <llama/dpf.h>
#include <llama/assert.h>
#include <cassert>

std::pair<LUTKeyPack, LUTKeyPack> keyGenLUT(int bin, int bout, GroupElement rin, GroupElement rout)
{
    LUTKeyPack key0, key1;
    key0.bin = bin;
    key1.bin = bin;
    key0.bout = bout;
    key1.bout = bout;
    auto dpfKeys = keyGenDPF(bin, bout, -rin, 1);
    key0.dpfKey = dpfKeys.first;
    key1.dpfKey = dpfKeys.second;
    auto routSplit = splitShare(rout, bout);
    key0.rout = routSplit.first;
    key1.rout = routSplit.second;
    return std::make_pair(key0, key1);
}

inline osuCrypto::block random_block()
{
    osuCrypto::block a;
    int tid = omp_get_thread_num();
    a = LlamaConfig::prngs[tid].get<osuCrypto::block>();
    return a;
}

std::pair<LUTSSKeyPack, LUTSSKeyPack> keyGenLUTSS(int bin, int bout, GroupElement rin, GroupElement rout)
{
    assert(bin == 8);
    LUTSSKeyPack key0, key1;
    key0.bin = bin;
    key1.bin = bin;
    key0.bout = bout;
    key1.bout = bout;

    uint64_t b0, b1, b2, b3;

    rin = -rin;
    mod(rin, bin);
    if (rin < 64) 
    {
        b0 = 1ULL << (63-rin);
        b1 = 0;
        b2 = 0;
        b3 = 0;
    }
    else if (rin < 128)
    {
        b0 = 0;
        b1 = 1ULL << (127-rin);
        b2 = 0;
        b3 = 0;
    }
    else if (rin < 192)
    {
        b0 = 0;
        b1 = 0;
        b2 = 1ULL << (191-rin);
        b3 = 0;
    }
    else
    {
        b0 = 0;
        b1 = 0;
        b2 = 0;
        b3 = 1ULL << (255-rin);
    }

    key0.b0 = random_ge(64);
    key0.b1 = random_ge(64);
    key0.b2 = random_ge(64);
    key0.b3 = random_ge(64);

    key1.b0 = b0 ^ key0.b0;
    key1.b1 = b1 ^ key0.b1;
    key1.b2 = b2 ^ key0.b2;
    key1.b3 = b3 ^ key0.b3;

    GroupElement routRes = random_ge(bout);
    auto routRes_split = splitShare(routRes, bout);
    key0.routRes = routRes_split.first;
    key1.routRes = routRes_split.second;

    GroupElement routCorr = random_ge(bout);
    auto routCorr_split = splitShare(routCorr, bout);
    key0.routCorr = routCorr_split.first;
    key1.routCorr = routCorr_split.second;

    rout = rout + routRes * routCorr;
    auto rout_split = splitShare(rout, bout);
    key0.rout = rout_split.first;
    key1.rout = rout_split.second;

    return std::make_pair(key0, key1);
}

std::pair<GroupElement, GroupElement> evalLUTSS_1(int party, GroupElement x, const std::vector<GroupElement> &tab, const LUTSSKeyPack &kp)
{
    int bin = kp.bin;
    assert(bin == 8);
    mod(x, bin);

    uint64_t bL0 = kp.b0;
    uint64_t bL1 = kp.b1;
    uint64_t bR0 = kp.b2;
    uint64_t bR1 = kp.b3;

    GroupElement res = 0, corr = 0;
    for (int i = 0; i < 64; i++)
    {
        GroupElement bit = (bL0 >> (63-i)) & 1;
        res += (1 - 2 * party) * bit * tab[(i+x)%256];
        corr += (1 - 2 * party) * bit;
    }

    for (int i = 0; i < 64; i++)
    {
        GroupElement bit = (bL1 >> (63-i)) & 1;
        res += (1 - 2 * party) * bit * tab[(i+64+x)%256];
        corr += (1 - 2 * party) * bit;
    }

    for (int i = 0; i < 64; i++)
    {
        GroupElement bit = (bR0 >> (63-i)) & 1;
        res += (1 - 2 * party) * bit * tab[(i+128+x)%256];
        corr += (1 - 2 * party) * bit;
    }

    for (int i = 0; i < 64; i++)
    {
        GroupElement bit = (bR1 >> (63-i)) & 1;
        res += (1 - 2 * party) * bit * tab[(i+192+x)%256];
        corr += (1 - 2 * party) * bit;
    }

    res = res + kp.routRes;
    corr = corr + kp.routCorr;

    return std::make_pair(res, corr);
}

GroupElement evalLUTSS_2(int party, GroupElement res, GroupElement corr, const LUTSSKeyPack &kp)
{
    return party * res * corr - res * kp.routCorr - corr * kp.routRes + kp.rout;
}

std::pair<LUTDPFETKeyPack, LUTDPFETKeyPack> keyGenLUTDPFET(int bin, int bout, GroupElement rin, GroupElement routRes, GroupElement routCorr)
{
    assert(bin == 8);
    LUTDPFETKeyPack key0, key1;
    key0.bin = bin;
    key1.bin = bin;
    key0.bout = bout;
    key1.bout = bout;

    auto dpfKeys = keyGenDPFET(bin, -rin);
    key0.dpfKey = dpfKeys.first;
    key1.dpfKey = dpfKeys.second;

    auto routRes_split = splitShare(routRes, bout);
    key0.routRes = routRes_split.first;
    key1.routRes = routRes_split.second;

    auto routCorr_split = splitShare(routCorr, 1);
    key0.routCorr = routCorr_split.first;
    key1.routCorr = routCorr_split.second;

    return std::make_pair(key0, key1);
}

std::pair<GroupElement, GroupElement> evalLUTDPFET_1(int party, GroupElement x, const std::vector<GroupElement> &tab, LUTDPFETKeyPack &kp)
{
    int bin = kp.bin;
    assert(bin == 8);
    mod(x, bin);

    GroupElement res = 0, corr = 0;
    auto res_corr = evalAll_reduce_et(party, kp.dpfKey, x, tab);

    res = res_corr.first + kp.routRes;
    corr = res_corr.second;

    // corr = -1 or 1
    if (party == 0) {
        corr = corr + 1; // corr = 0 or 2
    }
    mod(corr, 2);
    if (party == 0) 
    {
        corr = corr / 2;
    }
    else
    {
        corr = 2 - ((4 - corr) / 2);
    }
    corr = corr + kp.routCorr;
    mod(corr, 1);

    return std::make_pair(res, corr);
}
