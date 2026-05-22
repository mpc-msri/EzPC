#include "and.h"

// 64 ANDs at a time
std::pair<BitwiseAndKeyPack, BitwiseAndKeyPack> keyGenBitwiseAnd(GroupElement rin1, GroupElement rin2, GroupElement rout)
{
    BitwiseAndKeyPack k0, k1;
    GroupElement t[4];
    GroupElement ones = -1;
    t[0] = ((0 ^ rin1) & (0 ^ rin2)) ^ rout;
    t[1] = ((0 ^ rin1) & (ones ^ rin2)) ^ rout;
    t[2] = ((ones ^ rin1) & (0 ^ rin2)) ^ rout;
    t[3] = ((ones ^ rin1) & (ones ^ rin2)) ^ rout;

    auto t0Pair = splitShareXor(t[0], 64);
    auto t1Pair = splitShareXor(t[1], 64);
    auto t2Pair = splitShareXor(t[2], 64);
    auto t3Pair = splitShareXor(t[3], 64);

    k0.t[0] = t0Pair.first;
    k0.t[1] = t1Pair.first;
    k0.t[2] = t2Pair.first;
    k0.t[3] = t3Pair.first;

    k1.t[0] = t0Pair.second;
    k1.t[1] = t1Pair.second;
    k1.t[2] = t2Pair.second;
    k1.t[3] = t3Pair.second;

    return std::make_pair(k0, k1);
}

// 64 ANDs at a time
GroupElement evalBitwiseAnd(int party, GroupElement x, GroupElement y, const BitwiseAndKeyPack &key)
{
    GroupElement res = 0;
    for(uint64_t i = 0; i < 64; ++i)
    {
        uint8_t xBit = (x >> i) & 1;
        uint8_t yBit = (y >> i) & 1;
        GroupElement t = key.t[xBit * 2 + yBit];
        t = t & (1ULL << i);
        res = res ^ t;
    }
    return res;
}

GroupElement evalAnd(int party, GroupElement x, GroupElement y, const BitwiseAndKeyPack &key)
{
    uint8_t xBit = x & 1;
    uint8_t yBit = y & 1;
    GroupElement t = key.t[xBit * 2 + yBit] & 1;
    GroupElement res = res ^ t;
    return res;
}
