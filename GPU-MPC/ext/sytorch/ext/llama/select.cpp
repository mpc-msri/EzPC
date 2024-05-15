#include "select.h"

std::pair<SelectKeyPack, SelectKeyPack> keyGenSelect(int bin, GroupElement s, GroupElement y, GroupElement out)
{
    GroupElement a;
    GroupElement b;
    GroupElement c;
    GroupElement d1 = 0;
    GroupElement d2 = 0;
    a = s & 1;
    b = y;
    c = (s & 1) * y + out;
    if ((s & 1) == 1) {
        d1 = 2;
        d2 = -2 * y;
    }

    SelectKeyPack k0, k1;
    k0.Bin = bin;
    k1.Bin = bin;
    auto aSplit = splitShare(a, bin);
    k0.a = aSplit.first; k1.a = aSplit.second;
    auto bSplit = splitShare(b, bin);
    k0.b = bSplit.first; k1.b = bSplit.second;
    auto cSplit = splitShare(c, bin);
    k0.c = cSplit.first; k1.c = cSplit.second;
    auto d1Split = splitShare(d1, bin);
    k0.d1 = d1Split.first; k1.d1 = d1Split.second;
    auto d2Split = splitShare(d2, bin);
    k0.d2 = d2Split.first; k1.d2 = d2Split.second;
    return std::make_pair(k0, k1);
}

GroupElement evalSelect(int party, GroupElement s, GroupElement x, const SelectKeyPack &key)
{
    GroupElement t1 = 0;
    GroupElement t2 = 0;
    mod(s, 1);
    if (s == 0) {
        t1 = key.d1;
        t2 = key.d2;
    }
    GroupElement res;
    res = -key.a * x - key.b * s + key.c + x * t1 + t2;
    if (party == 1) {
        res += s * x;
    }
    mod(res, key.Bin);
    return res;
}