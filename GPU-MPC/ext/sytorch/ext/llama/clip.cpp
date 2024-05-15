#include "clip.h"
#include <llama/dcf.h>
#include "pubcmp.h"

std::pair<ClipKeyPack, ClipKeyPack> keyGenClip(int bin, GroupElement rin, GroupElement rout)
{
    GroupElement routCmp = random_ge(1);
    auto cmpKeys = keyGenPubCmp(bin, rin, routCmp);
    
    GroupElement a;
    GroupElement b;
    GroupElement c;
    GroupElement d1 = 0;
    GroupElement d2 = 0;
    a = routCmp & 1;
    b = rin;
    c = a * rin + rout;
    if (a == 1) {
        d1 = 2;
        d2 = -2 * rin;
    }

    ClipKeyPack k0, k1;
    k0.bin = bin; k1.bin = bin;
    k0.cmpKey = cmpKeys.first; k1.cmpKey = cmpKeys.second;
    
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

GroupElement evalClip_1(int party, int maxBw, GroupElement x, const ClipKeyPack &key)
{
    GroupElement s = evalPubCmp(party, x, (1LL<<maxBw), key.cmpKey);
    return s;
}

GroupElement evalClip_2(int party, int maxBw, GroupElement x, GroupElement y, const ClipKeyPack &key)
{
    GroupElement t1 = 0;
    GroupElement t2 = 0;
    mod(x, 1);
    if (x == 0) {
        t1 = key.d1;
        t2 = key.d2;
    }
    GroupElement res;
    res = -key.a * y - key.b * x + key.c + y * t1 + t2;
    if (party == 1) {
        res += x * y;
    }

    GroupElement maxval = (1LL<<maxBw) - 1;
    res = res - maxval * key.a;
    if (x == 1) {
        res = res + maxval * key.d1;
    }
    if ((party == 1) && (x == 0)) {
        res = res + maxval;
    }
    mod(res, key.bin);
    return res;
}