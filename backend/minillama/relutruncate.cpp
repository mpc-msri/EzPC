#include "relutruncate.h"

pair<ReluTruncateKeyPack> keyGenReluTruncate(int bin, int bout, int s, GroupElement rin, GroupElement routTruncate, GroupElement routRelu, GroupElement rout) 
{
    always_assert(bin == bout); // for now
    mod(rin);
    GroupElement r1, r0;
    r1.bitsize = bin;
    r0.bitsize = s;
    if (rin.value % (1ULL << s) == 0) {
        r1.value = (rin.value >> s);
        r0.value = 0;
    }
    else {
        r1.value = (rin.value >> s) + 1;
        r0.value = (r1.value << s) - rin.value;
    }
    GroupElement one(1, bin);
    auto dcfN = keyGenDCF(bin, bout, rin, one);
    auto dcfS = keyGenDCF(s,   bout, r0,  one);
    GroupElement zTruncate(0, bin);
    zTruncate.value = routTruncate.value - r1.value;
    mod(zTruncate);

    GroupElement a(0, bin);
    GroupElement b(0, bin);
    GroupElement c(0, bin);
    GroupElement d1(0, bin);
    GroupElement d2(0, bin);
    a.value = routRelu.value & 1;
    b.value = routTruncate.value;
    c.value = (routRelu.value & 1) * routTruncate.value + rout.value;
    if (routRelu.value & 1 == 1) {
        d1.value = 2;
        d2.value = -2 * routTruncate.value;
    }

    ReluTruncateKeyPack k0, k1;
    k0.Bin = bin; k1.Bin = bin;
    k0.Bout = bout; k1.Bout = bout;
    k0.shift = s; k1.shift = s;
    k0.dcfKeyN = dcfN.first; k1.dcfKeyN = dcfN.second;
    k0.dcfKeyS = dcfS.first; k1.dcfKeyS = dcfS.second;
    auto zTruncateSplit = splitShare(zTruncate);
    k0.zTruncate = zTruncateSplit.first; k1.zTruncate = zTruncateSplit.second;
    auto aSplit = splitShare(a);
    k0.a = aSplit.first; k1.a = aSplit.second;
    auto bSplit = splitShare(b);
    k0.b = bSplit.first; k1.b = bSplit.second;
    auto cSplit = splitShare(c);
    k0.c = cSplit.first; k1.c = cSplit.second;
    auto d1Split = splitShare(d1);
    k0.d1 = d1Split.first; k1.d1 = d1Split.second;
    auto d2Split = splitShare(d2);
    k0.d2 = d2Split.first; k1.d2 = d2Split.second;
    return std::make_pair(k0, k1);

}

GroupElement evalRT_lrs(int party, GroupElement x, const ReluTruncateKeyPack &key, GroupElement &cache) {
    GroupElement x0(0, key.shift);
    GroupElement x1(0, key.Bin);
    x0.value = x.value % (1ULL << key.shift);
    x1.value = x.value >> key.shift;
    mod(x0);
    mod(x1);
    GroupElement xs(0, key.shift);
    xs.value = (1ULL<<key.shift) - 1 - x0.value;
    mod(xs);
    GroupElement ts(0, key.Bin);
    evalDCF(party, &cache, x, key.dcfKeyN);
    evalDCF(party, &ts, xs, key.dcfKeyS);
    GroupElement res(0, key.Bin);
    if (party == 1) {
        res.value = x1.value + key.zTruncate.value + (1ULL<<(key.Bin - key.shift)) * cache.value + ts.value;
    }
    else {
        res.value = key.zTruncate.value + (1ULL<<(key.Bin - key.shift)) * cache.value + ts.value;
    }
    mod(res);
    return res;
}

GroupElement evalRT_drelu(int party, GroupElement x, const ReluTruncateKeyPack &key, const GroupElement &cached) {
    GroupElement xp(x.value + (1ULL<<(key.Bin - 1)), key.Bin);
    GroupElement t2(0, key.Bin);
    evalDCF(party, &t2, xp, key.dcfKeyN);
    GroupElement res(0, 1);
    res.value = t2.value - cached.value + key.a.value;
    mod(res);
    if (party == 1) {
        if (xp.value >= (1ULL<<(key.Bin - 1))) {
            res.value += 1;
        }
    }
    mod(res);
    return res;
}

GroupElement evalRT_mult(int party, GroupElement x, GroupElement y, const ReluTruncateKeyPack &key) {
    GroupElement t1 = GroupElement(0, key.Bin);
    GroupElement t2 = GroupElement(0, key.Bin);
    if (x.value == 0) {
        t1.value = key.d1.value;
        t2.value = key.d2.value;
    }
    GroupElement res(0, key.Bin);
    res.value = -key.a.value * y.value - key.b.value * x.value + key.c.value + y.value * t1.value + t2.value;
    if (party == 1) {
        res.value += x.value * y.value;
    }
    mod(res);
    return res;
}
