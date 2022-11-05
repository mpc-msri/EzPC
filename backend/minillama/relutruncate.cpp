#include "relutruncate.h"

pair<ReluTruncateKeyPack> keyGenReluTruncate(int bin, int bout, int s, GroupElement rin, GroupElement routTruncate, GroupElement routRelu, GroupElement rout) 
{
    always_assert(bin == bout); // for now
    mod(rin, bin);
    GroupElement r1, r0;
    if (rin % (1ULL << s) == 0) {
        r1 = (rin >> s);
        r0 = 0;
    }
    else {
        r1 = (rin >> s) + 1;
        r0 = (r1 << s) - rin;
    }
    if (LlamaConfig::stochasticRT) {
        r1 = (rin >> s) + 1;
    }
    auto dcfN = keyGenDCF(bin, s, rin, 1);
    GroupElement zTruncate;
    zTruncate = routTruncate - r1;
    mod(zTruncate, bin);

    GroupElement a;
    GroupElement b;
    GroupElement c;
    GroupElement d1 = 0;
    GroupElement d2 = 0;
    a = routRelu & 1;
    b = routTruncate;
    c = (routRelu & 1) * routTruncate + rout;
    if ((routRelu & 1) == 1) {
        d1 = 2;
        d2 = -2 * routTruncate;
    }

    ReluTruncateKeyPack k0, k1;
    k0.Bin = bin; k1.Bin = bin;
    k0.Bout = bout; k1.Bout = bout;
    k0.shift = s; k1.shift = s;
    k0.dcfKeyN = dcfN.first; k1.dcfKeyN = dcfN.second;
    if (!LlamaConfig::stochasticRT) {
        auto dcfS = keyGenDCF(s,   bout, r0,  1);
        k0.dcfKeyS = dcfS.first; k1.dcfKeyS = dcfS.second;
    }
    auto zTruncateSplit = splitShare(zTruncate, bin);
    k0.zTruncate = zTruncateSplit.first; k1.zTruncate = zTruncateSplit.second;
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

GroupElement evalRT_lrs(int party, GroupElement x, const ReluTruncateKeyPack &key, GroupElement &cache) {
    GroupElement x0;
    GroupElement x1;
    x0 = x % (1ULL << key.shift);
    x1 = x >> key.shift;
    mod(x0, key.shift);
    mod(x1, key.Bin);
    GroupElement xs;
    xs = (1ULL<<key.shift) - 1 - x0;
    mod(xs, key.shift);
    GroupElement ts = 0;
    evalDCF(party, &cache, x, key.dcfKeyN);
    if (LlamaConfig::stochasticRT) {
        ts = (party == 1) ? 1 : 0;
    }
    else {
        evalDCF(party, &ts, xs, key.dcfKeyS);
    }
    GroupElement res;
    if (party == 1) {
        res = x1 + key.zTruncate + (1ULL<<(key.Bin - key.shift)) * cache + ts;
    }
    else {
        res = key.zTruncate + (1ULL<<(key.Bin - key.shift)) * cache + ts;
    }
    mod(res, key.Bin);
    return res;
}

GroupElement evalRT_drelu(int party, GroupElement x, const ReluTruncateKeyPack &key, const GroupElement &cached) {
    GroupElement xp = x + (1ULL<<(key.Bin - 1));
    mod(xp, key.Bin);
    GroupElement t2 =0;
    evalDCF(party, &t2, xp, key.dcfKeyN);
    GroupElement res;
    res = t2 - cached + key.a;
    mod(res, 1);
    if (party == 1) {
        if (xp >= (1ULL<<(key.Bin - 1))) {
            res += 1;
        }
    }
    mod(res, 1);
    return res;
}

GroupElement evalRT_mult(int party, GroupElement x, GroupElement y, const ReluTruncateKeyPack &key) {
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
    mod(res, key.Bin);
    return res;
}
