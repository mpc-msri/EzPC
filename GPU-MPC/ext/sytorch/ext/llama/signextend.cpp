#include "signextend.h"
#include <llama/dcf.h>

std::pair<SignExtend2KeyPack, SignExtend2KeyPack> keyGenSignExtend2(int bin, int bout, GroupElement rin, GroupElement rout)
{
    std::pair<SignExtend2KeyPack, SignExtend2KeyPack> keys;
    
    auto dcfKeys = keyGenDCF(bin, 1, rin, 1);
    keys.first.dcfKey = dcfKeys.first;
    keys.second.dcfKey = dcfKeys.second;

    GroupElement rw = random_ge(1);
    auto rw_split = splitShare(rw, 1);
    keys.first.rw = rw_split.first;
    keys.second.rw = rw_split.second;

    GroupElement p[2];
    if ((rw % 2) == 0) {
        p[0] = rout - rin - (1ULL << (bin - 1));
        p[1] = rout - rin + (1ULL << (bin - 1));
    } else {
        p[0] = rout - rin + (1ULL << (bin - 1));
        p[1] = rout - rin - (1ULL << (bin - 1));
    }

    for (int i = 0; i < 2; ++i) {
        auto p_split = splitShare(p[i], bout);
        keys.first.p[i] = p_split.first;
        keys.second.p[i] = p_split.second;
    }

    return keys;
}

std::pair<SlothSignExtendKeyPack, SlothSignExtendKeyPack> keyGenSlothSignExtend(int bin, int bout, GroupElement rin, GroupElement w, GroupElement rout)
{
    SlothSignExtendKeyPack k0, k1;
    k0.bin = bin;
    k1.bin = bin;
    k0.bout = bout;
    k1.bout = bout;

    auto rout_split = splitShare(rout - rin - (1LL << (bin - 1)), bout);
    k0.rout = rout_split.first;
    k1.rout = rout_split.second;

    mod(w, 1);
    auto w_split = splitShare(w, bout);
    k0.select = w_split.first;
    k1.select = w_split.second;

    return std::make_pair(k0, k1);
}

GroupElement evalSlothSignExtend(int party, GroupElement x, GroupElement w, const SlothSignExtendKeyPack &kp)
{
    mod(w, 1);
    return party * x + kp.rout + (1LL << kp.bin) * ((1 - w) * kp.select + w * (party - kp.select));
}
