#include "signextend.h"
#include "dcf.h"

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
