#include "pubcmp.h"
#include <llama/dcf.h>

std::pair<PubCmpKeyPack, PubCmpKeyPack> keyGenPubCmp(int bin, GroupElement rin, GroupElement rout)
{
    mod(rin, bin);
    GroupElement r1, r0;
    
    auto dcfN = keyGenDCF(bin, 1, rin, 1);

    GroupElement a = rout & 1;

    PubCmpKeyPack k0, k1;
    k0.bin = bin; k1.bin = bin;
    k0.dcfKey = dcfN.first; k1.dcfKey = dcfN.second;
    
    auto aSplit = splitShare(a, 1);
    k0.rout = aSplit.first; k1.rout = aSplit.second;
    return std::make_pair(k0, k1);
}

GroupElement evalPubCmp(int party, GroupElement x, GroupElement c, const PubCmpKeyPack &key) {
    GroupElement y = x - c;
    mod(y, key.bin);
    mod(x, key.bin);
    GroupElement t2 = 0;
    GroupElement t1 = 0;
    evalDCF(party, &t2, y, key.dcfKey);
    evalDCF(party, &t1, x, key.dcfKey);
    GroupElement res = t2 - t1 + key.rout;
    if (party == 1) {
        GroupElement N = -c;
        mod(N, key.bin);
        if (y >= N) {
            res += 1;
        }
    }
    mod(res, 1);
    return res;
}
