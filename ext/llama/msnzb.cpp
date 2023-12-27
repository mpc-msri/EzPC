#include "msnzb.h"

std::pair<MSNZBKeyPack, MSNZBKeyPack> keyGenMSNZB(int bin, int bout, GroupElement rin, GroupElement rout, int start, int end)
{
    if (end == -1) end = (bin - 1);
    
    std::pair<MSNZBKeyPack, MSNZBKeyPack> keys;
    int m = end - start + 1;
    uint64_t p[m], q[m];

    for(uint64_t i = start; i <= end; ++i)
    {
        p[i - start] = (1ULL << i);
        q[i - start] = (1ULL << (i + 1)) - 1;
    }

    auto micKeys = keyGenMIC(bin, bout, m, p, q, rin, nullptr);
    keys.first.micKey = micKeys.first;
    keys.second.micKey = micKeys.second;
    auto rpair = splitShare(rout, bout);
    keys.first.r = rpair.first;
    keys.second.r = rpair.second;
    
    return keys;
}

GroupElement evalMSNZB(int party, int bin, int bout, GroupElement x, const MSNZBKeyPack &key, int start, int end, GroupElement *zcache)
{
    if (end == -1) end = (bin - 1);
    
    int m = end - start + 1;
    uint64_t p[m], q[m];
    GroupElement z[m];
    // for(int i = 0; i < m; ++i) z[m].bitsize = bout;

    for(uint64_t i = start; i <= end; ++i)
    {
        p[i - start] = (1ULL << i);
        q[i - start] = (1ULL << (i + 1)) - 1;
    }

    evalMIC(party, bin, bout, m, p, q, x, key.micKey, z);

    if (zcache != nullptr) {
        for(int i = 0; i < m; ++i) zcache[i] = z[i];
    }

    GroupElement sum = key.r;
    for(int i = start; i <= end; ++i)
    {
        sum = sum + i * z[i - start];
    }
    mod(sum, bout);
    return sum;
}
