#include "../ext/llama/pubcmp.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 40;
    int numTrials = 10000;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(1);
        GroupElement x = random_ge(bin);
        GroupElement c = random_ge(bin);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenPubCmp(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        GroupElement t0 = evalPubCmp(0, xhat, c, key0);
        GroupElement t1 = evalPubCmp(1, xhat, c, key1);
        GroupElement t = t0 ^ t1 ^ rout;
        mod(t, 1);

        if (x < c) {
            always_assert(t == 1);
        } else {
            always_assert(t == 0);
        }
    }
}