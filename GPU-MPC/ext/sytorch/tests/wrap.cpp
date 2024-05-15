#include "../ext/llama/wrap.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

void dpf_main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 64;
    int numTrials = 1000;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(1);
        GroupElement x = random_ge(bin);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenWrapDPF(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        GroupElement t0 = evalWrapDPF(0, xhat, key0);
        GroupElement t1 = evalWrapDPF(1, xhat, key1);
        GroupElement t = t0 ^ t1 ^ rout;
        if (xhat < r) {
            always_assert(t == 1);
        } else {
            always_assert(t == 0);
        }
    }
}

void ss_main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 6;
    int numTrials = 1000;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(1);
        GroupElement x = random_ge(bin);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenWrapSS(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        GroupElement t0 = evalWrapSS(0, xhat, key0);
        GroupElement t1 = evalWrapSS(1, xhat, key1);
        GroupElement t = t0 ^ t1 ^ rout;
        if (xhat < r) {
            always_assert(t == 1);
        } else {
            always_assert(t == 0);
        }
    }
}

int main()
{
    dpf_main();
    ss_main();
}
