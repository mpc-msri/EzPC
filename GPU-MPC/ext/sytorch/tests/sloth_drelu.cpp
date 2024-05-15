#include "../ext/llama/relu.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 64;
    int numTrials = 100;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(1);
        GroupElement x = random_ge(bin);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenSlothDrelu(bin, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        GroupElement t0 = evalSlothDrelu(0, xhat, key0);
        GroupElement t1 = evalSlothDrelu(1, xhat, key1);
        GroupElement t = t0 ^ t1 ^ rout;
        if (x < (1ULL<<(bin-1))) {
            always_assert(t == 1);
        } else {
            always_assert(t == 0);
        }
    }
}
