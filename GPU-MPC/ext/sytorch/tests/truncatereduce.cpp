#include "../ext/llama/pubdiv.h"
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
    int scale = 12;
    
    for (int i = 0; i < numTrials; ++i) {
        GroupElement r = random_ge(bin);
        GroupElement rout = random_ge(bin-scale);
        GroupElement x = random_ge(bin);
        GroupElement xhat = x + r;
        mod(xhat, bin);

        auto keys = keyGenTruncateReduce(bin, scale, r, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;

        GroupElement t0_2 = evalTruncateReduce(0, xhat, key0);
        GroupElement t1_2 = evalTruncateReduce(1, xhat, key1);

        GroupElement t_2 = t0_2 + t1_2 - rout;
        mod(t_2, bin - scale);

        always_assert(t_2 == (x >> scale));
    }
}
