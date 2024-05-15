#include "../ext/llama/fixtobfloat16.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 40;
    int numTrials = 255;
    
    for (int i = 1; i <= numTrials; ++i) {
        GroupElement rin = random_ge(bin);
        GroupElement rout = random_ge(14);
        GroupElement x = i;
        GroupElement xhat = x + rin;
        mod(xhat, bin);

        auto keys = keyGenF2BF16(bin, rin, rout);
        auto& key0 = keys.first;
        auto& key1 = keys.second;
        auto t0 = evalF2BF16_1(0, xhat, key0);
        auto t1 = evalF2BF16_1(1, xhat, key1);
        GroupElement k = t0.first + t1.first;
        GroupElement m = t0.second + t1.second;

        auto xm0 = evalF2BF16_2(0, xhat, k, m, key0);
        auto xm1 = evalF2BF16_2(1, xhat, k, m, key1);
        GroupElement xm = xm0 + xm1;

        auto r0 = evalF2BF16_3(0, k, xm, key0);
        auto r1 = evalF2BF16_3(1, k, xm, key1);

        GroupElement r = r0 + r1 - rout;

        mod(r, 13);

        k = r % (1LL << 6);
        m = r >> 6;

        always_assert(((m+128) >> (7-k)) == x);
    }
}