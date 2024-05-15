#include <llama/dpf.h>
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int samples = 1000;
    int bin = 12;

    for (int j = 0; j < samples; ++j)
    {
        GroupElement idx = rand() % (1LL << bin);
        auto keys = keyGenDPFET(bin, idx);
        auto& key0 = keys.first;
        auto& key1 = keys.second;

        for (int i = 0; i < (1LL<<bin); ++i)
        {
            auto res0 = evalDPFET_LT(0, key0, i);
            auto res1 = evalDPFET_LT(1, key1, i);
            auto res = res0 ^ res1;
            if (i < idx) 
            {
                always_assert(res == 1);
            }
            else 
            {
                always_assert(res == 0);
            }
        }
    }
}