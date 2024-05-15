#include <llama/dpf.h>
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 16;
    int bout = 64;

    std::vector<GroupElement> tab(1LL<<bin);
    for (int i = 0; i < (1LL<<bin); ++i)
    {
        tab[i] = rand();
    }

    for (int i = 0; i < 1000; ++i)
    {
        GroupElement idx = rand() % (1LL<<bin);
        auto keys = keyGenDPFET(bin, idx);
        auto& key0 = keys.first;
        auto& key1 = keys.second;

        GroupElement rot = rand() % (1LL<<bin);

        auto res0 = evalAll_reduce_et(0, key0, rot, tab);
        auto res1 = evalAll_reduce_et(1, key1, rot, tab);
        // always_assert(res0 + res1 == tab[idx]);
        always_assert((res0.first +  res1.first) * (res0.second + res1.second) == tab[(idx+rot) % (1LL<<bin)]);

    }
}