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
    int samples = 18432;

    std::vector<GroupElement> tab(1LL<<bin);
    for (int i = 0; i < tab.size(); ++i)
    {
        tab[i] = rand();
    }

    DPFKeyPack *keys = new DPFKeyPack[samples];
    GroupElement *res = new GroupElement[samples];
    for (int idx = 0; idx < samples; ++idx)
    {
        auto keypair = keyGenDPF(bin, bout, rand(), 1);
        keys[idx] = keypair.first;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int idx = 0; idx < samples; ++idx)
    {
        res[idx] = evalAll_reduce(0, keys[idx], 0, tab);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << compute_time / 1000.0 << " ms" << std::endl;

    for (int idx = 0; idx < samples; ++idx)
    {
        std::cerr << res[idx] << std::endl;
    }
}