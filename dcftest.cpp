#include "ext/llama/dcf.h"
#include "ext/llama/include/llama/assert.h"

int main()
{
    uint64_t seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    GroupElement alpha = 150;
    auto keys = keyGenDCFET1(10, alpha, 1);

    for (int i = 0; i < 128; ++i)
    {
        auto node0 = evalDCFET1(0, i, keys.first);
        auto node1 = evalDCFET1(1, i, keys.second);
        
        std::cout << node0.s << " " << node1.s << std::endl;
        always_assert(eq(node0.s, node1.s));
        always_assert(node0.t == node1.t);
        always_assert(((node0.v + node1.v) % 2) == 1);

        auto v0 = evalDCFET1_finalize(0, i, node0, keys.first);
        auto v1 = evalDCFET1_finalize(1, i, node1, keys.second);
        always_assert(v0 == (1 ^ v1));
    }

    for (int i = 129; i < 256; ++i)
    {
        auto node0 = evalDCFET1(0, i, keys.first);
        auto node1 = evalDCFET1(1, i, keys.second);
        
        always_assert(!eq(node0.s, node1.s)); // with high probability
        always_assert(node0.t == (1 ^ node1.t));
        auto v0 = evalDCFET1_finalize(0, i, node0, keys.first);
        auto v1 = evalDCFET1_finalize(1, i, node1, keys.second);
        // std::cout << i << std::endl;
        if (i < alpha)
            always_assert(v0 == (1 ^ v1));
        else
            always_assert(v0 == v1);
    }

    for (int i = 256; i < 1024; ++i)
    {
        auto node0 = evalDCFET1(0, i, keys.first);
        auto node1 = evalDCFET1(1, i, keys.second);
        
        always_assert(eq(node0.s, node1.s));
        always_assert(node0.t == node1.t);
        always_assert(((node0.v + node1.v) % 2) == 0);

        auto v0 = evalDCFET1_finalize(0, i, node0, keys.first);
        auto v1 = evalDCFET1_finalize(1, i, node1, keys.second);
        always_assert(v0 == v1);
    }

    // osuCrypto::block lol = osuCrypto::toBlock(1, 0);
    // std::cout << "lol: " << lol << std::endl;
    // std::cout << "sr(lol, 8): " << _mm_srli_si128(lol, 8) << std::endl;
    // std::cout << "lol: " << (osuCrypto::ZeroBlock | osuCrypto::toBlock(1LL << 5, 0)) << std::endl;

    return 0;
}
