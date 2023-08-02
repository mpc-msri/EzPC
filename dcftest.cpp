#include "ext/llama/dcf.h"
#include "ext/llama/include/llama/assert.h"

void test1()
{
    GroupElement alpha = 150;
    auto keys = keyGenDCFET1(10, alpha, 1);

    for (int i = 0; i < 128; ++i)
    {
        auto node0 = evalDCFET1(0, i, keys.first);
        auto node1 = evalDCFET1(1, i, keys.second);
        
        // std::cout << node0.s << " " << node1.s << std::endl;
        always_assert(eq(node0.s, node1.s));
        always_assert(node0.t == node1.t);
        always_assert(((node0.v + node1.v) % 2) == 1);

        auto v0 = evalDCFET1_finalize(0, i, node0, keys.first);
        auto v1 = evalDCFET1_finalize(1, i, node1, keys.second);
        always_assert(v0 == (1 ^ v1));
    }

    for (int i = 128; i < 256; ++i)
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
}

void test1_2bit()
{
    GroupElement alpha = 150;
    for (int j = 0; j < 10; ++j)
    {
        auto keys = keyGenDCFET2(10, alpha, 1);

        for (int i = 0; i < 128; ++i)
        {
            auto node0 = evalDCFET2(0, i, keys.first);
            auto node1 = evalDCFET2(1, i, keys.second);
            
            // std::cout << node0.s << " " << node1.s << std::endl;
            always_assert(eq(node0.s, node1.s));
            always_assert(node0.t == node1.t);
            always_assert(((node0.v + node1.v) % 4) == 1);

            auto v0 = evalDCFET2_finalize(0, i, node0, keys.first);
            auto v1 = evalDCFET2_finalize(1, i, node1, keys.second);
            always_assert(((v0 + v1) % 4) == 1);
        }

        for (int i = 128; i < 192; ++i)
        {
            auto node0 = evalDCFET2(0, i, keys.first);
            auto node1 = evalDCFET2(1, i, keys.second);
            always_assert(!eq(node0.s, node1.s)); // with high probability
            always_assert(node0.t == (1 ^ node1.t));

            auto v0 = evalDCFET2_finalize(0, i, node0, keys.first);
            auto v1 = evalDCFET2_finalize(1, i, node1, keys.second);

            if (i < alpha)
                always_assert(((v0 + v1) % 4) == 1);
            else
                always_assert(((v0 + v1) % 4) == 0);
        }

        for (int i = 192; i < 1024; ++i)
        {
            auto node0 = evalDCFET2(0, i, keys.first);
            auto node1 = evalDCFET2(1, i, keys.second);
            
            always_assert(eq(node0.s, node1.s));
            always_assert(node0.t == node1.t);
            always_assert(((node0.v + node1.v) % 4) == 0);

            auto v0 = evalDCFET2_finalize(0, i, node0, keys.first);
            auto v1 = evalDCFET2_finalize(1, i, node1, keys.second);
            always_assert(((v0 + v1) % 4) == 0);
        }
    }
}

void exhaustive_test_2bit(int bw)
{
    std::cout << "Exhaustive Test with bw = " << bw << std::endl;
    GroupElement alpha = random_ge(bw);
    for (int j = 0; j < 10; ++j)
    {
        auto keys = keyGenDCFET2(bw, alpha, 1);
        auto t0 = (alpha / 64) * 64;
        auto t1 = (alpha / 64) * 64 + 64;
        auto m = (1LL << bw);

        for (int i = 0; i < t0; ++i)
        {
            auto node0 = evalDCFET2(0, i, keys.first);
            auto node1 = evalDCFET2(1, i, keys.second);
            
            // std::cout << node0.s << " " << node1.s << std::endl;
            always_assert(eq(node0.s, node1.s));
            always_assert(node0.t == node1.t);
            always_assert(((node0.v + node1.v) % 4) == 1);

            auto v0 = evalDCFET2_finalize(0, i, node0, keys.first);
            auto v1 = evalDCFET2_finalize(1, i, node1, keys.second);
            always_assert(((v0 + v1) % 4) == 1);
        }

        for (int i = t0; i < t1; ++i)
        {
            auto node0 = evalDCFET2(0, i, keys.first);
            auto node1 = evalDCFET2(1, i, keys.second);
            always_assert(!eq(node0.s, node1.s)); // with high probability
            always_assert(node0.t == (1 ^ node1.t));

            auto v0 = evalDCFET2_finalize(0, i, node0, keys.first);
            auto v1 = evalDCFET2_finalize(1, i, node1, keys.second);

            if (i < alpha)
                always_assert(((v0 + v1) % 4) == 1);
            else
                always_assert(((v0 + v1) % 4) == 0);
        }

        for (int i = t1; i < m; ++i)
        {
            auto node0 = evalDCFET2(0, i, keys.first);
            auto node1 = evalDCFET2(1, i, keys.second);
            
            always_assert(eq(node0.s, node1.s));
            always_assert(node0.t == node1.t);
            always_assert(((node0.v + node1.v) % 4) == 0);

            auto v0 = evalDCFET2_finalize(0, i, node0, keys.first);
            auto v1 = evalDCFET2_finalize(1, i, node1, keys.second);
            always_assert(((v0 + v1) % 4) == 0);
        }
    }
}

void exhaustive_test(int bw)
{
    std::cout << "Exhaustive Test with bw = " << bw << std::endl;
    GroupElement alpha = random_ge(bw);
    auto keys = keyGenDCFET1(bw, alpha, 1);

    GroupElement t1 = (alpha / 128) * 128;
    GroupElement t2 = t1 + 128;
    GroupElement m = (1LL << bw);

    for (GroupElement i = 0; i < t1; ++i)
    {
        auto node0 = evalDCFET1(0, i, keys.first);
        auto node1 = evalDCFET1(1, i, keys.second);
        
        // std::cout << node0.s << " " << node1.s << std::endl;
        always_assert(eq(node0.s, node1.s));
        always_assert(node0.t == node1.t);
        always_assert(((node0.v + node1.v) % 2) == 1);

        auto v0 = evalDCFET1_finalize(0, i, node0, keys.first);
        auto v1 = evalDCFET1_finalize(1, i, node1, keys.second);
        always_assert(v0 == (1 ^ v1));
    }

    for (int i = t1; i < t2; ++i)
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

    for (int i = t2; i < m; ++i)
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
}

void random_test(int bw)
{
    std::cout << "Random Test with bw = " << bw << std::endl;
    GroupElement alpha = random_ge(bw);
    auto keys = keyGenDCFET1(bw, alpha, 1);

    for (GroupElement i = 0; i < 1000; ++i)
    {
        GroupElement idx = random_ge(bw);
        auto node0 = evalDCFET1(0, idx, keys.first);
        auto node1 = evalDCFET1(1, idx, keys.second);

        auto v0 = evalDCFET1_finalize(0, idx, node0, keys.first);
        auto v1 = evalDCFET1_finalize(1, idx, node1, keys.second);
        if (idx < alpha)
            always_assert(v0 == (1 ^ v1));
        else
            always_assert(v0 == v1);
    }
}

int main()
{
    uint64_t seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    test1();
    exhaustive_test(11);
    exhaustive_test(12);
    exhaustive_test(13);
    exhaustive_test(14);
    exhaustive_test(15);
    exhaustive_test(16);

    random_test(64);
    random_test(63);
    random_test(62);
    random_test(61);

    test1_2bit();
    exhaustive_test_2bit(11);
    exhaustive_test_2bit(12);
    exhaustive_test_2bit(13);
    exhaustive_test_2bit(14);
    exhaustive_test_2bit(15);
    exhaustive_test_2bit(16);

    return 0;
}
