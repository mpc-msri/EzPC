#include "../ext/llama/lut.h"
#include <iostream>
#include <sytorch/backend/llama_base.h>

int main()
{
    u64 seedKey = 0xdeadbeefbadc0ffe;
    for(int i = 0; i < 256; ++i) {
        LlamaConfig::prngs[i].SetSeed(osuCrypto::toBlock(time(NULL), seedKey));
    }

    int bin = 8;
    int bout = 64;
    std::vector<GroupElement> tab(256);
    for (int i = 0; i < 256; ++i)
    {
        tab[i] = i + 1;
    }

    for (int i = 0; i < 1000; ++i)
    {
        GroupElement rin = rand() % 256;
        GroupElement x = rand() % 256;
        GroupElement xhat = (x + rin) % 256;

        auto keys = keyGenLUTSS(bin, bout, rin, 0);
        auto y0 = evalLUTSS_1(0, xhat, tab, keys.first);
        auto y1 = evalLUTSS_1(1, xhat, tab, keys.second);
        GroupElement res = y0.first + y1.first;
        GroupElement corr = y0.second + y1.second;
        GroupElement f0 = evalLUTSS_2(0, res, corr, keys.first);
        GroupElement f1 = evalLUTSS_2(1, res, corr, keys.second);
        GroupElement y = f0 + f1;
        always_assert(y == tab[x]);
    }
}
