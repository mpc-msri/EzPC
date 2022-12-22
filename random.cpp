#include "random.h"

osuCrypto::PRNG prngWeights;

// samples a random float in range [0, 1)
double rand_float() {
    auto t = prngWeights.get<uint32_t>();
    return t / ((double)(1ULL<<32));
}

void rand_init() {
    prngWeights.SetSeed(osuCrypto::toBlock(time(NULL)));
}
