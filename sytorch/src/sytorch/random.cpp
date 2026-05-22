#include <sytorch/random.h>

osuCrypto::PRNG prngWeights;
osuCrypto::PRNG prngStr;

// samples a random float in range [0, 1)
double rand_float() {
    auto t = prngStr.get<uint32_t>();
    return t / ((double)(1ULL<<32));
}
