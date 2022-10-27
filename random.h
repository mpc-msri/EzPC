#pragma once

#include <cryptoTools/Crypto/PRNG.h>

extern osuCrypto::PRNG prngWeights;

// samples a random float in range [0, 1)
inline double rand_float() {
    auto t = prngWeights.get<uint32_t>();
    return t / ((double)(1ULL<<32));
}
