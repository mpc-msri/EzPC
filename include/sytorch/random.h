#pragma once

#include <cryptoTools/Crypto/PRNG.h>

extern osuCrypto::PRNG prngWeights;
extern osuCrypto::PRNG prngStr;

// samples a random float in range [0, 1)
double rand_float();
