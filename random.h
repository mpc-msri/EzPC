#pragma once

#include <cryptoTools/Crypto/PRNG.h>

extern osuCrypto::PRNG prngWeights;

// samples a random float in range [0, 1)
double rand_float();
void rand_init();
