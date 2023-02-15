#pragma once
#include <cryptoTools/Crypto/PRNG.h>

namespace LlamaConfig {
    extern osuCrypto::PRNG prngs[256];
}

extern osuCrypto::PRNG prngShared;
