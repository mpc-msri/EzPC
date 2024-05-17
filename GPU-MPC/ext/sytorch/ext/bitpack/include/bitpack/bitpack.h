#pragma once

#include <stdint.h>
#include <cstddef>

namespace bitpack {
    uint64_t mod(uint64_t x, int bw);
    std::size_t packed_size(std::size_t n, int bw);
    std::size_t pack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw);
    std::size_t unpack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw);
};
