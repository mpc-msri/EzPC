#include <bitpack/bitpack.h>

namespace bitpack {
    
    inline uint64_t mod(uint64_t x, int bw)
    {
        return x & ((1LL << bw) - 1);
    }

    std::size_t packed_size(std::size_t n, int bw)
    {
        return (n * bw + 63) / 64;
    }

    std::size_t pack_1bit(uint64_t *dst, const uint64_t *src, std::size_t n)
    {
        for (int i = 0; i < n; ++i)
        {
            std::size_t dsti = i / 64;
            std::size_t dstj = i % 64;
            dst[dsti] |= ((src[i] & 1) << dstj);
        }
        return packed_size(n, 1);
    }

    /// packs `n` `bw` bit integers from `src` into `dst`
    std::size_t pack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw)
    {
        if (bw == 64)
        {
            for (int i = 0; i < n; ++i)
            {
                dst[i] = src[i];
            }
            return n;
        }

        std::size_t ps = packed_size(n, bw);
        for (int i = 0; i < ps; ++i)
            dst[i] = 0;

        if (bw == 1)
        {
            return pack_1bit(dst, src, n);
        }

        std::size_t dsti = 0;
        std::size_t dstj = 0;

        
        for (int i = 0; i < n; ++i)
        {
            uint64_t x = mod(src[i], bw);
            std::size_t rem = 64 - dstj;
            if (bw <= rem)
            {
                dst[dsti] |= x << dstj;
                dstj += bw;
                if (dstj == 64)
                {
                    dstj = 0;
                    ++dsti;
                }
            }
            else
            {
                dst[dsti] |= x << dstj;
                dstj += bw;
                dst[dsti + 1] |= x >> rem;
                dstj -= 64;
                ++dsti;
            }
        }

        return dsti + (dstj > 0);
    }

    std::size_t unpack_1bit(uint64_t *dst, const uint64_t *src, std::size_t n)
    {
        for (int i = 0; i < n; ++i)
        {
            std::size_t srci = i / 64;
            std::size_t srcj = i % 64;
            dst[i] = src[srci] >> srcj;
        }
        return packed_size(n, 1);
    }

    /// unpacks `n` `bw` bit integers from `src` into `dst`
    std::size_t unpack(uint64_t *dst, const uint64_t *src, std::size_t n, int bw)
    {
        if (bw == 1)
        {
            return unpack_1bit(dst, src, n);
        }
        
        if (bw == 64)
        {
            for (int i = 0; i < n; ++i)
            {
                dst[i] = src[i];
            }
            return n;
        }

        std::size_t srci = 0;
        std::size_t srcj = 0;
        uint64_t cache = src[0];
        
        for (int i = 0; i < n; ++i)
        {
            uint64_t x = cache >> srcj;
            std::size_t rem = 64 - srcj;
            if (bw <= rem)
            {
                dst[i] = x;
                srcj += bw;
                if (srcj == 64)
                {
                    srcj = 0;
                    ++srci;
                    cache = src[srci];
                }
            }
            else
            {
                ++srci;
                cache = src[srci];
                dst[i] = x | (cache << rem);
                srcj = srcj + bw - 64;
            }
        }

        return srci + (srcj > 0);
    }
};
