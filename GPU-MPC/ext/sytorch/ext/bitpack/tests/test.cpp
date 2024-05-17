#include <bitpack/bitpack.h>
#include <cstdlib>
#include <iostream>

int main()
{
    std::size_t n = 1000;
    std::size_t bw = 64;
    std::size_t n_dst = bitpack::packed_size(n, bw);
    
    uint64_t src[n];
    uint64_t src2[n];

    for (int i = 0; i < n; ++i)
        src[i] = bitpack::mod(i, bw);

    uint64_t dst[n_dst];
    auto sz = bitpack::pack(dst, src, n, bw);

    if (sz != n_dst)
    {
        std::cerr << "sz != n_dst" << std::endl;
        return 1;
    }

    bitpack::unpack(src2, dst, n, bw);
    for (int i = 0; i < n; ++i)
    {
        if (src[i] != src2[i])
        {
            std::cerr << "src[" << i << "] != src2[" << i << "]" << std::endl;
            std::cout << "expected = " << src[i] << std::endl;
            std::cout << "got      = " << src2[i] << std::endl;
            return 1;
        }
    }
    return 0;
}
