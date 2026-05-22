#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/AES.h>
#include <random>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace osuCrypto {

    const block ZeroBlock = _mm_set_epi64x(0, 0);
    const block OneBlock = _mm_set_epi64x(0, 1);
    const block AllOneBlock = _mm_set_epi64x(u64(-1), u64(-1));
    const std::array<block, 2> zeroAndAllOne = { { ZeroBlock, AllOneBlock } };
    const block CCBlock = ([]() {block cc; memset(&cc, 0xcc, sizeof(block)); return cc; })();



    block PRF(const block& b, u64 i)
    {
		return AES(b).ecbEncBlock(toBlock(i));
    }

    void split(const std::string &s, char delim, std::vector<std::string> &elems) {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) {
            elems.push_back(item);
        }
    }

    std::vector<std::string> split(const std::string &s, char delim) {
        std::vector<std::string> elems;
        split(s, delim, elems);
        return elems;
    }

    const int tab64[64] = {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5 };


    u64 log2floor(u64 value)
    {
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        value |= value >> 32;
        return tab64[((uint64_t)((value - (value >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
    }

    u64 log2ceil(u64 value)
    {
        auto floor = log2floor(value);

        return floor + (value > (1ull << floor));
        //return u64(std::ceil(std::log2(value)));
    }

    block sysRandomSeed()
    {
        std::random_device rd;
		auto ret = std::array<unsigned int, 4>{rd(), rd(), rd(), rd()};
		return *(block*)&ret;
    }
}



std::ostream& operator<<(std::ostream& out, const oc::block& blk)
{
	using namespace oc;
	out << std::hex;
	u64* data = (u64*)&blk;

	out << std::setw(16) << std::setfill('0') << data[1]
		<< std::setw(16) << std::setfill('0') << data[0];

	out << std::dec << std::setw(0);
	return out;
}