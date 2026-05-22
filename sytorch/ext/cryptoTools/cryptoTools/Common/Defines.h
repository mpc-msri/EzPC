#pragma once
// This file and the associated implementation has been placed in the public domain, waiving all copyright. No restrictions are placed on its use.

#include <cstdint>
#include <iostream>
#include <memory>

#include <emmintrin.h>
#include <smmintrin.h>

#include <cryptoTools/Common/config.h>


#ifdef ENABLE_FULL_GSL
#include <cryptoTools/gsl/span>
#else
#include <cryptoTools/gsl/gls-lite.hpp>
#endif

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define LOCATION __FILE__ ":" STRINGIZE(__LINE__)
#define RTE_LOC std::runtime_error(LOCATION)

#ifdef _MSC_VER
#ifndef _WIN32_WINNT
// compile for win 7 and up.
#define _WIN32_WINNT 0x0601  
//#define _WINSOCK_DEPRECATED_NO_WARNINGS
#endif // !_WIN32_WINNT


	#pragma warning( disable : 4018) // signed unsigned comparison warning
	#define TODO(x) __pragma(message (__FILE__ ":" STRINGIZE(__LINE__) " Warning:TODO - " #x))
#else
	#define TODO(x)
#endif

// add instrinsics names that intel knows but clang doesn't…
#ifdef __clang__
#define _mm_cvtsi128_si64x _mm_cvtsi128_si64
#endif


namespace osuCrypto {
    template<typename T> using ptr = T*;
    template<typename T> using uPtr = std::unique_ptr<T>;
    template<typename T> using sPtr = std::shared_ptr<T>;
    template<typename T> using span = gsl::span<T>;

    typedef uint64_t u64;
    typedef int64_t i64;
    typedef uint32_t u32;
    typedef int32_t i32;
    typedef uint16_t u16;
    typedef int16_t i16;
    typedef uint8_t u8;
    typedef int8_t i8;


 //   template<typename T>
	//static std::string ToString(const T& t) { return boost::lexical_cast<std::string>(t); }

    typedef  __m128i block;
    inline block toBlock(u8*data) { return _mm_set_epi64x(((u64*)data)[1], ((u64*)data)[0]);}
    inline block toBlock(u64 low_u64)        { return _mm_set_epi64x(0, low_u64); }
    inline block toBlock(u64 high_u64, u64 low_u64) { return _mm_set_epi64x(high_u64, low_u64); }

    extern const block ZeroBlock;
    extern const block OneBlock;
    extern const block AllOneBlock;
    extern const block CCBlock;
    extern const std::array<block, 2> zeroAndAllOne;

    inline u64 roundUpTo(u64 val, u64 step) { return ((val + step - 1) / step) * step; }

    u64 log2ceil(u64);
    u64 log2floor(u64);

    block sysRandomSeed();
}


std::ostream& operator<<(std::ostream& out, const osuCrypto::block& block);
namespace osuCrypto
{
	using ::operator<<;
}

inline bool eq(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
    osuCrypto::block neq = _mm_xor_si128(lhs, rhs);
    return _mm_test_all_zeros(neq, neq) != 0;
}

inline bool neq(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
    osuCrypto::block neq = _mm_xor_si128(lhs, rhs);
    return _mm_test_all_zeros(neq, neq) == 0;
}

#ifdef _MSC_VER
inline bool operator<(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
    return lhs.m128i_u64[1] < rhs.m128i_u64[1] || (eq(lhs, rhs) && lhs.m128i_u64[0] < rhs.m128i_u64[0]);
}

inline osuCrypto::block operator^(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
	return _mm_xor_si128(lhs, rhs);
}
inline osuCrypto::block operator&(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
	return _mm_and_si128(lhs, rhs);
}

inline osuCrypto::block operator|(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
	return _mm_or_si128(lhs, rhs);
}
inline osuCrypto::block operator<<(const osuCrypto::block& lhs, const osuCrypto::u8& rhs)
{
	return _mm_slli_epi64(lhs, rhs);
}
inline osuCrypto::block operator>>(const osuCrypto::block& lhs, const osuCrypto::u8& rhs)
{
	return _mm_srli_epi64(lhs, rhs);
}
inline osuCrypto::block operator+(const osuCrypto::block& lhs, const osuCrypto::block& rhs)
{
	return _mm_add_epi64(lhs, rhs);
}


#ifdef ENABLE_RELIC
#pragma comment(lib, "relic_s.lib")
#endif

#ifdef ENABLE_MIRACL
#pragma comment(lib, "miracl.lib")
#endif

#ifdef ENABLE_WOLFSSL
#pragma comment(lib, "wolfssl.lib")
#endif

#endif

namespace oc = osuCrypto;
