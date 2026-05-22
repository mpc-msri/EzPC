/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <cryptoTools/Common/Defines.h>
#include <cryptoTools/Crypto/PRNG.h>

extern int32_t bitlength;
extern osuCrypto::PRNG prng;
extern osuCrypto::PRNG prngShared;

struct GroupElement {
    int bitsize = bitlength;
    uint64_t value;
    GroupElement(uint64_t value = 0, int bitsize = bitlength)
    {
        this->value = value;
        this->bitsize = bitsize;
        if (bitsize != 64)
            this->value = this->value % (uint64_t(1) << bitsize);
    }

    GroupElement(const GroupElement& other)
    {
        this->value = other.value;
        this->bitsize = other.bitsize;
    }

    uint8_t operator[](int index)
    {
        // a[0] gives msb, a[bitsize-1] gives lsb
        return (uint8_t)(value >> (bitsize - 1 - index)) & 1;
    }
};

inline void mod(GroupElement &a)
{
    if (a.bitsize != 64)
        a.value = a.value & ((uint64_t(1) << a.bitsize) - 1); 
}

inline GroupElement operator+(const GroupElement& a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = a.value + b.value;
    mod(c);
    return c;
}

inline GroupElement operator+(const GroupElement& a, const uint64_t& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = a.value + b;
    mod(c);
    return c;
}

inline GroupElement operator+(const uint64_t& a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = b.bitsize;
    c.value = (a + b.value);
    mod(c);
    return c;
}

inline GroupElement operator*(const GroupElement& a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value * b.value);
    mod(c);
    return c;
}

inline GroupElement operator*(const GroupElement& a, const uint64_t& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value * b);
    mod(c);
    return c;
}

inline GroupElement operator*(const uint64_t& a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = b.bitsize;
    c.value = (a * b.value);
    mod(c);
    return c;
}
inline GroupElement operator-(const GroupElement& a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value - b.value);
    mod(c);
    return c;
}

inline GroupElement operator-(const GroupElement& a, const uint64_t& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value - b);
    mod(c);
    return c;
}

inline GroupElement operator-(const uint64_t a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = b.bitsize;
    c.value = (a - b.value);
    mod(c);
    return c;
}

inline GroupElement operator-(const GroupElement& a)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = -a.value;
    mod(c);
    return c;
}

inline GroupElement operator/(const GroupElement& a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value / b.value);
    mod(c);
    return c;
}

inline GroupElement operator/(const GroupElement& a, const uint64_t& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value / b);
    mod(c);
    return c;
}

inline GroupElement operator%(const GroupElement& a, const GroupElement& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value % b.value);
    mod(c);
    return c;
}

inline bool operator==(const GroupElement &a, const GroupElement &b)
{
    return (a.value == b.value);
}
inline bool operator!=(const GroupElement &a, const GroupElement &b)
{
    return (a.value != b.value);
}

inline bool operator<(const GroupElement &a, const GroupElement &b)
{
    return (a.value < b.value);
}

inline bool operator>(const GroupElement &a, const GroupElement &b)
{
    return (a.value > b.value);
}

inline bool operator<=(const GroupElement &a, const GroupElement &b)
{
    return (a.value <= b.value);
}

inline bool operator>=(const GroupElement &a, const GroupElement &b)
{
    return (a.value >= b.value);
}

inline std::pair<GroupElement, GroupElement> splitShare(const GroupElement& a)
{
    GroupElement a1, a2;
    a1.bitsize = a.bitsize;
    a2.bitsize = a.bitsize;
    a1.value = rand() % (1 << a.bitsize);
    // a1.value = 0;
    mod(a1);
    a2.value = (a.value - a1.value);
    mod(a2);
    return std::make_pair(a1, a2);
}

inline std::pair<GroupElement, GroupElement> splitShareCommonPRNG(const GroupElement& a)
{
    GroupElement a1, a2;
    a1.bitsize = a.bitsize;
    a2.bitsize = a.bitsize;
    a1.value = prngShared.get<uint64_t>();
    // a1.value = 0;
    mod(a1);
    a2.value = (a.value - a1.value);
    mod(a2);
    return std::make_pair(a1, a2);
}

// inline std::pair<uint64_t, uint64_t> splitshare(uint64_t a, int bw)
// {
//     uint64_t a1, a2;
//     a1 = 0;//rand() & ((1 << bw) - 1);
//     a2 = (a - a1) & ((1 << bw) - 1);
//     return std::make_pair(a1, a2);
// }

inline GroupElement pow(GroupElement x, uint64_t e)
{
    if (e == 0)
    {
        return GroupElement(1, x.bitsize);
    }
    GroupElement res = pow(x, e / 2);
    if (e % 2 == 0)
    {
        return res * res;
    }
    else
    {
        return res * res * x;
    }
}

inline GroupElement random_ge(int bitlength)
{
    GroupElement a;
    a.bitsize = bitlength;
    a.value = prng.get<uint64_t>();
    mod(a);
    return a;
}

inline std::istream &operator>>(std::istream &is, GroupElement &a) {
    is >> a.value;
    mod(a);
    return is;
}

inline std::ostream &operator<<(std::ostream &os, const GroupElement &a) {
    if (a.bitsize == 64) {
        os << (int64_t)a.value;
    }
    else {
        uint64_t m = (1L << a.bitsize) - 1;
        int64_t v = (a.value + (1L << (a.bitsize - 1))) & m;
        os << v - (1L << (a.bitsize - 1));
        // os << (a.value & m);
    }
    return os;
}

inline GroupElement operator<<(const GroupElement& a, const int& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value << b);
    mod(c);
    return c;
}

inline GroupElement operator>>(const GroupElement& a, const int& b)
{
    GroupElement c;
    c.bitsize = a.bitsize;
    c.value = (a.value >> b);
    mod(c);
    return c;
}

