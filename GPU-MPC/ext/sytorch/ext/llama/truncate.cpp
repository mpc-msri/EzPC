// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "truncate.h"

std::pair<BulkyLRSKeyPack, BulkyLRSKeyPack> keyGenBulkyLRS(int bin, int bout, int m, uint64_t *scales, GroupElement rin, GroupElement rout)
{
    std::pair<BulkyLRSKeyPack, BulkyLRSKeyPack> keys;
    auto dcfN = keyGenDCF(bin, bout, rin, 1);
    keys.first.dcfKeyN = dcfN.first;
    keys.second.dcfKeyN = dcfN.second;

    keys.first.dcfKeyS = new DCFKeyPack[m];
    keys.second.dcfKeyS = new DCFKeyPack[m];
    keys.first.z = new GroupElement[m];
    keys.second.z = new GroupElement[m];

    for(int i = 0; i < m; ++i)
    {
        uint64_t s = scales[i];
        GroupElement r1, r0;
        if (rin % (1ULL << s) == 0) {
            r1 = (rin >> s);
            r0 = 0;
        }
        else {
            r1 = (rin >> s) + 1;
            r0 = (r1 << s) - rin;
        }
        mod(r0, s);
        mod(r1, bin);
        auto dcfS = keyGenDCF(s,   bout, r0,  1);
        keys.first.dcfKeyS[i] = dcfS.first;
        keys.second.dcfKeyS[i] = dcfS.second;
        GroupElement zTruncate = -r1;
        mod(zTruncate, bout);
        auto zTruncateSplit = splitShare(zTruncate, bout);
        keys.first.z[i] = zTruncateSplit.first;
        keys.second.z[i] = zTruncateSplit.second;
    }
    auto routSplit = splitShare(rout, bout);
    keys.first.out = routSplit.first;
    keys.second.out = routSplit.second;

    return keys;
}

inline void assert_failed(const char* file, int line, const char* function, const char* expression) {
    std::cout << "Assertion failed: " << expression << " in " << function << " at " << file << ":" << line << std::endl;
    exit(1);
}

#define always_assert(expr) (static_cast <bool> (expr) ? void (0) : assert_failed (__FILE__, __LINE__, __PRETTY_FUNCTION__, #expr))

GroupElement evalBulkyLRS(int party, int bin, int bout, int m, uint64_t *scales, GroupElement x, const BulkyLRSKeyPack &key, int s, uint64_t scalar)
{
    int idx = -1;
    for(int i = 0; i < m; ++i)
    {
        if(scales[i] == s)
        {
            idx = i;
            break;
        }
    }
    always_assert(idx != -1);

    GroupElement tn;
    evalDCF(party, &tn, x, key.dcfKeyN);

    GroupElement x0;
    GroupElement x1;
    x0 = x % (1ULL << s);
    x1 = x >> s;
    mod(x0, s);
    mod(x1, bin);
    GroupElement xs;
    xs = (1ULL<<s) - 1 - x0;
    GroupElement ts;
    evalDCF(party, &ts, xs, key.dcfKeyS[idx]);
    GroupElement res;
    if (party == 1) {
        res = scalar * (x1 + key.z[idx] + (1ULL<<(bin - s)) * tn + ts) + key.out;
    }
    else {
        res = scalar * (key.z[idx] + (1ULL<<(bin - s)) * tn + ts) + key.out;
    }
    mod(res, bout);
    return res;
}