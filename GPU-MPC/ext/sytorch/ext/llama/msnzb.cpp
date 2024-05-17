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

#include "msnzb.h"

std::pair<MSNZBKeyPack, MSNZBKeyPack> keyGenMSNZB(int bin, int bout, GroupElement rin, GroupElement rout, int start, int end)
{
    if (end == -1) end = (bin - 1);
    
    std::pair<MSNZBKeyPack, MSNZBKeyPack> keys;
    int m = end - start + 1;
    uint64_t p[m], q[m];

    for(uint64_t i = start; i <= end; ++i)
    {
        p[i - start] = (1ULL << i);
        q[i - start] = (1ULL << (i + 1)) - 1;
    }

    auto micKeys = keyGenMIC(bin, bout, m, p, q, rin, nullptr);
    keys.first.micKey = micKeys.first;
    keys.second.micKey = micKeys.second;
    auto rpair = splitShare(rout, bout);
    keys.first.r = rpair.first;
    keys.second.r = rpair.second;
    
    return keys;
}

GroupElement evalMSNZB(int party, int bin, int bout, GroupElement x, const MSNZBKeyPack &key, int start, int end, GroupElement *zcache)
{
    if (end == -1) end = (bin - 1);
    
    int m = end - start + 1;
    uint64_t p[m], q[m];
    GroupElement z[m];
    // for(int i = 0; i < m; ++i) z[m].bitsize = bout;

    for(uint64_t i = start; i <= end; ++i)
    {
        p[i - start] = (1ULL << i);
        q[i - start] = (1ULL << (i + 1)) - 1;
    }

    evalMIC(party, bin, bout, m, p, q, x, key.micKey, z);

    if (zcache != nullptr) {
        for(int i = 0; i < m; ++i) zcache[i] = z[i];
    }

    GroupElement sum = key.r;
    for(int i = start; i <= end; ++i)
    {
        sum = sum + i * z[i - start];
    }
    mod(sum, bout);
    return sum;
}
