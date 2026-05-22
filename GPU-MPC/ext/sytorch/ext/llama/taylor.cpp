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

#include "taylor.h"

std::pair<PrivateScaleKeyPack, PrivateScaleKeyPack> keyGenPrivateScale(int bin, int bout, GroupElement rin, GroupElement rout)
{
    std::pair<PrivateScaleKeyPack, PrivateScaleKeyPack> keys;
    auto rinSplit = splitShare(rin, bin);
    keys.first.rin = rinSplit.first;
    keys.second.rin = rinSplit.second;
    auto routSplit = splitShare(rout, bout);
    keys.first.rout = routSplit.first;
    keys.second.rout = routSplit.second;
    return keys;
}

GroupElement evalPrivateScale(int party, int bin, int bout, GroupElement x, const PrivateScaleKeyPack &key, uint64_t scalar)
{
    // bin == bout
    GroupElement res;
    if (party == 1)
    {
        res = scalar * (x - key.rin) + key.rout;
    }
    else
    {
        res = scalar * (-key.rin) + key.rout;
    }
    mod(res, bout);
    return res;
}

std::pair<TaylorSqKey, TaylorSqKey> keyGenTaylorSq(int bin, int bout, GroupElement rin, GroupElement rout)
{
    std::pair<TaylorSqKey, TaylorSqKey> keys;
    GroupElement a(rout + rin * rin);
    mod(a, bout);
    GroupElement b(2 * rin);
    mod(b, bout);
    auto apair = splitShare(a, bout);
    auto bpair = splitShare(b, bout);
    keys.first.a = apair.first;
    keys.first.b = bpair.first;
    keys.second.a = apair.second;
    keys.second.b = bpair.second;
    return keys;
}

GroupElement evalTaylorSq(int party, int bin, int bout, GroupElement x, const TaylorSqKey &key)
{
    GroupElement sum = key.a - key.b * x;
    if (party == 1)
    {
        sum = sum + x * x;
    }
    mod(sum, bout);
    return sum;
}

std::pair<TaylorKeyPack, TaylorKeyPack> keyGenTaylor(int bin, int bout, double a, double b, double c, GroupElement rin, GroupElement rout, int sf, int logk)
{
    // bin == bout only
    std::pair<TaylorKeyPack, TaylorKeyPack> keys;
    std::pair<MSNZBKeyPack, MSNZBKeyPack> msnzbKeys = keyGenMSNZB(bin, bout, rin, 0, sf, sf + logk);
    keys.first.msnzbKey = msnzbKeys.first;
    keys.second.msnzbKey = msnzbKeys.second;
    GroupElement routSquare = random_ge(bin);
    // GroupElement routSquare = rout;
    std::pair<TaylorSqKey, TaylorSqKey> squareKeys = keyGenTaylorSq(bin, bout, rin, routSquare);
    keys.first.squareKey = squareKeys.first;
    keys.second.squareKey = squareKeys.second;

    std::pair<BulkyLRSKeyPack, BulkyLRSKeyPack> lrsKeys[2];
    uint64_t scales[logk + 1];
    for (int i = 0; i < logk + 1; ++i)
    {
        scales[i] = sf + i;
    }

    GroupElement routASquare = random_ge(bin);
    lrsKeys[0] = keyGenBulkyLRS(bin, bout, logk + 1, scales, routSquare, routASquare);
    keys.first.lrsKeys[0] = lrsKeys[0].first;
    keys.second.lrsKeys[0] = lrsKeys[0].second;

    GroupElement routBscale = random_ge(bin);
    auto privateScaleKeys = keyGenPrivateScale(bin, bout, rin, routBscale);
    keys.first.privateScaleKey = privateScaleKeys.first;
    keys.second.privateScaleKey = privateScaleKeys.second;

    for (int i = 0; i < logk + 1; ++i)
    {
        scales[i] = sf + 3 * i;
    }
    lrsKeys[1] = keyGenBulkyLRS(bin, bout, logk + 1, scales, routASquare + routBscale, rout);
    keys.first.lrsKeys[1] = lrsKeys[1].first;
    keys.second.lrsKeys[1] = lrsKeys[1].second;

    return keys;
}

std::pair<GroupElement, GroupElement> evalTaylor_round1(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk)
{
    // bin == bout only
    GroupElement alpha = evalMSNZB(party, bin, bout, x, key.msnzbKey, sf, sf + logk);
    GroupElement square = evalTaylorSq(party, bin, bout, x, key.squareKey);
    return std::make_pair(alpha, square);
}

inline uint64_t flt2fx(double a, int pow)
{
    if (a > 0)
    {
        return (uint64_t)(a * (1ULL << pow));
    }
    else
    {
        return -(uint64_t)((-a) * (1ULL << pow));
    }
}

std::pair<GroupElement, GroupElement> evalTaylor_round2(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement square)
{
    // bin == bout only
    uint64_t pow = alpha + 1;
    uint64_t scalarA = flt2fx(a, pow);
    uint64_t scalarB = flt2fx(b, pow);

    uint64_t scales[logk + 1];
    for (int i = 0; i < logk + 1; ++i)
    {
        scales[i] = sf + i;
    }
    GroupElement ax2trunc = evalBulkyLRS(party, bin, bout, logk + 1, scales, square, key.lrsKeys[0], pow, scalarA);
    GroupElement bx = evalPrivateScale(party, bin, bout, x, key.privateScaleKey, scalarB);

    return std::make_pair(ax2trunc, bx);
}

GroupElement evalTaylor_round3(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement square, GroupElement ax2bx, uint64_t scalar)
{
    // bin == bout only
    uint64_t pow = alpha + 1;
    uint64_t scalarC = flt2fx(c, 2 * pow);
    mod(scalarC, bin);
    uint64_t scales[logk + 1];
    for (int i = 0; i < logk + 1; ++i)
    {
        scales[i] = sf + 3 * i;
    }
    GroupElement ax2bxctrunc = evalBulkyLRS(party, bin, bout, logk + 1, scales, ax2bx + scalarC, key.lrsKeys[1], pow + 2 * (pow - sf), scalar);

    return ax2bxctrunc;
}
