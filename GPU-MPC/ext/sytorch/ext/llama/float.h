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

#pragma once

#include <utility>
#include "mic.h"
#include "select.h"

void fill_pq(GroupElement *p, GroupElement *q, int n);
template <typename T>
using pair = std::pair<T, T>;

pair<FixToFloatKeyPack> keyGenFixToFloat(int bin, int scale, GroupElement rin, GroupElement *p, GroupElement *q);
void evalFixToFloat_1(int party, int bin, int scale, GroupElement x, const FixToFloatKeyPack &key, GroupElement *p, GroupElement *q, GroupElement &m, GroupElement &e, GroupElement &z, GroupElement &s, GroupElement &pow, GroupElement &sm);

GroupElement adjust(GroupElement m, GroupElement e);
pair<FloatToFixKeyPack> keyGenFloatToFix(int bin, int scale, GroupElement rout);

inline GroupElement pow_helper(int scale,GroupElement y) {
    GroupElement pow = 0;
    if (y < scale)
    {
        pow = (1ULL << y);        
    }
    return pow;

}