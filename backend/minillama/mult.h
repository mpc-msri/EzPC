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
#include "group_element.h"

std::pair<MultKey, MultKey> MultGen(GroupElement rin1, GroupElement rin2, GroupElement rout);
GroupElement MultEval(int party, const MultKey &k, const GroupElement &l, const GroupElement &r);
GroupElement mult_helper(uint8_t party, GroupElement x, GroupElement y, GroupElement x_mask, GroupElement y_mask);
std::pair<MultKeyNew, MultKeyNew> new_mult_unsigned_gen(int bw1, int bw2, uint64_t rin1, uint64_t rin2, uint64_t rout);
uint64_t new_mult_unsigned_eval(int party, int bw1, int bw2, const MultKeyNew &k, const uint64_t x, const uint64_t y);

std::pair<MultKeyNew, MultKeyNew> new_mult_signed_gen(int bw1, int bw2, uint64_t rin1, uint64_t rin2, uint64_t rout);
uint64_t new_mult_signed_eval(int party, int bw1, int bw2, const MultKeyNew &k, const uint64_t x, const uint64_t y);
