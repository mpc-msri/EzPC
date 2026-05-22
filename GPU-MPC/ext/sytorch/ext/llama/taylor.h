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
#include "msnzb.h"
#include "truncate.h"
#include <llama/keypack.h>

std::pair<PrivateScaleKeyPack, PrivateScaleKeyPack> keyGenPrivateScale(int bin, int bout, GroupElement rin, GroupElement rout);

GroupElement evalPrivateScale(int party, int bin, int bout, GroupElement x, const PrivateScaleKeyPack &key, uint64_t scalar);

std::pair<TaylorSqKey, TaylorSqKey> keyGenTaylorSq(int bin, int bout, GroupElement rin, GroupElement rout);

GroupElement evalTaylorSq(int party, int bin, int bout, GroupElement x, const TaylorSqKey &key);

std::pair<TaylorKeyPack, TaylorKeyPack> keyGenTaylor(int bin, int bout, double a, double b, double c, GroupElement rin, GroupElement rout, int sf, int logk);

std::pair<GroupElement, GroupElement> evalTaylor_round1(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk);

std::pair<GroupElement, GroupElement> evalTaylor_round2(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement square);

GroupElement evalTaylor_round3(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement square, GroupElement ax2bx, uint64_t scalar = 1);

GroupElement evalTaylor_round4(int party, int bin, int bout, double a, double b, double c, GroupElement x, const TaylorKeyPack &key, int sf, int logk, GroupElement alpha, GroupElement ax2bxctrunc);
