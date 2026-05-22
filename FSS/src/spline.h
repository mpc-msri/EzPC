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
#include "keypack.h"
#include "config.h"

std::pair<ReluKeyPack, ReluKeyPack> keyGenRelu(int Bin, int Bout,
                        GroupElement rin, GroupElement rout);

GroupElement evalRelu(int party, GroupElement x, const ReluKeyPack &k);

std::pair<MaxpoolKeyPack, MaxpoolKeyPack> keyGenMaxpool(int Bin, int Bout, GroupElement rin1, GroupElement rin2, GroupElement rout);
GroupElement evalMaxpool(int party, GroupElement x, GroupElement y, const MaxpoolKeyPack &k);

std::pair<SplineKeyPack, SplineKeyPack> keyGenSigmoid_main_wrapper(int Bin, int Bout, int scaleIn, int scaleOut,
                    GroupElement rin, GroupElement rout);

GroupElement evalSigmoid_main_wrapper(int party, GroupElement x, SplineKeyPack &k);

std::pair<SplineKeyPack, SplineKeyPack> keyGenTanh_main_wrapper(int Bin, int Bout, int scaleIn, int scaleOut,
                    GroupElement rin, GroupElement rout);

GroupElement evalTanh_main_wrapper(int party, GroupElement x, SplineKeyPack &k);

// Note: for input bitlen 12, octave spline's ulp is calculated over bitlen 6 = 12/2, but truncate-reduce is called keeping in mind output bitlen 11

std::pair<SplineKeyPack, SplineKeyPack> keyGenInvsqrt_main_wrapper(int Bin, int Bout, int scaleIn, int scaleOut,
                    GroupElement rin, GroupElement rout);

GroupElement evalInvsqrt_main_wrapper(int party, GroupElement x, SplineKeyPack &k);

std::pair<SplineKeyPack, SplineKeyPack> keyGenSigmoid(int Bin, int Bout, int numPoly, int degree, 
                    std::vector<std::vector<GroupElement>> polynomials,
                    std::vector<GroupElement> p,
                    GroupElement rin, GroupElement rout);

GroupElement evalSigmoid(int party, GroupElement x, SplineKeyPack &k);
