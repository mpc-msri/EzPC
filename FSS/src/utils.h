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

template <typename T>
using matrix = std::vector<std::vector<T>>;

void MatAdd(int s1, int s2, GroupElement *A, GroupElement* B, GroupElement *C);

void MatAdd4(int s0, int s1, int s2, int s3, GroupElement* A, GroupElement* B, GroupElement* C);

void MatSub(int s1, int s2, GroupElement *A, GroupElement* B, GroupElement *C);

void MatSub4(int s0, int s1, int s2, int s3, GroupElement* A, GroupElement* B, GroupElement* C);

void MatMul(int s1, int s2, int s3, GroupElement *A, GroupElement* B, GroupElement *C);

void MatCopy(int s1, int s2, GroupElement *input, GroupElement *output);

// C = C - A*B
void MatSubMul(int s1, int s2, int s3, GroupElement *A, GroupElement* B, GroupElement *C);

// C = C + A*B
void MatAddMul(int s1, int s2, int s3, GroupElement *A, GroupElement* B, GroupElement *C);

void MatCopy4(int s1, int s2, int s3, int s4, GroupElement *input, GroupElement *output);

void MatFinalize4(int s1, int s2, int s3, int s4, GroupElement *input);

void Conv2DReshapeFilter(int FH, int FW, int CI, int CO, GroupElement* filter, GroupElement* reshapedFilter);

void Conv2DReshapeInput(int N, int H, int W, int CI, int FH, int FW, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, int strideH, int strideW, int RRows, int RCols, GroupElement *inputArr, GroupElement *outputArr);

void Conv2DReshapeOutput(int N, int finalH, int finalW, int CO, GroupElement *inputArr, GroupElement *outputArr);

void Conv2DPlaintext(int N, int H, int W, int CI, 
				   int FH, int FW, int CO, 
				   int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, 
				   int strideH, int strideW, 
				   GroupElement *inputArr, 
				   GroupElement * filterArr, 
				   GroupElement * outArr);

std::vector<GroupElement> generateOffsetPolynomial(int bitsize, const std::vector<GroupElement> &poly, GroupElement rin);

std::vector<GroupElement> generateOffsetPolynomial_bitsize_accurate(int bitsize, const std::vector<GroupElement> &poly, GroupElement rin);

GroupElement evalPoly(std::vector<GroupElement> poly, GroupElement inp);

GroupElement changeBitsize(GroupElement x, int newbitsize);

GroupElement signedDivide(GroupElement x, GroupElement y);

GroupElement signedMod(GroupElement x, GroupElement y);

GroupElement flt2fxd(uint64_t x, int scale, int inp_bitlen);

long double fxd2flt(GroupElement x, int scale, int inp_bitlen);

int64_t getSignedValue(GroupElement x);

void matmul_eval_helper(int dim1, int dim2, int dim3, GroupElement *A,
                            GroupElement *B, GroupElement *C, GroupElement *ka, GroupElement *kb, GroupElement *kc);
