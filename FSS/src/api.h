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
#include "comms.h"
#include "api_varied.h"
#include <assert.h>
#include <cassert>

#define MASK_PAIR(x) x, x##_mask

extern int32_t bitlength;

inline void ClearMemSecret1(int32_t s1, MASK_PAIR(GroupElement *arr)) { 
    delete[] arr;
    delete[] arr_mask;
}

inline void ClearMemSecret2(int32_t s1, int32_t s2, MASK_PAIR(GroupElement *arr)) {
    delete[] arr;
    delete[] arr_mask;
}

inline void ClearMemSecret3(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *arr)) {
    delete[] arr;
    delete[] arr_mask;
}

inline void ClearMemSecret4(int32_t s1, int32_t s2, int32_t s3, int32_t s4,
                            MASK_PAIR(GroupElement *arr)) {
    delete[] arr;
    delete[] arr_mask;
}

inline void ClearMemSecret5(int32_t s1, int32_t s2, int32_t s3, int32_t s4,
                            int32_t s5, MASK_PAIR(GroupElement *arr)) {
    delete[] arr;
    delete[] arr_mask;
}

inline void ClearMemPublic1(int32_t s1, int32_t *arr) {
    delete[] arr;
}

inline void ClearMemPublic2(int32_t s1, int32_t s2, int32_t *arr) {
    delete[] arr;
}

inline void ClearMemPublic3(int32_t s1, int32_t s2, int32_t s3, int32_t *arr) {
    delete[] arr;
}

inline void ClearMemPublic4(int32_t s1, int32_t s2, int32_t s3, int32_t s4,
                            int32_t *arr) {
    delete[] arr;
}

inline void ClearMemPublic5(int32_t s1, int32_t s2, int32_t s3, int32_t s4,
                            int32_t s5, int32_t *arr) {
    delete[] arr;
}

// QUESTION: is it okay to remove const qualifier?
// void MatMul2D(int32_t s1, int32_t s2, int32_t s3, const intType *A,
//               const intType *B, intType *C, bool modelIsA);

void StartComputation();
void EndComputation();

void MatMul2D(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA);

void Conv2DWrapper(int32_t N, int32_t H, int32_t W,
                   int32_t CI, int32_t FH, int32_t FW,
                   int32_t CO, int32_t zPadHLeft,
                   int32_t zPadHRight, int32_t zPadWLeft,
                   int32_t zPadWRight, int32_t strideH,
                   int32_t strideW, MASK_PAIR(GroupElement *inputArr), MASK_PAIR(GroupElement *filterArr),
                   MASK_PAIR(GroupElement *outArr));

void Conv2DGroupWrapper(int64_t N, int64_t H, int64_t W,
                        int64_t CI, int64_t FH, int64_t FW,
                        int64_t CO, int64_t zPadHLeft,
                        int64_t zPadHRight, int64_t zPadWLeft,
                        int64_t zPadWRight, int64_t strideH,
                        int64_t strideW, int64_t G,
                        MASK_PAIR(GroupElement *inputArr), MASK_PAIR(GroupElement *filterArr), MASK_PAIR(GroupElement *outArr));

void ElemWiseActModelVectorMult(int32_t size, MASK_PAIR(GroupElement *inArr),
                                MASK_PAIR(GroupElement *multArrVec), MASK_PAIR(GroupElement *outputArr));

void ArgMax(int32_t s1, int32_t s2, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr));

void Relu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int sf,
          bool doTruncation);

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr));

void AvgPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr));

void ScaleDown(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf);

void ScaleUp(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf);

void ElemWiseVectorPublicDiv(int32_t s1, MASK_PAIR(GroupElement *arr1), int32_t divisor,
                             MASK_PAIR(GroupElement *outArr));

void ElemWiseSecretSharedVectorMult(int32_t size, MASK_PAIR(GroupElement *inArr),
                                    MASK_PAIR(GroupElement *multArrVec), MASK_PAIR(GroupElement *outputArr));

void Floor(int32_t s1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t sf);

inline GroupElement funcSSCons(uint64_t val) {
    GroupElement g(val, 64);
    // if (party == DEALER)
    //     g.value = 0;
    return g;
}

extern int32_t numRounds;
void reconstruct(int32_t size, GroupElement *arr, int bw);


inline void assert_failed(const char* file, int line, const char* function, const char* expression) {
    std::cout << "Assertion failed: " << expression << " in " << function << " at " << file << ":" << line << std::endl;
    exit(1);
}

#define always_assert(expr) (static_cast <bool> (expr) ? void (0) : assert_failed (__FILE__, __LINE__, __PRETTY_FUNCTION__, #expr))
