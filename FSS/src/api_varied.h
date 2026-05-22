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

#include <chrono>
#include <iostream>
#include <string>
#include <vector>


#define MASK_PAIR(x) x, x##_mask

void initialize();

void finalize();

void internalTruncateAndFix(int size, int shift, int bin, int bout, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), bool doReconstruct = true);

void internalExtend(int size, int bin, int bout, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), bool doReconstruct = true);

void AdjustScaleShr(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), int32_t I, int32_t J, int32_t bwA,
                    int32_t scale);

void AdjustScaleShr(int32_t I, int32_t J, int32_t scale, int64_t bwA,
                    MASK_PAIR(GroupElement *A));

// void MatAdd(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), int32_t I, int32_t J,
//             int32_t bwA, int32_t bwB, int32_t bwC, int32_t bwTemp, int32_t shrA,
//             int32_t shrB, int32_t shrC, int32_t demote,
//             bool subroutine = false);

void MatAddBroadCast(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), int32_t I,
                     int32_t J, int32_t bwA, int32_t bwB, int32_t bwC,
                     int32_t bwTemp, int32_t shrA, int32_t shrB, int32_t shrC,
                     int32_t demote, bool scalar_A = true);

void AddOrSubCir(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), int32_t I, int32_t J,
                 int32_t bwA, int32_t bwB, int32_t bwC, int32_t bwTemp,
                 int32_t shrA, int32_t shrB, int32_t shrC, bool add,
                 int32_t demote);

void AddOrSubCir4D(int32_t N, int32_t H, int32_t W, int32_t C, int32_t shrA,
                   int32_t shrB, int32_t shrC, bool add, int32_t demote,
                   int32_t bwA, int32_t bwB, int32_t bwTemp, int32_t bwC,
                   MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *X));

void Exp(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), int32_t I, int32_t J, int32_t bwA,
         int32_t bwB, int32_t sA, int32_t sB);

void Div(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), int32_t I, int32_t J,
         int32_t bwA, int32_t bwB, int32_t bwC, int32_t sA, int32_t sB,
         int32_t sC);

void ArgMax(MASK_PAIR(GroupElement *A), int32_t I, int32_t J, int32_t bwA, int32_t bw_index,
            MASK_PAIR(GroupElement *index));

void MaxPool2D(MASK_PAIR(GroupElement *A), int32_t I, int32_t J, int32_t bwA, int32_t bwB,
               MASK_PAIR(GroupElement *B));

void MaxPool2D(int I, int J, int bwA, int bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B));

void Convolution(int32_t N, int32_t H, int32_t W, int32_t CIN, int32_t HF,
                 int32_t WF, int32_t CINF, int32_t COUTF, int32_t HOUT,
                 int32_t WOUT, int32_t HPADL, int32_t HPADR, int32_t WPADL,
                 int32_t WPADR, int32_t HSTR, int32_t WSTR, int32_t HDL,
                 int32_t WDL, int32_t G, int32_t bwA, int32_t bwB, int32_t bwC,
                 int32_t bwTemp, int32_t shrA, int32_t shrB, int32_t H1,
                 int32_t H2, int32_t demote, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B),
                 MASK_PAIR(GroupElement *C));

void Convolution(int32_t N, int32_t H, int32_t W, int32_t CIN, int32_t HF,
                 int32_t WF, int32_t CINF, int32_t COUTF, int32_t HOUT,
                 int32_t WOUT, int32_t HPADL, int32_t HPADR, int32_t WPADL,
                 int32_t WPADR, int32_t HSTR, int32_t WSTR, int32_t HDL,
                 int32_t WDL, int32_t G, int32_t shrA, int32_t shrB, int32_t H1,
                 int32_t H2, int32_t demote, int32_t bwA, int32_t bwB,
                 int32_t bwTemp, int32_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B),
                 MASK_PAIR(GroupElement *C), MASK_PAIR(GroupElement *tmp), bool verbose = true);

void ReLU(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), int32_t I, int32_t J, int32_t bwA,
          int32_t bwB, uint64_t six, int32_t div);
void Relu6(int32_t N, int32_t H, int32_t W, int32_t C, int64_t six, int32_t div,
           int32_t bwA, int32_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B));

void BNorm(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *BNW), MASK_PAIR(GroupElement *BNB), MASK_PAIR(GroupElement *B), int32_t I,
           int32_t J, int32_t bwA, int32_t bwBNW, int32_t bwBNB, int32_t bwTemp,
           int32_t bwB, int32_t shA, int32_t shBNB, int32_t shB);

void NormaliseL2(MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), int32_t I, int32_t J, int32_t bwA,
                 int32_t scaleA, int32_t shrA);
void BNorm(int32_t I, int32_t J, int32_t shA, int32_t shBNB, int32_t shB,
           int32_t bwA, int32_t bwBNW, int32_t bwBNB, int32_t bwTemp,
           int32_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *BNW), MASK_PAIR(GroupElement *BNB), MASK_PAIR(GroupElement *B));

void NormaliseL2(int32_t N, int32_t H, int32_t W, int32_t C, int32_t scaleA,
                 int32_t shrA, int32_t bwA, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B));

void MBConv(int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Ct,
            int32_t HF, int32_t WF, int32_t Cout, int32_t Hout, int32_t Wout,
            int32_t HPADL, int32_t HPADR, int32_t WPADL, int32_t WPADR,
            int32_t HSTR, int32_t WSTR, int32_t D1, int32_t D2, int32_t D3,
            int64_t SIX_1, int64_t SIX_2, int32_t shr1, int32_t shr2,
            int32_t shr3, int32_t shr4, int32_t shr5, int32_t shr6,
            int32_t shr7, int32_t shr8, int32_t shr9, int32_t shl1,
            int32_t shl2, int32_t shl3, int32_t shl4, int32_t shl5,
            int32_t shl6, int32_t shl7, int32_t shl8, int32_t shl9, int32_t bwA,
            int32_t bwF1, int32_t bwB1W, int32_t bwB1B, int32_t bwF2,
            int32_t bwB2W, int32_t bwB2B, int32_t bwF3, int32_t bwB3W,
            int32_t bwB3B, int32_t bwC, int32_t bwX, int32_t bwT, int32_t bwU,
            int32_t bwUB1W, int32_t bwUB2W, int32_t bwUB3W, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *F1), MASK_PAIR(GroupElement *BN1W), MASK_PAIR(GroupElement *BN1B), MASK_PAIR(GroupElement *F2),
            MASK_PAIR(GroupElement *BN2W), MASK_PAIR(GroupElement *BN2B), MASK_PAIR(GroupElement *F3), MASK_PAIR(GroupElement *BN3W),
            MASK_PAIR(GroupElement *BN3B), MASK_PAIR(GroupElement *C), MASK_PAIR(GroupElement *X), MASK_PAIR(GroupElement *T), MASK_PAIR(GroupElement *U));

/*
    MatAdd fucntion for EzPC compatibility followed by MatAdd
    function developed using SeeDot.
    Kanav: needed?
*/
void MatAdd(int64_t I, int64_t J, int64_t shrA, int64_t shrB, int64_t shrC,
            int64_t demote, int64_t bwA, int64_t bwB, int64_t bwTemp,
            int64_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C),
            bool verbose = true);

void MatAdd4(int32_t N, int32_t H, int32_t W, int32_t C, int32_t shrA,
             int32_t shrB, int32_t shrC, int32_t demote, int32_t bwA,
             int32_t bwB, int32_t bwTemp, int32_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *X));

void MatSub(int64_t I, int64_t J, int64_t shrA, int64_t shrB, int64_t shrC,
            int64_t demote, int64_t bwA, int64_t bwB, int64_t bwTemp,
            int64_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C));

// Multiplication Layers

void MulCir(int64_t I, int64_t J, int64_t shrA, int64_t shrB, int64_t demote,
            int64_t bwA, int64_t bwB, int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C));

void MatMulUniform(int bw, int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C));

void MatMul(int64_t I, int64_t K, int64_t J, int64_t shrA, int64_t shrB,
            int64_t H1, int64_t H2, int64_t demote, int32_t bwA, int32_t bwB,
            int32_t bwTemp, int32_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C),
            MASK_PAIR(GroupElement *tmp), bool verbose = true);

void ScalarMul(int64_t I, int64_t J, int64_t shrA, int64_t shrB, int64_t demote,
               int64_t bwA, int64_t bwB, int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement A),
               MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C));

// Math layers

void Sigmoid(int64_t I, int64_t J, int64_t scale_in, int64_t scale_out,
             int64_t bwA, int64_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B));

void TanH(int64_t I, int64_t J, int64_t scale_in, int64_t scale_out,
          int64_t bwA, int64_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B));

void Sqrt(int64_t I, int64_t J, int64_t scale_in, int64_t scale_out,
          int64_t bwA, int64_t bwB, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement *B));

void AdjustScaleShl(int64_t I, int64_t J, int64_t scale, MASK_PAIR(GroupElement *A));

void ArgMax(int64_t I, int64_t J, int32_t bwA, int32_t bw_index, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *index));

void MatAddBroadCastA(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement A), MASK_PAIR(GroupElement *B),
                      MASK_PAIR(GroupElement *C));

void MatSubBroadCastA(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement A), MASK_PAIR(GroupElement *B),
                      MASK_PAIR(GroupElement *C));

void MatAddBroadCastB(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement B),
                      MASK_PAIR(GroupElement *C));

void MatSubBroadCastB(int64_t I, int64_t J, int64_t shrA, int64_t shrB,
                      int64_t shrC, int64_t demote, int64_t bwA, int64_t bwB,
                      int64_t bwTemp, int64_t bwC, MASK_PAIR(GroupElement *A), MASK_PAIR(GroupElement B),
                      MASK_PAIR(GroupElement *C));