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

#include <llama/group_element.h>

#define MASK_PAIR(x) x, x##_mask

namespace llama {
    void start();
    void end();
}

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

void Conv3DWrapper(int32_t N, int32_t D, int32_t H, int32_t W,
            int32_t CI, int32_t FD, int32_t FH, int32_t FW,
            int32_t CO, int32_t zPadDLeft, int32_t zPadDRight, int32_t zPadHLeft,
            int32_t zPadHRight, int32_t zPadWLeft,
            int32_t zPadWRight, int32_t strideD, int32_t strideH,
            int32_t strideW, GroupElement *inputArr, GroupElement *filterArr,
            GroupElement *outArr);

void ElemWiseActModelVectorMult(int32_t size, MASK_PAIR(GroupElement *inArr),
                                MASK_PAIR(GroupElement *multArrVec), MASK_PAIR(GroupElement *outputArr));

void ArgMax(int32_t s1, int32_t s2, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr));

void Relu(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu, std::string prefix = "");

void ReluTruncate(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int sf, GroupElement *drelu_cache);

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot, std::string prefix = "");

void AvgPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH,
             int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr));

void ScaleDown(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf);

void ScaleUp(int32_t size, MASK_PAIR(GroupElement *inArr), int32_t sf);

void ElemWiseMul(int32_t size, GroupElement *inArr, GroupElement *multArrVec, GroupElement *outputArr, std::string prefix = "");

void Floor(int32_t s1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t sf);

void PiranhaSoftmax(int32_t s1, int32_t s2, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t sf);

void ARS(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), int32_t shift);

void Select(int32_t size, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix = "", bool doReconstruct = true);
void Select(int32_t size, int bin, GroupElement *s, GroupElement *x, GroupElement *out, std::string prefix = "", bool doReconstruct = true);

void Relu2Round(int32_t size, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *drelu_cache, int effectiveInputBw);

void MaxPoolDouble(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot);

void MaxPoolOneHot(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH, int32_t FW, GroupElement *maxBits, GroupElement *oneHot);

void MaxPoolBackward(int32_t N, int32_t H, int32_t W, int32_t C, int32_t FH,
             int32_t FW, int32_t zPadHLeft, int32_t zPadHRight,
             int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH,
             int32_t strideW, int32_t N1, int32_t imgH, int32_t imgW,
             int32_t C1, MASK_PAIR(GroupElement *inArr), MASK_PAIR(GroupElement *outArr), GroupElement *oneHot);

void FixToFloat(int size, GroupElement *inp, GroupElement *out, int scale);
void FloatToFix(int size, GroupElement *inp, GroupElement *out, int scale);

void ReluExtend(int size, int bin, int bout, GroupElement *x, GroupElement *y, GroupElement *drelu);
void SignExtend2(int size, int bin, int bout, GroupElement *x, GroupElement *y);

void ConvTranspose3DWrapper(int64_t N, 
    int64_t D, 
    int64_t H, 
    int64_t W, 
    int64_t CI, 
    int64_t FD, 
    int64_t FH, 
    int64_t FW, 
    int64_t CO, 
    int64_t zPadDLeft, 
    int64_t zPadDRight, 
    int64_t zPadHLeft, 
    int64_t zPadHRight, 
    int64_t zPadWLeft, 
    int64_t zPadWRight, 
    int64_t strideD, 
    int64_t strideH, 
    int64_t strideW, 
    int64_t outD, 
    int64_t outH, 
    int64_t outW, 
    GroupElement* inputArr, 
    GroupElement* filterArr, 
    GroupElement* outArr);

void EdabitsPrTrunc(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");

void LUT_dpf(int size, int bin, int bout, const std::vector<GroupElement> &tab, GroupElement *x, GroupElement *y, std::string prefix = "", bool doReconstruct = true);

void nExp(int size, int bin, GroupElement *x, GroupElement *y, int scale);
void Tanh(int size, GroupElement *x, GroupElement *y, int scale);

void Clip(int size, int maxbw, GroupElement *x, GroupElement *y, std::string prefix = "");
void Softmax(int32_t s1, int32_t s2, int bin, GroupElement *x, GroupElement *y, int32_t scale);
void F2BF16(int size, GroupElement *x, GroupElement *y, std::string prefix = "");
void Rsqrt(int size, GroupElement *x, GroupElement *y, GroupElement extradiv, int scale, std::string prefix = "", std::vector<GroupElement>* lut = nullptr);
void Gelu(int size, int bin, GroupElement *x, GroupElement *y, int scale);
void TruncateReduce(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");
void LUT_dfpet(int size, int bin, int bout, const std::vector<GroupElement> &tab, GroupElement *x, GroupElement *y, std::string prefix, bool doReconstruct = true);
void SlothDrelu(int size, int bin, GroupElement *x, GroupElement *y, std::string prefix = "");
void SlothRelu(int size, int bin, GroupElement *x, GroupElement *y, std::string prefix = "");
void SlothClip(int size, int bin, int maxbw, int bout, GroupElement *x, GroupElement *y, std::string prefix = "");
void SlothMaxpool(int s1, int s2, int bin, GroupElement *x, GroupElement *y, std::string prefix = "");
void SlothMaxpoolTriangular(int s1, int s2, int bin, GroupElement *x, GroupElement *y, std::string prefix = "");
void SumOfSquare(int s1, int s2, GroupElement *x, GroupElement *y, std::string prefix = "");
void SlothLayerNorm(int s1, int s2, GroupElement *x, GroupElement *A, GroupElement *B, GroupElement *y, int scale);
void SlothRMSNorm(int s1, int s2, GroupElement *x, GroupElement *A, GroupElement *B, GroupElement *y, int scale);
void SlothGemm(int s1, int s2, int s3, GroupElement *x, GroupElement *A, GroupElement *y, int scale);
void SoftmaxTriangular(int32_t s1, int32_t s2, int bin, GroupElement *x, GroupElement *y, int32_t scale);
void MatMul2DTriangular(int32_t s1, int32_t s2, int32_t s3, MASK_PAIR(GroupElement *A),
            MASK_PAIR(GroupElement *B), MASK_PAIR(GroupElement *C), bool modelIsA);
void SlothAttentionTriangular(int n_seq, int n_embd, int n_heads, GroupElement *q, GroupElement *k, GroupElement *v, GroupElement *out, int scale);
void SlothLRS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");
void SlothARS(int size, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");
void SlothTR(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");
void SlothGelu(int size, int bin, GroupElement *x, GroupElement *out, int scale);
void SlothSilu(int size, int bin, GroupElement *x, GroupElement *out, int scale);
void SlothFaithfulARS(int size, int bin, GroupElement *x, GroupElement *y, int scale, std::string prefix = "");

void reconstruct(int32_t size, GroupElement *arr, int bw);
