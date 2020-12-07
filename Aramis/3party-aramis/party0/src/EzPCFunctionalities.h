/*
Authors: Mayank Rathee, Nishant Kumar.
Copyright:
Copyright (c) 2020 Microsoft Research
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
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include "Functionalities.h"
using namespace std;

void MatMulCSF2D(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<aramisSecretType> >& A, 
		vector< vector<aramisSecretType> >& B, 
		vector< vector<aramisSecretType> >& C, 
		int32_t consSF);

void ArgMax1(int32_t outArrS1, 
		int32_t inArrS1, 
		int32_t inArrS2, 
		vector< vector<aramisSecretType> >& inArr, 
		int32_t dim, 
		vector<aramisSecretType>& outArr);

void ArgMax3(int32_t outs1, 
		int32_t outs2, 
		int32_t outs3, 
		int32_t ins1, 
		int32_t ins2, 
		int32_t ins3, 
		int32_t ins4,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		int32_t dim, 
		vector< vector< vector<aramisSecretType> > >& outArr);

void Relu2(int32_t s1, 
		int32_t s2, 
		vector< vector<aramisSecretType> >& inArr, 
		vector< vector<aramisSecretType> >& outArr);

void Relu4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);

void MaxPool44(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C,	
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadHLeft, 
		int32_t zPadHRight,
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);

void MaxPool(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C, 
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadHLeft, 
		int32_t zPadHRight,
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);


void MaxPool44Hook(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C, 
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadH,
		int32_t zPadW, 
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);

void MaxPool4(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C, 
		int32_t imgH, 
		int32_t imgW,
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadHLeft, 
		int32_t zPadHRight,
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);


void ElemWiseMul2(int32_t s1, 
		int32_t s2, 
		vector< vector<aramisSecretType> >& arr1, 
		vector< vector<aramisSecretType> >& arr2, 
		vector< vector<aramisSecretType> >& outArr, 
		int32_t shrout);

void ElemWiseMul4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<aramisSecretType> > > >& arr1, 
		vector< vector< vector< vector<aramisSecretType> > > >& arr2, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr, 
		int32_t shrout);

void AvgPool44(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C, 
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadHLeft, 
		int32_t zPadHRight,
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);

void AvgPool(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C, 
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadHLeft, 
		int32_t zPadHRight,
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);

void AvgPool44Hook(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C, 
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadH,
		int32_t zPadW, 
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);

void ScalarMul4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		int64_t scalar, 
		vector< vector< vector< vector<aramisSecretType> > > >& inputArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outputArr, 
		int64_t consSF);

void ScalarMul2(int32_t s1, 
		int32_t s2,  
		int64_t scalar, 
		vector< vector<aramisSecretType> >& inputArr, 
		vector< vector<aramisSecretType> >& outputArr, 
		int64_t consSF);


void TempFusedBatchNorm4411(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		int32_t vecS1,
		vector<aramisSecretType>& multArr, 
		vector<aramisSecretType>& biasArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outputArr);

void FusedBatchNorm4411(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector<aramisSecretType>& multArr, 
		vector<aramisSecretType>& biasArr, 
		int32_t consSF,
		vector< vector< vector< vector<aramisSecretType> > > >& outputArr);

void Conv2DCSFNew(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW, 
		vector< vector< vector< vector<aramisSecretType> > > >& inputArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& filterArr, 
		int32_t consSF, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);


void ClearMemSecret1(int32_t s1, 
		vector<aramisSecretType>& arr);

void ClearMemSecret2(int32_t s1, 
		int32_t s2, 
		vector< vector< aramisSecretType > >& arr);

void ClearMemSecret3(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		vector< vector< vector< aramisSecretType > > >& arr);

void ClearMemSecret4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector< aramisSecretType > > > >& arr);

void ClearMemPublic2(int32_t s1, 
		int32_t s2, 
		vector< vector< int32_t > >& arr);

void StartComputation();

void EndComputation();
