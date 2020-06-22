/*

Authors: Nishant Kumar, Mayank Rathee.

Copyright:
Copyright (c) 2018 Microsoft Research
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

/***************** Functions for different layers in NN *********************/
void MatMulCSF2D(int64_t i, 
		int64_t j, 
		int64_t k, 
		vector< vector<porthosSecretType> >& A, 
		vector< vector<porthosSecretType> >& B, 
		vector< vector<porthosSecretType> >& C, 
		int64_t consSF);

void ArgMax1(int64_t outArrS1, 
		int64_t inArrS1, 
		int64_t inArrS2, 
		vector< vector<porthosSecretType> >& inArr, 
		int64_t dim, vector<porthosSecretType>& outArr);

void ArgMax3(int64_t outs1, 
		int64_t outs2, 
		int64_t outs3, 
		int64_t ins1, 
		int64_t ins2, 
		int64_t ins3, 
		int64_t ins4,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		int64_t dim, 
		vector< vector< vector<porthosSecretType> > >& outArr);

void Relu2(int64_t s1, 
		int64_t s2, 
		vector< vector<porthosSecretType> >& inArr, 
		vector< vector<porthosSecretType> >& outArr);

void Relu4(int64_t s1, 
		int64_t s2, 
		int64_t s3, 
		int64_t s4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void Relu5(int64_t s1, 
		int64_t s2, 
		int64_t s3, 
		int64_t s4, 
		int64_t s5, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inArr, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);

void MaxPool(int64_t N, 
		int64_t H, 
		int64_t W, 
		int64_t C, 
		int64_t ksizeH, 
		int64_t ksizeW,
		int64_t zPadHLeft, 
		int64_t zPadHRight, 
		int64_t zPadWLeft, 
		int64_t zPadWRight,
		int64_t strideH, 
		int64_t strideW,
		int64_t N1, 
		int64_t imgH, 
		int64_t imgW, 
		int64_t C1,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void ElemWiseMul2(int64_t s1, 
		int64_t s2, 
		vector< vector<porthosSecretType> >& arr1, 
		vector< vector<porthosSecretType> >& arr2, 
		vector< vector<porthosSecretType> >& outArr, 
		int64_t shrout);

void ElemWiseMul4(int64_t s1, 
		int64_t s2, 
		int64_t s3, 
		int64_t s4, 
		vector< vector< vector< vector<porthosSecretType> > > >& arr1, 
		vector< vector< vector< vector<porthosSecretType> > > >& arr2, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr, 
		int64_t shrout);

void ElemWiseMul5(int64_t a1, 
		int64_t a2, 
		int64_t a3, 
		int64_t a4, 
		int64_t a5, 
                int64_t b1, 
		int64_t b2, 
		int64_t b3, 
		int64_t b4, 
		int64_t b5, 
                int64_t s1, 
		int64_t s2, 
		int64_t s3, 
		int64_t s4, 
		int64_t s5, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& arr1, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& arr2, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr, 
		int64_t shrout);

void AvgPool(int64_t N, 
		int64_t H, 
		int64_t W, 
		int64_t C, 
		int64_t ksizeH, 
		int64_t ksizeW,
		int64_t zPadHLeft, 
		int64_t zPadHRight, 
		int64_t zPadWLeft, 
		int64_t zPadWRight,
		int64_t strideH, 
		int64_t strideW,
		int64_t N1, 
		int64_t imgH, 
		int64_t imgW, 
		int64_t C1,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void FusedBatchNorm4411(int64_t s1, 
		int64_t s2, 
		int64_t s3, 
		int64_t s4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector<porthosSecretType>& multArr, 
		vector<porthosSecretType>& biasArr, 
		int64_t consSF,
		vector< vector< vector< vector<porthosSecretType> > > >& outputArr);

void ReduceMean24(int64_t outS1, 
		int64_t outS2, 
		int64_t inS1, int64_t inS2, int64_t inS3, int64_t inS4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inputArr,
		vector<porthosSecretType>& axes,
		vector< vector<porthosSecretType> >& outputArr);

void ClearMemSecret1(int64_t s1, vector< porthosSecretType >& arr);
void ClearMemSecret2(int64_t s1, int64_t s2, vector< vector< porthosSecretType > >& arr);
void ClearMemSecret3(int64_t s1, int64_t s2, int64_t s3, vector< vector< vector< porthosSecretType > > >& arr);
void ClearMemSecret4(int64_t s1, int64_t s2, int64_t s3, int64_t s4, vector< vector< vector< vector< porthosSecretType > > > >& arr);
void ClearMemSecret5(int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, vector< vector< vector< vector< vector< porthosSecretType > > > > >& arr);
void ClearMemPublic2(int64_t s1, int64_t s2, vector< vector< int32_t > >& arr);

void StartComputation();
void EndComputation();

