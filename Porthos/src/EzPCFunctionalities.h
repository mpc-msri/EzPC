/*

Authors: Nishant Kumar, Mayank Rathee.

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

/***************** Functions for different layers in NN *********************/
void MatMul2D(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<porthosSecretType> >& A, 
		vector< vector<porthosSecretType> >& B, 
		vector< vector<porthosSecretType> >& C, 
		bool modelIsA);

void ArgMax(int32_t s1, 
		int32_t s2, 
		vector< vector<porthosSecretType> >& inArr, 
		vector<porthosSecretType>& outArr);

void Relu(int32_t size, 
		vector<porthosSecretType>& inArr, 
		vector<porthosSecretType>& outArr,
		int32_t sf,
		bool doTruncation);

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
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

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
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void ElemWiseSecretSharedVectorMult(int32_t size, vector<porthosSecretType>& arr1, vector<porthosSecretType>& arr2, vector<porthosSecretType>& outputArr);
void ElemWiseActModelVectorMult(int32_t size, vector<porthosSecretType>& arr1, vector<porthosSecretType>& arr2, vector<porthosSecretType>& outputArr);
void ElemWiseVectorPublicDiv(int32_t size, vector<porthosSecretType>& arr1, int32_t divisor, vector<porthosSecretType>& outputArr);


void ScaleUp(int32_t s1, vector<porthosSecretType>& arr, int32_t sf);
void ScaleDown(int32_t s1, vector<porthosSecretType>& arr, int32_t sf);

void Conv2D(int32_t N, int32_t H, int32_t W, int32_t CI, 
				int32_t FH, int32_t FW, int32_t CO, 
				int32_t zPadHLeft, int32_t zPadHRight, 
				int32_t zPadWLeft, int32_t zPadWRight, 
				int32_t strideH, int32_t strideW, 
				vector< vector< vector< vector<porthosSecretType> > > >& inputArr, 
				vector< vector< vector< vector<porthosSecretType> > > >& filterArr, 
				vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void Conv2DWrapper(int32_t N, int32_t H, int32_t W, int32_t CI, 
				int32_t FH, int32_t FW, int32_t CO, 
				int32_t zPadHLeft, int32_t zPadHRight, 
				int32_t zPadWLeft, int32_t zPadWRight, 
				int32_t strideH, int32_t strideW, 
				vector< vector< vector< vector<porthosSecretType> > > >& inputArr, 
				vector< vector< vector< vector<porthosSecretType> > > >& filterArr, 
				vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void Conv3D(int32_t N, int32_t D, int32_t H, int32_t W, int32_t CI,
			int32_t FD, int32_t FH, int32_t FW, int32_t CO,
			int32_t zPadDLeft, int32_t zPadDRight, 
			int32_t zPadHLeft, int32_t zPadHRight, 
			int32_t zPadWLeft, int32_t zPadWRight,
			int32_t strideD, int32_t strideH, int32_t strideW,
			vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
			vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
			vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);

void Conv3DWrapper(int32_t N, int32_t D, int32_t H, int32_t W, int32_t CI,
			int32_t FD, int32_t FH, int32_t FW, int32_t CO,
			int32_t zPadDLeft, int32_t zPadDRight, 
			int32_t zPadHLeft, int32_t zPadHRight, 
			int32_t zPadWLeft, int32_t zPadWRight,
			int32_t strideD, int32_t strideH, int32_t strideW,
			vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
			vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
			int32_t consSF,
			vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);

void ConvTranspose3D(int32_t N, int32_t DPrime, int32_t HPrime, int32_t WPrime, int32_t CI,
				int32_t FD, int32_t FH, int32_t FW, int32_t CO,
				int32_t D, int32_t H, int32_t W,
				int32_t zPadTrDLeft, int32_t zPadTrDRight,
				int32_t zPadTrHLeft, int32_t zPadTrHRight,
				int32_t zPadTrWLeft, int32_t zPadTrWRight,
				int32_t strideD, int32_t strideH, int32_t strideW,
				vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
				vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
				vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);

void ConvTranspose3DWrapper(int32_t N, int32_t DPrime, int32_t HPrime, int32_t WPrime, int32_t CI,
				int32_t FD, int32_t FH, int32_t FW, int32_t CO,
				int32_t D, int32_t H, int32_t W,
				int32_t zPadTrDLeft, int32_t zPadTrDRight,
				int32_t zPadTrHLeft, int32_t zPadTrHRight,
				int32_t zPadTrWLeft, int32_t zPadTrWRight,
				int32_t strideD, int32_t strideH, int32_t strideW,
				vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
				vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
				int32_t consSF,
				vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);

void ClearMemSecret1(int64_t s1, vector< porthosSecretType >& arr);
void ClearMemSecret2(int64_t s1, int64_t s2, vector< vector< porthosSecretType > >& arr);
void ClearMemSecret3(int64_t s1, int64_t s2, int64_t s3, vector< vector< vector< porthosSecretType > > >& arr);
void ClearMemSecret4(int64_t s1, int64_t s2, int64_t s3, int64_t s4, vector< vector< vector< vector< porthosSecretType > > > >& arr);
void ClearMemSecret5(int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, vector< vector< vector< vector< vector< porthosSecretType > > > > >& arr);

void ClearMemPublic(int64_t x);
void ClearMemPublic1(int64_t s1, vector< int32_t >& arr);
void ClearMemPublic2(int64_t s1, int64_t s2, vector< vector< int32_t > >& arr);
void ClearMemPublic3(int64_t s1, int64_t s2, int64_t s3, vector< vector< vector< int32_t > > >& arr);
void ClearMemPublic4(int64_t s1, int64_t s2, int64_t s3, int64_t s4, vector< vector< vector< vector< int32_t > > > >& arr);
void ClearMemPublic5(int64_t s1, int64_t s2, int64_t s3, int64_t s4, int64_t s5, vector< vector< vector< vector< vector< int32_t > > > > >& arr);

void Floor(int32_t size, vector<porthosSecretType>& inArr, vector<porthosSecretType>& outArr, int32_t sf);


void StartComputation();
void EndComputation();

