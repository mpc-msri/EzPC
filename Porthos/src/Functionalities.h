/*

Authors: Sameer Wagh, Mayank Rathee, Nishant Kumar.

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
using namespace std;

/******************************** Functionalities 2PC ********************************/ 
void funcTruncate2PC(vector<porthosSecretType> &a, 
		size_t power, 
		size_t size);

void funcTruncate2PC(vector<vector<vector<vector<vector<porthosSecretType>>>>> &a,
                size_t power,
                size_t d1,
                size_t d2,
                size_t d3,
                size_t d4,
                size_t d5);

void funcXORModuloOdd2PC(vector<smallType> &bit, 
		vector<porthosSecretType> &shares,
		vector<porthosSecretType> &output, 
		size_t size); 

void funcReconstruct2PC(const vector<porthosSecretType> &a, 
		size_t size, 
		string str, 
		vector<porthosSecretType>* b=NULL, 
		int revealToParties = 2);

void funcReconstructBit2PC(const vector<smallType> &a, 
		size_t size, 
		string str);

void funcConditionalSet2PC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<smallType> &c, 
		vector<porthosSecretType> &u, 
		vector<porthosSecretType> &v, 
		size_t size);

/******************************** Functionalities MPC ********************************/
void funcSecretShareConstant(const vector<porthosSecretType> &cons, 
		vector<porthosSecretType> &curShare, 
		size_t size);

/******************************** Main Functionalities MPC ********************************/
void funcMatMulMPC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b,
		uint32_t consSF = FLOAT_PRECISION,	
		bool doTruncation = true);

void funcDotProductMPC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c, 
		size_t size, 
		uint32_t conSF = FLOAT_PRECISION,
		bool doTruncation = true);

void funcPrivateCompareMPC(const vector<smallType> &share_m, 
		const vector<porthosSecretType> &r, 
		const vector<smallType> &beta, 
		vector<smallType> &betaPrime, 
		size_t size, 
		size_t dim);

void funcShareConvertMPC(vector<porthosSecretType> &a, 
		size_t size);

void funcComputeMSB3PC(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &b, 
		size_t size);

void funcSelectShares3PC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c, 
		size_t size);

void funcRELUPrime3PC(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &b, 
		size_t size);

void funcRELUMPC(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &b, 
		size_t size);

void funcDivisionMPC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &quotient, 
		size_t size);

void funcMaxMPC(vector<porthosSecretType> &a, 
		vector<porthosSecretType> &max, 
		vector<porthosSecretType> &maxIndex, 
		size_t rows, 
		size_t columns, 
		bool computeMaxIndex = false);

void funcMaxIndexMPC(vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &maxIndex, 
		size_t rows, 
		size_t columns);

/*************************************** Convolution and related utility functions **************************/
void funcConv2DCSF(int32_t N, 
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
		vector< vector< vector< vector<porthosSecretType> > > >& inputArr,
		vector< vector< vector< vector<porthosSecretType> > > >& filterArr,
		int32_t consSF,
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void funcConv3DMPC(
		int32_t N,
		int32_t D,
		int32_t H,
		int32_t W,
		int32_t CI,
		int32_t FD,
		int32_t FH,
		int32_t FW,
		int32_t CO,
		int32_t zPadDLeft,
		int32_t zPadDRight,
		int32_t zPadHLeft,
		int32_t zPadHRight,
		int32_t zPadWLeft,
		int32_t zPadWRight,
		int32_t strideD,
		int32_t strideH,
		int32_t strideW,
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
		int32_t consSF,
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);

void ConvTranspose3DSliding(int32_t N, 
		int32_t DPrime, 
		int32_t HPrime, 
		int32_t WPrime, 
		int32_t CI, 
		int32_t FD, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t D, 
		int32_t H, 
		int32_t W, 
		int32_t zPadTrDLeft, 
		int32_t zPadTrDRight, 
		int32_t zPadTrHLeft, 
		int32_t zPadTrHRight, 
		int32_t zPadTrWLeft, 
		int32_t zPadTrWRight, 
		int32_t strideD, 
		int32_t strideH, 
		int32_t strideW, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);

void ConvTranspose3DCSFMPC(int32_t N, 
		int32_t DPrime, 
		int32_t HPrime, 
		int32_t WPrime, 
		int32_t CI, 
		int32_t FD, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t D, 
		int32_t H, 
		int32_t W, 
		int32_t zPadTrDLeft, 
		int32_t zPadTrDRight, 
		int32_t zPadTrHLeft, 
		int32_t zPadTrHRight, 
		int32_t zPadTrWLeft, 
		int32_t zPadTrWRight, 
		int32_t strideD, 
		int32_t strideH, 
		int32_t strideW, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr, 
		int32_t consSF, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr);




/******************* Wrapper integer function calls ********************/
porthosSecretType funcMult(porthosSecretType a, 
		porthosSecretType b);

porthosSecretType funcReluPrime(porthosSecretType a);

porthosSecretType funcDiv(porthosSecretType a, 
		porthosSecretType b);

porthosSecretType funcSSCons(int32_t x);
porthosSecretType funcSSCons(int64_t x);


porthosSecretType funcReconstruct2PCCons(porthosSecretType a, 
		int revealToParties);


/******************************** Debugging functions ********************************/
void debugDotProd();
void debugComputeMSB();
void debugPC();
void debugDivision();
void debugMax();
void debugSS();
void debugMatMul();
void debugReLUPrime();
void debugMaxIndex();

/******************************** Testing functions ********************************/
void testMatMul(size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t iter);

void testRelu(size_t r, 
		size_t c, 
		size_t iter);

void testReluPrime(size_t r, 
		size_t c, 
		size_t iter);

void testMaxPool(size_t p_range, 
		size_t q_range, 
		size_t px, 
		size_t py, 
		size_t D, 
		size_t iter);

void testMaxPoolDerivative(size_t p_range, 
		size_t q_range, 
		size_t px, 
		size_t py, 
		size_t D, 
		size_t iter);

void testComputeMSB(size_t size);

void testShareConvert(size_t size);

/***************************************** End **************************************/
