/*
Authors: Mayank Rathee, Sameer Wagh, Nishant Kumar.
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
#include "../utils_sgx_port/utils_print_sgx.h"
#include "sgx_trts.h"
#include "../Enclave/Enclave_t.h"
using namespace std;

extern void start_time();
extern void start_communication();
extern void end_time(string str);
extern void end_communication(string str);

extern bool final_argmax;

/******************************** Functionalities 2PC ********************************/ 
void funcTruncate2PC(vector<aramisSecretType> &a, 
		size_t power, 
		size_t size);

void funcXORModuloOdd2PC(vector<smallType> &bit, 
		vector<aramisSecretType> &shares,
		vector<aramisSecretType> &output, 
		size_t size); 

void funcReconstruct2PC(const vector<aramisSecretType> &a, 
		size_t size, 
		string str, 
		vector<aramisSecretType>* b=NULL, 
		int revealToParties = 2);

void funcReconstructBit2PC(const vector<smallType> &a, 
		size_t size, 
		string str);

void funcConditionalSet2PC(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b, 
		vector<smallType> &c, 
		vector<aramisSecretType> &u, 
		vector<aramisSecretType> &v, 
		size_t size);

/******************************** Functionalities MPC ********************************/
void funcSecretShareConstant(const vector<aramisSecretType> &cons, 
		vector<aramisSecretType> &curShare, 
		size_t size);

/******************************** Main Functionalities MPC ********************************/

void funcMatMulMPC(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b, 
		bool areInputsScaled=true, 
		int coming_from_conv = 0);

void funcDotProductMPC(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t size, 
		bool bench_cross_bridge_calls = false);

void funcPrivateCompareMPC(const vector<smallType> &share_m, 
		const vector<aramisSecretType> &r, 
		const vector<smallType> &beta, 
		vector<smallType> &betaPrime, 
		size_t size, 
		size_t dim);

void funcShareConvertMPC(vector<aramisSecretType> &a, 
		size_t size);

void funcComputeMSB3PC(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		size_t size);

void funcSelectShares3PC(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c, 
		size_t size);

void funcRELUPrime3PC(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		size_t size);

void funcRELUMPC(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		size_t size);

void funcDivisionMPC(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b, 
		vector<aramisSecretType> &quotient, 
		size_t size);

void funcMaxMPC(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &max, 
		vector<aramisSecretType> &maxIndex, 
		size_t rows, 
		size_t columns, 
		bool calculate_max_idx = false);

void funcMaxIndexMPC(vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &maxIndex, 
		size_t rows, 
		size_t columns);

/******************* Wrapper integer function calls ********************/
aramisSecretType funcMult(aramisSecretType a, 
		aramisSecretType b);

aramisSecretType funcReluPrime(aramisSecretType a);

aramisSecretType funcDiv(aramisSecretType a, 
		aramisSecretType b);

aramisSecretType funcSSCons(aramisSecretType a);

aramisSecretType funcReconstruct2PCCons(aramisSecretType a, 
		int revealToParties);

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
		vector< vector< vector< vector<aramisSecretType> > > >& inputArr,
		vector< vector< vector< vector<aramisSecretType> > > >& filterArr,
		int64_t consSF,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr);

void funcConv2DCSFSplit(int32_t N, 
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
