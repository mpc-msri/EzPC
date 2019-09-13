#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

/******************************** Functionalities 2PC ********************************/ 
void funcTruncate2PC(vector<porthosSecretType> &a, 
		size_t power, 
		size_t size);

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
		bool areInputsScaled = true);

void funcDotProductMPC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c, 
		size_t size, 
		uint32_t conSF = FLOAT_PRECISION);

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

/******************* Wrapper integer function calls ********************/
porthosSecretType funcMult(porthosSecretType a, 
		porthosSecretType b);

porthosSecretType funcReluPrime(porthosSecretType a);

porthosSecretType funcDiv(porthosSecretType a, 
		porthosSecretType b);

porthosSecretType funcSSCons(porthosSecretType a);

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
