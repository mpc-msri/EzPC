/*
Authors: Anwesh Bhattacharya
Copyright:
Copyright (c) 2021 Microsoft Research
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

#ifndef LIBRARY_FLOAT_H__
#define LIBRARY_FLOAT_H__

#include "defines_float.h"
#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"

using namespace std;
using namespace sci;

#define WHICHPARTY tid&1?3-__party:__party

// Packs
extern IOPack *__iopack;
extern OTPack *__otpack;

// Operations
extern BoolOp *__bool_op; // bool
extern FixOp *__fix_op;	  // int
extern FPOp *__fp_op;	  // float
extern FPMath *__fp_math; // float math operations

// Floating point descriptors
extern int __m_bits; // mantissa bits
extern int __e_bits; // exponent bits

// Handy globals ;
extern int BATCH;
extern int __sz1 ;
extern int __sz2 ;
extern int __sz3 ;
extern int __sz4 ;

// Output operations
extern BoolArray __bool_pub; // bool
extern FixArray __fix_pub;	 // int
extern FPArray __fp_pub;	 // float

// Measurement
extern time_point<high_resolution_clock> __start ;
extern uint64_t __initial_rounds ;
extern float __comm_start ;

/********************* Public Variables *********************/

// Initialization
void __init(int __argc, char **__argv);

// Ending
void __end() ;

float __get_comm();

/********************* Primitive Vectors *********************/

template <typename T>
vector<T> make_vector(size_t size) {
	return std::vector<T>(size);
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes) {
	auto inner = make_vector<T>(sizes...);
	return vector<decltype(inner)>(first, inner);
}

/********************* Boolean Multidimensional Arrays *********************/

BoolArray __public_bool_to_boolean(uint8_t b, int party);

vector<BoolArray> make_vector_bool(int party, size_t last);

template <typename... Args>
auto make_vector_bool(int party, size_t first, Args... sizes) {
	auto _inner = make_vector_bool(party, sizes...);
	vector<decltype(_inner)> _ret;
	_ret.push_back(_inner);
	for (size_t i = 1; i < first; i++) {
		_ret.push_back(make_vector_bool(party, sizes...));
	}
	return _ret;
}

BoolArray __rand_bool(int party);

vector<BoolArray> make_vector_bool_rand(int party, size_t last);

template <typename... Args>
auto make_vector_bool_rand(int party, size_t first, Args... sizes)
{
	auto _inner = make_vector_bool_rand(party, sizes...);
	vector<decltype(_inner)> _ret;
	_ret.push_back(_inner);
	for (size_t i = 1; i < first; i++) {
		_ret.push_back(make_vector_bool_rand(party, sizes...));
	}
	return _ret;
}

/********************* Integer Multidimensional Arrays *********************/

FixArray __public_int_to_arithmetic(uint64_t i, bool sign, int len, int party);

vector<FixArray> make_vector_int(int party, bool sign, int len, size_t last);

template <typename... Args>
auto make_vector_int(int party, bool sign, int len, size_t first, Args... sizes) {
	auto _inner = make_vector_int(party, sign, len, sizes...);
	vector<decltype(_inner)> _ret;
	_ret.push_back(_inner);
	for (size_t i = 1; i < first; i++) {
		_ret.push_back(make_vector_int(party, sign, len, sizes...));
	}
	return _ret;
}

/********************* Floating Multidimensional Arrays *********************/

FPArray __public_float_to_fp(float f, int party);

vector<FPArray> make_vector_float(int party, size_t last);

template <typename... Args>
auto make_vector_float(int party, size_t first, Args... sizes)
{
	auto _inner = make_vector_float(party, sizes...);
	vector<decltype(_inner)> _ret;
	_ret.push_back(_inner);
	for (size_t i = 1; i < first; i++)
	{
		_ret.push_back(make_vector_float(party, sizes...));
	}
	return _ret;
}

FPArray __rand_float(int party);

vector<FPArray> make_vector_float_rand(int party, size_t last);

template <typename... Args>
auto make_vector_float_rand(int party, size_t first, Args... sizes)
{
	auto _inner = make_vector_float_rand(party, sizes...);
	vector<decltype(_inner)> _ret;
	_ret.push_back(_inner);
	for (size_t i = 1; i < first; i++)
	{
		_ret.push_back(make_vector_float_rand(party, sizes...));
	}
	return _ret;
}

/********************* Extern functions *********************/

vector<int> get_chunks(int items, int slots);
tuple<BoolArray,BoolArray,FixArray,FixArray> get_components(int tid, const FPArray &x);

void Transpose(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr) ;

void ElemWiseAdd(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);

void ElemWiseSub(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);

void ElemWiseMul(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);

void ElemWiseDiv(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr) ;

void scalarMultiplication(int32_t s, float scalar, vector<FPArray> &inArr, vector<FPArray> &outArr) ;

void AllOneDividedBySizeArray(int32_t s, vector<FPArray> &inArr) ;

void getOutDer(int32_t s1, int32_t s2, vector<vector<FPArray>> &P, vector<vector<FPArray>> &Phat, vector<vector<FPArray>> &der);

void MatMul(int32_t m, int32_t n, int32_t p,
			vector<vector<FPArray>> &A,
			vector<vector<FPArray>> &B,
			vector<vector<FPArray>> &C);

void MatMul3(
	int32_t d1, int32_t d2, int32_t d3, int32_t d4, 
	vector<vector<vector<FPArray>>> &arr1, 
	vector<vector<vector<FPArray>>> &arr2, 
	vector<vector<vector<FPArray>>> &arr3);

// use flip as true if
void IfElse(
	int32_t s1,
	vector<FPArray> &inArr,
	vector<BoolArray> &condArr,
	vector<FPArray> &outArr, bool flip = false);

void GemmAdd(int32_t s1, int32_t s2,
			 vector<vector<FPArray>> &inArr,
			 vector<FPArray> &bias,
			 vector<vector<FPArray>> &outArr);

void GemmAdd3(int32_t s1, int32_t s2, int32_t s3,
	vector<vector<vector<FPArray>>> &inArr, 
	vector<FPArray> &bias, 
	vector<vector<vector<FPArray>>> &outArr);

// hotArr is positive if input is negative
static vector<BoolArray> DEFAULT_VECTOR;
void Relu(
	int32_t s1,
	vector<FPArray> &inArr,
	vector<FPArray> &outArr,
	vector<BoolArray> &hotArr = DEFAULT_VECTOR);

// hotArr is positive if input is negative
void Leaky_Relu(
	int32_t s1,
	float alpha,
	vector<FPArray> &inArr,
	vector<FPArray> &outArr,
	vector<BoolArray> &hotArr = DEFAULT_VECTOR);

void getBiasDer(int32_t m, int32_t s2, vector<vector<FPArray>> &batchSoftDer, vector<FPArray> &biasDer);

void updateWeights(int32_t sz, float lr, vector<FPArray> &inArr, vector<FPArray> &derArr);

void updateWeightsMomentum(int32_t sz, float lr, float beta, vector<FPArray>& inArr, vector<FPArray> &derArr, vector<FPArray> &momArr) ;

void updateWeightsAdam(int32_t sz, vector<FPArray>& t, float lr, float beta1, float beta2, float eps, vector<FPArray>& inArr, vector<FPArray>& derArr, vector<FPArray>& mArr, vector<FPArray>& vArr) ;

void Softmax2(
	int32_t s1,
	int32_t s2,
	vector<vector<FPArray>> &inArr,
	vector<vector<FPArray>> &outArr);

void Ln(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr);

void Sqrt(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr);

void Sigmoid(int32_t s1, vector<FPArray>& inArr, vector<FPArray>& outArr) ; 

void Tanh(int32_t s1, vector<FPArray>& inArr, vector<FPArray>& outArr) ; 

void dotProduct2(int32_t s1, int32_t s2, vector<vector<FPArray>> &arr1, vector<vector<FPArray>> &arr2, vector<FPArray> &outArr);

void SubtractOne(int32_t s1, vector<FPArray>& inArr, vector<FPArray>& outArr) ; 

void dotProduct2(int32_t s1, int32_t s2, vector<vector<FPArray>>& arr1, vector<vector<FPArray>>& arr2, vector<FPArray>& outArr) ;

void vectorSum2(int32_t s1, int32_t s2, vector<vector<FPArray>>& arr1, vector<FPArray>& outArr) ;

void vsumIfElse(int32_t s1, int32_t s2, vector<vector<FPArray>>& arr1, vector<vector<BoolArray>>& arr2, vector<FPArray>& outArr) ;

void getLoss(int32_t s, vector<FPArray> &arr, vector<FPArray> &outArr);
/************** Conv network functions **************/

// Forward functions

void Conv2DGroupWrapper(
	int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
	int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, 
	int32_t strideH, int32_t strideW, int32_t G, 
	vector<vector<vector<vector<FPArray>>>>& inputArr, 
	vector<vector<vector<vector<FPArray>>>>& filterArr, 
	vector<vector<vector<vector<FPArray>>>>& outArr) ;

void ConvAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4, 
	vector<vector<vector<vector<FPArray>>>>& inArr, 
	vector<FPArray>& biasArr, 
	vector<vector<vector<vector<FPArray>>>>& outArr) ;

void MaxPool(
	int32_t N, int32_t H, int32_t W, int32_t C, 
	int32_t ksizeH, int32_t ksizeW, 
	int32_t strideH, int32_t strideW,
	int32_t imgH, int32_t imgW,
	vector<vector<vector<vector<FPArray>>>>& inArr, 
	vector<vector<vector<vector<BoolArray>>>> &poolmask, 
	vector<vector<vector<vector<FPArray>>>>& outArr) ;

// Der arr comes in as FH, FW, CI, CO
// Der arr is filled as CO, CI, FH, FW
void ConvDerWrapper(
	int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
	int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t G, 
	vector<vector<vector<vector<FPArray>>>>& inputArr, 
	vector<vector<vector<vector<FPArray>>>>& filterArr, 
	vector<vector<vector<vector<FPArray>>>>& outArr) ;

void ConvBiasDer(
	int N, int W, int H, int chan, vector<vector<vector<vector<FPArray>>>> &der, vector<FPArray> &biasDer) ;

vector<FPMatrix> batched_matrix_multiplication(vector<FPMatrix> &x, vector<FPMatrix> &y);

// conv_weights comes in as as FH, FW, CI, CO
// conv_weights required as CO, CI, FH, FW
void GetPooledDer(
	int batch_size,
	int inH, int inW, int inC,
	int outC, int outH, int outW,
	int filterH, int filterW,
	vector<vector<vector<vector<FPArray>>>> &conv_weights, 
	vector<vector<vector<vector<FPArray>>>> &outDer, 
	vector<vector<vector<vector<FPArray>>>> &inDer) ;

void PoolProp(
	int32_t BATCH, int32_t outc, int32_t img2, int32_t imgp, int32_t img1, 
	int32_t pk, int32_t ps,
	vector<vector<vector<vector<FPArray>>>> &PooledDer, 
	vector<vector<vector<vector<BoolArray>>>> &Pool, 
	vector<vector<vector<vector<FPArray>>>> &ActDer, 
	bool flip) ;

// Backward functions

void computeMSELoss(int32_t m, int32_t s, vector<vector<FPArray>> &target, vector<vector<FPArray>> &fwdOut, vector<FPArray> &loss);


// GPT

void Gelu(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr) ;


#endif