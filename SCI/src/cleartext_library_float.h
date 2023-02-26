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

#ifndef LIBRARY_CLEARTEXT_FLOAT_H__
#define LIBRARY_CLEARTEXT_FLOAT_H__

#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <tuple>
#include <map>

using namespace std ;

template<typename T>
vector<T> make_vector(size_t size) {
return std::vector<T>(size) ;
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
auto inner = make_vector<T>(sizes...) ;
return vector<decltype(inner)>(first, inner) ;
}

static vector<bool> DEFAULT_VECTOR;
void ElemWiseAdd(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) ;
void ElemWiseSub(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) ;
void ElemWiseMul(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) ;
void ElemWiseDiv(int32_t s1, vector<float>& arr1, vector<float>& arr2, vector<float>& outArr) ;

float intToFloat(int32_t m);
void Softmax2(int32_t s1, int32_t s2, vector<vector<float>>& inArr, vector<vector<float>>& outArr);
void Ln(int32_t s1, vector<float>& inArr, vector<float>& outArr);
void getOutDer(int32_t s1, int32_t s2, vector<vector<float>>& batchSoft, vector<vector<float>>& lab, vector<vector<float>>& der);
void MatMul(int32_t s1, int32_t s2, int32_t s3, vector<vector<float>>& mat1, vector<vector<float>>& mat2, vector<vector<float>>& mat3);
void GemmAdd(int32_t s1, int32_t s2, vector<vector<float>>& prod, vector<float>& bias, vector<vector<float>>& out);
void dotProduct2(int32_t s1, int32_t s2, vector<vector<float>>& arr1, vector<vector<float>>& arr2, vector<float>& outArr);
void vsumIfElse(int32_t s1, int32_t s2, vector<vector<float>> &arr, vector<vector<bool>>& hotarr, vector<float>& outArr) ;
void Relu(int32_t s1, vector<float>& inArr, vector<float>& outArr, vector<bool>& hotArr);
void getBiasDer(int32_t s1, int32_t s2, vector<vector<float>>& der, vector<float>& biasDer);
void IfElse(int32_t s1, vector<float>& dat, vector<bool>& hot, vector<float>& out, bool flip);
void updateWeights(int32_t s, float lr, vector<float>& bias, vector<float>& der);
void updateWeightsAdam(int32_t sz, int32_t t, float lr, float beta1, float beta2, float eps, vector<float>& wt, vector<float>& der, vector<float>& m_t, vector<float>& v_t) ;
void updateWeightsMomentum(int32_t sz, float lr, float beta, vector<float>& inArr, vector<float> &derArr, vector<float> &momArr) ;
void getLoss(int32_t m, vector<float>& lossTerms, vector<float>& loss);
void computeMSELoss(int32_t m, int32_t s, vector<vector<float>>& target, vector<vector<float>>& fwdOut, vector<float>& loss);
void Tanh(int32_t s1, vector<float>& inArr, vector<float>& outArr) ;

/******** Conv Functions **********/

void Conv2DGroupWrapper(
	int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
	int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, 
	int32_t strideH, int32_t strideW, int32_t G, 
	vector<vector<vector<vector<float>>>>& inputArr, 
	vector<vector<vector<vector<float>>>>& filterArr, 
	vector<vector<vector<vector<float>>>>& outArr) ;

void ConvAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4, 
	vector<vector<vector<vector<float>>>>& inArr, 
	vector<float>& biasArr, 
	vector<vector<vector<vector<float>>>>& outArr) ;

void MaxPool(
	int32_t N, int32_t H, int32_t W, int32_t C, 
	int32_t ksizeH, int32_t ksizeW, 
	int32_t strideH, int32_t strideW,
	int32_t imgH, int32_t imgW,
	vector<vector<vector<vector<float>>>>& inArr, 
	vector<vector<vector<vector<bool>>>> &poolmask, 
	vector<vector<vector<vector<float>>>>& outArr) ;

void ConvDerWrapper(
	int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
	int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t G, 
	vector<vector<vector<vector<float>>>>& inputArr, 
	vector<vector<vector<vector<float>>>>& filterArr, 
	vector<vector<vector<vector<float>>>>& outArr) ;

void ConvBiasDer(
	int N, int W, int H, int chan, vector<vector<vector<vector<float>>>> &der, vector<float> &biasDer) ;

void GetPooledDer(
	int batch_size,
	int inH, int inW, int inC,
	int outC, int outH, int outW,
	int filterH, int filterW,
	vector<vector<vector<vector<float>>>> &conv_weights, 
	vector<vector<vector<vector<float>>>> &outDer, 
	vector<vector<vector<vector<float>>>> &inDer) ;

void PoolProp(
	int32_t BATCH, int32_t outc, int32_t img2, int32_t imgp, int32_t img1, 
	int32_t pk, int32_t ps,
	vector<vector<vector<vector<float>>>> PooledDer, 
	vector<vector<vector<vector<bool>>>> Pool, 
	vector<vector<vector<vector<float>>>> ActDer, 
	bool flip) ;

#endif