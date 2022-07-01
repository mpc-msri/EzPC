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

#include "utils.h"
#include "array.h"
#include "comms.h"
#include <assert.h>
#include <iostream>
#include <Eigen/Dense>
#include <math.h>

void MatAdd(int s1, int s2, GroupElement *A, GroupElement* B, GroupElement *C)
{
    for (int i = 0; i < s1; i++)
    {
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdxRowM(C, s1, s2, i, j).value = Arr2DIdxRowM(A, s1, s2, i, j).value + Arr2DIdxRowM(B, s1, s2, i, j).value;
        }
    }
}

void MatAdd4(int s0, int s1, int s2, int s3, GroupElement* A, GroupElement* B, GroupElement* C)
{
    for (int i = 0; i < s0; i++)
    {
        for (int j = 0; j < s1; j++)
        {
            for (int k = 0; k < s2; k++)
            {
                for (int l = 0; l < s3; l++)
                {
                    Arr4DIdxRowM(C, s0, s1, s2, s3, i, j, k, l).value = Arr4DIdxRowM(A, s0, s1, s2, s3, i, j, k, l).value + Arr4DIdxRowM(B, s0, s1, s2, s3, i, j, k, l).value;
                }
            }
        }
    }
}

void MatSub(int s1, int s2, GroupElement *A, GroupElement* B, GroupElement *C)
{
    for (int i = 0; i < s1; i++)
    {
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdxRowM(C, s1, s2, i, j).value = Arr2DIdxRowM(A, s1, s2, i, j).value - Arr2DIdxRowM(B, s1, s2, i, j).value;
        }
    }
}

void MatSub4(int s0, int s1, int s2, int s3, GroupElement* A, GroupElement* B, GroupElement* C)
{
    for (int i = 0; i < s0; i++)
    {
        for (int j = 0; j < s1; j++)
        {
            for (int k = 0; k < s2; k++)
            {
                for (int l = 0; l < s3; l++)
                {
                    Arr4DIdxRowM(C, s0, s1, s2, s3, i, j, k, l).value = Arr4DIdxRowM(A, s0, s1, s2, s3, i, j, k, l).value - Arr4DIdxRowM(B, s0, s1, s2, s3, i, j, k, l).value;
                }
            }
        }
    }
}

void matmul_cleartext_eigen(int dim1, int dim2, int dim3, GroupElement *inA,
                            GroupElement *inB, GroupElement *outC) {
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_A(dim1, dim2);
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_B(dim2, dim3);
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_C(dim1, dim3);

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      eigen_A(i, j) = Arr2DIdxRowM(inA, dim1, dim2, i, j).value;
    }
  }
  for (int i = 0; i < dim2; i++) {
    for (int j = 0; j < dim3; j++) {
      eigen_B(i, j) = Arr2DIdxRowM(inB, dim2, dim3, i, j).value;
    }
  }
  eigen_C = eigen_A * eigen_B;
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3; j++) {
      Arr2DIdxRowM(outC, dim1, dim3, i, j).value = eigen_C(i, j);
    }
  }
}

void MatMul(int s1, int s2, int s3, GroupElement *A, GroupElement* B, GroupElement *C)
{
    // for (int i = 0; i < s1; i++)
    // {
    //     for (int k = 0; k < s3; k++)
    //     {
    //         Arr2DIdxRowM(C, s1, s3, i, k).value = 0;
    //         for (int j = 0; j < s2; j++)
    //         {
    //             Arr2DIdxRowM(C, s1, s3, i, k).value = Arr2DIdxRowM(C, s1, s3, i, k).value + Arr2DIdxRowM(A, s1, s2, i, j).value * Arr2DIdxRowM(B, s2, s3, j, k).value;
    //         }
    //     }
    // }
    matmul_cleartext_eigen(s1, s2, s3, A, B, C);
}

void Conv2DReshapeFilter(int FH, int FW, int CI, int CO, GroupElement* filter, GroupElement* reshapedFilter)
{
    for(int co = 0; co < CO; co++){
        for(int fh = 0; fh < FH; fh++){
            for(int fw = 0; fw < FW; fw++){
                for(int ci = 0; ci < CI; ci++){
                    Arr2DIdxRowM(reshapedFilter, CO, FH*FW*CI, co, (fh*FW*CI) + (fw*CI) + ci).value = Arr4DIdxRowM(filter, FH, FW, CI, CO, fh, fw, ci, co).value;
                }
            }
        }
    }
}

void Conv2DReshapeInput(int N, int H, int W, int CI, int FH, int FW, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, int strideH, int strideW, int RRows, int RCols, GroupElement *inputArr, GroupElement *outputArr)
{
    int linIdxFilterMult = 0;
	for (int n = 0; n < N; n++){
		int leftTopCornerH = 0 - zPadHLeft;
		int extremeRightBottomCornerH = H - 1 + zPadHRight;
		while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
			int leftTopCornerW = 0 - zPadWLeft;
			int extremeRightBottomCornerW = W - 1 + zPadWRight;
			while((leftTopCornerW + FW - 1) <= extremeRightBottomCornerW){

				for (int fh = 0; fh < FH; fh++){
					for (int fw = 0; fw < FW; fw++){
						int curPosH = leftTopCornerH + fh;
						int curPosW = leftTopCornerW + fw;
						for (int ci = 0; ci < CI; ci++){
							if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))){
								Arr2DIdxRowM(outputArr, RRows, RCols,(fh*FW*CI) + (fw*CI) + ci, linIdxFilterMult).value = 0L;
							}
							else{
								Arr2DIdxRowM(outputArr, RRows, RCols,(fh*FW*CI) + (fw*CI) + ci, linIdxFilterMult).value = Arr4DIdxRowM(inputArr, N, H, W, CI, n, curPosH, curPosW, ci).value;
							}
						}
					}
				}

				linIdxFilterMult = linIdxFilterMult + 1;
				leftTopCornerW = leftTopCornerW + strideW;
			}

			leftTopCornerH = leftTopCornerH + strideH;
		}
	}
}

void Conv2DReshapeOutput(int N, int finalH, int finalW, int CO, GroupElement *inputArr, GroupElement *outputArr)
{
    for (int co = 0; co < CO; ++co){
		for (int n = 0; n < N; ++n){
			for(int h = 0; h < finalH; ++h){
				for (int w = 0; w < finalW; ++w){
					Arr4DIdxRowM(outputArr, N, finalH, finalW, CO, n, h, w, co).value = Arr2DIdxRowM(inputArr, CO, N*finalH*finalW, co, (n*finalH*finalW) + (h*finalW) + w).value;
				}
			}
		}
	}
}



void PrintMatrix(matrix<GroupElement> matrix)
{
    for(int i=0; i<matrix.size(); i++){
        for(int j=0; j<matrix[0].size(); j++){
            std::cout << matrix[i][j].value << " ";
        }
        std::cout << std::endl;
    }
}

void Conv2DPlaintext(int N, int H, int W, int CI, 
				   int FH, int FW, int CO, 
				   int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, 
				   int strideH, int strideW, 
				   GroupElement *inputArr, 
				   GroupElement * filterArr, 
				   GroupElement * outArr)
{
    int reshapedFilterRows = CO;
	int reshapedFilterCols = FH*FW*CI;
	int reshapedIPRows = FH*FW*CI;
	int newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
	int newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);
	int reshapedIPCols = N * newH * newW;

    GroupElement *filterReshaped = make_array<GroupElement>(reshapedFilterRows, reshapedFilterCols);
	GroupElement *inputReshaped = make_array<GroupElement>(reshapedIPRows, reshapedIPCols);
	GroupElement *matmulOP = make_array<GroupElement>(reshapedFilterRows, reshapedIPCols);
    
    Conv2DReshapeInput(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
    Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, filterReshaped);
    MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP);
    Conv2DReshapeOutput(N, newH, newW, CO, matmulOP, outArr);

    delete[] filterReshaped;
    delete[] inputReshaped;
    delete[] matmulOP;

}

void VecCopy(int s, GroupElement *input, GroupElement *output)
{
    for(int i = 0; i < s; i++){
        output[i].value = input[i].value;
    }
}

void MatCopy(int s1, int s2, GroupElement *input, GroupElement *output){
    VecCopy(s1*s2, input, output);
}

// C = C - A*B
void MatSubMul(int s1, int s2, int s3, GroupElement *A, GroupElement* B, GroupElement *C)
{
    for (int i = 0; i < s1; i++)
    {
        for (int k = 0; k < s3; k++)
        {
            for (int j = 0; j < s2; j++)
            {
                Arr2DIdxRowM(C, s1, s3, i, k).value = Arr2DIdxRowM(C, s1, s3, i, k).value - Arr2DIdxRowM(A, s1, s2, i, j).value * Arr2DIdxRowM(B, s2, s3, j, k).value;
            }
        }
    }
}

// C = C + A*B
void MatAddMul(int s1, int s2, int s3, GroupElement *A, GroupElement* B, GroupElement *C)
{
    for (int i = 0; i < s1; i++)
    {
        for (int k = 0; k < s3; k++)
        {
            for (int j = 0; j < s2; j++)
            {
                Arr2DIdxRowM(C, s1, s3, i, k).value = Arr2DIdxRowM(C, s1, s3, i, k).value + Arr2DIdxRowM(A, s1, s2, i, j).value * Arr2DIdxRowM(B, s2, s3, j, k).value;
            }
        }
    }
}

void MatCopy4(int s1, int s2, int s3, int s4, GroupElement *input, GroupElement *output){
    for(int i = 0; i < s1; i++)
    {
        for(int j = 0; j < s2; j++)
        {
            for(int k = 0; k < s3; k++)
            {
                for(int l = 0; l < s4; l++)
                {
                    Arr4DIdxRowM(output, s1, s2, s3, s4, i, j, k, l).value = Arr4DIdxRowM(input, s1, s2, s3, s4, i, j, k, l).value;
                }
            }
        }
    }
}

void MatFinalize4(int s1, int s2, int s3, int s4, GroupElement *input)
{
    for(int i = 0; i < s1; i++)
    {
        for(int j = 0; j < s2; j++)
        {
            for(int k = 0; k < s3; k++)
            {
                for(int l = 0; l < s4; l++)
                {
                    mod(Arr4DIdxRowM(input, s1, s2, s3, s4, i, j, k, l));
                }
            }
        }
    }
}

std::vector<GroupElement> generateOffsetPolynomial(int bitsize, const std::vector<GroupElement> &poly, GroupElement rin)
{
    // given input coeffs of poly(x), output coeffs of poly(x - rin)
    int n = poly.size() - 1;

    // make sure rin has correct bitsize
    rin = GroupElement(rin.value, bitsize);

    GroupElement binomials[n + 1][n + 1];
    for (int i = 0; i < n + 1; ++i) {
        for(int j = 0; j < n + 1; ++j) {
            binomials[i][j] = 0;
        }
    }
    binomials[0][0] = GroupElement(1, bitsize);
    for (int i = 1; i <= n; ++i)
    {
        binomials[i][0] = GroupElement(1, bitsize);
        for (int j = 1; j < i; ++j)
        {
            // calculate iCj
            binomials[i][j] = binomials[i - 1][j - 1] + binomials[i - 1][j];
        }
        binomials[i][i] = GroupElement(1, bitsize);
    }

    std::vector<GroupElement> coeffs(n + 1);
    for (int i = n; i >= 0; --i)
    {
        GroupElement val = GroupElement(0, bitsize);
        for (int k = i; k <= n; ++k)
        {
            GroupElement t = binomials[k][i] * pow(rin, k - i) * poly[n - k];
            if ((k - i) % 2 == 1)
            {
                t = -t;
            }
            val = val + t;
        }
        coeffs[n - i] = val;
    }
    return coeffs;
}

std::vector<GroupElement> generateOffsetPolynomial_bitsize_accurate(int final_bitsize, const std::vector<GroupElement> &poly, GroupElement rin)
{

    /* for the special case where poly coefficients are set to bitsize just sufficient 
    final_bitsize is usually 64*/

    // given input coeffs of poly(x), output coeffs of poly(x - rin)
    int n = poly.size() - 1;

    // make sure rin has largest bitsize
    auto rin_upscaled = changeBitsize(rin, final_bitsize);

    GroupElement binomials[n + 1][n + 1];
    for (int i = 0; i < n + 1; ++i) {
        for(int j = 0; j < n + 1; ++j) {
            binomials[i][j] = 0;
        }
    }
    binomials[0][0] = GroupElement(1, final_bitsize);
    for (int i = 1; i <= n; ++i)
    {
        binomials[i][0] = GroupElement(1, final_bitsize);
        for (int j = 1; j < i; ++j)
        {
            // calculate iCj
            binomials[i][j] = binomials[i - 1][j - 1] + binomials[i - 1][j];
        }
        binomials[i][i] = GroupElement(1, final_bitsize);
    }

    std::vector<GroupElement> coeffs(n + 1);
    for (int i = n; i >= 0; --i)
    {
        // val will represent coeffs[n-i] so it should have same bitsize as poly[n-i]
        int cur_bitsize = poly[n - i].bitsize;
        GroupElement val = GroupElement(0, cur_bitsize);
        for (int k = i; k <= n; ++k)
        {
            // first do mult in 64 bits, then set to correct bitsize
            GroupElement t = binomials[k][i] * pow(rin, k - i) * changeBitsize(poly[n - k], cur_bitsize);
            if ((k - i) % 2 == 1)
            {
                t = -t;
            }
            t = changeBitsize(t, cur_bitsize);
            val = val + t;
        }
        coeffs[n - i] = val;
    }
    return coeffs;
}

GroupElement evalPoly(std::vector<GroupElement> poly, GroupElement inp)
{
    int inpB = inp.bitsize, coefB = poly[0].bitsize, degree = poly.size() - 1;
    int B = inpB;
    if (inpB != coefB) B = coefB;
    GroupElement res(0, B);
    GroupElement curr(1, B);
    for (int i = degree; i >= 0; --i)
    {
        res = res + curr * poly[i];
        curr = curr * inp;
    }
    return res;
}

GroupElement changeBitsize(GroupElement x, int newbitsize) {
    int oldbitsize = x.bitsize;
    if (oldbitsize == newbitsize) {
        return x;
    }
    else if (oldbitsize > newbitsize) {
        return GroupElement(x.value, newbitsize);
    }
    else {
        // oldbitsize < newbitsize
        // replace all bits to left of msb(x) with msb(x)
        uint8_t msb = x[0];

        GroupElement new_x(x.value, newbitsize);

        if (msb == 0) return new_x;

        // msb(x) is 1
        // std::cout << "msb is 1" << std::endl;
        for (int i = oldbitsize; i < newbitsize; i++) {
            new_x.value = new_x.value | ((uint64_t)1 << i);
        }
        return new_x;
    }
}

GroupElement signedDivide(GroupElement x, GroupElement y)
 {
    // assumes that underlying signed value of y is positive
    // todo: instead of recomputing N every time, store it in GE? what are the tradeoffs?

    assert(x.bitsize == y.bitsize);
    assert(x.bitsize <= 64);
    GroupElement N(0, x.bitsize);

    int64_t value, num, den;
    num = static_cast<int64_t>(x.value);
    den = static_cast<int64_t>(y.value);

    if (x.bitsize == 64){
        value = num / den;
    }
    else {
        N.value = ((uint64_t)1 << (x.bitsize));
        if (x.value >= (N.value >> 1)) {
            num = num - N.value;
        }
        value = num / den;
    }

    // -5/3 in c++ returns -1 but we want ans -2
    if (den * value > num) {
        value = value - 1;
    }

    return GroupElement(value, x.bitsize);

 }

 GroupElement signedMod(GroupElement x, GroupElement y)
 {
     // assumes that underlying signed value of y is positive
     // todo: handle other bitlengths
     // we dont use this anymore

    if (x.bitsize == 32) {
        int32_t value = static_cast<int32_t>(x.value) % static_cast<int32_t>(y.value);

            // using the above exprn as the formula gives a problem
            // for e.g. with -5%3 we expect the answer to be 1 because -5 = 3*-2 + 1
            // but above line says -5%3 = -2
            // therefore using this if condn

            // value = static_cast<uint64_t>((static_cast<int64_t>(x.value)) % (static_cast<int64_t>(y.value)) );

            if ((value != 0) && (static_cast<int32_t>(x.value) < 0)) {

                value += y.value;
            }

            return GroupElement(static_cast<uint32_t>(value), x.bitsize);
    }

    int64_t value = static_cast<int64_t>(x.value) % static_cast<int64_t>(y.value);

    // using the above exprn as the formula gives a problem
    // for e.g. with -5%3 we expect the answer to be 1 because -5 = 3*-2 + 1
    // but above line says -5%3 = -2
    // therefore using this if condn

    // value = static_cast<uint64_t>((static_cast<int64_t>(x.value)) % (static_cast<int64_t>(y.value)) );

    if ((value != 0) && (static_cast<int64_t>(x.value) < 0)) {

        value += y.value;
    }

    return GroupElement(static_cast<uint64_t>(value), x.bitsize);
 }

GroupElement flt2fxd(uint64_t x, int scale, int inp_bitlen){
    uint64_t fxdval = x << scale;
    return GroupElement(fxdval, inp_bitlen);
}

long double fxd2flt(GroupElement x, int scale, int inp_bitlen){
    assert(x.bitsize == inp_bitlen);
    uint64_t N_half = ((uint64_t)1 << (inp_bitlen - 1));
    long double fval;
    if (x.value >= N_half){
        fval = (long double) x.value - N_half - N_half;
    }
    else{
        fval = x.value;
    }

    return (fval / ((uint64_t)1<<scale));
}

int64_t getSignedValue(GroupElement x) {
    if (x.bitsize == 64) {
        return static_cast<int64_t>(x.value);
    }
    int msb = x[0];
    x.value = x.value % ((uint64_t)1 << x.bitsize);
    int64_t val = x.value;
    if (msb == 1) {
        val = val - ((uint64_t)1 << x.bitsize);
    }
    return val;
}

void matmul_eval_helper(int dim1, int dim2, int dim3, GroupElement *A,
                            GroupElement *B, GroupElement *C, GroupElement *ka, GroupElement *kb, GroupElement *kc) {
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_A(dim1, dim2);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_ka(dim1, dim2);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_B(dim2, dim3);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_kb(dim2, dim3);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_C(dim1, dim3);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_kc(dim1, dim3);

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            eigen_A(i, j) = Arr2DIdxRowM(A, dim1, dim2, i, j).value;
        }
    }
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim3; j++) {
            eigen_B(i, j) = Arr2DIdxRowM(B, dim2, dim3, i, j).value;
        }
    }
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            eigen_ka(i, j) = Arr2DIdxRowM(ka, dim1, dim2, i, j).value;
        }
    }
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim3; j++) {
            eigen_kb(i, j) = Arr2DIdxRowM(kb, dim2, dim3, i, j).value;
        }
    }
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            eigen_kc(i, j) = Arr2DIdxRowM(kc, dim1, dim3, i, j).value;
        }
    }
    if (party == SERVER) {
        eigen_C = eigen_A * eigen_B - eigen_ka * eigen_B - eigen_A * eigen_kb + eigen_kc;
    }
    else {
        eigen_C = eigen_kc - eigen_ka * eigen_B - eigen_A * eigen_kb;
    }
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            Arr2DIdxRowM(C, dim1, dim3, i, j).value = eigen_C(i, j);
        }
    }
}