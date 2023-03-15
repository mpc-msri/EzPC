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

#include <llama/utils.h>
#include <llama/array.h>
#include <llama/comms.h>
#include <assert.h>
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <chrono>

void MatAdd(int s1, int s2, GroupElement *A, GroupElement* B, GroupElement *C)
{
    for (int i = 0; i < s1; i++)
    {
        for (int j = 0; j < s2; j++)
        {
            Arr2DIdx(C, s1, s2, i, j) = Arr2DIdx(A, s1, s2, i, j) + Arr2DIdx(B, s1, s2, i, j);
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
                    Arr4DIdx(C, s0, s1, s2, s3, i, j, k, l) = Arr4DIdx(A, s0, s1, s2, s3, i, j, k, l) + Arr4DIdx(B, s0, s1, s2, s3, i, j, k, l);
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
            Arr2DIdx(C, s1, s2, i, j) = Arr2DIdx(A, s1, s2, i, j) - Arr2DIdx(B, s1, s2, i, j);
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
                    Arr4DIdx(C, s0, s1, s2, s3, i, j, k, l) = Arr4DIdx(A, s0, s1, s2, s3, i, j, k, l) - Arr4DIdx(B, s0, s1, s2, s3, i, j, k, l);
                }
            }
        }
    }
}

void MatMul(int s1, int s2, int s3, eigenMatrix &A, eigenMatrix &B, eigenMatrix &C)
{
    auto start = std::chrono::high_resolution_clock::now();
    C = A * B;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    eigenMicroseconds += duration.count();
}


void matmul_cleartext_eigen_llama(int dim1, int dim2, int dim3, GroupElement *inA,
                            GroupElement *inB, GroupElement *outC) {
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_A(dim1, dim2);
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_B(dim2, dim3);
  Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_C(dim1, dim3);

  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim2; j++) {
      eigen_A(i, j) = Arr2DIdx(inA, dim1, dim2, i, j);
    }
  }
  for (int i = 0; i < dim2; i++) {
    for (int j = 0; j < dim3; j++) {
      eigen_B(i, j) = Arr2DIdx(inB, dim2, dim3, i, j);
    }
  }
  eigen_C = eigen_A * eigen_B;
  for (int i = 0; i < dim1; i++) {
    for (int j = 0; j < dim3; j++) {
      Arr2DIdx(outC, dim1, dim3, i, j) = eigen_C(i, j);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  eigenMicroseconds += duration.count();
}


void MatMul(int s1, int s2, int s3, GroupElement *A, GroupElement *B, GroupElement *C)
{
    matmul_cleartext_eigen_llama(s1, s2, s3, A, B, C);
}

void Conv2DReshapeFilter(int FH, int FW, int CI, int CO, GroupElement* filter, GroupElement* reshapedFilter)
{
    for(int co = 0; co < CO; co++){
        for(int fh = 0; fh < FH; fh++){
            for(int fw = 0; fw < FW; fw++){
                for(int ci = 0; ci < CI; ci++){
                    // Arr2DIdx(reshapedFilter, CO, FH*FW*CI, co, (fh*FW*CI) + (fw*CI) + ci) = Arr4DIdx(filter, FH, FW, CI, CO, fh, fw, ci, co);
                    Arr2DIdx(reshapedFilter, CO, FH*FW*CI, co, (fh*FW*CI) + (fw*CI) + ci) = Arr2DIdx(filter, CO, FH*FW*CI, co, (fh*FW*CI) + (fw*CI) + ci);
                }
            }
        }
    }
}

void Conv2DReshapeFilter(int FH, int FW, int CI, int CO, GroupElement* filter, eigenMatrix &reshapedFilter)
{
    for(int co = 0; co < CO; co++){
        for(int fh = 0; fh < FH; fh++){
            for(int fw = 0; fw < FW; fw++){
                for(int ci = 0; ci < CI; ci++){
                    // reshapedFilter(co, (fh*FW*CI) + (fw*CI) + ci) = Arr4DIdx(filter, FH, FW, CI, CO, fh, fw, ci, co);
                    reshapedFilter(co, (fh*FW*CI) + (fw*CI) + ci) = Arr2DIdx(filter, CO, FH * FW * CI, co, (fh*FW*CI) + (fw*CI) + ci);
                }
            }
        }
    }
}

void Conv2DReshapeInput(size_t N, size_t H, size_t W, size_t CI, size_t FH, size_t FW, size_t zPadHLeft, size_t zPadHRight, size_t zPadWLeft, size_t zPadWRight, size_t strideH, size_t strideW, size_t RRows, size_t RCols, GroupElement *inputArr, GroupElement *outputArr)
{
    size_t linIdxFilterMult = 0;
	for (size_t n = 0; n < N; n++){
		size_t leftTopCornerH = 0 - zPadHLeft;
		size_t extremeRightBottomCornerH = H - 1 + zPadHRight;
		while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
			size_t leftTopCornerW = 0 - zPadWLeft;
			size_t extremeRightBottomCornerW = W - 1 + zPadWRight;
			while((leftTopCornerW + FW - 1) <= extremeRightBottomCornerW){

				for (size_t fh = 0; fh < FH; fh++){
					for (size_t fw = 0; fw < FW; fw++){
						size_t curPosH = leftTopCornerH + fh;
						size_t curPosW = leftTopCornerW + fw;
						for (size_t ci = 0; ci < CI; ci++){
                            size_t rowidx = (fh*FW*CI) + (fw*CI) + ci;
                            // std::cout << rowidx << std::endl;
							if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))){
								Arr2DIdx(outputArr, RRows, RCols, rowidx, linIdxFilterMult) = 0L;
							}
							else{
                                auto l = Arr4DIdx(inputArr, N, H, W, CI, n, curPosH, curPosW, ci);
								Arr2DIdx(outputArr, RRows, RCols, rowidx, linIdxFilterMult) = l;
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

void Conv2DReshapeInputPartial(size_t N, size_t H, size_t W, size_t CI, size_t FH, size_t FW, size_t zPadHLeft, size_t zPadHRight, size_t zPadWLeft, size_t zPadWRight, size_t strideH, size_t strideW, size_t RRows, size_t RCols, GroupElement *inputArr, eigenMatrix &outputArr, size_t batchIndex)
{
    size_t linIdxFilterMult = 0;
    size_t n = batchIndex;
    size_t leftTopCornerH = 0 - zPadHLeft;
    size_t extremeRightBottomCornerH = H - 1 + zPadHRight;
    while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
        size_t leftTopCornerW = 0 - zPadWLeft;
        size_t extremeRightBottomCornerW = W - 1 + zPadWRight;
        while((leftTopCornerW + FW - 1) <= extremeRightBottomCornerW){

            for (size_t fh = 0; fh < FH; fh++){
                for (size_t fw = 0; fw < FW; fw++){
                    size_t curPosH = leftTopCornerH + fh;
                    size_t curPosW = leftTopCornerW + fw;
                    for (size_t ci = 0; ci < CI; ci++){
                        if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))){
                            outputArr((fh*FW*CI) + (fw*CI) + ci, linIdxFilterMult) = 0L;
                        }
                        else{
                            outputArr((fh*FW*CI) + (fw*CI) + ci, linIdxFilterMult) = Arr4DIdx(inputArr, N, H, W, CI, n, curPosH, curPosW, ci);
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

void Conv2DReshapeOutput(int N, int finalH, int finalW, int CO, GroupElement *inputArr, GroupElement *outputArr)
{
    for (int co = 0; co < CO; ++co){
		for (int n = 0; n < N; ++n){
			for(int h = 0; h < finalH; ++h){
				for (int w = 0; w < finalW; ++w){
					Arr4DIdx(outputArr, N, finalH, finalW, CO, n, h, w, co) = Arr2DIdx(inputArr, CO, N*finalH*finalW, co, (n*finalH*finalW) + (h*finalW) + w);
				}
			}
		}
	}
}


void Conv2DReshapeOutputPartial(int N, int finalH, int finalW, int CO, eigenMatrix inputArr, GroupElement *outputArr, int batchIndex)
{
    for (int co = 0; co < CO; ++co){
        for(int h = 0; h < finalH; ++h){
            for (int w = 0; w < finalW; ++w){
                Arr4DIdx(outputArr, N, finalH, finalW, CO, batchIndex, h, w, co) = inputArr(co, (h*finalW) + w);
            }
        }
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
    size_t reshapedFilterRows = CO;
	size_t reshapedFilterCols = FH*FW*CI;
	size_t reshapedIPRows = FH*FW*CI;
	size_t newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
	size_t newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);
	size_t reshapedIPCols = N * newH * newW;

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

Conv2DCache allocateConv2DCache(int N, int H, int W, int CI, 
                                int FH, int FW, int CO, 
                                int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, 
                                int strideH, int strideW) {
    int reshapedFilterRows = CO;
	int reshapedFilterCols = FH*FW*CI;
	int reshapedIPRows = reshapedFilterCols;
	int newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
	int newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);
	int reshapedIPCols = newH * newW;

    Conv2DCache cache;
    cache.reshapedFilter = eigenMatrix(reshapedFilterRows, reshapedFilterCols);
	cache.reshapedInput = eigenMatrix(reshapedIPRows, reshapedIPCols);
	cache.matmulResult = eigenMatrix(reshapedFilterRows, reshapedIPCols);
    cache.temp = make_array<GroupElement>(N, newH, newW, CO);

    return cache;
}

void freeConv2DCache(const Conv2DCache &cache) {
    // cache.reshapedFilter.resize(0, 0);
    // cache.reshapedInput.resize(0, 0);
    // cache.matmulResult.resize(0, 0);
    delete[] cache.temp;
}

void Conv2DPlaintext(int N, int H, int W, int CI, 
				   int FH, int FW, int CO, 
				   int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, 
				   int strideH, int strideW, 
				   GroupElement *inputArr, 
				   GroupElement * filterArr, 
				   GroupElement * outArr,
                   Conv2DCache &cache)
{
    int reshapedFilterRows = CO;
	int reshapedFilterCols = FH*FW*CI;
	int reshapedIPRows = FH*FW*CI;
	int newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
	int newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);
	int reshapedIPCols = newH * newW;

    Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, cache.reshapedFilter);
    for(int i = 0; i < N; ++i) {
        Conv2DReshapeInputPartial(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, cache.reshapedInput, i);
        MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, cache.reshapedFilter, cache.reshapedInput, cache.matmulResult);
        Conv2DReshapeOutputPartial(N, newH, newW, CO, cache.matmulResult, outArr, i);
    }
}

void VecCopy(int s, GroupElement *input, GroupElement *output)
{
    for(int i = 0; i < s; i++){
        output[i] = input[i];
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
                Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(C, s1, s3, i, k) - Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k);
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
                Arr2DIdx(C, s1, s3, i, k) = Arr2DIdx(C, s1, s3, i, k) + Arr2DIdx(A, s1, s2, i, j) * Arr2DIdx(B, s2, s3, j, k);
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
                    Arr4DIdx(output, s1, s2, s3, s4, i, j, k, l) = Arr4DIdx(input, s1, s2, s3, s4, i, j, k, l);
                }
            }
        }
    }
}

void MatFinalize4(int bw, int s1, int s2, int s3, int s4, GroupElement *input)
{
    for(int i = 0; i < s1; i++)
    {
        for(int j = 0; j < s2; j++)
        {
            for(int k = 0; k < s3; k++)
            {
                for(int l = 0; l < s4; l++)
                {
                    mod(Arr4DIdx(input, s1, s2, s3, s4, i, j, k, l), bw);
                }
            }
        }
    }
}

void matmul_eval_helper(int party, int dim1, int dim2, int dim3, GroupElement *A,
                            GroupElement *B, GroupElement *C, GroupElement *ka, GroupElement *kb, GroupElement *kc) {
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_A(dim1, dim2);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_ka(dim1, dim2);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_B(dim2, dim3);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_kb(dim2, dim3);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_C(dim1, dim3);
    Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> eigen_kc(dim1, dim3);

    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            eigen_A(i, j) = Arr2DIdx(A, dim1, dim2, i, j);
        }
    }
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim3; j++) {
            eigen_B(i, j) = Arr2DIdx(B, dim2, dim3, i, j);
        }
    }
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            eigen_ka(i, j) = Arr2DIdx(ka, dim1, dim2, i, j);
        }
    }
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim3; j++) {
            eigen_kb(i, j) = Arr2DIdx(kb, dim2, dim3, i, j);
        }
    }
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            eigen_kc(i, j) = Arr2DIdx(kc, dim1, dim3, i, j);
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
            Arr2DIdx(C, dim1, dim3, i, j) = eigen_C(i, j);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    eigenMicroseconds += duration.count();
}

// void matmul_eval_helper(int party, int dim1, int dim2, int dim3, GroupElement *A,
//                             GroupElement *B, GroupElement *C, GroupElement *ka, GroupElement *kb, GroupElement *kc) {
//     auto start = std::chrono::high_resolution_clock::now();
//     Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_A(A, dim1, dim2);
//     Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_ka(ka, dim1, dim2);
//     Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_B(B, dim2, dim3);
//     Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_kb(kb, dim2, dim3);
//     Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_C(C, dim1, dim3);
//     Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_kc(kc, dim1, dim3);

//     if (party == SERVER) {
//         eigen_C = eigen_A * eigen_B - eigen_ka * eigen_B - eigen_A * eigen_kb + eigen_kc;
//     }
//     else {
//         eigen_C = eigen_kc - eigen_ka * eigen_B - eigen_A * eigen_kb;
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//     eigenMicroseconds += duration.count();
// }

void packBitArray(GroupElement *A, int size, uint8_t *out) {
    int bytesize = (size % 8 == 0) ? (size / 8) : (size / 8 + 1);
    for (int i = 0; i < bytesize; ++i) {
        out[i] = 0;
    }
    for (int i = 0; i < size; i++) {
        out[i / 8] = out[i / 8] | ((A[i] & 1) << (i % 8));
    }
}
