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
    // using eigen map
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A, s1, s2);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B, s1, s2);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_eigen(C, s1, s2);
    C_eigen = A_eigen + B_eigen;
}

void MatAdd4(int s0, int s1, int s2, int s3, GroupElement* A, GroupElement* B, GroupElement* C)
{
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A, s0, s1 * s2 * s3);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B, s0, s1 * s2 * s3);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_eigen(C, s0, s1 * s2 * s3);
    C_eigen = A_eigen + B_eigen;
}

void MatAdd5(int s0, int s1, int s2, int s3, int s4, GroupElement* A, GroupElement* B, GroupElement* C)
{
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A, s0, s1 * s2 * s3 * s4);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B, s0, s1 * s2 * s3 * s4);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_eigen(C, s0, s1 * s2 * s3 * s4);
    C_eigen = A_eigen + B_eigen;
}

void MatSub(int s1, int s2, GroupElement *A, GroupElement* B, GroupElement *C)
{
    // using eigen map
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A, s1, s2);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B, s1, s2);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_eigen(C, s1, s2);
    C_eigen = A_eigen - B_eigen;
}

void MatSub4(int s0, int s1, int s2, int s3, GroupElement* A, GroupElement* B, GroupElement* C)
{
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A, s0, s1 * s2 * s3);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B, s0, s1 * s2 * s3);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_eigen(C, s0, s1 * s2 * s3);
    C_eigen = A_eigen - B_eigen;
}

void MatSub5(int s0, int s1, int s2, int s3, int s4, GroupElement* A, GroupElement* B, GroupElement* C)
{
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_eigen(A, s0, s1 * s2 * s3 * s4);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B_eigen(B, s0, s1 * s2 * s3 * s4);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C_eigen(C, s0, s1 * s2 * s3 * s4);
    C_eigen = A_eigen - B_eigen;
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
  Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(inA, dim1, dim2);
  Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(inB, dim2, dim3);
  Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eC(outC, dim1, dim3);
  eC = eA * eB;
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
    // using eigen
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(reshapedFilter, CO, FH*FW*CI);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(filter, CO, FH*FW*CI);
    eA = eB;
}

void Conv2DReshapeFilter(int FH, int FW, int CI, int CO, GroupElement* filter, eigenMatrix &reshapedFilter)
{
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(filter, CO, FH*FW*CI);
    reshapedFilter = eB;
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

    GroupElement *filterReshaped = filterArr;
	GroupElement *inputReshaped = make_array<GroupElement>(reshapedIPRows, reshapedIPCols);
	GroupElement *matmulOP = make_array<GroupElement>(reshapedFilterRows, reshapedIPCols);
    
    Conv2DReshapeInput(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
    // Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, filterReshaped);
    MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP);
    Conv2DReshapeOutput(N, newH, newW, CO, matmulOP, outArr);

    // delete[] filterReshaped;
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
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, 1>> output_eigen(output, s);
    Eigen::Map<Eigen::Matrix<GroupElement, Eigen::Dynamic, 1>> input_eigen(input, s);
    output_eigen = input_eigen;
}

void MatCopy4(int s1, int s2, int s3, int s4, GroupElement *input, GroupElement *output){
    VecCopy(s1*s2*s3*s4, input, output);
}

void MatCopy5(int s1, int s2, int s3, int s4, int s5, GroupElement *input, GroupElement *output){
    VecCopy(s1*s2*s3*s4*s5, input, output);
}

void matmul_eval_helper(int party, int dim1, int dim2, int dim3, GroupElement *A,
                            GroupElement *B, GroupElement *C, GroupElement *ka, GroupElement *kb, GroupElement *kc) {
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_A(A, dim1, dim2);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_ka(ka, dim1, dim2);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_B(B, dim2, dim3);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_kb(kb, dim2, dim3);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_C(C, dim1, dim3);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_kc(kc, dim1, dim3);

    if (party == SERVER) {
        eigen_C = (eigen_A - eigen_ka) * eigen_B - eigen_A * eigen_kb + eigen_kc;
    }
    else {
        eigen_C = eigen_kc - eigen_ka * eigen_B - eigen_A * eigen_kb;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    eigenMicroseconds += duration.count();
}

void matmul_eval_helper_triangular(int party, int dim1, int dim2, int dim3, GroupElement *A,
                            GroupElement *B, GroupElement *C, GroupElement *ka, GroupElement *kb, GroupElement *kc) {
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_A(A, dim1, dim2);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_ka(ka, dim1, dim2);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_B(B, dim2, dim3);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_kb(kb, dim2, dim3);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_C(C, dim1, dim3);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_kc(kc, dim1, dim3);

    if (party == SERVER) {
        eigen_C = ((eigen_A - eigen_ka) * eigen_B - eigen_A * eigen_kb + eigen_kc).triangularView<Eigen::Lower>();
    }
    else {
        eigen_C = (eigen_kc - eigen_ka * eigen_B - eigen_A * eigen_kb).triangularView<Eigen::Lower>();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    eigenMicroseconds += duration.count();
}

void packBitArray(GroupElement *A, int size, uint8_t *out) {
    int bytesize = (size % 8 == 0) ? (size / 8) : (size / 8 + 1);
    for (int i = 0; i < bytesize; ++i) {
        out[i] = 0;
    }
    for (int i = 0; i < size; i++) {
        out[i / 8] = out[i / 8] | ((A[i] & 1) << (i % 8));
    }
}



// 3d

void Conv3DReshapeFilter(int FD, int FH, int FW, int CI, int CO, GroupElement* filter, GroupElement* reshapedFilter)
{
    // using eigen
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eA(reshapedFilter, CO, FD*FH*FW*CI);
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(filter, CO, FD*FH*FW*CI);
    eA = eB;
}

void Conv3DReshapeFilter(int FD, int FH, int FW, int CI, int CO, GroupElement* filter, eigenMatrix &reshapedFilter)
{
    Eigen::Map<Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eB(filter, CO, FD*FH*FW*CI);
    reshapedFilter = eB;
}

void Conv3DReshapeInput(size_t N, size_t D, size_t H, size_t W, size_t CI, size_t FD, size_t FH, size_t FW, size_t zPadDLeft, size_t zPadDRight, size_t zPadHLeft, size_t zPadHRight, size_t zPadWLeft, size_t zPadWRight, size_t strideD, size_t strideH, size_t strideW, size_t RRows, size_t RCols, GroupElement *inputArr, GroupElement *outputArr)
{
    size_t linIdxFilterMult = 0;
	for (size_t n = 0; n < N; n++){
        size_t leftTopCornerD = 0 - zPadDLeft;
        size_t extremeRightBottomCornerD = D - 1 + zPadDRight;
        while((leftTopCornerD + FD - 1) <= extremeRightBottomCornerD){
            size_t leftTopCornerH = 0 - zPadHLeft;
            size_t extremeRightBottomCornerH = H - 1 + zPadHRight;
            while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
                size_t leftTopCornerW = 0 - zPadWLeft;
                size_t extremeRightBottomCornerW = W - 1 + zPadWRight;
                while((leftTopCornerW + FW - 1) <= extremeRightBottomCornerW){

                    for (size_t fd = 0; fd < FD; fd++) {
                        for (size_t fh = 0; fh < FH; fh++){
                            for (size_t fw = 0; fw < FW; fw++){
                                size_t curPosD = leftTopCornerD + fd;
                                size_t curPosH = leftTopCornerH + fh;
                                size_t curPosW = leftTopCornerW + fw;
                                for (size_t ci = 0; ci < CI; ci++){
                                    size_t rowidx = (fd*FH*FW*CI) + (fh*FW*CI) + (fw*CI) + ci;
                                    // std::cout << rowidx << std::endl;
                                    if ((((curPosD < 0) || (curPosD >= D)) || ((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))){
                                        Arr2DIdx(outputArr, RRows, RCols, rowidx, linIdxFilterMult) = 0L;
                                    }
                                    else{
                                        auto l = Arr5DIdx(inputArr, N, D, H, W, CI, n, curPosD, curPosH, curPosW, ci);
                                        Arr2DIdx(outputArr, RRows, RCols, rowidx, linIdxFilterMult) = l;
                                    }
                                }
                            }
                        }
                    }

                    linIdxFilterMult = linIdxFilterMult + 1;
                    leftTopCornerW = leftTopCornerW + strideW;
                }
                leftTopCornerH = leftTopCornerH + strideH;
            }
            leftTopCornerD = leftTopCornerD + strideD;
        }
	}
}

void Conv3DReshapeInputPartial(size_t N, size_t D, size_t H, size_t W, size_t CI, size_t FD, size_t FH, size_t FW, size_t zPadDLeft, size_t zPadDRight, size_t zPadHLeft, size_t zPadHRight, size_t zPadWLeft, size_t zPadWRight, size_t strideD, size_t strideH, size_t strideW, size_t RRows, size_t RCols, GroupElement *inputArr, eigenMatrix &outputArr, size_t batchIndex)
{
    size_t linIdxFilterMult = 0;
    size_t n = batchIndex;

    size_t leftTopCornerD = 0 - zPadDLeft;
    size_t extremeRightBottomCornerD = D - 1 + zPadDRight;
    while((leftTopCornerD + FD - 1) <= extremeRightBottomCornerD) {
        size_t leftTopCornerH = 0 - zPadHLeft;
        size_t extremeRightBottomCornerH = H - 1 + zPadHRight;
        while((leftTopCornerH + FH - 1) <= extremeRightBottomCornerH){
            size_t leftTopCornerW = 0 - zPadWLeft;
            size_t extremeRightBottomCornerW = W - 1 + zPadWRight;
            while((leftTopCornerW + FW - 1) <= extremeRightBottomCornerW){

                for (size_t fd = 0; fd < FD; fd++) {
                    for (size_t fh = 0; fh < FH; fh++){
                        for (size_t fw = 0; fw < FW; fw++){
                            size_t curPosD = leftTopCornerD + fd;
                            size_t curPosH = leftTopCornerH + fh;
                            size_t curPosW = leftTopCornerW + fw;
                            for (size_t ci = 0; ci < CI; ci++){
                                if ((((curPosD < 0) || (curPosD >= D)) || ((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))){
                                    outputArr((fd*FH*FW*CI) + (fh*FW*CI) + (fw*CI) + ci, linIdxFilterMult) = 0L;
                                }
                                else{
                                    outputArr((fd*FH*FW*CI) + (fh*FW*CI) + (fw*CI) + ci, linIdxFilterMult) = Arr5DIdx(inputArr, N, D,H, W, CI, n, curPosD, curPosH, curPosW, ci);
                                }
                            }
                        }
                    }
                }

                linIdxFilterMult = linIdxFilterMult + 1;
                leftTopCornerW = leftTopCornerW + strideW;
            }
            leftTopCornerH = leftTopCornerH + strideH;
        }
        leftTopCornerD = leftTopCornerD + strideD;
    }
}

void Conv3DReshapeOutput(int N, int finalD, int finalH, int finalW, int CO, GroupElement *inputArr, GroupElement *outputArr)
{
    for (int co = 0; co < CO; ++co){
		for (int n = 0; n < N; ++n){
            for(int d = 0; d < finalD; ++d) {
                for(int h = 0; h < finalH; ++h){
                    for (int w = 0; w < finalW; ++w){
                        Arr5DIdx(outputArr, N, finalD, finalH, finalW, CO, n, d, h, w, co) = Arr2DIdx(inputArr, CO, N*finalD*finalH*finalW, co, (n*finalD*finalH*finalW) + (d*finalH*finalW) + (h*finalW) + w);
                    }
                }
            }
		}
	}
}


void Conv3DReshapeOutputPartial(int N, int finalD, int finalH, int finalW, int CO, eigenMatrix inputArr, GroupElement *outputArr, int batchIndex)
{
    for (int co = 0; co < CO; ++co){
        for(int d = 0; d < finalD; ++d) {
            for(int h = 0; h < finalH; ++h){
                for (int w = 0; w < finalW; ++w){
                    Arr5DIdx(outputArr, N, finalD, finalH, finalW, CO, batchIndex, d, h, w, co) = inputArr(co, (d*finalH*finalW) + (h*finalW) + w);
                }
            }
        }
	}
}

void Conv3DPlaintext(int N, int D, int H, int W, int CI, 
				   int FD, int FH, int FW, int CO, 
				   int zPadDLeft, int zPadDRight, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, 
				   int strideD, int strideH, int strideW, 
				   GroupElement *inputArr, 
				   GroupElement * filterArr, 
				   GroupElement * outArr)
{
    size_t reshapedFilterRows = CO;
	size_t reshapedFilterCols = FD*FH*FW*CI;
	size_t reshapedIPRows = FD*FH*FW*CI;
    size_t newD = (((D + (zPadDLeft+zPadDRight) - FD)/strideD) + 1);
	size_t newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
	size_t newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);
	size_t reshapedIPCols = N * newD * newH * newW;

    GroupElement *filterReshaped = filterArr;
	GroupElement *inputReshaped = make_array<GroupElement>(reshapedIPRows, reshapedIPCols);
	GroupElement *matmulOP = make_array<GroupElement>(reshapedFilterRows, reshapedIPCols);
    
    Conv3DReshapeInput(N, D, H, W, CI, FD, FH, FW, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
    // Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, filterReshaped);
    MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP);
    Conv3DReshapeOutput(N, newD, newH, newW, CO, matmulOP, outArr);

    // delete[] filterReshaped;
    delete[] inputReshaped;
    delete[] matmulOP;

}

Conv3DCache allocateConv3DCache(int N, int D, int H, int W, int CI, 
                                int FD, int FH, int FW, int CO, 
                                int zPadDLeft, int zPadDRight, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, 
                                int strideD, int strideH, int strideW) {
    int reshapedFilterRows = CO;
	int reshapedFilterCols = FD*FH*FW*CI;
	int reshapedIPRows = reshapedFilterCols;
    int newD = (((D + (zPadDLeft+zPadDRight) - FD)/strideD) + 1);
	int newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
	int newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);
	int reshapedIPCols = newD * newH * newW;

    Conv3DCache cache;
    cache.reshapedFilter = eigenMatrix(reshapedFilterRows, reshapedFilterCols);
	cache.reshapedInput = eigenMatrix(reshapedIPRows, reshapedIPCols);
	cache.matmulResult = eigenMatrix(reshapedFilterRows, reshapedIPCols);
    cache.temp = make_array<GroupElement>(N, newH, newW, CO);

    return cache;
}

void freeConv3DCache(const Conv3DCache &cache) {
    // cache.reshapedFilter.resize(0, 0);
    // cache.reshapedInput.resize(0, 0);
    // cache.matmulResult.resize(0, 0);
    delete[] cache.temp;
}

void Conv3DPlaintext(int N, int D, int H, int W, int CI, 
				   int FD, int FH, int FW, int CO, 
				   int zPadDLeft, int zPadDRight, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight, 
				   int strideD, int strideH, int strideW, 
				   GroupElement *inputArr, 
				   GroupElement * filterArr, 
				   GroupElement * outArr,
                   Conv3DCache &cache)
{
    int reshapedFilterRows = CO;
	int reshapedFilterCols = FD*FH*FW*CI;
	int reshapedIPRows = FD*FH*FW*CI;
    int newD = (((D + (zPadDLeft+zPadDRight) - FD)/strideD) + 1);
	int newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
	int newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);
	int reshapedIPCols = newD * newH * newW;

    Conv3DReshapeFilter(FD, FH, FW, CI, CO, filterArr, cache.reshapedFilter);
    for(int i = 0; i < N; ++i) {
        Conv3DReshapeInputPartial(N, D, H, W, CI, FD, FH, FW, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, cache.reshapedInput, i);
        MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, cache.reshapedFilter, cache.reshapedInput, cache.matmulResult);
        Conv3DReshapeOutputPartial(N, newD, newH, newW, CO, cache.matmulResult, outArr, i);
    }
}

void ConvTranspose3DLoopInnerClear(
    int64_t N, 
    int64_t D, 
    int64_t H, 
    int64_t W, 
    int64_t CI, 
    int64_t FD, 
    int64_t FH, 
    int64_t FW, 
    int64_t CO, 
    int64_t zPadDLeft, 
    int64_t zPadDRight, 
    int64_t zPadHLeft, 
    int64_t zPadHRight, 
    int64_t zPadWLeft, 
    int64_t zPadWRight, 
    int64_t strideD, 
    int64_t strideH, 
    int64_t strideW, 
    int64_t outD, 
    int64_t outH, 
    int64_t outW, 
    GroupElement* inputArr, 
    GroupElement* filterArr, 
    GroupElement* outArr)
{
    zPadDLeft = FD - 1 - zPadDLeft;
    zPadDRight = FD - 1 - zPadDRight;
    zPadHLeft = FH - 1 - zPadHLeft;
    zPadHRight = FH - 1 - zPadHRight;
    zPadWLeft = FW - 1 - zPadWLeft;
    zPadWRight = FW - 1 - zPadWRight;

    #pragma omp parallel for collapse(5)
    for (int64_t n =  0; n < N; n++){
        for (int64_t d =  0; d < outD; d++){
            for (int64_t h =  0; h < outH; h++){
                for (int64_t w =  0; w < outW; w++){
                    for (int64_t co =  0; co < CO; co++){
                        
                        GroupElement val =  0;
                        for (int64_t ci =  0; ci < CI; ci++){
                            for (int64_t fd = d; fd < (d + FD); fd++){
                                for (int64_t fh = h; fh < (h + FH); fh++){
                                    for (int64_t fw = w; fw < (w + FW); fw++){

                                        int64_t curPosD = ((fd - zPadDLeft) / strideD);
                                        int64_t curPosH = ((fh - zPadHLeft) / strideH);
                                        int64_t curPosW = ((fw - zPadWLeft) / strideW);

                                        if ((curPosD >=  0) &&
                                            (curPosH >=  0) &&
                                            (curPosW >=  0) &&
                                            (curPosD < D) &&
                                            (curPosH < H) &&
                                            (curPosW < W) &&
                                            (((fd - zPadDLeft) % strideD) == 0) &&
                                            (((fh - zPadHLeft) % strideH) == 0) &&
                                            (((fw - zPadWLeft) % strideW) == 0))
                                        {
                                            int32_t curFilterPosD = FD + d - fd -  1;
                                            int32_t curFilterPosH = FH + h - fh -  1;
                                            int32_t curFilterPosW = FW + w - fw -  1;
                                            val += (Arr5DIdx(inputArr, N, D, H, W, CI, n, curPosD, curPosH, curPosW, ci) * Arr5DIdx(filterArr, CO, FD, FH, FW, CI, co, curFilterPosD, curFilterPosH, curFilterPosW, ci));
                                        }
                                    }
                                }
                            }
                        }
                        Arr5DIdx(outArr, N, outD, outH, outW, CO, n, d, h, w, co) =  val;
                        // std::cout << "setting element at (" << n << " " << d << " " << h << " " << w << " " << co << ")" << std::endl;
                    }
                }
            }
        }
    }
}
