/*
Authors: Mayank Rathee, Nishant Kumar.
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
#include "EzPCFunctionalities.h"
#include <iostream> 

void MatMulCSF2D(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<aramisSecretType> >& A, 
		vector< vector<aramisSecretType> >& B, 
		vector< vector<aramisSecretType> >& C, 
		int32_t consSF)
{
	vector<aramisSecretType> X(i*j);
	vector<aramisSecretType> Y(j*k);
	vector<aramisSecretType> Z(i*k);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<j; jj++){
			X[ii*j + jj] = A[ii][jj]; //Each row is of size j
		}
	}
	for (int ii=0; ii<j; ii++){
		for (int jj=0; jj<k; jj++){
			Y[ii*k + jj] = B[ii][jj]; //Each row is of size k
		}
	}
	assert(consSF == FLOAT_PRECISION && "Please update FLOAT_PRECISION value in globals.h to be equal to consSF");
	funcMatMulMPC(X, Y, Z, i, j, k, 0, 0);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<k; jj++){
			C[ii][jj] = Z[ii*k + jj]; //Each row is of size k
		}
	}
}

void ArgMax1(int32_t outArrS1, 
		int32_t inArrS1, 
		int32_t inArrS2, 
		vector< vector<aramisSecretType> >& inArr, 
		int32_t dim, 
		vector<aramisSecretType>& outArr)
{
#ifdef SKIP_ARGMAX
	return;
#endif
	vector<aramisSecretType> Arr(inArrS1*inArrS2);
	vector<aramisSecretType> maxi(outArrS1);
	for(int ii=0;ii<inArrS1;ii++){
		for(int jj=0;jj<inArrS2;jj++){
			Arr[ii*inArrS2 + jj] = inArr[ii][jj]; //Each row is of size inArrS2
		}
	}
	final_argmax = true;
	funcMaxMPC(Arr, maxi, outArr, inArrS1, inArrS2, true);
}

void ArgMax3(int32_t outs1, 
		int32_t outs2, 
		int32_t outs3, 
		int32_t ins1, 
		int32_t ins2, 
		int32_t ins3, 
		int32_t ins4,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		int32_t dim, 
		vector< vector< vector<aramisSecretType> > >& outArr){
#ifdef SKIP_ARGMAX
	return;
#endif
	vector<aramisSecretType> Arr(ins3*ins4);
	vector<aramisSecretType> maxi(outs3);
	vector<aramisSecretType> maxIndex(outs3);
	for(int ii=0;ii<ins3;ii++){
		for(int jj=0;jj<ins4;jj++){
			Arr[ii*ins4 + jj] = inArr[0][0][ii][jj]; //Each row is of size inArrS2
		}
	}
	final_argmax = true;
	funcMaxMPC(Arr, maxi, maxIndex, ins3, ins4, true);
	for(int ii=0;ii<outs3;ii++){
		outArr[0][0][ii] = maxIndex[ii];
	}
}

void Relu2(int32_t s1, 
		int32_t s2, 
		vector< vector<aramisSecretType> >& inArr, 
		vector< vector<aramisSecretType> >& outArr){
#ifdef SKIP_RELU
	return;
#endif
	vector<aramisSecretType> finArr(s1*s2, 0);
	vector<aramisSecretType> foutArr(s1*s2, 0);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			finArr[i*s2+j] = inArr[i][j];
		}
	}
	funcRELUMPC(finArr, foutArr, s1*s2);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			outArr[i][j] = foutArr[i*s2+j];
		}
	}
}

void Relu4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<aramisSecretType> > > >& inArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr){
#ifdef SKIP_RELU
	return;
#endif
	vector<aramisSecretType> finArr(s1*s2*s3*s4, 0);
	vector<aramisSecretType> foutArr(s1*s2*s3*s4, 0);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					finArr[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = inArr[i][j][k][l];
				}
			}
		}
	}
	uint64_t total_relu_cost_mb = (uint64_t)((uint64_t)s1*s2*s3*s4*(1+8*(2+2+1+1+4+8+2+7+2)))/(1024*1024);
	uint64_t no_of_chunks = total_relu_cost_mb/(RELU_SPLIT_CHUNK_SIZE); 
	uint64_t relu_size = s1*s2*s3*s4;
#ifdef SPLIT_RELU
	if(total_relu_cost_mb > RELU_SPLIT_CHUNK_SIZE){
		uint64_t relu_done_till = 0;
		uint64_t relu_size_per_chunk = relu_size/no_of_chunks;
		uint64_t relu_size_last_chunk = relu_size - ((no_of_chunks)*relu_size_per_chunk); 

		vector<aramisSecretType> inarr_temp(relu_size_per_chunk+4, 0);
		vector<aramisSecretType> outarr_temp(relu_size_per_chunk+4, 0);
	
		for(int i=0; i<no_of_chunks; i++){

			if(relu_size_per_chunk%NO_CORES != 0){
				for(uint64_t j=0; j<relu_size_per_chunk; j++){
					inarr_temp[j] = finArr[relu_done_till+j];
				}
				funcRELUMPC(inarr_temp, outarr_temp, relu_size_per_chunk+(relu_size_per_chunk%NO_CORES));
			}
			else{
				for(uint64_t j=0; j<relu_size_per_chunk; j++){
					inarr_temp[j] = finArr[relu_done_till+j];
				}
		
				funcRELUMPC(inarr_temp, outarr_temp, relu_size_per_chunk);

			}

					
			for(uint64_t j=0; j<relu_size_per_chunk; j++){
				foutArr[relu_done_till+j] = outarr_temp[j];
			}
			
			relu_done_till += relu_size_per_chunk;

		}
		//Last chunk
		inarr_temp = vector<aramisSecretType>();
		outarr_temp = vector<aramisSecretType>();
		if(relu_size_last_chunk != 0){
			
			vector<aramisSecretType> outarr_temp_l(relu_size_last_chunk+4, 0);
			
			if(relu_size_last_chunk%NO_CORES != 0){
				vector<aramisSecretType> inarr_temp_l(relu_size_last_chunk+(relu_size_last_chunk%NO_CORES), 0);

				for(uint64_t j=0; j<relu_size_last_chunk; j++){
					inarr_temp_l[j] = finArr[relu_done_till+j];
				}
	
			funcRELUMPC(inarr_temp_l, outarr_temp_l, relu_size_last_chunk+(relu_size_last_chunk%NO_CORES));

			}
			else{
				vector<aramisSecretType> inarr_temp_l(relu_size_last_chunk, 0);

				for(uint64_t j=0; j<relu_size_last_chunk; j++){
					inarr_temp_l[j] = finArr[relu_done_till+j];
				}
	
			funcRELUMPC(inarr_temp_l, outarr_temp_l, relu_size_last_chunk);

			}
					
			for(uint64_t j=0; j<relu_size_last_chunk; j++){
				foutArr[relu_done_till+j] = outarr_temp_l[j];
			}
			relu_done_till += relu_size_last_chunk;
		}
		assert(relu_done_till == relu_size);
	}
	else{
		funcRELUMPC(finArr, foutArr, relu_size);

	}
#else
		funcRELUMPC(finArr, foutArr, relu_size);
#endif
		
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					outArr[i][j][k][l] = foutArr[i*s2*s3*s4 + j*s3*s4 + k*s4 + l];
				}
			}
		}
	}
}

void ElemWiseMul2(int32_t s1, 
		int32_t s2, 
		vector< vector<aramisSecretType> >& arr1, 
		vector< vector<aramisSecretType> >& arr2, 
		vector< vector<aramisSecretType> >& outArr, 
		int32_t shrout)
{
	vector<aramisSecretType> arr1Vec(s1*s2, 0);
	vector<aramisSecretType> arr2Vec(s1*s2, 0);
	vector<aramisSecretType> outArrVec(s1*s2, 0);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			arr1Vec[i*s2+j] = arr1[i][j];
		}
	}
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			arr2Vec[i*s2+j] = arr2[i][j];
		}
	}
	funcDotProductMPC(arr1Vec,arr2Vec,outArrVec,s1*s2);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			outArr[i][j] = outArrVec[i*s2+j];
		}
	}
}

void ElemWiseMul4(int32_t s1,
	       	int32_t s2,
	       	int32_t s3,
	       	int32_t s4,
		vector< vector< vector< vector<aramisSecretType> > > >& arr1,
		vector< vector< vector< vector<aramisSecretType> > > >& arr2,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr,
		int32_t shrout)
{
	vector<aramisSecretType> arr1Vec(s1*s2*s3*s4, 0);
	vector<aramisSecretType> arr2Vec(s1*s2*s3*s4, 0);
	vector<aramisSecretType> outArrVec(s1*s2*s3*s4, 0);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					arr1Vec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = arr1[i][j][k][l];
				}
			}
		}
	}
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					arr2Vec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = arr2[i][j][k][l];
				}
			}
		}
	}

	funcDotProductMPC(arr1Vec, arr2Vec, outArrVec, s1*s2*s3*s4);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					outArr[i][j][k][l] = outArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l];
				}
			}
		}
	}
}

void MaxPool44(int32_t N, 
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
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{
#ifdef SKIP_MAXPOOL
	return;
#endif
	int rows = N*H*W*C;
	int cols = ksizeH*ksizeW;

	vector<aramisSecretType> reInpArr(rows*cols, 0);
	vector<aramisSecretType> maxi(rows, 0);
	vector<aramisSecretType> maxiIdx(rows, 0);

	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t colIdx = fh*ksizeW + fw;
							int32_t finalIdx = rowIdx*(ksizeH*ksizeW) + colIdx;

							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							uint64_t temp = 0;
							if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
								temp = 0;
							}
							else{
								temp = inArr[n][curPosH][curPosW][c];
							}
							reInpArr[finalIdx] = temp;
						}
					}

					rowIdx += 1;
					leftTopCornerW = leftTopCornerW + strideW;
				}

				leftTopCornerH = leftTopCornerH + strideH;
			}
		}
	}

	funcMaxMPC(reInpArr, maxi, maxiIdx, rows, cols);
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					int iidx = n*C*H*W + c*H*W + h*W + w;
					outArr[n][h][w][c] = maxi[iidx];
				}
			}
		}
	}
	
}

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
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{
#ifdef SKIP_MAXPOOL
	return;
#endif

	int rows = N*H*W*C;
	int cols = ksizeH*ksizeW;

	vector<aramisSecretType> reInpArr(rows*cols, 0);
	vector<aramisSecretType> maxi(rows, 0);
	vector<aramisSecretType> maxiIdx(rows, 0);

	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t colIdx = fh*ksizeW + fw;
							int32_t finalIdx = rowIdx*(ksizeH*ksizeW) + colIdx;

							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							uint64_t temp = 0;
							if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
								temp = 0;
							}
							else{
								temp = inArr[n][curPosH][curPosW][c];
							}
							reInpArr[finalIdx] = temp;
						}
					}

					rowIdx += 1;
					leftTopCornerW = leftTopCornerW + strideW;
				}

				leftTopCornerH = leftTopCornerH + strideH;
			}
		}
	}

	funcMaxMPC(reInpArr, maxi, maxiIdx, rows, cols);
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					int iidx = n*C*H*W + c*H*W + h*W + w;
					outArr[n][h][w][c] = maxi[iidx];
				}
			}
		}
	}
	
}

void MaxPool44Hook(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C,
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadH, 
		int32_t zPadW,
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr){
	MaxPool44(N, H, W, C, ksizeH, ksizeW, zPadH, zPadH, zPadW, zPadW, strideH, strideW, N1, imgH, imgW, C1, inArr, outArr);
}

//////////////////////////////////////////////////
// AvgPool
/////////////////////////////////////////////////
extern int partyNum;

void AvgPool44(int32_t N, 
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
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{
#ifdef SKIP_AVGPOOL
	return;
#endif

	int rows = N*H*W*C;
	vector<aramisSecretType> filterAvg(rows, 0);

	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					uint64_t curFilterSum = 0;
					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							uint64_t temp = 0;
							if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
								temp = 0;
							}
							else{
								temp = inArr[n][curPosH][curPosW][c];
							}

							curFilterSum += temp;
						}
					}

					//IMP NOTE : The local division should always be signed division.
					//TODO : For now doing local truncation : but this will introduce error
					if (partyNum == PARTY_A)
						filterAvg[rowIdx] = static_cast<uint64_t>((static_cast<int64_t>(curFilterSum))/(ksizeH*ksizeW));

					if (partyNum == PARTY_B)
						filterAvg[rowIdx] = -static_cast<uint64_t>((static_cast<int64_t>(-curFilterSum))/(ksizeH*ksizeW));

					rowIdx += 1;
					leftTopCornerW = leftTopCornerW + strideW;
				}

				leftTopCornerH = leftTopCornerH + strideH;
			}
		}
	}

	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					int iidx = n*C*H*W + c*H*W + h*W + w;
					outArr[n][h][w][c] = filterAvg[iidx];
				}
			}
		}
	}
	
}

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
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{
#ifdef SKIP_AVGPOOL
	return;
#endif

	int rows = N*H*W*C;
	vector<aramisSecretType> filterAvg(rows, 0);

	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					uint64_t curFilterSum = 0;
					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							uint64_t temp = 0;
							if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
								temp = 0;
							}
							else{
								temp = inArr[n][curPosH][curPosW][c];
							}

							curFilterSum += temp;
						}
					}

					//IMP NOTE : The local division should always be signed division.
					//TODO : For now doing local truncation : but this will introduce error
					if (partyNum == PARTY_A)
						filterAvg[rowIdx] = static_cast<uint64_t>((static_cast<int64_t>(curFilterSum))/(ksizeH*ksizeW));

					if (partyNum == PARTY_B)
						filterAvg[rowIdx] = -static_cast<uint64_t>((static_cast<int64_t>(-curFilterSum))/(ksizeH*ksizeW));

					rowIdx += 1;
					leftTopCornerW = leftTopCornerW + strideW;
				}

				leftTopCornerH = leftTopCornerH + strideH;
			}
		}
	}

	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					int iidx = n*C*H*W + c*H*W + h*W + w;
					outArr[n][h][w][c] = filterAvg[iidx];
				}
			}
		}
	}
	
}

void AvgPool44Hook(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t C,
		int32_t ksizeH, 
		int32_t ksizeW,
		int32_t zPadH, 
		int32_t zPadW,
		int32_t strideH, 
		int32_t strideW,
		int32_t N1, 
		int32_t imgH, 
		int32_t imgW, 
		int32_t C1,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{
	AvgPool44(N, H, W, C, ksizeH, ksizeW, zPadH, zPadH, zPadW, zPadW, strideH, strideW, N1, imgH, imgW, C1, inArr, outArr);
}


void ScalarMul4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4,
		int64_t scalar,
		vector< vector< vector< vector<aramisSecretType> > > >& inputArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outputArr,
		int64_t consSF)
{
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					outputArr[i][j][k][l] = inputArr[i][j][k][l]*scalar;
				}
			}
		}
	}
}

void ScalarMul2(int32_t s1, 
		int32_t s2,
		int64_t scalar,
		vector< vector<aramisSecretType> >& inputArr,
		vector< vector<aramisSecretType> >& outputArr,
		int64_t consSF)
{
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			outputArr[i][j] = inputArr[i][j]*scalar;
		}
	}
}

void TempFusedBatchNorm4411(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		int32_t vecS1,
		vector<aramisSecretType>& multArr,
		vector<aramisSecretType>& biasArr,
		vector< vector< vector< vector<aramisSecretType> > > >& outputArr)
{
	vector<aramisSecretType> inArrVec(s1*s2*s3*s4, 0);
	vector<aramisSecretType> multArrVec(s1*s2*s3*s4, 0);
	vector<aramisSecretType> outArrVec(s1*s2*s3*s4, 0);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					inArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = inArr[i][j][k][l];
				}
			}
		}
	}
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					multArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = multArr[l];
				}
			}
		}
	}

	funcDotProductMPC(inArrVec, multArrVec, outArrVec, s1*s2*s3*s4);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					outputArr[i][j][k][l] = outArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] + biasArr[l];
				}
			}
		}
	}
}

void FusedBatchNorm4411(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4,
		vector< vector< vector< vector<aramisSecretType> > > >& inArr,
		vector<aramisSecretType>& multArr,
		vector<aramisSecretType>& biasArr,
		int32_t consSF,
		vector< vector< vector< vector<aramisSecretType> > > >& outputArr)
{
	vector<aramisSecretType> inArrVec(s1*s2*s3*s4, 0);
	vector<aramisSecretType> multArrVec(s1*s2*s3*s4, 0);
	vector<aramisSecretType> outArrVec(s1*s2*s3*s4, 0);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					inArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = inArr[i][j][k][l];
				}
			}
		}
	}
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					multArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = multArr[l];
				}
			}
		}
	}

	funcDotProductMPC(inArrVec, multArrVec, outArrVec, s1*s2*s3*s4);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					outputArr[i][j][k][l] = outArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] + biasArr[l];
				}
			}
		}
	}
}

void Conv2DCSFNew(int32_t N, 
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
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{
#ifdef SKIP_CONV
	return;
#endif
	funcConv2DCSF(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);
}

void ClearMemSecret1(int32_t s1, vector<aramisSecretType>& arr){
	arr = vector<aramisSecretType>();
}

void ClearMemSecret2(int32_t s1, int32_t s2, vector< vector< aramisSecretType > >& arr){
	arr = vector< vector< aramisSecretType > >();
}

void ClearMemSecret3(int32_t s1, int32_t s2, int32_t s3, vector< vector< vector< aramisSecretType > > >& arr){
	arr = vector< vector< vector< aramisSecretType > > >();

}

void ClearMemSecret4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector< vector< vector< vector< aramisSecretType > > > >& arr){
	arr = vector< vector< vector< vector< aramisSecretType > > > >();
}

void ClearMemPublic2(int32_t s1, int32_t s2, vector< vector< int32_t > >& arr){
	arr = vector< vector< int32_t > >();
}

void StartComputation(){
	print_string("[ARAMIS STATUS]: Aramis initilization completed. Bootstrapping main protocol now...");
	synchronize(2000000);
	print_string("[ARAMIS STATUS]: Starting main protocol...");
	touch_time();
}

void EndComputation(){
	leave_time();
}
