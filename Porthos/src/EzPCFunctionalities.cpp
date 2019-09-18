/*

Authors: Mayank Rathee, Nishant Kumar.

Copyright:
Copyright (c) 2018 Microsoft Research
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
#include <cassert>
#include <chrono>

extern int partyNum;
extern CommunicationObject commObject;

using namespace std::chrono;

/***************** Functions for different layers in NN *********************/
void MatMulCSF2D(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<porthosSecretType> >& A, 
		vector< vector<porthosSecretType> >& B, 
		vector< vector<porthosSecretType> >& C, 
		int32_t consSF)
{
	log_print("EzPCFunctionalities : Starting MatMulCSF2D ... ");

#if (LOG_LAYERWISE)	
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
#endif

	vector<porthosSecretType> X(i*j);
	vector<porthosSecretType> Y(j*k);
	vector<porthosSecretType> Z(i*k);
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
	funcMatMulMPC(X, Y, Z, i, j, k, 0, 0, consSF);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<k; jj++){
			C[ii][jj] = Z[ii*k + jj]; //Each row is of size k
		}
	}

#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeMatmul[1] += tt;
	commObject.dataMatmul[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataMatmul[1] += (commObject.totalDataReceived - bytesReceived);
#endif
}

//TODO: right now dim is getting ignored, but once the const optimization is done in tf compilation, dim should be a constant
//Then use dim to find out which axis to take max on

void ArgMax1(int32_t outArrS1, 
		int32_t inArrS1, 
		int32_t inArrS2, 
		vector< vector<porthosSecretType> >& inArr, 
		int32_t dim, 
		vector<porthosSecretType>& outArr)
{
		
	log_print("EzPCFunctionalities : Starting ArgMax1 ... ");
	vector<porthosSecretType> Arr(inArrS1*inArrS2);
	vector<porthosSecretType> maxi(outArrS1);
	for(int ii=0;ii<inArrS1;ii++){
		for(int jj=0;jj<inArrS2;jj++){
			Arr[ii*inArrS2 + jj] = inArr[ii][jj]; //Each row is of size inArrS2
		}
	}
	funcMaxMPC(Arr, maxi, outArr, inArrS1, inArrS2, true);
}

void ArgMax3(int32_t outs1, 
		int32_t outs2, 
		int32_t outs3, 
		int32_t ins1, 
		int32_t ins2, 
		int32_t ins3, 
		int32_t ins4,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		int32_t dim, 
		vector< vector< vector<porthosSecretType> > >& outArr)
{

	log_print("EzPCFunctionalities : Starting ArgMax3 ... ");
	vector<porthosSecretType> Arr(ins3*ins4);
	vector<porthosSecretType> maxi(outs3);
	vector<porthosSecretType> maxIndex(outs3);
	for(int ii=0;ii<ins3;ii++){
		for(int jj=0;jj<ins4;jj++){
			Arr[ii*ins4 + jj] = inArr[0][0][ii][jj]; //Each row is of size inArrS2
		}
	}
	funcMaxMPC(Arr, maxi, maxIndex, ins3, ins4, true);
	for(int ii=0;ii<outs3;ii++){
		outArr[0][0][ii] = maxIndex[ii];
	}
}

void Relu2(int32_t s1, 
		int32_t s2, 
		vector< vector<porthosSecretType> >& inArr, 
		vector< vector<porthosSecretType> >& outArr)
{
	log_print("EzPCFunctionalities : Starting Relu2 ... ");

#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
#endif
	vector<porthosSecretType> finArr(s1*s2, 0);
	vector<porthosSecretType> foutArr(s1*s2, 0);
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
#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeRelu += tt;
	commObject.dataRelu[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataRelu[1] += (commObject.totalDataReceived - bytesReceived);
#endif
}

void Relu4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
	log_print("EzPCFunctionalities : Starting Relu4 ... ");

#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
#endif

	vector<porthosSecretType> finArr(s1*s2*s3*s4, 0);
	vector<porthosSecretType> foutArr(s1*s2*s3*s4, 0);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					finArr[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] = inArr[i][j][k][l];
				}
			}
		}
	}
	funcRELUMPC(finArr, foutArr, s1*s2*s3*s4);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					outArr[i][j][k][l] = foutArr[i*s2*s3*s4 + j*s3*s4 + k*s4 + l];
				}
			}
		}
	}

#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeRelu += tt;
	commObject.dataRelu[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataRelu[1] += (commObject.totalDataReceived - bytesReceived);
#endif
}

void ElemWiseMul2(int32_t s1, 
		int32_t s2, 
		vector< vector<porthosSecretType> >& arr1, 
		vector< vector<porthosSecretType> >& arr2, 
		vector< vector<porthosSecretType> >& outArr, 
		int32_t shrout)
{
	log_print("EzPCFunctionalities : Starting ElemWiseMul2 ... ");
	vector<porthosSecretType> arr1Vec(s1*s2, 0);
	vector<porthosSecretType> arr2Vec(s1*s2, 0);
	vector<porthosSecretType> outArrVec(s1*s2, 0);
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
		vector< vector< vector< vector<porthosSecretType> > > >& arr1,
		vector< vector< vector< vector<porthosSecretType> > > >& arr2,
		vector< vector< vector< vector<porthosSecretType> > > >& outArr,
		int32_t shrout)
{
	log_print("EzPCFunctionalities : Starting ElemWiseMul4 ... ");
	vector<porthosSecretType> arr1Vec(s1*s2*s3*s4, 0);
	vector<porthosSecretType> arr2Vec(s1*s2*s3*s4, 0);
	vector<porthosSecretType> outArrVec(s1*s2*s3*s4, 0);
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

//////////////////////////////////////////////////
// MaxPool
/////////////////////////////////////////////////

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
		vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
	log_print("EzPCFunctionalities : Starting MaxPool44 ... ");

#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
#endif

	int rows = N*H*W*C;
	int cols = ksizeH*ksizeW;

	vector<porthosSecretType> reInpArr(rows*cols, 0);
	vector<porthosSecretType> maxi(rows, 0);
	vector<porthosSecretType> maxiIdx(rows, 0);

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

							porthosSecretType temp = 0;
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
	
#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeMaxpool += tt;
	commObject.dataMaxPool[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataMaxPool[1] += (commObject.totalDataReceived - bytesReceived);
#endif
}


//////////////////////////////////////////////////
// AvgPool
/////////////////////////////////////////////////

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
		vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
	log_print("EzPCFunctionalities : Starting AvgPool44 ... ");
#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
#endif

	int rows = N*H*W*C;
	vector<porthosSecretType> filterAvg(rows, 0);

	int rowIdx = 0;
	for(int n=0;n<N;n++){
		for(int c=0;c<C;c++){
			int32_t leftTopCornerH = -zPadHLeft;
			int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
			while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
				int32_t leftTopCornerW = -zPadWLeft;
				int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
				while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

					porthosSecretType curFilterSum = 0;
					for(int fh=0;fh<ksizeH;fh++){
						for(int fw=0;fw<ksizeW;fw++){
							int32_t curPosH = leftTopCornerH + fh;
							int32_t curPosW = leftTopCornerW + fw;

							porthosSecretType temp = 0;
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
						filterAvg[rowIdx] = static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(curFilterSum))/(ksizeH*ksizeW));

					if (partyNum == PARTY_B)
						filterAvg[rowIdx] = -static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(-curFilterSum))/(ksizeH*ksizeW));

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
	
#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeAvgPool += tt;
	commObject.dataAvgPool[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataAvgPool[1] += (commObject.totalDataReceived - bytesReceived);
#endif
}

void FusedBatchNorm4411(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr,
		vector<porthosSecretType>& multArr,
		vector<porthosSecretType>& biasArr,
		int32_t consSF,
		vector< vector< vector< vector<porthosSecretType> > > >& outputArr)
{
	log_print("EzPCFunctionalities : Starting TempFusedBatchNorm4411 ... ");
#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	porthosLongUnsignedInt bytesSent = commObject.totalDataSent;
	porthosLongUnsignedInt bytesReceived = commObject.totalDataReceived;
#endif

	vector<porthosSecretType> inArrVec(s1*s2*s3*s4, 0);
	vector<porthosSecretType> multArrVec(s1*s2*s3*s4, 0);
	vector<porthosSecretType> outArrVec(s1*s2*s3*s4, 0);
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

	funcDotProductMPC(inArrVec, multArrVec, outArrVec, s1*s2*s3*s4, consSF);
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			for(int k=0;k<s3;k++){
				for(int l=0;l<s4;l++){
					outputArr[i][j][k][l] = outArrVec[i*s2*s3*s4 + j*s3*s4 + k*s4 + l] + biasArr[l];
				}
			}
		}
	}

#if (LOG_LAYERWISE)
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	auto tt = time_span.count();
	commObject.timeBN += tt;
	commObject.dataBN[0] += (commObject.totalDataSent - bytesSent);
	commObject.dataBN[1] += (commObject.totalDataReceived - bytesReceived);
#endif
}

void ReduceMean24(int32_t outS1, 
		int32_t outS2, 
		int32_t inS1, 
		int32_t inS2, 
		int32_t inS3, 
		int32_t inS4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inputArr,
		vector<int32_t>& axes,
		vector< vector<porthosSecretType> >& outputArr)
{
	for(int i1=0;i1<outS1;i1++){
		for(int i2=0;i2<outS2;i2++){
			porthosSecretType summ = 0;
			for(int i=0;i<inS2;i++){
				for(int j=0;j<inS3;j++){
					summ = summ + inputArr[i1][i][j][i2];
				}
			}
			outputArr[i1][i2] = static_cast<porthosSecretType>((static_cast<porthosSignedSecretType>(summ))/(inS2*inS3));
		}
	}
}

void ClearMemSecret1(int32_t s1, vector< porthosSecretType >& arr){
	arr = vector< porthosSecretType >();
}

void ClearMemSecret2(int32_t s1, int32_t s2, vector< vector< porthosSecretType > >& arr){
	arr = vector< vector< porthosSecretType > >();
}

void ClearMemSecret3(int32_t s1, int32_t s2, int32_t s3, vector< vector< vector< porthosSecretType > > >& arr){
	arr = vector< vector< vector< porthosSecretType > > >();

}

void ClearMemSecret4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector< vector< vector< vector< porthosSecretType > > > >& arr){
	arr = vector< vector< vector< vector< porthosSecretType > > > >();
}

void ClearMemPublic2(int32_t s1, int32_t s2, vector< vector< int32_t > >& arr){
	arr = vector< vector< int32_t > >();
}

void StartComputation(){
	cout<<"Reached start of computation. Syncronizing across parties..."<<endl;
	synchronize(2000000);
	cout<<"Syncronized - now starting actual execution at "<<getCurrentTime()<<endl;
	start_m();
}

void EndComputation(){
	end_m();
}

