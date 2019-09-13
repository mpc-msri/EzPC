#pragma once
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include "Functionalities.h"
using namespace std;

/***************** Functions for different layers in NN *********************/
void MatMulCSF2D(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<porthosSecretType> >& A, 
		vector< vector<porthosSecretType> >& B, 
		vector< vector<porthosSecretType> >& C, 
		int32_t consSF);

void ArgMax1(int32_t outArrS1, 
		int32_t inArrS1, 
		int32_t inArrS2, 
		vector< vector<porthosSecretType> >& inArr, 
		int32_t dim, vector<porthosSecretType>& outArr);

void ArgMax3(int32_t outs1, 
		int32_t outs2, 
		int32_t outs3, 
		int32_t ins1, 
		int32_t ins2, 
		int32_t ins3, 
		int32_t ins4,
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		int32_t dim, 
		vector< vector< vector<porthosSecretType> > >& outArr);

void Relu2(int32_t s1, 
		int32_t s2, 
		vector< vector<porthosSecretType> >& inArr, 
		vector< vector<porthosSecretType> >& outArr);

void Relu4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

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
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void ElemWiseMul2(int32_t s1, 
		int32_t s2, 
		vector< vector<porthosSecretType> >& arr1, 
		vector< vector<porthosSecretType> >& arr2, 
		vector< vector<porthosSecretType> >& outArr, 
		int32_t shrout);

void ElemWiseMul4(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<porthosSecretType> > > >& arr1, 
		vector< vector< vector< vector<porthosSecretType> > > >& arr2, 
		vector< vector< vector< vector<porthosSecretType> > > >& outArr, 
		int32_t shrout);

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
		vector< vector< vector< vector<porthosSecretType> > > >& outArr);

void FusedBatchNorm4411(int32_t s1, 
		int32_t s2, 
		int32_t s3, 
		int32_t s4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inArr, 
		vector<porthosSecretType>& multArr, 
		vector<porthosSecretType>& biasArr, 
		int32_t consSF,
		vector< vector< vector< vector<porthosSecretType> > > >& outputArr);

void ReduceMean24(int32_t outS1, 
		int32_t outS2, 
		int32_t inS1, int32_t inS2, int32_t inS3, int32_t inS4, 
		vector< vector< vector< vector<porthosSecretType> > > >& inputArr,
		vector<int32_t>& axes,
		vector< vector<porthosSecretType> >& outputArr);

void ClearMemSecret1(int32_t s1, vector< porthosSecretType >& arr);
void ClearMemSecret2(int32_t s1, int32_t s2, vector< vector< porthosSecretType > >& arr);
void ClearMemSecret3(int32_t s1, int32_t s2, int32_t s3, vector< vector< vector< porthosSecretType > > >& arr);
void ClearMemSecret4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector< vector< vector< vector< porthosSecretType > > > >& arr);
void ClearMemPublic2(int32_t s1, int32_t s2, vector< vector< int32_t > >& arr);

void StartComputation();
void EndComputation();

