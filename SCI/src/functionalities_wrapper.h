/*
Authors: Nishant Kumar, Deevashwer Rathee
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

#ifndef FUNCTIONALITIES_WRAPPER_H__
#define FUNCTIONALITIES_WRAPPER_H__

#include "globals.h" //This should be the first file to be included, to make sure all #define's come first
#include "functionalities.h"
#include "NonLinear/relu-ring.h"
#include "NonLinear/relu-field.h"
#include <vector>
#include <cstdlib>
#include <fstream>

#ifdef VERIFY_LAYERWISE
#include "functionalities_pt.h"
#endif

void Conv2D(int32_t N, int32_t H, int32_t W, int32_t CI, 
        int32_t FH, int32_t FW, int32_t CO, 
        int32_t zPadHLeft, int32_t zPadHRight, 
        int32_t zPadWLeft, int32_t zPadWRight, 
        int32_t strideH, int32_t strideW, 
        intType* inputArr, intType* filterArr, 
        intType* outArr);

void MatMul2D(int32_t s1, int32_t s2, int32_t s3, const intType* A, const intType* B, intType* C, bool modelIsA)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif
    
    std::cout<<"Matmul called s1,s2,s3 = "<<s1<<" "<<s2<<" "<<s3<<std::endl;
    int partyWithAInAB_mul = sci::ALICE; //By default, the model is A and server/Alice has it
                                         // So, in the AB mult, party with A = server and party with B = client.
    int partyWithBInAB_mul = sci::BOB;
    if (!modelIsA){
        //Model is B
        partyWithAInAB_mul = sci::BOB;
        partyWithBInAB_mul = sci::ALICE;
    }

#if defined(SCI_OT)
#ifndef MULTITHREADED_MATMUL
    if (partyWithAInAB_mul==sci::ALICE){
        if (party==sci::ALICE){
            matmulImpl->funcOTSenderInputA(s1,s2,s3,A,C,iknpOT);
        }
        else{
            matmulImpl->funcOTReceiverInputB(s1,s2,s3,B,C,iknpOT);
        }
    }
    else{
        if (party==sci::BOB){
            matmulImpl->funcOTSenderInputA(s1,s2,s3,A,C,iknpOTRoleReversed);
        }
        else{
            matmulImpl->funcOTReceiverInputB(s1,s2,s3,B,C,iknpOTRoleReversed);
        }
    }

    if (party==sci::ALICE){
        //Now irrespective of whether A is the model or B is the model and whether
        //	server holds A or B, server should add locally A*B.
        //	
        // Add also A*own share of B
        intType* CTemp = new intType[s1*s3];
        matmulImpl->ideal_func(s1,s2,s3,A,B,CTemp);
        sci::elemWiseAdd<intType>(s1*s3,C,CTemp,C);
        delete[] CTemp;
    }
    else{
        //For minionn kind of hacky runs, switch this off
#ifndef HACKY_RUN
        if (modelIsA){
            for(int i=0;i<s1*s2;i++) assert(A[i]==0);
        }
        else{
            for(int i=0;i<s1*s2;i++) assert(B[i]==0);
        }
#endif
    }

#else //MULTITHREADED_MATMUL is ON
    intType* C_ans_arr[numThreads];
    std::thread matmulThreads[numThreads];
    for(int i=0;i<numThreads;i++){
        C_ans_arr[i] = new intType[s1*s3];
        matmulThreads[i] = std::thread(funcMatmulThread,i,numThreads,s1,s2,s3,A,B,C_ans_arr[i],partyWithAInAB_mul);
    }
    for(int i=0;i<numThreads;i++){
        matmulThreads[i].join();
    }
    for(int i=0;i<s1*s3;i++){
        C[i] = 0;
    }
    for(int i=0;i<numThreads;i++){
        for(int j=0;j<s1*s3;j++){
            C[j] += C_ans_arr[i][j];
        }
        delete[] C_ans_arr[i];
    }

    if (party==sci::ALICE){
        intType* CTemp = new intType[s1*s3];
        matmulImpl->ideal_func(s1,s2,s3,A,B,CTemp);
        sci::elemWiseAdd<intType>(s1*s3,C,CTemp,C);
        delete[] CTemp;
    }
    else{
        //For minionn kind of hacky runs, switch this off
#ifndef HACKY_RUN 
        if (modelIsA){
            for(int i=0;i<s1*s2;i++) assert(A[i]==0);
        }
        else{
            for(int i=0;i<s1*s2;i++) assert(B[i]==0);
        }
#endif
    }
#endif
    intType moduloMask = (1ULL<<bitlength)-1;
    if (bitlength==64) moduloMask = -1;
    for(int i=0;i<s1*s3;i++){
        C[i] = C[i] & moduloMask;
    }

#elif defined(SCI_HE)
    assert(modelIsA == false && "Assuming code generated by compiler produces B as the model.");
    std::vector<std::vector<intType>> At(s2);
    std::vector<std::vector<intType>> Bt(s3);
    std::vector<std::vector<intType>> Ct(s3);
    for(int i = 0; i < s2; i++) {
        At[i].resize(s1);
        for(int j = 0; j < s1; j++) {
            At[i][j] = Arr2DIdxRowM(A,s1,s2,j,i);
        }
    }
    for(int i = 0; i < s3; i++) {
        Bt[i].resize(s2);
        Ct[i].resize(s1);
        for(int j = 0; j < s2; j++) {
            Bt[i][j] = Arr2DIdxRowM(B,s2,s3,j,i);
        }
    }
    heFCImpl->matrix_multiplication(s3,s2,s1,Bt,At,Ct);
    for(int i = 0; i < s1; i++) {
        for(int j = 0; j < s3; j++) {
            Arr2DIdxRowM(C,s1,s3,i,j) = Ct[j][i];
        }
    }
#endif

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    MatmulTimeInMilliSec += temp;
    std::cout<<"Time in sec for current matmul = "<<(temp/1000.0)<<std::endl; 
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    MatmulCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
    for(int i = 0; i < s1; i++) {
        for(int j = 0; j < s3; j++) {
            assert(Arr2DIdxRowM(C,s1,s3,i,j) < prime_mod);
        }
    }
#endif
    if (party == SERVER) {
        funcReconstruct2PCCons(nullptr, A, s1*s2);
        funcReconstruct2PCCons(nullptr, B, s2*s3);
        funcReconstruct2PCCons(nullptr, C, s1*s3);
    } 
    else {
        signedIntType* VA = new signedIntType[s1*s2];
        funcReconstruct2PCCons(VA, A, s1*s2);
        signedIntType* VB = new signedIntType[s2*s3];
        funcReconstruct2PCCons(VB, B, s2*s3);
        signedIntType* VC = new signedIntType[s1*s3];
        funcReconstruct2PCCons(VC, C, s1*s3);

        std::vector<std::vector<uint64_t>> VAvec;
        std::vector<std::vector<uint64_t>> VBvec;
        std::vector<std::vector<uint64_t>> VCvec;
        VAvec.resize(s1,std::vector<uint64_t>(s2,0));
        VBvec.resize(s2,std::vector<uint64_t>(s3,0));
        VCvec.resize(s1,std::vector<uint64_t>(s3,0));

        for(int i = 0; i < s1; i++) {
            for(int j = 0; j < s2; j++) {
                VAvec[i][j] = getRingElt(Arr2DIdxRowM(VA,s1,s2,i,j));
            }
        }
        for(int i = 0; i < s2; i++) {
            for(int j = 0; j < s3; j++) {
                VBvec[i][j] = getRingElt(Arr2DIdxRowM(VB,s2,s3,i,j));
            }
        }

        MatMul2D_pt(s1, s2, s3, VAvec, VBvec, VCvec, 0);

        bool pass = true;
        for(int i = 0; i < s1; i++) {
            for(int j = 0; j < s3; j++) {
                if(Arr2DIdxRowM(VC,s1,s3,i,j) != getSignedVal(VCvec[i][j])) {
                    pass = false;
                }
            }
        }
        if (pass == true) std::cout << GREEN << "MatMul Output Matches" << RESET << std::endl;
        else std::cout << RED << "MatMul Output Mismatch" << RESET << std::endl;

        delete[] VA;
        delete[] VB;
        delete[] VC;
    }
#endif
}


void ArgMax(int32_t s1, int32_t s2, intType* inArr, intType* outArr)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    static int ctr = 1;
    std::cout<<"ArgMax "<<ctr<<" called, s1="<<s1<<", s2="<<s2<<std::endl;
    ctr++;

    assert(s1==1 && "ArgMax impl right now assumes s1==1");
    argmaxImpl->ArgMaxMPC(s2,inArr,outArr);

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    ArgmaxTimeInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    ArgmaxCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
    if (party == SERVER) {
        funcReconstruct2PCCons(nullptr, inArr, s1*s2);
        funcReconstruct2PCCons(nullptr, outArr, s1);
    } 
    else {
        signedIntType* VinArr = new signedIntType[s1*s2];
        funcReconstruct2PCCons(VinArr, inArr, s1*s2);
        signedIntType* VoutArr = new signedIntType[s1];
        funcReconstruct2PCCons(VoutArr, outArr, s1);

        std::vector<std::vector<uint64_t>> VinVec;
        VinVec.resize(s1,std::vector<uint64_t>(s2,0));
        std::vector<uint64_t> VoutVec(s1);

        for(int i = 0; i < s1; i++) {
            for(int j = 0; j < s2; j++) {
                VinVec[i][j] = getRingElt(Arr2DIdxRowM(VinArr,s1,s2,i,j));
            }
        }

        ArgMax_pt(s1, s2, VinVec, VoutVec);

        bool pass = true;
        for(int i = 0; i < s1; i++) {
            if(VoutArr[i] != getSignedVal(VoutVec[i])) {
                pass = false;
                std::cout << VoutArr[i] << "\t" << getSignedVal(VoutVec[i]) << std::endl;
            }
        }
        if (pass == true) std::cout << GREEN << "ArgMax1 Output Matches" << RESET << std::endl;
        else std::cout << RED << "ArgMax1 Output Mismatch" << RESET << std::endl;

        delete[] VinArr;
        delete[] VoutArr;
    }
#endif
}


void Relu(int32_t size, intType* inArr, intType* outArr, int sf, bool doTruncation)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    static int ctr = 1;
    std::cout<<"Relu "<<ctr<<" called size="<<size<<std::endl;
    ctr++;

    intType moduloMask = sci::all1Mask(bitlength);
    int eightDivElemts = ((size + 8 - 1)/8)*8; //(ceil of s1*s2/8.0)*8
    uint8_t* msbShare = new uint8_t[eightDivElemts];
    intType* tempInp = new intType[eightDivElemts];
    intType* tempOutp = new intType[eightDivElemts];
    sci::copyElemWisePadded(size, inArr, eightDivElemts, tempInp, 0);

#ifndef MULTITHREADED_NONLIN
    reluImpl->relu(tempOutp, tempInp, eightDivElemts, nullptr);
#else
    std::thread relu_threads[numThreads];
    int chunk_size = (eightDivElemts/(8*numThreads))*8;
    for (int i = 0; i < numThreads; ++i) {
        int offset = i*chunk_size;
        int lnum_relu;
        if (i == (numThreads - 1)) {
            lnum_relu = eightDivElemts - offset;
        } else {
            lnum_relu = chunk_size;
        }
        relu_threads[i] = std::thread(funcReLUThread, i, tempOutp+offset, tempInp+offset, lnum_relu, nullptr, false);
    }
    for (int i = 0; i < numThreads; ++i) {
        relu_threads[i].join();
    }
#endif

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    ReluTimeInMilliSec += temp;
    std::cout<<"Time in sec for current relu = "<<(temp/1000.0)<<std::endl; 
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    ReluCommSent += curComm;
#endif

    if (doTruncation) 
    {
#ifdef LOG_LAYERWISE
        INIT_ALL_IO_DATA_SENT;
        INIT_TIMER;
#endif
        for(int i=0;i<eightDivElemts;i++){
            msbShare[i] = 0; //After relu, all numbers are +ve
        }

#ifdef SCI_OT
        for(int i=0;i<eightDivElemts;i++){
            tempOutp[i] = tempOutp[i] & moduloMask;
        }
        funcTruncateTwoPowerRingWrapper(eightDivElemts, tempOutp, outArr, sf, msbShare, true);
#else
        funcFieldDivWrapper<intType>(size, tempOutp, outArr, 1ULL<<sf, msbShare);
#endif

#ifdef LOG_LAYERWISE
        auto temp = TIMER_TILL_NOW;
        TruncationTimeInMilliSec += temp;
        uint64_t curComm;
        FIND_ALL_IO_TILL_NOW(curComm);
        TruncationCommSent += curComm;
#endif
    } 
    else {
        for(int i=0;i<size;i++){
            outArr[i] = tempOutp[i];
        }
    }

#ifdef SCI_OT
    for(int i=0;i<size;i++){
        outArr[i] = outArr[i] & moduloMask;
    }
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
    for(int i=0;i<size;i++){
        assert(tempOutp[i] < prime_mod);
        assert(outArr[i] < prime_mod);
    }
#endif

    if (party == SERVER){
        funcReconstruct2PCCons(nullptr, inArr, size);
        funcReconstruct2PCCons(nullptr, tempOutp, size);
        funcReconstruct2PCCons(nullptr, outArr, size);
    }
    else{
        signedIntType* VinArr = new signedIntType[size];
        funcReconstruct2PCCons(VinArr, inArr, size);
        signedIntType* VtempOutpArr = new signedIntType[size];
        funcReconstruct2PCCons(VtempOutpArr, tempOutp, size);
        signedIntType* VoutArr = new signedIntType[size];
        funcReconstruct2PCCons(VoutArr, outArr, size);

        std::vector<uint64_t> VinVec;
        VinVec.resize(size,0);

        std::vector<uint64_t> VoutVec;
        VoutVec.resize(size,0);

        for(int i=0;i<size;i++){
            VinVec[i] = getRingElt(VinArr[i]);
        }

        Relu_pt(size, VinVec, VoutVec, 0, false); // sf = 0

        bool pass = true;
        for(int i=0;i<size;i++){
            if (VtempOutpArr[i] != getSignedVal(VoutVec[i])) {
                pass = false;
            }
        }
        if (pass == true) std::cout << GREEN << "ReLU Output Matches" << RESET << std::endl;
        else std::cout << RED << "ReLU Output Mismatch" << RESET << std::endl;

        ScaleDown_pt(size, VoutVec, sf);

        pass = true;
        for(int i=0;i<size;i++){
            if (VoutArr[i] != getSignedVal(VoutVec[i])) {
                pass = false;
            }
        }
        if (pass == true) std::cout << GREEN << "Truncation (after ReLU) Output Matches" << RESET << std::endl;
        else std::cout << RED << "Truncation (after ReLU) Output Mismatch" << RESET << std::endl;

        delete[] VinArr;
        delete[] VtempOutpArr;
        delete[] VoutArr;
    }
#endif

    delete[] tempInp;
    delete[] tempOutp;
    delete[] msbShare;
}

void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, 
        int32_t ksizeH, int32_t ksizeW,
        int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
        int32_t strideH, int32_t strideW,
        int32_t N1, int32_t imgH, int32_t imgW, int32_t C1,
        intType* inArr, 
        intType* outArr)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    static int ctr = 1;
    std::cout<<"Maxpool "<<ctr<<" called N="<<N<<", H="<<H<<", W="<<W<<", C="<<C<<", ksizeH="<<ksizeH<<", ksizeW="<<ksizeW<<std::endl;
    ctr++;

    uint64_t moduloMask = sci::all1Mask(bitlength);
    int rowsOrig = N*H*W*C;
    int rows = ((rowsOrig + 8 - 1)/8)*8; //(ceil of rows/8.0)*8
    int cols = ksizeH*ksizeW;

    intType* reInpArr = new intType[rows*cols];
    intType* maxi = new intType[rows];
    intType* maxiIdx = new intType[rows];

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

                            intType temp = 0;
                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
                                temp = 0;
                            }
                            else{
                                temp = Arr4DIdxRowM(inArr,N,imgH,imgW,C,n,curPosH,curPosW,c);
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

    for(int i=rowsOrig;i<rows;i++){
        reInpArr[i] = 0; //The extra padded values
    }

#ifndef MULTITHREADED_NONLIN
    maxpoolImpl->funcMaxMPC(rows, cols, reInpArr, maxi, maxiIdx);
#else
    std::thread maxpool_threads[numThreads];
    int chunk_size = (rows/(8*numThreads))*8;
    for (int i = 0; i < numThreads; ++i) {
        int offset = i*chunk_size;
        int lnum_rows;
        if (i == (numThreads - 1)) {
            lnum_rows = rows - offset;
        } else {
            lnum_rows = chunk_size;
        }
        maxpool_threads[i] = std::thread(funcMaxpoolThread, i, lnum_rows, cols,
                reInpArr+offset*cols, maxi+offset, maxiIdx+offset);
    }
    for (int i = 0; i < numThreads; ++i) {
        maxpool_threads[i].join();
    }
#endif

    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H;h++){
                for(int w=0;w<W;w++){
                    int iidx = n*C*H*W + c*H*W + h*W + w;
                    Arr4DIdxRowM(outArr,N,H,W,C,n,h,w,c) = maxi[iidx];
#ifdef SCI_OT
                    Arr4DIdxRowM(outArr,N,H,W,C,n,h,w,c) = Arr4DIdxRowM(outArr,N,H,W,C,n,h,w,c) & moduloMask;
#endif
                }
            }
        }
    }

    delete[] reInpArr;
    delete[] maxi;
    delete[] maxiIdx;

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    MaxpoolTimeInMilliSec += temp;
    std::cout<<"Time in sec for current maxpool = "<<(temp/1000.0)<<std::endl; 
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    MaxpoolCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
    for(int i=0;i<N;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                for(int p=0;p<C;p++){
                    assert(Arr4DIdxRowM(outArr,N,H,W,C,i,j,k,p) < prime_mod);
                }
            }
        }
    }
#endif
    if (party == SERVER) {
        funcReconstruct2PCCons(nullptr, inArr, N*imgH*imgW*C);
        funcReconstruct2PCCons(nullptr, outArr, N*H*W*C);
    } 
    else {
        signedIntType* VinArr = new signedIntType[N*imgH*imgW*C];
        funcReconstruct2PCCons(VinArr, inArr, N*imgH*imgW*C);
        signedIntType* VoutArr = new signedIntType[N*H*W*C];
        funcReconstruct2PCCons(VoutArr, outArr, N*H*W*C);

        std::vector< std::vector< std::vector< std::vector<uint64_t>>>> VinVec;
        VinVec.resize(N,std::vector< std::vector< std::vector<uint64_t>>>(imgH,std::vector< std::vector<uint64_t>>(imgW,std::vector<uint64_t>(C,0))));

        std::vector< std::vector< std::vector< std::vector<uint64_t>>>> VoutVec;
        VoutVec.resize(N,std::vector< std::vector< std::vector<uint64_t>>>(H,std::vector< std::vector<uint64_t>>(W,std::vector<uint64_t>(C,0))));

        for(int i=0;i<N;i++){
            for(int j=0;j<imgH;j++){
                for(int k=0;k<imgW;k++){
                    for(int p=0;p<C;p++){
                        VinVec[i][j][k][p] = getRingElt(Arr4DIdxRowM(VinArr,N,imgH,imgW,C,i,j,k,p));
                    }
                }
            }
        }

        MaxPool_pt(N, H, W, C, ksizeH, ksizeW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, N1, imgH, imgW, C1, VinVec, VoutVec);

        bool pass = true;
        for(int i=0;i<N;i++){
            for(int j=0;j<H;j++){
                for(int k=0;k<W;k++){
                    for(int p=0;p<C;p++){
                        if (Arr4DIdxRowM(VoutArr,N,H,W,C,i,j,k,p) != getSignedVal(VoutVec[i][j][k][p])) {
                            pass = false;
                        }
                    }
                }
            }
        }
        if (pass == true) std::cout << GREEN << "Maxpool Output Matches" << RESET << std::endl;
        else std::cout << RED << "Maxpool Output Mismatch" << RESET << std::endl;

        delete[] VinArr;
        delete[] VoutArr;
    }
#endif
}

void AvgPool(int32_t N, int32_t H, int32_t W, int32_t C, 
        int32_t ksizeH, int32_t ksizeW,
        int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
        int32_t strideH, int32_t strideW,
        int32_t N1, int32_t imgH, int32_t imgW, int32_t C1,
        intType* inArr, 
        intType* outArr)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    static int ctr = 1;
    std::cout<<"AvgPool "<<ctr<<" called N="<<N<<", H="<<H<<", W="<<W<<", C="<<C<<", ksizeH="<<ksizeH<<", ksizeW="<<ksizeW<<std::endl;
    ctr++;

    uint64_t moduloMask = sci::all1Mask(bitlength);
    int rows = N*H*W*C;
    int rowsPadded = ((rows + 8 - 1)/8)*8;
    intType* filterSum = new intType[rowsPadded];
    intType* filterAvg = new intType[rowsPadded];

    int rowIdx = 0;
    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            int32_t leftTopCornerH = -zPadHLeft;
            int32_t extremeRightBottomCornerH = imgH - 1 + zPadHRight;
            while((leftTopCornerH + ksizeH - 1) <= extremeRightBottomCornerH){
                int32_t leftTopCornerW = -zPadWLeft;
                int32_t extremeRightBottomCornerW = imgW - 1 + zPadWRight;
                while((leftTopCornerW + ksizeW - 1) <= extremeRightBottomCornerW){

                    intType curFilterSum = 0;
                    for(int fh=0;fh<ksizeH;fh++){
                        for(int fw=0;fw<ksizeW;fw++){
                            int32_t curPosH = leftTopCornerH + fh;
                            int32_t curPosW = leftTopCornerW + fw;

                            intType temp = 0;
                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))){
                                temp = 0;
                            }
                            else{
                                temp = Arr4DIdxRowM(inArr,N,imgH,imgW,C,n,curPosH,curPosW,c);
                            }
#ifdef SCI_OT
                            curFilterSum += temp;
#else
                            curFilterSum = sci::neg_mod(curFilterSum+temp,(int64_t)prime_mod);
#endif
                        }
                    }

                    filterSum[rowIdx] = curFilterSum;
                    rowIdx += 1;
                    leftTopCornerW = leftTopCornerW + strideW;
                }

                leftTopCornerH = leftTopCornerH + strideH;
            }
        }
    }

    for(int i=rows;i<rowsPadded;i++){
        filterSum[i] = 0;
    }

#ifdef SCI_OT
    for(int i=0;i<rowsPadded;i++){
        filterSum[i] = filterSum[i] & moduloMask;
    }
    funcAvgPoolTwoPowerRingWrapper(rowsPadded,filterSum,filterAvg,ksizeH*ksizeW);
#else
    funcFieldDivWrapper<intType>(rowsPadded,filterSum,filterAvg,ksizeH*ksizeW,nullptr);
#endif

    for(int n=0;n<N;n++){
        for(int c=0;c<C;c++){
            for(int h=0;h<H;h++){
                for(int w=0;w<W;w++){
                    int iidx = n*C*H*W + c*H*W + h*W + w;
                    Arr4DIdxRowM(outArr,N,H,W,C,n,h,w,c) = filterAvg[iidx];
#ifdef SCI_OT
                    Arr4DIdxRowM(outArr,N,H,W,C,n,h,w,c) = Arr4DIdxRowM(outArr,N,H,W,C,n,h,w,c) & moduloMask;
#endif
                }
            }
        }
    }

    delete[] filterSum;
    delete[] filterAvg;

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    AvgpoolTimeInMilliSec += temp;
    std::cout<<"Time in sec for current avgpool = "<<(temp/1000.0)<<std::endl; 
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    AvgpoolCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
    for(int i=0;i<N;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                for(int p=0;p<C;p++){
                    assert(Arr4DIdxRowM(outArr,N,H,W,C,i,j,k,p) < prime_mod);
                }
            }
        }
    }
#endif
    if (party == SERVER) {
        funcReconstruct2PCCons(nullptr, inArr, N*imgH*imgW*C);
        funcReconstruct2PCCons(nullptr, outArr, N*H*W*C);
    } 
    else {
        signedIntType* VinArr = new signedIntType[N*imgH*imgW*C];
        funcReconstruct2PCCons(VinArr, inArr, N*imgH*imgW*C);
        signedIntType* VoutArr = new signedIntType[N*H*W*C];
        funcReconstruct2PCCons(VoutArr, outArr, N*H*W*C);

        std::vector< std::vector< std::vector< std::vector<uint64_t>>>> VinVec;
        VinVec.resize(N,std::vector< std::vector< std::vector<uint64_t>>>(imgH,std::vector< std::vector<uint64_t>>(imgW,std::vector<uint64_t>(C,0))));

        std::vector< std::vector< std::vector< std::vector<uint64_t>>>> VoutVec;
        VoutVec.resize(N,std::vector< std::vector< std::vector<uint64_t>>>(H,std::vector< std::vector<uint64_t>>(W,std::vector<uint64_t>(C,0))));

        for(int i=0;i<N;i++){
            for(int j=0;j<imgH;j++){
                for(int k=0;k<imgW;k++){
                    for(int p=0;p<C;p++){
                        VinVec[i][j][k][p] = getRingElt(Arr4DIdxRowM(VinArr,N,imgH,imgW,C,i,j,k,p));
                    }
                }
            }
        }

        AvgPool_pt(N, H, W, C, ksizeH, ksizeW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, N1, imgH, imgW, C1, VinVec, VoutVec);

        bool pass = true;
        for(int i=0;i<N;i++){
            for(int j=0;j<H;j++){
                for(int k=0;k<W;k++){
                    for(int p=0;p<C;p++){
                        if (Arr4DIdxRowM(VoutArr,N,H,W,C,i,j,k,p) != getSignedVal(VoutVec[i][j][k][p])) {
                            pass = false;
                        }
                    }
                }
            }
        }

        if (pass == true) std::cout << GREEN << "AvgPool Output Matches" << RESET << std::endl;
        else std::cout << RED << "AvgPool Output Mismatch" << RESET << std::endl;

        delete[] VinArr;
        delete[] VoutArr;
    }
#endif
}

void Conv2DWrapper(signedIntType N, signedIntType H, signedIntType W, signedIntType CI, 
        signedIntType FH, signedIntType FW, signedIntType CO, 
        signedIntType zPadHLeft, signedIntType zPadHRight, 
        signedIntType zPadWLeft, signedIntType zPadWRight, 
        signedIntType strideH, signedIntType strideW, 
        intType* inputArr, 
        intType* filterArr, 
        intType* outArr)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    static int ctr = 1;
    std::cout<<"Conv2DCSF "<<ctr<<" called N="<<N<<", H="<<H<<", W="<<W<<", CI="<<CI<<", FH="<<FH<<", FW="<<FW<<", CO="<<CO<<", S="<<strideH<<std::endl;
    ctr++;

    signedIntType newH = (((H + (zPadHLeft+zPadHRight) - FH)/strideH) + 1);
    signedIntType newW = (((W + (zPadWLeft+zPadWRight) - FW)/strideW) + 1);

#ifdef SCI_OT
    // If its a ring, then its a OT based -- use the default Conv2DCSF implementation that comes from the EzPC library
    Conv2D(N,H,W,CI,FH,FW,CO,zPadHLeft,zPadHRight,zPadWLeft,zPadWRight,strideH,strideW,inputArr,filterArr,outArr);
#else
    // If its a field, then its a HE based -- use the HE based conv implementation
    std::vector< std::vector< std::vector< std::vector<intType>>>> inputVec;
    inputVec.resize(N,std::vector< std::vector< std::vector<intType>>>(H,std::vector< std::vector<intType>>(W,std::vector<intType>(CI,0))));

    std::vector< std::vector< std::vector< std::vector<intType>>>> filterVec;
    filterVec.resize(FH,std::vector< std::vector< std::vector<intType>>>(FW,std::vector< std::vector<intType>>(CI,std::vector<intType>(CO,0))));

    std::vector< std::vector< std::vector< std::vector<intType>>>> outputVec;
    outputVec.resize(N,std::vector< std::vector< std::vector<intType>>>(newH,std::vector< std::vector<intType>>(newW,std::vector<intType>(CO,0))));

    for(int i=0;i<N;i++){
        for(int j=0;j<H;j++){
            for(int k=0;k<W;k++){
                for(int p=0;p<CI;p++){
                    inputVec[i][j][k][p] = Arr4DIdxRowM(inputArr,N,H,W,CI,i,j,k,p);
                }
            }
        }
    }
    for(int i=0;i<FH;i++){
        for(int j=0;j<FW;j++){
            for(int k=0;k<CI;k++){
                for(int p=0;p<CO;p++){
                    filterVec[i][j][k][p] = Arr4DIdxRowM(filterArr,FH,FW,CI,CO,i,j,k,p);
                }
            }
        }
    }

    heConvImpl->convolution(N,H,W,CI,FH,FW,CO,
            zPadHLeft,zPadHRight,zPadWLeft,zPadWRight,strideH,strideW,inputVec,filterVec,outputVec);

    for(int i=0;i<N;i++){
        for(int j=0;j<newH;j++){
            for(int k=0;k<newW;k++){
                for(int p=0;p<CO;p++){
                    Arr4DIdxRowM(outArr,N,newH,newW,CO,i,j,k,p) = outputVec[i][j][k][p];
                }
            }
        }
    }

#endif

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    ConvTimeInMilliSec += temp;
    std::cout<<"Time in sec for current conv = "<<(temp/1000.0)<<std::endl; 
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    ConvCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
    for(int i=0;i<N;i++){
        for(int j=0;j<newH;j++){
            for(int k=0;k<newW;k++){
                for(int p=0;p<CO;p++){
                    assert(Arr4DIdxRowM(outArr,N,newH,newW,CO,i,j,k,p) < prime_mod);
                }
            }
        }
    }
#endif
    if (party == SERVER) {
        funcReconstruct2PCCons(nullptr, inputArr, N*H*W*CI);
        funcReconstruct2PCCons(nullptr, filterArr, FH*FW*CI*CO);
        funcReconstruct2PCCons(nullptr, outArr, N*newH*newW*CO);
    } 
    else {
        signedIntType* VinputArr = new signedIntType[N*H*W*CI];
        funcReconstruct2PCCons(VinputArr, inputArr, N*H*W*CI);
        signedIntType* VfilterArr = new signedIntType[FH*FW*CI*CO];
        funcReconstruct2PCCons(VfilterArr, filterArr, FH*FW*CI*CO);
        signedIntType* VoutputArr = new signedIntType[N*newH*newW*CO];
        funcReconstruct2PCCons(VoutputArr, outArr, N*newH*newW*CO);

        std::vector< std::vector< std::vector< std::vector<uint64_t>>>> VinputVec;
        VinputVec.resize(N,std::vector< std::vector< std::vector<uint64_t>>>(H,std::vector< std::vector<uint64_t>>(W,std::vector<uint64_t>(CI,0))));

        std::vector< std::vector< std::vector< std::vector<uint64_t>>>> VfilterVec;
        VfilterVec.resize(FH,std::vector< std::vector< std::vector<uint64_t>>>(FW,std::vector< std::vector<uint64_t>>(CI,std::vector<uint64_t>(CO,0))));

        std::vector< std::vector< std::vector< std::vector<uint64_t>>>> VoutputVec;
        VoutputVec.resize(N,std::vector< std::vector< std::vector<uint64_t>>>(newH,std::vector< std::vector<uint64_t>>(newW,std::vector<uint64_t>(CO,0))));

        for(int i=0;i<N;i++){
            for(int j=0;j<H;j++){
                for(int k=0;k<W;k++){
                    for(int p=0;p<CI;p++){
                        VinputVec[i][j][k][p] = getRingElt(Arr4DIdxRowM(VinputArr,N,H,W,CI,i,j,k,p));
                    }
                }
            }
        }
        for(int i=0;i<FH;i++){
            for(int j=0;j<FW;j++){
                for(int k=0;k<CI;k++){
                    for(int p=0;p<CO;p++){
                        VfilterVec[i][j][k][p] = getRingElt(Arr4DIdxRowM(VfilterArr,FH,FW,CI,CO,i,j,k,p));
                    }
                }
            }
        }

        Conv2DWrapper_pt(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, VinputVec, VfilterVec, VoutputVec); // consSF = 0

        bool pass = true;
        for(int i=0;i<N;i++){
            for(int j=0;j<newH;j++){
                for(int k=0;k<newW;k++){
                    for(int p=0;p<CO;p++){
                        if (Arr4DIdxRowM(VoutputArr,N,newH,newW,CO,i,j,k,p) != getSignedVal(VoutputVec[i][j][k][p])) {
                            pass = false;
                        }
                    }
                }
            }
        }
        if (pass == true) std::cout << GREEN << "Convolution Output Matches" << RESET << std::endl;
        else std::cout << RED << "Convolution Output Mismatch" << RESET << std::endl;

        delete[] VinputArr;
        delete[] VfilterArr;
        delete[] VoutputArr;
    }
#endif
}

void ElemWiseActModelVectorMult(int32_t size, intType* inArr, intType* multArrVec, intType* outputArr)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    if (party==CLIENT){
        for(int i=0;i<size;i++){
                assert((multArrVec[i] == 0)
                    && "The semantics of ElemWiseActModelVectorMult dictate multArrVec should be the model and client share should be 0 for it.");
        }
    }

    static int batchNormCtr = 1;
    std::cout<<"Starting fused batchNorm #"<<batchNormCtr<<std::endl;
    batchNormCtr++;

#ifdef SCI_OT
#ifdef MULTITHREADED_DOTPROD
    std::thread dotProdThreads[numThreads];
    int chunk_size = (size/numThreads);
    intType* inputArrPtr;
    if (party==SERVER){
        inputArrPtr = multArrVec;
    }
    else{
        inputArrPtr = inArr;
    }
    for (int i = 0; i < numThreads; i++) {
        int offset = i*chunk_size;
        int curSize;
        if (i == (numThreads - 1)) {
            curSize = size - offset;
        } 
        else{
            curSize = chunk_size;
        }
        dotProdThreads[i] = std::thread(funcDotProdThread, 
                i, numThreads, curSize, inputArrPtr+offset, outputArr+offset);
    }
    for (int i = 0; i < numThreads; ++i) {
        dotProdThreads[i].join();
    }
#else
    if (party==SERVER){
        matmulImpl->funcDotProdOTSender(size,multArrVec,outputArr,iknpOT);
    }
    else{
        matmulImpl->funcDotProdOTReceiver(size,inArr,outputArr,iknpOT);
    }
#endif

    if (party==SERVER){
        for(int i=0;i<size;i++){
            outputArr[i] += (inArr[i]*multArrVec[i]);
        }
    }
    else{
        for(int i=0;i<size;i++){
            assert(multArrVec[i]==0 && "Client's share of model is non-zero.");
        }
    }

#else // SCI-HE
    std::vector<uint64_t> tempInArr(size);
    std::vector<uint64_t> tempOutArr(size);
    std::vector<uint64_t> tempMultArr(size);

    for(int i = 0; i < size; i++) {
        tempInArr[i] = inArr[i];
        tempMultArr[i] = multArrVec[i];
    }

    heProdImpl->elemwise_product(size, tempInArr, tempMultArr, tempOutArr);

    for(int i = 0; i < size; i++) {
        outputArr[i] = tempOutArr[i];
    }
#endif

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    BatchNormInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    BatchNormCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
    for(int i=0;i<size;i++){
        assert(outputArr[i] < prime_mod);
    }
#endif
    if (party == SERVER) {
        funcReconstruct2PCCons(nullptr, inArr, size);
        funcReconstruct2PCCons(nullptr, multArrVec, size);
        funcReconstruct2PCCons(nullptr, outputArr, size);
    }
    else {
        signedIntType* VinArr = new signedIntType[size];
        funcReconstruct2PCCons(VinArr, inArr, size);
        signedIntType* VmultArr = new signedIntType[size];
        funcReconstruct2PCCons(VmultArr, multArrVec, size);
        signedIntType* VoutputArr = new signedIntType[size];
        funcReconstruct2PCCons(VoutputArr, outputArr, size);

        std::vector<uint64_t> VinVec(size);
        std::vector<uint64_t> VmultVec(size);
        std::vector<uint64_t> VoutputVec(size);

        for(int i=0;i<size;i++){
            VinVec[i] = getRingElt(VinArr[i]);
            VmultVec[i] = getRingElt(VmultArr[i]);
        }

        ElemWiseActModelVectorMult_pt(size, VinVec, VmultVec, VoutputVec);

        bool pass = true;
        for(int i=0;i<size;i++){
            if (VoutputArr[i] != getSignedVal(VoutputVec[i])){
                pass = false;
            }
        }
        if (pass == true) std::cout << GREEN << "ElemWiseSecretVectorMult Output Matches" << RESET << std::endl;
        else std::cout << RED << "ElemWiseSecretVectorMult Output Mismatch" << RESET << std::endl;

        delete[] VinArr;
        delete[] VmultArr;
        delete[] VoutputArr;
    }
#endif
}

void ScaleDown(int32_t size, intType* inArr, int32_t sf)
{
#ifdef LOG_LAYERWISE
    INIT_ALL_IO_DATA_SENT;
    INIT_TIMER;
#endif

    int eightDivElemts = ((size + 8 - 1)/8)*8; //(ceil of s1*s2/8.0)*8
    intType* tempInp;
    if(size != eightDivElemts) {
        tempInp = new intType[eightDivElemts];
        memcpy(tempInp, inArr, sizeof(intType)*size);
    } 
    else {
        tempInp = inArr;
    }
    uint64_t moduloMask = sci::all1Mask(bitlength);
    for(int i=0;i<eightDivElemts;i++){
        tempInp[i] = tempInp[i] & moduloMask;
    }
    intType* outp = new intType[eightDivElemts];

#ifdef SCI_OT    
    funcTruncateTwoPowerRingWrapper(eightDivElemts, tempInp, outp, sf, nullptr);
#else
    funcFieldDivWrapper<intType>(eightDivElemts, tempInp, outp, 1ULL<<sf, nullptr);
#endif

#ifdef LOG_LAYERWISE
    auto temp = TIMER_TILL_NOW;
    TruncationTimeInMilliSec += temp;
    uint64_t curComm;
    FIND_ALL_IO_TILL_NOW(curComm);
    TruncationCommSent += curComm;
#endif

#ifdef VERIFY_LAYERWISE
#ifdef SCI_HE
    for(int i=0;i<size;i++){
        assert(outp[i] < prime_mod);
    }
#endif

    if (party == SERVER) {
        funcReconstruct2PCCons(nullptr, inArr, size);
        funcReconstruct2PCCons(nullptr, outp, size);
    } 
    else {
        signedIntType* VinArr = new signedIntType[size];
        funcReconstruct2PCCons(VinArr, inArr, size);
        signedIntType* VoutpArr = new signedIntType[size];
        funcReconstruct2PCCons(VoutpArr, outp, size);

        std::vector<uint64_t> VinVec;
        VinVec.resize(size,0);

        for(int i=0;i<size;i++){
            VinVec[i] = getRingElt(VinArr[i]);
        }

        ScaleDown_pt(size, VinVec, sf);

        bool pass = true;
        for(int i=0;i<size;i++){
            if (VoutpArr[i] != getSignedVal(VinVec[i])){
                pass = false;
            }
        }

        if (pass == true) std::cout << GREEN << "Truncation4 Output Matches" << RESET << std::endl;
        else std::cout << RED << "Truncation4 Output Mismatch" << RESET << std::endl;

        delete[] VinArr;
        delete[] VoutpArr;
    }
#endif

    memcpy(inArr, outp, sizeof(intType)*size);
    delete[] outp;
    if(size != eightDivElemts)
        delete[] tempInp;
}

void ScaleUp(int32_t size, intType* arr, int32_t sf){
    for(int i=0;i<size;i++){
#ifdef SCI_OT
        arr[i] = (arr[i]<<sf);
#else
        arr[i] = sci::neg_mod(arr[i]<<sf, (int64_t)prime_mod);
#endif
    }
}

void StartComputation(){
    startTimeTracker = std::chrono::high_resolution_clock::now();
    for(int i=0;i<numThreads;i++){
        auto temp = ioArr[i]->counter;
        communicationTracker[i] = temp;
        std::cout<<"Thread i = "<<i<<", total data sent till now = "<<temp<<std::endl;
    }
    std::cout<<"-----------Syncronizing-----------"<<std::endl;
    io->sync();
    std::cout<<"-----------Syncronized - now starting execution-----------"<<std::endl;
}

void EndComputation(){
    auto endTimer = std::chrono::high_resolution_clock::now();
    auto execTimeInMilliSec = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer-startTimeTracker).count();
    uint64_t totalComm = 0;
    for(int i=0;i<numThreads;i++){
        auto temp = ioArr[i]->counter;
        std::cout<<"Thread i = "<<i<<", total data sent till now = "<<temp<<std::endl;
        totalComm += (temp - communicationTracker[i]);
    }
    uint64_t totalCommClient;
    std::cout<<"------------------------------------------------------\n";
    std::cout<<"------------------------------------------------------\n";
    std::cout<<"------------------------------------------------------\n";
    std::cout<<"Total time taken = "<<execTimeInMilliSec<<" milliseconds.\n";
    std::cout<<"Total data sent = "<<(totalComm/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    if (party==SERVER){
        io->recv_data(&totalCommClient, sizeof(uint64_t));
        std::cout<<"Total comm (sent+received) = "<<((totalComm+totalCommClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    }
    else if (party==CLIENT){
        io->send_data(&totalComm, sizeof(uint64_t));
        std::cout<<"Total comm (sent+received) = (see SERVER OUTPUT)"<<std::endl;
    }
    std::cout<<"------------------------------------------------------\n";

#ifdef LOG_LAYERWISE
    std::cout<<"Total time in Conv = "<<(ConvTimeInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"Total time in Matmul = "<<(MatmulTimeInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"Total time in BatchNorm = "<<(BatchNormInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"Total time in Truncation = "<<(TruncationTimeInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"Total time in Relu = "<<(ReluTimeInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"Total time in MaxPool = "<<(MaxpoolTimeInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"Total time in AvgPool = "<<(AvgpoolTimeInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"Total time in Argmax = "<<(ArgmaxTimeInMilliSec/1000.0)<<" seconds."<<std::endl;
    std::cout<<"------------------------------------------------------\n";
    std::cout<<"Conv data sent = "<<((ConvCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"Matmul data sent = "<<((MatmulCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"BatchNorm data sent = "<<((BatchNormCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"Truncation data sent = "<<((TruncationCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"Relu data sent = "<<((ReluCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"Maxpool data sent = "<<((MaxpoolCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"Avgpool data sent = "<<((AvgpoolCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"Argmax data sent = "<<((ArgmaxCommSent)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
    std::cout<<"------------------------------------------------------\n";
    if (party==SERVER){
        uint64_t ConvCommSentClient = 0;
        uint64_t MatmulCommSentClient = 0;
        uint64_t BatchNormCommSentClient = 0;
        uint64_t TruncationCommSentClient = 0;
        uint64_t ReluCommSentClient = 0;
        uint64_t MaxpoolCommSentClient = 0;
        uint64_t AvgpoolCommSentClient = 0;
        uint64_t ArgmaxCommSentClient = 0;
        io->recv_data(&ConvCommSentClient, sizeof(uint64_t));
        io->recv_data(&MatmulCommSentClient, sizeof(uint64_t));
        io->recv_data(&BatchNormCommSentClient, sizeof(uint64_t));
        io->recv_data(&TruncationCommSentClient, sizeof(uint64_t));
        io->recv_data(&ReluCommSentClient, sizeof(uint64_t));
        io->recv_data(&MaxpoolCommSentClient, sizeof(uint64_t));
        io->recv_data(&AvgpoolCommSentClient, sizeof(uint64_t));
        io->recv_data(&ArgmaxCommSentClient, sizeof(uint64_t));
        std::cout<<"Conv data (sent+received) = "<<((ConvCommSent+ConvCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
        std::cout<<"Matmul data (sent+received) = "<<((MatmulCommSent+MatmulCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
        std::cout<<"BatchNorm data (sent+received) = "<<((BatchNormCommSent+BatchNormCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
        std::cout<<"Truncation data (sent+received) = "<<((TruncationCommSent+TruncationCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
        std::cout<<"Relu data (sent+received) = "<<((ReluCommSent+ReluCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
        std::cout<<"Maxpool data (sent+received) = "<<((MaxpoolCommSent+MaxpoolCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
        std::cout<<"Avgpool data (sent+received) = "<<((AvgpoolCommSent+AvgpoolCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;
        std::cout<<"Argmax data (sent+received) = "<<((ArgmaxCommSent+ArgmaxCommSentClient)/(1.0*(1ULL<<20)))<<" MiB."<<std::endl;

#ifdef WRITE_LOG
        std::string file_addr = "results-Porthos2PC-server.csv";
        bool write_title = true; {
            std::fstream result(file_addr.c_str(), std::fstream::in);
            if(result.is_open())
                write_title = false;
            result.close();
        }
        std::fstream result(file_addr.c_str(), std::fstream::out|std::fstream::app);
        if(write_title){
            result << "Network,Algebra,Bitlen,Base,#Threads,Total Time,Total Comm,Conv Time,Conv Comm,MatMul Time,MatMul Comm,BatchNorm Time,BatchNorm Comm,Truncation Time,Truncation Comm,ReLU Time,ReLU Comm,MaxPool Time,MaxPool Comm,AvgPool Time,AvgPool Comm,ArgMax Time,ArgMax Comm" << std::endl;
        }
        result << get_network_label(network_name) << ","
            << (isNativeRing ? "Ring": "Field") << ","
            << bitlength << ","
            << baseForRelu << ","
            << numThreads << ","
            << execTimeInMilliSec/1000.0 << ","
            << (totalComm+totalCommClient)/(1.0*(1ULL<<20)) << ","
            << ConvTimeInMilliSec/1000.0 << ","
            << (ConvCommSent+ConvCommSentClient)/(1.0*(1ULL<<20)) << ","
            << MatmulTimeInMilliSec/1000.0 << ","
            << (MatmulCommSent+MatmulCommSentClient)/(1.0*(1ULL<<20)) << ","
            << BatchNormInMilliSec/1000.0 << ","
            << (BatchNormCommSent+BatchNormCommSentClient)/(1.0*(1ULL<<20)) << ","
            << TruncationTimeInMilliSec/1000.0 << ","
            << (TruncationCommSent+TruncationCommSentClient)/(1.0*(1ULL<<20)) << ","
            << ReluTimeInMilliSec/1000.0 << ","
            << (ReluCommSent+ReluCommSentClient)/(1.0*(1ULL<<20)) << ","
            << MaxpoolTimeInMilliSec/1000.0 << ","
            << (MaxpoolCommSent+MaxpoolCommSentClient)/(1.0*(1ULL<<20)) << ","
            << AvgpoolTimeInMilliSec/1000.0 << ","
            << (AvgpoolCommSent+AvgpoolCommSentClient)/(1.0*(1ULL<<20)) << ","
            << ArgmaxTimeInMilliSec/1000.0 << ","
            << (ArgmaxCommSent+ArgmaxCommSentClient)/(1.0*(1ULL<<20)) << std::endl;
        result.close();
#endif
    }
    else if (party==CLIENT){
        io->send_data(&ConvCommSent, sizeof(uint64_t));
        io->send_data(&MatmulCommSent, sizeof(uint64_t));
        io->send_data(&BatchNormCommSent, sizeof(uint64_t));
        io->send_data(&TruncationCommSent, sizeof(uint64_t));
        io->send_data(&ReluCommSent, sizeof(uint64_t));
        io->send_data(&MaxpoolCommSent, sizeof(uint64_t));
        io->send_data(&AvgpoolCommSent, sizeof(uint64_t));
        io->send_data(&ArgmaxCommSent, sizeof(uint64_t));
    }
#endif
}

inline void ClearMemSecret1(int32_t s1, intType* arr){
    delete[] arr;
}

inline void ClearMemSecret2(int32_t s1, int32_t s2, intType* arr){
    delete[] arr; //At the end of the day, everything is done using 1D array
}

inline void ClearMemSecret3(int32_t s1, int32_t s2, int32_t s3, intType* arr){
    delete[] arr;
}

inline void ClearMemSecret4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, intType* arr){
    delete[] arr;
}

inline void ClearMemSecret5(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, intType* arr){
    delete[] arr;
}

inline void ClearMemPublic(int32_t x){
    return;
}

inline void ClearMemPublic1(int32_t s1, int32_t* arr){
    delete[] arr;
}

inline void ClearMemPublic2(int32_t s1, int32_t s2, int32_t* arr){
    delete[] arr;
}

inline void ClearMemPublic3(int32_t s1, int32_t s2, int32_t s3, int32_t* arr){
    delete[] arr;
}

inline void ClearMemPublic4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t* arr){
    delete[] arr;
}

inline void ClearMemPublic5(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, int32_t* arr){
    delete[] arr;
}

inline void ClearMemPublic(int64_t x){
    return;
}

inline void ClearMemPublic1(int32_t s1, int64_t* arr){
    delete[] arr;
}

inline void ClearMemPublic2(int32_t s1, int32_t s2, int64_t* arr){
    delete[] arr;
}

inline void ClearMemPublic3(int32_t s1, int32_t s2, int32_t s3, int64_t* arr){
    delete[] arr;
}

inline void ClearMemPublic4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int64_t* arr){
    delete[] arr;
}

inline void ClearMemPublic5(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, int64_t* arr){
    delete[] arr;
}

intType SecretAdd(intType x, intType y){
#ifdef SCI_OT
    return (x+y);
#else
    return sci::neg_mod(x+y, (int64_t)prime_mod);
#endif
}

intType SecretSub(intType x, intType y){
#ifdef SCI_OT
    return (x-y);
#else
    return sci::neg_mod(x-y, (int64_t)prime_mod);
#endif
}

intType SecretMult(intType x, intType y){
    //Not being used in any of our networks right now
    assert(false);    
}

void ElemWiseVectorPublicDiv(int32_t s1, intType* arr1, int32_t divisor, intType* outArr){
    //Not being used in any of our networks right now
    assert(false);
}

void ElemWiseSecretSharedVectorMult(int32_t size, intType* inArr, intType* multArrVec, intType* outputArr){
    //Not being used in any of our networks right now
    assert(false);
}

void Floor(int32_t s1, intType* inArr, intType* outArr, int32_t sf){
    //Not being used in any of our networks right now
    assert(false);
}

#endif //FUNCTIONALITIES_WRAPPER_H__
