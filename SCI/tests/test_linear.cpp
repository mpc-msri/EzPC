/*
Authors: Nishant Kumar
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

#define BITLEN_35
#define SCI_OT

#include "globals.h"
#include "linear-primary.h"
using namespace std;
using namespace sci;

const int s1 = 2, s2 = 2, s3 = 2;

void testCOT_matmul(NetIO* io, IKNP<NetIO>* iknp, int party, OTIdeal<NetIO>* otIdeal){
	int numOTs = 3;
	intType corr[numOTs];
	uint32_t chunkSizes[numOTs];
	uint32_t numChunks[numOTs];
	intType data[numOTs];
	uint8_t choices[numOTs];
	for(int i=0;i<numOTs;i++) numChunks[i] = 1;
	for(int i=0;i<numOTs;i++) chunkSizes[i] = 64;
	for(int i=0;i<numOTs;i++) corr[i] = 0;
	for(int i=0;i<numOTs;i++) choices[i] = 0;
	corr[0] = 1;
	choices[0] = 1;
	if (party==ALICE){
		// otIdeal->send_cot_matmul(data, corr, chunkSizes, numChunks, numOTs);
		iknp->send_cot_matmul<intType>(data, corr, chunkSizes, numChunks, numOTs, 100);
	}
	else{
		// otIdeal->recv_cot_matmul(data, choices, chunkSizes, numChunks, numOTs);
		iknp->recv_cot_matmul<intType>(data, choices, chunkSizes, numChunks, numOTs, 100);
	}
	for(int i=0;i<numOTs;i++){
		cout<<"OT# = "<<i<<", data[i] = "<<data[i]<<endl;
	}
}

int main(int argc, char** argv){
	ArgMapping amap;
	int port = 32000;
	string serverAddr = "127.0.0.1";

	amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
	amap.arg("p", port, "Port Number");
	amap.arg("ip", serverAddr, "IP Address of server (ALICE)");
	amap.parse(argc, argv);

	assert(party==sci::ALICE || party==sci::BOB);

	checkIfUsingEigen();
	int numThreads=4;

	NetIO* io = new NetIO(party==ALICE ? nullptr : serverAddr.c_str(), port);
	NetIO* io1 = new NetIO(party==ALICE ? nullptr : serverAddr.c_str(), port+500);
	IKNP<NetIO>* iknpOT = new IKNP<NetIO>(io);
	IKNP<NetIO>* iknpOTRoleReversed = new IKNP<NetIO>(io1);

	Matmul<NetIO, intType, IKNP<NetIO>>* matmulImpl = 
					new Matmul<NetIO, intType, IKNP<NetIO>>(party, bitlength, io, iknpOT, iknpOTRoleReversed);

	// Following is the ideal ot declaration -- use if in doubt of OT implementation
	// OTIdeal<NetIO>* otIdeal = new OTIdeal<NetIO>(io);
	// Matmul<NetIO, intType, OTIdeal<NetIO>>* matmulImpl = 
	// 				new Matmul<NetIO, intType, OTIdeal<NetIO>>(party, bitlength, io, otIdeal, otIdeal);

	// testCOT_matmul(io, iknpOT, party, otIdeal);

	if (party==ALICE){
		iknpOT->setup_send();
		iknpOTRoleReversed->setup_recv();
	}
	else if (party==BOB){
		iknpOT->setup_recv();
		iknpOTRoleReversed->setup_send();
	}
	cout<<"After base ots, communication bytes = "<<(io->counter)<<endl;

	INIT_TIMER;
	intType* A = new intType[s1*s2];
	intType* B = new intType[s2*s3];
	intType* C = new intType[s1*s3];
	for(int i=0;i<s1;i++){
		for(int j=0;j<s2;j++){
			Arr2DIdxRowM(A,s1,s2,i,j) = (i+j+1)<<(bitlength-1);
		}
	}
	for(int i=0;i<s2;i++){
		for(int j=0;j<s3;j++){
			Arr2DIdxRowM(B,s2,s3,i,j) = (i+j+1)<<(bitlength-1);
		}
	}

	matmulImpl->ideal_func(s1,s2,s3,A,B,C);
	cout<<"Printing A : "<<endl;
	print2DArr<intType>(s1,s2,A);
	cout<<"Printing B : "<<endl;
	print2DArr<intType>(s2,s3,B);
	cout<<"Printing C : "<<endl;
	print2DArr<intType>(s1,s3,C);

	intType* C_share = new intType[s1*s3];
	cout<<"Starting single matmul test"<<endl;
	START_TIMER;
	if (party==ALICE){
		matmulImpl->funcOTSenderInputA(s1,s2,s3,A,C_share,iknpOT);
		// matmulImpl->funcOTReceiverInputA(s1,s2,s3,A,C_share,iknpOTRoleReversed);
		// matmulImpl->funcOTReceiverInputA(s1,s2,s3,A,C_share,otIdeal);
	}
	else if (party==BOB){
		matmulImpl->funcOTReceiverInputB(s1,s2,s3,B,C_share,iknpOT);
		// matmulImpl->funcOTSenderInputB(s1,s2,s3,B,C_share,iknpOTRoleReversed);
		// matmulImpl->funcOTSenderInputB(s1,s2,s3,B,C_share,otIdeal);
	}
	else{
		assert(false);
	}
	io->flush();
	PAUSE_TIMER("after single matmul");
	cout<<"Comm bytes till now = "<<(io->counter)<<endl;
	cout<<"Myshare for mult of A*B: "<<endl;
	print2DArr<intType>(s1,s3,C_share);

	intType* C_temp_share = new intType[s1*s3];
	intType* C_clear = new intType[s1*s3];
	if (party==ALICE){
		io->recv_data(C_temp_share,sizeof(intType)*s1*s3);
	}
	else if (party==BOB){
		io->send_data(C_share,sizeof(intType)*s1*s3);
	}

	// assert for equality
	if (party==ALICE){
		elemWiseAdd<intType>(s1*s3,C_share,C_temp_share,C_clear);
		for(int i=0;i<s1;i++){
			for(int j=0;j<s3;j++){
				Arr2DIdxRowM(C_clear,s1,s3,i,j) = Arr2DIdxRowM(C_clear,s1,s3,i,j) & ((1ULL<<bitlength)-1);
				assert((Arr2DIdxRowM(C_clear,s1,s3,i,j)==Arr2DIdxRowM(C,s1,s3,i,j)) && "equality failed");
			}
		}
	}

	// ------------------------------------------
	// beaver testing code

	cout<<"Beaver matmul test "<<endl;
	PRG128 prg;
	intType* bv_a = new intType[s1*s2];
	intType* bv_b = new intType[s2*s3];
	intType* bv_c = new intType[s1*s3];
	cout<<"Beaver starting timer"<<endl;
	START_TIMER
	matmulImpl->generateBeaverMatrixTriplet(s1,s2,s3,prg,bv_a,bv_b,bv_c);
	PAUSE_TIMER("Done with beaver triplets")
	matmulImpl->verifyMatmulShares(s1,s2,s3,bv_a,bv_b,bv_c);
	cout<<"Beaver triplets verified"<<endl;

	cout<<"Testing beaver online phase"<<endl;
	intType* bvo_x = new intType[s1*s2];
	intType* bvo_y = new intType[s2*s3];
	intType* bvo_z = new intType[s1*s3];
	prg.random_data(bvo_x,sizeof(intType)*s1*s2);
	prg.random_data(bvo_y,sizeof(intType)*s2*s3);
	cout<<"starting online phase"<<endl;
	START_TIMER
	matmulImpl->runBeaverOnlinePhase(s1,s2,s3,bvo_x,bvo_y,bvo_z,bv_a,bv_b,bv_c);
	PAUSE_TIMER("done with beaver online")
	matmulImpl->verifyMatmulShares(s1,s2,s3,bvo_x,bvo_y,bvo_z);

	cout<<"Beaver online phase testing done"<<endl;
	STOP_TIMER("foo")
}
