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

#define BITLEN_64
#define SCI_OT

#include "functionalities.h"
#include <iostream>
#include <cstdlib>
using namespace std;

/*
	Contains test cases for ring based truncation and avgpool + field based division.
	The three functions can be found at the bottom of main. 
	Change the args defined below to vary parameters.
	When switching from ring to field or vice-versa, switch the bitlen and SCI_OT/SCI_HE flag.
*/

const int N = 8000;
const int sf = 12;
const uint32_t divisor = 11;
const int server_random_seed = 84342;
const int client_random_seed = 32243;

#ifdef SCI_OT
void test_ring_truncation(){
	if (party==SERVER){
		srand(server_random_seed);
	}
	else{
		srand(client_random_seed);
	}
	intType inp[N];
	uint8_t msbShare[N];
	intType outp[N];
	uint64_t moduloMask = sci::all1Mask(bitlength);
	for(int i=0;i<N;i++){
		if (party==SERVER) inp[i] = rand() & moduloMask;
		else inp[i] = rand() & moduloMask;
	}
	funcTruncateTwoPowerRingWrapper(N,inp,outp,sf,nullptr);
	funcTruncationIdeal(N,inp,sf);
	bool anythingFailed = false;
	for(int i=0;i<N;i++){
		signedIntType ans = funcReconstruct2PCCons(inp[i],2);
		auto ansigot = funcReconstruct2PCCons(outp[i],2);
		if (party==sci::BOB){
			if(!((ans==ansigot))) {
				cout<<RED<<"Error "<<i<<" ans,ansigot = "<<ans<<" "<<ansigot<<endl;
				anythingFailed = true;
			}
			else {
				// cout<<GREEN<<"right ans "<<i<<" ans,ansigot = "<<ans<<" "<<ansigot<<endl;
			}
		}
	}
	cout<<"anythingFailed? = "<<anythingFailed<<endl;
}

void test_ring_avgpool(){
	if (party==SERVER){
		srand(server_random_seed);
	}
	else{
		srand(client_random_seed);
	}
	intType inp[N];
	intType outp[N];
	uint64_t moduloMask = sci::all1Mask(bitlength);
	for(int i=0;i<N;i++){
		if (party==SERVER) inp[i] = rand() & moduloMask;
		else inp[i] = rand() & moduloMask;
	}

	funcAvgPoolTwoPowerRingWrapper(N,inp,outp,divisor);
	bool anythingFailed = false;
	for(int i=0;i<N;i++){
		signedIntType inpClear = funcReconstruct2PCCons(inp[i],2);
		auto ans = div_floor(inpClear,divisor);
		auto ansigot = funcReconstruct2PCCons(outp[i],2);
		if (party==sci::BOB){
			if(ans!=ansigot) {
				cout<<RED<<"Error ans, ansigot :: "<<ans<<" "<<ansigot<<endl;
				anythingFailed = true;
			}
			else{
				// cout<<GREEN<<"right ans "<<i<<endl;
			}
		}
	}
	cout<<"anythingFailed? = "<<anythingFailed<<endl;
}
#endif

#ifdef SCI_HE
void test_field_div(){
	if (party==SERVER){
		srand(server_random_seed);
	}
	else{
		srand(client_random_seed);
	}
	int N = 32;
	int consSF = 3;
	uint32_t divisor = 1<<consSF;
	intType inp[N];
	intType outp[N];
	for(int i=0;i<N;i++){
		if (party==SERVER) inp[i] = rand()%prime_mod;
		else inp[i] = rand()%prime_mod;
	}

	cout<<"IO counter before === "<<io->counter<<" bytes"<<endl;
	funcFieldDivWrapper<intType>(N,inp,outp,divisor,nullptr);
	cout<<"IO counter after === "<<io->counter<<" bytes"<<endl;
	funcTruncationIdeal(N, inp, consSF);
	bool anythingFailed = false;
	for(int i=0;i<N;i++){
		auto ans = funcReconstruct2PCCons(inp[i],2);
		auto ansigot = funcReconstruct2PCCons(outp[i],2);
		if (party==sci::BOB){
			if(ans!=ansigot){
				cout<<RED<<"Error "<<i<<" ans,ansigot = "<<ans<<" "<<ansigot<<endl;
				anythingFailed = true;
			} 
			else {
				// cout<<GREEN<<"right ans "<<i<<" ans,ansigot = "<<ans<<" "<<ansigot<<endl;
			}
		}
	}
	cout<<"anythingFailed? = "<<anythingFailed<<endl;
}
#endif

int main(int argc, char** argv)
{
	ArgMapping amap;
	int port = 32000;
	string serverAddr = "127.0.0.1";

	amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
	amap.arg("p", port, "Port Number");
	amap.arg("ip", serverAddr, "IP Address of server (ALICE)");
	amap.parse(argc, argv);

	assert(party==sci::ALICE || party==sci::BOB);

	checkIfUsingEigen();
	for(int i=0;i<numThreads;i++){
	ioArr[i] = new sci::NetIO(party==sci::ALICE ? nullptr : serverAddr.c_str(), port+i);
	otInstanceArr[i] = new sci::IKNP<sci::NetIO>(ioArr[i]);
	prgInstanceArr[i] = new sci::PRG128();
	kkotInstanceArr[i] = new sci::KKOT < sci::NetIO > (ioArr[i]);
	matmulInstanceArr[i] = new Matmul<sci::NetIO, intType, sci::IKNP<sci::NetIO>>(party, bitlength, ioArr[i], otInstanceArr[i], nullptr);
	if (i == 0) {
	otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], party, baseForRelu, bitlength);
	} 
	else if (i == 1) {
	otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], 3-party, baseForRelu, bitlength);
	} 
	else if (i & 1) {
	otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], 3-party, baseForRelu, bitlength, false);
	otpackArr[i]->copy(otpackArr[1]);
	} 
	else {
	otpackArr[i] = new sci::OTPack<sci::NetIO>(ioArr[i], party, baseForRelu, bitlength, false);
	otpackArr[i]->copy(otpackArr[0]);
	}
	}

	io = ioArr[0];
	iknpOT = new sci::IKNP<sci::NetIO>(io);
	iknpOTRoleReversed = new sci::IKNP<sci::NetIO>(io); //TCP is full duplex -- so both side OT on same TCP should be good
	kkot = new sci::KKOT<sci::NetIO>(io);
	prg128Instance = new sci::PRG128();
	otpack = new sci::OTPack<sci::NetIO>(io, party, baseForRelu, bitlength);

	matmulImpl = new Matmul<sci::NetIO, intType, sci::IKNP<sci::NetIO>>(party, bitlength, io, iknpOT, iknpOTRoleReversed);


	#ifdef SCI_OT
	reluImpl = new ReLURingProtocol<sci::NetIO, intType>(party,RING,io,bitlength,baseForRelu,otpack);
	maxpoolImpl = new MaxPoolProtocol<sci::NetIO, intType>(party,RING,io,bitlength,baseForRelu,0,otpack,reluImpl);
	argmaxImpl = new ArgMaxProtocol<sci::NetIO, intType>(party,RING,io,bitlength,baseForRelu,0,otpack,reluImpl);
	#endif

	#ifdef SCI_HE
	reluImpl = new ReLUFieldProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack);
	maxpoolImpl = new MaxPoolProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack,reluImpl);
	argmaxImpl = new ArgMaxProtocol<sci::NetIO, intType>(party,FIELD,io,bitlength,baseForRelu,prime_mod,otpack,reluImpl);
	heConvImpl = new ConvField(party,io);
	heFCImpl = new FCField(party,io);
	heProdImpl = new ElemWiseProdField(party, io);
	assertFieldRun();
	#endif
	#ifdef MULTITHREADED_NONLIN
	#ifdef SCI_OT
	for(int i = 0; i < numThreads; i++) {
	if (i & 1) {
	reluImplArr[i] = new ReLURingProtocol<sci::NetIO, intType>(3-party,RING,ioArr[i],bitlength,baseForRelu,otpackArr[i]);
	maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(3-party,RING,ioArr[i],bitlength,baseForRelu,0,otpackArr[i],reluImplArr[i]);
	} 
	else {
	reluImplArr[i] = new ReLURingProtocol<sci::NetIO, intType>(party,RING,ioArr[i],bitlength,baseForRelu,otpackArr[i]);
	maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(party,RING,ioArr[i],bitlength,baseForRelu,0,otpackArr[i],reluImplArr[i]);
	}
	}
	#endif
	#ifdef SCI_HE
	for(int i = 0; i < numThreads; i++) {
	if (i & 1) {
	reluImplArr[i] = new ReLUFieldProtocol<sci::NetIO, intType>(3-party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i]);
	maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(3-party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i],reluImplArr[i]);
	} 
	else {
	reluImplArr[i] = new ReLUFieldProtocol<sci::NetIO, intType>(party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i]);
	maxpoolImplArr[i] = new MaxPoolProtocol<sci::NetIO, intType>(party,FIELD,ioArr[i],bitlength,baseForRelu,prime_mod,otpackArr[i],reluImplArr[i]);
	}
	}
	#endif
	#endif

	if (party==sci::ALICE){
	iknpOT->setup_send();
	iknpOTRoleReversed->setup_recv();
	}
	else if (party==sci::BOB){
	iknpOT->setup_recv();
	iknpOTRoleReversed->setup_send();
	}
	cout<<"After base ots, communication = "<<(io->counter)<<" bytes"<<endl;

	#ifdef SCI_OT
	test_ring_truncation();
	test_ring_avgpool();
	#else
	test_field_div();
	#endif

	cout<<"After function call, communication = "<<(io->counter)<<" bytes"<<endl;
}
