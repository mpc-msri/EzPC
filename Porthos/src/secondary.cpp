/*

Authors: Sameer Wagh, Mayank Rathee, Nishant Kumar.

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

#include "secondary.h"
#include <iostream>
using namespace std;

//this player number
int partyNum;
//aes_key of the party
char *party_aes_key;

int instanceID;

//For faster DGK computation
smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];
smallType numberModuloPrime[256];
smallType numberModuloOtherNumbers[BITLENUSED][256];

//communication
extern string * addrs;
extern PorthosNet ** communicationSenders;
extern PorthosNet ** communicationReceivers;

void parseInputs(int argc, 
		char* argv[])
{	
	assert((sizeof(double) == sizeof(porthosSecretType)) && "sizeof(double) != sizeof(porthosSecretType)");
	if(argc == 3){
		instanceID = 0;
	}
	else if(argc == 4){
		instanceID = atoi(argv[3]);
	}
	else{
		porthos_throw_error(PARSE_ERROR);
		cout<<"Porthos expects either 3 or 4 CLI arguments!"<<endl;
	}
	NUM_OF_PARTIES = 3;
	partyNum = atoi(argv[1]);
}


void initializeMPC()
{
	//populate offline module prime addition and multiplication tables
	for (int i = 0; i < PRIME_NUMBER; ++i)
		for (int j = 0; j < PRIME_NUMBER; ++j)
		{
			additionModPrime[i][j] = (i + j) % PRIME_NUMBER;
			additionModPrime[j][i] = additionModPrime[i][j];
			multiplicationModPrime[i][j] = (i * j) % PRIME_NUMBER;
			multiplicationModPrime[j][i] = multiplicationModPrime[i][j];
		}

	for(int i=0;i<256;i++){
		numberModuloPrime[i] = i%PRIME_NUMBER;
	}

	for(int i=1;i<BITLENUSED;i++){
		for(int j=0;j<256;j++){
			numberModuloOtherNumbers[i][j] = j%i;
		}
	}
}

void deleteObjects()
{
	//close connection
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;

	delete[] addrs;
	delete[] party_aes_key;
}

