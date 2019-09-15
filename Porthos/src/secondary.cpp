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

