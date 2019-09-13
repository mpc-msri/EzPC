#ifndef AESOBJECT_H
#define AESOBJECT_H

#pragma once
#include <algorithm>
#include "globals.h"


//NOTE: This class isn't thread safe.
class AESObject
{
private:
	//AES variables
	__m128i pseudoRandomString[RANDOM_COMPUTE];
	__m128i tempSecComp[RANDOM_COMPUTE];
	unsigned long rCounter = -1;
	AES_KEY_TED aes_key;

	//Extraction variables
	__m128i randomBitNumber {0};
	uint8_t randomBitCounter = 0;
	__m128i random8BitNumber {0};
	uint8_t random8BitCounter = 0; 
	__m128i random64BitNumber {0};
	bool fetch64New = true;

	//Private extraction functions
	__m128i newRandomNumber();

	//Private helper functions
	smallType AES_random(int i);

#ifdef PRECOMPUTEAES
	__m128i* preComputedKeys = NULL;
	__m128i* tempKeyArray = NULL;
	
	void PreComputeKeysFunc(porthosLongUnsignedInt startKeyNum, 
			porthosLongUnsignedInt numKeys);
#endif

public:
	//Constructor
	AESObject(string filename);
	
	~AESObject();

#ifdef PRECOMPUTEAES
	void PreComputeKeys(porthosLongUnsignedInt numKeysToPrecompute, 
			int32_t numThreads);
#endif
	
	//Randomness functions
	porthosSecretType get64Bits();
	
	smallType get8Bits();
	
	smallType getBit();
	
	//Other randomness functions
	smallType randModPrime();
	
	smallType randNonZeroModPrime();
	
	porthosSecretType randModuloOdd();
	
	void AES_random_shuffle(vector<smallType> &vec, 
			size_t begin_offset, 
			size_t end_offset);

	unsigned long getRCounter()
	{
		return rCounter;
	}

#ifdef PRECOMPUTEAES

	__m128i* getPreComputedKeysPtr()
	{
		return preComputedKeys;
	}
	
	void fillWithRandomBits64(porthosSecretType* arr, 
			size_t size);
	
	void fillWithRandomBits8(uint8_t* arr, 
			size_t size);
	
	void fillWithRandomModuloPrimeBits(uint8_t* arr, 
			size_t size);
	
	void fillWithRandomModuloOddBits(porthosSecretType* arr, 
			size_t size);
	
#endif
};



#endif
