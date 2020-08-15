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
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef AESOBJECT_H
#define AESOBJECT_H

#pragma once
#include <algorithm>
#include "globals.h"
#include <openssl/evp.h>
#include <openssl/aes.h>

typedef __m128i block;
//NOTE: This class isn't thread safe.
class AESObject
{
private:
	//AES variables
	__m128i pseudoRandomString[RANDOM_COMPUTE];
	__m128i tempSecComp[RANDOM_COMPUTE];
	unsigned long rCounter = -1;

	//Extraction variables
	__m128i randomBitNumber {0};
	uint8_t randomBitCounter = 0;
	__m128i random8BitNumber {0};
	uint8_t random8BitCounter = 0; 
	__m128i random64BitNumber {0};
	bool fetch64New = true;

	unsigned char* inbuf_ssl;
	unsigned char* outbuf_ssl;
	unsigned char* key_ssl;
	unsigned char* iv_ssl;
	
	EVP_CIPHER_CTX* ctx_ssl;
	

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
	
	void SSL_AES_ecb_encrypt_chunk_in_out(block *in, block *out, unsigned nblks);
	
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
