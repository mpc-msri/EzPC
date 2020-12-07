/*
Authors: Mayank Rathee, Sameer Wagh, Nishant Kumar.
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

#ifndef AESOBJECT_H
#define AESOBJECT_H

#pragma once
#include <algorithm>
#include "globals.h"
#include "../utils_sgx_port/utils_print_sgx.h"
#include "sgx_trts.h"
#include "../Enclave/Enclave_t.h"

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
	__m128i newRandomNumberFillArray();

	//Private helper functions
	smallType AES_random(int i);


public:
	std::string keystr;
	//Constructor
	AESObject(char* filename);
	~AESObject();
	void ResetKey(char* filename);
	
	//Randomness functions
	aramisSecretType get64Bits();
	smallType get8Bits();
	smallType getBit();

	//Other randomness functions
	smallType randModPrime();
	smallType randNonZeroModPrime();
	aramisSecretType randModuloOdd();
	void AES_random_shuffle(vector<smallType> &vec, size_t begin_offset, size_t end_offset);
	void fillWithRandomBitsModuloPrime(uint8_t* arr, uint64_t size);

	unsigned long getRCounter()
	{
		return rCounter;
	}

};


#endif
