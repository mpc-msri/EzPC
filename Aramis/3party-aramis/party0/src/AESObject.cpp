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

#pragma once
#include "../util/TedKrovetzAesNiWrapperC.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include "AESObject.h"
//strcpy
#include <string.h>
#include "../files/all_keys.h"

using namespace std;


AESObject::AESObject(char* filename)
{
	std::string str;

	if(strcmp(filename, "KeyA") == 0){
		str = keyA;
	}
	else if(strcmp(filename, "KeyB") == 0){
		str = keyB;
	}
	else if(strcmp(filename, "KeyC") == 0){
		str = keyC;
	}
	else if(strcmp(filename, "KeyD") == 0){
		str = keyD;
	}
	else if(strcmp(filename, "KeyAB") == 0){
		str = keyAB;
	}
	else if(strcmp(filename, "KeyAC") == 0){
		str = keyAC;
	}
	else if(strcmp(filename, "KeyAD") == 0){
		str = keyAD;
	}
	else if(strcmp(filename, "KeyBC") == 0){
		str = keyBC;
	}
	else if(strcmp(filename, "KeyBD") == 0){
		str = keyBD;
	}
	else if(strcmp(filename, "KeyCD") == 0){
		str = keyCD;
	}
	else{
		print_string("ERROR NO KEY");
	}
	if(str.length() == 0){
		return;
	}

	int len = str.length();
	char common_aes_key[len+1];
	memset(common_aes_key, '\0', len+1);
	common_aes_key[len] = '\0';
	strncpy(common_aes_key, str.c_str(), len);
	AES_set_encrypt_key((unsigned char*)common_aes_key, 256, &aes_key);
	keystr = str;
}

void AESObject::ResetKey(char* filename)
{
	std::string str;

	if(strcmp(filename, "KeyA") == 0){
		str = keyA;
	}
	else if(strcmp(filename, "KeyB") == 0){
		str = keyB;
	}
	else if(strcmp(filename, "KeyC") == 0){
		str = keyC;
	}
	else if(strcmp(filename, "KeyD") == 0){
		str = keyD;
	}
	else if(strcmp(filename, "KeyAB") == 0){
		str = keyAB;
	}
	else if(strcmp(filename, "KeyAC") == 0){
		str = keyAC;
	}
	else if(strcmp(filename, "KeyAD") == 0){
		str = keyAD;
	}
	else if(strcmp(filename, "KeyBC") == 0){
		str = keyBC;
	}
	else if(strcmp(filename, "KeyBD") == 0){
		str = keyBD;
	}
	else if(strcmp(filename, "KeyCD") == 0){
		str = keyCD;
	}
	else{
		print_string("ERROR NO KEY");
	}
	if(str.length() == 0){
		return;
	}

	int len = str.length();
	char common_aes_key[len+1];
	memset(common_aes_key, '\0', len+1);
	common_aes_key[len] = '\0';
	strncpy(common_aes_key, str.c_str(), len);
	AES_set_encrypt_key((unsigned char*)common_aes_key, 256, &aes_key);
	keystr = str;
	rCounter = -1;
}

AESObject::~AESObject()
{
	return;
}

__m128i AESObject::newRandomNumber()
{
	rCounter++;
	if (rCounter % RANDOM_COMPUTE == 0)//generate more random seeds
	{
		for (int i = 0; i < RANDOM_COMPUTE; i++)
			tempSecComp[i] = _mm_set1_epi32(rCounter+i);//not exactly counter mode - (rcounter+i,rcouter+i,rcounter+i,rcounter+i)
		AES_ecb_encrypt_chunk_in_out(tempSecComp, pseudoRandomString, RANDOM_COMPUTE, &aes_key);
	}
	return pseudoRandomString[rCounter%RANDOM_COMPUTE];
}

aramisSecretType AESObject::get64Bits()
{
	int idx = 0;
	if (fetch64New)
	{
		random64BitNumber = newRandomNumber();
	}
	else
	{
		idx = 1;
	}
	fetch64New ^= 1;
	return (((uint64_t*)(&random64BitNumber))[idx]);
}

smallType AESObject::get8Bits()
{
	smallType ret;

	if (random8BitCounter == 0)
	{
		random8BitNumber = newRandomNumber();
	}

	uint8_t *temp = (uint8_t*)&random8BitNumber;
	ret = (smallType)temp[random8BitCounter];

	random8BitCounter++;
	if (random8BitCounter == 16)
		random8BitCounter = 0;

	return ret;
}


smallType AESObject::getBit()
{
	smallType ret;
	__m128i temp;

	if (randomBitCounter == 0)
		randomBitNumber = newRandomNumber();

	int x = randomBitCounter % 8;
	switch(x)
	{
		case 0 : temp = _mm_and_si128(randomBitNumber, BIT1);
			 break;
		case 1 : temp = _mm_and_si128(randomBitNumber, BIT2);
			 break;
		case 2 : temp = _mm_and_si128(randomBitNumber, BIT4);
			 break;
		case 3 : temp = _mm_and_si128(randomBitNumber, BIT8);
			 break;
		case 4 : temp = _mm_and_si128(randomBitNumber, BIT16);
			 break;
		case 5 : temp = _mm_and_si128(randomBitNumber, BIT32);
			 break;
		case 6 : temp = _mm_and_si128(randomBitNumber, BIT64);
			 break;
		case 7 : temp = _mm_and_si128(randomBitNumber, BIT128);
			 break;
	}
	uint8_t *val = (uint8_t*)&temp;
	ret = (val[0] >> x);

	randomBitCounter++;
	if (randomBitCounter % 8 == 0)
		randomBitNumber = _mm_srli_si128(randomBitNumber, 1);

	if (randomBitCounter == 128)
		randomBitCounter = 0;

	return ret;
}

extern smallType numberModuloPrime[256];
smallType AESObject::randModPrime()
{
	smallType ret;
	smallType boundary = BOUNDARY;
	do
	{
		ret = get8Bits();
	} while (ret >= boundary);

	return numberModuloPrime[ret];
}

smallType AESObject::randNonZeroModPrime()
{
	smallType ret;
	do
	{
		ret = randModPrime();
	} while (ret == 0);

	return ret;
}

aramisSecretType AESObject::randModuloOdd()
{
	aramisSecretType ret;
	do
	{
		ret = get64Bits();
	} while (ret == MINUS_ONE);
	return ret;
}

extern smallType numberModuloOtherNumbers[BITLENUSED][256];
smallType AESObject::AES_random(int i)
{
#ifdef DEBUG
	assert(i < 64);
#endif
	smallType ret;
	smallType boundary = ((256/i) * i);
	do
	{
		ret = get8Bits();
	} while (ret >= boundary);

	return (numberModuloOtherNumbers[i][ret]);
}

void AESObject::AES_random_shuffle(vector<smallType> &vec, size_t begin_offset, size_t end_offset)
{
	vector<smallType>::iterator it = vec.begin();
	vector<smallType>::iterator first = it + begin_offset;
	vector<smallType>::iterator last = it + end_offset;
	long int n = last - first;

	for (long int i = n-1; i > 0; --i)
	{
		using std::swap;
		swap(first[i], first[AES_random(i+1)]);
	}
}


__m128i AESObject::newRandomNumberFillArray()
{
	if (rCounter % RANDOM_COMPUTE == 0)//generate more random seeds
	{
		for (int i = 0; i < RANDOM_COMPUTE; i++)
			tempSecComp[i] = _mm_set1_epi32(rCounter+i);//not exactly counter mode - (rcounter+i,rcouter+i,rcounter+i,rcounter+i)
		AES_ecb_encrypt_chunk_in_out(tempSecComp, pseudoRandomString, RANDOM_COMPUTE, &aes_key);
	}

}

void AESObject::fillWithRandomBitsModuloPrime(uint8_t* arr, uint64_t size){
	if(size == 0){
		return;
	}
	int leftoverBytesToRead = (16 - (random8BitCounter%16))%16;
	smallType temp;
	uint64_t arrIdx = 0;
	for(int i=0; i<leftoverBytesToRead; i++){
		temp = get8Bits();
		if(temp < BOUNDARY){
			arr[arrIdx++] = temp;
			size--;
			if(size <= 0)
				break;
		}
	}
	assert(random8BitCounter == 0);
	if(size > 0){
		newRandomNumberFillArray();
		//We now have RANDOM_COMPUTE number of 128 bit values
		//This means we have 16*RANDOM_COMPUTE number of 8 bit values
		int iters_left = size/(16*RANDOM_COMPUTE);
		int iter_size = 16*RANDOM_COMPUTE;
		uint8_t* prsptr = (uint8_t*)pseudoRandomString;
		int prsidx = 0;
		int consumed = 0;
		while(size > 0){
			if(consumed == iter_size){
				consumed = 0;
				prsidx = 0;
				newRandomNumberFillArray();
			}
			temp = prsptr[prsidx++];
			consumed++;
			if(temp < BOUNDARY){
				arr[arrIdx++] = temp;
				size--;
			}
		}
		//Now settle the state of AES object.
		if(consumed % 16 == 0){
			rCounter += (consumed/16);
			random8BitCounter = 0;
		}
		else{
			rCounter += 1+(rCounter/16);
			random8BitCounter = consumed%16;
			random8BitNumber = pseudoRandomString[rCounter-1];
		}
	}


}
