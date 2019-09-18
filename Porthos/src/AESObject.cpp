/*

Authors: Sameer Wagh, Mayank Rathee, Nishant Kumar.

Copyright:
Copyright (c) 2018 Microsoft Research
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
#include "TedKrovetzAesNiWrapperC.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <thread>
#include <cstring>
#include "AESObject.h"

using namespace std;

AESObject::AESObject(string filename)
{
	ifstream f(filename);
	string str { istreambuf_iterator<char>(f), istreambuf_iterator<char>() };
	f.close();
	int len = str.length();
	char common_aes_key[len+1];
	memset(common_aes_key, '\0', len+1);
	strcpy(common_aes_key, str.c_str());
	AES_set_encrypt_key((unsigned char*)common_aes_key, 256, &aes_key);
}

AESObject::~AESObject()
{
#ifdef PRECOMPUTEAES
	if (preComputedKeys!=NULL) delete[] preComputedKeys;
	if (tempKeyArray!=NULL) delete[] tempKeyArray;
#endif
}

#ifdef PRECOMPUTEAES

void AESObject::PreComputeKeysFunc(porthosLongUnsignedInt startKeyNum,
		porthosLongUnsignedInt numKeys)
{
	for(porthosLongUnsignedInt i=0;i<numKeys;i++)
	{
		porthosLongUnsignedInt curKeyNum = startKeyNum + i;
		tempKeyArray[curKeyNum] = _mm_set1_epi32(curKeyNum);
	}
	AES_ecb_encrypt_chunk_in_out(tempKeyArray+startKeyNum, preComputedKeys+startKeyNum, numKeys, &aes_key);
}

void AESObject::PreComputeKeys(porthosLongUnsignedInt numKeysToPrecompute,
		int32_t numThreads)
{
	tempKeyArray = new __m128i[numKeysToPrecompute];
	preComputedKeys = new __m128i[numKeysToPrecompute];
	porthosLongUnsignedInt numKeysPerThread = numKeysToPrecompute/numThreads;
	thread* threads = new thread[numThreads];
	for(int i=0;i<numThreads;i++)
	{
		porthosLongUnsignedInt startKeyNum = i*numKeysPerThread;
		int32_t numKeys = numKeysPerThread;
		if (i == numThreads-1)
		{
			numKeys = numKeysToPrecompute - ((numThreads-1)*numKeysPerThread);
		}
		threads[i] = thread(&AESObject::PreComputeKeysFunc, this, startKeyNum, numKeys);
	}
	for (int i = 0; i < numThreads; i++)
		threads[i].join();

	delete[] threads;
}

#endif

__m128i AESObject::newRandomNumber()
{
#ifndef PRECOMPUTEAES
	rCounter++;
	if (rCounter % RANDOM_COMPUTE == 0)//generate more random seeds
	{
		for (int i = 0; i < RANDOM_COMPUTE; i++)
			tempSecComp[i] = _mm_set1_epi32(rCounter+i);//not exactly counter mode - (rcounter+i,rcouter+i,rcounter+i,rcounter+i)
		AES_ecb_encrypt_chunk_in_out(tempSecComp, pseudoRandomString, RANDOM_COMPUTE, &aes_key);
	}
	return pseudoRandomString[rCounter%RANDOM_COMPUTE];

#else

#ifdef DEBUG
	assert(preComputedKeys!=NULL);
#endif

	rCounter++;
	__builtin_prefetch(&preComputedKeys[rCounter+1],0,1);
	return preComputedKeys[rCounter];
#endif

}

porthosSecretType AESObject::get64Bits()
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
	return (((porthosSecretType*)(&random64BitNumber))[idx]);
}

smallType AESObject::get8Bits()
{
	smallType ret;

	if (random8BitCounter == 0)
	{
		random8BitNumber = newRandomNumber();
		//random8BitNumber = _mm_set1_epi32(1);
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

porthosSecretType AESObject::randModuloOdd()
{
	porthosSecretType ret;
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

void AESObject::AES_random_shuffle(vector<smallType> &vec, 
		size_t begin_offset, 
		size_t end_offset)
{
	vector<smallType>::iterator it = vec.begin();
	auto first = it + begin_offset;
	auto last = it + end_offset;
	auto n = last - first;

	for (auto i = n-1; i > 0; --i)
	{
		using std::swap;
		swap(first[i], first[AES_random(i+1)]);
	}
}

#ifdef PRECOMPUTEAES

void AESObject::fillWithRandomBits64(porthosSecretType* arr, 
		size_t size)
{
#ifdef DEBUG
	assert(preComputedKeys!=NULL);
#endif

	if (size == 0)
		return;
	if (!fetch64New){
		porthosSecretType* random64BitNumberPtr = (porthosSecretType*)(&random64BitNumber);
		arr[0] = random64BitNumberPtr[1];
		arr = arr+1;
		size--;
		fetch64New = true;
	}

	if (size > 0){
		rCounter++;
		memcpy(arr, preComputedKeys + rCounter, size*8); //size*8 bytes to copy
		if (size&1){
			//size is odd
			rCounter += (size/2);
			fetch64New = false;
			random64BitNumber = preComputedKeys[rCounter];
		}
		else{
			//size is even
			rCounter += ((size/2)-1);
		}
	}
}

void AESObject::fillWithRandomBits8(uint8_t* arr, 
		size_t size)
{
#ifdef DEBUG
	assert(preComputedKeys!=NULL);
#endif

	if (size == 0)
		return;
	int leftoverBytesToRead = (16 - (random8BitCounter%16))%16;
	for(int i=0 ; (i<leftoverBytesToRead) && (size>0) ; i++, size--){
		arr[i] = get8Bits();
	}

	if (size > 0){
#ifdef DEBUG
		assert(random8BitCounter == 0);
#endif
		arr = arr + leftoverBytesToRead;
		//size is already updated
		rCounter += 1;
		memcpy(arr, preComputedKeys + rCounter, size);
		int unalignedRead = size%16;
		if (unalignedRead == 0){
			rCounter += ((size/16)-1);
			random8BitCounter = 0;
		}
		else{
			rCounter += (size/16);
			random8BitNumber = preComputedKeys[rCounter];
			random8BitCounter = unalignedRead;
		}
	}
}

void AESObject::fillWithRandomModuloPrimeBits(uint8_t* arr, 
		size_t size)
{
#ifdef DEBUG
	assert(preComputedKeys!=NULL);
#endif

	if (size == 0)
		return;
	int leftoverBytesToRead = (16 - (random8BitCounter%16))%16;
	smallType temp;
	uint32_t arrIdx = 0;
	for(int i=0 ; i<leftoverBytesToRead ; i++){
		temp = get8Bits();
		if (temp < BOUNDARY){
			arr[arrIdx++] = numberModuloPrime[temp];
			size--;
			if (size <= 0)
				break;
		}
	}

	if (size > 0){
#ifdef DEBUG
		assert(random8BitCounter == 0);
#endif
		//no need to update size and arr
		rCounter += 1;
		uint8_t* preComputedKeysPtr = (uint8_t*)(preComputedKeys+rCounter);
		uint32_t preComputedKeysIdx = 0;
		smallType boundary = BOUNDARY;
		while(size > 0){
			smallType temp = preComputedKeysPtr[preComputedKeysIdx++];
			if (temp < boundary){
				arr[arrIdx++] = numberModuloPrime[temp];
				size--;
			}
		}
		int unalignedRead = preComputedKeysIdx%16;
		if (unalignedRead == 0){
			rCounter += ((preComputedKeysIdx/16)-1);
			random8BitCounter = 0;
		}
		else{
			rCounter += (preComputedKeysIdx/16);
			random8BitNumber = preComputedKeys[rCounter];
			random8BitCounter = unalignedRead;
		}
	}
}


void AESObject::fillWithRandomModuloOddBits(porthosSecretType* arr, 
		size_t size)
{
#ifdef DEBUG
	assert(preComputedKeys!=NULL);
#endif

	if (size==0)
		return;
	uint32_t arrIdx = 0;
	if (!fetch64New){
		porthosSecretType* random64BitNumberPtr = (porthosSecretType*)(&random64BitNumber);
		porthosSecretType temp = random64BitNumberPtr[1];
		if (temp != MINUS_ONE){
			arr[arrIdx++] = temp;
			size--;
		}
		fetch64New = true;
	}

	if (size > 0){
		rCounter++;
		porthosSecretType* preComputedKeysPtr = (porthosSecretType*)(preComputedKeys + rCounter);
		uint32_t preComputedKeysIdx = 0;
		while(size > 0){
			porthosSecretType temp = preComputedKeysPtr[preComputedKeysIdx++];
			if (temp != MINUS_ONE){
				arr[arrIdx++] = temp;
				size--;
			}
		}
		if (preComputedKeysIdx & 1){
			//preComputedKeysIdx is odd
			rCounter += (preComputedKeysIdx/2);
			fetch64New = false;
			random64BitNumber = preComputedKeys[rCounter];
		}
		else{
			//preComputedKeysIdx is even
			rCounter += ((preComputedKeysIdx/2)-1);
		}
	}
}


#endif

