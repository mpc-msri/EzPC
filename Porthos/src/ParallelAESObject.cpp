#pragma once
#include "TedKrovetzAesNiWrapperC.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include <thread>
#include "AESObject.h"
#include "ParallelAESObject.h"

//For precomputing
extern AESObject* aes_common;

using namespace std;

void swapSmallTypes(smallType *a, 
		smallType *b)
{
    smallType temp = *a;
    *a = *b;
    *b = temp;
}

ParallelAESObject::precompute()
{
	//TODO (Mayank): Fix hardcoded 1 values.
	for (size_t i = 0; i < PC_CALLS_MAX*SHUFFLE_MAX*NO_CORES; ++i)
		randomNumber[i] = 1; //randomNumber[i] = aes_common->get8Bits();

	for (size_t i = 0; i < PC_CALLS_MAX*NONZERO_MAX*NO_CORES; ++i)
		randomNonZero[i] = 1; //randomNonZero[i] = aes_common->randNonZeroModPrime();
}


smallType ParallelAESObject::randNonZeroModPrime(int t, 
		int &offset)
{
	return randomNonZero[(counterPC*NONZERO_MAX*NO_CORES + t*NONZERO_MAX + offset++)%PC_CALLS_MAX*NONZERO_MAX*NO_CORES];
}


smallType ParallelAESObject::AES_random(int i, 
		int t, 
		int &offset)
{
	smallType ret;

	do
	{
		ret = randomNumber[(counterPC*SHUFFLE_MAX*NO_CORES + t*SHUFFLE_MAX + offset++)%PC_CALLS_MAX*SHUFFLE_MAX*NO_CORES];
	} while (ret >= ((256/i) * i));

	return ret;
}

void ParallelAESObject::AES_random_shuffle(smallType* vec, 
		size_t begin_offset, 
		size_t end_offset, 
		int t, 
		int &offset)
{
    auto n = end_offset - begin_offset;

    for (auto i = n-1; i > 0; --i)
        swapSmallTypes(&vec[begin_offset + i], &vec[begin_offset + (AES_random(i+1, t, offset)%n)]);
}
