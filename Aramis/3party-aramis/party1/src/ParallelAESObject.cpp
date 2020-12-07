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
#include "TedKrovetzAesNiWrapperC.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include "AESObject.h"
#include "ParallelAESObject.h"

//For precomputing
extern AESObject* aes_common;
extern smallType numberModuloPrime[256];

using namespace std;

void swapSmallTypes(smallType *a, 
		smallType *b)
{
    smallType temp = *a;
    *a = *b;
    *b = temp;
}

void ParallelAESObject::precompute()
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

smallType ParallelAESObject::randModPrime(int t, int &offset)
{
	smallType ret;

	do
	{
		ret = randomNumber[(counterPC*SHUFFLE_MAX*NO_CORES + t*SHUFFLE_MAX + offset++)%PC_CALLS_MAX*SHUFFLE_MAX*NO_CORES];
	} while (ret >= BOUNDARY);

	return numberModuloPrime[ret];
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

