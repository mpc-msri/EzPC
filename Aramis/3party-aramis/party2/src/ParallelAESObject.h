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


#ifndef PARALLEL_AESOBJECT_H
#define PARALLEL_AESOBJECT_H

#pragma once
#include <algorithm>
#include "globals.h"

#define NONZERO_MAX 100
#define SHUFFLE_MAX 100
#define PC_CALLS_MAX 100



class ParallelAESObject
{
	private:
		//Private helper functions
		smallType AES_random(int i, 
				int t, 
				int &offset);

		//precomputed random numbers
		smallType randomNumber[PC_CALLS_MAX*SHUFFLE_MAX*NO_CORES];
		smallType randomNonZero[PC_CALLS_MAX*NONZERO_MAX*NO_CORES];

		uint64_t counterPC = 0;

	public:
		//Constructor
		ParallelAESObject(char* filename) {};
		void precompute();
		smallType randModPrime(int t, 
				int &offset);

		//Other randomness functions
		smallType randNonZeroModPrime(int t, 
				int &offset);
		
		void AES_random_shuffle(smallType *vec, 
				size_t begin_offset, 
				size_t end_offset, 
				int t, 
				int &offset);
		
		void counterIncrement()
		{
			counterPC++;
		};
};



#endif
