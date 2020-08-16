/*
Authors: Mayank Rathee.
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

#include "utils_rand_sgx.h"

// returns a random integer from 0 to SGX_RAND_MAX-1
int sgx_rand(){
	unsigned char* randbuf = (unsigned char*)malloc(SGX_RAND_MAX_BYTES);
	sgx_read_rand(randbuf, SGX_RAND_MAX_BYTES);
	int randvalue = *(int*)randbuf;
	if(randvalue<0){
		randvalue *= -1;
	}
	return randvalue;
}

// from 0 to modulus-1
int sgx_rand(int modulus){
	int fullvalue = sgx_rand();
	return fullvalue%modulus;
}
