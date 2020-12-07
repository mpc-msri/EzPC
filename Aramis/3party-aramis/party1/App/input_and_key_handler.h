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

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include "sgx_urts.h"
#include "Enclave_u.h"

#define LAYER0 784
#define LAST_LAYER_SIZE 10
#ifdef MNIST
	#define TRAINING_DATA_SIZE 60000
	#define TEST_DATA_SIZE 10000
	#define LOG_MINI_BATCH 7
#else
	#define TRAINING_DATA_SIZE 8
	#define TEST_DATA_SIZE 8
	#define LOG_MINI_BATCH 0
#endif

void read_input_and_keys(sgx_enclave_id_t enclave_id, int party_num);
void read_keys_securenn(sgx_enclave_id_t enclave_id, int party_num);
