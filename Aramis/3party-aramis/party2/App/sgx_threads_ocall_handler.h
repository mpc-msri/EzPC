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
#include <string.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <pwd.h>
#define MAX_PATH FILENAME_MAX
#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include "sgx_urts.h"
#include "App.h"
#include "Enclave_u.h"
#include <thread>

extern sgx_enclave_id_t global_eid;

void ocall_parallelPC_spawn_threads(int thread_num);

void ocall_parallelPC_join_threads();

void ocall_matmul_spawn_threads(int thread_num);

void ocall_matmul_join_threads();

void ocall_join_threads();

void ocall_populate_aes_spawn_threads(int thread_num);

void ocall_deduce_zero_spawn_threads(int thread_num);

void ocall_populate_flat_vector_aes_spawn_threads(int thread_num);

void ocall_join_threads_half();

void ocall_create_testing_threads(int thread_num);

void ocall_join_testing_threads(int thread_num);

