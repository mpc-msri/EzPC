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

#ifndef GLOBALS_H
#define GLOBALS_H

#pragma once
#include "sgx_intrin.h"
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include "selective_run_config.h"
#include "optimizations_config.h"
#include "network_config.h"
using namespace std;

//Run attestation or skip it
#define RUN_ATTESTATION

#define BITLENUSED 64

//Use or not use make_vector functionality
#define MAKE_VECTOR

//AES and other globals
#define LOG_DEBUG false
#define RANDOM_COMPUTE 128//Size of buffer for random elements
#define FIXED_KEY_AES "43739841701238781571456410093f43"
#define STRING_BUFFER_SIZE 256
#define true 1
#define false 0
#define DEBUG_CONST 16
#define DEBUG_INDEX 0
#define DEBUG_PRINT "SIGNED"
#define CPP_ASSEMBLY 1
#define MNIST false
#define PARALLEL true
#define NO_CORES 4

//MPC globals
extern int NUM_OF_PARTIES;
#define STANDALONE (NUM_OF_PARTIES == 1)
#define THREE_PC (NUM_OF_PARTIES == 3)
#define PARTY_A 0
#define PARTY_B 1
#define PARTY_C 2
#define PARTY_S 3

#define PRIME_NUMBER 127
#define FLOAT_PRECISION 15
#define PRIMARY (partyNum == PARTY_A or partyNum == PARTY_B)
#define	NON_PRIMARY (partyNum == PARTY_C)
#define HELPER (partyNum == PARTY_C)
#define MPC (THREE_PC)

#define MINI_BATCH_SIZE 1

//Typedefs and others
typedef __m128i superLongType;
typedef uint64_t myType;
typedef uint64_t aramisSecretType;
typedef uint8_t smallType;

const int BIT_SIZE = (sizeof(aramisSecretType) * CHAR_BIT);
const aramisSecretType LARGEST_NEG = ((aramisSecretType)1 << (BIT_SIZE - 1));
const aramisSecretType MINUS_ONE = (aramisSecretType)-1;
const smallType BOUNDARY = (256/PRIME_NUMBER) * PRIME_NUMBER;

const __m128i BIT1 = _mm_setr_epi8(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT2 = _mm_setr_epi8(2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT4 = _mm_setr_epi8(4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT8 = _mm_setr_epi8(8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT16 = _mm_setr_epi8(16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT32 = _mm_setr_epi8(32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT64 = _mm_setr_epi8(64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT128 = _mm_setr_epi8(128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

//Macros
#define _aligned_malloc(size,alignment) aligned_alloc(alignment,size)
#define _aligned_free free
#define getrandom(min, max) ((rand()%(int)(((max) + 1)-(min)))+ (min))
#define floatToMyType(a) ((aramisSecretType)(a * (1 << FLOAT_PRECISION)))
#define Arr2DIdx(arr,rows,cols,i,j) (*(arr + i*cols + j))
#define Arr4DIdx(arr,s0,s1,s2,s3,i,j,k,l) (*(arr + i*s1*s2*s3 + j*s2*s3 + k*s3 + l))

#endif
