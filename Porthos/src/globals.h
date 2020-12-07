/*

Authors: Sameer Wagh, Mayank Rathee, Nishant Kumar.

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
#include <emmintrin.h>
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
using namespace std;

//Uncomment to run the examples in example_neural_nets/ dir
#define RUN_INCLUDED_NN_EXAMPLES

//Run default NN and not the ones in example_neural_nets/ dir
#ifndef RUN_INCLUDED_NN_EXAMPLES
#define DEFAULT_NN
#endif

/**********************SWITCH OPTIMIZATIONS************/
//This includes flags to turn on/off Porthos
//optimizations proposed over SecureNN in crypTFlow.

//To run all optimizations
#define PORTHOS_OPTIMIZATIONS

//Switch all optimizations on or off
#ifdef PORTHOS_OPTIMIZATIONS
#define RUN_SHARECONV_OPTI
#define RUN_MSB_OPTI
#define CONV_OPTI
//#define PRECOMPUTEAES
#define PARALLIZE_CRITICAL
#endif
/******************************************************/

//Some flags for debugging
#define LOG_DEBUG false
#define LOG_LAYERWISE false
//Uncomment this if you want to send and receive
//serially for debugging
#define PARALLEL_COMM
//#define VERIFYLAYERWISE 

//AES and other globals
#define RANDOM_COMPUTE 256//Size of buffer for random elements
#define FIXED_KEY_AES "43739841701238781571456410093f43"
#define STRING_BUFFER_SIZE 256
#define true 1
#define false 0
#define DEBUG_CONST 16
#define DEBUG_INDEX 0
#define DEBUG_PRINT "SIGNED"
#define CPP_ASSEMBLY 1
#define PARALLEL true
#define NO_CORES 4
#define NUM_ITERATIONS 1

//Sizes for debug/test_functions
#define TEST_BATCH_SIZE 1

//MPC globals
extern int NUM_OF_PARTIES;
#define STANDALONE (NUM_OF_PARTIES == 1)
#define THREE_PC (NUM_OF_PARTIES == 3)
#define PARTY_A 0
#define PARTY_B 1
#define PARTY_C 2

extern uint32_t FLOAT_PRECISION;

#define PRIME_NUMBER 127
#define PRIMARY (partyNum == PARTY_A or partyNum == PARTY_B)
#define HELPER (partyNum == PARTY_C)
#define MPC (THREE_PC)

//Some parameters for ParallelPC function's AES computation
//Tweak these params if you get SEGFAULT in ParallelPC execution
#define NONZERO_MAX 100
#define SHUFFLE_MAX 100
#define PC_CALLS_MAX 100

//Some other useful flags
#define DEBUG //DEBUG build or not
#define BITLENUSED 64
#define RUNTIME_DETAILED //Prints wall clock time and CPU time.

//Important: To compile Porthos without Eigen support
//turn off/comment out this flag.
#define USE_EIGEN //Uses Eigen for MatMul

//Typedefs and others
typedef __m128i superLongType;
//Secret types are used to represent secret shared values
typedef uint64_t porthosSecretType;
typedef int64_t porthosSignedSecretType;

typedef uint64_t porthosLongUnsignedInt;
typedef int64_t porthosLongSignedInt;
typedef uint8_t smallType;

const int BIT_SIZE = (sizeof(porthosSecretType) * CHAR_BIT);
const porthosSecretType LARGEST_NEG = ((porthosSecretType)1 << (BIT_SIZE - 1));
const porthosSecretType MINUS_ONE = (porthosSecretType)-1;
const smallType BOUNDARY = (256/PRIME_NUMBER) * PRIME_NUMBER;

const __m128i BIT1 = _mm_setr_epi8(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT2 = _mm_setr_epi8(2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT4 = _mm_setr_epi8(4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT8 = _mm_setr_epi8(8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT16 = _mm_setr_epi8(16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT32 = _mm_setr_epi8(32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT64 = _mm_setr_epi8(64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT128 = _mm_setr_epi8(128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

//Some useful macros
#define _aligned_malloc(size,alignment) aligned_alloc(alignment,size)
#define _aligned_free free
#define getrandom(min, max) ((rand()%(int)(((max) + 1)-(min)))+ (min))
#define Arr2DIdx(arr,rows,cols,i,j) (*(arr + i*cols + j))
#define Arr4DIdx(arr,s0,s1,s2,s3,i,j,k,l) (*(arr + i*s1*s2*s3 + j*s2*s3 + k*s3 + l))

inline porthosSecretType floatToMyType(porthosSecretType a){
	return (a * (1 << FLOAT_PRECISION));
}

// The following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

#endif
