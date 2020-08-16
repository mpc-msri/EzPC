/*
Authors: Nishant Kumar, Deevashwer Rathee
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

#ifndef GLOBALS_H___
#define GLOBALS_H___

#include <cstdint> //Only keep standard headers over here
#include <chrono>  //Keep the local repo based headers below, once constants are defined
#include <thread>
#include <map>

// #define NDEBUG //This must come first -- so that this marco is used throughout code
			   	 //Defining this will disable all asserts throughout code
#define LOG_LAYERWISE
#define USE_EIGEN
#define RUNOPTI
#ifdef RUNOPTI
#define MULTITHREADED_MATMUL
#define MULTITHREADED_NONLIN
#define MULTITHREADED_TRUNC
#define MULTITHREADED_DOTPROD
#endif
#define VERIFY_LAYERWISE
#define WRITE_LOG

enum NetworkName {
    Default, MiniONN, SqNet, ResNet18, ResNet50, DenseNet121, ResNet32_Cifar100
};

NetworkName network_name = Default;

inline std::string get_network_label(NetworkName network_name) {
    switch(network_name) {
        case MiniONN:
            return "MiniONN";
        case SqNet:
            return "SqNet";
        case ResNet18:
            return "ResNet18";
        case ResNet50:
            return "ResNet50";
        case DenseNet121:
            return "DenseNet121";
        case ResNet32_Cifar100:
            return "ResNet32_Cifar100";
    }
    return "Unknown-Network";
}

// To use 64 bitlen, define BITLEN_64 in the main program before including this.
// Otherwise default to 32 bits
#if defined(SCI_HE)
    typedef uint64_t intType;
    typedef int64_t signedIntType;
    static const bool isNativeRing = false;
#elif defined(SCI_OT)
    #if defined(BITLEN_64)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_33)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_34)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_35)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_36)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_37)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_38)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_39)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_40)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_41)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_42)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_43)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #elif defined(BITLEN_44)
        typedef uint64_t intType;
        typedef int64_t signedIntType;
        static const bool isNativeRing = true;
    #else
        typedef uint32_t intType;
        typedef int32_t signedIntType;
        static const bool isNativeRing = true;
    #endif
#endif

#if defined(BITLEN_64)
    int32_t bitlength = 64;
#elif defined(BITLEN_32)
    int32_t bitlength = 32;
#elif defined(BITLEN_33)
    int32_t bitlength = 33;
#elif defined(BITLEN_34)
    int32_t bitlength = 34;
#elif defined(BITLEN_35)
    int32_t bitlength = 35;
#elif defined(BITLEN_36)
    int32_t bitlength = 36;
#elif defined(BITLEN_37)
    int32_t bitlength = 37;
#elif defined(BITLEN_38)
    int32_t bitlength = 38;
#elif defined(BITLEN_39)
    int32_t bitlength = 39;
#elif defined(BITLEN_40)
    int32_t bitlength = 40;
#elif defined(BITLEN_41)
    int32_t bitlength = 41;
#elif defined(BITLEN_42)
    int32_t bitlength = 42;
#elif defined(BITLEN_43)
    int32_t bitlength = 43;
#elif defined(BITLEN_44)
    int32_t bitlength = 44;
#else
    #if defined(SCI_OT)
        int32_t bitlength = 32; //In ring case, default to 32
    #else
        int32_t bitlength = -1; //In field case, error condition: expecting one of the above macros
    #endif
#endif

/*
Bitlength 32 prime: 4293918721
Bitlength 33 prime: 8589475841
Bitlength 34 prime: 17179672577
Bitlength 35 prime: 34359410689
Bitlength 36 prime: 68718428161
Bitlength 37 prime: 137438822401
Bitlength 38 prime: 274876334081
Bitlength 39 prime: 549755486209
Bitlength 40 prime: 1099510054913
Bitlength 41 prime: 2199023190017
*/

#ifdef SCI_HE
    const std::map<int32_t, uint64_t> default_prime_mod {
        { 32, 4293918721 },
        { 33, 8589475841 },
        { 34, 17179672577 },
        { 35, 34359410689 },
        { 36, 68718428161 },
        { 37, 137438822401 },
        { 38, 274876334081 },
        { 39, 549755486209 },
        { 40, 1099510054913 },
        { 41, 2199023190017 },
    };
    uint64_t prime_mod = default_prime_mod.at(bitlength);
#else
#if defined(BITLEN_64)
    uint64_t prime_mod = 0ULL;
#else
    uint64_t prime_mod = 1ULL << bitlength;
#endif
#endif

const int32_t baseForRelu = 4;
int32_t party = 0;
extern const int numThreads = 4;

#ifdef LOG_LAYERWISE
uint64_t ConvTimeInMilliSec = 0;
uint64_t MatmulTimeInMilliSec = 0;
uint64_t BatchNormInMilliSec = 0;
uint64_t TruncationTimeInMilliSec = 0;
uint64_t ReluTimeInMilliSec = 0;
uint64_t MaxpoolTimeInMilliSec = 0;
uint64_t AvgpoolTimeInMilliSec = 0;
uint64_t ArgmaxTimeInMilliSec = 0;

uint64_t ConvCommSent = 0;
uint64_t MatmulCommSent = 0;
uint64_t BatchNormCommSent = 0;
uint64_t TruncationCommSent = 0;
uint64_t ReluCommSent = 0;
uint64_t MaxpoolCommSent = 0;
uint64_t AvgpoolCommSent = 0;
uint64_t ArgmaxCommSent = 0;

#endif

#include "utils/constants.h"
#include "utils/net_io_channel.h"
#include "OT/emp-ot.h"
const int SERVER = sci::ALICE;
const int CLIENT = sci::BOB;

// Keep above order of headers same -- constants.h has definitions of ALICE and BOB 
// 	Other headers are needed first to define io, iknpOT etc. -- so that in rest of the files
// 	these can be directly used.

sci::NetIO* io;
sci::IKNP<sci::NetIO>* iknpOT; //ALICE/server is the sender, Bob/client is the receiver
sci::IKNP<sci::NetIO>* iknpOTRoleReversed;//Reverse as above 
sci::KKOT<sci::NetIO>* kkot;
sci::OTPack<sci::NetIO> *otpack;

// For multiThreading
// NOTE : The otInstances are defined as follows:
// 	If threadNum is even, then ALICE is sender and BOB is receiver ; reverse if threadNum is odd
// 	Here threadNum \in [0,numThreads)
sci::NetIO* ioArr[numThreads];
sci::IKNP<sci::NetIO>* otInstanceArr[numThreads];
sci::OTPack<sci::NetIO>* otpackArr[numThreads];
sci::KKOT<sci::NetIO>* kkotInstanceArr[numThreads];
sci::PRG128* prg128Instance;
sci::PRG128* prgInstanceArr[numThreads];

#include "linear-primary.h"
Matmul<sci::NetIO, intType, sci::IKNP<sci::NetIO>>* matmulInstanceArr[numThreads];
Matmul<sci::NetIO, intType, sci::IKNP<sci::NetIO>>* matmulImpl;
#include "NonLinear/relu-interface.h"
ReLUProtocol<sci::NetIO, intType>* reluImplArr[numThreads];
ReLUProtocol<sci::NetIO, intType>* reluImpl;
#include "NonLinear/maxpool.h"
MaxPoolProtocol<sci::NetIO, intType>* maxpoolImplArr[numThreads];
MaxPoolProtocol<sci::NetIO, intType>* maxpoolImpl;
#include "NonLinear/argmax.h"
#include "LinearHE/conv-field.h"
#include "LinearHE/fc-field.h"
#include "LinearHE/elemwise-prod-field.h"
//Add extra headers here


// sci::OTIdeal<sci::NetIO>* otIdeal;
// Matmul<sci::NetIO, intType, sci::OTIdeal<sci::NetIO>>* matmulImpl;

ConvField* heConvImpl;
FCField* heFCImpl;
ElemWiseProdField* heProdImpl;

ArgMaxProtocol<sci::NetIO, intType>* argmaxImpl;
std::chrono::time_point<std::chrono::high_resolution_clock> startTimeTracker;
uint64_t communicationTracker[numThreads];

void checkIfUsingEigen(){
#ifdef USE_EIGEN
	std::cout<<"Using Eigen for Matmul"<<std::endl;
#else
	std::cout<<"Using normal Matmul"<<std::endl;
#endif
}

void assertFieldRun(){
	assert(sizeof(intType)==sizeof(uint64_t));
	assert(sizeof(signedIntType)==sizeof(int64_t));
	assert(bitlength>=32 && bitlength<=41);
}

#endif //GLOBALS_H__
