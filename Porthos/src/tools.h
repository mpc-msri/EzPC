/*
 * 	BMR_BGW_aux.cpp
 * 
 *      Author: Aner Ben-Efraim, Satyanarayana
 * 	
 * 	year: 2016
 *	
 *	Modified for crypTFlow. 
 */

#ifndef TOOLS_H
#define TOOLS_H
#pragma once

#include <stdio.h> 
#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <vector>
#include <time.h>
#include <string>
#include <openssl/sha.h>
#include <math.h>
#include <sstream>
#include "AESObject.h"
#include "ParallelAESObject.h"
#include "connect.h"
#include "globals.h"

//Error codes
#define PARSE_ERROR 101

//For time benchmarking
/********************Nested Timing Support*************/

//Usage:
//INIT_TIMER to create timer instance
//START_TIMER to register timer
//STOP_TIMER("") to get the time taken since START_TIMER
//PAUSE_TIMER to pause the timer

//NOTE: This can even be used in nested form. For e.g.
/*
INIT_TIMER;
START_TIMER;
for(int i=0; i<x; i++){
	INIT_TIMER;
	START_TIMER;
	sleep(1);
	STOP_TIMER("Time taken for 1 loop iteration");
}
STOP_TIMER("Time taken for full loop");
*/

#define TIMING
 
#ifdef TIMING

#define INIT_TIMER auto start_timer = std::chrono::high_resolution_clock::now(); \
	int pause_timer = 0;
#define START_TIMER  start_timer = std::chrono::high_resolution_clock::now();
#define PAUSE_TIMER(name) pause_timer += std::chrono::duration_cast<std::chrono::milliseconds>( \
            std::chrono::high_resolution_clock::now()-start_timer).count(); \
	std::cout << "[PAUSING TIMER] RUNTIME till now of " << name << ": " << pause_timer<<" ms"<<endl;
#define STOP_TIMER(name) cout << "------------------------------------" << endl; std::cout << "[STOPPING TIMER] Total RUNTIME of " << name << ": " << \
	std::chrono::duration_cast<std::chrono::milliseconds>( \
            std::chrono::high_resolution_clock::now()-start_timer \
    ).count() + pause_timer << " ms " << std::endl; 

#else

#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)

#endif
/***************************************************/

extern int partyNum;

extern AESObject* aes_common;
extern AESObject* aes_indep;
extern AESObject* aes_a_1;
extern AESObject* aes_a_2;
extern AESObject* aes_b_1;
extern AESObject* aes_b_2;
extern AESObject* aes_c_1;
extern AESObject* aes_share_conv_bit_shares_p0_p2;
extern AESObject* aes_share_conv_bit_shares_p1_p2;
extern AESObject* aes_share_conv_shares_mod_odd_p0_p2;
extern AESObject* aes_share_conv_shares_mod_odd_p1_p2;
extern AESObject* aes_comp_msb_shares_lsb_p0_p2;
extern AESObject* aes_comp_msb_shares_lsb_p1_p2;
extern AESObject* aes_comp_msb_shares_bit_vec_p0_p2;
extern AESObject* aes_comp_msb_shares_bit_vec_p1_p2;
extern AESObject* aes_conv_opti_a_1;
extern AESObject* aes_conv_opti_a_2;
extern AESObject* aes_conv_opti_b_1;
extern AESObject* aes_conv_opti_b_2;
extern AESObject* aes_conv_opti_c_1;

extern ParallelAESObject* aes_parallel;

extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

/************************************ Some statistics functions ***********************/
void start_time();

void end_time();

void start_rounds();

void end_rounds();

void aggregateCommunication();

void start_m();

double diff(timespec start, timespec end);

void end_m();

/************************************ Some MatMul functions ************************/
void matrixMultEigen(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c, 			
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b);

void matrixMultEigen(const vector<vector<porthosSecretType>> &a, 
		const vector<vector<porthosSecretType>> &b, 
		vector<vector<porthosSecretType>> &c, 			
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b);

void matrixMultEigen(porthosSecretType* a, 
		porthosSecretType* b, 
		porthosSecretType* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b);

/*************************************** End of MatMul functions ***********************************/

/*************************************** Some other STANDALONE EXECTION utility functions **************************/

porthosSecretType divideMyTypeSA(porthosSecretType a, 
		porthosSecretType b);

porthosSecretType dividePlainSA(porthosSecretType a, 
		int b);

void dividePlainSA(vector<porthosSecretType> &vec, 
		int divisor);

porthosSecretType multiplyMyTypesSA(porthosSecretType a, 
		porthosSecretType b, 
		int shift);

/*************************************** Other small utility functions ************************************/
void XORVectors(const vector<smallType> &a, 
		const vector<smallType> &b, 
		vector<smallType> &c, 
		size_t size);

void log_print(string str);

void error(string str);

size_t adversary(size_t party);

smallType subtractModPrime(smallType a, 
		smallType b);

void wrapAround(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<smallType> &c, 
		size_t size);

inline smallType addModPrime(smallType a, 
		smallType b)
{
	return additionModPrime[a][b];
}

inline smallType multiplyModPrime(smallType a, 
		smallType b)
{
	return multiplicationModPrime[a][b];
}

inline smallType wrapAround(porthosSecretType a, 
		porthosSecretType b)
{
	return (a > MINUS_ONE - b);
}

/************************************* Some functions with AES and resharing ****************************/
void populateBitsVector(vector<smallType> &vec, 
		string r_type, 
		size_t size);

void sharesOfBits(vector<smallType> &bit_shares_x_1, 
		vector<smallType> &bit_shares_x_2, 
		const vector<porthosSecretType> &x, 
		size_t size, 
		string r_type);

void sharesOfLSB(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<porthosSecretType> &r, 
		size_t size, 
		string r_type);

void sharesOfLSB(vector<porthosSecretType> &share_1, 
		vector<porthosSecretType> &share_2, 
		const vector<porthosSecretType> &r, 
		size_t size, 
		string r_type);

void sharesOfBitVector(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type);

void sharesOfBitVector(vector<porthosSecretType> &share_1, 
		vector<porthosSecretType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type);

//Split shares of a vector of porthosSecretType into shares (randomness is independent)
void splitIntoShares(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &a1, 
		vector<porthosSecretType> &a2, 
		size_t size);

/***************************** Basic utility functions for Convolution drivers ************************/
void zero_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& vec, 
		int d1, 
		int d2, 
		int d3, 
		int d4);

void subtract_2D_vectors(const vector< vector<porthosSecretType> >& inp_l, 
			const vector< vector<porthosSecretType> >& inp_r, 
			vector< vector<porthosSecretType> >& out, 
			int d1, 
			int d2);

void add_2D_vectors(const vector< vector<porthosSecretType> >& inp_l, 
		const vector< vector<porthosSecretType> >& inp_r, 
		vector< vector<porthosSecretType> >& out, 
		int d1, 
		int d2);

void zero_2D_vector(vector< vector<porthosSecretType> >& vec, 
		int d1, 
		int d2);


void add_5D_vectors(vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_l, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_r, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4, 
		int d5);

void subtract_5D_vectors(vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_l, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_r, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4, 
		int d5);

void flatten_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& input,
                        vector<porthosSecretType>& output,
                        int d1,
                        int d2,
                        int d3,
                        int d4,
                        int d5);

void deflatten_5D_vector(vector<porthosSecretType>& input,
                        vector< vector< vector< vector< vector<porthosSecretType> > > > >& output,
                        int d1,
                        int d2,
                        int d3,
                        int d4,
                        int d5);

void add_4D_vectors(vector< vector< vector< vector<porthosSecretType> > > >& inp_l, 
		vector< vector< vector< vector<porthosSecretType> > > >& inp_r, 
		vector< vector< vector< vector<porthosSecretType> > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4);

void subtract_4D_vectors(vector< vector< vector< vector<porthosSecretType> > > >& inp_l, 
			vector< vector< vector< vector<porthosSecretType> > > >& inp_r, 
			vector< vector< vector< vector<porthosSecretType> > > >& out, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void flatten_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input, 
			vector<porthosSecretType>& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void deflatten_4D_vector(vector<porthosSecretType>& input, 
			vector< vector< vector< vector<porthosSecretType> > > >& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void flatten_2D_vector(vector< vector<porthosSecretType> >& input, 
			vector<porthosSecretType>& output, 
			int d1, 
			int d2);

void deflatten_2D_vector(vector<porthosSecretType>& input, 
			vector< vector<porthosSecretType> >& output, 
			int d1, 
			int d2);

void send_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& input,
			int party,
			int d1, 
			int d2, 
			int d3, 
			int d4,
			int d5);

void receive_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& recv,
			int party,
			int d1, 
			int d2, 
			int d3, 
			int d4,
			int d5);

void send_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void receive_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& recv, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void send_2_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input1,
			vector< vector< vector< vector<porthosSecretType> > > >& input2,
			int d11, 
			int d12, 
			int d13, 
			int d14,
			int d21, 
			int d22, 
			int d23, 
			int d24);

void receive_2_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input1,
			vector< vector< vector< vector<porthosSecretType> > > >& input2,
			int d11, 
			int d12, 
			int d13, 
			int d14,
			int d21, 
			int d22, 
			int d23, 
			int d24);

void send_2D_vector(vector< vector<porthosSecretType> >& input, 
			int d1, 
			int d2);

void receive_2D_vector(vector< vector<porthosSecretType> >& recv, 
			int d1, 
			int d2);

void populate_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& vec, 
			int d1, 
			int d2, 
			int d3, 
			int d4, 
			int d5, 
			string type);

void populate_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& vec, 
			int d1, 
			int d2, 
			int d3, 
			int d4, 
			string type);

void populate_2D_vector(vector< vector<porthosSecretType> >& vec, 
			int d1, 
			int d2, 
			string type);

void populate_AES_Arr(porthosSecretType* arr, 
		porthosLongUnsignedInt size, 
		string type);

void add_2_Arr(porthosSecretType* arr1, 
		porthosSecretType* arr2, 
		porthosSecretType* arr, 
		porthosLongUnsignedInt size);

void subtract_2_Arr(porthosSecretType* arr1, 
		porthosSecretType* arr2, 
		porthosSecretType* arr, 
		porthosLongUnsignedInt size);

/********************************** Porthos info display functions *****************************/
void porthos_throw_error(int code);

void show_porthos_mode();

void print_linear(porthosSecretType var, 
		string type);

void maxPoolReshape(const vector<porthosSecretType> &vec, 
		vector<porthosSecretType> &vecShaped,
		size_t ih, 
		size_t iw, 
		size_t D, 
		size_t B,  
		size_t fh, 
		size_t fw, 
		size_t sy, 
		size_t sx);

/*=============================================================================================*/

/********************************* Some function defined in this header file itself **********************/
// Template functions
template<typename T>
void populateRandomVector(vector<T> &vec, 
		size_t size,  
		string r_type, 
		string neg_type);

template<typename T>
void addVectors(const vector<T> &a, 
		const vector<T> &b, 
		vector<T> &c, 
		size_t size);

template<typename T>
void subtractVectors(const vector<T> &a, 
		const vector<T> &b, 
		vector<T> &c, 
		size_t size);

template<typename T>
void copyVectors(const vector<T> &a, 
		vector<T> &b, 
		size_t size);

template<typename T1, typename T2>
void addModuloOdd(const vector<T1> &a, 
		const vector<T2> &b, 
		vector<porthosSecretType> &c, 
		size_t size);

template<typename T1, typename T2>
void subtractModuloOdd(const vector<T1> &a, 
		const vector<T2> &b, 
		vector<porthosSecretType> &c, 
		size_t size);

template<typename T1, typename T2>
void addModuloOddArr(T1* a, 
		T2* b, 
		porthosSecretType* c, 
		size_t size);

template<typename T1, typename T2>
void subtractModuloOddArr(T1* a, 
		T2* b, 
		porthosSecretType* c, 
		size_t size);

template<typename T1, typename T2>
porthosSecretType addModuloOdd(T1 a, 
		T2 b);

template<typename T1, typename T2>
porthosSecretType subtractModuloOdd(T1 a, 
		T2 b);

template<typename T>
void sharesModuloOdd(vector<porthosSecretType> &shares_1, 
		vector<porthosSecretType> &shares_2, 
		const vector<T> &x, 
		size_t size, 
		string r_type);


/***************************** Functions defined in header itself *******************************/

template<typename T>
void populateRandomVector(vector<T> &vec, 
		size_t size, 
		string r_type, 
		string neg_type)
{	
#ifdef DEBUG
	assert((neg_type == "NEGATIVE" or neg_type == "POSITIVE") && "invalid negativeness type for populateRandomVector");
	assert(sizeof(T) == sizeof(porthosSecretType) && "Probably only need 64-bit numbers");
#endif
	porthosSecretType sign = 1;
	if (r_type == "COMMON")
	{
		if (neg_type == "NEGATIVE")
		{		
			if (partyNum == PARTY_B)
				sign = MINUS_ONE;

			if (sizeof(T) == sizeof(porthosSecretType))
			{
#ifdef PRECOMPUTEAES
				aes_common->fillWithRandomBits64((porthosSecretType*)vec.data(), size);
				for (size_t i = 0; i < size; ++i)
					vec[i] *= sign;

#else
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_common->get64Bits();
#endif		
			}
			else
			{
#ifdef PRECOMPUTEAES
				aes_common->fillWithRandomBits8((uint8_t *)vec.data(), size);
				for (size_t i = 0; i < size; ++i)
					vec[i] *= sign;
#else
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_common->get8Bits();
#endif
			}
		}
		
		if (neg_type == "POSITIVE")
		{
			if (sizeof(T) == sizeof(porthosSecretType))
			{
#ifdef PRECOMPUTEAES
				aes_common->fillWithRandomBits64((porthosSecretType*)vec.data(), size);
#else
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_common->get64Bits();		
#endif
			}
			else
			{
#ifdef PRECOMPUTEAES
				aes_common->fillWithRandomBits8((uint8_t *)vec.data(), size);
#else
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_common->get8Bits();
#endif
			}			
		}
	}

	if (r_type == "INDEP")
	{
		if (neg_type == "NEGATIVE")
		{		
			if (partyNum == PARTY_B)
				sign = MINUS_ONE;

			if (sizeof(T) == sizeof(porthosSecretType))
			{
#ifdef PRECOMPUTEAES
				aes_indep->fillWithRandomBits64((porthosSecretType*)vec.data(), size);
				for (size_t i = 0; i < size; ++i)
					vec[i] *= sign;			
#else
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_indep->get64Bits();		
#endif
			}
			else
			{
#ifdef PRECOMPUTEAES
				aes_indep->fillWithRandomBits8((uint8_t *)vec.data(), size);
				for (size_t i = 0; i < size; ++i)
					vec[i] *= sign;
#else

				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_indep->get8Bits();		
#endif
			}		
		}
		
		if (neg_type == "POSITIVE")
		{
			if (sizeof(T) == sizeof(porthosSecretType))
			{
#ifdef PRECOMPUTEAES
				aes_indep->fillWithRandomBits64((porthosSecretType*)vec.data(), size);
#else
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_indep->get64Bits();		
#endif
			}
			else
			{
#ifdef PRECOMPUTEAES
				aes_indep->fillWithRandomBits8((uint8_t *)vec.data(), size);
#else
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_indep->get8Bits();		
#endif
			}		
		}
	}

	if (r_type == "a_1")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for a_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(porthosSecretType) && "sizeof(T) == sizeof(porthosSecretType)");
#endif

#ifdef PRECOMPUTEAES
		aes_a_1->fillWithRandomBits64((porthosSecretType *)vec.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_a_1->get64Bits();
#endif
	}
	else if (r_type == "b_1")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for b_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(porthosSecretType) && "sizeof(T) == sizeof(porthosSecretType)");
#endif
	
#ifdef PRECOMPUTEAES
		aes_b_1->fillWithRandomBits64((porthosSecretType *)vec.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_b_1->get64Bits();
#endif
	}
	else if (r_type == "c_1")
	{	
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for c_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(porthosSecretType) && "sizeof(T) == sizeof(porthosSecretType)");
#endif

#ifdef PRECOMPUTEAES
		aes_c_1->fillWithRandomBits64((porthosSecretType *)vec.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_c_1->get64Bits();
#endif
	}
	else if (r_type == "a_2")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for a_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(porthosSecretType) && "sizeof(T) == sizeof(porthosSecretType)");
#endif

#ifdef PRECOMPUTEAES
		aes_a_2->fillWithRandomBits64((porthosSecretType *)vec.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_a_2->get64Bits();
#endif
	}
	else if (r_type == "b_2")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for b_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(porthosSecretType) && "sizeof(T) == sizeof(porthosSecretType)");
#endif

#ifdef PRECOMPUTEAES
		aes_b_2->fillWithRandomBits64((porthosSecretType *)vec.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_b_2->get64Bits();
#endif
	}
}


template<typename T>
void fillVector(T val, 
		vector<T> c, 
		size_t size)
{
	for (size_t i = 0; i < size; i++)
	{
		c[i] = val;
	}
}

template<typename T>
void addVectors(const vector<T> &a, 
		const vector<T> &b, 
		vector<T> &c, 
		size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] + b[i];
}

template<typename T>
void subtractVectors(const vector<T> &a, 
		const vector<T> &b, 
		vector<T> &c, 
		size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] - b[i];
}

template<typename T>
void copyVectors(const vector<T> &a, 
		vector<T> &b, 
		size_t size)
{
	for (size_t i = 0; i < size; ++i)
		b[i] = a[i];
}


template<typename T1, typename T2>
void addModuloOdd(const vector<T1> &a, 
		const vector<T2> &b, 
		vector<porthosSecretType> &c, 
		size_t size)
{
	assert((sizeof(T1) == sizeof(porthosSecretType) or sizeof(T2) == sizeof(porthosSecretType)) && "At least one type should be porthosSecretType for typecast to work");

	for (size_t i = 0; i < size; ++i)
	{
		if (a[i] == MINUS_ONE and b[i] == MINUS_ONE)
			c[i] = 0;
		else 
			c[i] = (a[i] + b[i] + wrapAround(a[i], b[i])) % MINUS_ONE;
	}
}

template<typename T1, typename T2>
void subtractModuloOdd(const vector<T1> &a, 
		const vector<T2> &b, 
		vector<porthosSecretType> &c, 
		size_t size)
{
	vector<porthosSecretType> temp(size);
	for (size_t i = 0; i < size; ++i)
		temp[i] = MINUS_ONE - b[i];

	addModuloOdd<T1, porthosSecretType>(a, temp, c, size);
}

template<typename T1, typename T2>
void addModuloOddArr(T1* a, 
		T2* b, 
		porthosSecretType* c, 
		size_t size)
{
	assert((sizeof(T1) == sizeof(porthosSecretType) or sizeof(T2) == sizeof(porthosSecretType)) && "At least one type should be porthosSecretType for typecast to work");

	for (size_t i = 0; i < size; ++i)
	{
		if (a[i] == MINUS_ONE and b[i] == MINUS_ONE)
			c[i] = 0;
		else 
			c[i] = (a[i] + b[i] + wrapAround(a[i], b[i])) % MINUS_ONE;
	}
}


template<typename T1, typename T2>
void subtractModuloOddArr(T1* a, 
		T2* b, 
		porthosSecretType* c, 
		size_t size)
{
	porthosSecretType* temp = new porthosSecretType[size];
	for (size_t i = 0; i < size; ++i){
		temp[i] = MINUS_ONE - b[i];
	}
	addModuloOddArr<T1, porthosSecretType>(a, temp, c, size);
	delete[] temp;
}



template<typename T>
void sharesModuloOdd(vector<porthosSecretType> &shares_1, 
		vector<porthosSecretType> &shares_2, 
		const vector<T> &x, 
		size_t size, 
		string r_type)
{
#ifdef DEBUG
	assert((r_type == "PRG_COMM_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfBits");
	assert(partyNum == PARTY_C);
#endif

	if (r_type == "COMMON")
	{
#ifdef PRECOMPUTEAES
		aes_common->fillWithRandomModuloOddBits(shares_1.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = aes_common->randModuloOdd();
#endif
		subtractModuloOdd<T, porthosSecretType>(x, shares_1, shares_2, size);
	}

	else if (r_type == "INDEP")
	{
#ifdef PRECOMPUTEAES
		aes_indep->fillWithRandomModuloOddBits(shares_1.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = aes_indep->randModuloOdd();
#endif
		subtractModuloOdd<T, porthosSecretType>(x, shares_1, shares_2, size);
	}
	
	else if(r_type == "PRG_COMM_OPTI")
	{
#ifdef PRECOMPUTEAES
		aes_share_conv_shares_mod_odd_p0_p2->fillWithRandomModuloOddBits(shares_1.data(), size/2);
		aes_share_conv_shares_mod_odd_p1_p2->fillWithRandomModuloOddBits(shares_2.data() + (size/2), (size - (size/2)));
#else
		for(size_t i=0;i<(size/2);i++)
		{
			shares_1[i] = aes_share_conv_shares_mod_odd_p0_p2->randModuloOdd();
		}
		for(size_t i=(size/2);i<size;i++)
		{
			shares_2[i] = aes_share_conv_shares_mod_odd_p1_p2->randModuloOdd();
		}
#endif

		subtractModuloOddArr(x.data(), shares_1.data(), shares_2.data(), (size/2));
		subtractModuloOddArr(x.data() + (size/2), shares_2.data() + (size/2), shares_1.data() + (size/2), (size - (size/2)));
	}
}

template<typename T1, typename T2>
porthosSecretType addModuloOdd(T1 a, 
		T2 b)
{
	assert((sizeof(T1) == sizeof(porthosSecretType) or sizeof(T2) == sizeof(porthosSecretType)) && "At least one type should be porthosSecretType for typecast to work");

	if (a == MINUS_ONE and b == MINUS_ONE)
		return 0;
	else 
		return (a + b + wrapAround(a, b)) % MINUS_ONE;
}

template<typename T1, typename T2>
porthosSecretType subtractModuloOdd(T1 a, 
		T2 b)
{
	porthosSecretType temp = MINUS_ONE - b;
	return addModuloOdd<T1, porthosSecretType>(a, temp);
}

template<typename T>
T** mallocDyn2DArr(int32_t s0, 
		int32_t s1)
{
	T** arr = new T*[s0];
	for(int i=0;i<s0;i++)
		arr[i] = new T[s1];
	return arr;
}

template<typename T>
T**** mallocDyn4DArr(int32_t s0, 
		int32_t s1, 
		int32_t s2, 
		int32_t s3)
{
	T**** arr = new T***[s0];
	for(int i=0;i<s0;i++){
		arr[i] = new T**[s1];
		for(int j=0;j<s1;j++){
			arr[i][j] = new T*[s2];
			for(int k=0;k<s2;k++){
				arr[i][j][k] = new T[s3];
			}
		}
	}
	return arr;
}

#endif

