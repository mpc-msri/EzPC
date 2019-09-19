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
#include "../util/Config.h"
#include "../util/TedKrovetzAesNiWrapperC.h"
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
#include "sgx_trts.h"
#include "../Enclave/Enclave_t.h"

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
extern AESObject* threaded_aes_common[NO_CORES];
extern AESObject* threaded_aes_indep[NO_CORES];
extern AESObject* threaded_aes_a_1[NO_CORES];
extern AESObject* threaded_aes_a_2[NO_CORES];
extern AESObject* threaded_aes_b_1[NO_CORES];
extern AESObject* threaded_aes_b_2[NO_CORES];
extern AESObject* threaded_aes_c_1[NO_CORES];
extern AESObject* threaded_aes_share_conv_bit_shares_p0_p2[NO_CORES];
extern AESObject* threaded_aes_share_conv_bit_shares_p1_p2[NO_CORES];
extern AESObject* threaded_aes_share_conv_shares_mod_odd_p0_p2[NO_CORES];
extern AESObject* threaded_aes_share_conv_shares_mod_odd_p1_p2[NO_CORES];
extern AESObject* threaded_aes_comp_msb_shares_lsb_p0_p2[NO_CORES];
extern AESObject* threaded_aes_comp_msb_shares_lsb_p1_p2[NO_CORES];
extern AESObject* threaded_aes_comp_msb_shares_bit_vec_p0_p2[NO_CORES];
extern AESObject* threaded_aes_comp_msb_shares_bit_vec_p1_p2[NO_CORES];
extern AESObject* threaded_aes_conv_opti_a_1[NO_CORES];
extern AESObject* threaded_aes_conv_opti_a_2[NO_CORES];
extern AESObject* threaded_aes_conv_opti_b_1[NO_CORES];
extern AESObject* threaded_aes_conv_opti_b_2[NO_CORES];
extern AESObject* threaded_aes_conv_opti_c_1[NO_CORES];

extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

/************************************ Some MatMul functions ************************/

void matrixMultEigen(vector<aramisSecretType> &a, vector<aramisSecretType> &b, vector<aramisSecretType> &c, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b);

void matrixMultEigen_fast_with_Eigen(const vector<aramisSecretType> &a, const vector<aramisSecretType> &b, vector<aramisSecretType> &c, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b);

void matrixMultEigen(vector< vector<aramisSecretType> > &a, vector< vector<aramisSecretType> > &b, vector< vector<aramisSecretType> > &c, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b);

void matrixMultEigen(uint64_t* a, uint64_t* b, uint64_t* c,
						size_t rows, size_t common_dim, size_t columns,
						size_t transpose_a, size_t transpose_b);

/*************************************** End of MatMul functions ***********************************/

/*************************************** Some other STANDALONE EXECTION utility functions **************************/
aramisSecretType divideMyTypeSA(aramisSecretType a, 
		aramisSecretType b);

aramisSecretType dividePlainSA(aramisSecretType a, 
		int b);

void dividePlainSA(vector<aramisSecretType> &vec, 
		int divisor);

aramisSecretType multiplyMyTypesSA(aramisSecretType a, 
		aramisSecretType b, 
		int shift);
aramisSecretType divideMyTypeSA(aramisSecretType a, aramisSecretType b);

/*************************************** Other small utility functions ************************************/
void XORVectors(const vector<smallType> &a, 
		const vector<smallType> &b, 
		vector<smallType> &c, 
		size_t size);

size_t adversary(size_t party);

smallType subtractModPrime(smallType a, 
		smallType b);

void wrapAround(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b, 
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

inline smallType wrapAround(aramisSecretType a, 
		aramisSecretType b)
{
	return (a > MINUS_ONE - b);
}

/************************************* Some functions with AES and resharing ****************************/
void populateBitsVector(vector<smallType> &vec, 
		string r_type, 
		size_t size);

void sharesOfBits(vector<smallType> &bit_shares_x_1, 
		vector<smallType> &bit_shares_x_2, 
		const vector<aramisSecretType> &x, 
		size_t size, 
		string r_type);

void sharesOfLSB(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<aramisSecretType> &r, 
		size_t size, 
		string r_type);

void sharesOfLSB(vector<aramisSecretType> &share_1, 
		vector<aramisSecretType> &share_2, 
		const vector<aramisSecretType> &r, 
		size_t size, 
		string r_type);

void sharesOfBitVector(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type);

void sharesOfBitVector(vector<aramisSecretType> &share_1, 
		vector<aramisSecretType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type);

//Split shares of a vector of aramisSecretType into shares (randomness is independent)
void splitIntoShares(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &a1, 
		vector<aramisSecretType> &a2, 
		size_t size);

void populateRandomVectorParallel(vector<uint64_t> vec, 
		uint64_t size, 
		string type, 
		string extra);

void sharesModuloOddParallel(vector<aramisSecretType> &shares_1, 
		vector<aramisSecretType> &shares_2, 
		vector<smallType> &x, 
		size_t size, 
		string r_type);

void sharesOfBitsParallel(vector<smallType> &bit_shares_x_1, 
		vector<smallType> &bit_shares_x_2, 
		vector<aramisSecretType> &x, 
		size_t size, 
		string r_type);

void populate_4D_vectorParallel(vector< vector< vector< vector<aramisSecretType> > > >& vec, 
		int d1, 
		int d2, 
		int d3, 
		int d4, 
		string type);

void populate_2D_vectorParallel(vector< vector<aramisSecretType> >& vec, 
		int d1, 
		int d2, 
		string type);

void populate_AES_ArrParallel(uint64_t* arr, 
		uint64_t size, 
		string type);

void sharesOfBitsPrimaryParallel(smallType* arr, 
		size_t size, 
		int which_party);


/***************************** Basic utility functions for Convolution drivers ************************/
void zero_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& vec, 
		int d1, 
		int d2, 
		int d3, 
		int d4);

void subtract_2D_vectors(vector< vector<aramisSecretType> >& inp_l, 
			vector< vector<aramisSecretType> >& inp_r, 
			vector< vector<aramisSecretType> >& out, 
			int d1, 
			int d2);

void add_2D_vectors(vector< vector<aramisSecretType> >& inp_l, 
		vector< vector<aramisSecretType> >& inp_r, 
		vector< vector<aramisSecretType> >& out, 
		int d1, 
		int d2);

void zero_2D_vector(vector< vector<aramisSecretType> >& vec, 
		int d1, 
		int d2);

void add_4D_vectors(vector< vector< vector< vector<aramisSecretType> > > >& inp_l, 
		vector< vector< vector< vector<aramisSecretType> > > >& inp_r, 
		vector< vector< vector< vector<aramisSecretType> > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4);

void subtract_4D_vectors(vector< vector< vector< vector<aramisSecretType> > > >& inp_l, 
			vector< vector< vector< vector<aramisSecretType> > > >& inp_r, 
			vector< vector< vector< vector<aramisSecretType> > > >& out, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void flatten_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& input, 
			vector<aramisSecretType>& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void deflatten_4D_vector(vector<aramisSecretType>& input, 
			vector< vector< vector< vector<aramisSecretType> > > >& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void flatten_2D_vector(vector< vector<aramisSecretType> >& input, 
			vector<aramisSecretType>& output, 
			int d1, 
			int d2);

void deflatten_2D_vector(vector<aramisSecretType>& input, 
			vector< vector<aramisSecretType> >& output, 
			int d1, 
			int d2);

void send_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& input, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void receive_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& recv, 
			int d1, 
			int d2, 
			int d3, 
			int d4);

void send_2_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& input1,
			vector< vector< vector< vector<aramisSecretType> > > >& input2,
			int d11, 
			int d12, 
			int d13, 
			int d14,
			int d21, 
			int d22, 
			int d23, 
			int d24);

void receive_2_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& input1,
			vector< vector< vector< vector<aramisSecretType> > > >& input2,
			int d11, 
			int d12, 
			int d13, 
			int d14,
			int d21, 
			int d22, 
			int d23, 
			int d24);

void send_2D_vector(vector< vector<aramisSecretType> >& input, 
			int d1, 
			int d2);

void receive_2D_vector(vector< vector<aramisSecretType> >& recv, 
			int d1, 
			int d2);

void populate_4D_vector(vector< vector< vector< vector<aramisSecretType> > > >& vec, 
			int d1, 
			int d2, 
			int d3, 
			int d4, 
			string type);

void populate_2D_vector(vector< vector<aramisSecretType> >& vec, 
			int d1, 
			int d2, 
			string type);

void populate_AES_Arr(aramisSecretType* arr, 
		uint64_t size, 
		string type);

void add_2_Arr(aramisSecretType* arr1, 
		aramisSecretType* arr2, 
		aramisSecretType* arr, 
		uint64_t size);

void subtract_2_Arr(aramisSecretType* arr1, 
		aramisSecretType* arr2, 
		aramisSecretType* arr, 
		uint64_t size);

/********************************** Aramis info display functions *****************************/
void show_aramis_mode();

void peek_inside_enclave(int worker_thread_num);

void maxPoolReshape(const vector<aramisSecretType> &vec, 
		vector<aramisSecretType> &vecShaped,
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
		vector<aramisSecretType> &c, 
		size_t size);

template<typename T1, typename T2>
void subtractModuloOdd(const vector<T1> &a, 
		const vector<T2> &b, 
		vector<aramisSecretType> &c, 
		size_t size);

template<typename T1, typename T2>
void addModuloOddArr(T1* a, 
		T2* b, 
		aramisSecretType* c, 
		size_t size);

template<typename T1, typename T2>
void subtractModuloOddArr(T1* a, 
		T2* b, 
		aramisSecretType* c, 
		size_t size);

template<typename T1, typename T2>
aramisSecretType addModuloOdd(T1 a, 
		T2 b);

template<typename T1, typename T2>
aramisSecretType subtractModuloOdd(T1 a, 
		T2 b);

template<typename T>
void sharesModuloOdd(vector<aramisSecretType> &shares_1, 
		vector<aramisSecretType> &shares_2, 
		const vector<T> &x, 
		size_t size, 
		string r_type);

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

/***************************** Functions defined in header itself *******************************/

template<typename T>
void populateRandomVector(vector<T> &vec, 
		size_t size, 
		string r_type, 
		string neg_type)
{	
	// assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for populateRandomVector");
	assert((neg_type == "NEGATIVE" or neg_type == "POSITIVE") && "invalid negativeness type for populateRandomVector");
	// assert(sizeof(T) == sizeof(aramisSecretType) && "Probably only need 64-bit numbers");
	// assert(r_type == "COMMON" && "Only common randomness mode required currently");

	aramisSecretType sign = 1;
	if (r_type == "COMMON")
	{
		if (neg_type == "NEGATIVE")
		{		
			if (partyNum == PARTY_B)
				sign = MINUS_ONE;

			if (sizeof(T) == sizeof(aramisSecretType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_common->get64Bits();
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_common->get8Bits();
			}
		}
		
		if (neg_type == "POSITIVE")
		{
			if (sizeof(T) == sizeof(aramisSecretType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_common->get64Bits();		
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_common->get8Bits();
			}			
		}
	}

	if (r_type == "INDEP")
	{
		if (neg_type == "NEGATIVE")
		{		
			if (partyNum == PARTY_B)
				sign = MINUS_ONE;

			if (sizeof(T) == sizeof(aramisSecretType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_indep->get64Bits();		
			}
			else
			{

				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_indep->get8Bits();		
			}		
		}
		
		if (neg_type == "POSITIVE")
		{
			if (sizeof(T) == sizeof(aramisSecretType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_indep->get64Bits();		
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_indep->get8Bits();		
			}		
		}
	}

	if (r_type == "a_1")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for a_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_a_1->get64Bits();
	}
	else if (r_type == "b_1")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for b_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif
	
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_b_1->get64Bits();
	}
	else if (r_type == "c_1")
	{	
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for c_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_c_1->get64Bits();
	}
	else if (r_type == "a_2")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for a_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_a_2->get64Bits();
	}
	else if (r_type == "b_2")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for b_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_b_2->get64Bits();
	}
	//Calls linked to the convolution optimization. 
	else if (r_type == "opti_a_1")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for a_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_conv_opti_a_1->get64Bits();
	}
	else if (r_type == "opti_b_1")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for b_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_conv_opti_b_1->get64Bits();
	}
	else if (r_type == "opti_c_1")
	{	
#ifdef DEBUG
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for c_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_conv_opti_c_1->get64Bits();
	}
	else if (r_type == "opti_a_2")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for a_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_conv_opti_a_2->get64Bits();
	}
	else if (r_type == "opti_b_2")
	{
#ifdef DEBUG
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for b_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(aramisSecretType) && "sizeof(T) == sizeof(aramisSecretType)");
#endif

		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_conv_opti_b_2->get64Bits();
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
	//print_integer(size);
	for (size_t i = 0; i < size; ++i){
		//print_integer(i);
		c[i] = a[i] - b[i];
	}
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
		vector<aramisSecretType> &c, 
		size_t size)
{
	assert((sizeof(T1) == sizeof(aramisSecretType) or sizeof(T2) == sizeof(aramisSecretType)) && "At least one type should be aramisSecretType for typecast to work");

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
		vector<aramisSecretType> &c, 
		size_t size)
{
	vector<aramisSecretType> temp(size);
	for (size_t i = 0; i < size; ++i)
		temp[i] = MINUS_ONE - b[i];

	addModuloOdd<T1, aramisSecretType>(a, temp, c, size);
}

template<typename T1, typename T2>
void addModuloOddArr(T1* a, 
		T2* b, 
		aramisSecretType* c, 
		size_t size)
{
	assert((sizeof(T1) == sizeof(aramisSecretType) or sizeof(T2) == sizeof(aramisSecretType)) && "At least one type should be aramisSecretType for typecast to work");

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
		aramisSecretType* c, 
		size_t size)
{
	aramisSecretType temp[size];
	for (size_t i = 0; i < size; ++i)
		temp[i] = MINUS_ONE - b[i];

	addModuloOddArr<T1, aramisSecretType>(a, temp, c, size);
}

template<typename T>
void sharesModuloOdd(vector<aramisSecretType> &shares_1, 
		vector<aramisSecretType> &shares_2, 
		const vector<T> &x, 
		size_t size, 
		string r_type)
{
	assert((r_type == "SHARE_CONV_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfBits");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = aes_common->randModuloOdd();
		subtractModuloOdd<T, aramisSecretType>(x, shares_1, shares_2, size);
	}

	else if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = aes_indep->randModuloOdd();
		subtractModuloOdd<T, aramisSecretType>(x, shares_1, shares_2, size);
	}

	else if(r_type == "SHARE_CONV_OPTI"){
		for(size_t i=0;i<(size/2);i++)
		{
			shares_1[i] = aes_share_conv_shares_mod_odd_p0_p2->randModuloOdd();
		}
		for(size_t i=(size/2);i<size;i++)
		{
			shares_2[i] = aes_share_conv_shares_mod_odd_p1_p2->randModuloOdd();
		}

		subtractModuloOddArr(x.data(), shares_1.data(), shares_2.data(), (size/2));
		subtractModuloOddArr(x.data() + (size/2), shares_2.data() + (size/2), shares_1.data() + (size/2), (size - (size/2)));

	}
}

template<typename T1, typename T2>
aramisSecretType addModuloOdd(T1 a, 
		T2 b)
{
	assert((sizeof(T1) == sizeof(aramisSecretType) or sizeof(T2) == sizeof(aramisSecretType)) && "At least one type should be aramisSecretType for typecast to work");

	if (a == MINUS_ONE and b == MINUS_ONE)
		return 0;
	else 
		return (a + b + wrapAround(a, b)) % MINUS_ONE;
}

template<typename T1, typename T2>
aramisSecretType subtractModuloOdd(T1 a, 
		T2 b)
{
	aramisSecretType temp = MINUS_ONE - b;
	return addModuloOdd<T1, aramisSecretType>(a, temp);
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

