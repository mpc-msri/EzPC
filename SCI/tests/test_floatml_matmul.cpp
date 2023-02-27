#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

// #include "FloatingPoint/floating-point.h"
// #include "FloatingPoint/fp-math.h"
// #include "floatml_utils.h"
#include "library_float.h"

using namespace std ;
using namespace sci ;

// vector<int> get_chunks(int items, int slots) {
// 	int allocated, chunk, remaining ;

// 	chunk = items/slots ;
// 	vector<int> ret(slots, chunk) ;

// 	allocated = chunk*slots ;
// 	remaining = items - allocated ;
// 	for (int i = 0 ; i < remaining ; i++)
// 		ret[i]++ ;

// 	return ret ;
// }

// void MatMul_thread(
// 	int tid, int m_chunk, int n, int p, int m_bits, int e_bits, FPMatrix B,
// 	uint8_t *A_s, uint8_t *A_z, uint64_t *A_m, uint64_t *A_e,
// 	uint8_t *res_s, uint8_t *res_z, uint64_t *res_m, uint64_t *res_e
// 	) {

// 	FPMatrix A_chunk = fpopArr[tid]->input(tid&1?3-__party:__party, m_chunk, n, A_s, A_z, A_m, A_e, m_bits, e_bits) ;
// 	FPMatrix res ;
// 	if (__old) {
// 		res = fpopArr[tid]->matrix_multiplication_secfloat(A_chunk, B, __chunk_exp) ;
// 	} else {
// 		res = fpopArr[tid]->matrix_multiplication_beacon(A_chunk, B, __chunk_exp) ;
// 	}

// 	memcpy(res_s, res.s, m_chunk*p*sizeof(uint8_t)) ;
// 	memcpy(res_z, res.z, m_chunk*p*sizeof(uint8_t)) ;
// 	memcpy(res_m, res.m, m_chunk*p*sizeof(uint64_t)) ;
// 	memcpy(res_e, res.e, m_chunk*p*sizeof(uint64_t)) ;
// }

// void MatMul(int32_t m, int32_t n, int32_t p, 
// 	vector<vector<FPArray>> &A, 
// 	vector<vector<FPArray>> &B, 
// 	vector<vector<FPArray>> &C) {

// 	if (m <= __nt && p > __nt) {
// 		auto BT = make_vector_float(ALICE, p, n) ;
// 		auto AT = make_vector_float(ALICE, n, m) ;
// 		auto CT = make_vector_float(ALICE, p, m) ;

// 		for (int i = 0 ; i < n ; i++) {
// 			for (int j = 0 ; j < m ; j++) {
// 				AT[i][j] = A[j][i] ;
// 			}
// 		}

// 		for (int i = 0 ; i < p ; i++)
// 			for (int j = 0 ; j < n ; j++)
// 				BT[i][j] = B[j][i] ;

// 		MatMul (p, n, m, BT, AT, CT) ;

// 		for (int i = 0 ; i < m ; i++)
// 			for (int j = 0 ; j < p ; j++)
// 				C[i][j] = CT[j][i] ;

// 		return ;
// 	} 

// 	int m_bits = A[0][0].m_bits ;
// 	int e_bits = B[0][0].e_bits ;

// 	uint8_t *A_s = new uint8_t[m*n] ;
// 	uint8_t *A_z = new uint8_t[m*n] ;
// 	uint64_t *A_m = new uint64_t[m*n] ;
// 	uint64_t *A_e = new uint64_t[m*n] ;
// 	for (int i = 0, k=0 ; i < m ; i++) {
// 		for (int j = 0 ; j < n ; j++, k++) {
// 			A_s[k] = A[i][j].s[0] ;
// 			A_z[k] = A[i][j].z[0] ;
// 			A_m[k] = A[i][j].m[0] ;
// 			A_e[k] = A[i][j].e[0] ;
// 		}
// 	}

// 	uint8_t *B_s = new uint8_t[n*p] ;
// 	uint8_t *B_z = new uint8_t[n*p] ;
// 	uint64_t *B_m = new uint64_t[n*p] ;
// 	uint64_t *B_e = new uint64_t[n*p] ;
// 	for (int i = 0, k = 0 ; i < n ; i++) {
// 		for (int j = 0 ; j < p ; j++, k++) {
// 			B_s[k] = B[i][j].s[0] ;
// 			B_z[k] = B[i][j].z[0] ;
// 			B_m[k] = B[i][j].m[0] ;
// 			B_e[k] = B[i][j].e[0] ;
// 		}
// 	}
// 	FPMatrix mat2 = __fp_op->input(__party, n, p, B_s, B_z, B_m, B_e, m_bits, e_bits) ;

// 	uint8_t *res_s = new uint8_t[m*p] ;
// 	uint8_t *res_z = new uint8_t[m*p] ;
// 	uint64_t *res_m = new uint64_t[m*p] ;
// 	uint64_t *res_e = new uint64_t[m*p] ;

// 	vector<int> chunks = get_chunks(m, __nt) ;
// 	thread threads[MAX_THREADS] ;
// 	int m_offset, A_offset, res_offset ;
// 	m_offset = A_offset = res_offset = 0 ;
// 	for (int i = 0 ; i < __nt ; i++) {
// 		if (chunks[i] > 0) {
// 			threads[i] = thread(MatMul_thread,
// 				i, chunks[i], n, p, m_bits, e_bits, mat2,
// 				A_s+A_offset, A_z+A_offset, A_m+A_offset, A_e+A_offset,
// 				res_s+res_offset, res_z+res_offset, res_m+res_offset, res_e+res_offset
// 			) ;

// 			m_offset += chunks[i] ;
// 			A_offset += chunks[i]*n ;
// 			res_offset += chunks[i]*p ;
// 		}
// 	}

// 	for (int i = 0 ; i < __nt ; i++) {
// 		if (chunks[i] > 0)
// 			threads[i].join() ;
// 	}


// 	for (int i = 0, k = 0 ; i < m ; i++) {
// 		for (int j = 0 ; j < p ; j++, k++) {
// 			C[i][j].s[0] = res_s[k] ;
// 			C[i][j].z[0] = res_z[k] ;
// 			C[i][j].m[0] = res_m[k] ;
// 			C[i][j].e[0] = res_e[k] ;
// 		}
// 	}

// 	delete[] A_s ; delete[] B_s ; delete[] res_s ;
// 	delete[] A_z ; delete[] B_z ; delete[] res_z ;
// 	delete[] A_m ; delete[] B_m ; delete[] res_m ;
// 	delete[] A_e ; delete[] B_e ; delete[] res_e ; 
// }


int main (int __argc, char **__argv) {
	int m_bits, e_bits ;
	__init(__argc, __argv) ;
	m_bits = __m_bits ;
	e_bits = __e_bits ;

	int m, n, p ;
	m = __sz1 ;
	n = __sz2 ;
	p = __sz3 ;

	vector<vector<FPArray>> matA = make_vector_float_rand(ALICE, m, n) ;
	vector<vector<FPArray>> matB = make_vector_float_rand(ALICE, n, p) ;
	vector<vector<FPArray>> matC = make_vector_float_rand(ALICE, m, p) ;

	MatMul(m, n, p, matA, matB, matC) ;
	__end() ;

	return 0;
}


