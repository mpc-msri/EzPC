#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include "floatml_utils.h"

using namespace std ;
using namespace sci ;

vector<int> get_chunks(int items, int slots) {
	int allocated, chunk, remaining ;

	chunk = items/slots ;
	vector<int> ret(slots, chunk) ;

	allocated = chunk*slots ;
	remaining = items - allocated ;
	for (int i = 0 ; i < remaining ; i++)
		ret[i]++ ;

	return ret ;
}

void dot_product_thread(
	int tid, int chunk, int len, int mbits, int ebits,
	uint8_t **dot1_s, uint8_t **dot1_z, uint64_t **dot1_m, uint64_t **dot1_e,
	uint8_t **dot2_s, uint8_t **dot2_z, uint64_t **dot2_m, uint64_t **dot2_e,
	uint8_t *out_s, uint8_t *out_z, uint64_t *out_m, uint64_t *out_e
	) {

	vector<FPArray> dot1(chunk), dot2(chunk) ;
	for (int i = 0 ; i < chunk ; i++) {
		dot1[i] = fpopArr[tid]->input(tid&1?3-__party:__party, len, dot1_s[i], dot1_z[i], dot1_m[i], dot1_e[i], mbits, ebits) ;
		dot2[i] = fpopArr[tid]->input(tid&1?3-__party:__party, len, dot2_s[i], dot2_z[i], dot2_m[i], dot2_e[i], mbits, ebits) ;
	}

	FPArray out = fpopArr[tid]->dot_product(dot1, dot2) ;
	memcpy(out_s, out.s, chunk*sizeof(uint8_t)) ;
	memcpy(out_z, out.z, chunk*sizeof(uint8_t)) ;
	memcpy(out_m, out.m, chunk*sizeof(uint64_t)) ;
	memcpy(out_e, out.e, chunk*sizeof(uint64_t)) ;
}

void dot_product(vector<FPArray> &dot1, vector<FPArray> &dot2, vector<FPArray> &out) {
	int mbits, ebits ;
	int N, n, sz ;

	mbits = dot1[0].m_bits ;
	ebits = dot1[0].e_bits ;
	N = dot1.size() ;
	n = dot1[0].size ;

	uint8_t **dot1_s = new uint8_t*[N] ;
	uint8_t **dot1_z = new uint8_t*[N] ;
	uint64_t **dot1_m = new uint64_t*[N] ;
	uint64_t **dot1_e = new uint64_t*[N] ;

	uint8_t **dot2_s = new uint8_t*[N] ;
	uint8_t **dot2_z = new uint8_t*[N] ;
	uint64_t **dot2_m = new uint64_t*[N] ;
	uint64_t **dot2_e = new uint64_t*[N] ;

	for (int i = 0 ; i < N ; i++) {
		dot1_s[i] = new uint8_t[n] ; dot2_s[i] = new uint8_t[n] ;
		dot1_z[i] = new uint8_t[n] ; dot2_z[i] = new uint8_t[n] ;
		dot1_m[i] = new uint64_t[n] ; dot2_m[i] = new uint64_t[n] ;
		dot1_e[i] = new uint64_t[n] ; dot2_e[i] = new uint64_t[n] ;

		for (int j = 0 ; j < N ; j++) {
			dot1_s[i][j] = dot1[i].s[j] ; dot2_s[i][j] = dot2[i].s[j] ;
			dot1_z[i][j] = dot1[i].z[j] ; dot2_z[i][j] = dot2[i].z[j] ;
			dot1_m[i][j] = dot1[i].m[j] ; dot2_m[i][j] = dot2[i].m[j] ;
			dot1_e[i][j] = dot1[i].e[j] ; dot2_e[i][j] = dot2[i].e[j] ;
		}
	}

	uint8_t *out_s = new uint8_t[N] ;
	uint8_t *out_z = new uint8_t[N] ;
	uint64_t *out_m = new uint64_t[N] ;
	uint64_t *out_e = new uint64_t[N] ;

	vector<int> chunks = get_chunks(N, __nt) ;
	thread threads[MAX_THREADS] ;
	int offset = 0 ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			threads[i] = thread(dot_product_thread,
				i, chunks[i], n, mbits, ebits,
				dot1_s + offset, dot1_z + offset, dot1_m + offset, dot1_e + offset,
				dot2_s + offset, dot2_z + offset, dot2_m + offset, dot2_e + offset,
				out_s + offset, out_z + offset, out_m + offset, out_e + offset
			) ;
			offset += chunks[i] ;
		}
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;

	for (int i = 0 ; i < N ; i++) {
		out[i].s[0] = out_s[i] ;
		out[i].z[0] = out_z[i] ;
		out[i].m[0] = out_m[i] ;
		out[i].e[0] = out_e[i] ;
	}

	delete[] dot1_s ; delete[] dot2_s ; delete[] out_s ;
	delete[] dot1_z ; delete[] dot2_z ; delete[] out_z ;
	delete[] dot1_m ; delete[] dot2_m ; delete[] out_m ;
	delete[] dot1_e ; delete[] dot2_e ; delete[] out_e ;
}

void dot_product_old_thread(
	int tid, int chunk, int len, int mbits, int ebits,
	uint8_t **dot1_s, uint8_t **dot1_z, uint64_t **dot1_m, uint64_t **dot1_e,
	uint8_t **dot2_s, uint8_t **dot2_z, uint64_t **dot2_m, uint64_t **dot2_e,
	uint8_t *out_s, uint8_t *out_z, uint64_t *out_m, uint64_t *out_e
	) {

	vector<FPArray> dot1(chunk), dot2(chunk) ;
	for (int i = 0 ; i < chunk ; i++) {
		dot1[i] = fpopArr[tid]->input(tid&1?3-__party:__party, len, dot1_s[i], dot1_z[i], dot1_m[i], dot1_e[i], mbits, ebits) ;
		dot2[i] = fpopArr[tid]->input(tid&1?3-__party:__party, len, dot2_s[i], dot2_z[i], dot2_m[i], dot2_e[i], mbits, ebits) ;
	}

	FPArray out = fpopArr[tid]->treesum(fpopArr[tid]->mul(dot1, dot2)) ;
	memcpy(out_s, out.s, chunk*sizeof(uint8_t)) ;
	memcpy(out_z, out.z, chunk*sizeof(uint8_t)) ;
	memcpy(out_m, out.m, chunk*sizeof(uint64_t)) ;
	memcpy(out_e, out.e, chunk*sizeof(uint64_t)) ;
}


void dot_product_old(vector<FPArray> &dot1, vector<FPArray> &dot2, vector<FPArray> &out) {
	int mbits, ebits ;
	int N, n, sz ;

	mbits = dot1[0].m_bits ;
	ebits = dot1[0].e_bits ;
	N = dot1.size() ;
	n = dot1[0].size ;

	uint8_t **dot1_s = new uint8_t*[N] ;
	uint8_t **dot1_z = new uint8_t*[N] ;
	uint64_t **dot1_m = new uint64_t*[N] ;
	uint64_t **dot1_e = new uint64_t*[N] ;

	uint8_t **dot2_s = new uint8_t*[N] ;
	uint8_t **dot2_z = new uint8_t*[N] ;
	uint64_t **dot2_m = new uint64_t*[N] ;
	uint64_t **dot2_e = new uint64_t*[N] ;

	for (int i = 0 ; i < N ; i++) {
		dot1_s[i] = new uint8_t[n] ; dot2_s[i] = new uint8_t[n] ;
		dot1_z[i] = new uint8_t[n] ; dot2_z[i] = new uint8_t[n] ;
		dot1_m[i] = new uint64_t[n] ; dot2_m[i] = new uint64_t[n] ;
		dot1_e[i] = new uint64_t[n] ; dot2_e[i] = new uint64_t[n] ;

		for (int j = 0 ; j < n ; j++) {
			dot1_s[i][j] = dot1[i].s[j] ; dot2_s[i][j] = dot2[i].s[j] ;
			dot1_z[i][j] = dot1[i].z[j] ; dot2_z[i][j] = dot2[i].z[j] ;
			dot1_m[i][j] = dot1[i].m[j] ; dot2_m[i][j] = dot2[i].m[j] ;
			dot1_e[i][j] = dot1[i].e[j] ; dot2_e[i][j] = dot2[i].e[j] ;
		}
	}

	uint8_t *out_s = new uint8_t[N] ;
	uint8_t *out_z = new uint8_t[N] ;
	uint64_t *out_m = new uint64_t[N] ;
	uint64_t *out_e = new uint64_t[N] ;

	vector<int> chunks = get_chunks(N, __nt) ;
	thread threads[MAX_THREADS] ;
	int offset = 0 ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			threads[i] = thread(dot_product_old_thread,
				i, chunks[i], n, mbits, ebits,
				dot1_s + offset, dot1_z + offset, dot1_m + offset, dot1_e + offset,
				dot2_s + offset, dot2_z + offset, dot2_m + offset, dot2_e + offset,
				out_s + offset, out_z + offset, out_m + offset, out_e + offset
			) ;
			offset += chunks[i] ;
		}
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;

	for (int i = 0 ; i < N ; i++) {
		out[i].s[0] = out_s[i] ;
		out[i].z[0] = out_z[i] ;
		out[i].m[0] = out_m[i] ;
		out[i].e[0] = out_e[i] ;
	}

	for (int i = 0 ; i < N ; i++) {
		delete[] dot1_s[i] ; delete[] dot2_s[i] ;
		delete[] dot1_z[i] ; delete[] dot2_z[i] ;
		delete[] dot1_m[i] ; delete[] dot2_m[i] ;
		delete[] dot1_e[i] ; delete[] dot2_e[i] ;
	}

	delete[] dot1_s ; delete[] dot2_s ; delete[] out_s ;
	delete[] dot1_z ; delete[] dot2_z ; delete[] out_z ;
	delete[] dot1_m ; delete[] dot2_m ; delete[] out_m ;
	delete[] dot1_e ; delete[] dot2_e ; delete[] out_e ;
}

int main (int __argc, char **__argv) {
int m_bits, e_bits ;
__init(__argc, __argv) ;
m_bits = __m_bits ;
e_bits = __e_bits ;

int sz1, sz2 ;
sz1 = __sz1 ;
sz2 = __sz2 ;

float* inp = new float[sz2] ;
vector<FPArray> dot1(sz1) ;
for (int i = 0 ; i < sz1 ; i++) {
	for (int j = 0 ; j < sz2 ; j++) {
		if (__party == ALICE)
			inp[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;
	}
	dot1[i] = __fp_op->input(ALICE, sz2, inp, m_bits, e_bits) ;
}

vector<FPArray> dot2(sz1) ;
for (int i = 0 ; i < sz1 ; i++) {
	for (int j = 0 ; j < sz2 ; j++) {
		if (__party == BOB)
			inp[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;
	}
	dot2[i] = __fp_op->input(BOB, sz2, inp, m_bits, e_bits) ;
}

auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = 0 ;
for (int i = 0 ; i < __nt ; i++)
	comm_start += (float)iopackArr[i]->get_comm() ;

auto out = make_vector_float(ALICE, sz1) ;
if (__old) {
	dot_product_old(dot1, dot2, out) ;	
} else {
	dot_product(dot1, dot2, out) ;	
}


long long t = time_from(start);
float comm_end = 0 ;
for (int i = 0 ; i < __nt ; i++)
	comm_end += (float)iopackArr[i]->get_comm() ;

float add_time = t/1000.0 ;
float add_comm = (comm_end - comm_start)/(1<<20) ;
int add_round = __iopack->get_rounds() - initial_rounds ;

printf("Time = %f\n", add_time) ;
printf("Comm = %f\n", add_comm) ;
printf("Round = %d\n", add_round) ;

return 0;
}

