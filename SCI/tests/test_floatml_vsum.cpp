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


void accum_thread(
	int tid, int chunk, int len, int mbits, int ebits,
	uint8_t **vec_s, uint8_t **vec_z, uint64_t **vec_m, uint64_t **vec_e,
	uint8_t *out_s, uint8_t *out_z, uint64_t *out_m, uint64_t *out_e
	) {

	vector<FPArray> vec(chunk) ;
	for (int i = 0 ; i < chunk ; i++)
		vec[i] = fpopArr[tid]->input(tid&1?3-__party:__party, len, vec_s[i], vec_z[i], vec_m[i], vec_e[i], mbits, ebits) ;

	FPArray out ;
	if (__old) {
		out = fpopArr[tid]->treesum(vec) ;
	} else {
		out = fpopArr[tid]->vector_sum(vec) ;
	}

	memcpy(out_s, out.s, chunk*sizeof(uint8_t)) ;
	memcpy(out_z, out.z, chunk*sizeof(uint8_t)) ;
	memcpy(out_m, out.m, chunk*sizeof(uint64_t)) ;
	memcpy(out_e, out.e, chunk*sizeof(uint64_t)) ;
}


void accum(vector<FPArray> &vec, vector<FPArray> &out) {
	int mbits, ebits ;
	int N, n, sz ;

	mbits = vec[0].m_bits ;
	ebits = vec[0].e_bits ;
	N = vec.size() ;
	n = vec[0].size ;

	uint8_t **vec_s = new uint8_t*[N] ;
	uint8_t **vec_z = new uint8_t*[N] ;
	uint64_t **vec_m = new uint64_t*[N] ;
	uint64_t **vec_e = new uint64_t*[N] ;

	for (int i = 0 ; i < N ; i++) {
		vec_s[i] = new uint8_t[n] ;
		vec_z[i] = new uint8_t[n] ;
		vec_m[i] = new uint64_t[n] ;
		vec_e[i] = new uint64_t[n] ;

		for (int j = 0 ; j < n ; j++) {
			vec_s[i][j] = vec[i].s[j] ; 
			vec_z[i][j] = vec[i].z[j] ; 
			vec_m[i][j] = vec[i].m[j] ; 
			vec_e[i][j] = vec[i].e[j] ; 
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
			threads[i] = thread(accum_thread,
				i, chunks[i], n, mbits, ebits,
				vec_s + offset, vec_z + offset, vec_m + offset, vec_e + offset,
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
		delete[] vec_s[i] ;
		delete[] vec_z[i] ;
		delete[] vec_m[i] ;
		delete[] vec_e[i] ;
	}

	delete[] vec_s ; delete[] out_s ;
	delete[] vec_z ; delete[] out_z ;
	delete[] vec_m ; delete[] out_m ;
	delete[] vec_e ; delete[] out_e ;
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
vector<FPArray> vec(sz1) ;
for (int i = 0 ; i < sz1 ; i++) {
	for (int j = 0 ; j < sz2 ; j++) {
		if (__party == ALICE)
			inp[j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;
	}
	vec[i] = __fp_op->input(ALICE, sz2, inp, m_bits, e_bits) ;
}

auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = 0 ;
for (int i = 0 ; i < __nt ; i++)
	comm_start += (float)iopackArr[i]->get_comm() ;

vector<FPArray> out = make_vector_float(ALICE, sz1) ;

accum(vec, out) ;	

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

