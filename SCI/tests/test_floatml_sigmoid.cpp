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

void Sigmoid_old_thread(
	int tid, int chunk, int m_bits, int e_bits,
	uint8_t *in_s, uint8_t *in_z, uint64_t *in_m, uint64_t *in_e, 
	uint8_t *out_s, uint8_t *out_z, uint64_t *out_m, uint64_t *out_e
	) {

	FPArray in_flat = fpopArr[tid]->input(tid&1?3-__party:__party, chunk, in_s, in_z, in_m, in_e, m_bits, e_bits) ;
	float zero = 0.0 ;
	FPArray zero_flat = fpopArr[tid]->input<float>(ALICE, chunk, zero, m_bits, e_bits) ;
	float one = 1.0 ;
	FPArray one_flat = fpopArr[tid]->input<float>(ALICE, chunk, one, m_bits, e_bits) ;

	FPArray out_flat = fpopArr[tid]->sub(zero_flat, in_flat) ;
	out_flat = fpmathArr[tid]->exp(out_flat) ;
	out_flat = fpopArr[tid]->add(one_flat, out_flat) ;
	out_flat = fpopArr[tid]->div(one_flat, out_flat) ;

	memcpy(out_s, out_flat.s, chunk*sizeof(uint8_t)) ;
	memcpy(out_z, out_flat.z, chunk*sizeof(uint8_t)) ;
	memcpy(out_m, out_flat.m, chunk*sizeof(uint64_t)) ;
	memcpy(out_e, out_flat.e, chunk*sizeof(uint64_t)) ;
}

void Sigmoid_old(
	int32_t s1, 
	vector<FPArray> &inArr, 
	vector<FPArray> &outArr) {
	int m_bits, e_bits ;
	m_bits = inArr[0].m_bits ;
	e_bits = inArr[0].e_bits ;

	uint8_t *in_s = new uint8_t[s1] ;
	uint8_t *in_z = new uint8_t[s1] ;
	uint64_t *in_m = new uint64_t[s1] ;
	uint64_t *in_e = new uint64_t[s1] ;
	for (int i = 0 ; i < s1 ; i++) {
		in_s[i] = inArr[i].s[0] ;
		in_z[i] = inArr[i].z[0] ;
		in_m[i] = inArr[i].m[0] ;
		in_e[i] = inArr[i].e[0] ;
	}

	uint8_t *out_s = new uint8_t[s1] ;
	uint8_t *out_z = new uint8_t[s1] ;
	uint64_t *out_m = new uint64_t[s1] ;
	uint64_t *out_e = new uint64_t[s1] ;

	vector<int> chunks = get_chunks(s1, __nt) ;
	int offset = 0 ;
	thread threads[MAX_THREADS] ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			threads[i] = thread(Sigmoid_old_thread,
				i, chunks[i], m_bits, e_bits,
				in_s+offset, in_z+offset, in_m+offset, in_e+offset,
				out_s+offset, out_z+offset, out_m+offset, out_e+offset
			) ;
			offset += chunks[i] ;
		}
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;

	for (int i = 0 ; i < s1 ; i++) {
		outArr[i].s[0] = out_s[i] ;
		outArr[i].z[0] = out_z[i] ;
		outArr[i].m[0] = out_m[i] ;
		outArr[i].e[0] = out_e[i] ;
	}

	delete[] in_s ; delete[] out_s ; 
	delete[] in_z ; delete[] out_z ; 
	delete[] in_m ; delete[] out_m ; 
	delete[] in_e ; delete[] out_e ; 
}

void Sigmoid_new_thread(
	int tid, int chunk, int m_bits, int e_bits,
	uint8_t *in_s, uint8_t *in_z, uint64_t *in_m, uint64_t *in_e, 
	uint8_t *out_s, uint8_t *out_z, uint64_t *out_m, uint64_t *out_e
	) {

	FPArray in_flat = fpopArr[tid]->input(tid&1?3-__party:__party, chunk, in_s, in_z, in_m, in_e, m_bits, e_bits) ;
	FPArray out_flat ;
	if (m_bits == 7 && e_bits == 8)
		out_flat = fpmathArr[tid]->sigmoid_bf16(in_flat) ;
	else if (m_bits == 23 && e_bits == 8)
		out_flat = fpmathArr[tid]->sigmoid_fp32(in_flat) ;

	memcpy(out_s, out_flat.s, chunk*sizeof(uint8_t)) ;
	memcpy(out_z, out_flat.z, chunk*sizeof(uint8_t)) ;
	memcpy(out_m, out_flat.m, chunk*sizeof(uint64_t)) ;
	memcpy(out_e, out_flat.e, chunk*sizeof(uint64_t)) ;
}

void Sigmoid_new(
	int32_t s1, 
	vector<FPArray> &inArr, 
	vector<FPArray> &outArr) {
	int m_bits, e_bits ;
	m_bits = inArr[0].m_bits ;
	e_bits = inArr[0].e_bits ;

	uint8_t *in_s = new uint8_t[s1] ;
	uint8_t *in_z = new uint8_t[s1] ;
	uint64_t *in_m = new uint64_t[s1] ;
	uint64_t *in_e = new uint64_t[s1] ;
	for (int i = 0 ; i < s1 ; i++) {
		in_s[i] = inArr[i].s[0] ;
		in_z[i] = inArr[i].z[0] ;
		in_m[i] = inArr[i].m[0] ;
		in_e[i] = inArr[i].e[0] ;
	}

	uint8_t *out_s = new uint8_t[s1] ;
	uint8_t *out_z = new uint8_t[s1] ;
	uint64_t *out_m = new uint64_t[s1] ;
	uint64_t *out_e = new uint64_t[s1] ;

	vector<int> chunks = get_chunks(s1, __nt) ;
	int offset = 0 ;
	thread threads[MAX_THREADS] ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			threads[i] = thread(Sigmoid_new_thread,
				i, chunks[i], m_bits, e_bits,
				in_s+offset, in_z+offset, in_m+offset, in_e+offset,
				out_s+offset, out_z+offset, out_m+offset, out_e+offset
			) ;
			offset += chunks[i] ;
		}
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;

	for (int i = 0 ; i < s1 ; i++) {
		outArr[i].s[0] = out_s[i] ;
		outArr[i].z[0] = out_z[i] ;
		outArr[i].m[0] = out_m[i] ;
		outArr[i].e[0] = out_e[i] ;
	}

	delete[] in_s ; delete[] out_s ; 
	delete[] in_z ; delete[] out_z ; 
	delete[] in_m ; delete[] out_m ; 
	delete[] in_e ; delete[] out_e ; 
}


int main (int __argc, char **__argv) {
int m_bits, e_bits ;
__init(__argc, __argv) ;
m_bits = __m_bits ;
e_bits = __e_bits ;

int sz = __sz1 ;

float* inp1_tmp = new float[1] ;
vector<FPArray> inp1 = make_vector_float(ALICE, sz) ;
vector<FPArray> out = make_vector_float(ALICE, sz) ;
for (int i = 0 ; i < sz ; i++) {
	if (__party == ALICE)
			inp1_tmp[0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;

	inp1[i] = __fp_op->input<float>(ALICE, 1, inp1_tmp, m_bits, e_bits) ;
}

auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = 0 ;
for (int i = 0 ; i < __nt ; i++)
	comm_start += (float)iopackArr[i]->get_comm() ;

if (__old) {
	Sigmoid_old(sz, inp1, out) ;
} else {
	Sigmoid_new(sz, inp1, out) ;
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

