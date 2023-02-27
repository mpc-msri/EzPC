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

void Softmax2_thread(
	int tid, int mchunk, int n, int m_bits, int e_bits,
	uint8_t **in_s, uint8_t **in_z, uint64_t **in_m, uint64_t **in_e,
	uint8_t **out_s, uint8_t **out_z, uint64_t **out_m, uint64_t **out_e
	) {
	vector<FPArray> softin, softout ;

	for (int i = 0 ; i < mchunk ; i++)
		softin.push_back(fpopArr[tid]->input(tid&1?3-__party:__party, n, in_s[i], in_z[i], in_m[i], in_e[i], m_bits, e_bits)) ;

	if (__old)
		softout = fpmathArr[tid]->softmax_secfloat(softin) ;
	else
		softout = fpmathArr[tid]->softmax_beacon(softin) ;

	for (int i = 0 ; i < mchunk ; i++) {
		memcpy(out_s[i], softout[i].s, n*sizeof(uint8_t)) ;
		memcpy(out_z[i], softout[i].z, n*sizeof(uint8_t)) ;
		memcpy(out_m[i], softout[i].m, n*sizeof(uint64_t)) ;
		memcpy(out_e[i], softout[i].e, n*sizeof(uint64_t)) ;
	}
}

void Softmax2(
	int32_t s1, 
	int32_t s2, 
	vector<vector<FPArray>> &inArr, 
	vector<vector<FPArray>> &outArr) {
	int m_bits = inArr[0][0].m_bits ;
	int e_bits = inArr[0][0].e_bits ;

	uint8_t **row_s = new uint8_t*[s1] ;
	uint8_t **row_z = new uint8_t*[s1] ;
	uint64_t **row_m = new uint64_t*[s1] ;
	uint64_t **row_e = new uint64_t*[s1] ;

	uint8_t **out_s = new uint8_t*[s1] ;
	uint8_t **out_z = new uint8_t*[s1] ;
	uint64_t **out_m = new uint64_t*[s1] ;
	uint64_t **out_e = new uint64_t*[s1] ;

	for (int i = 0 ; i < s1 ; i++) {
		row_s[i] = new uint8_t[s2] ;
		row_z[i] = new uint8_t[s2] ;
		row_m[i] = new uint64_t[s2] ;
		row_e[i] = new uint64_t[s2] ;

		out_s[i] = new uint8_t[s2] ;
		out_z[i] = new uint8_t[s2] ;
		out_m[i] = new uint64_t[s2] ;
		out_e[i] = new uint64_t[s2] ;

		for (int j = 0 ; j < s2 ; j++) {
			row_s[i][j] = inArr[i][j].s[0] ;
			row_z[i][j] = inArr[i][j].z[0] ;
			row_m[i][j] = inArr[i][j].m[0] ;
			row_e[i][j] = inArr[i][j].e[0] ;
		}
	}

	vector<int> chunks = get_chunks(s1, __nt) ;
	thread threads[MAX_THREADS] ;
	int offset = 0 ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			threads[i] = thread(Softmax2_thread,
				i, chunks[i], s2, m_bits, e_bits,
				row_s+offset, row_z+offset, row_m+offset, row_e+offset,
				out_s+offset, out_z+offset, out_m+offset, out_e+offset
			) ;
			offset += chunks[i] ;
		}
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;


	for (int i = 0 ; i < s1 ; i++) {
		for (int j = 0 ; j < s2 ; j++) {
			outArr[i][j].s[0] = out_s[i][j] ;
			outArr[i][j].z[0] = out_z[i][j] ;
			outArr[i][j].m[0] = out_m[i][j] ;
			outArr[i][j].e[0] = out_e[i][j] ;
		}
	}

	for (int i = 0 ; i < s1 ; i++) {
		delete[] row_s[i] ; delete[] out_s[i] ;
		delete[] row_z[i] ; delete[] out_z[i] ;
		delete[] row_m[i] ; delete[] out_m[i] ;
		delete[] row_e[i] ; delete[] out_e[i] ;
	}

	delete[] row_s ; delete[] out_s ;
	delete[] row_z ; delete[] out_z ;
	delete[] row_m ; delete[] out_m ;
	delete[] row_e ; delete[] out_e ;
}

int main (int __argc, char **__argv) {
int m_bits, e_bits ;
__init(__argc, __argv) ;
m_bits = __m_bits ;
e_bits = __e_bits ;

int rows, sz ;
rows = __sz1 ;
sz = __sz2 ;

float* inp1_tmp = new float[1] ;
vector<vector<FPArray>> inp1 = make_vector_float(ALICE, rows, sz) ;
for (int i = 0 ; i < rows ; i++) {
	for (int j = 0 ; j < sz ; j++) {
		if (__party == ALICE)
			inp1_tmp[0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;
		inp1[i][j] = __fp_op->input<float>(ALICE, 1, inp1_tmp, m_bits, e_bits) ;
	}
}

vector<vector<FPArray>> out = make_vector_float(ALICE, rows, sz) ;

auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = 0 ;
for (int i = 0 ; i < __nt ; i++)
	comm_start += (float)iopackArr[i]->get_comm() ;

Softmax2(rows, sz, inp1, out) ;

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

