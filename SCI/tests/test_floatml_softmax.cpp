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

void Softmax2_old_thread(int tid, int chunk, int row_size, int m_bits, int e_bits,
	uint8_t** Row_s, uint8_t** Row_z, uint64_t** Row_m, uint64_t** Row_e) {

	int sz = chunk*row_size ;
	vector<FPArray> maxvecs ;

	for (int i = 0 ; i < chunk ; i++) 
		maxvecs.push_back(fpopArr[tid]->input(tid&1?3-__party:__party, row_size, Row_s[i], Row_z[i], Row_m[i], Row_e[i], m_bits, e_bits)) ;
	FPArray allmax = fpopArr[tid]->max(maxvecs) ;

	uint8_t *sub1_s = new uint8_t[sz] ; uint8_t *sub2_s = new uint8_t[sz] ;
	uint8_t *sub1_z = new uint8_t[sz] ; uint8_t *sub2_z = new uint8_t[sz] ;
	uint64_t *sub1_m = new uint64_t[sz] ; uint64_t *sub2_m = new uint64_t[sz] ;
	uint64_t *sub1_e = new uint64_t[sz] ; uint64_t *sub2_e = new uint64_t[sz] ;
	for (int i = 0, k = 0 ; i < chunk ; i++) {
		for (int j = 0 ; j < row_size ; j++, k++) {
			sub1_s[k] = Row_s[i][j] ; sub2_s[k] = allmax.s[i] ;
			sub1_z[k] = Row_z[i][j] ; sub2_z[k] = allmax.z[i] ;
			sub1_m[k] = Row_m[i][j] ; sub2_m[k] = allmax.m[i] ;
			sub1_e[k] = Row_e[i][j] ; sub2_e[k] = allmax.e[i] ;
		}
	}

	FPArray in_exp = fpmathArr[tid]->exp(fpopArr[tid]->sub(
		fpopArr[tid]->input(tid&1?3-__party:__party, sz, sub1_s, sub1_z, sub1_m, sub1_e, m_bits, e_bits), 
		fpopArr[tid]->input(tid&1?3-__party:__party, sz, sub2_s, sub2_z, sub2_m, sub2_e, m_bits, e_bits)
	)) ;

	uint8_t *sum_s = new uint8_t[row_size] ;
	uint8_t *sum_z = new uint8_t[row_size] ;
	uint64_t *sum_m = new uint64_t[row_size] ;
	uint64_t *sum_e = new uint64_t[row_size] ;
	vector<FPArray> sumvecs ;
	for (int i=0, k=0 ; i < chunk ; i++) {
		for (int j = 0 ; j < row_size ; j++, k++) {
			sum_s[j] = in_exp.s[k] ;
			sum_z[j] = in_exp.z[k] ;
			sum_m[j] = in_exp.m[k] ;
			sum_e[j] = in_exp.e[k] ;
		}
		sumvecs.push_back(fpopArr[tid]->input(tid&1?3-__party:__party, row_size, sum_s, sum_z, sum_m, sum_e, m_bits, e_bits)) ;
	}
	FPArray sums = fpopArr[tid]->treesum(sumvecs) ;

	uint8_t *den_s = new uint8_t[sz] ;
	uint8_t *den_z = new uint8_t[sz] ;
	uint64_t *den_m = new uint64_t[sz] ;
	uint64_t *den_e = new uint64_t[sz] ;
	for (int i = 0, k = 0 ; i < chunk ; i++) {
		for (int j = 0 ; j < row_size ; j++, k++) {
			den_s[k] = sums.s[i] ; 
			den_z[k] = sums.z[i] ;
			den_m[k] = sums.m[i] ;
			den_e[k] = sums.e[i] ;
		}
	}
	FPArray den = fpopArr[tid]->input(tid&1?3-__party:__party, sz, den_s, den_z, den_m, den_e, m_bits, e_bits) ;
	FPArray ans = fpopArr[tid]->div(in_exp, den) ;

	for (int i = 0, k = 0 ; i < chunk ; i++) {
		memcpy(Row_s[i], ans.s + i*row_size, row_size*sizeof(uint8_t)) ;
		memcpy(Row_z[i], ans.z + i*row_size, row_size*sizeof(uint8_t)) ;
		memcpy(Row_m[i], ans.m + i*row_size, row_size*sizeof(uint64_t)) ;
		memcpy(Row_e[i], ans.e + i*row_size, row_size*sizeof(uint64_t)) ;
	}
	
	delete[] sub1_s ; delete[] sub2_s ; delete[] sum_s ; delete[] den_s ;
	delete[] sub1_z ; delete[] sub2_z ; delete[] sum_z ; delete[] den_z ;
	delete[] sub1_m ; delete[] sub2_m ; delete[] sum_m ; delete[] den_m ;
	delete[] sub1_e ; delete[] sub2_e ; delete[] sum_e ; delete[] den_e ;
}

void Softmax2_old(
	int s1, int s2,
	vector<vector<FPArray>> &inArr,
	vector<vector<FPArray>> &outArr
	) {

	int m_bits, e_bits ;
	m_bits = inArr[0][0].m_bits ;
	e_bits = inArr[0][0].e_bits ;

	uint8_t** Row_s = new uint8_t*[s1] ;
	uint8_t** Row_z = new uint8_t*[s1] ;
	uint64_t** Row_m = new uint64_t*[s1] ;
	uint64_t** Row_e = new uint64_t*[s1] ;

	for (int i = 0 ; i < s1 ; i++) {
		Row_s[i] = new uint8_t[s2] ;
		Row_z[i] = new uint8_t[s2] ;
		Row_m[i] = new uint64_t[s2] ;
		Row_e[i] = new uint64_t[s2] ;

		for (int j = 0 ; j < s2 ; j++) {
			Row_s[i][j] = inArr[i][j].s[0] ;
			Row_z[i][j] = inArr[i][j].z[0] ;
			Row_m[i][j] = inArr[i][j].m[0] ;
			Row_e[i][j] = inArr[i][j].e[0] ;
		}
	}

	vector<int> chunks = get_chunks(s1, __nt) ;
	thread threads[MAX_THREADS] ;
	int offset = 0 ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			threads[i] = thread(Softmax2_old_thread, 
				i, chunks[i], s2, m_bits, e_bits,
				Row_s+offset, Row_z+offset, Row_m+offset, Row_e+offset
			) ;
			offset += chunks[i] ;
		}
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;

	for (int i = 0 ; i < s1 ; i++) {
		for (int j = 0 ; j < s2 ; j++) {
			outArr[i][j].s[0] = Row_s[i][j] ;
			outArr[i][j].z[0] = Row_z[i][j] ;
			outArr[i][j].m[0] = Row_m[i][j] ;
			outArr[i][j].e[0] = Row_e[i][j] ;
		}
	}

	for (int i = 0 ; i < s1 ; i++) {
		delete[] Row_s[i] ;
		delete[] Row_z[i] ;
		delete[] Row_m[i] ;
		delete[] Row_e[i] ;
	}

	delete[] Row_s ;
	delete[] Row_z ;
	delete[] Row_m ;
	delete[] Row_e ;
}

void Softmax2_thread(
	int tid, int mchunk, int n, int m_bits, int e_bits,
	uint8_t **in_s, uint8_t **in_z, uint64_t **in_m, uint64_t **in_e,
	uint8_t **out_s, uint8_t **out_z, uint64_t **out_m, uint64_t **out_e
	) {
	vector<FPArray> softin, softout ;

	for (int i = 0 ; i < mchunk ; i++)
		softin.push_back(fpopArr[tid]->input(tid&1?3-__party:__party, n, in_s[i], in_z[i], in_m[i], in_e[i], m_bits, e_bits)) ;

	softout = fpmathArr[tid]->softmax_secfloat(softin) ;

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

if (__old) {
	Softmax2_old(rows, sz, inp1, out) ;
} else {
	Softmax2(rows, sz, inp1, out) ;
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

