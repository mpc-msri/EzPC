/*
Authors: Anwesh Bhattacharya
Copyright:
Copyright (c) 2021 Microsoft Research
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

#include "globals_float.h"
#include "library_float.h"
#include <math.h>

using namespace std;
using namespace sci;

int __chunk_exp = 15 ;

void MatMul_thread(
	int tid, int m_chunk, int n, int p, int m_bits, int e_bits, FPMatrix B,
	uint8_t *A_s, uint8_t *A_z, uint64_t *A_m, uint64_t *A_e,
	uint8_t *res_s, uint8_t *res_z, uint64_t *res_m, uint64_t *res_e)
{

	FPMatrix A_chunk = fpopArr[tid]->input(WHICHPARTY, m_chunk, n, A_s, A_z, A_m, A_e, m_bits, e_bits);
	FPMatrix res = fpopArr[tid]->matrix_multiplication_secfloat(A_chunk, B, __chunk_exp);

	memcpy(res_s, res.s, m_chunk * p * sizeof(uint8_t));
	memcpy(res_z, res.z, m_chunk * p * sizeof(uint8_t));
	memcpy(res_m, res.m, m_chunk * p * sizeof(uint64_t));
	memcpy(res_e, res.e, m_chunk * p * sizeof(uint64_t));
}

void MatMul(int32_t m, int32_t n, int32_t p,
			vector<vector<FPArray>> &A,
			vector<vector<FPArray>> &B,
			vector<vector<FPArray>> &C)
{

	if (m <= __nt && p > __nt)
	{
		vector<vector<FPArray>> BT = make_vector_float(ALICE, p, n);
		vector<vector<FPArray>> AT = make_vector_float(ALICE, n, m);
		vector<vector<FPArray>> CT = make_vector_float(ALICE, p, m);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				AT[i][j] = A[j][i];
			}
		}

		for (int i = 0; i < p; i++)
			for (int j = 0; j < n; j++)
				BT[i][j] = B[j][i];

		MatMul(p, n, m, BT, AT, CT);

		for (int i = 0; i < m; i++)
			for (int j = 0; j < p; j++)
				C[i][j] = CT[j][i];

		return;
	}

	int m_bits = A[0][0].m_bits;
	int e_bits = A[0][0].e_bits;

	uint8_t *A_s = new uint8_t[m * n];
	uint8_t *A_z = new uint8_t[m * n];
	uint64_t *A_m = new uint64_t[m * n];
	uint64_t *A_e = new uint64_t[m * n];
	for (int i = 0, k = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++, k++)
		{
			A_s[k] = A[i][j].s[0];
			A_z[k] = A[i][j].z[0];
			A_m[k] = A[i][j].m[0];
			A_e[k] = A[i][j].e[0];
		}
	}

	uint8_t *B_s = new uint8_t[n * p];
	uint8_t *B_z = new uint8_t[n * p];
	uint64_t *B_m = new uint64_t[n * p];
	uint64_t *B_e = new uint64_t[n * p];
	for (int i = 0, k = 0; i < n; i++)
	{
		for (int j = 0; j < p; j++, k++)
		{
			B_s[k] = B[i][j].s[0];
			B_z[k] = B[i][j].z[0];
			B_m[k] = B[i][j].m[0];
			B_e[k] = B[i][j].e[0];
		}
	}
	FPMatrix mat2 = __fp_op->input(__party, n, p, B_s, B_z, B_m, B_e, m_bits, e_bits);

	uint8_t *res_s = new uint8_t[m * p];
	uint8_t *res_z = new uint8_t[m * p];
	uint64_t *res_m = new uint64_t[m * p];
	uint64_t *res_e = new uint64_t[m * p];

	vector<int> chunks = get_chunks(m, __nt);
	thread threads[MAX_THREADS];
	int m_offset, A_offset, res_offset;
	m_offset = A_offset = res_offset = 0;
	for (int i = 0; i < __nt; i++)
	{
		if (chunks[i] > 0)
		{
			threads[i] = thread(MatMul_thread,
								i, chunks[i], n, p, m_bits, e_bits, mat2,
								A_s + A_offset, A_z + A_offset, A_m + A_offset, A_e + A_offset,
								res_s + res_offset, res_z + res_offset, res_m + res_offset, res_e + res_offset);

			m_offset += chunks[i];
			A_offset += chunks[i] * n;
			res_offset += chunks[i] * p;
		}
	}

	for (int i = 0; i < __nt; i++)
	{
		if (chunks[i] > 0)
			threads[i].join();
	}

	for (int i = 0, k = 0; i < m; i++)
	{
		for (int j = 0; j < p; j++, k++)
		{
			C[i][j].m_bits = m_bits;
			C[i][j].e_bits = e_bits;

			C[i][j].s[0] = res_s[k];
			C[i][j].z[0] = res_z[k];
			C[i][j].m[0] = res_m[k];
			C[i][j].e[0] = res_e[k];
		}
	}

	delete[] A_s;
	delete[] B_s;
	delete[] res_s;
	delete[] A_z;
	delete[] B_z;
	delete[] res_z;
	delete[] A_m;
	delete[] B_m;
	delete[] res_m;
	delete[] A_e;
	delete[] B_e;
	delete[] res_e;
}

void vectorSum_thread(
	int tid, int chunk, int colsize, int m_bits, int e_bits,
	uint8_t **Row_s, uint8_t **Row_z, uint64_t **Row_m, uint64_t **Row_e,
	uint8_t *row_s, uint8_t *row_z, uint64_t *row_m, uint64_t *row_e)
{
	vector<FPArray> sums;
	for (int i = 0; i < chunk; i++)
	{
		sums.push_back(
			fpopArr[tid]->input(
				WHICHPARTY, colsize, Row_s[i], Row_z[i], Row_m[i], Row_e[i], m_bits, e_bits));
	}

	FPArray vsum = fpopArr[tid]->treesum(sums);

	memcpy(row_s, vsum.s, chunk * sizeof(uint8_t));
	memcpy(row_z, vsum.z, chunk * sizeof(uint8_t));
	memcpy(row_m, vsum.m, chunk * sizeof(uint64_t));
	memcpy(row_e, vsum.e, chunk * sizeof(uint64_t));
}

void getBiasDer(int32_t m, int32_t s2, vector<vector<FPArray>> &batchSoftDer, vector<FPArray> &biasDer)
{
	int m_bits, e_bits;
	m_bits = batchSoftDer[0][0].m_bits;
	e_bits = batchSoftDer[0][0].e_bits;

	uint8_t **Row_s = new uint8_t *[s2];
	uint8_t **Row_z = new uint8_t *[s2];
	uint64_t **Row_m = new uint64_t *[s2];
	uint64_t **Row_e = new uint64_t *[s2];

	uint8_t *row_s = new uint8_t[s2];
	uint8_t *row_z = new uint8_t[s2];
	uint64_t *row_m = new uint64_t[s2];
	uint64_t *row_e = new uint64_t[s2];

	for (int i = 0; i < s2; i++)
	{
		Row_s[i] = new uint8_t[m];
		Row_z[i] = new uint8_t[m];
		Row_m[i] = new uint64_t[m];
		Row_e[i] = new uint64_t[m];

		for (int j = 0; j < m; j++)
		{
			Row_s[i][j] = batchSoftDer[j][i].s[0];
			Row_z[i][j] = batchSoftDer[j][i].z[0];
			Row_m[i][j] = batchSoftDer[j][i].m[0];
			Row_e[i][j] = batchSoftDer[j][i].e[0];
		}
	}

	vector<int> chunks = get_chunks(s2, __nt);
	thread threads[MAX_THREADS];
	int offset = 0;
	for (int i = 0; i < __nt; i++)
	{
		if (chunks[i] > 0)
		{
			threads[i] = thread(vectorSum_thread,
								i, chunks[i], m, m_bits, e_bits,
								Row_s + offset, Row_z + offset, Row_m + offset, Row_e + offset,
								row_s + offset, row_z + offset, row_m + offset, row_e + offset);
			offset += chunks[i];
		}
	}

	for (int i = 0; i < __nt; i++)
		if (chunks[i] > 0)
			threads[i].join();

	for (int i = 0; i < s2; i++)
	{
		biasDer[i].m_bits = m_bits;
		biasDer[i].e_bits = e_bits;

		biasDer[i].s[0] = row_s[i];
		biasDer[i].z[0] = row_z[i];
		biasDer[i].m[0] = row_m[i];
		biasDer[i].e[0] = row_e[i];
	}

	for (int i = 0; i < s2; i++)
	{
		delete[] Row_s[i];
		delete[] Row_z[i];
		delete[] Row_m[i];
		delete[] Row_e[i];
	}

	delete[] Row_s;
	delete[] row_s;
	delete[] Row_z;
	delete[] row_z;
	delete[] Row_m;
	delete[] row_m;
	delete[] Row_e;
	delete[] row_e;
}

void Softmax2_thread(
	int tid, int mchunk, int n, int m_bits, int e_bits,
	uint8_t **in_s, uint8_t **in_z, uint64_t **in_m, uint64_t **in_e,
	uint8_t **out_s, uint8_t **out_z, uint64_t **out_m, uint64_t **out_e)
{
	vector<FPArray> softin, softout;

	for (int i = 0; i < mchunk; i++)
		softin.push_back(fpopArr[tid]->input(WHICHPARTY, n, in_s[i], in_z[i], in_m[i], in_e[i], m_bits, e_bits));

	softout = fpmathArr[tid]->softmax_secfloat(softin);
	for (int i = 0; i < mchunk; i++)
	{
		memcpy(out_s[i], softout[i].s, n * sizeof(uint8_t));
		memcpy(out_z[i], softout[i].z, n * sizeof(uint8_t));
		memcpy(out_m[i], softout[i].m, n * sizeof(uint64_t));
		memcpy(out_e[i], softout[i].e, n * sizeof(uint64_t));
	}
}

void Softmax2(
	int32_t s1,
	int32_t s2,
	vector<vector<FPArray>> &inArr,
	vector<vector<FPArray>> &outArr)
{
	int m_bits = inArr[0][0].m_bits;
	int e_bits = inArr[0][0].e_bits;

	uint8_t **row_s = new uint8_t *[s1];
	uint8_t **row_z = new uint8_t *[s1];
	uint64_t **row_m = new uint64_t *[s1];
	uint64_t **row_e = new uint64_t *[s1];

	uint8_t **out_s = new uint8_t *[s1];
	uint8_t **out_z = new uint8_t *[s1];
	uint64_t **out_m = new uint64_t *[s1];
	uint64_t **out_e = new uint64_t *[s1];

	for (int i = 0; i < s1; i++)
	{
		row_s[i] = new uint8_t[s2];
		row_z[i] = new uint8_t[s2];
		row_m[i] = new uint64_t[s2];
		row_e[i] = new uint64_t[s2];

		out_s[i] = new uint8_t[s2];
		out_z[i] = new uint8_t[s2];
		out_m[i] = new uint64_t[s2];
		out_e[i] = new uint64_t[s2];

		for (int j = 0; j < s2; j++)
		{
			row_s[i][j] = inArr[i][j].s[0];
			row_z[i][j] = inArr[i][j].z[0];
			row_m[i][j] = inArr[i][j].m[0];
			row_e[i][j] = inArr[i][j].e[0];
		}
	}

	vector<int> chunks = get_chunks(s1, __nt);
	thread threads[MAX_THREADS];
	int offset = 0;
	for (int i = 0; i < __nt; i++)
	{
		if (chunks[i] > 0)
		{
			threads[i] = thread(Softmax2_thread,
								i, chunks[i], s2, m_bits, e_bits,
								row_s + offset, row_z + offset, row_m + offset, row_e + offset,
								out_s + offset, out_z + offset, out_m + offset, out_e + offset);
			offset += chunks[i];
		}
	}

	for (int i = 0; i < __nt; i++)
		if (chunks[i] > 0)
			threads[i].join();

	for (int i = 0; i < s1; i++)
	{
		for (int j = 0; j < s2; j++)
		{
			outArr[i][j].s[0] = out_s[i][j];
			outArr[i][j].z[0] = out_z[i][j];
			outArr[i][j].m[0] = out_m[i][j];
			outArr[i][j].e[0] = out_e[i][j];
		}
	}

	for (int i = 0; i < s1; i++)
	{
		delete[] row_s[i];
		delete[] out_s[i];
		delete[] row_z[i];
		delete[] out_z[i];
		delete[] row_m[i];
		delete[] out_m[i];
		delete[] row_e[i];
		delete[] out_e[i];
	}

	delete[] row_s;
	delete[] out_s;
	delete[] row_z;
	delete[] out_z;
	delete[] row_m;
	delete[] out_m;
	delete[] row_e;
	delete[] out_e;
}

void dotProduct_thread(
	int tid, int chunk, int colsize, int m_bits, int e_bits,
	uint8_t **Row1_s, uint8_t **Row1_z, uint64_t **Row1_m, uint64_t **Row1_e,
	uint8_t **Row2_s, uint8_t **Row2_z, uint64_t **Row2_m, uint64_t **Row2_e,
	uint8_t *row_s, uint8_t *row_z, uint64_t *row_m, uint64_t *row_e
	) {
	vector<FPArray> sums;
	for (int i = 0; i < chunk; i++)
	{
		FPArray mul1 = fpopArr[tid]->input(WHICHPARTY, colsize, Row1_s[i], Row1_z[i], Row1_m[i], Row1_e[i], m_bits, e_bits) ;
		FPArray mul2 = fpopArr[tid]->input(WHICHPARTY, colsize, Row2_s[i], Row2_z[i], Row2_m[i], Row2_e[i], m_bits, e_bits) ;
		sums.push_back(fpopArr[tid]->mul(mul1, mul2)) ;
	}

	FPArray vsum = fpopArr[tid]->treesum(sums);
	memcpy(row_s, vsum.s, chunk * sizeof(uint8_t));
	memcpy(row_z, vsum.z, chunk * sizeof(uint8_t));
	memcpy(row_m, vsum.m, chunk * sizeof(uint64_t));
	memcpy(row_e, vsum.e, chunk * sizeof(uint64_t));
}

void dotProduct2(int32_t s1, int32_t s2, 
	vector<vector<FPArray>> &arr1, vector<vector<FPArray>> &arr2, vector<FPArray> &outArr
	) {
	int m_bits, e_bits;
	m_bits = arr1[0][0].m_bits ;
	e_bits = arr1[0][0].e_bits ;

	uint8_t **Row1_s = new uint8_t*[s1] ;
	uint8_t **Row1_z = new uint8_t*[s1] ;
	uint64_t **Row1_m = new uint64_t*[s1] ;
	uint64_t **Row1_e = new uint64_t*[s1] ;

	uint8_t **Row2_s = new uint8_t*[s1] ;
	uint8_t **Row2_z = new uint8_t*[s1] ;
	uint64_t **Row2_m = new uint64_t*[s1] ;
	uint64_t **Row2_e = new uint64_t*[s1] ;

	uint8_t *row_s = new uint8_t[s1] ;
	uint8_t *row_z = new uint8_t[s1] ;
	uint64_t *row_m = new uint64_t[s1] ;
	uint64_t *row_e = new uint64_t[s1] ;

	for (int i = 0 ; i < s1 ; i++) {
		Row1_s[i] = new uint8_t[s2] ; Row2_s[i] = new uint8_t[s2] ;
		Row1_z[i] = new uint8_t[s2] ; Row2_z[i] = new uint8_t[s2] ;
		Row1_m[i] = new uint64_t[s2] ; Row2_m[i] = new uint64_t[s2] ;
		Row1_e[i] = new uint64_t[s2] ; Row2_e[i] = new uint64_t[s2] ;

		for (int j = 0 ; j < s2 ; j++) {
			Row1_s[i][j] = arr1[i][j].s[0] ; Row2_s[i][j] = arr2[i][j].s[0] ;
			Row1_z[i][j] = arr1[i][j].z[0] ; Row2_z[i][j] = arr2[i][j].z[0] ;
			Row1_m[i][j] = arr1[i][j].m[0] ; Row2_m[i][j] = arr2[i][j].m[0] ;
			Row1_e[i][j] = arr1[i][j].e[0] ; Row2_e[i][j] = arr2[i][j].e[0] ;
		}
	}

	vector<int> chunks = get_chunks(s1, __nt);
	thread threads[MAX_THREADS];
	int offset = 0;
	for (int i = 0; i < __nt; i++)
	{
		if (chunks[i] > 0)
		{
			threads[i] = thread(dotProduct_thread,
								i, chunks[i], s2, m_bits, e_bits,
								Row1_s + offset, Row1_z + offset, Row1_m + offset, Row1_e + offset,
								Row2_s + offset, Row2_z + offset, Row2_m + offset, Row2_e + offset,
								row_s + offset, row_z + offset, row_m + offset, row_e + offset);
			offset += chunks[i];
		}
	}

	for (int i = 0; i < __nt; i++)
		if (chunks[i] > 0)
			threads[i].join();

	for (int i = 0; i < s1; i++)
	{
		outArr[i].m_bits = m_bits;
		outArr[i].e_bits = e_bits;

		outArr[i].s[0] = row_s[i];
		outArr[i].z[0] = row_z[i];
		outArr[i].m[0] = row_m[i];
		outArr[i].e[0] = row_e[i];
	}


	for (int i = 0 ; i < s1 ; i++) {
		delete[] Row1_s[i] ; delete[] Row2_s[i] ;
		delete[] Row1_z[i] ; delete[] Row2_z[i] ;
		delete[] Row1_m[i] ; delete[] Row2_m[i] ;
		delete[] Row1_e[i] ; delete[] Row2_e[i] ;
	}

	delete[] Row1_s ; delete[] Row2_s ; delete[] row_s ;
	delete[] Row1_z ; delete[] Row2_z ; delete[] row_z ;
	delete[] Row1_m ; delete[] Row2_m ; delete[] row_m ;
	delete[] Row1_e ; delete[] Row2_e ; delete[] row_e ;
}

void vectorSum2(
	int32_t s1, int32_t s2, 
	vector<vector<FPArray>> &inArr, vector<FPArray> &outArr
	) {
	int m_bits, e_bits;
	m_bits = inArr[0][0].m_bits;
	e_bits = inArr[0][0].e_bits;

	uint8_t **Row_s = new uint8_t *[s1];
	uint8_t **Row_z = new uint8_t *[s1];
	uint64_t **Row_m = new uint64_t *[s1];
	uint64_t **Row_e = new uint64_t *[s1];

	uint8_t *row_s = new uint8_t[s1];
	uint8_t *row_z = new uint8_t[s1];
	uint64_t *row_m = new uint64_t[s1];
	uint64_t *row_e = new uint64_t[s1];

	for (int i = 0; i < s1; i++)
	{
		Row_s[i] = new uint8_t[s2];
		Row_z[i] = new uint8_t[s2];
		Row_m[i] = new uint64_t[s2];
		Row_e[i] = new uint64_t[s2];

		for (int j = 0; j < s2; j++)
		{
			Row_s[i][j] = inArr[j][i].s[0];
			Row_z[i][j] = inArr[j][i].z[0];
			Row_m[i][j] = inArr[j][i].m[0];
			Row_e[i][j] = inArr[j][i].e[0];
		}
	}

	vector<int> chunks = get_chunks(s1, __nt);
	thread threads[MAX_THREADS];
	int offset = 0;
	for (int i = 0; i < __nt; i++)
	{
		if (chunks[i] > 0)
		{
			threads[i] = thread(vectorSum_thread,
								i, chunks[i], s2, m_bits, e_bits,
								Row_s + offset, Row_z + offset, Row_m + offset, Row_e + offset,
								row_s + offset, row_z + offset, row_m + offset, row_e + offset);
			offset += chunks[i];
		}
	}

	for (int i = 0; i < __nt; i++)
		if (chunks[i] > 0)
			threads[i].join();

	for (int i = 0; i < s1; i++)
	{
		outArr[i].m_bits = m_bits;
		outArr[i].e_bits = e_bits;

		outArr[i].s[0] = row_s[i];
		outArr[i].z[0] = row_z[i];
		outArr[i].m[0] = row_m[i];
		outArr[i].e[0] = row_e[i];
	}

	for (int i = 0; i < s1; i++)
	{
		delete[] Row_s[i];
		delete[] Row_z[i];
		delete[] Row_m[i];
		delete[] Row_e[i];
	}

	delete[] Row_s;
	delete[] row_s;
	delete[] Row_z;
	delete[] row_z;
	delete[] Row_m;
	delete[] row_m;
	delete[] Row_e;
	delete[] row_e;
}

void getLoss(int32_t s, vector<FPArray> &arr, vector<FPArray> &outArr)
{
	int m_bits, e_bits;
	m_bits = arr[0].m_bits;
	e_bits = arr[0].e_bits;

	uint8_t *in_s = new uint8_t[s];
	uint8_t *in_z = new uint8_t[s];
	uint64_t *in_m = new uint64_t[s];
	uint64_t *in_e = new uint64_t[s];

	for (int i = 0; i < s; i++)
	{
		in_s[i] = arr[i].s[0];
		in_z[i] = arr[i].z[0];
		in_m[i] = arr[i].m[0];
		in_e[i] = arr[i].e[0];
	}

	vector<FPArray> sum;
	sum.push_back(__fp_op->input(__party, s, in_s, in_z, in_m, in_e, m_bits, e_bits));

	FPArray res = __fp_op->treesum(sum);
	FPArray div = __fp_op->input<float>(ALICE, 1, (float)-1.0 / s, m_bits, e_bits);
	res = __fp_op->mul(res, div);

	outArr[0].s[0] = res.s[0];
	outArr[0].z[0] = res.z[0];
	outArr[0].m[0] = res.m[0];
	outArr[0].e[0] = res.e[0];

	delete[] in_s;
	delete[] in_z;
	delete[] in_m;
	delete[] in_e;
}

void ConvBiasDer(
	int m, int W, int H, int chan, vector<vector<vector<vector<FPArray>>>> &der, vector<FPArray> &biasDer) {
	int m_bits, e_bits ;
	m_bits = der[0][0][0][0].m_bits ;
	e_bits = der[0][0][0][0].e_bits ;

	uint8_t **Row_s = new uint8_t*[chan] ;
	uint8_t **Row_z = new uint8_t*[chan] ;
	uint64_t **Row_m = new uint64_t*[chan] ;
	uint64_t **Row_e = new uint64_t*[chan] ;

	uint8_t *der_s = new uint8_t[chan] ;
	uint8_t *der_z = new uint8_t[chan] ;
	uint64_t *der_m = new uint64_t[chan] ;
	uint64_t *der_e = new uint64_t[chan] ;

	int sz = W*H*m ;
	for (int i = 0 ; i < chan ; i++) {
		Row_s[i] = new uint8_t[sz] ;
		Row_z[i] = new uint8_t[sz] ;
		Row_m[i] = new uint64_t[sz] ;
		Row_e[i] = new uint64_t[sz] ;
	}

	for (int i1 = 0 ; i1 < chan ; i1++) {
		for (int i2 = 0, k = 0 ; i2 < W ; i2++) {
			for (int i3 = 0 ; i3 < H ; i3++) {
				for (int i4 = 0 ; i4 < m ; i4++, k++) {
					Row_s[i1][k] = der[i4][i2][i3][i1].s[0] ;
					Row_z[i1][k] = der[i4][i2][i3][i1].z[0] ;
					Row_m[i1][k] = der[i4][i2][i3][i1].m[0] ;
					Row_e[i1][k] = der[i4][i2][i3][i1].e[0] ;
				}
			}
		}
	}

	// cout << "\tGonna chunk\n" ;

	vector<int> chunks = get_chunks(chan, __nt) ;
	thread threads[MAX_THREADS] ;
	int offset = 0 ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			threads[i] = thread(vectorSum_thread, 
				i, chunks[i], sz, m_bits, e_bits,
				Row_s+offset, Row_z+offset, Row_m+offset, Row_e+offset,
				der_s+offset, der_z+offset, der_m+offset, der_e+offset
			) ;
			offset += chunks[i] ;
		}
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;

	// cout << "\tJoined threads\n" ;

	for (int i = 0 ; i < chan ; i++) {
		biasDer[i].s[0] = der_s[i] ;
		biasDer[i].z[0] = der_z[i] ;
		biasDer[i].m[0] = der_m[i] ;
		biasDer[i].e[0] = der_e[i] ;
	}

	// cout << "\tCopied to vector<FPArray>\n" ;

	for (int i = 0 ; i < chan ; i++) {
		delete[] Row_s[i] ;
		delete[] Row_z[i] ;
		delete[] Row_m[i] ;
		delete[] Row_e[i] ;
	}

	delete[] Row_s ; delete[] der_s ;
	delete[] Row_z ; delete[] der_z ;
	delete[] Row_m ; delete[] der_m ;
	delete[] Row_e ; delete[] der_e ;
}

void batched_matrix_multiplication_thread(int tid, int chunk, int matsize, int m_bits, int e_bits,
	vector<FPMatrix> x_chunk, vector<FPMatrix> y_chunk,
	uint8_t** Row_s, uint8_t** Row_z, uint64_t** Row_m, uint64_t** Row_e
	) {

	vector<FPMatrix> res ;
	res = fpopArr[tid]->matrix_multiplication_secfloat(x_chunk, y_chunk, __chunk_exp) ;

	for (int i = 0 ; i < chunk ; i++) {
		memcpy(Row_s[i], res[i].s, matsize*sizeof(uint8_t)) ;
		memcpy(Row_z[i], res[i].z, matsize*sizeof(uint8_t)) ;
		memcpy(Row_m[i], res[i].m, matsize*sizeof(uint64_t)) ;
		memcpy(Row_e[i], res[i].e, matsize*sizeof(uint64_t)) ;
	}
}

vector<FPMatrix> batched_matrix_multiplication(vector<FPMatrix> &x, vector<FPMatrix> &y) {
	int m, n, p, L ;
	int m_bits, e_bits ;

	m = x[0].dim1 ;
	n = x[0].dim2 ;
	p = y[0].dim2 ;
	L = x.size() ;

	m_bits = x[0].m_bits ;
	e_bits = x[0].e_bits ;

	uint8_t** Row_s = new uint8_t*[L] ;
	uint8_t** Row_z = new uint8_t*[L] ;
	uint64_t** Row_m = new uint64_t*[L] ;
	uint64_t** Row_e = new uint64_t*[L] ;

	for (int i = 0 ; i < L ; i++) {
		Row_s[i] = new uint8_t[m*p] ;
		Row_z[i] = new uint8_t[m*p] ;
		Row_m[i] = new uint64_t[m*p] ;
		Row_e[i] = new uint64_t[m*p] ;
	}

	vector<int> chunks = get_chunks(L, __nt) ;
	thread threads[MAX_THREADS] ;
	int offset = 0 ;
	for (int i = 0 ; i < __nt ; i++) {
		if (chunks[i] > 0) {
			vector<FPMatrix> x_chunk = {x.begin()+offset, x.begin()+offset+chunks[i]} ;
			vector<FPMatrix> y_chunk = {y.begin()+offset, y.begin()+offset+chunks[i]} ;
			threads[i] = thread(batched_matrix_multiplication_thread,
				i, chunks[i], m*p, m_bits, e_bits,
				x_chunk, y_chunk,
				Row_s+offset, Row_z+offset, Row_m+offset, Row_e+offset
			) ;
		}
		offset += chunks[i] ;
	}

	for (int i = 0 ; i < __nt ; i++)
		if (chunks[i] > 0)
			threads[i].join() ;

	vector<FPMatrix> ret ;
	for (int i = 0 ; i < L ; i++) {
		FPMatrix ret_l = __fp_op->input(__party, m, p, Row_s[i], Row_z[i], Row_m[i], Row_e[i], m_bits, e_bits) ;
		ret.push_back(ret_l) ;

		delete[] Row_s[i] ;	
		delete[] Row_z[i] ;	
		delete[] Row_m[i] ;	
		delete[] Row_e[i] ;	
	}

	delete[] Row_s ;
	delete[] Row_z ;
	delete[] Row_m ;
	delete[] Row_e ;

	return ret ;
}