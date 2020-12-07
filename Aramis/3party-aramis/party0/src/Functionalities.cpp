/*
Authors: Mayank Rathee, Sameer Wagh, Nishant Kumar.
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

#pragma once
#include "Functionalities.h"
#include <algorithm>    // std::rotate
#include <thread>
using namespace std;

template<typename T>
vector<T> make_vector(size_t size) {
	return std::vector<T>(size);
}

	template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
	auto inner = make_vector<T>(sizes...);
	return vector<decltype(inner)>(first, inner);
}

// For sgx specific threaded parallelPC implementation
smallType *c_location, *share_m_location, *beta_location, *betaPrime_location;
aramisSecretType *r_location;
int start_parallelPC[NO_CORES];
int end_parallelPC[NO_CORES];
int dim_parallelPC[NO_CORES];
smallType* c1_deduceZero, *c2_deduceZero, *betaPrime_deduceZero;
int start_deduceZero[NO_CORES];
int end_deduceZero[NO_CORES];
int dim_deduceZero;

bool final_argmax = false;

extern vector<uint64_t*> toFreeMemoryLaterArr;

volatile int threads_done[NO_CORES] = {0};

/******************************** Functionalities 2PC ********************************/
// Share Truncation, truncate shares of a by power (in place) (power is logarithmic)
void funcTruncate2PC(vector<aramisSecretType> &a, 
		size_t power, 
		size_t size, 
		size_t party_1, 
		size_t party_2)
{
	assert((partyNum == party_1 or partyNum == party_2) && "Truncate called by spurious parties");

	if (partyNum == party_1)
		for (size_t i = 0; i < size; ++i)
			a[i] = static_cast<uint64_t>(static_cast<int64_t>(a[i]) >> power);

	if (partyNum == party_2)
		for (size_t i = 0; i < size; ++i)
			a[i] = - static_cast<uint64_t>(static_cast<int64_t>(- a[i]) >> power);
}

void funcTruncate2PC(vector<vector<aramisSecretType>> &a, 
		size_t power, 
		size_t rows, 
		size_t cols, 
		size_t party_1, 
		size_t party_2)
{
	assert((partyNum == party_1 or partyNum == party_2) && "Truncate called by spurious parties");

	if (partyNum == party_1){
		for(size_t i=0;i<rows;i++){
			for(size_t j=0;j<cols;j++){
				a[i][j] = static_cast<uint64_t>(static_cast<int64_t>(a[i][j]) >> power);
			}
		}
	}
	else if (partyNum == party_2){
		for(size_t i=0;i<rows;i++){
			for(size_t j=0;j<cols;j++){
				a[i][j] = - static_cast<uint64_t>(static_cast<int64_t>(- a[i][j]) >> power);
			}
		}
	}
}



// XOR shares with a public bit into output.
void funcXORModuloOdd2PC(vector<smallType> &bit, 
		vector<aramisSecretType> &shares, 
		vector<aramisSecretType> &output, 
		size_t size)
{
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (bit[i] == 1)
				output[i] = subtractModuloOdd<smallType, aramisSecretType>(1, shares[i]);
			else
				output[i] = shares[i];
		}
	}

	if (partyNum == PARTY_B)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (bit[i] == 1)
				output[i] = subtractModuloOdd<smallType, aramisSecretType>(0, shares[i]);
			else
				output[i] = shares[i];
		}
	}
}

void funcReconstruct2PC(const vector<aramisSecretType> &a, 
		size_t size, 
		string str, 
		vector<aramisSecretType>* b, 
		int revealToParties)
{
	assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Reconstruct called by spurious parties");
	assert((revealToParties >= 1) && (revealToParties <= 3) && ("Reconstruct/Reveal bitmask should be between 1 and 3 inclusive."));

	vector<aramisSecretType> temp(size);
	int partyToSend = PARTY_B;
	int partyToReceive = PARTY_A;
	if (revealToParties == 1)
	{
		//bitmask = 01
		partyToSend = PARTY_A;
		partyToReceive = PARTY_B;
	}

	if (partyNum == partyToSend)
	{
		sendVector<aramisSecretType>(a, partyToReceive, size);
		if (revealToParties == 3)
		{
			//bitmask = 11
			//both parties are supposed to get output - wait for reciver to send back results
			if (b)
			{
				receiveVector<aramisSecretType>((*b), partyToReceive, size);
			}
			else
			{
				receiveVector<aramisSecretType>(temp, partyToReceive, size);
			}
		}
	}

	if (partyNum == partyToReceive)
	{
		receiveVector<aramisSecretType>(temp, partyToSend, size);

		if (b)
		{
			addVectors<aramisSecretType>(temp, a, (*b), size);
			if (revealToParties == 3)
			{
				//bitmask = 11
				//send the reconstructed vector to the other party
				sendVector<aramisSecretType>(*b, partyToSend, size);
			}
		}
		else
		{
			addVectors<aramisSecretType>(temp, a, temp, size);
			if (revealToParties == 3)
			{
				//bitmask = 11
				//send the reconstructed vector to the other party
				sendVector<aramisSecretType>(temp, partyToSend, size);
			}
		}
	}
}


void funcReconstructBit2PC(const vector<smallType> &a, 
		size_t size, 
		string str)
{
	assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Reconstruct called by spurious parties");

	vector<smallType> temp(size);
	if (partyNum == PARTY_B)
		sendVector<smallType>(a, PARTY_A, size);

	if (partyNum == PARTY_A)
	{
		receiveVector<smallType>(temp, PARTY_B, size);
		XORVectors(temp, a, temp, size);

		print_string(str+": ");
		for (size_t i = 0; i < size; ++i)
			print_integer((int)temp[i]);

		print_string("");
	}
}


void funcConditionalSet2PC(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b, 
		vector<smallType> &c,
		vector<aramisSecretType> &u, 
		vector<aramisSecretType> &v, 
		size_t size)
{
	assert((partyNum == PARTY_C) && "ConditionalSet called by spurious parties");

	for (size_t i = 0; i < size; ++i)
	{
		if (c[i] == 0)
		{
			u[i] = a[i];
			v[i] = b[i];
		}
		else
		{
			u[i] = b[i];
			v[i] = a[i];
		}
	}
}


/******************************** Functionalities MPC ********************************/
void funcSecretShareConstant(const vector<aramisSecretType> &cons, 
		vector<aramisSecretType> &curShare, 
		size_t size)
{
	if (PRIMARY)
	{
		populateRandomVector<aramisSecretType>(curShare, size, "COMMON", "NEGATIVE");
		if (partyNum == PARTY_A)
		{
			addVectors<aramisSecretType>(curShare, cons, curShare, size);
		}
	}
	else
	{
		fillVector<aramisSecretType>(0, curShare, size);
	}
}

// Matrix Multiplication of a*b = c with transpose flags for a,b.
// Output is a share between PARTY_A and PARTY_B.
// a^transpose_a is rows*common_dim and b^transpose_b is common_dim*columns
void funcMatMulMPC(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		vector<aramisSecretType> &c,
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b, 
		bool areInputsScaled, 
		int coming_from_conv)
{

	int conv_comm = 0;
	size_t size = rows*columns;
	size_t size_left = rows*common_dim;
	size_t size_right = common_dim*columns;
	vector<aramisSecretType> A(size_left, 0), B(size_right, 0), C(size, 0);

	if (HELPER)
	{
		vector<aramisSecretType> A1(size_left, 0), A2(size_left, 0),
			B1(size_right, 0), B2(size_right, 0),
			C1(size, 0), C2(size, 0);

#ifdef PARALLEL_AES_ALL
		populateRandomVectorParallel(A1, size_left, "a_1", "POSITIVE");
		populateRandomVectorParallel(A2, size_left, "a_2", "POSITIVE");
		populateRandomVectorParallel(B1, size_right, "b_1", "POSITIVE");
		populateRandomVectorParallel(B2, size_right, "b_2", "POSITIVE");
		populateRandomVectorParallel(C1, size, "c_1", "POSITIVE");
#else
		populateRandomVector<aramisSecretType>(A1, size_left, "a_1", "POSITIVE");
		populateRandomVector<aramisSecretType>(A2, size_left, "a_2", "POSITIVE");
		populateRandomVector<aramisSecretType>(B1, size_right, "b_1", "POSITIVE");
		populateRandomVector<aramisSecretType>(B2, size_right, "b_2", "POSITIVE");
		populateRandomVector<aramisSecretType>(C1, size, "c_1", "POSITIVE");

#endif
		addVectors<aramisSecretType>(A1, A2, A, size_left);
		addVectors<aramisSecretType>(B1, B2, B, size_right);

		matrixMultEigen(A, B, C, rows, common_dim, columns, 0, 0);
		subtractVectors<aramisSecretType>(C, C1, C2, size);

		if(coming_from_conv)
			conv_comm += size;
		sendVector<aramisSecretType>(C2, PARTY_B, size);
	}

	if (PRIMARY)
	{
		vector<aramisSecretType> E(size_left), F(size_right);
		vector<aramisSecretType> temp_E(size_left), temp_F(size_right);
		vector<aramisSecretType> temp_c(size);

		if (partyNum == PARTY_A)
		{
#ifdef PARALLEL_AES_ALL
			populateRandomVectorParallel(A, size_left, "a_1", "POSITIVE");
			populateRandomVectorParallel(B, size_right, "b_1", "POSITIVE");
#else
			populateRandomVector<aramisSecretType>(A, size_left, "a_1", "POSITIVE");
			populateRandomVector<aramisSecretType>(B, size_right, "b_1", "POSITIVE");

#endif
		}

		if (partyNum == PARTY_B)
		{
#ifdef PARALLEL_AES_ALL
			populateRandomVectorParallel(A, size_left, "a_2", "POSITIVE");
			populateRandomVectorParallel(B, size_right, "b_2", "POSITIVE");
#else
			populateRandomVector<aramisSecretType>(A, size_left, "a_2", "POSITIVE");
			populateRandomVector<aramisSecretType>(B, size_right, "b_2", "POSITIVE");

#endif
		}

		subtractVectors<aramisSecretType>(a, A, E, size_left);
		subtractVectors<aramisSecretType>(b, B, F, size_right);

		if(coming_from_conv)
			conv_comm += (size_left+size_right);

		if (partyNum == PARTY_A)
			sendTwoVectors<aramisSecretType>(E, F, adversary(partyNum), size_left, size_right);
		else
			receiveTwoVectors<aramisSecretType>(temp_E, temp_F, adversary(partyNum), size_left, size_right);

		if (partyNum == PARTY_B)
			sendTwoVectors<aramisSecretType>(E, F, adversary(partyNum), size_left, size_right);
		else
			receiveTwoVectors<aramisSecretType>(temp_E, temp_F, adversary(partyNum), size_left, size_right);

		addVectors<aramisSecretType>(E, temp_E, E, size_left);
		addVectors<aramisSecretType>(F, temp_F, F, size_right);

		if (partyNum == PARTY_A)
		{
			subtractVectors<aramisSecretType>(a, E, A, size_left);
			matrixMultEigen(A, F, c, rows, common_dim, columns, 0, 0);
			matrixMultEigen(E, b, temp_c, rows, common_dim, columns, 0, 0);
		}
		else
		{
			matrixMultEigen(a, F, c, rows, common_dim, columns, 0, 0);
			matrixMultEigen(E, b, temp_c, rows, common_dim, columns, 0, 0);
		}

		addVectors<aramisSecretType>(c, temp_c, c, size);
		if (partyNum == PARTY_A)
		{
#ifdef PARALLEL_AES_ALL
			populateRandomVectorParallel(C, size, "c_1", "POSITIVE");
#else
			populateRandomVector<aramisSecretType>(C, size, "c_1", "POSITIVE");
#endif
		}
		else if (partyNum == PARTY_B)
		{
			//Receive C1 from P2 after E and F have been revealed.
			receiveVector<aramisSecretType>(C, PARTY_C, size);
		}

		addVectors<aramisSecretType>(c, C, c, size);

		if (areInputsScaled)
			funcTruncate2PC(c, FLOAT_PRECISION, size, PARTY_A, PARTY_B);
	}

}


void funcDotProductMPC(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b,
		vector<aramisSecretType> &c, 
		size_t size, 
		bool bench_cross_bridge_calls)
{


	vector<aramisSecretType> A(size, 0), B(size, 0), C(size, 0);
	if (HELPER)
	{
		vector<aramisSecretType> A1(size, 0), A2(size, 0),
			B1(size, 0), B2(size, 0),
			C1(size, 0), C2(size, 0);

#ifdef PARALLEL_AES_ALL
		populateRandomVectorParallel(A1, size, "a_1", "POSITIVE");
		populateRandomVectorParallel(A2, size, "a_2", "POSITIVE");
		populateRandomVectorParallel(B1, size, "b_1", "POSITIVE");
		populateRandomVectorParallel(B2, size, "b_2", "POSITIVE");
		populateRandomVectorParallel(C1, size, "c_1", "POSITIVE");
#else
		populateRandomVector<aramisSecretType>(A1, size, "a_1", "POSITIVE");
		populateRandomVector<aramisSecretType>(A2, size, "a_2", "POSITIVE");
		populateRandomVector<aramisSecretType>(B1, size, "b_1", "POSITIVE");
		populateRandomVector<aramisSecretType>(B2, size, "b_2", "POSITIVE");
		populateRandomVector<aramisSecretType>(C1, size, "c_1", "POSITIVE");

#endif
		addVectors<aramisSecretType>(A1, A2, A, size);
		addVectors<aramisSecretType>(B1, B2, B, size);

		for (size_t i = 0; i < size; ++i)
			C[i] = A[i] * B[i];

		// splitIntoShares(C, C1, C2, size);
		subtractVectors<aramisSecretType>(C, C1, C2, size);
		sendVector<aramisSecretType>(C2, PARTY_B, size);

	}

	if (PRIMARY)
	{
		if (partyNum == PARTY_A)
		{
#ifdef PARALLEL_AES_ALL
			populateRandomVectorParallel(A, size, "a_1", "POSITIVE");
			populateRandomVectorParallel(B, size, "b_1", "POSITIVE");
#else
			populateRandomVector<aramisSecretType>(A, size, "a_1", "POSITIVE");
			populateRandomVector<aramisSecretType>(B, size, "b_1", "POSITIVE");

#endif
		}

		if (partyNum == PARTY_B)
		{
#ifdef PARALLEL_AES_ALL
			populateRandomVectorParallel(A, size, "a_2", "POSITIVE");
			populateRandomVectorParallel(B, size, "b_2", "POSITIVE");
#else
			populateRandomVector<aramisSecretType>(A, size, "a_2", "POSITIVE");
			populateRandomVector<aramisSecretType>(B, size, "b_2", "POSITIVE");
#endif
		}

		vector<aramisSecretType> E(size), F(size), temp_E(size), temp_F(size);
		aramisSecretType temp;

		subtractVectors<aramisSecretType>(a, A, E, size);
		subtractVectors<aramisSecretType>(b, B, F, size);

		if (partyNum == PARTY_A)
			sendTwoVectors<aramisSecretType>(E, F, adversary(partyNum), size, size /*bench_cross_bridge_calls*/);
		else
			receiveTwoVectors<aramisSecretType>(temp_E, temp_F, adversary(partyNum), size, size);

		if (partyNum == PARTY_B)
			sendTwoVectors<aramisSecretType>(E, F, adversary(partyNum), size, size /*bench_cross_bridge_calls*/);
		else
			receiveTwoVectors<aramisSecretType>(temp_E, temp_F, adversary(partyNum), size, size);

		addVectors<aramisSecretType>(E, temp_E, E, size);
		addVectors<aramisSecretType>(F, temp_F, F, size);

		for (size_t i = 0; i < size; ++i)
		{

			if (partyNum == PARTY_A)
			{
				c[i] = (a[i]-E[i]) * F[i];
			}
			else
			{
				c[i] = a[i] * F[i];
			}

			temp = E[i] * b[i];
			c[i] = c[i] + temp;

		}

		if (partyNum == PARTY_A){
#ifdef PARALLEL_AES_ALL
			populateRandomVectorParallel(C, size, "c_1", "POSITIVE");
#else
			populateRandomVector<aramisSecretType>(C, size, "c_1", "POSITIVE");
#endif
		}
		else if (partyNum == PARTY_B)
			receiveVector<aramisSecretType>(C, PARTY_C, size);

		addVectors<aramisSecretType>(c, C, c, size);
		funcTruncate2PC(c, FLOAT_PRECISION, size, PARTY_A, PARTY_B);
	}
}



void parallelPC(smallType* c, 
		size_t start, 
		size_t end, 
		int t,
		const smallType* share_m, 
		const aramisSecretType* r,
		const smallType* beta, 
		const smallType* betaPrime, 
		size_t dim)
{
	size_t index3, index2;
	size_t PARTY;

	smallType bit_r, a, tempM;
	aramisSecretType valueX;
	thread_local int shuffle_counter = 0;
	thread_local int nonZero_counter = 0;

	for (size_t index2 = start; index2 < end; ++index2)
	{
		if (beta[index2] == 1 and r[index2] != MINUS_ONE)
			valueX = r[index2] + 1;
		else
			valueX = r[index2];

		if (beta[index2] == 1 and r[index2] == MINUS_ONE)
		{
			//One share of zero and other shares of 1
			//assert(false);
			for (size_t k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				c[index3] = aes_parallel->randModPrime(t, nonZero_counter);
				if (partyNum == PARTY_A)
					c[index3] = subtractModPrime((k!=0), c[index3]);

				c[index3] = multiplyModPrime(c[index3], aes_parallel->randNonZeroModPrime(t, nonZero_counter));
			}
		}
		else
		{
			//Single for loop
			a = 0;
			for (size_t k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				c[index3] = a;
				tempM = share_m[index3];

				bit_r = (smallType)((valueX >> (63-k)) & 1);

				if (bit_r == 0)
					a = addModPrime(a, tempM);
				else
					a = addModPrime(a, subtractModPrime((partyNum == PARTY_A), tempM));

				if (!beta[index2])
				{
					if (partyNum == PARTY_A)
						c[index3] = addModPrime(c[index3], 1+bit_r);
					c[index3] = subtractModPrime(c[index3], tempM);
				}
				else
				{
					if (partyNum == PARTY_A)
						c[index3] = addModPrime(c[index3], 1-bit_r);
					c[index3] = addModPrime(c[index3], tempM);
				}

				c[index3] = multiplyModPrime(c[index3], aes_parallel->randNonZeroModPrime(t, nonZero_counter));
			}
		}
		aes_parallel->AES_random_shuffle(c, index2*dim, (index2+1)*dim, t, shuffle_counter);
	}
	aes_parallel->counterIncrement();
}

void deduceIfAnyZeroPC(smallType* p0Data, 
		smallType* p1Data,
		size_t start, 
		size_t end, 
		size_t dim,
		smallType* betaPrime)
{
	for(size_t i=start; i<end; i++)
	{
		betaPrime[i] = 0;
		for(size_t j=0; j<dim; j++)
		{
			size_t curIdx = i*dim + j;
			if (addModPrime(p0Data[curIdx], p1Data[curIdx]) == 0)
			{
				betaPrime[i] = 1;
				break;
			}
		}
	}
}

void parallelPC_sgx_threaded(int worker_thread_num){
	parallelPC(c_location, start_parallelPC[worker_thread_num],
			end_parallelPC[worker_thread_num], worker_thread_num,
			(const smallType*)share_m_location, (const aramisSecretType*)r_location,
			(const smallType*)beta_location, (const smallType*)betaPrime_location,
			dim_parallelPC[worker_thread_num]);
}

void deduceZeroThreadDispatcher(int thread_id){
	deduceIfAnyZeroPC(c1_deduceZero, c2_deduceZero, start_deduceZero[thread_id],
			end_deduceZero[thread_id], dim_deduceZero, betaPrime_deduceZero);
}

// Private Compare functionality
void funcPrivateCompareMPC(vector<smallType> &share_m, 
		vector<aramisSecretType> &r,
		vector<smallType> &beta, 
		vector<smallType> &betaPrime,
		size_t size, 
		size_t dim)
{

	assert(dim == BIT_SIZE && "Private Compare assert issue");
	size_t sizeLong = size*dim;
	size_t index3, index2;
	size_t PARTY;

	PARTY = PARTY_C;


	if (PRIMARY)
	{
		smallType bit_r, a, tempM;
		vector<smallType> c(sizeLong);
		aramisSecretType valueX;

		if (PARALLEL)
		{
#ifndef PARALLELPC
			parallelPC(c.data(), 0, size, 1, share_m.data(), r.data(), beta.data(), betaPrime.data(), dim);
#else
			c_location = c.data();
			share_m_location = share_m.data();
			r_location = r.data();
			beta_location = beta.data();
			betaPrime_location = betaPrime.data();
			//thread *threads = new thread[NO_CORES];
			int chunksize = size/NO_CORES;

			for (int i = 0; i < NO_CORES; i++)
			{
				int start = i*chunksize;
				int end = (i+1)*chunksize;
				if (i == NO_CORES - 1)
					end = size;
				start_parallelPC[i] = start;
				end_parallelPC[i] = end;
				dim_parallelPC[i] = dim;

				//parallelPC_sgx_threaded.
				ocall_parallelPC_spawn_threads(i);

			}

			ocall_parallelPC_join_threads();
#endif
		}
		else
		{
			//Check the security of the first if condition
			for (size_t index2 = 0; index2 < size; ++index2)
			{
				if (beta[index2] == 1 and r[index2] != MINUS_ONE)
					valueX = r[index2] + 1;
				else
					valueX = r[index2];

				if (beta[index2] == 1 and r[index2] == MINUS_ONE)
				{
					//One share of zero and other shares of 1
					//Then multiply and shuffle
					for (size_t k = 0; k < dim; ++k)
					{
						index3 = index2*dim + k;
						c[index3] = aes_common->randModPrime();
						if (partyNum == PARTY_A)
							c[index3] = subtractModPrime((k!=0), c[index3]);

						c[index3] = multiplyModPrime(c[index3], aes_common->randNonZeroModPrime());
					}
				}
				else
				{
					//Single for loop
					a = 0;
					for (size_t k = 0; k < dim; ++k)
					{
						index3 = index2*dim + k;
						c[index3] = a;
						tempM = share_m[index3];

						bit_r = (smallType)((valueX >> (63-k)) & 1);

						if (bit_r == 0)
							a = addModPrime(a, tempM);
						else
							a = addModPrime(a, subtractModPrime((partyNum == PARTY_A), tempM));

						if (!beta[index2])
						{
							if (partyNum == PARTY_A)
								c[index3] = addModPrime(c[index3], 1+bit_r);
							c[index3] = subtractModPrime(c[index3], tempM);
						}
						else
						{
							if (partyNum == PARTY_A)
								c[index3] = addModPrime(c[index3], 1-bit_r);
							c[index3] = addModPrime(c[index3], tempM);
						}

						c[index3] = multiplyModPrime(c[index3], aes_common->randNonZeroModPrime());
					}
				}
				aes_common->AES_random_shuffle(c, index2*dim, (index2+1)*dim);
			}
		}
		sendVector<smallType>(c, PARTY, sizeLong);
	}

	if (partyNum == PARTY)
	{
		vector<smallType> c1(sizeLong);
		vector<smallType> c2(sizeLong);

		receiveVector<smallType>(c1, PARTY_A, sizeLong);
		receiveVector<smallType>(c2, PARTY_B, sizeLong);

#ifdef PARALLELIZE_CRITICAL
		int chunksize = size/NO_CORES;
		c1_deduceZero = c1.data();
		c2_deduceZero = c2.data();
		betaPrime_deduceZero = betaPrime.data();
		dim_deduceZero = dim;
		for (int i = 0; i < NO_CORES; i++)
		{
			int start = i*chunksize;
			int end = (i+1)*chunksize;
			if (i == NO_CORES - 1)
				end = size;
			start_deduceZero[i] = start;
			end_deduceZero[i] = end;
			ocall_deduce_zero_spawn_threads(i);
		}
		ocall_join_threads();

#else
		for (size_t index2 = 0; index2 < size; ++index2)
		{
			betaPrime[index2] = 0;
			for (int k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				if (addModPrime(c1[index3], c2[index3]) == 0)
				{
					betaPrime[index2] = 1;
					break;
				}
			}
		}
#endif
	}
}

// Convert shares of a in \Z_L to shares in \Z_{L-1} (in place)
// a \neq L-1
void funcShareConvertMPC(vector<aramisSecretType> &a, 
		size_t size)
{

	vector<aramisSecretType> r;
	vector<smallType> etaDP;
	vector<smallType> alpha;
	vector<smallType> betai;
	vector<smallType> bit_shares;
	vector<aramisSecretType> delta_shares;
	vector<smallType> etaP;
	vector<aramisSecretType> eta_shares;
	vector<aramisSecretType> theta_shares;
	size_t PARTY;

	PARTY = PARTY_C;

	if (PRIMARY)
	{
		r.resize(size);
		etaDP.resize(size);
		alpha.resize(size);
		betai.resize(size);
		bit_shares.resize(size*BIT_SIZE);
		delta_shares.resize(size);
		eta_shares.resize(size);
		theta_shares.resize(size);
	}
	else if (HELPER)
	{
		etaP.resize(size);
	}

	PARTY = PARTY_C;

	if (PRIMARY)
	{
		vector<aramisSecretType> r1(size);
		vector<aramisSecretType> r2(size);
		vector<aramisSecretType> a_tilde(size);

#ifdef PARALLEL_AES_ALL
		populateRandomVectorParallel(r1, size, "COMMON", "POSITIVE");
		populateRandomVectorParallel(r2, size, "COMMON", "POSITIVE");
#else
		populateRandomVector<aramisSecretType>(r1, size, "COMMON", "POSITIVE");
		populateRandomVector<aramisSecretType>(r2, size, "COMMON", "POSITIVE");
#endif
		addVectors<aramisSecretType>(r1, r2, r, size);

		if (partyNum == PARTY_A)
			wrapAround(r1, r2, alpha, size);

		if (partyNum == PARTY_A)
		{
			addVectors<aramisSecretType>(a, r1, a_tilde, size);
			wrapAround(a, r1, betai, size);
		}
		if (partyNum == PARTY_B)
		{
			addVectors<aramisSecretType>(a, r2, a_tilde, size);
			wrapAround(a, r2, betai, size);
		}

		populateBitsVector(etaDP, "COMMON", size);
		//return;
		sendVector<aramisSecretType>(a_tilde, PARTY_C, size);
	}
	// Change Mayank
#ifndef RUN_SHARECONV_OPTI
	if (partyNum == PARTY_C)
	{
		vector<aramisSecretType> x(size);
		vector<smallType> delta(size);
		vector<aramisSecretType> a_tilde_1(size);
		vector<aramisSecretType> a_tilde_2(size);
		vector<smallType> bit_shares_x_1(size*BIT_SIZE);
		vector<smallType> bit_shares_x_2(size*BIT_SIZE);
		vector<aramisSecretType> delta_shares_1(size);
		vector<aramisSecretType> delta_shares_2(size);

		receiveVector<aramisSecretType>(a_tilde_1, PARTY_A, size);
		receiveVector<aramisSecretType>(a_tilde_2, PARTY_B, size);

		addVectors<aramisSecretType>(a_tilde_1, a_tilde_2, x, size);
#ifdef PARALLEL_AES
		sharesOfBitsParallel(bit_shares_x_1, bit_shares_x_2, x, size, "INDEP");
#else
		sharesOfBits(bit_shares_x_1, bit_shares_x_2, x, size, "INDEP");
#endif
		sendVector<smallType>(bit_shares_x_1, PARTY_A, size*BIT_SIZE);
		sendVector<smallType>(bit_shares_x_2, PARTY_B, size*BIT_SIZE);
		wrapAround(a_tilde_1, a_tilde_2, delta, size);
#ifdef PARALLEL_AES_ALL
		sharesModuloOddParallel(delta_shares_1, delta_shares_2, delta, size, "INDEP");
#else
		sharesModuloOdd(delta_shares_1, delta_shares_2, delta, size, "INDEP");
#endif
		sendVector<aramisSecretType>(delta_shares_1, PARTY_A, size);
		sendVector<aramisSecretType>(delta_shares_2, PARTY_B, size);
	}

	else if (PRIMARY)
	{
		receiveVector<smallType>(bit_shares, PARTY_C, size*BIT_SIZE);
		receiveVector<aramisSecretType>(delta_shares, PARTY_C, size);
	}

#else
	if (partyNum == PARTY_C)
	{
		vector<aramisSecretType> x(size);
		vector<smallType> delta(size);
		vector<aramisSecretType> a_tilde_1(size);
		vector<aramisSecretType> a_tilde_2(size);
		vector<smallType> bit_shares_x_1(size*BIT_SIZE);
		vector<smallType> bit_shares_x_2(size*BIT_SIZE);
		vector<aramisSecretType> delta_shares_1(size);
		vector<aramisSecretType> delta_shares_2(size);

		receiveVector<aramisSecretType>(a_tilde_1, PARTY_A, size);
		receiveVector<aramisSecretType>(a_tilde_2, PARTY_B, size);
		addVectors<aramisSecretType>(a_tilde_1, a_tilde_2, x, size);
		wrapAround(a_tilde_1, a_tilde_2, delta, size);
#ifdef PARALLEL_AES
		sharesOfBitsParallel(bit_shares_x_1, bit_shares_x_2, x, size, "SHARE_CONV_OPTI");
#else
		sharesOfBits(bit_shares_x_1, bit_shares_x_2, x, size, "SHARE_CONV_OPTI");
#endif

		//Send first half x2 to B and second half x1 to A.
		sendArr<smallType>(bit_shares_x_1.data() + (size/2) , PARTY_A, (size - (size/2))*BIT_SIZE);
		sendArr<smallType>(bit_shares_x_2.data(), PARTY_B, (size/2)*BIT_SIZE);

		wrapAround(a_tilde_1, a_tilde_2, delta, size);
		sharesModuloOdd(delta_shares_1, delta_shares_2, delta, size, "SHARE_CONV_OPTI");

		sendArr<aramisSecretType>(delta_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		sendArr<aramisSecretType>(delta_shares_2.data(), PARTY_B, (size/2));

	}

	else if (PRIMARY)
	{
		size_t localStart, localEnd, receiveStart, receiveEnd; //start - inclusive, end - exclusive
		AESObject* aesObjectForBitShares;
		AESObject* aesObjectForDeltaShares;
		if (partyNum == PARTY_A)
		{
			localStart = 0;
			localEnd = size/2;
			receiveStart = size/2;
			receiveEnd = size;
			aesObjectForBitShares = aes_share_conv_bit_shares_p0_p2;
			aesObjectForDeltaShares = aes_share_conv_shares_mod_odd_p0_p2;
		}
		else if (partyNum == PARTY_B)
		{
			receiveStart = 0;
			receiveEnd = size/2;
			localStart = size/2;
			localEnd = size;
			aesObjectForBitShares = aes_share_conv_bit_shares_p1_p2;
			aesObjectForDeltaShares = aes_share_conv_shares_mod_odd_p1_p2;
		}

		//First do all bit computation and then wait for P2 to get remaining.
		//Call parallel version of this.
#ifdef PARALLEL_AES
		if(partyNum == PARTY_A)
			sharesOfBitsPrimaryParallel(bit_shares.data() + localStart*BIT_SIZE, (localEnd-localStart)*BIT_SIZE, 0);
		else
			sharesOfBitsPrimaryParallel(bit_shares.data() + localStart*BIT_SIZE, (localEnd-localStart)*BIT_SIZE, 1);
#else
		for(size_t i=localStart; i<localEnd; i++)
		{
			for(size_t j=0; j<BIT_SIZE; j++)
			{
				bit_shares[i*BIT_SIZE + j] = aesObjectForBitShares->randModPrime();
			}
		}
#endif
		for(size_t i=localStart; i<localEnd; i++)
		{
			delta_shares[i] = aesObjectForDeltaShares->randModuloOdd();
		}

		//Now get the remaining bits from P2.
		receiveArr<smallType>(bit_shares.data() + receiveStart, PARTY_C, (receiveEnd - receiveStart)*BIT_SIZE);
		receiveArr<aramisSecretType>(delta_shares.data() + receiveStart, PARTY_C, (receiveEnd - receiveStart));
	}

#endif //end of share convert opti
	if (PRIMARY)
	{
		for(size_t i=0;i<size;i++)
		{
			r[i] = r[i] - 1;
		}
	}

	funcPrivateCompareMPC(bit_shares, r, etaDP, etaP, size, BIT_SIZE);
	//184 MB here
	if (partyNum == PARTY)
	{
		vector<aramisSecretType> eta_shares_1(size);
		vector<aramisSecretType> eta_shares_2(size);

		for (size_t i = 0; i < size; ++i)
			etaP[i] = 1 - etaP[i];
#ifdef PARALLEL_AES_ALL
		sharesModuloOddParallel(eta_shares_1, eta_shares_2, etaP, size, "INDEP");
#else
		sharesModuloOdd(eta_shares_1, eta_shares_2, etaP, size, "INDEP");
#endif
		sendVector<aramisSecretType>(eta_shares_1, PARTY_A, size);
		sendVector<aramisSecretType>(eta_shares_2, PARTY_B, size);
	}

	if (PRIMARY)
	{
		receiveVector<aramisSecretType>(eta_shares, PARTY, size);
		funcXORModuloOdd2PC(etaDP, eta_shares, theta_shares, size);
		addModuloOdd<aramisSecretType, smallType>(theta_shares, betai, theta_shares, size);
		subtractModuloOdd<aramisSecretType, aramisSecretType>(theta_shares, delta_shares, theta_shares, size);

		if (partyNum == PARTY_A)
			subtractModuloOdd<aramisSecretType, smallType>(theta_shares, alpha, theta_shares, size);

		subtractModuloOdd<aramisSecretType, aramisSecretType>(a, theta_shares, a, size);
	}
}


//Compute MSB of a and store it in b
//3PC: output is shares of MSB in \Z_L
void funcComputeMSB3PC(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		size_t size)
{
	assert(THREE_PC && "funcComputeMSB3PC called in non-3PC mode");

	vector<aramisSecretType> ri;
	vector<smallType> bit_shares;
	vector<aramisSecretType> LSB_shares;
	vector<smallType> beta;
	vector<aramisSecretType> c;
	vector<smallType> betaP;
	vector<aramisSecretType> theta_shares;

	if (PRIMARY)
	{
		ri.resize(size);
		bit_shares.resize(size * BIT_SIZE);
		LSB_shares.resize(size);
		beta.resize(size);
		c.resize(size);
		theta_shares.resize(size);
	}
	else if (HELPER)
	{
		betaP.resize(size);
	}

#ifndef RUN_MSB_OPTI
	if (partyNum == PARTY_C)
	{
		vector<aramisSecretType> r1(size);
		vector<aramisSecretType> r2(size);
		vector<aramisSecretType> r(size);
		vector<smallType> bit_shares_r_1(size*BIT_SIZE);
		vector<smallType> bit_shares_r_2(size*BIT_SIZE);
		vector<aramisSecretType> LSB_shares_1(size);
		vector<aramisSecretType> LSB_shares_2(size);

		for (size_t i = 0; i < size; ++i)
		{
			r1[i] = aes_indep->randModuloOdd();
		}
		for (size_t i = 0; i < size; ++i)
		{
			r2[i] = aes_indep->randModuloOdd();
		}

		addModuloOdd<aramisSecretType, aramisSecretType>(r1, r2, r, size);
#ifdef PARALLEL_AES
		sharesOfBitsParallel(bit_shares_r_1, bit_shares_r_2, r, size, "INDEP");
#else
		sharesOfBits(bit_shares_r_1, bit_shares_r_2, r, size, "INDEP");
#endif
		sendVector<smallType>(bit_shares_r_1, PARTY_A, size*BIT_SIZE);
		sendVector<smallType>(bit_shares_r_2, PARTY_B, size*BIT_SIZE);

		sharesOfLSB(LSB_shares_1, LSB_shares_2, r, size, "INDEP");
		sendTwoVectors<aramisSecretType>(r1, LSB_shares_1, PARTY_A, size, size);
		sendTwoVectors<aramisSecretType>(r2, LSB_shares_2, PARTY_B, size, size);
	}

	else if (PRIMARY)
	{
		receiveVector<smallType>(bit_shares, PARTY_C, size*BIT_SIZE);
		receiveTwoVectors<aramisSecretType>(ri, LSB_shares, PARTY_C, size, size);

	}
	//Optimization
#else

	if (partyNum == PARTY_C)
	{
		vector<aramisSecretType> r1(size);
		vector<aramisSecretType> r2(size);
		vector<aramisSecretType> r(size);
		vector<smallType> bit_shares_r_1(size*BIT_SIZE);
		vector<smallType> bit_shares_r_2(size*BIT_SIZE);
		vector<aramisSecretType> LSB_shares_1(size);
		vector<aramisSecretType> LSB_shares_2(size);

		for (size_t i = 0; i < size; ++i)
		{
			r1[i] = aes_share_conv_shares_mod_odd_p0_p2->randModuloOdd();
		}
		for (size_t i = 0; i < size; ++i)
		{
			r2[i] = aes_share_conv_shares_mod_odd_p1_p2->randModuloOdd();
		}
		// Now r vector is not even required to be sent.

		addModuloOdd<aramisSecretType, aramisSecretType>(r1, r2, r, size);
#ifdef PARALLEL_AES
		if(size <= 1)
			sharesOfBits(bit_shares_r_1, bit_shares_r_2, r, size, "SHARE_CONV_OPTI");
		else
			sharesOfBitsParallel(bit_shares_r_1, bit_shares_r_2, r, size, "SHARE_CONV_OPTI");
#else
		sharesOfBits(bit_shares_r_1, bit_shares_r_2, r, size, "SHARE_CONV_OPTI");
#endif

		sendArr<smallType>(bit_shares_r_1.data() + (size/2)*BIT_SIZE, PARTY_A, (size - (size/2))*BIT_SIZE);
		sendArr<smallType>(bit_shares_r_2.data(), PARTY_B, (size/2)*BIT_SIZE);


		sharesOfLSB(LSB_shares_1, LSB_shares_2, r, size, "MSB_OPTI");

		sendArr<aramisSecretType>(LSB_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		sendArr<aramisSecretType>(LSB_shares_2.data(), PARTY_B, (size/2));

	}
	else if (PRIMARY)
	{
		size_t localStart, localEnd, receiveStart, receiveEnd; //start - inclusive, end - exclusive
		AESObject* aesObjectForRi;
		AESObject* aesObjectForBitShares;
		AESObject* aesObjectForLSBShares;
		if (partyNum == PARTY_A)
		{
			localStart = 0;
			localEnd = size/2;
			receiveStart = size/2;
			receiveEnd = size;
			aesObjectForRi = aes_share_conv_shares_mod_odd_p0_p2;
			aesObjectForBitShares = aes_share_conv_bit_shares_p0_p2;
			aesObjectForLSBShares = aes_comp_msb_shares_lsb_p0_p2;
		}
		else
		{
			localStart = size/2;
			localEnd = size;
			receiveStart = 0;
			receiveEnd = size/2;
			aesObjectForRi = aes_share_conv_shares_mod_odd_p1_p2;
			aesObjectForBitShares = aes_share_conv_bit_shares_p1_p2;
			aesObjectForLSBShares = aes_comp_msb_shares_lsb_p1_p2;
		}

		//First compute ri shares locally.
		for (size_t i = 0; i < size; ++i)
		{
			ri[i] = aesObjectForRi->randModuloOdd();
		}

		//Also fill what you can fill locally for bit_shares and LSB_shares.
		//Then wait on P2 to get the other half.
#ifdef PARALLEL_AES
		//Call parallel version of this.
		if(size <= 1){
			for(size_t i=localStart; i<localEnd; i++)
			{
				for(size_t j=0; j<BIT_SIZE; j++)
				{
					bit_shares[i*BIT_SIZE + j] = aesObjectForBitShares->randModPrime();
				}
			}

		}
		else{
			if(partyNum == PARTY_A)
				sharesOfBitsPrimaryParallel(bit_shares.data() + localStart*BIT_SIZE, (localEnd-localStart)*BIT_SIZE, 0);
			else
				sharesOfBitsPrimaryParallel(bit_shares.data() + localStart*BIT_SIZE, (localEnd-localStart)*BIT_SIZE, 1);
		}
#else
		for(size_t i=localStart; i<localEnd; i++)
		{
			for(size_t j=0; j<BIT_SIZE; j++)
			{
				bit_shares[i*BIT_SIZE + j] = aesObjectForBitShares->randModPrime();
			}
		}
#endif
		for(size_t i=localStart; i<localEnd; i++)
		{
			LSB_shares[i] = aesObjectForLSBShares->get64Bits();
		}

		//Now that all local computation is done, wait on p2 to get the remaining half
		receiveArr<smallType>(bit_shares.data() + receiveStart*BIT_SIZE, PARTY_C, (receiveEnd - receiveStart)*BIT_SIZE);
		receiveArr<aramisSecretType>(LSB_shares.data() + receiveStart, PARTY_C, receiveEnd - receiveStart);
	}

#endif
	if (PRIMARY)
	{
		vector<aramisSecretType> temp(size);
		addModuloOdd<aramisSecretType, aramisSecretType>(a, a, c, size);
		addModuloOdd<aramisSecretType, aramisSecretType>(c, ri, c, size);

		if (partyNum == PARTY_A){
			sendVector<aramisSecretType>(ref(c), adversary(partyNum), size);
			receiveVector<aramisSecretType>(ref(temp), adversary(partyNum), size);
		}
		else if (partyNum == PARTY_B){
			receiveVector<aramisSecretType>(ref(temp), adversary(partyNum), size);
			sendVector<aramisSecretType>(ref(c), adversary(partyNum), size);
		}
		addModuloOdd<aramisSecretType, aramisSecretType>(c, temp, c, size);
		populateBitsVector(beta, "COMMON", size);
	}

	funcPrivateCompareMPC(bit_shares, c, beta, betaP, size, BIT_SIZE);

#ifndef RUN_MSB_OPTI
	if (partyNum == PARTY_C)
	{
		vector<aramisSecretType> theta_shares_1(size);
		vector<aramisSecretType> theta_shares_2(size);

		sharesOfBitVector(theta_shares_1, theta_shares_2, betaP, size, "INDEP");
		sendVector<aramisSecretType>(theta_shares_1, PARTY_A, size);
		sendVector<aramisSecretType>(theta_shares_2, PARTY_B, size);
	}
	else if (PRIMARY)
	{
		if(partyNum == PARTY_A)
		{
			receiveVector<aramisSecretType>(theta_shares, PARTY_C, size);
		}
		else if(partyNum == PARTY_B)
		{
			receiveVector<aramisSecretType>(theta_shares, PARTY_C, size);
		}
	}

#else
	if (partyNum == PARTY_C)
	{
		vector<aramisSecretType> theta_shares_1(size);
		vector<aramisSecretType> theta_shares_2(size);

		sharesOfBitVector(theta_shares_1, theta_shares_2, betaP, size, "MSB_OPTI");

		sendArr<aramisSecretType>(theta_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		sendArr<aramisSecretType>(theta_shares_2.data(), PARTY_B, (size/2));

	}
	else if (PRIMARY)
	{
		size_t localStart, localEnd, receiveStart, receiveEnd; //start - inclusive, end - exclusive
		AESObject* aesObjectForBitShares;
		if (partyNum == PARTY_A)
		{
			localStart = 0;
			localEnd = size/2;
			receiveStart = size/2;
			receiveEnd = size;
			aesObjectForBitShares = aes_comp_msb_shares_bit_vec_p0_p2;
		}
		else
		{
			localStart = size/2;
			localEnd = size;
			receiveStart = 0;
			receiveEnd = size/2;
			aesObjectForBitShares = aes_comp_msb_shares_bit_vec_p1_p2;
		}

		for(size_t i=localStart; i<localEnd; i++)
		{
			theta_shares[i] = aesObjectForBitShares->get64Bits();
		}

		//Now receive remaining from P2
		receiveArr<aramisSecretType>(theta_shares.data() + receiveStart, PARTY_C, (receiveEnd - receiveStart));
	}

#endif

	if (PRIMARY)
	{
		// theta_shares is the same as gamma (in older versions);
		// LSB_shares is the same as delta (in older versions);


		aramisSecretType j = 0;
		if (partyNum == PARTY_A)
			j = floatToMyType(1);

		for (size_t i = 0; i < size; ++i)
			theta_shares[i] = (1 - 2*beta[i])*theta_shares[i] + j*beta[i];

		for (size_t i = 0; i < size; ++i)
			LSB_shares[i] = (1 - 2*(c[i] & 1))*LSB_shares[i] + j*(c[i] & 1);
	}
	vector<aramisSecretType> prod(size), temp(size);

#ifdef BENCH_CROSS_BRIDGE_CALLS
	funcDotProductMPC(theta_shares, LSB_shares, prod, size, true);
#else
	funcDotProductMPC(theta_shares, LSB_shares, prod, size);
#endif
	if (PRIMARY)
	{
		populateRandomVector<aramisSecretType>(temp, size, "COMMON", "NEGATIVE");
		for (size_t i = 0; i < size; ++i)
			b[i] = theta_shares[i] + LSB_shares[i] - 2*prod[i] + temp[i];
	}
}

// 3PC SelectShares: c contains shares of selector bit (encoded in aramisSecretType).
// a,b,c are shared across PARTY_A, PARTY_B
void funcSelectShares3PC(const vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &b,
		vector<aramisSecretType> &c, 
		size_t size)
{

	assert(THREE_PC && "funcSelectShares3PC called in non-3PC mdoe");
	funcDotProductMPC(a, b, c, size);
}

// 3PC: PARTY_A, PARTY_B hold shares in a, want shares of RELU' in b.
void funcRELUPrime3PC(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		size_t size)
{
	assert(THREE_PC && "funcRELUPrime3PC called in non-3PC mode");

	vector<aramisSecretType> twoA(size, 0);
	aramisSecretType j = 0;

	for (size_t i = 0; i < size; ++i)
		twoA[i] = (a[i] << 1);

	funcShareConvertMPC(twoA, size);
	funcComputeMSB3PC(twoA, b, size);
	if (partyNum == PARTY_A)
		j = floatToMyType(1);

	if (PRIMARY)
		for (size_t i = 0; i < size; ++i)
			b[i] = j - b[i];
}

//PARTY_A, PARTY_B hold shares in a, want shares of RELU in b.
void funcRELUMPC(const vector<aramisSecretType> &a, 
		vector<aramisSecretType> &b, 
		size_t size)
{

	vector<aramisSecretType> reluPrime(size);
	funcRELUPrime3PC(a, reluPrime, size);
	funcSelectShares3PC(a, reluPrime, b, size);
}


//Chunk wise maximum of a vector of size rows*columns and maximum is caclulated of every
//column number of elements. max is a vector of size rows. maxIndex contains the index of
//the maximum value.
//PARTY_A, PARTY_B start with the shares in a and {A,B} and {C,D} have the results in
//max and maxIndex.
void funcMaxMPC(vector<aramisSecretType> &a, 
		vector<aramisSecretType> &max, 
		vector<aramisSecretType> &maxIndex,
		size_t rows, 
		size_t columns, 
		bool calculate_max_idx)
{

	vector<aramisSecretType> diff(rows), diffIndex(rows), rp(rows), indexShares(rows*columns, 0);

	for (size_t i = 0; i < rows; ++i)
	{
		max[i] = a[i*columns];
		maxIndex[i] = 0;
	}

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			if (partyNum == PARTY_A)
				indexShares[i*columns + j] = j;

	for (size_t i = 1; i < columns; ++i)
	{
		for (size_t	j = 0; j < rows; ++j)
			diff[j] = max[j] - a[j*columns + i];

		for (size_t	j = 0; j < rows; ++j)
			diffIndex[j] = maxIndex[j] - indexShares[j*columns + i];

		if(final_argmax){
			funcRELUPrime3PC(diff, rp, rows);
		}
		else{
#ifdef SPLIT_MAXPOOL
			uint64_t total_maxpool_cost_mb = (rows*(1+8*(2+1+1+4+8+2+7+2)))/(1024*1024);
			uint64_t no_of_chunks = total_maxpool_cost_mb/MAXPOOL_SPLIT_CHUNK_SIZE;
			if(total_maxpool_cost_mb <= MAXPOOL_SPLIT_CHUNK_SIZE){
				funcRELUPrime3PC(diff, rp, rows);
			}
			else{
				uint64_t maxpool_done_till = 0;
				uint64_t maxpool_size_per_chunk = rows/no_of_chunks;
				uint64_t maxpool_size_last_chunk = rows - (no_of_chunks*maxpool_size_per_chunk);

				vector<aramisSecretType> diff_cut(maxpool_size_per_chunk), rp_cut(maxpool_size_per_chunk);

				for(int lc=0; lc<no_of_chunks; lc++){
					for(int aa=0; aa<maxpool_size_per_chunk; aa++){
						diff_cut[aa] = diff[maxpool_done_till+aa];
					}
					funcRELUPrime3PC(diff_cut, rp_cut, maxpool_size_per_chunk);
					for(int aa=0; aa<maxpool_size_per_chunk; aa++){
						rp[maxpool_done_till+aa] = rp_cut[aa];
					}
					maxpool_done_till += maxpool_size_per_chunk;
				}
				diff_cut = vector<aramisSecretType>();
				rp_cut = vector<aramisSecretType>();

				if(maxpool_size_last_chunk != 0){
					vector<aramisSecretType> diff_cut_l(maxpool_size_last_chunk), rp_cut_l(maxpool_size_last_chunk);
					for(int aa=0; aa<maxpool_size_last_chunk; aa++){
						diff_cut_l[aa] = diff[maxpool_done_till+aa];
					}
					funcRELUPrime3PC(diff_cut_l, rp_cut_l, maxpool_size_last_chunk);
					for(int aa=0; aa<maxpool_size_last_chunk; aa++){
						rp[maxpool_done_till+aa] = rp_cut_l[aa];
					}

				}

			}
#else
			funcRELUPrime3PC(diff, rp, rows);
#endif
		}
		funcSelectShares3PC(diff, rp, max, rows);
		if(calculate_max_idx)
			funcSelectShares3PC(diffIndex, rp, maxIndex, rows);
		for (size_t	j = 0; j < rows; ++j)
			max[j] = max[j] + a[j*columns + i];
		if(calculate_max_idx){
			for (size_t	j = 0; j < rows; ++j)
				maxIndex[j] = maxIndex[j] + indexShares[j*columns + i];
		}
	}

}


//MaxIndex is of size rows. a is of size rows*columns.
//a will be set to 0's except at maxIndex (in every set of column)
void funcMaxIndexMPC(vector<aramisSecretType> &a, 
		const vector<aramisSecretType> &maxIndex,
		size_t rows, 
		size_t columns)
{
	assert(((1 << (BIT_SIZE-1)) % columns) == 0 && "funcMaxIndexMPC works only for power of 2 columns");
	assert(columns < 257 && "This implementation does not support larger than 257 columns");

	vector<smallType> random(rows);

	if (PRIMARY)
	{
		vector<smallType> toSend(rows);
		for (size_t i = 0; i < rows; ++i)
			toSend[i] = (smallType)maxIndex[i] % columns;

		populateRandomVector<smallType>(random, rows, "COMMON", "POSITIVE");
		if (partyNum == PARTY_A)
			addVectors<smallType>(toSend, random, toSend, rows);

		sendVector<smallType>(toSend, PARTY_C, rows);
	}

	if (partyNum == PARTY_C)
	{
		vector<smallType> index(rows), temp(rows);
		vector<aramisSecretType> vector(rows*columns, 0), share_1(rows*columns), share_2(rows*columns);
		receiveVector<smallType>(index, PARTY_A, rows);
		receiveVector<smallType>(temp, PARTY_B, rows);
		addVectors<smallType>(index, temp, index, rows);

		for (size_t i = 0; i < rows; ++i)
			index[i] = index[i] % columns;

		for (size_t i = 0; i < rows; ++i)
			vector[i*columns + index[i]] = 1;

		splitIntoShares(vector, share_1, share_2, rows*columns);
		sendVector<aramisSecretType>(share_1, PARTY_A, rows*columns);
		sendVector<aramisSecretType>(share_2, PARTY_B, rows*columns);
	}

	if (PRIMARY)
	{
		receiveVector<aramisSecretType>(a, PARTY_C, rows*columns);
		size_t offset = 0;
		for (size_t i = 0; i < rows; ++i)
		{
			rotate(a.begin()+offset, a.begin()+offset+(random[i] % columns), a.begin()+offset+columns);
			offset += columns;
		}
	}
}



/*****************************************
 * New Functions for Convolution
 *****************************************/

void Conv2DReshapeFilter_new(int32_t FH, 
		int32_t FW, 
		int32_t CI, 
		int32_t CO, 
		auto& inputArr, 
		auto& outputArr)
{
	for (uint32_t co =  (int32_t)0; co < CO; co++){
		for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
			for (uint32_t fw =  (int32_t)0; fw < FW; fw++){
				for (uint32_t ci =  (int32_t)0; ci < CI; ci++){

					int32_t linIdx = ((((fh * FW) * CI) + (fw * CI)) + ci);
					outputArr[co][linIdx] = inputArr[fh][fw][ci][co];
				}
			}
		}
	}
}

void Conv2DReshapeFilterArr_new(int32_t FH, 
		int32_t FW, 
		int32_t CI, 
		int32_t CO, 
		uint64_t* inputArr, 
		uint64_t* outputArr)
{
	int32_t outputArrCols = FH*FW*CI;
	for (uint32_t co =  (int32_t)0; co < CO; co++){
		for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
			for (uint32_t fw =  (int32_t)0; fw < FW; fw++){
				for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
					int32_t linIdx = ((((fh * FW) * CI) + (fw * CI)) + ci);
					Arr2DIdx(outputArr, CO, outputArrCols, co, linIdx) = Arr4DIdx(inputArr, FH, FW, CI, CO, fh, fw, ci, co);
				}
			}
		}
	}
}

void Conv2DReshapeMatMulOP_new(int32_t N, 
		int32_t finalH, 
		int32_t finalW, 
		int32_t CO, 
		auto& inputArr, 
		auto& outputArr)
{
	for (uint32_t co =  (int32_t)0; co < CO; co++){
		for (uint32_t n =  (int32_t)0; n < N; n++){
			for (uint32_t h =  (int32_t)0; h < finalH; h++){
				for (uint32_t w =  (int32_t)0; w < finalW; w++){
					outputArr[n][h][w][co] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)];
				}
			}
		}
	}
}

void Conv2DReshapeInput_new(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FH, 
		int32_t FW, 
		int32_t zPadHLeft, 
		uint32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW, 
		int32_t RRows, 
		int32_t RCols, 
		auto& inputArr, 
		auto& outputArr)
{

	int32_t linIdxFilterMult =  (int32_t)0;
	for (uint32_t n =  (int32_t)0; n < N; n++){

		int32_t leftTopCornerH = ( (int32_t)0 - zPadHLeft);

		int32_t extremeRightBottomCornerH = ((H -  (int32_t)1) + zPadHRight);
		while ((((leftTopCornerH + FH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

			int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft);

			int32_t extremeRightBottomCornerW = ((W -  (int32_t)1) + zPadWRight);
			while ((((leftTopCornerW + FW) -  (int32_t)1) <= extremeRightBottomCornerW)) {
				for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
					for (uint32_t fw =  (int32_t)0; fw < FW; fw++){

						int32_t curPosH = (leftTopCornerH + fh);

						int32_t curPosW = (leftTopCornerW + fw);

						uint64_t val = 0;
						for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
							if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
								val = 0;
							} else {
								val = inputArr[n][curPosH][curPosW][ci];
							}
							outputArr[((((fh * FW) * CI) + (fw * CI)) + ci)][linIdxFilterMult] = val;
						}
					}
				}
				linIdxFilterMult = (linIdxFilterMult +  (int32_t)1);
				leftTopCornerW = (leftTopCornerW + strideW);
			}

			leftTopCornerH = (leftTopCornerH + strideH);
		}

	}
}

void Conv2DReshapeInputArr_new(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t CI,
		int32_t FH, 
		int32_t FW,
		int32_t zPadHLeft, 
		int32_t zPadHRight,
		int32_t zPadWLeft, 
		int32_t zPadWRight,
		int32_t strideH, 
		int32_t strideW,
		int32_t RRows, 
		int32_t RCols,
		uint64_t* inputArr, 
		uint64_t* outputArr)
{
	int32_t linIdxFilterMult =  (int32_t)0;
	for (uint32_t n =  (int32_t)0; n < N; n++){

		int32_t leftTopCornerH = ( (int32_t)0 - zPadHLeft);

		int32_t extremeRightBottomCornerH = ((H -  (int32_t)1) + zPadHRight);
		while ((((leftTopCornerH + FH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

			int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft);

			int32_t extremeRightBottomCornerW = ((W -  (int32_t)1) + zPadWRight);
			while ((((leftTopCornerW + FW) -  (int32_t)1) <= extremeRightBottomCornerW)) {
				for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
					for (uint32_t fw =  (int32_t)0; fw < FW; fw++){

						int32_t curPosH = (leftTopCornerH + fh);

						int32_t curPosW = (leftTopCornerW + fw);

						uint64_t val = 0;
						for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
							if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
								val = 0;
							} else {
								val = Arr4DIdx(inputArr, N, H, W, CI, n, curPosH, curPosW, ci);
							}
							int32_t firstIdxOutputArr = ((((fh * FW) * CI) + (fw * CI)) + ci);
							Arr2DIdx(outputArr, RRows, RCols, firstIdxOutputArr, linIdxFilterMult) = val;
						}
					}
				}
				linIdxFilterMult = (linIdxFilterMult +  (int32_t)1);
				leftTopCornerW = (leftTopCornerW + strideW);
			}

			leftTopCornerH = (leftTopCornerH + strideH);
		}

	}
}



void Conv2DCSF_optimized_backend(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW,
		vector< vector< vector< vector<aramisSecretType> > > >& y,
		vector< vector< vector< vector<aramisSecretType> > > >& x,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr,
		int64_t consSF,
		auto& e_clear,
		auto& f_clear,
		auto& m_out)
{

	//Assign some helpful variables here.
	int r_x_cols = FH*FW*CI;
	int r_x_rows = CO;
	int r_y_rows = FH*FW*CI;
	int32_t newH = (((H + (zPadHLeft+zPadHRight) - FH) / strideH) +  (int32_t)1);
	int32_t newW = (((W + (zPadWLeft+zPadWRight) - FW) / strideW) +  (int32_t)1);
	int r_y_cols = N*newH*newW;
	int x_size = FH*FW*CI*CO;
	int y_size = N*H*W*CI;
	int r_x_size = r_x_cols*r_x_rows;
	int r_y_size = r_y_cols*r_y_rows;

	int r_f_cols = r_x_cols;
	int r_f_rows = r_x_rows;
	int r_i_cols = r_y_cols;
	int r_i_rows = r_y_rows;

	int size_left = x_size;
	int size_right = y_size;
	int size_out = r_f_rows*r_i_cols;

	int32_t reshapedFilterRows = CO;
	int32_t reshapedFilterCols = ((FH * FW) * CI);
	int32_t reshapedIPRows = ((FH * FW) * CI);
	int32_t reshapedIPCols = ((N * newH) * newW);

	if (partyNum == PARTY_A || partyNum == PARTY_B){
		// First generate E and F

#ifndef MAKE_VECTOR
		vector<vector<vector<vector<aramisSecretType>>>> m_x(FH, vector<vector<vector<aramisSecretType>>>(FW, vector<vector<aramisSecretType>>(CI, vector<aramisSecretType>(CO))));
		vector<vector<vector<vector<aramisSecretType>>>> m_y(N, vector<vector<vector<aramisSecretType>>>(H, vector<vector<aramisSecretType>>(W, vector<aramisSecretType>(CI))));
#else
		auto m_x = make_vector<uint64_t>(FH, FW, CI, CO);
		auto m_y = make_vector<uint64_t>(N, H, W, CI);
#endif
		//populate here based on party number.

		if(partyNum == PARTY_A){
#ifdef PARALLEL_AES_CONV_ALL
			populate_4D_vectorParallel(m_x, FH, FW, CI, CO, "a1");
			populate_4D_vectorParallel(m_y, N, H, W, CI, "b1");
#else
			populate_4D_vector(m_x, FH, FW, CI, CO, "a1");
			populate_4D_vector(m_y, N, H, W, CI, "b1");

#endif			
		}
		else if(partyNum == PARTY_B){
#ifdef PARALLEL_AES_CONV_ALL
			populate_4D_vectorParallel(m_x, FH, FW, CI, CO, "a2");
			populate_4D_vectorParallel(m_y, N, H, W, CI, "b2");
#else
			populate_4D_vector(m_x, FH, FW, CI, CO, "a2");
			populate_4D_vector(m_y, N, H, W, CI, "b2");
#endif
		}

#ifndef MAKE_VECTOR
		vector<vector<vector<vector<aramisSecretType>>>> e(FH, vector<vector<vector<aramisSecretType>>>(FW, vector<vector<aramisSecretType>>(CI, vector<aramisSecretType>(CO))));
		vector<vector<vector<vector<aramisSecretType>>>> f(N, vector<vector<vector<aramisSecretType>>>(H, vector<vector<aramisSecretType>>(W, vector<aramisSecretType>(CI))));
		vector<vector<vector<vector<aramisSecretType>>>> e_other(FH, vector<vector<vector<aramisSecretType>>>(FW, vector<vector<aramisSecretType>>(CI, vector<aramisSecretType>(CO))));
		vector<vector<vector<vector<aramisSecretType>>>> f_other(N, vector<vector<vector<aramisSecretType>>>(H, vector<vector<aramisSecretType>>(W, vector<aramisSecretType>(CI))));
#else
		auto e = make_vector<uint64_t>(FH, FW, CI, CO);
		auto f = make_vector<uint64_t>(N, H, W, CI);
		auto e_other = make_vector<uint64_t>(FH, FW, CI, CO);
		auto f_other = make_vector<uint64_t>(N, H, W, CI);
#endif
		subtract_4D_vectors(x, m_x, e, FH, FW, CI, CO);
		subtract_4D_vectors(y, m_y, f, N, H, W, CI);

		//Reveal e and f.
		if(partyNum == PARTY_A){
			send_4D_vector(e, FH, FW, CI, CO);
			send_4D_vector(f, N, H, W, CI);
			receive_4D_vector(e_other, FH, FW, CI, CO);
			receive_4D_vector(f_other, N, H, W, CI);
		}
		else if(partyNum == PARTY_B){
			receive_4D_vector(e_other, FH, FW, CI, CO);
			receive_4D_vector(f_other, N, H, W, CI);
			send_4D_vector(e, FH, FW, CI, CO);
			send_4D_vector(f, N, H, W, CI);
		}

		add_4D_vectors(e, e_other, e_clear, FH, FW, CI, CO);
		add_4D_vectors(f, f_other, f_clear, N, H, W, CI);

	}

}

void ConvMatMul_new(vector<aramisSecretType> &a_r, 
		vector<aramisSecretType> &b_r, 
		vector<aramisSecretType> &c,
		vector<aramisSecretType> &clear_e_r, 
		vector<aramisSecretType> &clear_f_r,
		const vector<aramisSecretType> &m_out,
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b, 
		int64_t consSF)
{
	size_t size = rows*columns;
	size_t size_left = rows*common_dim;
	size_t size_right = common_dim*columns;
	vector<aramisSecretType> C(size, 0);


	if (PRIMARY)
	{
		vector<aramisSecretType> temp_c(size);

		if (partyNum == PARTY_A)
		{	vector<aramisSecretType> A(size_left, 0);
			subtractVectors<aramisSecretType>(a_r, clear_e_r, A, size_left);
			matrixMultEigen(A, clear_f_r, c, rows, common_dim, columns, 0, 0);
			matrixMultEigen(clear_e_r, b_r, temp_c, rows, common_dim, columns, 0, 0);
		}
		else
		{
			matrixMultEigen(a_r, clear_f_r, c, rows, common_dim, columns, 0, 0);
			matrixMultEigen(clear_e_r, b_r, temp_c, rows, common_dim, columns, 0, 0);
		}

		addVectors<aramisSecretType>(c, temp_c, c, size);
		addVectors<aramisSecretType>(c, m_out, c, size);

		funcTruncate2PC(c, consSF, size, PARTY_A, PARTY_B);
	}
}

void MatMulCSF2D_new(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<aramisSecretType> >& A, 
		vector< vector<aramisSecretType> >& B, 
		vector< vector<aramisSecretType> >& C, 
		int32_t consSF, 
		vector< vector<aramisSecretType> >& e, 
		vector< vector<aramisSecretType> >& f, 
		vector< vector<aramisSecretType> >& m_C)
{
	vector<aramisSecretType> X(i*j);
	vector<aramisSecretType> Y(j*k);
	vector<aramisSecretType> Z(i*k);
	vector<aramisSecretType> E(i*j);
	vector<aramisSecretType> F(j*k);
	vector<aramisSecretType> m_Z(i*k);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<j; jj++){
			X[ii*j + jj] = A[ii][jj]; //Each row is of size j
		}
	}
	for (int ii=0; ii<j; ii++){
		for (int jj=0; jj<k; jj++){
			Y[ii*k + jj] = B[ii][jj]; //Each row is of size k
		}
	}

	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<j; jj++){
			E[ii*j + jj] = e[ii][jj]; //Each row is of size j
		}
	}
	for (int ii=0; ii<j; ii++){
		for (int jj=0; jj<k; jj++){
			F[ii*k + jj] = f[ii][jj]; //Each row is of size k
		}
	}

	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<k; jj++){
			m_Z[ii*k + jj] = m_C[ii][jj];
		}
	}

	ConvMatMul_new(X, Y, Z, E, F, m_Z, i, j, k, 0, 0, consSF);

	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<k; jj++){
			C[ii][jj] = Z[ii*k + jj]; //Each row is of size k
		}
	}
}

void MatMulCSF2D_Plain_Eigen(int32_t i, 
		int32_t j, 
		int32_t k, 
		vector< vector<aramisSecretType> >& A, 
		vector< vector<aramisSecretType> >& B, 
		vector< vector<aramisSecretType> >& C, 
		int32_t consSF)
{
	vector<aramisSecretType> X(i*j);
	vector<aramisSecretType> Y(j*k);
	vector<aramisSecretType> Z(i*k);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<j; jj++){
			X[ii*j + jj] = A[ii][jj]; //Each row is of size j
		}
	}
	for (int ii=0; ii<j; ii++){
		for (int jj=0; jj<k; jj++){
			Y[ii*k + jj] = B[ii][jj]; //Each row is of size k
		}
	}
	matrixMultEigen(X, Y, Z, i, j, k, 0, 0);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<k; jj++){
			C[ii][jj] = Z[ii*k + jj]; //Each row is of size k
		}
	}
}

void ConvLocalMatMulOps(vector< vector<aramisSecretType> >& X,
	       	vector< vector<aramisSecretType> >& Y,
	       	vector< vector<aramisSecretType> >& Z, //Z is the output of the function
		vector< vector<aramisSecretType> >& E_clear, 
		vector< vector<aramisSecretType> >& F_clear,
		vector< vector<aramisSecretType> >& C,
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		int64_t consSF)
{
#ifdef DEBUG
	assert(PRIMARY && "Should have been called only from PRIMARY.");
#endif

	vector<vector<aramisSecretType>> temp_Z(rows, vector<aramisSecretType>(columns));
	if (partyNum == PARTY_A)
	{
		//Calculate X - E_clear
		vector<vector<aramisSecretType>> tempSubHolder(rows, vector<aramisSecretType>(common_dim));
		subtract_2D_vectors(X, E_clear, tempSubHolder, rows, common_dim);
		matrixMultEigen(tempSubHolder, F_clear, Z, rows, common_dim, columns, 0, 0);
		matrixMultEigen(E_clear, Y, temp_Z, rows, common_dim, columns, 0, 0);
	}
	else
	{
		matrixMultEigen(X, F_clear, Z, rows, common_dim, columns, 0, 0);
		matrixMultEigen(E_clear, Y, temp_Z, rows, common_dim, columns, 0, 0);
	}

	add_2D_vectors(Z, temp_Z, Z, rows, columns);
}

void funcConv2DCSF(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW,
		vector< vector< vector< vector<aramisSecretType> > > >& inputArr,
		vector< vector< vector< vector<aramisSecretType> > > >& filterArr,
		int64_t consSF,
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{
	assert(consSF == FLOAT_PRECISION && "Please update FLOAT_PRECISION value in globals.h to be equal to consSF");	
	int32_t reshapedFilterRows = CO;
	int32_t reshapedFilterCols = ((FH * FW) * CI);
	int32_t reshapedIPRows = ((FH * FW) * CI);
	int32_t newH = (((H + (zPadHLeft+zPadHRight) - FH) / strideH) +  (int32_t)1);
	int32_t newW = (((W + (zPadWLeft+zPadWRight) - FW) / strideW) +  (int32_t)1);
	int32_t reshapedIPCols = ((N * newH) * newW);


	//PartyC works here
	if(partyNum == PARTY_C){

		uint64_t* m_x1 = (uint64_t*)malloc(sizeof(uint64_t)*FH*FW*CI*CO);
		uint64_t* m_y1 = (uint64_t*)malloc(sizeof(uint64_t)*N*H*W*CI);
		uint64_t* m_x = (uint64_t*)malloc(sizeof(uint64_t)*FH*FW*CI*CO);
		uint64_t* m_y = (uint64_t*)malloc(sizeof(uint64_t)*N*H*W*CI);
		uint64_t* m_z0 = (uint64_t*)malloc(sizeof(uint64_t)*reshapedFilterRows*reshapedIPCols);
		uint64_t* m_z = (uint64_t*)malloc(sizeof(uint64_t)*reshapedFilterRows*reshapedIPCols);
		uint64_t* r_m_x = (uint64_t*)malloc(sizeof(uint64_t)*reshapedFilterRows*reshapedFilterCols);
		uint64_t* r_m_y = (uint64_t*)malloc(sizeof(uint64_t)*reshapedIPRows*reshapedIPCols);
		if(m_x1 == NULL){
			print_string("Malloc 1 error in conv opti. Returns NULL");
		}
		if(m_y1 == NULL){
			print_string("Malloc 2 error in conv opti. Returns NULL");
		}
		if(m_x == NULL){
			print_string("Malloc 3 error in conv opti. Returns NULL");
		}
		if(m_y == NULL){
			print_string("Malloc 4 error in conv opti. Returns NULL");
		}
		if(m_z0 == NULL){
			print_string("Malloc 5 error in conv opti. Returns NULL");
		}
		if(m_z == NULL){
			print_string("Malloc 6 error in conv opti. Returns NULL");
		}
		if(r_m_x == NULL){
			print_string("Malloc 7 error in conv opti. Returns NULL");
		}
		if(r_m_y == NULL){
			print_string("Malloc 8 error in conv opti. Returns NULL");
		}

#ifdef PARALLEL_AES_CONV_ALL
		populate_AES_ArrParallel(m_x, ((uint64_t)FH)*FW*CI*CO, "a1");
		populate_AES_ArrParallel(m_x1, ((uint64_t)FH)*FW*CI*CO, "a2");
		populate_AES_ArrParallel(m_y, ((uint64_t)N)*H*W*CI, "b1");
		populate_AES_ArrParallel(m_y1, ((uint64_t)N)*H*W*CI, "b2");
		populate_AES_ArrParallel(m_z0, reshapedFilterRows*reshapedIPCols, "c1");
#else
		populate_AES_Arr(m_x, ((uint64_t)FH)*FW*CI*CO, "a1");
		populate_AES_Arr(m_x1, ((uint64_t)FH)*FW*CI*CO, "a2");
		populate_AES_Arr(m_y, ((uint64_t)N)*H*W*CI, "b1");
		populate_AES_Arr(m_y1, ((uint64_t)N)*H*W*CI, "b2");
		populate_AES_Arr(m_z0, reshapedFilterRows*reshapedIPCols, "c1");

#endif
		add_2_Arr(m_x, m_x1, m_x, ((uint64_t)FH)*FW*CI*CO);
		add_2_Arr(m_y, m_y1, m_y, ((uint64_t)N)*H*W*CI);

		Conv2DReshapeFilterArr_new(FH, FW, CI, CO, m_x, r_m_x);
		Conv2DReshapeInputArr_new(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, m_y, r_m_y);

		//Do MatMul and find the value of m_z1
		matrixMultEigen(r_m_x, r_m_y, m_z, reshapedFilterRows, reshapedFilterCols, reshapedIPCols, 0, 0);

		subtract_2_Arr(m_z, m_z0, m_z, reshapedFilterRows*reshapedIPCols);
		sendArr<aramisSecretType>(m_z, PARTY_B, reshapedFilterRows*reshapedIPCols);

		free(m_x1);
		free(m_y1);
		free(m_x);
		free(m_y);
		free(m_z0);
		free(m_z);
		free(r_m_x);
		free(r_m_y);

	}
	if(PRIMARY)
	{

#ifndef MAKE_VECTOR
		vector<vector<aramisSecretType>> filterReshaped(reshapedFilterRows, vector<aramisSecretType>(reshapedFilterCols));
		vector<vector<aramisSecretType>> E_filterReshaped(reshapedFilterRows, vector<aramisSecretType>(reshapedFilterCols));
#else
		auto filterReshaped = make_vector<uint64_t>(reshapedFilterRows, reshapedFilterCols);
		auto E_filterReshaped = make_vector<uint64_t>(reshapedFilterRows, reshapedFilterCols);
#endif

#ifndef MAKE_VECTOR
		vector<vector<aramisSecretType>> inputReshaped(reshapedIPRows, vector<aramisSecretType>(reshapedIPCols));
		vector<vector<aramisSecretType>> F_inputReshaped(reshapedIPRows, vector<aramisSecretType>(reshapedIPCols));
#else
		auto inputReshaped = make_vector<uint64_t>(reshapedIPRows, reshapedIPCols);
		auto F_inputReshaped = make_vector<uint64_t>(reshapedIPRows, reshapedIPCols);
#endif

#ifndef MAKE_VECTOR
		vector<vector<aramisSecretType>> matmulOP(reshapedFilterRows, vector<aramisSecretType>(reshapedIPCols));
		vector<vector<aramisSecretType>> m_matmulOP(reshapedFilterRows, vector<aramisSecretType>(reshapedIPCols));
#else
		auto matmulOP = make_vector<uint64_t>(reshapedFilterRows, reshapedIPCols);
		auto m_matmulOP = make_vector<uint64_t>(reshapedFilterRows, reshapedIPCols);
#endif

#ifndef MAKE_VECTOR
		vector<vector<vector<vector<aramisSecretType>>>> e_clear(FH, vector<vector<vector<aramisSecretType>>>(FW, vector<vector<aramisSecretType>>(CI, vector<aramisSecretType>(CO))));
		vector<vector<vector<vector<aramisSecretType>>>> f_clear(N, vector<vector<vector<aramisSecretType>>>(H, vector<vector<aramisSecretType>>(W, vector<aramisSecretType>(CI))));
#else
		auto e_clear = make_vector<uint64_t>(FH, FW, CI, CO);
		auto f_clear = make_vector<uint64_t>(N, H, W, CI);
#endif
		Conv2DCSF_optimized_backend(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr, consSF, e_clear, f_clear, m_matmulOP);


		Conv2DReshapeFilter_new(FH, FW, CI, CO, filterArr, filterReshaped);
		Conv2DReshapeFilter_new(FH, FW, CI, CO, e_clear, E_filterReshaped);

		Conv2DReshapeInput_new(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
		Conv2DReshapeInput_new(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, f_clear, F_inputReshaped);

		ConvLocalMatMulOps(filterReshaped, inputReshaped, matmulOP, E_filterReshaped, F_inputReshaped, m_matmulOP, reshapedFilterRows, reshapedFilterCols, reshapedIPCols, consSF);

		if(partyNum == PARTY_A){
#ifdef PARALLEL_AES_CONV_ALL
			populate_2D_vectorParallel(m_matmulOP, reshapedFilterRows, reshapedIPCols, "c1");
#else
			populate_2D_vector(m_matmulOP, reshapedFilterRows, reshapedIPCols, "c1");
#endif
		}
		else if(partyNum == PARTY_B){
			receive_2D_vector(m_matmulOP, reshapedFilterRows, reshapedIPCols);
		}

		add_2D_vectors(matmulOP, m_matmulOP, matmulOP, reshapedFilterRows, reshapedIPCols);
		funcTruncate2PC(matmulOP, consSF, reshapedFilterRows, reshapedIPCols, PARTY_A, PARTY_B);

		Conv2DReshapeMatMulOP_new(N, newH, newW, CO, matmulOP, outArr);

	}
}

#ifdef SPLIT_CONV
/********************************** Aramis specific convolution splitting ************************************/

void Conv2DReshapeFilterSplit(int32_t FH, 
		int32_t FW, 
		int32_t CI, 
		int32_t CO, 
		auto& inputArr, 
		auto& outputArr){
	for (uint32_t co =  (int32_t)0; co < CO; co++){
		for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
			for (uint32_t fw =  (int32_t)0; fw < FW; fw++){
				for (uint32_t ci =  (int32_t)0; ci < CI; ci++){

					int32_t linIdx = ((((fh * FW) * CI) + (fw * CI)) + ci);
					outputArr[co][linIdx] = inputArr[fh][fw][ci][co];
				}
			}
		}
	}
}

void Conv2DReshapeMatMulOPSplit(int32_t N, 
		int32_t finalH, 
		int32_t finalW, 
		int32_t CO, 
		auto& inputArr, 
		auto& outputArr)
{
	for (uint32_t co =  (int32_t)0; co < CO; co++){
		for (uint32_t n =  (int32_t)0; n < N; n++){
			for (uint32_t h =  (int32_t)0; h < finalH; h++){
				for (uint32_t w =  (int32_t)0; w < finalW; w++){
					outputArr[n][h][w][co] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)];
				}
			}
		}
	}
}

void Conv2DReshapeInputSplit(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FH, 
		int32_t FW, 
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW, 
		int32_t RRows, 
		int32_t RCols, 
		auto& inputArr, 
		auto& outputArr, 
		int32_t begin, 
		int32_t end)
{

	int32_t linIdxFilterMult =  (int32_t)0;
	for (uint32_t n =  (int32_t)0; n < N; n++){

		int32_t leftTopCornerH = begin;

		int32_t extremeRightBottomCornerH = end;
		while ((((leftTopCornerH + FH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

			int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft);

			int32_t extremeRightBottomCornerW = ((W -  (int32_t)1) + zPadWRight);
			while ((((leftTopCornerW + FW) -  (int32_t)1) <= extremeRightBottomCornerW)) {
				for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
					for (uint32_t fw =  (int32_t)0; fw < FW; fw++){

						int32_t curPosH = (leftTopCornerH + fh);

						int32_t curPosW = (leftTopCornerW + fw);

						uint64_t val =  (int64_t)0;
						for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
							if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
								val = ( (int64_t)0);
							} else {
								val = inputArr[n][curPosH][curPosW][ci];
							}
							outputArr[((((fh * FW) * CI) + (fw * CI)) + ci)][linIdxFilterMult] = val;
						}
					}
				}
				linIdxFilterMult = (linIdxFilterMult +  (int32_t)1);
				leftTopCornerW = (leftTopCornerW + strideW);
			}

			leftTopCornerH = (leftTopCornerH + strideH);
		}

	}
}

void append_columns(vector<vector<uint64_t>> input, 
		vector<vector<uint64_t>> &output, 
		int d1, 
		int d2, 
		int32_t column_id)
{
	for(int i=0; i<d1; i++){ // goes till CO
		for(int j=0; j<d2; j++){
			output[i][column_id+j] = input[i][j];
		}
	}
}

void MatMulCSF2DSplit(int32_t i,
	       	int32_t j,
	       	int32_t k,
	       	vector< vector<aramisSecretType> >& A,
	       	vector< vector<aramisSecretType> >& B,
	       	vector< vector<aramisSecretType> >& C,
	       	int32_t consSF)
{
	assert(consSF == FLOAT_PRECISION && "Please update FLOAT_PRECISION in globals.h to be equal to consSF");
	vector<aramisSecretType> X(i*j);
	vector<aramisSecretType> Y(j*k);
	vector<aramisSecretType> Z(i*k);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<j; jj++){
			X[ii*j + jj] = A[ii][jj]; //Each row is of size j
		}
	}
	for (int ii=0; ii<j; ii++){
		for (int jj=0; jj<k; jj++){
			Y[ii*k + jj] = B[ii][jj]; //Each row is of size k
		}
	}
	funcMatMulMPC(X, Y, Z, i, j, k, 0, 0);
	for (int ii=0; ii<i; ii++){
		for (int jj=0; jj<k; jj++){
			C[ii][jj] = Z[ii*k + jj]; //Each row is of size k
		}
	}
}

void funcConv2DCSFSplit(int32_t N, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideH, 
		int32_t strideW, 
		vector< vector< vector< vector<aramisSecretType> > > >& inputArr, 
		vector< vector< vector< vector<aramisSecretType> > > >& filterArr, 
		int32_t consSF, 
		vector< vector< vector< vector<aramisSecretType> > > >& outArr)
{

	int32_t reshapedFilterRows = CO;

	int32_t reshapedFilterCols = ((FH * FW) * CI);

	int32_t reshapedIPRows = ((FH * FW) * CI);

	int32_t newH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) +  (int32_t)1);

	int32_t newW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) +  (int32_t)1);

	int32_t reshapedIPCols = ((N * newH) * newW);

	auto matmulOP_full = make_vector<uint64_t>(reshapedFilterRows, reshapedIPCols);

	uint64_t size_of_conv_mb = (8*((CO*FH*FW*CI)*(1+1+1+2+1) + (FH*FW*CI*N*newH*newW)*(1+1+1+2+1) + (CO*N*newH*newW)*(1+1+1+1+1)))/(1024*1024);
	uint64_t no_of_chunks = size_of_conv_mb/CONV_SPLIT_CHUNK_SIZE;
	//Reused.
	auto filterReshaped = make_vector<uint64_t>(reshapedFilterRows, reshapedFilterCols);
	Conv2DReshapeFilterSplit(FH, FW, CI, CO, filterArr, filterReshaped);

	if(size_of_conv_mb <= CONV_SPLIT_CHUNK_SIZE){
		auto inputReshaped = make_vector<uint64_t>(reshapedIPRows, reshapedIPCols);

		auto matmulOP = make_vector<uint64_t>(reshapedFilterRows, reshapedIPCols);
		Conv2DReshapeInputSplit(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped, ( (int32_t)0 - zPadHLeft), ((H -  (int32_t)1) + zPadHRight));
		MatMulCSF2DSplit(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP, consSF);
		Conv2DReshapeMatMulOPSplit(N, newH, newW, CO, matmulOP, outArr);
		return;
	}

	int32_t nh_by_c = (newH/no_of_chunks);
	int32_t last_nh_by_c = newH - (no_of_chunks*nh_by_c);

	int32_t col_id = 0;

	for(int i=0; i<no_of_chunks; i++){
		int32_t begin_h = 0-zPadHLeft + (nh_by_c*strideH*i);
		int32_t end_h = 0-zPadHLeft + ((((i+1)*nh_by_c)-1)*strideH) + FH - 1;
		auto inputReshaped = make_vector<uint64_t>(reshapedIPRows, N*(nh_by_c)*newW);
		auto matmulOP = make_vector<uint64_t>(reshapedFilterRows, N*(nh_by_c)*newW);

		Conv2DReshapeInputSplit(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped, begin_h, end_h);
		MatMulCSF2DSplit(reshapedFilterRows, reshapedFilterCols, N*nh_by_c*newW, filterReshaped, inputReshaped, matmulOP, consSF);

		append_columns(matmulOP, matmulOP_full, reshapedFilterRows, N*nh_by_c*newW, col_id);
		col_id += N*nh_by_c*newW;

	}

	if(last_nh_by_c != 0){
		int32_t begin_h = 0-zPadHLeft + (nh_by_c*strideH*no_of_chunks);
		int32_t end_h = H - 1 + zPadHRight;
		auto inputReshaped = make_vector<uint64_t>(reshapedIPRows, N*(last_nh_by_c)*newW);
		auto matmulOP = make_vector<uint64_t>(reshapedFilterRows, N*(last_nh_by_c)*newW);

		Conv2DReshapeInputSplit(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped, begin_h, end_h);

		MatMulCSF2DSplit(reshapedFilterRows, reshapedFilterCols, N*last_nh_by_c*newW, filterReshaped, inputReshaped, matmulOP, consSF);

		append_columns(matmulOP, matmulOP_full, reshapedFilterRows, N*last_nh_by_c*newW, col_id);
		col_id += N*last_nh_by_c*newW;

	}
	Conv2DReshapeMatMulOPSplit(N, newH, newW, CO, matmulOP_full, outArr);
}
#endif

/******************* Wrapper integer function calls ********************/
aramisSecretType funcMult(aramisSecretType a, 
		aramisSecretType b)
{
	vector<aramisSecretType> tmp1(1, a);
	vector<aramisSecretType> tmp2(1, b);
	vector<aramisSecretType> tmp3(1, 0);
	funcMatMulMPC(tmp1, tmp2, tmp3, 1, 1, 1, 0, 0, false);
	return tmp3[0];
}

aramisSecretType funcReluPrime(aramisSecretType a)
{
	vector<aramisSecretType> tmp1(1, a);
	vector<aramisSecretType> tmp2(1, 0);
	funcRELUPrime3PC(tmp1, tmp2, 1);
	return tmp2[0];
}

aramisSecretType funcSSCons(aramisSecretType a)
{
	vector<aramisSecretType> tmp1(1,a);
	vector<aramisSecretType> tmp2(1,0);
	funcSecretShareConstant(tmp1, tmp2, 1);
	return tmp2[0];
}

//Arg2 revealToParties is a bitmask as to which parties should see the reconstructed values
//10 - party 0, 01 - party 1, 11 - party 1&2
aramisSecretType funcReconstruct2PCCons(aramisSecretType a, 
		int revealToParties)
{
	if (HELPER)
	{
		//skip
		return a;
	}
	vector<aramisSecretType> tmp1(1,a);
	vector<aramisSecretType> tmp2(1,0);
	funcReconstruct2PC(tmp1, 1, "", &tmp2, revealToParties);
	return tmp2[0];
}


/******************************** Debug ********************************/
void debugDotProd()
{
	size_t size = 10;
	vector<aramisSecretType> a(size, 0), b(size, 0), c(size);
	vector<aramisSecretType> temp(size);

	populateRandomVector<aramisSecretType>(temp, size, "COMMON", "NEGATIVE");
	for (size_t i = 0; i < size; ++i)
	{
		if (partyNum == PARTY_A)
			a[i] = temp[i] + floatToMyType(i);
		else
			a[i] = temp[i];
	}

	populateRandomVector<aramisSecretType>(temp, size, "COMMON", "NEGATIVE");
	for (size_t i = 0; i < size; ++i)
	{
		if (partyNum == PARTY_A)
			b[i] = temp[i] + floatToMyType(i);
		else
			b[i] = temp[i];
	}

	funcDotProductMPC(a, b, c, size);

	if (PRIMARY)
		funcReconstruct2PC(c, size, "c");
}

void debugComputeMSB()
{
	size_t size = 10;
	vector<aramisSecretType> a(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			a[i] = i - 5;

	vector<aramisSecretType> c(size);
	funcComputeMSB3PC(a, c, size);

	if (PRIMARY)
		funcReconstruct2PC(c, size, "c");

}

void debugPC()
{
	size_t size = 10;
	vector<aramisSecretType> r(size);
	vector<smallType> bit_shares(size*BIT_SIZE, 0);

	for (size_t i = 0; i < size; ++i)
		r[i] = 5+i;

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			for (size_t j = 0; j < BIT_SIZE; ++j)
				if (j == BIT_SIZE - 1 - i)
					bit_shares[i*BIT_SIZE + j] = 1;

	vector<smallType> beta(size);
	vector<smallType> betaPrime(size);

	if (PRIMARY)
		populateBitsVector(beta, "COMMON", size);

	funcPrivateCompareMPC(bit_shares, r, beta, betaPrime, size, BIT_SIZE);

	if (PRIMARY)
		for (size_t i = 0; i < size; ++i)
			print_integer((int)beta[i]);

}

void debugDivision()
{
	size_t size = 10;
	vector<aramisSecretType> numerator(size);
	vector<aramisSecretType> denominator(size);
	vector<aramisSecretType> quotient(size,0);

	for (size_t i = 0; i < size; ++i)
		numerator[i] = 50;

	for (size_t i = 0; i < size; ++i)
		denominator[i] = 50*size;

	funcDivisionMPC(numerator, denominator, quotient, size);

	if (PRIMARY)
	{
		funcReconstruct2PC(numerator, size, "Numerator");
		funcReconstruct2PC(denominator, size, "Denominator");
		funcReconstruct2PC(quotient, size, "Quotient");
	}
}

void debugMax()
{
	size_t rows = 1;
	size_t columns = 10;
	vector<aramisSecretType> a(rows*columns, 0);

	if (partyNum == PARTY_A or partyNum == PARTY_C){
		a[0] = 0; a[1] = 1; a[2] = 0; a[3] = 4; a[4] = 5;
		a[5] = 3; a[6] = 10; a[7] = 6, a[8] = 41; a[9] = 9;
	}

	vector<aramisSecretType> max(rows), maxIndex(rows);
	funcMaxMPC(a, max, maxIndex, rows, columns);

	if (PRIMARY)
	{
		funcReconstruct2PC(a, columns, "a");
		funcReconstruct2PC(max, rows, "max");
		funcReconstruct2PC(maxIndex, rows, "maxIndex");
		//cout << "-----------------" << endl;
	}
}


void debugSS()
{
	size_t size = 10;
	vector<aramisSecretType> inputs(size, 0), outputs(size, 0);

	vector<aramisSecretType> selector(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			selector[i] = (aramisSecretType)(aes_indep->getBit() << FLOAT_PRECISION);

	if (PRIMARY)
		funcReconstruct2PC(selector, size, "selector");

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			inputs[i] = (aramisSecretType)aes_indep->get8Bits();

	funcSelectShares3PC(inputs, selector, outputs, size);

	if (PRIMARY)
	{
		funcReconstruct2PC(inputs, size, "inputs");
		funcReconstruct2PC(outputs, size, "outputs");
	}
}


void debugMatMul()
{
	size_t rows = 3;
	size_t common_dim = 2;
	size_t columns = 3;
	size_t transpose_a = 0, transpose_b = 0;

	vector<aramisSecretType> a(rows*common_dim);
	vector<aramisSecretType> b(common_dim*columns);
	vector<aramisSecretType> c(rows*columns);

	for (size_t i = 0; i < a.size(); ++i)
		a[i] = floatToMyType(i);

	for (size_t i = 0; i < b.size(); ++i)
		b[i] = floatToMyType(i);

	if (PRIMARY)
		funcReconstruct2PC(a, a.size(), "a");

	if (PRIMARY)
		funcReconstruct2PC(b, b.size(), "b");

	funcMatMulMPC(a, b, c, rows, common_dim, columns, transpose_a, transpose_b);

	if (PRIMARY)
		funcReconstruct2PC(c, c.size(), "c");
}

void debugReLUPrime()
{
	size_t size = 10;
	vector<aramisSecretType> inputs(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			inputs[i] = aes_indep->get8Bits() - aes_indep->get8Bits();

	vector<aramisSecretType> outputs(size, 0);
	funcRELUPrime3PC(inputs, outputs, size);
	if (PRIMARY)
	{
		funcReconstruct2PC(inputs, size, "inputs");
		funcReconstruct2PC(outputs, size, "outputs");
	}

}


void debugMaxIndex()
{
	size_t rows = 10;
	size_t columns = 4;

	vector<aramisSecretType> maxIndex(rows, 0);
	if (partyNum == PARTY_A)
		for (size_t i = 0; i < rows; ++i)
			maxIndex[i] = (aes_indep->get8Bits())%columns;

	vector<aramisSecretType> a(rows*columns);
	funcMaxIndexMPC(a, maxIndex, rows, columns);

	if (PRIMARY)
	{
		funcReconstruct2PC(maxIndex, maxIndex.size(), "maxIndex");

		vector<aramisSecretType> temp(rows*columns);
		if (partyNum == PARTY_B)
			sendVector<aramisSecretType>(a, PARTY_A, rows*columns);

		if (partyNum == PARTY_A)
		{
			receiveVector<aramisSecretType>(temp, PARTY_B, rows*columns);
			addVectors<aramisSecretType>(temp, a, temp, rows*columns);

			print_string("a: ");
			for (size_t i = 0; i < rows; ++i)
			{
				for (int j = 0; j < columns; ++j)
				{
					//print_linear(temp[i*columns + j], DEBUG_PRINT);
				}
				print_string("");
			}
			print_string("");
		}
	}
}




/******************************** Test ********************************/
void testMatMul(size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t iter)
{
	vector<aramisSecretType> a(rows*common_dim, 1);
	vector<aramisSecretType> b(common_dim*columns, 1);
	vector<aramisSecretType> c(rows*columns);

	if (STANDALONE)
	{
		for (int runs = 0; runs < iter; ++runs)
		{
			matrixMultEigen(a, b, c, rows, common_dim, columns, 0, 0);
			dividePlainSA(c, (1 << FLOAT_PRECISION));
		}
	}

	if (MPC)
	{
		for (int runs = 0; runs < iter; ++runs)
			funcMatMulMPC(a, b, c, rows, common_dim, columns, 0, 0);
	}
	print_integer(c[0]);
}

void testRelu(size_t r, 
		size_t c, 
		size_t iter)
{
	vector<aramisSecretType> a(r*c, 1);
	vector<smallType> reluPrimeSmall(r*c, 1);
	vector<aramisSecretType> reluPrimeLarge(r*c, 1);
	vector<aramisSecretType> b(r*c, 0);

	for (int runs = 0; runs < iter; ++runs)
	{
		if (STANDALONE)
			for (size_t i = 0; i < r*c; ++i)
				b[i] = a[i] * reluPrimeSmall[i];

		if (THREE_PC){
			funcRELUPrime3PC(a, reluPrimeLarge, r*c);
			funcSelectShares3PC(a, reluPrimeLarge, b, r*c);
		}
	}
}


void testReluPrime(size_t r, 
		size_t c, 
		size_t iter)
{
	vector<aramisSecretType> a(r*c, 1);
	vector<aramisSecretType> b(r*c, 0);
	vector<smallType> d(r*c, 0);

	for (int runs = 0; runs < iter; ++runs)
	{
		if (STANDALONE)
			for (size_t i = 0; i < r*c; ++i)
				b[i] = (a[i] < LARGEST_NEG ? 1:0);

		if (THREE_PC)
			funcRELUPrime3PC(a, b, r*c);

	}
}


void testMaxPool(size_t p_range, 
		size_t q_range, 
		size_t px, 
		size_t py, 
		size_t D, 
		size_t iter)
{
	size_t B = MINI_BATCH_SIZE;
	size_t size_x = p_range*q_range*D*B;

	vector<aramisSecretType> y(size_x, 0);
	vector<aramisSecretType> maxPoolShaped(size_x, 0);
	vector<aramisSecretType> act(size_x/(px*py), 0);
	vector<aramisSecretType> maxIndex(size_x/(px*py), 0);

	for (size_t i = 0; i < iter; ++i)
	{
		maxPoolReshape(y, maxPoolShaped, p_range, q_range, D, B, py, px, py, px);

		if (STANDALONE)
		{
			size_t size = (size_x/(px*py))*(px*py);
			vector<aramisSecretType> diff(size);

			for (size_t i = 0; i < (size_x/(px*py)); ++i)
			{
				act[i] = maxPoolShaped[i*(px*py)];
				maxIndex[i] = 0;
			}

			for (size_t i = 1; i < (px*py); ++i)
				for (size_t j = 0; j < (size_x/(px*py)); ++j)
				{
					if (maxPoolShaped[j*(px*py) + i] > act[j])
					{
						act[j] = maxPoolShaped[j*(px*py) + i];
						maxIndex[j] = i;
					}
				}
		}

		if (MPC)
			funcMaxMPC(maxPoolShaped, act, maxIndex, size_x/(px*py), px*py);
	}
}

void testMaxPoolDerivative(size_t p_range, 
		size_t q_range, 
		size_t px, 
		size_t py, 
		size_t D, 
		size_t iter)
{
	size_t B = MINI_BATCH_SIZE;
	size_t alpha_range = p_range/py;
	size_t beta_range = q_range/px;
	size_t size_y = (p_range*q_range*D*B);
	vector<aramisSecretType> deltaMaxPool(size_y, 0);
	vector<aramisSecretType> deltas(size_y/(px*py), 0);
	vector<aramisSecretType> maxIndex(size_y/(px*py), 0);

	size_t size_delta = alpha_range*beta_range*D*B;
	vector<aramisSecretType> thatMatrixTemp(size_y, 0), thatMatrix(size_y, 0);


	for (size_t i = 0; i < iter; ++i)
	{
		if (STANDALONE)
			for (size_t i = 0; i < size_delta; ++i)
				thatMatrixTemp[i*px*py + maxIndex[i]] = 1;

		if (MPC)
			funcMaxIndexMPC(thatMatrixTemp, maxIndex, size_delta, px*py);


		//Reshape thatMatrix
		size_t repeat_size = D*B;
		size_t alpha_offset, beta_offset, alpha, beta;
		for (size_t r = 0; r < repeat_size; ++r)
		{
			size_t size_temp = p_range*q_range;
			for (size_t i = 0; i < size_temp; ++i)
			{
				alpha = (i/(px*py*beta_range));
				beta = (i/(px*py)) % beta_range;
				alpha_offset = (i%(px*py))/px;
				beta_offset = (i%py);
				thatMatrix[((py*alpha + alpha_offset)*q_range) +
					(px*beta + beta_offset) + r*size_temp]
					= thatMatrixTemp[r*size_temp + i];
			}
		}

		//Replicate delta martix appropriately
		vector<aramisSecretType> largerDelta(size_y, 0);
		size_t index_larger, index_smaller;
		for (size_t r = 0; r < repeat_size; ++r)
		{
			size_t size_temp = p_range*q_range;
			for (size_t i = 0; i < size_temp; ++i)
			{
				index_smaller = r*size_temp/(px*py) + (i/(q_range*py))*beta_range + ((i%q_range)/px);
				index_larger = r*size_temp + (i/q_range)*q_range + (i%q_range);
				largerDelta[index_larger] = deltas[index_smaller];
			}
		}

		if (STANDALONE)
			for (size_t i = 0; i < size_y; ++i)
				deltaMaxPool[i] = largerDelta[i] * thatMatrix[i];

		if (MPC)
			funcDotProductMPC(largerDelta, thatMatrix, deltaMaxPool, size_y);
	}
}


