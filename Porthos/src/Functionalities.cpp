/*

Authors: Sameer Wagh, Mayank Rathee, Nishant Kumar.

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
#include <chrono>

using namespace std;
using namespace std::chrono;

extern CommunicationObject commObject;
extern vector<porthosSecretType*> toFreeMemoryLaterArr;

//For n-dimensional vector creation
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


/******************************** Functionalities 2PC ********************************/ 

void funcTruncateIdeal(vector<porthosSecretType>& vec, int sf, int size){
	assert(sf == FLOAT_PRECISION);
	if (partyNum == PARTY_A){
		sendVector<porthosSecretType>(vec, PARTY_B, size);
		for(int i=0;i<size;i++){
			vec[i] = 0;
		}
	}
	if (partyNum == PARTY_B){
		vector<porthosSecretType> recv_vec(size, 0);
		receiveVector<porthosSecretType>(recv_vec, PARTY_A, size);
		for(int i=0;i<size;i++){
			vec[i] = recv_vec[i] + vec[i];
			vec[i] = ((porthosSignedSecretType)vec[i])>>sf;
		}
	}
}

// Share Truncation, truncate shares of a by power (in place) (power is logarithmic)
void funcTruncate2PC(vector<porthosSecretType> &a, 
		size_t power, 
		size_t size)
{
	assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Truncate called by spurious parties");
	//funcTruncateIdeal(a, power, size);
	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			a[i] = static_cast<porthosSecretType>(static_cast<porthosSignedSecretType>(a[i]) >> power);

	if (partyNum == PARTY_B)
		for (size_t i = 0; i < size; ++i)
			a[i] = - static_cast<porthosSecretType>(static_cast<porthosSignedSecretType>(- a[i]) >> power);
}

void funcTruncate2PC(vector<vector<porthosSecretType>> &a, 
		size_t power, 
		size_t rows, 
		size_t cols)
{
	assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Truncate called by spurious parties");
	vector<porthosSecretType> flatten_vec(rows*cols, 0);
	flatten_2D_vector(a, flatten_vec, rows, cols);
	funcTruncate2PC(flatten_vec, power, rows*cols);
	deflatten_2D_vector(flatten_vec, a, rows, cols);
}

void funcTruncate2PC(vector<vector<vector<vector<vector<porthosSecretType>>>>> &a,
                size_t power,
                size_t d1,
                size_t d2,
                size_t d3,
                size_t d4,
                size_t d5)
{
        assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Truncate called by spurious parties");

        size_t size = d1*d2*d3*d4*d5;
        vector<porthosSecretType> flatten_vec(size, 0);
        flatten_5D_vector(a, flatten_vec, d1,d2,d3,d4,d5);
        funcTruncate2PC(flatten_vec, power, size);
        deflatten_5D_vector(flatten_vec, a, d1,d2,d3,d4,d5);
}

// XOR shares with a public bit into output.
void funcXORModuloOdd2PC(vector<smallType> &bit, 
		vector<porthosSecretType> &shares, 
		vector<porthosSecretType> &output, 
		size_t size)
{
	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (bit[i] == 1)
				output[i] = subtractModuloOdd<smallType, porthosSecretType>(1, shares[i]);
			else
				output[i] = shares[i];
		}
	}

	if (partyNum == PARTY_B)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (bit[i] == 1)
				output[i] = subtractModuloOdd<smallType, porthosSecretType>(0, shares[i]);
			else
				output[i] = shares[i];
		}
	}
}

//Reconstruct clear value by revealing shares to other parties
void funcReconstruct2PC(const vector<porthosSecretType> &a, 
		size_t size, 
		string str, 
		vector<porthosSecretType>* b=NULL, 
		int revealToParties = 2)
{
	assert((partyNum == PARTY_A or partyNum == PARTY_B) && "Reconstruct called by spurious parties");
	assert((revealToParties >= 1) && (revealToParties <= 3) && ("Reconstruct/Reveal bitmask should be between 1 and 3 inclusive."));

	vector<porthosSecretType> temp(size);
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
		sendVector<porthosSecretType>(a, partyToReceive, size);
		if (revealToParties == 3)
		{
			//bitmask = 11
			//Both parties are supposed to get output. Wait for reciver to send back results
			if (b)
			{
				receiveVector<porthosSecretType>((*b), partyToReceive, size);
			}
			else
			{
				receiveVector<porthosSecretType>(temp, partyToReceive, size);
			}
		}
	}

	if (partyNum == partyToReceive)
	{
		receiveVector<porthosSecretType>(temp, partyToSend, size);

		if (b)
		{
			addVectors<porthosSecretType>(temp, a, (*b), size);
			if (revealToParties == 3)
			{
				//bitmask = 11
				//Send the reconstructed vector to the other party
				sendVector<porthosSecretType>(*b, partyToSend, size);
			}
		}
		else
		{
			addVectors<porthosSecretType>(temp, a, temp, size);
			cout << str << ": ";
			for (size_t i = 0; i < size; ++i)
				print_linear(temp[i], DEBUG_PRINT);
			cout << endl;
			if (revealToParties == 3)
			{
				//bitmask = 11
				//Send the reconstructed vector to the other party
				sendVector<porthosSecretType>(temp, partyToSend, size);
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

		cout << str << ": ";
		for (size_t i = 0; i < size; ++i)
			cout << (int)temp[i] << " ";
		cout << endl;
	}
}

void funcConditionalSet2PC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<smallType> &c,
		vector<porthosSecretType> &u, 
		vector<porthosSecretType> &v, 
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
//Secret share a value between primary parties.
void funcSecretShareConstant(const vector<porthosSecretType> &cons, 
		vector<porthosSecretType> &curShare, 
		size_t size)
{
	if (PRIMARY)
	{
		populateRandomVector<porthosSecretType>(curShare, size, "COMMON", "NEGATIVE");
		if (partyNum == PARTY_A)
		{
			addVectors<porthosSecretType>(curShare, cons, curShare, size);
		}
	}
	else
	{
		fillVector<porthosSecretType>(0, curShare, size);
	}
}

/******************************** Main Functionalities MPC ********************************/

// Matrix Multiplication of a*b = c with transpose flags for a,b.
// Output is a share between PARTY_A and PARTY_B.
// a^transpose_a is rows*common_dim and b^transpose_b is common_dim*columns
void funcMatMulMPC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c,
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b, 
		uint32_t consSF,
		bool doTruncation)
{
	log_print("funcMatMulMPC");
#if (LOG_DEBUG)
	cout << "Rows, Common_dim, Columns: " << rows << "x" << common_dim << "x" << columns << endl;
#endif

	size_t size = rows*columns;
	size_t size_left = rows*common_dim;
	size_t size_right = common_dim*columns;
	vector<porthosSecretType> A(size_left, 0), B(size_right, 0), C(size, 0);

	if (HELPER)
	{
		vector<porthosSecretType> A1(size_left, 0), A2(size_left, 0),
			B1(size_right, 0), B2(size_right, 0),
			C1(size, 0), C2(size, 0);

		populateRandomVector<porthosSecretType>(A1, size_left, "a_1", "POSITIVE");
		populateRandomVector<porthosSecretType>(A2, size_left, "a_2", "POSITIVE");
		populateRandomVector<porthosSecretType>(B1, size_right, "b_1", "POSITIVE");
		populateRandomVector<porthosSecretType>(B2, size_right, "b_2", "POSITIVE");
		populateRandomVector<porthosSecretType>(C1, size, "c_1", "POSITIVE");

		addVectors<porthosSecretType>(A1, A2, A, size_left);
		addVectors<porthosSecretType>(B1, B2, B, size_right);

		matrixMultEigen(A, B, C, rows, common_dim, columns, 0, 0);
		subtractVectors<porthosSecretType>(C, C1, C2, size);

#if (LOG_LAYERWISE)
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
#endif
		sendVector<porthosSecretType>(C2, PARTY_B, size);
#if (LOG_LAYERWISE)
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		auto tt = time_span.count();
		commObject.timeMatmul[0] += tt;
#endif
	}

	if (PRIMARY)
	{
		vector<porthosSecretType> E(size_left), F(size_right);
		vector<porthosSecretType> temp_E(size_left), temp_F(size_right);
		vector<porthosSecretType> temp_c(size);

		if (partyNum == PARTY_A)
		{
			populateRandomVector<porthosSecretType>(A, size_left, "a_1", "POSITIVE");
			populateRandomVector<porthosSecretType>(B, size_right, "b_1", "POSITIVE");
		}

		if (partyNum == PARTY_B)
		{
			populateRandomVector<porthosSecretType>(A, size_left, "a_2", "POSITIVE");
			populateRandomVector<porthosSecretType>(B, size_right, "b_2", "POSITIVE");
		}

		subtractVectors<porthosSecretType>(a, A, E, size_left);
		subtractVectors<porthosSecretType>(b, B, F, size_right);
#if (LOG_LAYERWISE)
		auto t1 = high_resolution_clock::now();
#endif

#ifdef PARALLEL_COMM
		thread *threads = new thread[2];

		threads[0] = thread(sendTwoVectors<porthosSecretType>, ref(E), ref(F), adversary(partyNum), size_left, size_right);
		threads[1] = thread(receiveTwoVectors<porthosSecretType>, ref(temp_E), ref(temp_F), adversary(partyNum), size_left, size_right);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		delete[] threads;
#else
		if (partyNum == PARTY_A){
			sendTwoVectors<porthosSecretType>(ref(E), ref(F), adversary(partyNum), size_left, size_right);
			receiveTwoVectors<porthosSecretType>(ref(temp_E), ref(temp_F), adversary(partyNum), size_left, size_right);
		}
		else if (partyNum == PARTY_B){
			receiveTwoVectors<porthosSecretType>(ref(temp_E), ref(temp_F), adversary(partyNum), size_left, size_right);
			sendTwoVectors<porthosSecretType>(ref(E), ref(F), adversary(partyNum), size_left, size_right);
		}
#endif

#if (LOG_LAYERWISE)
		auto t2 = high_resolution_clock::now();
		auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
		commObject.timeMatmul[0] += tt;
#endif

		addVectors<porthosSecretType>(E, temp_E, E, size_left);
		addVectors<porthosSecretType>(F, temp_F, F, size_right);

		if (partyNum == PARTY_A)
		{
			subtractVectors<porthosSecretType>(a, E, A, size_left);
#if (LOG_LAYERWISE)
			auto t1 = high_resolution_clock::now();
#endif
			matrixMultEigen(A, F, c, rows, common_dim, columns, 0, 0);
			matrixMultEigen(E, b, temp_c, rows, common_dim, columns, 0, 0);
#if (LOG_LAYERWISE)
			auto t2 = high_resolution_clock::now();
			auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
			cout<<"funcMatMulMPC : Local eigen matmuls took "<<tt<<" seconds."<<endl;
#endif
		}
		else
		{
#if (LOG_LAYERWISE)
			auto t1 = high_resolution_clock::now();
#endif
			matrixMultEigen(a, F, c, rows, common_dim, columns, 0, 0);
			matrixMultEigen(E, b, temp_c, rows, common_dim, columns, 0, 0);
#if (LOG_LAYERWISE)
			auto t2 = high_resolution_clock::now();
			auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
			cout<<"funcMatMulMPC : Local eigen matmuls took "<<tt<<" seconds."<<endl;
#endif
		}

		addVectors<porthosSecretType>(c, temp_c, c, size);

		if (partyNum == PARTY_A)
		{
			populateRandomVector<porthosSecretType>(C, size, "c_1", "POSITIVE");
		}
		else if (partyNum == PARTY_B)
		{
			//Receive C1 from P2 after E and F have been revealed.
#if (LOG_LAYERWISE)
			auto t1 = high_resolution_clock::now();
#endif
			receiveVector<porthosSecretType>(C, PARTY_C, size);
#if (LOG_LAYERWISE)
			auto t2 = high_resolution_clock::now();
			auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
			commObject.timeMatmul[0] += tt;
#endif
		}

		addVectors<porthosSecretType>(c, C, c, size);

		if (doTruncation){
			assert(FLOAT_PRECISION == consSF && "Please correct FLOAT_PRECISION value to be equal to consSF");
			funcTruncate2PC(c, consSF, size);
		}
	}
}


void funcDotProductMPC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b,
		vector<porthosSecretType> &c, 
		size_t size,
		uint32_t consSF,
		bool doTruncation)
{
	log_print("funcDotProductMPC");

	vector<porthosSecretType> A(size, 0), B(size, 0), C(size, 0);

	if (HELPER)
	{
		vector<porthosSecretType> A1(size, 0), A2(size, 0),
			B1(size, 0), B2(size, 0),
			C1(size, 0), C2(size, 0);

		populateRandomVector<porthosSecretType>(A1, size, "a_1", "POSITIVE");
		populateRandomVector<porthosSecretType>(A2, size, "a_2", "POSITIVE");
		populateRandomVector<porthosSecretType>(B1, size, "b_1", "POSITIVE");
		populateRandomVector<porthosSecretType>(B2, size, "b_2", "POSITIVE");
		populateRandomVector<porthosSecretType>(C1, size, "c_1", "POSITIVE");

		addVectors<porthosSecretType>(A1, A2, A, size);
		addVectors<porthosSecretType>(B1, B2, B, size);

		for (size_t i = 0; i < size; ++i)
			C[i] = A[i] * B[i];

		// splitIntoShares(C, C1, C2, size);
		subtractVectors<porthosSecretType>(C, C1, C2, size);
		sendVector<porthosSecretType>(C2, PARTY_B, size);
	}

	if (PRIMARY)
	{
		if (partyNum == PARTY_A)
		{
			populateRandomVector<porthosSecretType>(A, size, "a_1", "POSITIVE");
			populateRandomVector<porthosSecretType>(B, size, "b_1", "POSITIVE");
		}

		if (partyNum == PARTY_B)
		{
			populateRandomVector<porthosSecretType>(A, size, "a_2", "POSITIVE");
			populateRandomVector<porthosSecretType>(B, size, "b_2", "POSITIVE");
		}

		vector<porthosSecretType> E(size), F(size), temp_E(size), temp_F(size);
		porthosSecretType temp;

		subtractVectors<porthosSecretType>(a, A, E, size);
		subtractVectors<porthosSecretType>(b, B, F, size);

#ifdef PARALLEL_COMM
		thread *threads = new thread[2];

		threads[0] = thread(sendTwoVectors<porthosSecretType>, ref(E), ref(F), adversary(partyNum), size, size);
		threads[1] = thread(receiveTwoVectors<porthosSecretType>, ref(temp_E), ref(temp_F), adversary(partyNum), size, size);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		delete[] threads;
#else
		if (partyNum == PARTY_A){
			sendTwoVectors<porthosSecretType>(ref(E), ref(F), adversary(partyNum), size, size);
			receiveTwoVectors<porthosSecretType>(ref(temp_E), ref(temp_F), adversary(partyNum), size, size);
		}
		else if (partyNum == PARTY_B){
			receiveTwoVectors<porthosSecretType>(ref(temp_E), ref(temp_F), adversary(partyNum), size, size);
			sendTwoVectors<porthosSecretType>(ref(E), ref(F), adversary(partyNum), size, size);
		}
#endif
		addVectors<porthosSecretType>(E, temp_E, E, size);
		addVectors<porthosSecretType>(F, temp_F, F, size);

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

		if (partyNum == PARTY_A)
			populateRandomVector<porthosSecretType>(C, size, "c_1", "POSITIVE");
		else if (partyNum == PARTY_B)
			receiveVector<porthosSecretType>(C, PARTY_C, size);

		addVectors<porthosSecretType>(c, C, c, size);

		if (doTruncation){
			assert(FLOAT_PRECISION == consSF && "Please correct FLOAT_PRECISION value to be equal to consSF");
			funcTruncate2PC(c, consSF, size);
		}
	}
#ifdef VERIFYLAYERWISE

		if (partyNum == PARTY_A){
				sendVector<porthosSecretType>(a, PARTY_B, size);
				sendVector<porthosSecretType>(b, PARTY_B, size);
				sendVector<porthosSecretType>(c, PARTY_B, size);
		}
		if (partyNum == PARTY_B){
				auto aOther = make_vector<porthosSecretType>(size);
				auto bOther = make_vector<porthosSecretType>(size);
				auto localAns = make_vector<porthosSecretType>(size);
				auto cOther = make_vector<porthosSecretType>(size);

				receiveVector<porthosSecretType>(aOther, PARTY_A, size);
				receiveVector<porthosSecretType>(bOther, PARTY_A, size);
				receiveVector<porthosSecretType>(cOther, PARTY_A, size);

				for(int i=0;i<size;i++){
						aOther[i] = a[i] + aOther[i];
						bOther[i] = b[i] + bOther[i];
						cOther[i] = c[i] + cOther[i];

						localAns[i] = ((porthosSignedSecretType)(aOther[i]*bOther[i]))>>consSF;
				}

				static int ctr = 1;
				bool pass = true;
				for(int i1=0;i1<size;i1++){
						if (localAns[i1] != cOther[i1]){
								std::cerr<<RED<<"ERROR in dotProd #"<<ctr<<" "<<i1<<" "<<localAns[i1]<<" "<<cOther[i1]<<RESET<<std::endl;
								pass = false;
						}
				}

				if (!pass){
						std::cerr<<RED<<"There was an error in dotProd#"<<ctr<<RESET<<std::endl;
				}
				else{
						std::cout<<GREEN<<"Executed dotProd#"<<ctr<<" correctly."<<RESET<<std::endl;
				}
				ctr++;
		}

#endif
}

//Multithreaded function for parallel private compare
void parallelPC(smallType* c, 
		size_t start, 
		size_t end, 
		int t,
		const smallType* share_m, 
		const porthosSecretType* r,
		const smallType* beta, 
		const smallType* betaPrime, 
		size_t dim)
{
	size_t index3, index2;
	size_t PARTY;

	smallType bit_r, a, tempM;
	porthosSecretType valueX;

	thread_local int shuffle_counter = 0;
	thread_local int nonZero_counter = 0;

	//Check the security of the first if condition
	for (size_t index2 = start; index2 < end; ++index2)
	{
		if (beta[index2] == 1 and r[index2] != MINUS_ONE)
			valueX = r[index2] + 1;
		else
			valueX = r[index2];

		if (beta[index2] == 1 and r[index2] == MINUS_ONE)
		{
			//This assert hits with a negligible probability. If it does
			//change #define PARALLEL true to false
			//This will make Porthos system use a single threaded version
			assert(false && "aes_common object isn't thread safe currently");

#ifdef PRECOMPUTEAES
			aes_common->fillWithRandomModuloPrimeBits(c + index2*dim, dim);
#else
			for (size_t k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				c[index3] = aes_common->randModPrime();
			}
#endif
			for (size_t k = 0; k < dim; ++k)
			{
				index3 = index2*dim + k;
				if (partyNum == PARTY_A)
					c[index3] = subtractModPrime((k!=0), c[index3]);

				c[index3] = multiplyModPrime(c[index3], aes_parallel->randNonZeroModPrime(t, nonZero_counter)); //Use common randomness for P0 and P1 here.
			}
		}
		else
		{
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

				c[index3] = multiplyModPrime(c[index3], aes_parallel->randNonZeroModPrime(t, nonZero_counter)); //Use common randomness for P0 and P1 here.
			}
		}
		aes_parallel->AES_random_shuffle(c, index2*dim, (index2+1)*dim, t, shuffle_counter); //Use common randomness for P0 and P1 here.
	}
	aes_parallel->counterIncrement();
}

//Multithreaded function called by PARTY_C to check
//for 0s in an a vector.
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


// Private Compare functionality
void funcPrivateCompareMPC(const vector<smallType> &share_m, 
		const vector<porthosSecretType> &r,
		const vector<smallType> &beta, 
		vector<smallType> &betaPrime,
		size_t size, 
		size_t dim)
{
	log_print("funcPrivateCompareMPC");

#ifdef DEBUG
	assert(dim == BIT_SIZE && "Private Compare assert issue");
#endif
	size_t sizeLong = size*dim;
	size_t index3, index2;

	if (PRIMARY)
	{
		smallType bit_r, a, tempM;
		vector<smallType> c(sizeLong);
		porthosSecretType valueX;

		if (PARALLEL)
		{
			thread *threads = new thread[NO_CORES];
			int chunksize = size/NO_CORES;
			for (int i = 0; i < NO_CORES; i++)
			{
				int start = i*chunksize;
				int end = (i+1)*chunksize;
				if (i == NO_CORES - 1)
					end = size;

				threads[i] = thread(parallelPC, c.data(), start, end, i, share_m.data(), r.data(), beta.data(), betaPrime.data(), dim);
			}
			for (int i = 0; i < NO_CORES; i++)
				threads[i].join();
			delete[] threads;
		}
		else
		{
			//Run single threaded version
			for (size_t index2 = 0; index2 < size; ++index2)
			{
				if (beta[index2] == 1 and r[index2] != MINUS_ONE)
					valueX = r[index2] + 1;
				else
					valueX = r[index2];

				if (beta[index2] == 1 and r[index2] == MINUS_ONE)
				{
#ifdef PRECOMPUTE_AES
					aes_common->fillWithRandomModuloPrimeBits(c.data() + index2*dim, dim);
#else
					for (size_t k = 0; k < dim; ++k)
					{
						index3 = index2*dim + k;
						c[index3] = aes_common->randModPrime();
					}
#endif
					for (size_t k = 0; k < dim; ++k)
					{
						index3 = index2*dim + k;
						if (partyNum == PARTY_A)
							c[index3] = subtractModPrime((k!=0), c[index3]);

						c[index3] = multiplyModPrime(c[index3], aes_common->randNonZeroModPrime());
					}
				}
				else
				{
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
		sendVector<smallType>(c, PARTY_C, sizeLong);
	}

	if (partyNum == PARTY_C)
	{
		vector<smallType> c1(sizeLong);
		vector<smallType> c2(sizeLong);


		receiveVector<smallType>(c1, PARTY_A, sizeLong);
		receiveVector<smallType>(c2, PARTY_B, sizeLong);


#ifdef PARALLIZE_CRITICAL
		thread *threads = new thread[NO_CORES];
		int chunksize = size/NO_CORES;

		for (int i = 0; i < NO_CORES; i++)
		{
			int start = i*chunksize;
			int end = (i+1)*chunksize;
			if (i == NO_CORES - 1)
				end = size;

			threads[i] = thread(deduceIfAnyZeroPC, c1.data(), c2.data(), start, end, dim, betaPrime.data());
		}

		for (int i = 0; i < NO_CORES; i++)
			threads[i].join();

		delete[] threads;

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
void funcShareConvertMPC(vector<porthosSecretType> &a, 
		size_t size)
{
	log_print("funcShareConvertMPC");

	vector<porthosSecretType> r;
	vector<smallType> etaDP;
	vector<smallType> alpha;
	vector<smallType> betai;
	vector<smallType> bit_shares;
	vector<porthosSecretType> delta_shares;
	vector<smallType> etaP;
	vector<porthosSecretType> eta_shares;
	vector<porthosSecretType> theta_shares;

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

	if (PRIMARY)
	{
		vector<porthosSecretType> r1(size);
		vector<porthosSecretType> r2(size);
		vector<porthosSecretType> a_tilde(size);

		populateRandomVector<porthosSecretType>(r1, size, "COMMON", "POSITIVE");
		populateRandomVector<porthosSecretType>(r2, size, "COMMON", "POSITIVE");
		addVectors<porthosSecretType>(r1, r2, r, size);

		if (partyNum == PARTY_A)
			wrapAround(r1, r2, alpha, size);

		if (partyNum == PARTY_A)
		{
			addVectors<porthosSecretType>(a, r1, a_tilde, size);
			wrapAround(a, r1, betai, size);
		}
		if (partyNum == PARTY_B)
		{
			addVectors<porthosSecretType>(a, r2, a_tilde, size);
			wrapAround(a, r2, betai, size);
		}

		populateBitsVector(etaDP, "COMMON", size);
		sendVector<porthosSecretType>(a_tilde, PARTY_C, size);
	}

#ifndef RUN_SHARECONV_OPTI
	
	if (partyNum == PARTY_C)
	{
		vector<porthosSecretType> x(size);
		vector<smallType> delta(size);
		vector<porthosSecretType> a_tilde_1(size);
		vector<porthosSecretType> a_tilde_2(size);
		vector<smallType> bit_shares_x_1(size*BIT_SIZE);
		vector<smallType> bit_shares_x_2(size*BIT_SIZE);
		vector<porthosSecretType> delta_shares_1(size);
		vector<porthosSecretType> delta_shares_2(size);

		receiveVector<porthosSecretType>(a_tilde_1, PARTY_A, size);
		receiveVector<porthosSecretType>(a_tilde_2, PARTY_B, size);

		addVectors<porthosSecretType>(a_tilde_1, a_tilde_2, x, size);
		sharesOfBits(bit_shares_x_1, bit_shares_x_2, x, size, "INDEP");

		sendVector<smallType>(bit_shares_x_1, PARTY_A, size*BIT_SIZE);
		sendVector<smallType>(bit_shares_x_2, PARTY_B, size*BIT_SIZE);
		wrapAround(a_tilde_1, a_tilde_2, delta, size);
		sharesModuloOdd<smallType>(delta_shares_1, delta_shares_2, delta, size, "INDEP");
		sendVector<porthosSecretType>(delta_shares_1, PARTY_A, size);
		sendVector<porthosSecretType>(delta_shares_2, PARTY_B, size);
	}

	else if (PRIMARY)
	{
		receiveVector<smallType>(bit_shares, PARTY_C, size*BIT_SIZE);
		receiveVector<porthosSecretType>(delta_shares, PARTY_C, size);
	}

#else //Use share convert optimization
	if (partyNum == PARTY_C)
	{
		vector<porthosSecretType> x(size);
		vector<smallType> delta(size);
		vector<porthosSecretType> a_tilde_1(size);
		vector<porthosSecretType> a_tilde_2(size);
		vector<smallType> bit_shares_x_1(size*BIT_SIZE);
		vector<smallType> bit_shares_x_2(size*BIT_SIZE);
		vector<porthosSecretType> delta_shares_1(size);
		vector<porthosSecretType> delta_shares_2(size);

		receiveVector<porthosSecretType>(a_tilde_1, PARTY_A, size);
		receiveVector<porthosSecretType>(a_tilde_2, PARTY_B, size);

		addVectors<porthosSecretType>(a_tilde_1, a_tilde_2, x, size);
		sharesOfBits(bit_shares_x_1, bit_shares_x_2, x, size, "PRG_COMM_OPTI");

		//Send first half of x2 to Party B and second half of x1 to party A.
#ifdef PARALLEL_COMM
		thread *threads_a = new thread[2];
		threads_a[0] = thread(sendArr<smallType>, bit_shares_x_1.data() + (size/2)*BIT_SIZE , PARTY_A, (size - (size/2))*BIT_SIZE);	
		threads_a[1] = thread(sendArr<smallType>, bit_shares_x_2.data(), PARTY_B, (size/2)*BIT_SIZE);

		for(int th=0; th<2; th++){
			threads_a[th].join();
		}
		delete[] threads_a;
#else
		sendArr<smallType>(bit_shares_x_1.data() + (size/2)*BIT_SIZE, PARTY_A, (size - (size/2))*BIT_SIZE);
		sendArr<smallType>(bit_shares_x_2.data(), PARTY_B, (size/2)*BIT_SIZE);
#endif

		wrapAround(a_tilde_1, a_tilde_2, delta, size);
		sharesModuloOdd<smallType>(delta_shares_1, delta_shares_2, delta, size, "PRG_COMM_OPTI");

#ifdef PARALLEL_COMM
		thread *threads_b = new thread[2];
		threads_b[0] = thread(sendArr<porthosSecretType>, delta_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		threads_b[1] = thread(sendArr<porthosSecretType>, delta_shares_2.data(), PARTY_B, (size/2));

		for(int th=0; th<2; th++){
			threads_b[th].join();
		}
		delete[] threads_b;
#else
		sendArr<porthosSecretType>(delta_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		sendArr<porthosSecretType>(delta_shares_2.data(), PARTY_B, (size/2));
#endif

	}
	else if (PRIMARY)
	{
		size_t localStart, localEnd, receiveStart, receiveEnd; //start is inclusive and end is exclusive
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
#ifdef PRECOMPUTEAES
		aesObjectForBitShares->fillWithRandomModuloPrimeBits(bit_shares.data() + localStart*BIT_SIZE, (localEnd - localStart)*BIT_SIZE);
		aesObjectForDeltaShares->fillWithRandomModuloOddBits(delta_shares.data() + localStart, (localEnd - localStart));
#else
		for(size_t i=localStart; i<localEnd; i++)
		{
			for(size_t j=0; j<BIT_SIZE; j++)
			{
				bit_shares[i*BIT_SIZE + j] = aesObjectForBitShares->randModPrime();
			}
		}

		for(size_t i=localStart; i<localEnd; i++)
		{
			delta_shares[i] = aesObjectForDeltaShares->randModuloOdd();
		}
#endif

		//Now get the remaining bits from P2.
		receiveArr<smallType>(bit_shares.data() + receiveStart*BIT_SIZE, PARTY_C, (receiveEnd - receiveStart)*BIT_SIZE);
		receiveArr<porthosSecretType>(delta_shares.data() + receiveStart, PARTY_C, (receiveEnd - receiveStart));
	}
#endif //end of share convert optimization

	if (PRIMARY)
	{
		for(size_t i=0;i<size;i++)
		{
			r[i] = r[i] - 1ULL;
		}
	}

	funcPrivateCompareMPC(bit_shares, r, etaDP, etaP, size, BIT_SIZE);

	if (partyNum == PARTY_C)
	{
		vector<porthosSecretType> eta_shares_1(size);
		vector<porthosSecretType> eta_shares_2(size);

		sharesModuloOdd<smallType>(eta_shares_1, eta_shares_2, etaP, size, "INDEP");
		sendVector<porthosSecretType>(eta_shares_1, PARTY_A, size);
		sendVector<porthosSecretType>(eta_shares_2, PARTY_B, size);
	}

	if (PRIMARY)
	{
		receiveVector<porthosSecretType>(eta_shares, PARTY_C, size);
		funcXORModuloOdd2PC(etaDP, eta_shares, theta_shares, size);
		addModuloOdd<porthosSecretType, smallType>(theta_shares, betai, theta_shares, size);
		addModuloOdd<porthosSecretType, porthosSecretType>(theta_shares, delta_shares, theta_shares, size);	

		if (partyNum == PARTY_A){
			for(size_t i=0; i<size; i++){
				alpha[i] = alpha[i] + 1;
			}	
			subtractModuloOdd<porthosSecretType, smallType>(theta_shares, alpha, theta_shares, size);
		}

		subtractModuloOdd<porthosSecretType, porthosSecretType>(a, theta_shares, a, size);
	}
}

//Compute MSB of a and store it in b
//3PC: output is shares of MSB in \Z_L
void funcComputeMSB3PC(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &b, 
		size_t size)
{
	log_print("funcComputeMSB3PC");
#ifdef DEBUG
	assert(THREE_PC && "funcComputeMSB3PC called in non-3PC mode");
#endif

	vector<porthosSecretType> ri;
	vector<smallType> bit_shares;
	vector<porthosSecretType> LSB_shares;
	vector<smallType> beta;
	vector<porthosSecretType> c;
	vector<smallType> betaP;
	vector<porthosSecretType> theta_shares;

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

	//ComputeMSB Code part 1: contains part where P0 and P1 receive bit_shares, lsb_shares, r.
#ifndef RUN_MSB_OPTI //Code without compute msb optimization

	if (partyNum == PARTY_C)
	{
		vector<porthosSecretType> r1(size);
		vector<porthosSecretType> r2(size);
		vector<porthosSecretType> r(size);
		vector<smallType> bit_shares_r_1(size*BIT_SIZE);
		vector<smallType> bit_shares_r_2(size*BIT_SIZE);
		vector<porthosSecretType> LSB_shares_1(size);
		vector<porthosSecretType> LSB_shares_2(size);

#ifdef PRECOMPUTEAES
		aes_indep->fillWithRandomModuloOddBits(r1.data(), size);
		aes_indep->fillWithRandomModuloOddBits(r2.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
		{
			r1[i] = aes_indep->randModuloOdd();
		}
		for (size_t i = 0; i < size; ++i)
		{
			r2[i] = aes_indep->randModuloOdd();
		}
#endif

		addModuloOdd<porthosSecretType, porthosSecretType>(r1, r2, r, size);
		sharesOfBits(bit_shares_r_1, bit_shares_r_2, r, size, "INDEP");
		sendVector<smallType>(bit_shares_r_1, PARTY_A, size*BIT_SIZE);
		sendVector<smallType>(bit_shares_r_2, PARTY_B, size*BIT_SIZE);

		sharesOfLSB(LSB_shares_1, LSB_shares_2, r, size, "INDEP");
		sendTwoVectors<porthosSecretType>(r1, LSB_shares_1, PARTY_A, size, size);
		sendTwoVectors<porthosSecretType>(r2, LSB_shares_2, PARTY_B, size, size);
	}

	else if (PRIMARY)
	{
		receiveVector<smallType>(bit_shares, PARTY_C, size*BIT_SIZE);
		receiveTwoVectors<porthosSecretType>(ri, LSB_shares, PARTY_C, size, size);
	}

#else //Begin of Compute MSB optimization 1

	if (partyNum == PARTY_C)
	{
		vector<porthosSecretType> r1(size);
		vector<porthosSecretType> r2(size);
		vector<porthosSecretType> r(size);
		vector<smallType> bit_shares_r_1(size*BIT_SIZE);
		vector<smallType> bit_shares_r_2(size*BIT_SIZE);
		vector<porthosSecretType> LSB_shares_1(size);
		vector<porthosSecretType> LSB_shares_2(size);

#ifdef PRECOMPUTEAES
		aes_share_conv_shares_mod_odd_p0_p2->fillWithRandomModuloOddBits(r1.data(), size);
		aes_share_conv_shares_mod_odd_p1_p2->fillWithRandomModuloOddBits(r2.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
		{
			r1[i] = aes_share_conv_shares_mod_odd_p0_p2->randModuloOdd();
		}
		for (size_t i = 0; i < size; ++i)
		{
			r2[i] = aes_share_conv_shares_mod_odd_p1_p2->randModuloOdd();
		}
#endif
		// Now r vector is not even required to be sent.
		addModuloOdd<porthosSecretType, porthosSecretType>(r1, r2, r, size);
		sharesOfBits(bit_shares_r_1, bit_shares_r_2, r, size, "PRG_COMM_OPTI");

#ifdef PARALLEL_COMM
		thread *threads_a = new thread[2];
		threads_a[0] = thread(sendArr<smallType>, bit_shares_r_1.data() + (size/2)*BIT_SIZE, PARTY_A, (size - (size/2))*BIT_SIZE);
		threads_a[1] = thread(sendArr<smallType>, bit_shares_r_2.data(), PARTY_B, (size/2)*BIT_SIZE);

		for(int th=0; th<2; th++){
			threads_a[th].join();
		}
		delete[] threads_a;
#else
		sendArr<smallType>(bit_shares_r_1.data() + (size/2)*BIT_SIZE, PARTY_A, (size - (size/2))*BIT_SIZE);
		sendArr<smallType>(bit_shares_r_2.data(), PARTY_B, (size/2)*BIT_SIZE);
#endif

		sharesOfLSB(LSB_shares_1, LSB_shares_2, r, size, "PRG_COMM_OPTI");

#ifdef PARALLEL_COMM
		thread *threads_b = new thread[2];
		threads_b[0] = thread(sendArr<porthosSecretType>, LSB_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		threads_b[1] = thread(sendArr<porthosSecretType>, LSB_shares_2.data(), PARTY_B, (size/2));
		for(int th=0; th<2; th++){
			threads_b[th].join();
		}
		delete[] threads_b;
#else
		sendArr<porthosSecretType>(LSB_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		sendArr<porthosSecretType>(LSB_shares_2.data(), PARTY_B, (size/2));
#endif

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
#ifdef PRECOMPUTEAES
		aesObjectForRi->fillWithRandomModuloOddBits(ri.data(), size);
#else
		for (size_t i = 0; i < size; ++i)
		{
			ri[i] = aesObjectForRi->randModuloOdd();
		}
#endif

		//Also fill what you can fill locally for bit_shares and LSB_shares.
		//Then wait on P2 to get the other half.
#ifdef PRECOMPUTEAES
		aesObjectForBitShares->fillWithRandomModuloPrimeBits(bit_shares.data() + localStart*BIT_SIZE, (localEnd - localStart)*BIT_SIZE);
		aesObjectForLSBShares->fillWithRandomBits64(LSB_shares.data() + localStart, localEnd - localStart);
#else
		for(size_t i=localStart; i<localEnd; i++)
		{
			for(size_t j=0; j<BIT_SIZE; j++)
			{
				bit_shares[i*BIT_SIZE + j] = aesObjectForBitShares->randModPrime();
			}
		}

		for(size_t i=localStart; i<localEnd; i++)
		{
			LSB_shares[i] = aesObjectForLSBShares->get64Bits();
		}
#endif

		//Now that all local computation is done, wait on p2 to get the remaining half
		receiveArr<smallType>(bit_shares.data() + receiveStart*BIT_SIZE, PARTY_C, (receiveEnd - receiveStart)*BIT_SIZE);
		receiveArr<porthosSecretType>(LSB_shares.data() + receiveStart, PARTY_C, receiveEnd - receiveStart);
	}

#endif //End compute msb optimization 1


	if (PRIMARY)
	{
		vector<porthosSecretType> temp(size);
		addModuloOdd<porthosSecretType, porthosSecretType>(a, a, c, size);
		addModuloOdd<porthosSecretType, porthosSecretType>(c, ri, c, size);

#ifdef PARALLEL_COMM
		thread *threads = new thread[2];

		threads[0] = thread(sendVector<porthosSecretType>, ref(c), adversary(partyNum), size);
		threads[1] = thread(receiveVector<porthosSecretType>, ref(temp), adversary(partyNum), size);

		for (int i = 0; i < 2; i++)
			threads[i].join();

		delete[] threads;
#else
		if (partyNum == PARTY_A){
			sendVector<porthosSecretType>(ref(c), adversary(partyNum), size);
			receiveVector<porthosSecretType>(ref(temp), adversary(partyNum), size);
		}
		else if (partyNum == PARTY_B){
			receiveVector<porthosSecretType>(ref(temp), adversary(partyNum), size);
			sendVector<porthosSecretType>(ref(c), adversary(partyNum), size);
		}
#endif
		addModuloOdd<porthosSecretType, porthosSecretType>(c, temp, c, size);
		populateBitsVector(beta, "COMMON", size);
	}

	funcPrivateCompareMPC(bit_shares, c, beta, betaP, size, BIT_SIZE);

	//Code part 2 - involves sending of shares of betaP from P2 to P0 and P1.
#ifndef RUN_MSB_OPTI //Code without compute msb optimization

	if (partyNum == PARTY_C)
	{
		vector<porthosSecretType> theta_shares_1(size);
		vector<porthosSecretType> theta_shares_2(size);

		sharesOfBitVector(theta_shares_1, theta_shares_2, betaP, size, "INDEP");
		sendVector<porthosSecretType>(theta_shares_1, PARTY_A, size);
		sendVector<porthosSecretType>(theta_shares_2, PARTY_B, size);
	}
	else if (PRIMARY)
	{
		if(partyNum == PARTY_A)
		{
			receiveVector<porthosSecretType>(theta_shares, PARTY_C, size);
		}
		else if(partyNum == PARTY_B)
		{
			receiveVector<porthosSecretType>(theta_shares, PARTY_C, size);
		}
	}

#else //Begin compute msb optimization 2

	if (partyNum == PARTY_C)
	{
		vector<porthosSecretType> theta_shares_1(size);
		vector<porthosSecretType> theta_shares_2(size);

		sharesOfBitVector(theta_shares_1, theta_shares_2, betaP, size, "PRG_COMM_OPTI");

#ifdef PARALLEL_COMM
		thread *threads_c = new thread[2];

		threads_c[0] = thread(sendArr<porthosSecretType>, theta_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		threads_c[1] = thread(sendArr<porthosSecretType>, theta_shares_2.data(), PARTY_B, (size/2));

		for(int th=0; th<2; th++){
			threads_c[th].join();
		}

		delete[] threads_c;
#else
		sendArr<porthosSecretType>(theta_shares_1.data() + (size/2), PARTY_A, (size - (size/2)));
		sendArr<porthosSecretType>(theta_shares_2.data(), PARTY_B, (size/2));
#endif

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

#ifdef PRECOMPUTEAES
		aesObjectForBitShares->fillWithRandomBits64(theta_shares.data() + localStart, (localEnd - localStart));
#else
		for(size_t i=localStart; i<localEnd; i++)
		{
			theta_shares[i] = aesObjectForBitShares->get64Bits();
		}
#endif

		//Now receive remaining from P2
		receiveArr<porthosSecretType>(theta_shares.data() + receiveStart, PARTY_C, (receiveEnd - receiveStart));
	}

#endif //End compute msb optimization 2

	if (PRIMARY)
	{
		// theta_shares is the same as gamma (in older versions);
		// LSB_shares is the same as delta (in older versions);

		porthosSecretType j = 0;
		if (partyNum == PARTY_A)
			j = floatToMyType(1);

		for (size_t i = 0; i < size; ++i)
			theta_shares[i] = (1 - 2*beta[i])*theta_shares[i] + j*beta[i];

		for (size_t i = 0; i < size; ++i)
			LSB_shares[i] = (1 - 2*(c[i] & 1))*LSB_shares[i] + j*(c[i] & 1);
	}


	vector<porthosSecretType> prod(size), temp(size);
	funcDotProductMPC(theta_shares, LSB_shares, prod, size);

	if (PRIMARY)
	{
		populateRandomVector<porthosSecretType>(temp, size, "COMMON", "NEGATIVE");
		for (size_t i = 0; i < size; ++i)
			b[i] = theta_shares[i] + LSB_shares[i] - 2*prod[i] + temp[i];
	}
}

// 3PC SelectShares: c contains shares of selector bit (encoded in porthosSecretType).
// a,b,c are shared across PARTY_A, PARTY_B
void funcSelectShares3PC(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b,
		vector<porthosSecretType> &c, 
		size_t size)
{
	log_print("funcSelectShares3PC");

	assert(THREE_PC && "funcSelectShares3PC called in non-3PC mdoe");
	funcDotProductMPC(a, b, c, size);
}

// 3PC: PARTY_A, PARTY_B hold shares in a, want shares of RELU' in b.
void funcRELUPrime3PC(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &b, 
		size_t size)
{
	log_print("funcRELUPrime3PC");
	assert(THREE_PC && "funcRELUPrime3PC called in non-3PC mode");

	vector<porthosSecretType> twoA(size, 0);
	porthosSecretType j = 0;

	for (size_t i = 0; i < size; ++i)
		twoA[i] = (a[i] * 2);	

#ifdef VERIFYLAYERWISE
	if (partyNum == PARTY_A){
		sendVector<porthosSecretType>(twoA, PARTY_B, size);
	}
	if (partyNum == PARTY_B){
		auto inOther = make_vector<porthosSecretType>(size);
	
		receiveVector<porthosSecretType>(inOther, PARTY_A, size);

		for(int i=0;i<size;i++){
			inOther[i] = twoA[i] + inOther[i];
		}
	}	
#endif	
	funcShareConvertMPC(twoA, size);
#ifdef VERIFYLAYERWISE
	if (partyNum == PARTY_A){
		sendVector<porthosSecretType>(twoA, PARTY_B, size);
	}
	if (partyNum == PARTY_B){
		auto aOther = make_vector<porthosSecretType>(size);
	
		receiveVector<porthosSecretType>(aOther, PARTY_A, size);
		addModuloOdd<porthosSecretType, porthosSecretType>(aOther, twoA, aOther, size);
	}
#endif	
	funcComputeMSB3PC(twoA, b, size);
#ifdef VERIFYLAYERWISE
	if (partyNum == PARTY_A){
		sendVector<porthosSecretType>(b, PARTY_B, size);
	}
	if (partyNum == PARTY_B){
		auto bOther = make_vector<porthosSecretType>(size);
		//auto localAns = make_vector<porthosSecretType>(size);
	
		receiveVector<porthosSecretType>(bOther, PARTY_A, size);

		for(int i=0;i<size;i++){
			bOther[i] = b[i] + bOther[i];
		}
	}	
#endif
	if (partyNum == PARTY_A)
		j = floatToMyType(1);

	if (PRIMARY)
		for (size_t i = 0; i < size; ++i)
			b[i] = j - b[i];
}

//PARTY_A, PARTY_B hold shares in a, want shares of RELU in b.
void funcRELUMPC(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &b, 
		size_t size)
{
	log_print("funcRELUMPC");

	vector<porthosSecretType> reluPrime(size);

	funcRELUPrime3PC(a, reluPrime, size);
	funcSelectShares3PC(a, reluPrime, b, size);

#ifdef VERIFYLAYERWISE
	if (partyNum == PARTY_A){
		sendVector<porthosSecretType>(a, PARTY_B, size);
		sendVector<porthosSecretType>(b, PARTY_B, size);
	}
	if (partyNum == PARTY_B){
		auto aOther = make_vector<porthosSecretType>(size);
		auto bOther = make_vector<porthosSecretType>(size);
		auto localAns = make_vector<porthosSecretType>(size);
	
		receiveVector<porthosSecretType>(aOther, PARTY_A, size);
		receiveVector<porthosSecretType>(bOther, PARTY_A, size);

		for(int i=0;i<size;i++){
			aOther[i] = a[i] + aOther[i];
			bOther[i] = b[i] + bOther[i];

			localAns[i] = (((porthosSignedSecretType)aOther[i]) > 0) ? aOther[i] : 0;
		}

		static int ctr = 1;
		bool pass = true;
		for(int i1=0;i1<size;i1++){
			if (localAns[i1] != bOther[i1]){
				std::cerr<<RED<<"ERROR in relu#"<<ctr<<" "<<i1<<" "<<localAns[i1]<<" "<<bOther[i1]<<" "<<aOther[i1]<<RESET<<std::endl;
				pass = false;
			}
		}

		if (!pass){
			std::cerr<<RED<<"There was an error in relu#"<<ctr<<RESET<<std::endl;
		}
		else{
			std::cout<<GREEN<<"Executed relu#"<<ctr<<" correctly."<<RESET<<std::endl;
		}
		ctr++;
	}	
#endif	
}

//Chunk wise maximum of a vector of size rows*columns and maximum is caclulated of every
//column number of elements. max is a vector of size rows. maxIndex contains the index of
//the maximum value.
//PARTY_A, PARTY_B start with the shares in a and {A,B} and {C,D} have the results in
//max and maxIndex.
void funcMaxMPC(vector<porthosSecretType> &a, 
		vector<porthosSecretType> &max, 
		vector<porthosSecretType> &maxIndex,
		size_t rows, 
		size_t columns,
		bool computeMaxIndex)
{
	log_print("funcMaxMPC");

	vector<porthosSecretType> diff(rows);
	vector<porthosSecretType> rp(rows);
	vector<porthosSecretType> diffIndex;
	vector<porthosSecretType> indexShares;

	for (size_t i = 0; i < rows; ++i)
	{
		max[i] = a[i*columns];
	}

	if (computeMaxIndex)
	{
		diffIndex.resize(rows);
		indexShares.resize(rows*columns, 0);
		for (size_t i = 0; i < rows; ++i)
		{
			maxIndex[i] = 0;
		}
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				if (partyNum == PARTY_A)
					indexShares[i*columns + j] = j;
	}

	for (size_t i = 1; i < columns; ++i)
	{
		for (size_t j = 0; j < rows; ++j)
			diff[j] = max[j] - a[j*columns + i];

		if (computeMaxIndex)
		{
			for (size_t j = 0; j < rows; ++j)
				diffIndex[j] = maxIndex[j] - indexShares[j*columns + i];
		}

		funcRELUPrime3PC(diff, rp, rows);
		funcSelectShares3PC(diff, rp, max, rows);

		if (computeMaxIndex)
			funcSelectShares3PC(diffIndex, rp, maxIndex, rows);

		for (size_t j = 0; j < rows; ++j)
			max[j] = max[j] + a[j*columns + i];

		if (computeMaxIndex)
			for (size_t j = 0; j < rows; ++j)
				maxIndex[j] = maxIndex[j] + indexShares[j*columns + i];
	}

}

//MaxIndex is of size rows. a is of size rows*columns.
//a will be set to 0's except at maxIndex (in every set of column)
void funcMaxIndexMPC(vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &maxIndex,
		size_t rows, 
		size_t columns)
{
	log_print("funcMaxIndexMPC");
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
		vector<porthosSecretType> vector(rows*columns, 0), share_1(rows*columns), share_2(rows*columns);
		receiveVector<smallType>(index, PARTY_A, rows);
		receiveVector<smallType>(temp, PARTY_B, rows);
		addVectors<smallType>(index, temp, index, rows);

		for (size_t i = 0; i < rows; ++i)
			index[i] = index[i] % columns;

		for (size_t i = 0; i < rows; ++i)
			vector[i*columns + index[i]] = 1;

		splitIntoShares(vector, share_1, share_2, rows*columns);
		sendVector<porthosSecretType>(share_1, PARTY_A, rows*columns);
		sendVector<porthosSecretType>(share_2, PARTY_B, rows*columns);
	}

	if (PRIMARY)
	{
		receiveVector<porthosSecretType>(a, PARTY_C, rows*columns);
		size_t offset = 0;
		for (size_t i = 0; i < rows; ++i)
		{
			rotate(a.begin()+offset, a.begin()+offset+(random[i] % columns), a.begin()+offset+columns);
			offset += columns;
		}
	}
}

/*************************************** Convolution and related utility functions **************************/

//The following functions (with _porthos) are used to implement convolution in Porthos.
//These implement optimized convolution as proposed in crypTFlow paper.
//There are 2 skeletons for most of these functions. One which takes in 
//vector and the other which takes in a pointer.

//Reshape filter
void Conv2DReshapeFilter_porthos(int32_t FH, 
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

void Conv2DReshapeFilterArr_porthos(int32_t FH, 
		int32_t FW, 
		int32_t CI, 
		int32_t CO, 
		porthosSecretType* inputArr, 
		porthosSecretType* outputArr)
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

//Reshape output
void Conv2DReshapeMatMulOP_porthos(int32_t N, 
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

//Reshape input image
void Conv2DReshapeInput_porthos(int32_t N, 
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
						porthosSecretType val = 0;
						
						for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
							if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
								val = 0;
							}
							else {
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

void Conv2DReshapeInputArr_porthos(int32_t N, 
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
		porthosSecretType* inputArr, 
		porthosSecretType* outputArr)
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
						porthosSecretType val = 0;
						
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
		vector< vector< vector< vector<porthosSecretType> > > >& y,
		vector< vector< vector< vector<porthosSecretType> > > >& x,
		vector< vector< vector< vector<porthosSecretType> > > >& outArr,
		porthosLongSignedInt consSF,
		auto& e_clear,
		auto& f_clear,
		auto& m_out)
{

	//Assign some helpful variables here.
	int r_x_cols = FH*FW*CI;
	int r_x_rows = CO;
	int r_y_rows = FH*FW*CI;
	int32_t newH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) +  (int32_t)1);
	int32_t newW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) +  (int32_t)1);
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

	if (PRIMARY){
		
		// First generate E and F
		auto m_x = make_vector<porthosSecretType>(FH, FW, CI, CO);
		auto m_y = make_vector<porthosSecretType>(N, H, W, CI);
		
		//Now populate here based on party number.
		if(partyNum == PARTY_A){
			populate_4D_vector(m_x, FH, FW, CI, CO, "a1");
			populate_4D_vector(m_y, N, H, W, CI, "b1");
		}
		else if(partyNum == PARTY_B){
			populate_4D_vector(m_x, FH, FW, CI, CO, "a2");
			populate_4D_vector(m_y, N, H, W, CI, "b2");
		}

		auto e = make_vector<porthosSecretType>(FH, FW, CI, CO);
		auto f = make_vector<porthosSecretType>(N, H, W, CI);
		auto e_other = make_vector<porthosSecretType>(FH, FW, CI, CO);
		auto f_other = make_vector<porthosSecretType>(N, H, W, CI);
		subtract_4D_vectors(x, m_x, e, FH, FW, CI, CO);
		subtract_4D_vectors(y, m_y, f, N, H, W, CI);

		//Reveal e and f.
		if(partyNum == PARTY_A){
#ifdef PARALLEL_COMM
			thread* threads1 = new thread[2];
			threads1[0] = thread(send_2_4D_vector, ref(e), ref(f), FH, FW, CI, CO, N, H, W, CI);
			threads1[1] = thread(receive_2_4D_vector, ref(e_other), ref(f_other), FH, FW, CI, CO, N, H, W, CI);
			for(int i=0; i<2; i++)
				threads1[i].join();
			delete[] threads1;
#else
			send_2_4D_vector(ref(e), ref(f), FH, FW, CI, CO, N, H, W, CI);
			receive_2_4D_vector(ref(e_other), ref(f_other), FH, FW, CI, CO, N, H, W, CI);
#endif
		}
		else if(partyNum == PARTY_B){
#ifdef PARALLEL_COMM
			thread* threads1 = new thread[2];
			threads1[0] = thread(receive_2_4D_vector, ref(e_other), ref(f_other), FH, FW, CI, CO, N, H, W, CI);
			threads1[1] = thread(send_2_4D_vector, ref(e), ref(f), FH, FW, CI, CO, N, H, W, CI);
			for(int i=0; i<2; i++)
				threads1[i].join();
			delete[] threads1;
#else
			receive_2_4D_vector(ref(e_other), ref(f_other), FH, FW, CI, CO, N, H, W, CI);
			send_2_4D_vector(ref(e), ref(f), FH, FW, CI, CO, N, H, W, CI);
#endif
		}

		add_4D_vectors(e, e_other, e_clear, FH, FW, CI, CO);
		add_4D_vectors(f, f_other, f_clear, N, H, W, CI);
	}
}

//Called only by primary parties
void ConvLocalMatMulOps(const vector< vector<porthosSecretType> >& X,
		const vector< vector<porthosSecretType> >& Y,
		vector< vector<porthosSecretType> >& Z, //Z is the output of the function
		const vector< vector<porthosSecretType> >& E_clear,
		const vector< vector<porthosSecretType> >& F_clear,
		const vector< vector<porthosSecretType> >& C,
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		porthosLongSignedInt consSF)
{
#ifdef DEBUG
	assert(PRIMARY && "Should have been called only from PRIMARY.");
#endif

	vector<vector<porthosSecretType>> temp_Z(rows, vector<porthosSecretType>(columns));
	if (partyNum == PARTY_A)
	{
		//Calculate X - E_clear
		vector<vector<porthosSecretType>> tempSubHolder(rows, vector<porthosSecretType>(common_dim));
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
		vector< vector< vector< vector<porthosSecretType> > > >& inputArr,
		vector< vector< vector< vector<porthosSecretType> > > >& filterArr,
		int32_t consSF,
		vector< vector< vector< vector<porthosSecretType> > > >& outArr)
{
	log_print("funcConv2DCSF");

	int32_t reshapedFilterRows = CO;
	int32_t reshapedFilterCols = ((FH * FW) * CI);
	int32_t reshapedIPRows = ((FH * FW) * CI);
	int32_t newH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) +  (int32_t)1);
	int32_t newW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) +  (int32_t)1);
	int32_t reshapedIPCols = ((N * newH) * newW);

	if(HELPER)
	{
		porthosSecretType* m_x1 = new porthosSecretType[FH*FW*CI*CO];
		porthosSecretType* m_y1 = new porthosSecretType[N*H*W*CI];
		porthosSecretType* m_x = new porthosSecretType[FH*FW*CI*CO];
		porthosSecretType* m_y = new porthosSecretType[N*H*W*CI];
		porthosSecretType* m_z0 = new porthosSecretType[reshapedFilterRows*reshapedIPCols];
		porthosSecretType* m_z = new porthosSecretType[reshapedFilterRows*reshapedIPCols];
		porthosSecretType* r_m_x = new porthosSecretType[reshapedFilterRows*reshapedFilterCols];
		porthosSecretType* r_m_y = new porthosSecretType[reshapedIPRows*reshapedIPCols];

#if (LOG_LAYERWISE)
		auto t1 = high_resolution_clock::now();
#endif
		populate_AES_Arr(m_x, ((porthosLongUnsignedInt)FH)*FW*CI*CO, "a1");
		populate_AES_Arr(m_x1, ((porthosLongUnsignedInt)FH)*FW*CI*CO, "a2");
		populate_AES_Arr(m_y, ((porthosLongUnsignedInt)N)*H*W*CI, "b1");
		populate_AES_Arr(m_y1, ((porthosLongUnsignedInt)N)*H*W*CI, "b2");
		populate_AES_Arr(m_z0, reshapedFilterRows*reshapedIPCols, "c1");

#if (LOG_LAYERWISE)
		auto t2 = high_resolution_clock::now();
		auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
		cout<<"funcConv2DCSF: Time for AES populate (in sec) : "<<tt<<endl;
#endif

		add_2_Arr(m_x, m_x1, m_x, ((porthosLongUnsignedInt)FH)*FW*CI*CO);
		add_2_Arr(m_y, m_y1, m_y, ((porthosLongUnsignedInt)N)*H*W*CI);

#if (LOG_LAYERWISE)
		t1 = high_resolution_clock::now();
#endif
		Conv2DReshapeFilterArr_porthos(FH, FW, CI, CO, m_x, r_m_x);
		Conv2DReshapeInputArr_porthos(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, m_y, r_m_y);

#if (LOG_LAYERWISE)
		t2 = high_resolution_clock::now();
		tt = (duration_cast<duration<double>>(t2 - t1)).count();
		cout<<"funcConv2DCSF: Time for reshape (in sec) : "<<tt<<endl;
		t1 = high_resolution_clock::now();
#endif
		matrixMultEigen(r_m_x, r_m_y, m_z, reshapedFilterRows, reshapedFilterCols, reshapedIPCols, 0, 0);

#if (LOG_LAYERWISE)
		t2 = high_resolution_clock::now();
		tt = (duration_cast<duration<double>>(t2 - t1)).count();
		cout<<"funcConv2DCSF: Time for matmul (in sec) : "<<tt<<endl;
#endif
		subtract_2_Arr(m_z, m_z0, m_z, reshapedFilterRows*reshapedIPCols);
		sendArr<porthosSecretType>(m_z, PARTY_B, reshapedFilterRows*reshapedIPCols);

		//Free memmory for these later at the end.
		toFreeMemoryLaterArr.push_back(m_x1);
		toFreeMemoryLaterArr.push_back(m_y1);
		toFreeMemoryLaterArr.push_back(m_x);
		toFreeMemoryLaterArr.push_back(m_y);
		toFreeMemoryLaterArr.push_back(m_z0);
		toFreeMemoryLaterArr.push_back(m_z);
		toFreeMemoryLaterArr.push_back(r_m_x);
		toFreeMemoryLaterArr.push_back(r_m_y);

	}
	else if (PRIMARY)
	{
#if (LOG_LAYERWISE)
		auto t1 = high_resolution_clock::now();
#endif
		auto filterReshaped = make_vector<porthosSecretType>(reshapedFilterRows, reshapedFilterCols);
		auto E_filterReshaped = make_vector<porthosSecretType>(reshapedFilterRows, reshapedFilterCols);
		auto inputReshaped = make_vector<porthosSecretType>(reshapedIPRows, reshapedIPCols);
		auto F_inputReshaped = make_vector<porthosSecretType>(reshapedIPRows, reshapedIPCols);
		auto matmulOP = make_vector<porthosSecretType>(reshapedFilterRows, reshapedIPCols);
		auto m_matmulOP = make_vector<porthosSecretType>(reshapedFilterRows, reshapedIPCols);
		//Communicate E and F before reshaping filter and input.
		//E will be filter-m_filter and F will be input-m_input

		auto e_clear = make_vector<porthosSecretType>(FH, FW, CI, CO);
		auto f_clear = make_vector<porthosSecretType>(N, H, W, CI);

#if (LOG_LAYERWISE)
		auto t2 = high_resolution_clock::now();
		auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
		cout<<"funcConv2DCSF: Time for make_vector (in sec) : "<<tt<<endl;
		t1 = high_resolution_clock::now();
#endif
		Conv2DCSF_optimized_backend(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr, consSF, e_clear, f_clear, m_matmulOP);

#if (LOG_LAYERWISE)
		t2 = high_resolution_clock::now();
		tt = (duration_cast<duration<double>>(t2 - t1)).count();
		cout<<"funcConv2DCSF: Time for optimized backend func (in sec) : "<<tt<<endl;
		t1 = high_resolution_clock::now();
#endif
		Conv2DReshapeFilter_porthos(FH, FW, CI, CO, filterArr, filterReshaped);
		Conv2DReshapeFilter_porthos(FH, FW, CI, CO, e_clear, E_filterReshaped);

		Conv2DReshapeInput_porthos(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
		Conv2DReshapeInput_porthos(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, f_clear, F_inputReshaped);

#if (LOG_LAYERWISE)
		t2 = high_resolution_clock::now();
		tt = (duration_cast<duration<double>>(t2 - t1)).count();
		cout<<"funcConv2DCSF: Time for reshape (in sec) : "<<tt<<endl;
		t1 = high_resolution_clock::now();
#endif
		ConvLocalMatMulOps(filterReshaped, inputReshaped, matmulOP, E_filterReshaped, F_inputReshaped, m_matmulOP, reshapedFilterRows, reshapedFilterCols, reshapedIPCols, consSF);

#if (LOG_LAYERWISE)
		t2 = high_resolution_clock::now();
		tt = (duration_cast<duration<double>>(t2 - t1)).count();
		cout<<"funcConv2DCSF: Time for matmul (in sec) : "<<tt<<endl;
#endif

		if (partyNum == PARTY_A){
			populate_2D_vector(m_matmulOP, reshapedFilterRows, reshapedIPCols, "c1");
		}
		else if (partyNum == PARTY_B){
			// Receive from Party_C
			receive_2D_vector(m_matmulOP, reshapedFilterRows, reshapedIPCols);
		}

		add_2D_vectors(matmulOP, m_matmulOP, matmulOP, reshapedFilterRows, reshapedIPCols);
		funcTruncate2PC(matmulOP, consSF, reshapedFilterRows, reshapedIPCols);

		Conv2DReshapeMatMulOP_porthos(N, newH, newW, CO, matmulOP, outArr);
	}
}

void Conv3DSliding(int32_t N, 
		int32_t D, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FD, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t zPadDLeft, 
		int32_t zPadDRight, 
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideD, 
		int32_t strideH, 
		int32_t strideW, 
		int32_t outD, 
		int32_t outH, 
		int32_t outW, 
		auto& inputArr, 
		auto& filterArr, 
		int32_t consSF, 
		auto& outArr){
#pragma omp parallel for collapse(5)
	for (uint32_t n =  (int32_t)0; n < N; n++){
		for (uint32_t co =  (int32_t)0; co < CO; co++){
			for (uint32_t d =  (int32_t)0; d < outD; d++){
				for (uint32_t h =  (int32_t)0; h < outH; h++){
					for (uint32_t w =  (int32_t)0; w < outW; w++){
						for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
							int64_t val =  (int64_t)0;
							for (uint32_t fd = (d * strideD); fd < ((d * strideD) + FD); fd++){
								for (uint32_t fh = (h * strideH); fh < ((h * strideH) + FH); fh++){
									for (uint32_t fw = (w * strideW); fw < ((w * strideW) + FW); fw++){
										int32_t curPosD = (fd - zPadDLeft);
										int32_t curPosH = (fh - zPadHLeft);
										int32_t curPosW = (fw - zPadWLeft);
										if (((((((curPosD >=  (int32_t)0) && (curPosH >=  (int32_t)0)) && (curPosW >=  (int32_t)0)) && (curPosD < D)) && (curPosH < H)) && (curPosW < W))) {

											int32_t curFilterPosD = (fd - (d * strideD));

											int32_t curFilterPosH = (fh - (h * strideH));

											int32_t curFilterPosW = (fw - (w * strideW));
											val = (val + (inputArr[n][curPosD][curPosH][curPosW][ci] * filterArr[curFilterPosD][curFilterPosH][curFilterPosW][ci][co]));
										}
									}
								}
							}
							//outArr[n][d][h][w][co] = (outArr[n][d][h][w][co] + (val >> consSF));
							outArr[n][d][h][w][co] = (outArr[n][d][h][w][co] + (val));
						}
					}
				}
			}
		}
	}
}

void funcConv3DMPC(
		int32_t N,
		int32_t D,
		int32_t H,
		int32_t W,
		int32_t CI,
		int32_t FD,
		int32_t FH,
		int32_t FW,
		int32_t CO,
		int32_t zPadDLeft,
		int32_t zPadDRight,
		int32_t zPadHLeft,
		int32_t zPadHRight,
		int32_t zPadWLeft,
		int32_t zPadWRight,
		int32_t strideD,
		int32_t strideH,
		int32_t strideW,
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr,
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr,
		int32_t consSF,
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr) 
{
	log_print("funcConv3DCSF");
	// out -> n, outd, outh, outw, co
	// in -> n, d, h, w, cin
	// filter -> filter_depth, filter_height, filter_width, in_channels, out_channels
	int32_t outD = ((((D - FD) + (zPadDLeft + zPadDRight)) / strideD) +  (int32_t)1);
	int32_t outH = ((((H - FH) + (zPadHLeft + zPadHRight)) / strideH) +  (int32_t)1);
	int32_t outW = ((((W - FW) + (zPadWLeft + zPadWRight)) / strideW) +  (int32_t)1);
	std::cout<<"D, H, W of output: "<<outD<<" "<<outH<<" "<<outW<<std::endl;

	auto A = make_vector<porthosSecretType>(N, D, H, W, CI);
	auto B = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
	auto C = make_vector<porthosSecretType>(N, outD, outH, outW, CO);

	if(HELPER) {
		auto A1 = make_vector<porthosSecretType>(N, D, H, W, CI);
		auto A2 = make_vector<porthosSecretType>(N, D, H, W, CI);
		auto B1 = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
		auto B2 = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
		auto C1 = make_vector<porthosSecretType>(N, outD, outH, outW, CO);
		auto C2 = make_vector<porthosSecretType>(N, outD, outH, outW, CO);

		populate_5D_vector(A1, N, D, H, W, CI, "a1");
		populate_5D_vector(A2, N, D, H, W, CI, "a2");
		populate_5D_vector(B1, FD, FH, FW, CI, CO, "b1");
		populate_5D_vector(B2, FD, FH, FW, CI, CO, "b2");
		populate_5D_vector(C1, N, outD, outH, outW, CO, "c1");

		add_5D_vectors(A1, A2, A, N, D, H, W, CI);
		add_5D_vectors(B1, B2, B, FD, FH, FW, CI, CO);

		Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, A, B, consSF, C);

		subtract_5D_vectors(C, C1, C2, N, outD, outH, outW, CO);

#if (LOG_LAYERWISE)
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
#endif

		send_5D_vector(C2, PARTY_B, N, outD, outH, outW, CO);

#if (LOG_LAYERWISE)
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		auto tt = time_span.count();
		commObject.timeMatmul[0] += tt;
#endif
#ifdef DEBUGGING
		send_5D_vector(A, PARTY_A, N, D, H, W, CI);
		send_5D_vector(B, PARTY_A, FD, FH, FW, CI, CO);
		send_5D_vector(C, PARTY_A, N, outD, outH, outW, CO);
#endif

	}

	if (PRIMARY) {
#ifdef DEBUGGING
		if(partyNum == PARTY_B){
			send_5D_vector(inputArr, PARTY_A, N, D, H, W, CI);
			send_5D_vector(filterArr, PARTY_A, FD, FH, FW, CI, CO);
			return;
		}
		else if(partyNum == PARTY_A){
			auto x = make_vector<porthosSecretType>(N, D, H, W, CI);
			auto y = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
			auto a = make_vector<porthosSecretType>(N, D, H, W, CI);
			auto b = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
			auto c = make_vector<porthosSecretType>(N, outD, outH, outW, CO);
			auto c_ans = make_vector<porthosSecretType>(N, outD, outH, outW, CO);
			auto c_temp = make_vector<porthosSecretType>(N, outD, outH, outW, CO);
			auto e = make_vector<porthosSecretType>(N, D, H, W, CI);
			auto f = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);

			receive_5D_vector(ref(a), PARTY_C, N, D, H, W, CI);
			receive_5D_vector(ref(b), PARTY_C, FD, FH, FW, CI, CO);
			receive_5D_vector(ref(c), PARTY_C, N, outD, outH, outW, CO);
			receive_5D_vector(ref(x), PARTY_B, N, D, H, W, CI);
			receive_5D_vector(ref(y), PARTY_B, FD, FH, FW, CI, CO);
			add_5D_vectors(x, inputArr, x, N, D, H, W, CI);
			add_5D_vectors(y, filterArr, y, FD, FH, FW, CI, CO);
			subtract_5D_vectors(x, a, e, N, D, H, W, CI);
			subtract_5D_vectors(y, b, f, FD, FH, FW, CI, CO);
			subtract_5D_vectors(x, e, A, N, D, H, W, CI);
			Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, A, f, consSF, c_temp);
			Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, e, y, consSF, c_ans);
			std::cout<<"From reconstructed code:"<<std::endl;
			std::cout<<e[0][0][0][0][0]<<std::endl;
			std::cout<<f[0][0][0][0][0]<<std::endl;
			std::cout<<"From reconstructed code ends"<<std::endl;
			add_5D_vectors(c_temp, c, c, N, outD, outH, outW, CO);
			add_5D_vectors(c_ans, c, outArr, N, outD, outH, outW, CO);
			funcTruncate2PC(outArr, consSF, N, outD, outH, outW, CO);
			return;
				
		}
#endif
		auto E = make_vector<porthosSecretType>(N, D, H, W, CI);
		auto F = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
		auto temp_C = make_vector<porthosSecretType>(N, outD, outH, outW, CO);
		auto temp_E = make_vector<porthosSecretType>(N, D, H, W, CI);
		auto temp_F = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
		if (partyNum == PARTY_A) {
			populate_5D_vector(A, N, D, H, W, CI, "a1");
			populate_5D_vector(B, FD, FH, FW, CI, CO, "b1");
			populate_5D_vector(C, N, outD, outH, outW, CO, "c1");
		}

		if (partyNum == PARTY_B) {
			populate_5D_vector(A, N, D, H, W, CI, "a2");
			populate_5D_vector(B, FD, FH, FW, CI, CO, "b2");
		}

		subtract_5D_vectors(inputArr, A, E, N, D, H, W, CI);
		subtract_5D_vectors(filterArr, B, F, FD, FH, FW, CI, CO);

#if (LOG_LAYERWISE)
		auto t1 = high_resolution_clock::now();
#endif
		if (partyNum == PARTY_A) {
			send_5D_vector(ref(E), adversary(partyNum), N, D, H, W, CI);
			send_5D_vector(ref(F), adversary(partyNum), FD, FH, FW, CI, CO);
			receive_5D_vector(ref(temp_E), adversary(partyNum), N, D, H, W, CI);
			receive_5D_vector(ref(temp_F), adversary(partyNum), FD, FH, FW, CI, CO);
		} else {
			receive_5D_vector(ref(temp_E), adversary(partyNum), N, D, H, W, CI);
			receive_5D_vector(ref(temp_F), adversary(partyNum), FD, FH, FW, CI, CO);
			send_5D_vector(ref(E), adversary(partyNum), N, D, H, W, CI);
			send_5D_vector(ref(F), adversary(partyNum), FD, FH, FW, CI, CO);
		}
#if (LOG_LAYERWISE)
		auto t2 = high_resolution_clock::now();
		auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
		commObject.timeMatmul[0] += tt;
#endif
		add_5D_vectors(E, temp_E, E, N, D, H, W, CI);
		add_5D_vectors(F, temp_F, F, FD, FH, FW, CI, CO);
		
		//Receive C1 from P2 after E and F have been revealed.
		if (partyNum == PARTY_B) {
			receive_5D_vector(ref(C), PARTY_C, N, outD, outH, outW, CO);
			subtract_5D_vectors(inputArr, E, A, N, D, H, W, CI);
			Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, A, F, consSF, outArr);
			Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, E, filterArr, consSF, temp_C);
		}

		if (partyNum == PARTY_A) {
			Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, inputArr, F, consSF, outArr);
			Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, zPadDLeft, zPadDRight, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideD, strideH, strideW, outD, outH, outW, E, filterArr, consSF, temp_C);
		}
		//subtract_5D_vectors(inputArr, E, A, N, D, H, W, CI);
		//std::cout<<"From secure code:"<<std::endl;
		//std::cout<<E[0][0][0][0][0]<<std::endl;
		//std::cout<<F[0][0][0][0][0]<<std::endl;
		//std::cout<<"From secure code ends"<<std::endl;

		add_5D_vectors(outArr, temp_C, outArr, N, outD, outH, outW, CO);
		add_5D_vectors(C, outArr, outArr, N, outD, outH, outW, CO);
		//if (areInputsScaled)
	//	funcTruncate2PC(outArr, consSF, N, outD, outH, outW, CO);

	}

#ifdef VERIFYLAYERWISE
	
	if (partyNum == PARTY_A){
		send_5D_vector(inputArr, PARTY_B, N, D, H, W, CI);
		send_5D_vector(filterArr, PARTY_B, FD, FH, FW, CI, CO);
		send_5D_vector(outArr, PARTY_B, N, outD, outH, outW, CO);
	}
	if (partyNum == PARTY_B){
		auto inputArrOther = make_vector<porthosSecretType>(N, D, H, W, CI);
		auto filterArrOther = make_vector<porthosSecretType>(FD, FH, FW, CI, CO);
		auto outArrLocalAns = make_vector<porthosSecretType>(N, outD, outH, outW, CO);
		auto outArrOther = make_vector<porthosSecretType>(N, outD, outH, outW, CO);
	
		receive_5D_vector(inputArrOther, PARTY_A, N, D, H, W, CI);
		receive_5D_vector(filterArrOther, PARTY_A, FD, FH, FW, CI, CO);
		receive_5D_vector(outArrOther, PARTY_A, N, outD, outH, outW, CO);

		add_5D_vectors(inputArrOther, inputArr, inputArrOther, N, D, H, W, CI);
		add_5D_vectors(filterArrOther, filterArr, filterArrOther, FD, FH, FW, CI, CO);
		add_5D_vectors(outArrOther, outArr, outArrOther, N, outD, outH, outW, CO);

		Conv3DSliding(N, D, H, W, CI, FD, FH, FW, CO, 
					zPadDLeft, zPadDRight, 
					zPadHLeft, zPadHRight, 
					zPadWLeft, zPadWRight, 
					strideD, strideH, strideW, 
					outD, outH, outW, 
					inputArrOther, filterArrOther, consSF, outArrLocalAns);

		static int ctr = 1;
		bool pass = true;
		for(int i1=0;i1<N;i1++){
			for(int i2=0;i2<outD;i2++){
				for(int i3=0;i3<outH;i3++){
					for(int i4=0;i4<outW;i4++){
						for(int i5=0;i5<CO;i5++){
							if ((((porthosSignedSecretType)outArrLocalAns[i1][i2][i3][i4][i5])>>consSF) != (((porthosSignedSecretType)outArrOther[i1][i2][i3][i4][i5]) >> consSF)){
								std::cerr<<RED<<"ERROR in conv3d #"<<ctr<<" "<<i1<<" "<<i2<<" "<<i3<<" "<<i4<<" "<<i5<<" "<<
									(((porthosSignedSecretType)outArrLocalAns[i1][i2][i3][i4][i5])>>consSF)<<" "<<
									outArrOther[i1][i2][i3][i4][i5]<<RESET<<std::endl;
								pass = false;
							}
						}
					}
				}
			}
		}

		if (!pass){
			std::cerr<<RED<<"There was an error in conv3d#"<<ctr<<RESET<<std::endl;
		}
		else{
			std::cout<<GREEN<<"Executed conv3d#"<<ctr<<" correctly."<<RESET<<std::endl;
		}
		ctr++;
	}	

#endif	
}

void ConvTranspose3DLoopInnerClear(int32_t N, 
		int32_t D, 
		int32_t H, 
		int32_t W, 
		int32_t CI, 
		int32_t FD, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t zPadDLeft, 
		int32_t zPadDRight, 
		int32_t zPadHLeft, 
		int32_t zPadHRight, 
		int32_t zPadWLeft, 
		int32_t zPadWRight, 
		int32_t strideD, 
		int32_t strideH, 
		int32_t strideW, 
		int32_t outD, 
		int32_t outH, 
		int32_t outW, 
		auto& inputArr, 
		auto& filterArr, 
		auto& outArr)
{
#pragma omp parallel for collapse(5)
for (uint32_t n =  (int32_t)0; n < N; n++){
for (uint32_t co =  (int32_t)0; co < CO; co++){
for (uint32_t d =  (int32_t)0; d < outD; d++){
for (uint32_t h =  (int32_t)0; h < outH; h++){
for (uint32_t w =  (int32_t)0; w < outW; w++){
for (uint32_t ci =  (int32_t)0; ci < CI; ci++){

int64_t val =  (int64_t)0;
for (uint32_t fd = d; fd < (d + FD); fd++){
for (uint32_t fh = h; fh < (h + FH); fh++){
for (uint32_t fw = w; fw < (w + FW); fw++){

int32_t curPosD = ((fd - zPadDLeft) / strideD);

int32_t curPosH = ((fh - zPadHLeft) / strideD);

int32_t curPosW = ((fw - zPadWLeft) / strideD);
if ((((((((((curPosD >=  (int32_t)0) && (curPosH >=  (int32_t)0)) && (curPosW >=  (int32_t)0)) && (curPosD < D)) && (curPosH < H)) && (curPosW < W)) && (((fd - zPadDLeft) % strideD) ==  (int32_t)0)) && (((fh - zPadHLeft) % strideH) ==  (int32_t)0)) && (((fw - zPadWLeft) % strideW) ==  (int32_t)0))) {

int32_t curFilterPosD = (((FD + d) - fd) -  (int32_t)1);

int32_t curFilterPosH = (((FH + h) - fh) -  (int32_t)1);

int32_t curFilterPosW = (((FW + w) - fw) -  (int32_t)1);
val = (val + (inputArr[n][curPosD][curPosH][curPosW][ci] * filterArr[curFilterPosD][curFilterPosH][curFilterPosW][co][ci]));
}
}
}
}
outArr[n][d][h][w][co] = (outArr[n][d][h][w][co] + val);
}
}
}
}
}
}
}

void ConvTranspose3DSliding(int32_t N, 
		int32_t DPrime, 
		int32_t HPrime, 
		int32_t WPrime, 
		int32_t CI, 
		int32_t FD, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t D, 
		int32_t H, 
		int32_t W, 
		int32_t zPadTrDLeft, 
		int32_t zPadTrDRight, 
		int32_t zPadTrHLeft, 
		int32_t zPadTrHRight, 
		int32_t zPadTrWLeft, 
		int32_t zPadTrWRight, 
		int32_t strideD, 
		int32_t strideH, 
		int32_t strideW, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr){
ConvTranspose3DLoopInnerClear(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, D, H, W, inputArr, filterArr, outArr);
}

void ConvTranspose3DCSFMPC(int32_t N, 
		int32_t DPrime, 
		int32_t HPrime, 
		int32_t WPrime, 
		int32_t CI, 
		int32_t FD, 
		int32_t FH, 
		int32_t FW, 
		int32_t CO, 
		int32_t D, 
		int32_t H, 
		int32_t W, 
		int32_t zPadTrDLeft, 
		int32_t zPadTrDRight, 
		int32_t zPadTrHLeft, 
		int32_t zPadTrHRight, 
		int32_t zPadTrWLeft, 
		int32_t zPadTrWRight, 
		int32_t strideD, 
		int32_t strideH, 
		int32_t strideW, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inputArr, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& filterArr, 
		int32_t consSF, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& outArr)
{
	log_print("ConvTranspose3DCSFMPC");
	auto A = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
	auto B = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
	auto C = make_vector<porthosSecretType>(N, D, H, W, CO);

	if(HELPER) {
		auto A1 = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
		auto A2 = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
		auto B1 = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
		auto B2 = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
		auto C1 = make_vector<porthosSecretType>(N, D, H, W, CO);
		auto C2 = make_vector<porthosSecretType>(N, D, H, W, CO);

		populate_5D_vector(A1, N, DPrime, HPrime, WPrime, CI, "a1");
		populate_5D_vector(A2, N, DPrime, HPrime, WPrime, CI, "a2");
		populate_5D_vector(B1, FD, FH, FW, CO, CI, "b1");
		populate_5D_vector(B2, FD, FH, FW, CO, CI, "b2");
		populate_5D_vector(C1, N, D, H, W, CO, "c1");

		add_5D_vectors(A1, A2, A, N, DPrime, HPrime, WPrime, CI);
		add_5D_vectors(B1, B2, B, FD, FH, FW, CO, CI);

		ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, A, B, C);

		subtract_5D_vectors(C, C1, C2, N, D, H, W, CO);

#if (LOG_LAYERWISE)
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
#endif

		send_5D_vector(C2, PARTY_B, N, D, H, W, CO);

#if (LOG_LAYERWISE)
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
		auto tt = time_span.count();
		commObject.timeMatmul[0] += tt;
#endif
#ifdef DEBUGGING
		send_5D_vector(A, PARTY_A, N, DPrime, HPrime, WPrime, CI);
		send_5D_vector(B, PARTY_A, FD, FH, FW, CO, CI);
		send_5D_vector(C, PARTY_A, N, D, H, W, CO);
#endif

	}

	if (PRIMARY) {
#ifdef DEBUGGING
		if(partyNum == PARTY_B){
			send_5D_vector(inputArr, PARTY_A, N, DPrime, HPrime, WPrime, CI);
			send_5D_vector(filterArr, PARTY_A, FD, FH, FW, CO, CI);
			return;
		}
		else if(partyNum == PARTY_A){
			auto x = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
			auto y = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
			auto a = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
			auto b = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
			auto c = make_vector<porthosSecretType>(N, D, H, W, CO);
			auto c_ans = make_vector<porthosSecretType>(N, D, H, W, CO);
			auto c_temp = make_vector<porthosSecretType>(N, D, H, W, CO);
			auto e = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
			auto f = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);

			receive_5D_vector(ref(a), PARTY_C, N, DPrime, HPrime, WPrime, CI);
			receive_5D_vector(ref(b), PARTY_C, FD, FH, FW, CO, CI);
			receive_5D_vector(ref(c), PARTY_C, N, D, H, W, CO);
			receive_5D_vector(ref(x), PARTY_B, N, DPrime, HPrime, WPrime, CI);
			receive_5D_vector(ref(y), PARTY_B, FD, FH, FW, CO, CI);
			add_5D_vectors(x, inputArr, x, N, DPrime, HPrime, WPrime, CI);
			add_5D_vectors(y, filterArr, y, FD, FH, FW, CO, CI);
			subtract_5D_vectors(x, a, e, N, DPrime, HPrime, WPrime, CI);
			subtract_5D_vectors(y, b, f, FD, FH, FW, CO, CI);
			subtract_5D_vectors(x, e, A, N, DPrime, HPrime, WPrime, CI);
			ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, A, f, c_temp);
			ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, e, y, c_ans);
			std::cout<<"From reconstructed code:"<<std::endl;
			std::cout<<e[0][0][0][0][0]<<std::endl;
			std::cout<<f[0][0][0][0][0]<<std::endl;
			std::cout<<"From reconstructed code ends"<<std::endl;
			add_5D_vectors(c_temp, c, c, N, D, H, W, CO);
			add_5D_vectors(c_ans, c, outArr, N, D, H, W, CO);
			funcTruncate2PC(outArr, consSF, N, D, H, W, CO);
			return;
				
		}
#endif
		auto E = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
		auto F = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
		auto temp_C = make_vector<porthosSecretType>(N, D, H, W, CO);
		auto temp_E = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
		auto temp_F = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
		if (partyNum == PARTY_A) {
			populate_5D_vector(A, N, DPrime, HPrime, WPrime, CI, "a1");
			populate_5D_vector(B, FD, FH, FW, CO, CI, "b1");
			populate_5D_vector(C, N, D, H, W, CO, "c1");
		}

		if (partyNum == PARTY_B) {
			populate_5D_vector(A, N, DPrime, HPrime, WPrime, CI, "a2");
			populate_5D_vector(B, FD, FH, FW, CO, CI, "b2");
		}

		subtract_5D_vectors(inputArr, A, E, N, DPrime, HPrime, WPrime, CI);
		subtract_5D_vectors(filterArr, B, F, FD, FH, FW, CO, CI);

#if (LOG_LAYERWISE)
		auto t1 = high_resolution_clock::now();
#endif
		if (partyNum == PARTY_A) {
			send_5D_vector(ref(E), adversary(partyNum), N, DPrime, HPrime, WPrime, CI);
			send_5D_vector(ref(F), adversary(partyNum), FD, FH, FW, CO, CI);
			receive_5D_vector(ref(temp_E), adversary(partyNum), N, DPrime, HPrime, WPrime, CI);
			receive_5D_vector(ref(temp_F), adversary(partyNum), FD, FH, FW, CO, CI);
		} else {
			receive_5D_vector(ref(temp_E), adversary(partyNum), N, DPrime, HPrime, WPrime, CI);
			receive_5D_vector(ref(temp_F), adversary(partyNum), FD, FH, FW, CO, CI);
			send_5D_vector(ref(E), adversary(partyNum), N, DPrime, HPrime, WPrime, CI);
			send_5D_vector(ref(F), adversary(partyNum), FD, FH, FW, CO, CI);
		}
#if (LOG_LAYERWISE)
		auto t2 = high_resolution_clock::now();
		auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
		commObject.timeMatmul[0] += tt;
#endif
		add_5D_vectors(E, temp_E, E, N, DPrime, HPrime, WPrime, CI);
		add_5D_vectors(F, temp_F, F, FD, FH, FW, CO, CI);
		
		//Receive C1 from P2 after E and F have been revealed.
		if (partyNum == PARTY_B) {
			receive_5D_vector(ref(C), PARTY_C, N, D, H, W, CO);
			subtract_5D_vectors(inputArr, E, A, N, DPrime, HPrime, WPrime, CI);
			ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, A, F, outArr);
			ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, E, filterArr, temp_C);
		}

		if (partyNum == PARTY_A) {
			ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, inputArr, F, outArr);
			ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, zPadTrDLeft, zPadTrDRight, zPadTrHLeft, zPadTrHRight, zPadTrWLeft, zPadTrWRight, strideD, strideH, strideW, E, filterArr, temp_C);
		}
		//subtract_5D_vectors(inputArr, E, A, N, D, H, W, CI);
		//std::cout<<"From secure code:"<<std::endl;
		//std::cout<<E[0][0][0][0][0]<<std::endl;
		//std::cout<<F[0][0][0][0][0]<<std::endl;
		//std::cout<<"From secure code ends"<<std::endl;

		add_5D_vectors(outArr, temp_C, outArr, N, D, H, W, CO);
		add_5D_vectors(C, outArr, outArr, N, D, H, W, CO);
		//if (areInputsScaled)
	//	funcTruncate2PC(outArr, consSF, N, D, H, W, CO);

	}

#ifdef VERIFYLAYERWISE
		if (partyNum == PARTY_A){
			send_5D_vector(inputArr, PARTY_B, N, DPrime, HPrime, WPrime, CI);
			send_5D_vector(filterArr, PARTY_B, FD, FH, FW, CO, CI);
			send_5D_vector(outArr, PARTY_B, N, D, H, W, CO);
		}
		if (partyNum == PARTY_B){
			auto inputArrOther = make_vector<porthosSecretType>(N, DPrime, HPrime, WPrime, CI);
			auto filterArrOther = make_vector<porthosSecretType>(FD, FH, FW, CO, CI);
			auto outArrLocalAns = make_vector<porthosSecretType>(N, D, H, W, CO);
			auto outArrOther = make_vector<porthosSecretType>(N, D, H, W, CO);
		
			receive_5D_vector(inputArrOther, PARTY_A, N, DPrime, HPrime, WPrime, CI);
			receive_5D_vector(filterArrOther, PARTY_A, FD, FH, FW, CO, CI);
			receive_5D_vector(outArrOther, PARTY_A, N, D, H, W, CO);
	
			add_5D_vectors(inputArrOther, inputArr, inputArrOther, N, DPrime, HPrime, WPrime, CI);
			add_5D_vectors(filterArrOther, filterArr, filterArrOther, FD, FH, FW, CO, CI);
			add_5D_vectors(outArrOther, outArr, outArrOther, N, D, H, W, CO);
	
			ConvTranspose3DSliding(N, DPrime, HPrime, WPrime, CI, FD, FH, FW, CO, D, H, W, 
				zPadTrDLeft, zPadTrDRight, 
				zPadTrHLeft, zPadTrHRight, 
				zPadTrWLeft, zPadTrWRight, 
				strideD, strideH, strideW, 
				inputArrOther, 
				filterArrOther, 
				outArrLocalAns);

			static int ctr = 1;
	
			bool pass = true;
			for(int i1=0;i1<N;i1++){
				for(int i2=0;i2<D;i2++){
					for(int i3=0;i3<H;i3++){
						for(int i4=0;i4<W;i4++){
							for(int i5=0;i5<CO;i5++){
								if ((((porthosSignedSecretType)outArrLocalAns[i1][i2][i3][i4][i5])>>consSF) != (((porthosSignedSecretType)outArrOther[i1][i2][i3][i4][i5]) >> consSF)){
									std::cerr<<RED<<"ERROR in convtranspose #"<<ctr<<" "<<i1<<" "<<i2<<" "<<i3<<" "<<i4<<" "<<i5<<" "<<
										(((porthosSignedSecretType)outArrLocalAns[i1][i2][i3][i4][i5])>>consSF)<<" "<<
										outArrOther[i1][i2][i3][i4][i5]<<RESET<<std::endl;
									pass = false;
								}
							}
						}
					}
				}
			}

			if (!pass){
				std::cerr<<RED<<"There was an error in convTranspose3d#"<<ctr<<RESET<<std::endl;
			}			
			else{
				std::cout<<GREEN<<"Executed convTranspose3d#"<<ctr<<" correctly."<<RESET<<std::endl;
			}
			ctr++;
		}	
	
#endif	
}


/************************************* End of main MPC functions ********************************************/

/******************* Wrapper integer function calls ********************/
porthosSecretType funcMult(porthosSecretType a, 
		porthosSecretType b)
{
	vector<porthosSecretType> tmp1(1, a);
	vector<porthosSecretType> tmp2(1, b);
	vector<porthosSecretType> tmp3(1, 0);
	funcMatMulMPC(tmp1, tmp2, tmp3, 1, 1, 1, 0, 0, false);
	return tmp3[0];
}

porthosSecretType funcReluPrime(porthosSecretType a)
{
	vector<porthosSecretType> tmp1(1, a);
	vector<porthosSecretType> tmp2(1, 0);
	funcRELUPrime3PC(tmp1, tmp2, 1);
	return tmp2[0];
}

porthosSecretType funcSSCons(int32_t x)
{
	/*
		Secret share public value x between the two parties. 
		Corresponding ezpc statement would be int32_al x = 0;
		Set one party share as x and other party's share as 0.
	*/
	if (partyNum == PARTY_A){
		return x;
	}
	if (partyNum == PARTY_B){
		return 0;
	}
}

porthosSecretType funcSSCons(int64_t x)
{
	/*
		Secret share public value x between the two parties. 
		Corresponding ezpc statement would be int32_al x = 0;
		Set one party share as x and other party's share as 0.
	*/
	if (partyNum == PARTY_A){
		return x;
	}
	if (partyNum == PARTY_B){
		return 0;
	}
}


//Arg2 revealToParties is a bitmask as to which parties should see the reconstructed values
//10 - party 0, 01 - party 1, 11 - party 0 & 1
porthosSecretType funcReconstruct2PCCons(porthosSecretType a, 
		int revealToParties)
{
	if (HELPER)
	{
		//skip
		return a;
	}
	vector<porthosSecretType> tmp1(1,a);
	vector<porthosSecretType> tmp2(1,0);
	funcReconstruct2PC(tmp1, 1, "", &tmp2, revealToParties);
	return tmp2[0];
}

/******************************** Debugging functions ********************************/
void debugDotProd()
{
	size_t size = 10;
	vector<porthosSecretType> a(size, 0), b(size, 0), c(size);
	vector<porthosSecretType> temp(size);

	populateRandomVector<porthosSecretType>(temp, size, "COMMON", "NEGATIVE");
	for (size_t i = 0; i < size; ++i)
	{
		if (partyNum == PARTY_A)
			a[i] = temp[i] + floatToMyType(i);
		else
			a[i] = temp[i];
	}

	populateRandomVector<porthosSecretType>(temp, size, "COMMON", "NEGATIVE");
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
	vector<porthosSecretType> a(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			a[i] = i - 5;

	vector<porthosSecretType> c(size);
	funcComputeMSB3PC(a, c, size);

	if (PRIMARY)
		funcReconstruct2PC(c, size, "c");

}

void debugPC()
{
	size_t size = 10;
	vector<porthosSecretType> r(size);
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
			cout << (int) beta[i] << endl;

}

void debugMax()
{
	size_t rows = 1;
	size_t columns = 10;
	vector<porthosSecretType> a(rows*columns, 0);

	if (partyNum == PARTY_A or partyNum == PARTY_C){
		a[0] = 0; a[1] = 1; a[2] = 0; a[3] = 4; a[4] = 5;
		a[5] = 3; a[6] = 10; a[7] = 6, a[8] = 41; a[9] = 9;
	}

	vector<porthosSecretType> max(rows), maxIndex(rows);
	funcMaxMPC(a, max, maxIndex, rows, columns);

	if (PRIMARY)
	{
		funcReconstruct2PC(a, columns, "a");
		funcReconstruct2PC(max, rows, "max");
		funcReconstruct2PC(maxIndex, rows, "maxIndex");
		cout << "-----------------" << endl;
	}
}

void debugSS()
{
	size_t size = 10;
	vector<porthosSecretType> inputs(size, 0), outputs(size, 0);

	vector<porthosSecretType> selector(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			selector[i] = (porthosSecretType)(aes_indep->getBit() << FLOAT_PRECISION);

	if (PRIMARY)
		funcReconstruct2PC(selector, size, "selector");

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			inputs[i] = (porthosSecretType)aes_indep->get8Bits();

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

	vector<porthosSecretType> a(rows*common_dim);
	vector<porthosSecretType> b(common_dim*columns);
	vector<porthosSecretType> c(rows*columns);

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
	vector<porthosSecretType> inputs(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			inputs[i] = aes_indep->get8Bits() - aes_indep->get8Bits();

	vector<porthosSecretType> outputs(size, 0);
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

	vector<porthosSecretType> maxIndex(rows, 0);
	if (partyNum == PARTY_A)
		for (size_t i = 0; i < rows; ++i)
			maxIndex[i] = (aes_indep->get8Bits())%columns;

	vector<porthosSecretType> a(rows*columns);
	funcMaxIndexMPC(a, maxIndex, rows, columns);

	if (PRIMARY)
	{
		funcReconstruct2PC(maxIndex, maxIndex.size(), "maxIndex");

		vector<porthosSecretType> temp(rows*columns);
		if (partyNum == PARTY_B)
			sendVector<porthosSecretType>(a, PARTY_A, rows*columns);

		if (partyNum == PARTY_A)
		{
			receiveVector<porthosSecretType>(temp, PARTY_B, rows*columns);
			addVectors<porthosSecretType>(temp, a, temp, rows*columns);

			cout << "a: " << endl;
			for (size_t i = 0; i < rows; ++i)
			{
				for (int j = 0; j < columns; ++j)
				{
					print_linear(temp[i*columns + j], DEBUG_PRINT);
				}
				cout << endl;
			}
			cout << endl;
		}
	}
}


/******************************** Testing functions ********************************/
void testMatMul(size_t rows, 
		size_t common_dim, 
		size_t columns, 
		size_t iter)
{
	vector<porthosSecretType> a(rows*common_dim, 1);
	vector<porthosSecretType> b(common_dim*columns, 1);
	vector<porthosSecretType> c(rows*columns);

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
}

void testRelu(size_t r, 
		size_t c, 
		size_t iter)
{
	vector<porthosSecretType> a(r*c, 1);
	vector<smallType> reluPrimeSmall(r*c, 1);
	vector<porthosSecretType> reluPrimeLarge(r*c, 1);
	vector<porthosSecretType> b(r*c, 0);

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
	vector<porthosSecretType> a(r*c, 1);
	vector<porthosSecretType> b(r*c, 0);
	vector<smallType> d(r*c, 0);

	for (int runs = 0; runs < iter; ++runs)
	{
		if (STANDALONE)
			for (size_t i = 0; i < r*c; ++i)
				b[i] = (a[i] < LARGEST_NEG ? 1:0);
		else if(MPC)
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
	size_t B = TEST_BATCH_SIZE;
	size_t size_x = p_range*q_range*D*B;

	vector<porthosSecretType> y(size_x, 0);
	vector<porthosSecretType> maxPoolShaped(size_x, 0);
	vector<porthosSecretType> act(size_x/(px*py), 0);
	vector<porthosSecretType> maxIndex(size_x/(px*py), 0);

	for (size_t i = 0; i < iter; ++i)
	{
		maxPoolReshape(y, maxPoolShaped, p_range, q_range, D, B, py, px, py, px);

		if (STANDALONE)
		{
			size_t size = (size_x/(px*py))*(px*py);
			vector<porthosSecretType> diff(size);

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

		if (MPC){
			funcMaxMPC(maxPoolShaped, act, maxIndex, size_x/(px*py), px*py);
		}
	}
}

void testMaxPoolDerivative(size_t p_range, 
		size_t q_range, 
		size_t px, 
		size_t py, 
		size_t D, 
		size_t iter)
{
	size_t B = TEST_BATCH_SIZE;
	size_t alpha_range = p_range/py;
	size_t beta_range = q_range/px;
	size_t size_y = (p_range*q_range*D*B);
	vector<porthosSecretType> deltaMaxPool(size_y, 0);
	vector<porthosSecretType> deltas(size_y/(px*py), 0);
	vector<porthosSecretType> maxIndex(size_y/(px*py), 0);

	size_t size_delta = alpha_range*beta_range*D*B;
	vector<porthosSecretType> thatMatrixTemp(size_y, 0), thatMatrix(size_y, 0);


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
		vector<porthosSecretType> largerDelta(size_y, 0);
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

void testComputeMSB(size_t size)
{
	vector<porthosSecretType> a(size, 0);

	if (partyNum == PARTY_A)
		for (size_t i = 0; i < size; ++i)
			a[i] = i - (size/2);

	if (THREE_PC)
	{
		vector<porthosSecretType> c(size);
		funcComputeMSB3PC(a, c, size);
	}
}

void testShareConvert(size_t size)
{
	vector<porthosSecretType> a(size, 0);

	if (partyNum == PARTY_A)
	{
		for (size_t i = 0; i < size; ++i)
		{
			a[i] = i - (size/2);
			if (a[i] == -1){
				a[i] -= 1;
			}
		}
	}

	funcShareConvertMPC(a, size);
}

/***************************************** End testing functions *****************************/

