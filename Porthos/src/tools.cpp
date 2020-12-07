/*
 * 	BMR_BGW_aux.cpp
 * 
 *      Author: Aner Ben-Efraim, Satyanarayana
 * 	
 * 	year: 2016
 *	
 *	Modified for crypTFlow. 
 */

#include "tools.h"
#include <stdint.h>
#include <mutex> 
#include <bitset>

//If Porthos compiled with Eigen support
#ifdef USE_EIGEN
#include <Eigen/Dense>
using namespace Eigen;
#endif

using namespace std;

#define NANOSECONDS_PER_SEC 1E9

//For time measurements
clock_t tStart;
struct timespec requestStart, requestEnd;
bool alreadyMeasuringTime = false;

int roundComplexitySend = 0;
int roundComplexityRecv = 0;
bool alreadyMeasuringRounds = false;

extern void aggregateCommunication();
extern CommunicationObject commObject;

/************************************ Some statistics functions ***********************/

void start_time()
{
	if (alreadyMeasuringTime)
	{
		cout << "Nested timing measurements" << endl;
		exit(-1);
	}

	tStart = clock();
	clock_gettime(CLOCK_REALTIME, &requestStart);
	alreadyMeasuringTime = true;
}

double diff(timespec start, timespec end)
{
    timespec temp;

    if ((end.tv_nsec-start.tv_nsec)<0)
    {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    }
    else 
    {
            temp.tv_sec = end.tv_sec-start.tv_sec;
            temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp.tv_sec + (double)temp.tv_nsec/NANOSECONDS_PER_SEC;
}


void end_time()
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_time() never called" << endl;
		exit(-1);
	}

	clock_gettime(CLOCK_REALTIME, &requestEnd);
#ifdef RUNTIME_DETAILED
	cout << "------------------------------------" << endl;
	cout << "Wall Clock time for execution: " << diff(requestStart, requestEnd) << " sec\n";
	cout << "CPU time for execution: " << (double)(clock() - tStart)/CLOCKS_PER_SEC << " sec\n";
#endif
	cout << "------------------------------------" << endl;	
	alreadyMeasuringTime = false;
}

void start_rounds()
{
	if (alreadyMeasuringRounds)
	{
		cout << "Nested round measurements" << endl;
		exit(-1);
	}

	roundComplexitySend = 0;
	roundComplexityRecv = 0;
	alreadyMeasuringRounds = true;
}

void end_rounds()
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_rounds() never called" << endl;
		exit(-1);
	}

	cout << "------------------------------------" << endl;
	cout << "Send Round Complexity of execution: " << roundComplexitySend << endl;
	cout << "Recv Round Complexity of execution: " << roundComplexityRecv << endl;
	cout << "------------------------------------" << endl;	
	alreadyMeasuringRounds = false;
}

//This function should be useed for some layer wise timing analysis
#if (LOG_LAYERWISE)
void analyseComputeVsCommTime()
{
/*
	assert(commObject.dataSent.size() == commObject.timeInSending.size());
	assert(commObject.dataReceived.size() == commObject.timeInReceiving.size());
*/

	double totalTimeInSending = commObject.totalTimeInSending;
	double totalTimeInReceiving = commObject.totalTimeInReceiving;
	porthosLongUnsignedInt totalDataSent = commObject.totalDataSent;
	porthosLongUnsignedInt totalDataReceived = commObject.totalDataReceived;

/*
	for(int i=0,end=commObject.dataSent.size();i<end;i++){
		totalDataSent += commObject.dataSent[i];
		totalTimeInSending += commObject.timeInSending[i];
	}
	for(int i=0,end=commObject.dataReceived.size();i<end;i++){
		totalDataReceived += commObject.dataReceived[i];
		totalTimeInReceiving += commObject.timeInReceiving[i];
	}
*/
	cout<<"Total data sent = "<<totalDataSent<<", total time in sending = "<<totalTimeInSending<<endl;
	cout<<"Total data received = "<<totalDataReceived<<", total time in receving = "<<totalTimeInReceiving<<endl;
	cout<<"Average b/w in sending = "<<(totalDataSent/(1000*1000*totalTimeInSending))<<" MBps"<<endl;
	cout<<"Average b/w in receiving = "<<(totalDataReceived/(1000*1000*totalTimeInReceiving))<<" MBps"<<endl;
	cout<<"Min size sent = "<<commObject.minSizeSent<<endl;

	cout<<"matmul time comm, matmul time total, data sent, data received = "
		<<commObject.timeMatmul[0]<<" "
		<<commObject.timeMatmul[1]<<" "
		<<(commObject.dataMatmul[0]/(1000.0*1000.0))<<" "
		<<(commObject.dataMatmul[1]/(1000.0*1000.0))<<endl;

	cout<<"relu time, data sent, data received = "
		<<commObject.timeRelu<<" "
		<<(commObject.dataRelu[0]/(1000.0*1000.0))<<" "
		<<(commObject.dataRelu[1]/(1000.0*1000.0))<<endl;

	cout<<"maxpool time, data sent, data received = "
		<<commObject.timeMaxpool<<" "
		<<(commObject.dataMaxPool[0]/(1000.0*1000.0))<<" "
		<<(commObject.dataMaxPool[1]/(1000.0*1000.0))<<endl;

	cout<<"avgpool time, data sent, data received = "
		<<commObject.timeAvgPool<<" "
		<<(commObject.dataAvgPool[0]/(1000.0*1000.0))<<" "
		<<(commObject.dataAvgPool[1]/(1000.0*1000.0))<<endl;

	cout<<"bn time, data sent, data received = "
		<<commObject.timeBN<<" "
		<<(commObject.dataBN[0]/(1000.0*1000.0))<<" "
		<<(commObject.dataBN[1]/(1000.0*1000.0))<<endl;
}
#endif

//Aggregate some stats about ongoing communication
void aggregateCommunication()
{
	vector<size_t> vec(4, 0), temp(4, 0);
	vec[0] = commObject.getSent();
	vec[1] = commObject.getRecv();
	vec[2] = commObject.getRoundsSent();
	vec[3] = commObject.getRoundsRecv();

	if (partyNum == PARTY_B or partyNum == PARTY_C)
		sendVector<size_t>(vec, PARTY_A, 4);

	if (partyNum == PARTY_A)
	{
		receiveVector<size_t>(temp, PARTY_B, 4);
		addVectors<size_t>(vec, temp, vec, 4);
		receiveVector<size_t>(temp, PARTY_C, 4);
		addVectors<size_t>(vec, temp, vec, 4);
	}

	if (partyNum == PARTY_A)
	{
		cout << "------------------------------------" << endl;
		cout << "Total communication: " << (double)vec[0]/1000000 << "MB (sent) and " << (double)vec[1]/1000000 << "MB (recv)\n";
		cout << "Total #calls: " << vec[2] << " (sends) and " << vec[3] << " (recvs)" << endl;
		cout << "------------------------------------" << endl;
	}
}

void start_m()
{
	cout << endl;
	start_time();
	start_communication();
}

void end_m()
{
	end_time();
	pause_communication();
	aggregateCommunication();
#if (LOG_LAYERWISE)
	analyseComputeVsCommTime();
#endif
	end_communication();
}

/************************************ Some MatMul functions ************************/

// If Eigen is not to be used for matmul
#ifndef USE_EIGEN 
void matrixMultEigen(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c, 			
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0 && "Transpose support not added yet");
	int ta, tc;
	porthosSecretType cumu_ele;
	for(int r=0; r<rows; r++){
		ta = r*common_dim;
		tc = r*columns;
		for(int col=0; col<columns; col++){
			cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += a[ta + com] * b[com*columns + col];
			}
			c[tc + col] = cumu_ele;
		}
	}
	
}

void matrixMultEigen(const vector<vector<porthosSecretType>> &a, 
		const vector<vector<porthosSecretType>> &b, 
		vector<vector<porthosSecretType>> &c, 			
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0 && "Transpose support not added yet");
	porthosSecretType cumu_ele;
	for(int r=0; r<rows; r++){
		for(int col=0; col<columns; col++){
			cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += a[r][com] * b[com][col];
			}
			c[r][col] = cumu_ele;
		}
	}
	
}

void matrixMultEigen(porthosSecretType* a, 
		porthosSecretType* b, 
		porthosSecretType* c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b)
{
	assert(transpose_a == 0 && transpose_b == 0 && "Transpose support not added yet");
	int ta, tc;
	porthosSecretType cumu_ele;
	for(int r=0; r<rows; r++){
		ta = r*common_dim;
		tc = r*columns;
		for(int col=0; col<columns; col++){
			cumu_ele = 0;
			for(int com=0; com<common_dim; com++){
				cumu_ele += a[ta + com] * b[com*columns + col];
			}
			c[tc + col] = cumu_ele;
		}
	}
	
}

#else
//Use Eigen MatMul
void matrixMultEigen(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<porthosSecretType> &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b)
{
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_a(rows, common_dim);
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_b(common_dim, columns);
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_c(rows, columns);

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < common_dim; ++j)
		{
			if (transpose_a)
				eigen_a(i, j) = a[j*rows + i];
			else
				eigen_a(i, j) = a[i*common_dim + j];
		}
	}

	for (size_t i = 0; i < common_dim; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			if (transpose_b)
				eigen_b(i, j) = b[j*common_dim + i];	
			else
				eigen_b(i, j) = b[i*columns + j];	
		}
	}

	eigen_c = eigen_a * eigen_b;

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
				c[i*columns + j] = eigen_c(i,j);
}


void matrixMultEigen(const vector< vector<porthosSecretType> > &a, 
		const vector< vector<porthosSecretType> > &b, 
		vector< vector<porthosSecretType> > &c, 
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b)
{
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_a(rows, common_dim);
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_b(common_dim, columns);
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_c(rows, columns);

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < common_dim; ++j)
		{
			if (transpose_a)
				eigen_a(i, j) = a[j][i];
			else
				eigen_a(i, j) = a[i][j];
		}
	}

	for (size_t i = 0; i < common_dim; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			if (transpose_b)
				eigen_b(i, j) = b[j][i];
			else
				eigen_b(i, j) = b[i][j];	
		}
	}

	eigen_c = eigen_a * eigen_b;

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
				c[i][j] = eigen_c(i,j);
}

void matrixMultEigen(porthosSecretType* a, 
		porthosSecretType* b, 
		porthosSecretType* c,
		size_t rows, 
		size_t common_dim, 
		size_t columns,
		size_t transpose_a, 
		size_t transpose_b)
{
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_a(rows, common_dim);
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_b(common_dim, columns);
	Matrix<porthosSecretType, Dynamic, Dynamic> eigen_c(rows, columns);

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < common_dim; ++j)
		{
			if (transpose_a)
				eigen_a(i, j) = Arr2DIdx(a, rows, common_dim, j, i);
			else
				eigen_a(i, j) = Arr2DIdx(a, rows, common_dim, i, j);
		}
	}

	for (size_t i = 0; i < common_dim; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			if (transpose_b)
				eigen_b(i, j) = Arr2DIdx(b, common_dim, columns, j, i);
			else
				eigen_b(i, j) = Arr2DIdx(b, common_dim, columns, i, j);
		}
	}

	eigen_c = eigen_a * eigen_b;

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
				Arr2DIdx(c, rows, columns, i, j) = eigen_c(i,j);
}

#endif

/*************************************** End of MatMul functions ***********************************/

/*************************************** Some other STANDALONE EXECTION utility functions **************************/

porthosSecretType divideMyTypeSA(porthosSecretType a, 
		porthosSecretType b)
{
	assert((sizeof(double) == sizeof(porthosSecretType)) && "sizeof(double) != sizeof(porthosSecretType)");
	assert((b != 0) && "Cannot divide by 0");
	return floatToMyType((double)((porthosLongSignedInt)a)/(double)((porthosLongSignedInt)b));
}

porthosSecretType dividePlainSA(porthosSecretType a, 
		int b)
{
	assert((b != 0) && "Cannot divide by 0");
	return static_cast<porthosSecretType>(static_cast<porthosLongSignedInt>(a)/static_cast<porthosLongSignedInt>(b));
}

void dividePlainSA(vector<porthosSecretType> &vec, 
		int divisor)
{
	assert((sizeof(double) == sizeof(porthosSecretType)) && "sizeof(double) != sizeof(porthosSecretType)");
	assert((divisor != 0) && "Cannot divide by 0");
	for (int i = 0; i < vec.size(); ++i)
		vec[i] = (porthosSecretType)((double)((porthosLongSignedInt)vec[i])/(double)((porthosLongSignedInt)divisor)); 	
}

porthosSecretType multiplyMyTypesSA(porthosSecretType a, 
		porthosSecretType b, 
		int shift)
{
	porthosSecretType ret;
	ret = static_cast<porthosSecretType>((static_cast<porthosLongSignedInt>(a) * static_cast<porthosLongSignedInt>(b))/ (1 << shift));
	return ret;
}

/*************************************** Other small utility functions ************************************/

void XORVectors(const vector<smallType> &a, 
		const vector<smallType> &b, 
		vector<smallType> &c, 
		size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] ^ b[i];
}


void log_print(string str)
{
#if (LOG_DEBUG)
	cout << "----------------------------" << endl;
	cout << "Started " << str << " at " << getCurrentTime() << endl;
	cout << "----------------------------" << endl;	
#endif
}

void error(string str)
{
	cout << "Error: " << str << endl;
	exit(-1);
}

size_t adversary(size_t party)
{
	size_t ret;

	switch(party)
	{
    		case PARTY_A : ret = PARTY_B;
             			break;       
    		case PARTY_B : ret = PARTY_A;
             			break;
	}	
	return ret;
}

smallType subtractModPrime(smallType a, 
		smallType b)
{
	if (b == 0)
		return a;
	else 
	{
		b = (PRIME_NUMBER - b); 
		return additionModPrime[a][b];
	}
}

void wrapAround(const vector<porthosSecretType> &a, 
		const vector<porthosSecretType> &b, 
		vector<smallType> &c, 
		size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = wrapAround(a[i], b[i]);
}

/************************************* Some functions with AES and resharing ****************************/

void populateBitsVector(vector<smallType> &vec, 
		string r_type, 
		size_t size)
{
	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for populateBitsVector");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_common->getBit();
	}

	if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_indep->getBit();
	}
}

//Returns shares of MSB...LSB of first number and so on.  
void sharesOfBits(vector<smallType> &bit_shares_x_1, 
		vector<smallType> &bit_shares_x_2, 
		const vector<porthosSecretType> &x, 
		size_t size, 
		string r_type)
{
#ifdef DEBUG
	assert((r_type == "PRG_COMM_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfBits");
	assert(partyNum == PARTY_C);
#endif
	smallType temp;

	if (r_type == "COMMON")
	{
#ifdef PRECOMPUTEAES
		aes_common->fillWithRandomModuloPrimeBits(bit_shares_x_1.data(), size*BIT_SIZE);
#else
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				bit_shares_x_1[i*BIT_SIZE + k] = aes_common->randModPrime();
			}
		}
#endif
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), bit_shares_x_1[i*BIT_SIZE + k]);
			}
		}
	}

	else if (r_type == "INDEP")
	{
#ifdef PRECOMPUTEAES
		aes_indep->fillWithRandomModuloPrimeBits(bit_shares_x_1.data(), size*BIT_SIZE);
#else
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				bit_shares_x_1[i*BIT_SIZE + k] = aes_indep->randModPrime();
			}
		}
#endif
		for (size_t i = 0; i < size; ++i)
		{
			for (size_t k = 0; k < BIT_SIZE; ++k)
			{
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), bit_shares_x_1[i*BIT_SIZE + k]);
			}
		}
	}

	else if (r_type == "PRG_COMM_OPTI")
	{
#ifdef PRECOMPUTEAES
		aes_share_conv_bit_shares_p0_p2->fillWithRandomModuloPrimeBits(bit_shares_x_1.data(), (size/2)*BIT_SIZE);
		aes_share_conv_bit_shares_p1_p2->fillWithRandomModuloPrimeBits(bit_shares_x_2.data() + (size/2)*BIT_SIZE, (size - (size/2))*BIT_SIZE);
#else
		for(size_t i = 0; i < (size/2); i++)
		{
			for(size_t k=0; k<BIT_SIZE; k++)
			{
				bit_shares_x_1[i*BIT_SIZE + k] = aes_share_conv_bit_shares_p0_p2->randModPrime();
			}
		}
		for(size_t i = (size/2); i < size; i++)
		{
			for(size_t k=0; k<BIT_SIZE; k++)
			{
				bit_shares_x_2[i*BIT_SIZE + k] = aes_share_conv_bit_shares_p1_p2->randModPrime();
			}
		}
#endif

		for(size_t i = 0; i < (size/2); i++)
		{
			for(size_t k=0; k<BIT_SIZE; k++)
			{
				bit_shares_x_2[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), bit_shares_x_1[i*BIT_SIZE + k]);
			}
		}
		for(size_t i = (size/2); i < size; i++)
		{
			for(size_t k=0; k<BIT_SIZE; k++)
			{
				bit_shares_x_1[i*BIT_SIZE + k] = subtractModPrime((x[i] >> (BIT_SIZE - 1 - k) & 1), bit_shares_x_2[i*BIT_SIZE + k]);
			}
		}
	}
}

//Returns boolean shares of LSB of r.  
void sharesOfLSB(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<porthosSecretType> &r, 
		size_t size, 
		string r_type)
{
#ifdef DEBUG
	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for sharesOfLSB");
	assert(partyNum == PARTY_C);
#endif

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->getBit();
			share_2[i] = share_1[i] ^ (r[i] % 2);
		}
	}

	else if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->getBit();
			share_2[i] = share_1[i] ^ (r[i] % 2);
		}
	}
}

//Returns \Z_L shares of LSB of r.  
void sharesOfLSB(vector<porthosSecretType> &share_1, 
		vector<porthosSecretType> &share_2, 
		const vector<porthosSecretType> &r, 
		size_t size, 
		string r_type)
{
#ifdef DEBUG
	assert((r_type == "PRG_COMM_OPTI" || (r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfLSB");
	assert(partyNum == PARTY_C);
#endif 

	if (r_type == "COMMON")
	{
#ifdef PRECOMPUTEAES
		aes_common->fillWithRandomBits64(share_1.data(), size);
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}
#else	
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->get64Bits();
		}
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}
#endif
	}

	else if (r_type == "INDEP")
	{
#ifdef PRECOMPUTEAES
		aes_indep->fillWithRandomBits64(share_1.data(), size);
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}
#else
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->get64Bits();
		}
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}
#endif
	}

	else if(r_type == "PRG_COMM_OPTI"){
#ifdef PRECOMPUTEAES
		aes_comp_msb_shares_lsb_p0_p2->fillWithRandomBits64(share_1.data(), (size/2));
		aes_comp_msb_shares_lsb_p1_p2->fillWithRandomBits64(share_2.data() + (size/2), (size-(size/2)));
#else
		for (size_t i = 0; i < (size/2); ++i)
		{
			share_1[i] = aes_comp_msb_shares_lsb_p0_p2->get64Bits();
		}
		for (size_t i = (size/2); i < size; ++i)
		{
			share_2[i] = aes_comp_msb_shares_lsb_p1_p2->get64Bits();
		}
#endif

		for (size_t i = 0; i < (size/2); ++i)
		{
			share_2[i] = floatToMyType(r[i] % 2) - share_1[i];
		}
		for (size_t i = (size/2); i < size; ++i)
		{
			share_1[i] = floatToMyType(r[i] % 2) - share_2[i];
		}
	}
}

//Returns boolean shares of a bit vector vec.  
void sharesOfBitVector(vector<smallType> &share_1, 
		vector<smallType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type)
{
	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for sharesOfLSB");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->getBit();
			share_2[i] = share_1[i] ^ vec[i];
		}
	}

	if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->getBit();
			share_2[i] = share_1[i] ^ vec[i];
		}
	}
}

//Returns \Z_L shares of a bit vector vec.  
void sharesOfBitVector(vector<porthosSecretType> &share_1, 
		vector<porthosSecretType> &share_2, 
		const vector<smallType> &vec, 
		size_t size, 
		string r_type)
{
#ifdef DEBUG
	assert((r_type == "PRG_COMM_OPTI" ||(r_type == "COMMON" or r_type == "INDEP")) && "invalid randomness type for sharesOfLSB");
	assert(partyNum == PARTY_C);
#endif

	if (r_type == "COMMON")
	{
#ifdef PRECOMPUTEAES
		aes_common->fillWithRandomBits64(share_1.data(), size);
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
#else
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_common->get64Bits();
		}
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
#endif
	}
	else if (r_type == "INDEP")
	{
#ifdef PRECOMPUTEAES
		aes_indep->fillWithRandomBits64(share_1.data(), size);		
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
#else
		for (size_t i = 0; i < size; ++i)
		{
			share_1[i] = aes_indep->get64Bits();
		}
		for (size_t i = 0; i < size; ++i)
		{
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
#endif
	}
	else if(r_type == "PRG_COMM_OPTI"){
#ifdef PRECOMPUTEAES
		aes_comp_msb_shares_bit_vec_p0_p2->fillWithRandomBits64(share_1.data(), (size/2));
		aes_comp_msb_shares_bit_vec_p1_p2->fillWithRandomBits64(share_2.data() + (size/2), (size-(size/2)));
#else
		for (size_t i = 0; i < (size/2); ++i)
		{
			share_1[i] = aes_comp_msb_shares_bit_vec_p0_p2->get64Bits();
		}
		for (size_t i = (size/2); i < size; ++i)
		{
			share_2[i] = aes_comp_msb_shares_bit_vec_p1_p2->get64Bits();
		}
#endif

		for (size_t i = 0; i < (size/2); ++i)
		{
			share_2[i] = floatToMyType(vec[i]) - share_1[i];
		}
		
		for (size_t i = (size/2); i < size; ++i)
		{
			share_1[i] = floatToMyType(vec[i]) - share_2[i];
		}
	}
}

//Split shares of a vector of porthosSecretType into shares (randomness is independent)
void splitIntoShares(const vector<porthosSecretType> &a, 
		vector<porthosSecretType> &a1, 
		vector<porthosSecretType> &a2, 
		size_t size)
{
	populateRandomVector<porthosSecretType>(a1, size, "INDEP", "POSITIVE");
	subtractVectors<porthosSecretType>(a, a1, a2, size);
}

/***************************** Basic utility functions for Convolution drivers ************************/

void zero_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& vec, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					vec[i][j][k][l] = 0;
				}
			}
		}
	}
}

void subtract_2D_vectors(const vector< vector<porthosSecretType> >& inp_l, 
			const vector< vector<porthosSecretType> >& inp_r, 
			vector< vector<porthosSecretType> >& out, 
			int d1, 
			int d2)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			out[i][j] = inp_l[i][j] - inp_r[i][j];
		}
	}
}

void add_2D_vectors(const vector< vector<porthosSecretType> >& inp_l, 
		const vector< vector<porthosSecretType> >& inp_r, 
		vector< vector<porthosSecretType> >& out, 
		int d1, 
		int d2)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			out[i][j] = inp_l[i][j] + inp_r[i][j];
		}
	}
}						 

void zero_2D_vector(vector< vector<porthosSecretType> >& vec, 
		int d1, 
		int d2)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			vec[i][j] = 0;
		}
	}
}

void add_4D_vectors(vector< vector< vector< vector<porthosSecretType> > > >& inp_l, 
		vector< vector< vector< vector<porthosSecretType> > > >& inp_r, 
		vector< vector< vector< vector<porthosSecretType> > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					out[i][j][k][l] = inp_l[i][j][k][l] + inp_r[i][j][k][l];
				}
			}
		}
	}
}

void add_5D_vectors(vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_l, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_r, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4,
		int d5)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					for(int m=0; m<d5; m++){
						out[i][j][k][l][m] = inp_l[i][j][k][l][m] + inp_r[i][j][k][l][m];
					}
				}
			}
		}
	}
}

void subtract_4D_vectors(vector< vector< vector< vector<porthosSecretType> > > >& inp_l, 
			vector< vector< vector< vector<porthosSecretType> > > >& inp_r, 
			vector< vector< vector< vector<porthosSecretType> > > >& out, 
			int d1, 
			int d2, 
			int d3, 
			int d4)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					out[i][j][k][l] = inp_l[i][j][k][l] - inp_r[i][j][k][l];
				}
			}
		}
	}
}

void subtract_5D_vectors(vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_l, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& inp_r, 
		vector< vector< vector< vector< vector<porthosSecretType> > > > >& out, 
		int d1, 
		int d2, 
		int d3, 
		int d4,
		int d5)
{
	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					for(int m=0; m<d5; m++){
						out[i][j][k][l][m] = inp_l[i][j][k][l][m] - inp_r[i][j][k][l][m];
					}
				}
			}
		}
	}
}

void flatten_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& input, 
			vector<porthosSecretType>& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4, 
			int d5)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			for(int k=0;k<d3;k++){
				for(int l=0;l<d4;l++){
					for(int m=0;m<d5;m++){
						output[i*d2*d3*d4*d5 + j*d3*d4*d5 + k*d4*d5 + l*d5 + m] = input[i][j][k][l][m];
					}
				}
			}
		}
	}

}

void deflatten_5D_vector(vector<porthosSecretType>& input, 
			vector< vector< vector< vector< vector<porthosSecretType> > > > >& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4, 
			int d5)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			for(int k=0;k<d3;k++){
				for(int l=0;l<d4;l++){
					for(int m=0;m<d5;m++){
						output[i][j][k][l][m] = input[i*d2*d3*d4*d5 + j*d3*d4*d5 + k*d4*d5 + l*d5 + m];
					}
				}
			}
		}
	}

}

void flatten_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input, 
			vector<porthosSecretType>& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			for(int k=0;k<d3;k++){
				for(int l=0;l<d4;l++){
					output[i*d2*d3*d4 + j*d3*d4 + k*d4 + l] = input[i][j][k][l];
				}
			}
		}
	}

}

void deflatten_4D_vector(vector<porthosSecretType>& input, 
			vector< vector< vector< vector<porthosSecretType> > > >& output, 
			int d1, 
			int d2, 
			int d3, 
			int d4)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			for(int k=0;k<d3;k++){
				for(int l=0;l<d4;l++){
					output[i][j][k][l] = input[i*d2*d3*d4 + j*d3*d4 + k*d4 + l];
				}
			}
		}
	}

}

void flatten_2D_vector(vector< vector<porthosSecretType> >& input, 
			vector<porthosSecretType>& output, 
			int d1, 
			int d2)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			output[i*d2 + j] = input[i][j];
		}
	}

}

void deflatten_2D_vector(vector<porthosSecretType>& input, 
			vector< vector<porthosSecretType> >& output, 
			int d1, 
			int d2)
{
	for(int i=0;i<d1;i++){
		for(int j=0;j<d2;j++){
			output[i][j] = input[i*d2 + j];
		}
	}

}

void send_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& input,
			int party,
                        int d1,
                        int d2,
                        int d3,
                        int d4,
                        int d5)
{
	vector<porthosSecretType> flat_input(d1*d2*d3*d4*d5, 0);

	//Flatten and send.
	flatten_5D_vector(input, flat_input, d1, d2, d3, d4, d5);

	sendVector<porthosSecretType>(ref(flat_input), party, d1*d2*d3*d4*d5);

}

void receive_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& recv,
			int party,
                        int d1,
                        int d2,
                        int d3,
                        int d4,
                        int d5)
{
	vector<porthosSecretType> flat_recv(d1*d2*d3*d4*d5, 0);

	//Receive and deflatten.
	receiveVector<porthosSecretType>(ref(flat_recv), party, d1*d2*d3*d4*d5);

	deflatten_5D_vector(flat_recv, recv, d1, d2, d3, d4, d5);

}


void send_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input, 
			int d1, 
			int d2, 
			int d3, 
			int d4)
{
	vector<porthosSecretType> flat_input(d1*d2*d3*d4, 0);

	//Flatten and send.
	flatten_4D_vector(input, flat_input, d1, d2, d3, d4);

	sendVector<porthosSecretType>(ref(flat_input), adversary(partyNum), d1*d2*d3*d4);

}

void receive_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& recv, 
			int d1, 
			int d2, 
			int d3, 
			int d4)
{
	vector<porthosSecretType> flat_recv(d1*d2*d3*d4, 0);

	//Receive and deflatten.
	receiveVector<porthosSecretType>(ref(flat_recv), adversary(partyNum), d1*d2*d3*d4);

	deflatten_4D_vector(flat_recv, recv, d1, d2, d3, d4);

}

void send_2_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input1,
			vector< vector< vector< vector<porthosSecretType> > > >& input2,
			int d11, 
			int d12, 
			int d13, 
			int d14,
			int d21, 
			int d22, 
			int d23, 
			int d24)
{
	int32_t len1 = d11*d12*d13*d14;
	int32_t len2 = d21*d22*d23*d24;
	vector<porthosSecretType> sendArr(len1 + len2, 0);
	for(int i=0;i<d11;i++){
		for(int j=0;j<d12;j++){
			for(int k=0;k<d13;k++){
				for(int l=0;l<d14;l++){
					sendArr[i*d12*d13*d14 + j*d13*d14 + k*d14 + l] = input1[i][j][k][l];
				}
			}
		}
	}
	for(int i=0;i<d21;i++){
		for(int j=0;j<d22;j++){
			for(int k=0;k<d23;k++){
				for(int l=0;l<d24;l++){
					sendArr[len1 + i*d22*d23*d24 + j*d23*d24 + k*d24 + l] = input2[i][j][k][l];
				}
			}
		}
	}
	sendVector<porthosSecretType>(ref(sendArr), adversary(partyNum), len1+len2);
}

void receive_2_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& input1,
			vector< vector< vector< vector<porthosSecretType> > > >& input2,
			int d11, 
			int d12, 
			int d13, 
			int d14,
			int d21, 
			int d22, 
			int d23, 
			int d24)
{
	int32_t len1 = d11*d12*d13*d14;
	int32_t len2 = d21*d22*d23*d24;
	vector<porthosSecretType> rcvArr(len1 + len2, 0);
	receiveVector<porthosSecretType>(ref(rcvArr), adversary(partyNum), len1+len2);
	for(int i=0;i<d11;i++){
		for(int j=0;j<d12;j++){
			for(int k=0;k<d13;k++){
				for(int l=0;l<d14;l++){
					input1[i][j][k][l] = rcvArr[i*d12*d13*d14 + j*d13*d14 + k*d14 + l];
				}
			}
		}
	}
	for(int i=0;i<d21;i++){
		for(int j=0;j<d22;j++){
			for(int k=0;k<d23;k++){
				for(int l=0;l<d24;l++){
					input2[i][j][k][l] = rcvArr[len1 + i*d22*d23*d24 + j*d23*d24 + k*d24 + l];
				}
			}
		}
	}
}

void send_2D_vector(vector< vector<porthosSecretType> >& input, 
			int d1, 
			int d2)
{
	vector<porthosSecretType> flat_input(d1*d2, 0);

	//Flatten and send.
	flatten_2D_vector(input, flat_input, d1, d2);

	sendVector<porthosSecretType>(ref(flat_input), PARTY_B, d1*d2);

}

void receive_2D_vector(vector< vector<porthosSecretType> >& recv, 
			int d1, 
			int d2)
{
	vector<porthosSecretType> flat_recv(d1*d2, 0);

	//Receive and deflatten.
	receiveVector<porthosSecretType>(ref(flat_recv), PARTY_C, d1*d2);

	deflatten_2D_vector(flat_recv, recv, d1, d2);

}

void populate_4D_vector(vector< vector< vector< vector<porthosSecretType> > > >& vec, 
			int d1, 
			int d2, 
			int d3, 
			int d4, 
			string type)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = aes_conv_opti_a_1;
	else if(type == "a2") aesObject = aes_conv_opti_a_2;
	else if(type == "b1") aesObject = aes_conv_opti_b_1;
	else if(type == "b2") aesObject = aes_conv_opti_b_2;
	else if(type == "c1") aesObject = aes_conv_opti_c_1;
	else assert(false);

	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					vec[i][j][k][l] = aesObject->get64Bits();
				}
			}
		}
	}
}

void populate_5D_vector(vector< vector< vector< vector< vector<porthosSecretType> > > > >& vec, 
			int d1, 
			int d2, 
			int d3, 
			int d4, 
			int d5, 
			string type)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = aes_conv_opti_a_1;
	else if(type == "a2") aesObject = aes_conv_opti_a_2;
	else if(type == "b1") aesObject = aes_conv_opti_b_1;
	else if(type == "b2") aesObject = aes_conv_opti_b_2;
	else if(type == "c1") aesObject = aes_conv_opti_c_1;
	else assert(false);

	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			for(int k=0; k<d3; k++){
				for(int l=0; l<d4; l++){
					for(int m=0; m<d5; m++){
						vec[i][j][k][l][m] = aesObject->get64Bits();
					}
				}
			}
		}
	}
}

void populate_2D_vector(vector< vector<porthosSecretType> >& vec, 
			int d1, 
			int d2, 
			string type)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = aes_conv_opti_a_1;
	else if(type == "a2") aesObject = aes_conv_opti_a_2;
	else if(type == "b1") aesObject = aes_conv_opti_b_1;
	else if(type == "b2") aesObject = aes_conv_opti_b_2;
	else if(type == "c1") aesObject = aes_conv_opti_c_1;
	else assert(false);

	for(int i=0; i<d1; i++){
		for(int j=0; j<d2; j++){
			vec[i][j] = aesObject->get64Bits();
		}
	}
}

void populate_AES_Arr(porthosSecretType* arr, 
		porthosLongUnsignedInt size, 
		string type)
{
	AESObject* aesObject;
	if(type == "a1") aesObject = aes_conv_opti_a_1;
	else if(type == "a2") aesObject = aes_conv_opti_a_2;
	else if(type == "b1") aesObject = aes_conv_opti_b_1;
	else if(type == "b2") aesObject = aes_conv_opti_b_2;
	else if(type == "c1") aesObject = aes_conv_opti_c_1;
	else assert(false);

#ifdef PRECOMPUTEAES
	aesObject->fillWithRandomBits64(arr, size);
#else
	for(porthosLongUnsignedInt i=0; i<size; i++){
		arr[i] = aesObject->get64Bits();
	}
#endif
}

void add_2_Arr(porthosSecretType* arr1, 
		porthosSecretType* arr2, 
		porthosSecretType* arr, 
		porthosLongUnsignedInt size)
{
	for(porthosLongUnsignedInt i=0; i<size; i++){
		arr[i] = arr1[i] + arr2[i];
	}
}

void subtract_2_Arr(porthosSecretType* arr1, 
		porthosSecretType* arr2, 
		porthosSecretType* arr, 
		porthosLongUnsignedInt size)
{
	for(porthosLongUnsignedInt i=0; i<size; i++){
		arr[i] = arr1[i] - arr2[i];
	}
}

/********************************** Porthos info display functions *****************************/

void porthos_throw_error(int code)
{
	switch(code){
		case PARSE_ERROR:
			cout<<endl<<"*********************PORTHOS SYSTEM: ERROR DETECTED********************"<<endl;
			cout<<"Usage: ./Porthos.out <PARTY_NUMBER> <PATH_IP_ADDRESSES> [ < <ANY_INPUT> ]"<<endl<<"For bugs, contact at mayankrathee.japan@gmail.com or nishant.kr10@gmail.com"<<endl<<endl;
			break;
		default:
			cout<<endl<<"*********************PORTHOS SYSTEM: ERROR DETECTED********************"<<endl;
	};
}

void show_porthos_mode()
{
	bool all_opti, share_conv_opti, msb_opti, conv_opti, precompute_aes, parallelize_crit_opti;
	cout<<endl<<"**********************PORTHOS MODE**********************"<<endl<<">>> Running Porthos in the following mode: "<<endl;
#ifndef PORTHOS_OPTIMIZATIONS
	cout<<"PORTHOS OPTIMIZATIONS        OFF"<<endl<<endl;
#else
#ifdef RUN_SHARECONV_OPTI
	cout<<"SHARE CONVERT OPTIMIZATION   ON"<<endl;
#else
	cout<<"SHARE CONVERT OPTIMIZATION   OFF"<<endl;
#endif
#ifdef RUN_MSB_OPTI
	cout<<"COMPUTE MSB OPTIMIZATION     ON"<<endl;
#else
	cout<<"COMPUTE MSB OPTIMIZATION     OFF"<<endl;
#endif
#ifdef CONV_OPTI
	cout<<"CONVOLUTION OPTIMIZATION     ON"<<endl;
#else
	cout<<"CONVOLUTION OPTIMIZATION     OFF"<<endl;
#endif
#ifdef PRECOMPUTEAES
	cout<<"PRECOMP. AES OPTIMIZATION    ON"<<endl;
#else
	cout<<"PRECOMP. AES OPTIMIZATION    OFF"<<endl;
#endif
#ifdef PARALLIZE_CRITICAL
	cout<<"PRIVATE COMP. OPTIMIZATION   ON"<<endl;
#else
	cout<<"PRIVATE COMP. OPTIMIZATION   OFF"<<endl;
#endif

#endif

#ifdef DEBUG
	cout<<"PORTHOS DEBUG BUILD(ASSERTS) ON"<<endl;
#else
	cout<<"PORTHOS DEBUG BUILD(ASSERTS) OFF"<<endl;
#endif
#ifdef USE_EIGEN
	cout<<"PORTHOS EIGEN MATMUL SUPPORT ON"<<endl;
#else
	cout<<"PORTHOS EIGEN MATMUL SUPPORT OFF"<<endl;
#endif

	if(sizeof(uint64_t) == sizeof(porthosSecretType)){
		cout<<"Running Porthos in Z_2^64"<<endl;
	}
	else{
		cout<<"Support for custom ring not available"<<endl;
	}
	cout<<endl;
}

/********************************** Some helper functions invoked only by test functions **************/

void print_linear(porthosSecretType var, 
		string type)
{
	if (type == "BITS")
		cout << bitset<64>(var) << " ";
	else if (type == "FLOAT")
		cout << (static_cast<porthosLongSignedInt>(var))/(float)(1 << FLOAT_PRECISION) << " ";
	else if (type == "SIGNED")
		cout << static_cast<porthosLongSignedInt>(var) << " ";
	else if (type == "UNSIGNED")
		cout << var << " ";	
}

void maxPoolReshape(const vector<porthosSecretType> &vec, 
		vector<porthosSecretType> &vecShaped,
		size_t ih, 
		size_t iw, 
		size_t D, 
		size_t B,  
		size_t fh, 
		size_t fw, 
		size_t sy, 
		size_t sx)
{
	assert(fw >= sx and fh >= sy && "Check implementation");
	assert((iw - fw)%sx == 0 && "Check implementations for this unmet condition");
	assert((ih - fh)%sy == 0 && "Check implementations for this unmet condition");
	assert(vec.size() == vecShaped.size() && "Dimension issue with convolutionReshape");

	size_t loc = 0, counter = 0;
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < D; ++j)
			for (size_t k = 0; k < ih-fh+1; k += sy) 
				for (size_t l = 0; l < iw-fw+1; l += sx)
				{
					loc = i*iw*ih*D + j*iw*ih + k*iw + l;
					for (size_t a = 0; a < fh; ++a)
						for (size_t b = 0; b < fw; ++b)
							vecShaped[counter++] = vec[loc + a*iw + b];
				}
}
