/*
 * Slightly modified for Aramis from
 * basicSockets.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: froike(Roi Inbar)
 *	Modified: Aner Ben-Efraim
 *
 * 	Modified by: Mayank Rathee
 */

#ifndef CONNECT_H
#define CONNECT_H

#include "basicSockets.h"
#include <sstream>
#include <vector>
#include "../util/TedKrovetzAesNiWrapperC.h"
#include <stdint.h>
#include <iomanip>
#include <fstream>
#include "globals.h"
#include "sgx_trts.h"
#include "../Enclave/Enclave_t.h"

using namespace std;

extern AramisNet ** communicationSenders;
extern AramisNet ** communicationReceivers;

extern int partyNum;

//setting up communication
void initCommunication(string addr, int port, int player, int mode);
void initializeCommunication(int* ports);
void initializeCommunicationSerial(int* ports); //Use this for many parties
void initializeCommunication(char* filename, int p);


//synchronization functions
void sendByte(int player, char* toSend, int length, int conn);
void receiveByte(int player, int length, int conn);
void synchronize(int length = 1);

void start_communication();
void pause_communication();
void resume_communication();
void end_communication(string str);


/****************************** These functions are defined in header itself ******************/

template<typename T>
void sendVector(const vector<T> &vec, 
		size_t player, 
		size_t size);

template<typename T>
void sendArr(T* arr, 
		size_t player, 
		size_t size);

template<typename T>
void receiveVector(vector<T> &vec, 
		size_t player, 
		size_t size);

template<typename T>
void receiveArr(T* arr, 
		size_t player, 
		size_t size);

template<typename T>
void sendTwoVectors(const vector<T> &vec1, 
		const vector<T> &vec2, 
		size_t player, 
		size_t size1, 
		size_t size2);

template<typename T>
void receiveTwoVectors(vector<T> &vec1, 
		vector<T> &vec2, 
		size_t player, 
		size_t size1, 
		size_t size2);

/************************ Function defined in header itself ****************************/

template<typename T>
void sendVector(const vector<T> &vec, size_t player, size_t size)
{
	communicationSenders[player]->sendMsg(vec.data(), size * sizeof(T), 0, player);
}

template<typename T>
void sendArr(T* arr, size_t player, size_t size)
{
	uint64_t bytesSent = size*sizeof(T);

	communicationSenders[player]->sendMsg(arr, size * sizeof(T), 0, player);
}

template<typename T>
void receiveVector(vector<T> &vec, size_t player, size_t size)
{
	communicationReceivers[player]->receiveMsg(vec.data(), size * sizeof(T), 0, player);
}

template<typename T>
void receiveArr(T* arr, size_t player, size_t size)
{
	uint64_t bytesReceived = size*sizeof(T);

	communicationReceivers[player]->receiveMsg(arr, size * sizeof(T), 0, player);		
}

template<typename T>
void sendTwoVectors(const vector<T> &vec1, const vector<T> &vec2, size_t player, size_t size1, size_t size2)
{
	vector<T> temp(size1+size2);
	for (size_t i = 0; i < size1; ++i)
		temp[i] = vec1[i];

	for (size_t i = 0; i < size2; ++i)
		temp[size1 + i] = vec2[i];

	sendVector<T>(temp, player, size1+size2);
}

template<typename T>
void receiveTwoVectors(vector<T> &vec1, vector<T> &vec2, size_t player, size_t size1, size_t size2)
{
	vector<T> temp(size1+size2);
	receiveVector<T>(temp, player, size1+size2);

	for (size_t i = 0; i < size1; ++i)
		vec1[i] = temp[i];

	for (size_t i = 0; i < size2; ++i)
		vec2[i] = temp[size1 + i];
}

#endif
