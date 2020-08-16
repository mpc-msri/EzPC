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

#ifndef CONNECT_H
#define CONNECT_H

#include "basicSockets.h"
#include <sstream>
#include <vector>
#include <stdint.h>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <string.h>

using namespace std;
using namespace std::chrono;

extern PorthosNet ** communicationSenders;
extern PorthosNet ** communicationReceivers;

extern int partyNum;

//setting up communication
void initCommunication(string addr, 
		int port, 
		int player, 
		int mode);

void initializeCommunication(int* ports);

void initializeCommunicationSerial(int* ports); //Use this for many parties

void initializeCommunication(char* filename, 
		int p);

//synchronization functions
void sendByte(int player, 
		char* toSend, 
		int length, 
		int conn);

void receiveByte(int player, 
		int length, 
		int conn);

void synchronize(int length = 1);

void start_communication();

void pause_communication();

void resume_communication();

void end_communication();

porthosLongUnsignedInt getCurrentTime();

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
void sendVector(const vector<T> &vec, 
		size_t player, 
		size_t size)
{
	porthosLongUnsignedInt bytesSent = size*sizeof(T);

#if (LOG_DEBUG)
	cout << "Sending " << bytesSent << " Bytes to player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType" << endl;
	else 
		cout << "smallType" << endl;
	
	auto t1 = high_resolution_clock::now();
#endif

	if(!communicationSenders[player]->sendMsg(vec.data(), size * sizeof(T), 0))
		cout << "Send vector error" << endl;

#if (LOG_DEBUG)
	
	auto t2 = high_resolution_clock::now();
	auto tt = (duration_cast<duration<double>>(t2 - t1)).count();

	cout << "Done Sending " << bytesSent << " Bytes to player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType";
	else 
		cout << "smallType";
	cout<<" in time "<<tt<<" seconds at speed = "<<(bytesSent/(1.0*tt*1000000))<<" MBps."<<endl;
#endif
}


template<typename T>
void sendArr(T* arr, 
		size_t player, 
		size_t size)
{
	porthosLongUnsignedInt bytesSent = size*sizeof(T);

#if (LOG_DEBUG)
	cout << "Sending " << bytesSent << " Bytes to player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType" << endl;
	else 
		cout << "smallType" << endl;
	
	auto t1 = high_resolution_clock::now();
#endif

	if(!communicationSenders[player]->sendMsg(arr, size * sizeof(T), 0))
		cout << "Send array error" << endl;

#if (LOG_DEBUG)
	
	auto t2 = high_resolution_clock::now();
	auto tt = (duration_cast<duration<double>>(t2 - t1)).count();

	cout << "Done Sending " << bytesSent << " Bytes to player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType";
	else 
		cout << "smallType";
	cout<<" in time "<<tt<<" seconds at speed = "<<(bytesSent/(1.0*tt*1000000))<<" MBps."<<endl;
#endif
}

template<typename T>
void receiveVector(vector<T> &vec, 
		size_t player, 
		size_t size)
{
	porthosLongUnsignedInt bytesReceived = size*sizeof(T);

#if (LOG_DEBUG)
	cout << "Receiving " << bytesReceived << " Bytes from player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType" << endl;
	else 
		cout << "smallType" << endl;

	auto t1 = high_resolution_clock::now();
#endif

	if(!communicationReceivers[player]->receiveMsg(vec.data(), size * sizeof(T), 0))
		cout << "Receive porthosSecretType vector error" << endl;
		
#if (LOG_DEBUG)
	
	auto t2 = high_resolution_clock::now();
	auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
	
	cout << "Done Receiving " << bytesReceived << " Bytes from player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType";
	else 
		cout << "smallType";
	cout<<" in time "<<tt<<" seconds at speed = "<<(bytesReceived/(1.0*tt*1000000))<<" MBps."<<endl;
#endif

}

template<typename T>
void receiveArr(T* arr, 
		size_t player, 
		size_t size)
{
	porthosLongUnsignedInt bytesReceived = size*sizeof(T);

#if (LOG_DEBUG)
	cout << "Receiving " << bytesReceived << " Bytes from player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType" << endl;
	else 
		cout << "smallType" << endl;

	auto t1 = high_resolution_clock::now();
#endif

	if(!communicationReceivers[player]->receiveMsg(arr, size * sizeof(T), 0))
		cout << "Receive porthosSecretType vector error" << endl;
		
#if (LOG_DEBUG)
	
	auto t2 = high_resolution_clock::now();
	auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
	
	cout << "Done Receiving " << bytesReceived << " Bytes from player " << player << " at "<<getCurrentTime()<< " via ";
	if (sizeof(T) == 8)
		cout << "porthosSecretType";
	else 
		cout << "smallType";
	cout<<" in time "<<tt<<" seconds at speed = "<<(bytesReceived/(1.0*tt*1000000))<<" MBps."<<endl;
#endif
}

template<typename T>
void sendTwoVectors(const vector<T> &vec1, 
		const vector<T> &vec2, 
		size_t player, 
		size_t size1, 
		size_t size2)
{
	vector<T> temp(size1+size2);
	for (size_t i = 0; i < size1; ++i)
		temp[i] = vec1[i];

	for (size_t i = 0; i < size2; ++i)
		temp[size1 + i] = vec2[i];

	sendVector<T>(temp, player, size1+size2);
}

template<typename T>
void receiveTwoVectors(vector<T> &vec1, 
		vector<T> &vec2, 
		size_t player, 
		size_t size1, 
		size_t size2)
{
	vector<T> temp(size1+size2);
	receiveVector<T>(temp, player, size1+size2);

	for (size_t i = 0; i < size1; ++i)
		vec1[i] = temp[i];

	for (size_t i = 0; i < size2; ++i)
		vec2[i] = temp[size1 + i];
}

#endif
