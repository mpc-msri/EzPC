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

#include "connect.h"
#include <thread>
#include <mutex> 
#include <vector>

using namespace std;

#define STRING_BUFFER_SIZE 256
extern void error(string str);

//this player number
extern int partyNum;
extern int instanceID;

//communication
string * addrs;
PorthosNet ** communicationSenders;
PorthosNet ** communicationReceivers;

//Communication measurements object
extern CommunicationObject commObject;

//setting up communication
void initCommunication(string addr, 
		int port, 
		int player, 
		int mode)
{
	char temp[25];
	strcpy(temp, addr.c_str());
	if (mode == 0)
	{
		communicationSenders[player] = new PorthosNet(temp, port);
		communicationSenders[player]->connectNow();
	}
	else
	{
		communicationReceivers[player] = new PorthosNet(port);
		communicationReceivers[player]->listenNow();
	}
}

void initializeCommunication(int* ports)
{
	int i;
	communicationSenders = new PorthosNet*[NUM_OF_PARTIES];
	communicationReceivers = new PorthosNet*[NUM_OF_PARTIES];
	thread *threads = new thread[NUM_OF_PARTIES * 2];
	for (i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			threads[i * 2 + 1] = thread(initCommunication, addrs[i], ports[i * 2 + 1], i, 0);
			threads[i * 2] = thread(initCommunication, "127.0.0.1", ports[i * 2], i, 1);
		}
	}
	for (int i = 0; i < 2 * NUM_OF_PARTIES; i++)
	{
		if (i != 2 * partyNum && i != (2 * partyNum + 1))
			threads[i].join();//wait for all threads to finish
	}

	delete[] threads;
}

void initializeCommunicationSerial(int* ports)//Use this for many parties
{
	communicationSenders = new PorthosNet*[NUM_OF_PARTIES];
	communicationReceivers = new PorthosNet*[NUM_OF_PARTIES];
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i<partyNum)
		{
		  initCommunication( addrs[i], ports[i * 2 + 1], i, 0);
		  initCommunication("127.0.0.1", ports[i * 2], i, 1);
		}
		else if (i>partyNum)
		{
		  initCommunication("127.0.0.1", ports[i * 2], i, 1);
		  initCommunication( addrs[i], ports[i * 2 + 1], i, 0);
		}
	}
}

void initializeCommunication(char* filename, 
		int p)
{
	FILE * f = fopen(filename, "r");
	partyNum = p;
	char buff[STRING_BUFFER_SIZE];
	char ip[STRING_BUFFER_SIZE];
	
	addrs = new string[NUM_OF_PARTIES];
	int * ports = new int[NUM_OF_PARTIES * 2];


	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		fgets(buff, STRING_BUFFER_SIZE, f);
		sscanf(buff, "%s\n", ip);
		addrs[i] = string(ip);
		//cout << addrs[i] << endl;
		ports[2 * i] = 32000 + i*NUM_OF_PARTIES + partyNum + 100*instanceID;
		ports[2 * i + 1] = 32000 + partyNum*NUM_OF_PARTIES + i + 100*instanceID;
	}

	fclose(f);
	initializeCommunicationSerial(ports);

	delete[] ports;
}

//synchronization functions
void sendByte(int player, 
		char* toSend, 
		int length, 
		int conn)
{
	communicationSenders[player]->sendMsg(toSend, length, conn);
}

void receiveByte(int player, 
		int length, 
		int conn)
{
	char *sync = new char[length+1];
	communicationReceivers[player]->receiveMsg(sync, length, conn);
	delete[] sync;
}

void synchronize(int length = 1)
{
	char* toSend = new char[length+1];
	memset(toSend, '0', length+1);
	vector<thread *> threads;
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i == partyNum) continue;
		for (int conn = 0; conn < NUMCONNECTIONS; conn++)
		{
			threads.push_back(new thread(sendByte, i, toSend, length, conn));
			threads.push_back(new thread(receiveByte, i, length, conn));
		}
	}
	for (vector<thread *>::iterator it = threads.begin(); it != threads.end(); it++)
	{
		(*it)->join();
		delete *it;
	}
	threads.clear();
	delete[] toSend;
}


void start_communication()
{
	if (commObject.getMeasurement())
		error("Nested communication measurements");

	commObject.reset();
	commObject.setMeasurement(true);
}

void pause_communication()
{
	if (!commObject.getMeasurement())
		error("Communication never started to pause");

	commObject.setMeasurement(false);
}

void resume_communication()
{
	if (commObject.getMeasurement())
		error("Communication is not paused");

	commObject.setMeasurement(true);
}

void end_communication()
{
	cout << "------------------------------------" << endl;
	cout << "Communication for execution, P" << partyNum << ": " 
		 << (float)commObject.getSent()/1000000 << "MB (sent) " 
		 << (float)commObject.getRecv()/1000000 << "MB (recv)" << endl;
	cout << "#Calls, P" << partyNum << ": " 
		 << commObject.getRoundsSent() << "(sends) " 
		 << commObject.getRoundsRecv() << "(recvs)" << endl; 
	cout << "------------------------------------" << endl;	
	commObject.reset();
}

porthosLongUnsignedInt getCurrentTime()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch();
	auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	return millis;
}


