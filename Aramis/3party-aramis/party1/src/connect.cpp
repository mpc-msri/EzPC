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

#include "connect.h"
#include <mutex> 
#include <vector>
#include "../utils_sgx_port/utils_print_sgx.h"
using namespace std;

#define STRING_BUFFER_SIZE 256
extern void error(string str);

//this player number
extern int partyNum;

//communication
string * addrs;
AramisNet ** communicationSenders;
AramisNet ** communicationReceivers;

//Communication measurements object
extern CommunicationObject commObject;

//setting up communication
void initCommunication(string addr, int port, int player, int mode)
{
	char temp[25];
	int ln = addr.length();
	strncpy(temp, addr.c_str(), ln);
	temp[ln] = '\0';
	if (mode == 0)
	{
		communicationSenders[player] = new AramisNet(temp, port, player);
		communicationSenders[player]->connectNow();
	}
	else
	{
		communicationReceivers[player] = new AramisNet(port);
		communicationReceivers[player]->listenNow();
	}
}

void initializeCommunicationSerial(int* ports)//Use this for many parties
{
	communicationSenders = new AramisNet*[NUM_OF_PARTIES];
	communicationReceivers = new AramisNet*[NUM_OF_PARTIES];
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if(i==partyNum){
			continue;
		}
		if(partyNum==0){
			if(i==1){
				initCommunication("127.0.0.1", ports[0], 1, 1);//server0 for client1
				initCommunication(addrs[1], ports[2], 1, 0); //client0 for server 1
			}
			else if(i==2){
				initCommunication("127.0.0.1", ports[1], 2, 1);// server 0 for client2
				initCommunication(addrs[2], ports[4], 2, 0); // client0 for server2
	
			}

		}
		else if(partyNum==1){
			if(i==0){
				initCommunication(addrs[0], ports[0], 0, 0);//client1 for server0
				initCommunication("127.0.0.1", ports[2], 0, 1);//server1 for client0
			}
			else if(i==2){
				initCommunication(addrs[2], ports[5], 2, 0);//client1 for server2
				initCommunication("127.0.0.1", ports[3], 2, 1);//server1 for client2
			}
		}
		else if(partyNum==2){
			if(i==0){
				initCommunication("127.0.0.1", ports[5], 1, 1);//server2 for client1
				initCommunication(addrs[0], ports[1], 0, 0);//client2 for server0
			}
			else if(i==1){
				initCommunication("127.0.0.1", ports[4], 0, 1);//server2 for client0
				initCommunication(addrs[1], ports[3], 1, 0);//client2 for server1
			}
		}
	}
}

void initializeCommunication(char* filename, int p)
{
	partyNum = p;
	char buff[STRING_BUFFER_SIZE];
	char ip[STRING_BUFFER_SIZE];
	
	addrs = new string[NUM_OF_PARTIES];
	int * ports = new int[NUM_OF_PARTIES * 2];


	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		//This address is just a place holder
		//artifact of porting. Actual address
		//of other parties is entered in
		//App/basicSocketsPort.cpp
		addrs[i] = "127.0.0.1";
		ports[2*i] = 32000+2*i;
		ports[2*i + 1] = 32000+(2*i)+1; // Server listens to both clients on same port
	}

	initializeCommunicationSerial(ports);
}


//synchronization functions
void sendByte(int player, char* toSend, int length, int conn)
{
	communicationSenders[player]->sendMsg(toSend, length, conn, player);
	// totalBytesSent += 1;
}

void receiveByte(int player, int length, int conn)
{
	char *sync = new char[length+1];
	communicationReceivers[player]->receiveMsg(sync, length, conn, player);
	delete[] sync;
	// totalBytesReceived += 1;
}

void synchronize(int length /*=1*/)
{
#ifdef VERBOSE_PRINT
	print_string("Beginning synchronization");
#endif	
	char* toSend = new char[length+1];
	memset(toSend, '0', length+1);
#ifdef VERBOSE_PRINT
	print_string("------This is party num-------");
	print_integer(partyNum);
#endif
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i == partyNum) continue;
		for (int conn = 0; conn < NUMCONNECTIONS; conn++)
		{	
			if(partyNum == 0)
			{
				if(i==1){
					sendByte(i, toSend, length, conn);
					receiveByte(i, length, conn);
				}
				else if(i==2){
					sendByte(i, toSend, length, conn);
					receiveByte(i, length, conn);
				}
			}
			else if(partyNum == 1){
				if(i==0){
					receiveByte(i, length, conn);
					sendByte(i, toSend, length, conn);
				}
				else if(i==2){
					sendByte(i, toSend, length, conn);
					receiveByte(i, length, conn);
				}
			}
			else if(partyNum == 2){
				if(i==0){
					receiveByte(i, length, conn);
					sendByte(i, toSend, length, conn);
				}
				else if(i==1){
					receiveByte(i, length, conn);
					sendByte(i, toSend, length, conn);
				}
			}
		}
	}
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

void end_communication(string str)
{
	commObject.reset();
}
