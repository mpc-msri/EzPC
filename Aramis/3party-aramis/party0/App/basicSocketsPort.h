/*
 * basicSockets.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: froike(Roi Inbar)
 *	Modified: Aner Ben-Efraim
 *	Modified by: Mayank Rathee
 */

#include <stdio.h>
#include <stropts.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/netdevice.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string>

using namespace std;

#define PRINT_APP

#ifndef ARAMISNET_H_
#define ARAMISNET_H_

#ifdef _WIN32
 #include<winsock2.h>
#else
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <stdbool.h>
#endif
extern char** localIPaddrs;
extern int numberOfAddresses;
#define NUMCONNECTIONS 1

#define STRING_BUFFER_SIZE 256


char** getIPAddresses();
int getPartyNum(char* filename);


class CommunicationObject
{
private:
	uint64_t bytesSent = 0;
	uint64_t bytesReceived = 0;
	uint64_t numberOfSends = 0;
	uint64_t numberOfRecvs = 0;	
	bool measurement = false;	

public: 
	void reset()
	{
		bytesSent = 0;
		bytesReceived = 0;
		numberOfSends = 0;
		numberOfRecvs = 0;
		measurement = false;
	}

	void setMeasurement(bool a)
	{
		measurement = a;
	}

	void incrementSent(int size)
	{
		if (measurement)
		{
			bytesSent += size;
			numberOfSends++;
		}
	}

	void incrementRecv(int size)
	{
		if (measurement)
		{
			bytesReceived += size;
			numberOfRecvs++;
		}
	}

	uint64_t getSent() {return bytesSent;}
	uint64_t getRecv() {return bytesReceived;}
	uint64_t getRoundsSent() {return numberOfSends;}
	uint64_t getRoundsRecv() {return numberOfRecvs;}
	bool getMeasurement() {return measurement;}
};


class AramisNet {
private:
	char * host;
	unsigned int port;
	bool is_JustServer;
	int socketFd[NUMCONNECTIONS];
	#ifdef _WIN32
	    PCSTR Cport;
		WSADATA wsa;
		DWORD dwRetval;
	#endif


public:
	AramisNet(char * host, int port, int player);

	AramisNet(int portno);

	void* sendAndRecive(const void* data, int get_size, int send_size);

	virtual ~AramisNet();

	bool listenNow();

	bool connectNow();

	bool sendMsg(const void* data, int size, int conn);

	bool receiveMsg(void* buff, int buffSize, int conn);


};



#endif 
