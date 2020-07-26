/*
 * Slightly modified for Porthos from
 * basicSockets.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: froike(Roi Inbar) 
 * 	Modified: Aner Ben-Efraim
 * 
 */

#include <stdio.h>
//#include <stropts.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#ifdef __APPLE__
#include <net/if.h>
#else
#include <linux/netdevice.h>
#endif

#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "globals.h"
using namespace std;

#ifndef PORTHOSNET_H_
#define PORTHOSNET_H_

#ifdef _WIN32
 #include<winsock2.h>
#else
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <stdbool.h>
#endif
/*GLOBAL VARIABLES - LIST OF IP ADDRESSES*/
extern char** localIPaddrs;
extern int numberOfAddresses;
#define NUMCONNECTIONS 1


//gets the list of IP addresses
char** getIPAddresses();
int getPartyNum(char* filename);


class CommunicationObject
{
private:
	porthosLongUnsignedInt bytesSent = 0;
	porthosLongUnsignedInt bytesReceived = 0;
	porthosLongUnsignedInt numberOfSends = 0;
	porthosLongUnsignedInt numberOfRecvs = 0;	
	bool measurement = false;	

public:
#if (LOG_LAYERWISE)
	double totalTimeInSending;
	double totalTimeInReceiving;
	porthosLongUnsignedInt totalDataSent;
	porthosLongUnsignedInt totalDataReceived;
	porthosLongUnsignedInt minSizeSent = 0;

	porthosLongUnsignedInt dataMatmul[2] = {0,0};
	double timeMatmul[2] = {0.0,0.0};

	porthosLongUnsignedInt dataRelu[2] = {0,0};
	double timeRelu = 0.0;

	porthosLongUnsignedInt dataMaxPool[2] = {0,0};
	double timeMaxpool = 0.0;

	porthosLongUnsignedInt dataBN[2] = {0,0};
	double timeBN = 0.0;

	porthosLongUnsignedInt dataAvgPool[2] = {0,0};
	double timeAvgPool = 0.0;
#endif
	
	void reset()
	{
		bytesSent = 0;
		bytesReceived = 0;
		numberOfSends = 0;
		numberOfRecvs = 0;
		measurement = false;
#if (LOG_LAYERWISE)
		totalTimeInSending = 0.0;
		totalTimeInReceiving = 0.0;
		totalDataSent = 0;
		totalDataReceived = 0;
		minSizeSent = 5000000000;
#endif
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

	porthosLongUnsignedInt getSent() {return bytesSent;}
	porthosLongUnsignedInt getRecv() {return bytesReceived;}
	porthosLongUnsignedInt getRoundsSent() {return numberOfSends;}
	porthosLongUnsignedInt getRoundsRecv() {return numberOfRecvs;}
	bool getMeasurement() {return measurement;}
};


class PorthosNet {
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
	/**
	 * Constructor for servers and clients, got the host and the port for connect or listen.
	 * After creation call listenNow() or connectNow() function.
	 */
	PorthosNet(char * host, int port);

	/**
	 * Constructor for servers only. got the port it will listen to.
	 * After creation call listenNow() function.
	 */
	PorthosNet(int portno);

	/**
	 * got data and send it to the other side, wait for response and return it.
	 * return pointer for the data that recived.
	 */
	void* sendAndRecive(const void* data, int get_size, int send_size);

	
	virtual ~PorthosNet();

	/**
	 * Start listen on the given port.
	 */
	bool listenNow();

	/**
	 * Try to connect to server by given host and port.
	 * return true for success or false for failure.
	 */
	bool connectNow();

	/**
	 * Send Data to the other side.
	 * return true for success or false for failure.
	 */
	bool sendMsg(const void* data, size_t size, int conn);

	/**
	 * Recive data from other side.
	 * return true for success or false for failure.
	 */
	bool receiveMsg(void* buff, size_t buffSize, int conn);
};



#endif /* PORTHOSNET_H_ */
