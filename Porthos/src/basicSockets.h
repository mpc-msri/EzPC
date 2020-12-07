/*
 * Slightly modified for Porthos from
 * basicSockets.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: froike(Roi Inbar)
 * 	Modified: Aner Ben-Efraim
 *
 */
#ifndef PORTHOSNET_H_
#define PORTHOSNET_H_

#include <stdio.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <vector>
#include "globals.h"
using namespace std;

#include <boost/array.hpp>
#include <boost/asio.hpp>
using boost::asio::ip::tcp;



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
	size_t bytesSent = 0;
	size_t bytesReceived = 0;
	size_t numberOfSends = 0;
	size_t numberOfRecvs = 0;
	bool measurement = false;

public:
#if (LOG_LAYERWISE)
	double totalTimeInSending;
	double totalTimeInReceiving;
	size_t totalDataSent;
	size_t totalDataReceived;
	size_t minSizeSent = 0;

	size_t dataMatmul[2] = {0,0};
	double timeMatmul[2] = {0.0,0.0};

	size_t dataRelu[2] = {0,0};
	double timeRelu = 0.0;

	size_t dataMaxPool[2] = {0,0};
	double timeMaxpool = 0.0;

	size_t dataBN[2] = {0,0};
	double timeBN = 0.0;

	size_t dataAvgPool[2] = {0,0};
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

	void incrementSent(size_t size)
	{
		if (measurement)
		{
			bytesSent += size;
			numberOfSends++;
		}
	}

	void incrementRecv(size_t size)
	{
		if (measurement)
		{
			bytesReceived += size;
			numberOfRecvs++;
		}
	}

	size_t getSent() {return bytesSent;}
	size_t getRecv() {return bytesReceived;}
	size_t getRoundsSent() {return numberOfSends;}
	size_t getRoundsRecv() {return numberOfRecvs;}
	bool getMeasurement() {return measurement;}
};


class PorthosNet {
private:
	char * host;
	unsigned int port;
	bool is_JustServer;
	tcp::socket* sockets[NUMCONNECTIONS];

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
