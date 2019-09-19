/*
 * basicSockets.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: froike(Roi Inbar)
 *	Modified: Aner Ben-Efraim
 *	Modified by: Mayank Rathee
 */

#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include "basicSocketsPort.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <iostream>
#include <netinet/tcp.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

//sleep
#include <iostream>
#include <thread>
#include <chrono>

#include "cross_call_counter.h"

using namespace std;

#define bufferSize 256

#ifdef __linux__
	#include <unistd.h>
	#include <sys/socket.h>
	#include <netinet/in.h>
	#include <netdb.h>
	#define Sockwrite(sock, data, size) write(sock, data, size) 
	#define Sockread(sock, buff, bufferSize) read(sock, buff, bufferSize)
#elif _WIN32
	#pragma comment (lib, "Ws2_32.lib")
	#include<winsock.h>
	#define socklen_t int
	#define close closesocket
	#define Sockwrite(sock, data, size) send(sock, (char*)data, size, 0)
	#define Sockread(sock, buff, bufferSize) recv(sock, (char*)buff, bufferSize, 0)
	
#endif

char** localIPaddrs;
int numberOfAddresses;

CommunicationObject commObject;

char* addresses[4] = {"127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"};


std::string exec(const char* cmd) 
{
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get())) {
        if (fgets(buffer.data(), 128, pipe.get()) != NULL)
            result += buffer.data();
    }
    return result;
}

string getPublicIp()
{
	string s = exec("dig TXT +short o-o.myaddr.l.google.com @ns1.google.com");
    s = s.substr(1, s.length()-3); 
    return s;
}

char** getIPAddresses(const int domain)
{
  char** ans;
  int s;
  struct ifconf ifconf;
  struct ifreq ifr[50];
  int ifs;
  int i;

  s = socket(domain, SOCK_STREAM, 0);
  if (s < 0) {
    perror("socket");
    return 0;
  }

  ifconf.ifc_buf = (char *) ifr;
  ifconf.ifc_len = sizeof ifr;

  if (ioctl(s, SIOCGIFCONF, &ifconf) == -1) {
    perror("ioctl");
    return 0;
  }

  ifs = ifconf.ifc_len / sizeof(ifr[0]);
  numberOfAddresses = ifs+1;
  ans = new char*[ifs+1];

  string ip = getPublicIp(); 
  ans[0] = new char[ip.length()+1];
  strcpy(ans[0], ip.c_str());
  ans[0][ip.length()] = '\0';

  for (i = 1; i <= ifs; i++) {
    char* ip=new char[INET_ADDRSTRLEN];
    struct sockaddr_in *s_in = (struct sockaddr_in *) &ifr[i].ifr_addr;

    if (!inet_ntop(domain, &s_in->sin_addr, ip, INET_ADDRSTRLEN)) {
      perror("inet_ntop");
      return 0;
    }

    ans[i]=ip;
  }

  close(s);

  return ans;
}

int getPartyNum(char* filename)
{

	FILE * f = fopen(filename, "r");

	char buff[STRING_BUFFER_SIZE];
	char ip[STRING_BUFFER_SIZE];

	localIPaddrs=getIPAddresses(AF_INET);
	string tmp;
	int player = 0;
	while (true)
	{
		fgets(buff, STRING_BUFFER_SIZE, f);
		sscanf(buff, "%s\n", ip);
		for (int i = 0; i < numberOfAddresses; i++)
			if (strcmp(localIPaddrs[i], ip) == 0 || strcmp("127.0.0.1", ip)==0)
				return player;
		player++;
	}
	fclose(f);

}

AramisNet::AramisNet(char* host, int portno, int player) {
	this->port = portno;
#ifdef _WIN32
	this->Cport = (PCSTR)portno;
#endif
	this->host = host;
	this->host = addresses[player]; 
	this->is_JustServer = false;
	for (int i = 0; i < NUMCONNECTIONS; i++) this->socketFd[i] = -1;
#ifdef _WIN32
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
	{
		printf("Failed. Error Code : %d\n", WSAGetLastError());
	}
	else printf("WSP Initialised.\n");
#endif

}

AramisNet::AramisNet(int portno) {
	this->port = portno;
	this->host = "";
	this->is_JustServer = true;
	for (int i = 0; i < NUMCONNECTIONS; i++) this->socketFd[i] = -1;
#ifdef _WIN32
	if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
	{
		printf("Failed. Error Code : %d\n", WSAGetLastError());
	}
	else printf("WSP Initialised.\n");
#endif
}

AramisNet::~AramisNet() {
	return;
}

bool AramisNet::listenNow(){
	int serverSockFd;
	socklen_t clilen;

	struct sockaddr_in serv_addr, cli_addr[NUMCONNECTIONS];


	serverSockFd = socket(AF_INET, SOCK_STREAM, 0);
	if (serverSockFd < 0){
		cout<<"[ARAMIS NET]: ERROR opening socket"<<endl;
		return false;
	}
	memset(&serv_addr, 0,sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(this->port);
	
	int yes=1;
	
	if (setsockopt(serverSockFd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == -1) 
	{
		perror("setsockopt");
		exit(1);
	}
	
	int testCounter=0;//
	while (bind(serverSockFd, (struct sockaddr *) &serv_addr,
			sizeof(serv_addr)) < 0 && testCounter<10){
		cout<<"[ARAMIS NET]: ERROR on binding: "<< port <<endl;
		cout<<"[ARAMIS NET]: Count: "<< testCounter<<endl;///
		perror("Binding");
		testCounter++;///
		sleep(2);
	}
	if (testCounter==10) return false;//
	listen(serverSockFd, 10);
	
	for (int conn = 0; conn < NUMCONNECTIONS; conn++)
	{
		clilen = sizeof(cli_addr[conn]);
		cout<<"[ARAMIS NET]: Server listening on port "<<this->port<<endl;
		this->socketFd[conn] = accept(serverSockFd,
				(struct sockaddr *) &cli_addr[conn],
				 &clilen);
		//cout<<socketFd[conn]<<" for conn "<<conn<<endl;
		if (this->socketFd[conn] < 0){
			cout<<"\n\n\n[ARAMIS NET]: ERROR on accept\n\n\n"<<endl;
			return false;
		}
		int flag = 1;
		int result = setsockopt(this->socketFd[conn],            /* socket affected */
	                          IPPROTO_TCP,     /* set option at TCP level */
	                          TCP_NODELAY,     /* name of option */
	                          (char *) &flag,  /* the cast is historical */
	                          sizeof(int));    /* length of option value */
		if (result < 0) {
		    cout << "[ARAMIS NET]: error setting NODELAY. exiting" << endl;
		    exit (-1);
		}

	}
	close(serverSockFd);
	return true;
}


bool AramisNet::connectNow(){
	//sleep for a second, so that server is ready
	std::this_thread::sleep_for(std::chrono::seconds(2));
	struct sockaddr_in serv_addr[NUMCONNECTIONS];
	struct hostent *server;
	int n;

	if (is_JustServer){
		cout<<"[ARAMIS NET]: ERROR: Never got a host... please use the second constructor"<<endl;;
		return false;
	}

	for (int conn = 0; conn < NUMCONNECTIONS; conn++)
	{
		//fprintf(stderr,"usage %s hostname port\n", host);
		socketFd[conn] = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (socketFd[conn] < 0){
			cout << ("[ARAMIS NET]: ERROR opening socket") << endl;
			return false;
		}

		//use TCP_NODELAY on the socket 
		int flag = 1;
		int result = setsockopt(socketFd[conn],            /* socket affected */
	                          IPPROTO_TCP,     /* set option at TCP level */
	                          TCP_NODELAY,     /* name of option */
	                          (char *) &flag,  /* the cast is historical */
	                          sizeof(int));    /* length of option value */
		if (result < 0) {
		    cout << "[ARAMIS NET]: error setting NODELAY. exiting" << endl;
		    exit (-1);
	       }
		

		memset(&serv_addr[conn], 0, sizeof(serv_addr[conn]));
		serv_addr[conn].sin_family		= AF_INET;
		serv_addr[conn].sin_addr.s_addr	= inet_addr(host);
		serv_addr[conn].sin_port			= htons(port); 
		
		int count = 0;
		cout << "[ARAMIS NET]: Trying to connect to server " << endl; 
		cout << "[ARAMIS NET]: IP Address: " << host << " || Port " << port << endl;
		while (connect(socketFd[conn], (struct sockaddr *) &serv_addr[conn], sizeof(serv_addr[conn])) < 0)
		{
				count++;
				if (count % 50 == 0)
				    cout << "[ARAMIS NET]: Not managing to connect. " << "Count=" << count << endl;
				sleep(1);
		}

	}

	return true;
}



bool AramisNet::sendMsg(const void* data, int size, int conn){
	socket_calls++;	
	int left = size;
	int n;
	while (left > 0)
	{
		n = Sockwrite(this->socketFd[conn], &((char*)data)[size - left], left);
		if (n < 0) {
			cout << "[ARAMIS NET]: ERROR writing to socket" << endl;
			return false;
		}
		left -= n;
	}

	return true;
}

bool AramisNet::receiveMsg(void* buff, int size, int conn){
	socket_calls++;
	int left = size;
	int n;
	while (left > 0)
	{
		n = Sockread(this->socketFd[conn], &((char*)buff)[size-left], left);
		if (n < 0) {
			cout << "[ARAMIS NET]: ERROR reading from socket" << endl;
			cout << "[ARAMIS NET]: Size = " << size << ", Left = " << left << ", n = " << n << endl;
			return false;
		}
		if (n == 0){
			break;
		}
		left -= n;
	}
	return true;
}
