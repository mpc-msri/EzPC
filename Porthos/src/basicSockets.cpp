/*
 * Slightly modifier for Porthos from
 * basicSockets.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: froike(Roi Inbar)
 *	Modified: Aner Ben-Efraim
 *
 */

#include "basicSockets.h"
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
#include <chrono>
#include "globals.h"
using namespace std;
using namespace std::chrono;

#define bufferSize 256

/*GLOBAL VARIABLES - LIST OF IP ADDRESSES*/
char** localIPaddrs;
int numberOfAddresses;

//For communication measurements
CommunicationObject commObject;

// Global for maintaining all io context
boost::asio::io_service io_service;

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

	localIPaddrs = getIPAddresses(AF_INET);
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
	return -1;
}

PorthosNet::PorthosNet(char* host, int portno) {
	this->port = portno;
	this->host = host;
	this->is_JustServer = false;
	for (int i = 0; i < NUMCONNECTIONS; i++)
		this->sockets[i] = new tcp::socket(io_service);
}

PorthosNet::PorthosNet(int portno) {
	this->port = portno;
	this->host = "";
	this->is_JustServer = true;
	for (int i = 0; i < NUMCONNECTIONS; i++)
		this->sockets[i] = new tcp::socket(io_service);
}

PorthosNet::~PorthosNet() {
	for (int conn = 0; conn < NUMCONNECTIONS; conn++) {
		boost::system::error_code ec;
		sockets[conn]->shutdown(tcp::socket::shutdown_both, ec);
		if (ec)
			std::cout << "Shutdown of send/recv on socket failed: " << ec.message() << std::endl;
		sockets[conn]->close(ec);
		if (ec)
			std::cout << "Closing of socket failed: " << ec.message() << std::endl;
		free(sockets[conn]);
	}
}

bool PorthosNet::listenNow(){
	tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port));
	for (int conn = 0; conn < NUMCONNECTIONS; conn++) {
		boost::system::error_code ec;
		acceptor.accept(*sockets[conn], ec);
		if (ec) {
			std::cout << "Error in accepting socket connection on port "
				<< port << ". " << ec.message() << std::endl;
			return false;
		}
		tcp::no_delay tcpNoDelayOption(true);
		sockets[conn]->set_option(tcpNoDelayOption, ec);
		if (ec) {
			std::cout << ec.message() << std::endl;
			return false;
		}
		boost::asio::socket_base::keep_alive keepAliveOption(true);
		sockets[conn]->set_option(keepAliveOption, ec);
		if (ec) {
			std::cout << ec.message() << std::endl;
			return false;
		}
	}
	return true;
}

bool PorthosNet::connectNow(){
	if (is_JustServer){
		cout<<"ERROR: Never got a host... please use the second constructor"<<endl;;
		return false;
	}

	tcp::resolver resolver(io_service);
	tcp::resolver::query query(host, std::to_string(port));
	tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

	for (int conn = 0; conn < NUMCONNECTIONS; conn++) {
		std::cout << "Trying to connect\n";
		boost::system::error_code ec;
		boost::asio::connect(*sockets[conn], endpoint_iterator, ec);
		int count = 1;
		while(ec) {
			ec.clear();
			if (count % 50 == 0) {
				std::cout << "Not managing to connect. " << "Count=" << count << endl;
				std::cout << ec.message() << ". Trying again." << std::endl;
			}
			sleep(1);
			boost::asio::connect(*sockets[conn], endpoint_iterator, ec);
			count++;
		}
		tcp::no_delay tcpNoDelayOption(true);
		sockets[conn]->set_option(tcpNoDelayOption, ec);
		if (ec) {
			std::cout << ec.message() << std::endl;
			return false;
		}
		boost::asio::socket_base::keep_alive keepAliveOption(true);
		sockets[conn]->set_option(keepAliveOption, ec);
		if (ec) {
			std::cout << ec.message() << std::endl;
			return false;
		}
	}
	return true;
}

bool sendChunk(const void* data, size_t size, tcp::socket &socket) {
	boost::system::error_code ec;
	boost::asio::write(socket, boost::asio::buffer(data, size), ec);
	if (ec) {
		std::cout << "sendChunk: ERROR writing to socket" << std::endl;
		std::cout << ec.message() << "\n";
		return false;
	}
	return true;
}

bool PorthosNet::sendMsg(const void* data, size_t size, int conn){
#if (LOG_LAYERWISE)
	auto t1 = high_resolution_clock::now();
#endif
	boost::system::error_code ec;
	boost::asio::write(*sockets[conn], boost::asio::buffer(data, size), ec);
	if (ec) {
		std::cout << "ERROR writing to socket" << std::endl;
		std::cout << ec.message() << "\n";
		return false;
	}
#if (LOG_LAYERWISE)
	auto t2 = high_resolution_clock::now();
	auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
	commObject.totalDataSent += size;
	commObject.totalTimeInSending += tt;
	if (size < commObject.minSizeSent){
		commObject.minSizeSent = size;
	}
#endif
	commObject.incrementSent(size);
	return true;
}

bool PorthosNet::receiveMsg(void* buff, size_t size, int conn){
	memset(buff, 0, size);
#if (LOG_LAYERWISE)
	auto t1 = high_resolution_clock::now();
#endif
	boost::system::error_code ec;
	boost::asio::read(*sockets[conn], boost::asio::buffer(buff, size), ec);
	if (ec) {
		std::cout << "ERROR reading from socket" << std::endl;
		std::cout << ec.message() << "\n";
		return false;
	}
#if (LOG_LAYERWISE)
	auto t2 = high_resolution_clock::now();
	auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
	commObject.totalDataReceived += size;
	commObject.totalTimeInReceiving += tt;
#endif
	commObject.incrementRecv(size);
	return true;
}
