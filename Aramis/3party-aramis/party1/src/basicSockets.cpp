/*
 * Slightly modified for Aramis from
 * basicSockets.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: froike(Roi Inbar) 
 * 	Modified: Aner Ben-Efraim
 * 	Modified by: Mayank Rathee
 */

#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include "basicSockets.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include "../utils_sgx_port/utils_print_sgx.h"

//To make OCALLS
#include "Enclave_t.h"
#include "sgx_trts.h"
#include "Enclave.h"
#include "globals.h"

using namespace std;

#define bufferSize 256
//#define PRINT_SENT_RECVD

#define MAC_CHUNKS_SIZE 50000
#define CHUNKS_SIZE 80000 // faster was 5000
#define WITH_MAC

#ifdef __linux__
#include <unistd.h>
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

AramisNet::AramisNet(char* host, int portno, int player) {
	this->ukey = primary_key_bmr++;
	int hostlen = strlen(host);
	OCALL_AramisNet_Constructor(host, hostlen, portno, ukey, player);
}

AramisNet::AramisNet(int portno) {
	this->ukey = primary_key_bmr++;
	OCALL_AramisNet_Constructor_server(portno, ukey);
}

AramisNet::~AramisNet() {
	OCALL_AramisNet_Destructor(this->ukey);
}

bool AramisNet::listenNow(){
	bool res = false;
	return OCALL_AramisNet_listenNow(&res, this->ukey);
}


bool AramisNet::connectNow(){
	bool res = false;
	return OCALL_AramisNet_connectNow(&res, this->ukey);
}


bool AramisNet::sendMsg(const void* data, int size, int conn, int target){
	bool res = false;
#ifdef VERBOSE_PRINT
	//print_string("Sending some data to other party");
#endif

	bool ret = false;
#ifdef WITH_MAC
	if(size <= MAC_CHUNKS_SIZE){
		int lenwithmac = size+16+12;
		uint8_t* msgwithmac = (uint8_t*)malloc(lenwithmac);
		uint8_t* msgoriginal = (uint8_t*)malloc(size);
		memcpy(msgoriginal, data, size);
		aes_encrypt_message_mac(msgoriginal, size, msgwithmac, lenwithmac, target);
#ifdef HALT_COMM
		ret = true;
#else
		ret = OCALL_AramisNet_sendMsg(&res, lenwithmac, conn, this->ukey, (const void*)msgwithmac);
#endif
		free(msgwithmac);
		free(msgoriginal);
	}
	else{
		// Here message has to be splitted.
		int ctr_orig = 0;
		while(ctr_orig < size){
			int size_this_iter = 0;
			int left_to_end = size-ctr_orig;
			size_this_iter = (MAC_CHUNKS_SIZE<left_to_end)?MAC_CHUNKS_SIZE:left_to_end;

			int lenwithmac = size_this_iter+16+12;
			uint8_t* msgwithmac = (uint8_t*)malloc(lenwithmac);
			uint8_t* msgoriginal = (uint8_t*)malloc(size_this_iter);
			memcpy(msgoriginal, data+ctr_orig, size_this_iter);
			aes_encrypt_message_mac(msgoriginal, size_this_iter, msgwithmac, lenwithmac, target);
#ifdef HALT_COMM
			ret = true;
#else			
			ret = OCALL_AramisNet_sendMsg(&res, lenwithmac, conn, this->ukey, (const void*)msgwithmac);
#endif
			free(msgwithmac);
			free(msgoriginal);

			ctr_orig += MAC_CHUNKS_SIZE;
		}
	}
#else
	if(size <= CHUNKS_SIZE){
#ifdef HALT_COMM
		ret = true;
#else	
		ret = OCALL_AramisNet_sendMsg(&res, size, conn, this->ukey, data);
#endif	
	}
	else{
		// Here message has to be splitted.
		int ctr_orig = 0;
		while(ctr_orig < size){
			int size_this_iter = 0;
			int left_to_end = size-ctr_orig;
			size_this_iter = (CHUNKS_SIZE<left_to_end)?CHUNKS_SIZE:left_to_end;

			uint8_t* msgoriginal = (uint8_t*)malloc(size_this_iter);
			memcpy(msgoriginal, data+ctr_orig, size_this_iter);
#ifdef HALT_COMM
			ret = true;
#else			
			ret = OCALL_AramisNet_sendMsg(&res, size_this_iter, conn, this->ukey, (const void*)msgoriginal);
#endif
			free(msgoriginal);

			ctr_orig += CHUNKS_SIZE;
		}
	}
#endif
#ifdef VERBOSE_PRINT
	//print_string("Sent the data to other party");
#endif
#ifdef PRINT_SENT_RECVD
	print_string("====>This is the data that is being sent: ");
	memset(datacpy, '0', 10);
	print_c_string((const char*)data, 10);
	print_integer(target);
#endif
	return ret;
}

bool AramisNet::receiveMsg(void* buff, int size, int conn, int target){
	bool res = false;
	bool ret = false;
#ifdef WITH_MAC
	if(size <= MAC_CHUNKS_SIZE){
		int lenwithmac = size+16+12;
		uint8_t* msgwithmac = (uint8_t*)malloc(lenwithmac);
#ifdef HALT_COMM
		ret = true;
#else
		ret =  OCALL_AramisNet_receiveMsg(&res, (void*)msgwithmac, lenwithmac, conn, this->ukey);
		assert(SGX_SUCCESS == aes_decrypt_message_mac(msgwithmac, lenwithmac, (uint8_t*)buff, size, target));
#endif
		free(msgwithmac);
	}
	else{
		// Receive data in chunks
		int recvd_upto = 0;
		while(recvd_upto < size){
			int size_this_iter = 0;
			int left_to_end = size-recvd_upto;
			size_this_iter = (MAC_CHUNKS_SIZE<left_to_end)?MAC_CHUNKS_SIZE:left_to_end;

			int lenwithmac = size_this_iter+16+12;
			uint8_t* msgwithmac = (uint8_t*)malloc(lenwithmac);
#ifdef HALT_COMM
			ret = true;
#else
			ret =  OCALL_AramisNet_receiveMsg(&res, (void*)msgwithmac, lenwithmac, conn, this->ukey);
			assert(SGX_SUCCESS == aes_decrypt_message_mac(msgwithmac, lenwithmac, (uint8_t*)(buff+recvd_upto), size_this_iter, target));
#endif
			free(msgwithmac);
			recvd_upto += MAC_CHUNKS_SIZE;
		}
	}
#else
	if(size <= CHUNKS_SIZE){
#ifdef HALT_COMM
		ret = true;
#else	
		ret =  OCALL_AramisNet_receiveMsg(&res, buff, size, conn, this->ukey);
#endif	
	}
	else{
		// Receive data in chunks
		int recvd_upto = 0;
		while(recvd_upto < size){
			int size_this_iter = 0;
			int left_to_end = size-recvd_upto;
			size_this_iter = (CHUNKS_SIZE<left_to_end)?CHUNKS_SIZE:left_to_end;

			uint8_t* msgrecvd = (uint8_t*)malloc(size_this_iter);
#ifdef HALT_COMM
			ret = true;
#else
			ret =  OCALL_AramisNet_receiveMsg(&res, (void*)msgrecvd, size_this_iter, conn, this->ukey);
#endif
			memcpy(buff+recvd_upto, msgrecvd, size_this_iter);
			free(msgrecvd);

			recvd_upto += CHUNKS_SIZE;
		}
	}
#endif

#ifdef PRINT_SENT_RECVD
	print_string("---->This is the data that was receieved: ");
	print_c_string((char*)buff, 10);
	print_integer(target);
#endif
	return ret;
}
