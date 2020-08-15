/*
Authors: Mayank Rathee.
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

#include "socket_interface.h"
#include <thread>
#include <mutex>

// This cpp file is used to take in the appropriate primary
// key from the enclave and load the respective AramisNet class object
// on the application side and then call relevant funtions on that object.
//
// Therefore, we keep an vector of available AramisNet objects
// here and call the actual function inside basicSocketsPort.cpp
// with this AramisNet object.

vector<AramisNet> net_objects;
std::mutex mut;
//Print sent and received messages
#define PRINT_SENT_RECVD_APP
extern volatile int deadcond = 0;
volatile int initjob = 0;
volatile struct job_socket_app* job_send;

void worker_thread_app_send(){
	while(1){
		cout<<"a";
		if(deadcond)
			break;
		mut.lock();
		if(job_send->is_new == 1){

			// Call send
			job_send->netobj->sendMsg((const char*)(job_send->data), job_send->size, job_send->conn);	
			job_send->is_new = 0;
		}
		mut.unlock();
	}
}

void app_print_c_string(char* buf, int len){
	char out[10];
	snprintf(out, 10, "Len %d \n", len);
	printf("%s", out);
	for(int i=0; i<len; i++){
		snprintf(out, 10, " %c", buf[i]);
		printf("%s", out);
	}
	printf("\n");
}

void app_print_c_string(const char* buf, int len){
	char out[10];
	snprintf(out, 10, "Len %d \n", len);
	printf("%s", out);
	for(int i=0; i<len; i++){
		snprintf(out, 10, " %c", buf[i]);
		printf("%s", out);
	}
	printf("\n");
}

// Interface functions
void OCALL_AramisNet_Constructor(char* host, int hostlen, int portno, int ukey, int host_p_num){
	assert(ukey == net_objects.size());
	AramisNet obj = AramisNet(host, portno, host_p_num);
	net_objects.push_back(obj);
	return;
}

void OCALL_AramisNet_Constructor_server(int portno, int ukey){
	assert(ukey == net_objects.size());
	AramisNet obj = AramisNet(portno);
	net_objects.push_back(obj);
	return;
}

void OCALL_AramisNet_Destructor(int ukey){
	//delete(&(net_objects[ukey]));
	return;
}

bool OCALL_AramisNet_listenNow(int ukey){
	AramisNet *obj = &(net_objects[ukey]);
	assert(true == obj->listenNow());
	return true;	
}

bool OCALL_AramisNet_connectNow(int ukey){
	AramisNet *obj = &(net_objects[ukey]);
	assert(true == obj->connectNow());
	return true;
}

bool OCALL_AramisNet_sendMsg(int size, int conn, int ukey, const void* data){
	AramisNet *obj = &(net_objects[ukey]);
	
	assert(true == obj->sendMsg(data, size, conn));
	return true;
}

bool OCALL_AramisNet_receiveMsg(void* buff, int size, int conn, int ukey){
	AramisNet *obj = &(net_objects[ukey]);
	assert(true == obj->receiveMsg(buff, size, conn));
	return true;
}

// End - Interface functions
