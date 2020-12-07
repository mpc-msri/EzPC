/*

Authors: Aseem Rastogi, Lohith Ravuru.

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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <obliv.h>
#include <obliv.oh>
#include "ezpc.h"
#include "../common/util.h"


int currentParty;

const char* mySide()
{
	if(currentParty==1) return "Generator";
	else return "Evaluator";
}

void flush_output_queue(protocolIO *io)
{   
	for (int i = 0; i < io->size ; ++i)
	{
		int role = io->outq[i].role;
		bool rolematch = false;
		rolematch = (role==0) ? true : ( (role == io->role) ? true : false ) ;
		if (rolematch)
		{
			uint32_t output = (io->outq[i]).output;
			fprintf(stderr,"%" PRIu32,output);
			fprintf(stderr,"\n");
		}
	}
	return;
}

uint32_t* add_to_output_queue(protocolIO *io, int role)
{
	(io->outq[io->size]).role = role;
	io->size = io->size + 1;
	return &((io->outq[(io->size) - 1]).output);
}

double lap;

int main(int argc, char *argv[])
{
	ProtocolDesc pd;
	int opt;
	opt=getopt(argc,argv,"r:");
	switch(opt){
	   case 'r':
		currentParty = atoi(optarg);
		currentParty += 1; 
		break;
	   default:
		printf("Usage: ./a.out -r <0|1> \n");
		exit(0);
	}
	if(currentParty!=1 && currentParty!=2)
	{	
		printf("Usage: ./a.out -r <0|1> \n");
		exit(0);
	}
	const char* remote_host = currentParty==1? "localhost" : NULL;
	ocTestUtilTcpOrDie(&pd,remote_host,"1234");
	// scanf("%d",&(io.myinput));

	setCurrentParty(&pd,currentParty);
	lap=wallClock();
	protocolIO io;
	io.role = currentParty;
	io.size = 0;
	execYaoProtocol(&pd,ezpc_main,&io);
	flush_output_queue(&io);
	fprintf(stderr,"%s total time: %lf s\n",mySide(),wallClock()-lap);
	fprintf(stderr,"Yao gate count: %u\n",io.gatecount);
	cleanupProtocol(&pd);
	return 0;
}


obliv uint32_t uarshift(obliv uint32_t a, uint32_t b) obliv {
	obliv uint32_t result;
	obliv if(a > (0 - a))
	{
		result = (0 - ((0 - a) >> b));
	}
	else
	{
		result = (a >> b);
	}
	return result;
}
