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

#define MAX 10
#include <inttypes.h>
void ezpc_main(void* arg);

typedef struct output_queue_elmt {
  uint32_t output;
  int role;  //who should we output the clear value to
} output_queue_elmt;


typedef struct protocolIO
{
	int role;
	int size;
	uint64_t gatecount;
	output_queue_elmt outq[MAX];
} protocolIO;

void flush_output_queue(protocolIO *io);

uint32_t* add_to_output_queue(protocolIO *io, int role);

obliv uint32_t uarshift(obliv uint32_t a, uint32_t b) obliv;
