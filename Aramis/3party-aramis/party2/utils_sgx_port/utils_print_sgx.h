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

#include <string>
#include <inttypes.h>

extern int VERBOSE_COUT;

//ocall_print_string
#include "../Enclave/Enclave.h"
#include "sgx_intrin.h"

void print_bool_arr(bool* arr, int len);
	
void print_m128i(__m128i val);

void print_m128i_arr(const __m128i* arr, int len);

void print_m128i_arr(__m128i* arr, int len);

void print_string_cont(std::string str);

// Force print. Will override the VERBOSE_PRINT director.
void print_string_f(std::string str);

void print_string_cout(std::string str);
// Printing ring 1 (priority after force print). Used to print
// Upper level directions.
void print_string_r1(std::string str);

void print_string(std::string str);

void print_string_octal(std::string str);

void print_integer_cont(int val);

void print_integer_cont(uint64_t val);

void print_integer_cont(int64_t val);

void print_integer(int val);

void print_integer(unsigned int val);

void print_integer(uint64_t val);

void print_integer(int64_t val);

void print_integer(std::string pretext, int val);

void print_integer(std::string pretext, uint64_t val);

void print_integer(std::string pretext, int64_t val);

void print_c_string(uint8_t* buf, int len);

void print_c_string(const uint8_t* buf, int len);

void print_c_string(char* buf, int len);

void print_c_string(const char* buf, int len);

void fprintf(void* fileptr, const char* buf);

void fprintf(void* fileptr, const char* buf, int val);

#ifndef SGX_OSTREAM
#define SGX_OSTREAM

class sgx_ostream {
	public:
		int auto_flush = 0;
		//std::string coutpretext;
		std::string output;

		sgx_ostream(std::string initstring);

		sgx_ostream();

		sgx_ostream operator <<(sgx_ostream val);

		sgx_ostream operator <<(int val);

		sgx_ostream operator <<(std::string str);

		sgx_ostream operator <<(char* cstr);

		sgx_ostream operator <<(const char* cstr);

		~sgx_ostream();

};

void sgx_cout_flush(sgx_ostream stream);

#endif
