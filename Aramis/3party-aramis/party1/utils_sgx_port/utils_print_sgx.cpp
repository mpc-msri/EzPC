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

#include "utils_print_sgx.h"
#include <string>

int VERBOSE_COUT = 0;
#define VERBOSE_PRINT 1

void print_bool_arr(bool* arr, int len){
	//print_string_cont("__bool array value: ");
	//for(int i=0; i<len; i++){
		//print_string_cont(arr[i]?"1":"0");
	//}
	//print_string_cont("\n");
}

void print_m128i(__m128i val){
	//int64_t* val64 = (int64_t*)&val;
	//print_string_cont("__m128i value: ");
	//print_integer_cont(val64[1]);
	//print_integer_cont(val64[0]);
	//print_string_cont("\n");
}

void print_m128i_arr(const __m128i* arr, int len){
	//print_string_cont("__m128i array values: ");
	for(int i=0; i<len; i++){
		int64_t* val64 = (int64_t*)&arr[i];
		//print_integer_cont(val64[1]);
		//print_integer_cont(val64[0]);
		//if(i==len-1)
			//print_string_cont(".");
		//else{
			//print_string_cont(", ");
		//}
	}
	//print_string_cont("\n");
}

void print_m128i_arr(__m128i* arr, int len){
	//print_string_cont("__m128i array values: ");
	for(int i=0; i<len; i++){
		int64_t* val64 = (int64_t*)&arr[i];
		//print_integer_cont(val64[1]);
		//print_integer_cont(val64[0]);
		//if(i==len-1)
			//print_string_cont(".");
		//else{
			//print_string_cont(", ");
		//}
	}
	//print_string_cont("\n");
}

void print_string_cout(std::string str){ char out[10];
	const char* buf = str.c_str();
	for(int i=0; i<str.size(); i++){
		snprintf(out, 10, "%c", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}

void print_string_f(std::string str){ char out[10];
	//const char* buf = str.c_str();
	//for(int i=0; i<str.size(); i++){
		//snprintf(out, 10, "%c", buf[i]);
		//ocall_print_string(out);
	//}
	//ocall_print_string("\n");
}

void print_string_r1(std::string str){ char out[10];
	//const char* buf = str.c_str();
	//for(int i=0; i<str.size(); i++){
		//snprintf(out, 10, "%c", buf[i]);
		//if(VERBOSE_PRINT_RING_1 == 1)
			//ocall_print_string(out);
	//}
	//if(VERBOSE_PRINT_RING_1 == 1)
		//ocall_print_string("\n");
}


void print_string(std::string str){ char out[10];
	const char* buf = str.c_str();
	for(int i=0; i<str.size(); i++){
		snprintf(out, 10, "%c", buf[i]);
		if(VERBOSE_PRINT == 1 || VERBOSE_COUT == 1)
			ocall_print_string(out);
	}
	if(VERBOSE_PRINT == 1 || VERBOSE_COUT == 1)
		ocall_print_string("\n");
}

void print_string_cont(std::string str){ char out[10];
	//const char* buf = str.c_str();
	//for(int i=0; i<str.size(); i++){
		//snprintf(out, 10, "%c", buf[i]);
		//if(VERBOSE_PRINT == 1 || VERBOSE_COUT == 1)
			//ocall_print_string(out);
	//}
}

void print_string_octal(std::string str){
	char out[10];
	const char* buf = str.c_str();
	for(int i=0; i<str.size(); i++){
		snprintf(out, 10, " %2X", buf[i]);
		if(VERBOSE_PRINT == 1)
			ocall_print_string(out);
	}
	if(VERBOSE_PRINT == 1)
		ocall_print_string("\n");
}

void print_integer_cont(int val){
	//char out[25];
	//snprintf(out, 25, "%u" PRIu64 "", val);
	//if(VERBOSE_PRINT == 1)
		//ocall_print_string(out);
}

void print_integer_cont(uint64_t val){
	//char out[40];
	//snprintf(out, 40, "%u" PRIu64 "", val);
	//if(VERBOSE_PRINT == 1)
		//ocall_print_string(out);
}

void print_integer_cont(int64_t val){
	//char out[40];
	//snprintf(out, 40, "%u" PRId64 "", val);
	//if(VERBOSE_PRINT == 1)
		//ocall_print_string(out);
}


void print_integer(int val){
	char out[25];
	snprintf(out, 25, "%d" PRId32 "\n", val);
	//if(VERBOSE_PRINT == 1 || VERBOSE_COUT == 1)
		ocall_print_string(out);
}

void print_integer(unsigned int val){
	char out[25];
	snprintf(out, 25, "%u" PRIu32 "\n", val);
	//if(VERBOSE_PRINT == 1 || VERBOSE_COUT == 1)
		ocall_print_string(out);
}

void print_integer(uint64_t val){
	char out[80];
	print_string("Printing big int");
	snprintf(out, 80, "%llu" PRIu64 "\n", val);
	//if(VERBOSE_PRINT == 1 || VERBOSE_COUT == 1)
		ocall_print_string(out);
}

void print_integer(int64_t val){
	char out[40];
	snprintf(out, 40, "%lld" PRId64 "\n", val);
	//if(VERBOSE_PRINT == 1 || VERBOSE_COUT == 1)
		ocall_print_string(out);
}

void print_integer(std::string pretext, int val){
	//char out[25];
	//print_string_cont(pretext);
	//snprintf(out, 25, " (int): %u" PRIu64 "\n", val);
	//if(VERBOSE_PRINT == 1)
		//ocall_print_string(out);
}

void print_integer(std::string pretext, uint64_t val){
	//char out[40];
	//print_string_cont(pretext);
	//snprintf(out, 40, " (uint64_t): %u" PRIu64 "\n", val);
	//if(VERBOSE_PRINT == 1)
		//ocall_print_string(out);
}

void print_integer(std::string pretext, int64_t val){
	//char out[40];
	//print_string_cont(pretext);
	//snprintf(out, 40, " (int64_t): %u" PRId64 "\n", val);
	//if(VERBOSE_PRINT == 1)
		//ocall_print_string(out);
}

void print_c_string(uint8_t* buf, int len){
	char out[10];
	snprintf(out, 10, "Len %d \n", len);
	ocall_print_string(out);
	for(int i=0; i<len; i++){
		snprintf(out, 10, " %c", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}

void print_c_string(const uint8_t* buf, int len){
	char out[10];
	snprintf(out, 10, "Len %d \n", len);
	ocall_print_string(out);
	for(int i=0; i<len; i++){
		snprintf(out, 10, " %c", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}

void print_c_string(char* buf, int len){
	char out[10];
	snprintf(out, 10, "Len %d \n", len);
	ocall_print_string(out);
	for(int i=0; i<len; i++){
		snprintf(out, 10, " %c", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}

void print_c_string(const char* buf, int len){
	char out[10];
	snprintf(out, 10, "Len %d \n", len);
	ocall_print_string(out);
	for(int i=0; i<len; i++){
		snprintf(out, 10, " %c", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}


// To port placeholders for fprintf and other c print functions.
void fprintf(void* fileptr, const char* buf){
	//print_c_string(buf, strlen(buf));
}

void fprintf(void* fileptr, const char* buf, int val){
	//print_c_string(buf, strlen(buf));
}

sgx_ostream::sgx_ostream(std::string initstring){
	output = initstring;
}

sgx_ostream::sgx_ostream(){
	output = "";
}

sgx_ostream sgx_ostream::operator <<(sgx_ostream val){
	sgx_ostream conjugated = sgx_ostream(output+val.output);
	
	return conjugated;
}


sgx_ostream sgx_ostream::operator <<(int val){
	sgx_ostream conjugated = sgx_ostream(output+std::to_string(val));
	return conjugated;
}

sgx_ostream sgx_ostream::operator <<(std::string str){
	sgx_ostream conjugated = sgx_ostream(output+str);
	return conjugated;

}

sgx_ostream sgx_ostream::operator <<(char* cstr){
	std::string str(cstr);
	sgx_ostream conjugated = sgx_ostream(output+str);
	return conjugated;

}

sgx_ostream sgx_ostream::operator <<(const char* cstr){
	std::string str(cstr);
	sgx_ostream conjugated = sgx_ostream(output+str);
	return conjugated;

}

sgx_ostream::~sgx_ostream(){
	output.clear();
}

void sgx_cout_flush(sgx_ostream stream){
	VERBOSE_COUT = 1;
	print_string_cout(">>>SGX_OSTREAM: "+stream.output);
	VERBOSE_COUT = 0;
}


/*class sgx_ostream {
  public:
  std::string coutpretext;
  std::string output = "";

  sgx_ostream(){
  coutpretext = ">>>SGX_OSTREAM: ";
  output += coutpretext;
  }

  void operator <<(int val){
  output += stoi(val);
  }

  void operator <<(std::string str){

  }

  void operator <<(char* cstr){

  }

  void operator <<(const char* cstr){

  }


  };*/
