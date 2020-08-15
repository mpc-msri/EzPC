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

#include "utils_time_sgx.h"

int sgx_time::create_session(){
	if(session_running){
		return SGX_ERROR_UNEXPECTED;
	}
	error_code_pse = sgx_create_pse_session();
	return error_code_pse;
}

int sgx_time::close_session(){
	if(!session_running){
		return SGX_SUCCESS;
	}
	return sgx_close_pse_session();
}

//int sgx_time::sgx_time(){
	//return create_session();
//}

sgx_time::sgx_time(){
	create_session();
}

int sgx_time::start_recording(){
	error_code_time = sgx_get_trusted_time(&time_pre, &nonce_pre);
	return error_code_time;
}

int sgx_time::end_recording(){
	error_code_time = sgx_get_trusted_time(&time_post, &nonce_post);
	if(memcmp(&nonce_pre, &nonce_post, sizeof(sgx_time_source_nonce_t)) == 0){
		return 1;
	}
	else{
		return -1;
	}
}

uint64_t sgx_time::get_measurement(){
	uint64_t measurement = time_post - time_pre;
	// We can reset session_running
	session_running = 0;
	return measurement;
}

// TODO: Support for milliseconds and
// other time formats.

uint64_t sgx_time::get_measurement_ms(){
	uint64_t measurement = time_post - time_pre;
	// We can reset session_running
	session_running = 0;
	return measurement*MILLISECMULT;

}

// This is the time unit used
// as defualt in EMP-Toolkit
uint64_t sgx_time::get_measurement_us(){
	uint64_t measurement = time_post - time_pre;
	// We can reset session_running
	session_running = 0;
	return measurement*MICROSECMULT;

}

void sgx_time::print_error_info(){
	int error_code = error_code_time;
	if(error_code != SGX_SUCCESS){
		switch(error_code){
			case SGX_ERROR_SERVICE_UNAVAILABLE:
				print_string("Architecture does not support time");
				break;
			case SGX_ERROR_SERVICE_TIMEOUT:
				print_string("Service timeout");
				break;
			case SGX_ERROR_BUSY:
				print_string("Server is busy right now");
				break;
			case SGX_ERROR_INVALID_PARAMETER:
				print_string("Invalid pointers");
				break;
			case SGX_ERROR_NETWORK_FAILURE:
				print_string("Network issue was encountered");
				break;
			case SGX_ERROR_OUT_OF_MEMORY:
				print_string("SGX out of memory error");
				break;
			case SGX_ERROR_OUT_OF_EPC:
				print_string("Out of EPC pages error");
				break;
			case SGX_ERROR_UNEXPECTED:
				print_string("Unexpected error occured");
				break;
			default:
				print_string("Unknown error as to why time is not working");
				print_integer(error_code);
				print_string("This is error code");
				break;
		}
	}

}

sgx_time::~sgx_time(){
	close_session();
}

