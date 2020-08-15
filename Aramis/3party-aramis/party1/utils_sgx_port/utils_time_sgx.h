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

#include "sgx_trts.h"
// sgx_trusted_time
#include "sgx_tae_service.h"

//print routine
#include "utils_print_sgx.h"

#define MICROSECMULT 1000000
#define MILLISECMULT 1000

// TODO: Include error handler with this

class sgx_time{
	private:
		// Nonce ensured that time was measured
		// correctly. If nonce_pre == nonce_post
		// correct time measurement is
		// guaranteed. Otherwise, no guarantees.
		sgx_time_source_nonce_t nonce_pre = {0};
		sgx_time_source_nonce_t nonce_post = {0};

		// sgx_time_t a.k.a uint64_t records
		// the current time in seconds
		sgx_time_t time_pre;
		sgx_time_t time_post;

		bool session_running = 0; // 0 is Off.

		int create_session();

		int close_session();

	public:
		int error_code_pse = SGX_ERROR_UNEXPECTED;
		int error_code_time = SGX_ERROR_UNEXPECTED;
		
		//int sgx_time();

		sgx_time();

		int start_recording();

		int end_recording();

		uint64_t get_measurement();

		// TODO: Support for milliseconds and
		// other time formats.
		
		uint64_t get_measurement_ms();
		
		// This is the time unit used
		// as defualt in EMP-Toolkit
		uint64_t get_measurement_us();
		
		void print_error_info();

		~sgx_time();;
};
