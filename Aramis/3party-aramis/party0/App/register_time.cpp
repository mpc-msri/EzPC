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

#include "register_time.h"
#include "../src/globals.h"

std::chrono::high_resolution_clock::time_point reg_start;
std::chrono::high_resolution_clock::time_point reg_start_master_bridge;

void init_time_register(){
	reg_start = std::chrono::high_resolution_clock::now();
}

void touch_time(){
	reg_start = std::chrono::high_resolution_clock::now();
}

void leave_time(){
	std::cout << "[REGISTER_TIME] >> RUNTIME of " << "SGX process" << ": " <<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-reg_start).count() << " ms " << std::endl; 

}

void leave_time_get_val(uint64_t* val){
	uint64_t t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-reg_start).count();
	memcpy(val, &t, sizeof(uint64_t));
}

void tread_cross_bridge(int size, uint64_t* vec){
	vec[0] = 1;
}
