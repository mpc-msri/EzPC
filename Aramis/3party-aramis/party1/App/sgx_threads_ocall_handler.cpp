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

#include "sgx_threads_ocall_handler.h"
#include "cross_call_counter.h"
#include "../src/globals.h"

using namespace std;

thread *threads[NO_CORES];

void ocall_parallelPC_spawn_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(parallelPC_sgx_threaded, global_eid, thread_num);	
	
}

void ocall_parallelPC_join_threads(){
	thread_calls++;	
	for(int i=0; i<NO_CORES; i++){
		threads[i][0].join();
	}
}

void ocall_matmul_spawn_threads(int thread_num){
	thread_calls++;	
	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(matmul_sgx_threaded, global_eid, thread_num);	
}

void ocall_matmul_join_threads(){
	thread_calls++;	
	for(int i=0; i<NO_CORES; i++){
		threads[i][0].join();
	}
}

void ocall_join_threads(){
	thread_calls++;	
	for(int i=0; i<NO_CORES; i++){
		threads[i][0].join();
	}
}

void ocall_populate_aes_spawn_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(aes_parallel_populate_thread_dispatcher, global_eid, thread_num);	
}

void ocall_sharesModuloOdd_aes_spawn_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(aes_parallel_sharesModuloOdd_thread_dispatcher, global_eid, thread_num);	

}

void ocall_sharesOfBits_aes_spawn_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(aes_parallel_sharesOfBits_thread_dispatcher, global_eid, thread_num);	

}

void ocall_deduce_zero_spawn_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(deduceZeroThreadDispatcher, global_eid, thread_num);	

}

void ocall_populate_flat_vector_aes_spawn_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(aes_parallel_populate_flat_vector_thread_dispatcher, global_eid, thread_num);	
}

void ocall_sharesOfBits_primary_aes_spawn_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(aes_parallel_sharesOfBits_primary_thread_dispatcher, global_eid, thread_num);	

}

void ocall_join_threads_half(){
	thread_calls++;	
	for(int i=0; i<(NO_CORES/2); i++){
		threads[i][0].join();
	}
}

void ocall_create_testing_threads(int thread_num){
	thread_calls++;	
	thread* new_thread = new thread[1];
	threads[thread_num] = new_thread;
	new_thread[0] = thread(peek_inside_enclave, global_eid, thread_num);	
}

void ocall_join_testing_threads(){
	thread_calls++;
	for(int i=0; i<NO_CORES; i++){
		threads[i][0].join();
	}
}
