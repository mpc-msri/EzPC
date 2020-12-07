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

//Which matmul to run
#define MATMUL_THREADED //Fastest
//#define SLOW_MULT_CACHE_OPTI
//#define SLOW_MULT //Sequential 3-nested loop for matmul

//Other Optimizations
#define ALL_OPTI
//#define PARALLEL_SENDS_P2

//Switch all optimizations on or off
#ifdef ALL_OPTI
#define PARALLELPC
#define RUN_SHARECONV_OPTI
#define RUN_MSB_OPTI
#define CONV_OPTI
#define PARALLELIZE_CRITICAL
#define PARALLEL_AES
#define SPLIT_RELU
#define SPLIT_MAXPOOL
#define SPLIT_CONV
#define SPLIT_LOCAL_MATMUL
//#define PARALLEL_AES_ALL //This sometimes makes runtime slow
#define PARALLEL_AES_CONV_ALL
#endif

#define MATMUL_STACK //If OFF, then HEAP is used.
#define RELU_SPLIT_CHUNK_SIZE 40
#define MAXPOOL_SPLIT_CHUNK_SIZE 10
#define CONV_SPLIT_CHUNK_SIZE 40
#define LOCAL_MATMUL_CHUNK_COUNT 8
#define LOCAL_MATMUL_CHUNK_COUNT_2 2
#define DONT_CHUNK_LOCAL_MATMUL
#define DONT_CHUNK_LOCAL_MATMUL_2


