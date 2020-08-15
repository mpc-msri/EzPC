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

#include "sgx_intrin.h"
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdint.h>

#define RUN_FULL_NETWORK

#ifndef RUN_FULL_NETWORK
//To benchmark specific functions only
//#define SKIP_FIRST_RELU
//#define SKIP_FIRST_MAXPOOL
//#define SKIP_RELU
//#define SKIP_MAXPOOL
//#define SKIP_AVGPOOL
//#define SKIP_ARGMAX
//#define SKIP_CINS
//#define SKIP_CONV
//#define HALT_COMM //This will make the execution only local without any comm. OCALLs for threads will still happen.
#endif

