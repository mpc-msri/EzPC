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

// Written by Mayank (t-may@microsoft.com)
// Microsoft Research India
// This contains the headers of the code which hook the functions
// that are not present in SGX enclave SDK.


#include "sgx_trts.h"
#include "sgx_tprotected_fs.h"
#include <map>
#include <string>
#include <vector>

#include <iterator>
#include <typeinfo>
#include <functional>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <initializer_list>
#include <tuple>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "../Enclave/Enclave.h"
#include "Enclave_t.h"

// Error classes
#define NORMAL_EXEC 0
#define FSCANF_ERROR 1

/*extern std::map<int, std::string> error_info = 
{
	{1, "fscanf error"}
};
*/

/*
 fscanf alternative
 */

template<typename T>
int sgx_fscanf(SGX_FILE *f, std::string regex, T* first);

template<typename T, typename... Tothers>
int sgx_fscanf(SGX_FILE* f, std::string regex, T* first, Tothers*... args);

template<typename T, typename U>
void sgx_assign_veradic(int cnt, std::string regex, T newval, U* oldvalref);

template<typename T, typename U, typename... Uargs>
void sgx_assign_veradic(int cnt, std::string regex, T newval, U* oldvalref, Uargs*... others);

// Slice off whitespaces and return vector of values
std::vector<std::string> sgx_slice(std::string str);

void sgx_read_parsed_int(std::vector<std::string> store, int ctr, int &input);

void sgx_read_parsed_string(std::vector<std::string> store, int ctr, std::string strinput);
