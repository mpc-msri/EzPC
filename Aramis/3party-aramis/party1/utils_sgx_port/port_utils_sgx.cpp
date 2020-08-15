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
// This contains the code to hook the functions that are
// not present in SGX enclave SDK.

#include "port_utils_sgx.h"

template<typename T>
int sgx_fscanf(SGX_FILE *f, std::string regex, T* first){
	std::string str="";
	char ch = '0';
	sgx_fread(&ch, 1, 1, f);
	while(ch!='\n' && sgx_feof(f)==0){
		str += ch;
		sgx_fread(&ch, 1, 1, f);
	}
	if(regex.length()==1 && regex[0]=='\n'){
		return NORMAL_EXEC; 
	}
	std::vector<std::string> res = sgx_slice(str);
	if(sizeof(regex)==2){
		if(regex[0] != '%'){
			return FSCANF_ERROR;
		}
		switch(regex[1]){
			case 'd': *first = stoi(res[0]);
				  break;
			case 's': *first = res[0];
				  break;
			default: return FSCANF_ERROR;

		}
	}
	return NORMAL_EXEC;
}

template<typename T, typename... Tothers>
int sgx_fscanf(SGX_FILE* f, std::string regex, T* first, Tothers*... args){
	std::string str="";
	char ch = '0';
	sgx_fread(&ch, 1, 1, f);
	while(ch!='\n' && sgx_feof(f)==0){
		str+=ch;
		sgx_fread(&ch, 1, 1, f);
	}
	std::vector<std::string> res = sgx_slice(str);
	
	sgx_assign_veradic(0, regex, res, first, args...);

	return NORMAL_EXEC;
}

template<typename T, typename U>
void sgx_assign_veradic(int cnt, std::string regex, T newval, U* oldvalref){
	switch(regex[1]){
		case 'd': *oldvalref = stoi(newval[0]);
			  break;
		case 's': *oldvalref = newval[0];
			  break;
	}
}

template<typename T, typename U, typename... Uargs>
void sgx_assign_veradic(int cnt, std::string regex, T newval, U* oldvalref, Uargs*... others){
	std::string temp = ""+regex[2*cnt]+regex[2*cnt+1];
	switch(temp[1]){
		case 'd': *oldvalref = stoi(newval[cnt]);
			  break;
		case 's': *oldvalref = newval[cnt];
	}
	sgx_assign_veradic(cnt+1, regex, newval, others...);
}

// Slice off whitespaces and return vector of values
std::vector<std::string> sgx_slice(std::string str){
	std::vector<std::string> res;
	int i=0;
	while(str[0]==' '){
		str = str.substr(1);
	}
	while(i<str.size()){
		int j=i;
		std::string temp = "";
		while(str[j]!=' '){
			temp += str[j];
			j++;
			if(j==str.size()){
				break;
			}
		}
		res.push_back(temp);
		while(str[j]==' '){
			j++;
			if(j==str.size())
				break;
		}
		i = j;
		//i++;
	}
	return res;
}

void sgx_read_parsed_int(std::vector<std::string> store, int ctr, int &input){
	input = stoi(store[ctr]);
}

void sgx_read_parsed_string(std::vector<std::string> store, int ctr, std::string strinput){
	strinput = store[ctr];
}
