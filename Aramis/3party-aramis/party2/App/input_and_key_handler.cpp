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

#include "input_and_key_handler.h"
#include <vector>

using namespace std;

void read_keys_securenn(sgx_enclave_id_t enclave_id, int party_num){
	if(party_num == 0){
	// First read the keys and store them in all_keys.h
	// Send keyA, keyAB and keyD
		ifstream f1("files/keyA");
		string str1{istreambuf_iterator<char>(f1), istreambuf_iterator<char>() };
		f1.close();
		if(str1[str1.length()-1]=='\n'){
			str1 = str1.substr(0, str1.length()-1);
		}
		const char* key1 = str1.c_str();
		//cout<<str1.length()<<endl;
		
		ifstream f2("files/keyAB");
		string str2{istreambuf_iterator<char>(f2), istreambuf_iterator<char>() };
		f2.close();
		const char* key2 = str2.c_str();
		//cout<<str2.length()<<endl;
	
		ifstream f3("files/keyD");
		string str3{istreambuf_iterator<char>(f3), istreambuf_iterator<char>() };
		f3.close();
		const char* key3 = str3.c_str();
		//cout<<str3.length()<<endl;
	
		register_keys(enclave_id, key1, str1.length(), key2, str2.length(), key3, str3.length(), 0);	
	}
	else if(party_num == 1){
	// Send keyB, keyAB and keyD
		ifstream f1("files/keyB");
		string str1{istreambuf_iterator<char>(f1), istreambuf_iterator<char>() };
		f1.close();
		if(str1[str1.length()-1]=='\n'){
			str1 = str1.substr(0, str1.length()-1);
		}
		const char* key1 = str1.c_str();
		
		ifstream f2("files/keyAB");
		string str2{istreambuf_iterator<char>(f2), istreambuf_iterator<char>() };
		f2.close();
		const char* key2 = str2.c_str();
		
		ifstream f3("files/keyD");
		string str3{istreambuf_iterator<char>(f3), istreambuf_iterator<char>() };
		f3.close();
		const char* key3 = str3.c_str();
		
		register_keys(enclave_id, key1, str1.length(), key2, str2.length(), key3, str3.length(), 1);	

	}
	else if(party_num == 2){
	// Send keyC, keyCD and keyD
		ifstream f1("files/keyC");
		string str1{istreambuf_iterator<char>(f1), istreambuf_iterator<char>() };
		f1.close();
		if(str1[str1.length()-1]=='\n'){
			str1 = str1.substr(0, str1.length()-1);
		}
		const char* key1 = str1.c_str();
		
		ifstream f2("files/keyCD");
		string str2{istreambuf_iterator<char>(f2), istreambuf_iterator<char>() };
		f2.close();
		const char* key2 = str2.c_str();
		
		ifstream f3("files/keyD");
		string str3{istreambuf_iterator<char>(f3), istreambuf_iterator<char>() };
		f3.close();
		const char* key3 = str3.c_str();
		
		register_keys(enclave_id, key1, str1.length(), key2, str2.length(), key3, str3.length(), 2);	

	}
	return;	
}

void read_input_and_keys(sgx_enclave_id_t enclave_id, int party_num){
	read_keys_securenn(enclave_id, party_num);
}

