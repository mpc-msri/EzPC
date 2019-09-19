/*
 * Copyright (C) 2011-2019 Intel Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of Intel Corporation nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * Modified by Mayank Rathee, Microsoft Research India.
 * mayankrathee.japan@gmail.com
 */

#include <stdarg.h>
#include <stdio.h>      /* vsnprintf */
#include <stdlib.h>

#include "Enclave.h"
#include "Enclave_t.h"  /* print_string */
//#include "sgx_tgmp.h"
#include "tsgxsslio.h"
#include "sgx_tcrypto.h"
#include "sgx_tseal.h"
#include "sgx_utils.h"
#include "sgx_trts.h"

#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>

#include <stdio.h>
#include <string>

#include "utils_ported_sgx.h"

// sgx_trusted_time
#include "sgx_tae_service.h"

// Attestation stuff
#include "truce_enclave.h"

// SecureNN Link
#include "main.h"

// Keys
#include "../files/all_keys.h"

// Data
#include "../files/data/dataparsed.h"

// Instream sgx
#include "../utils_sgx_port/utils_input_sgx.h"

int party_num = -1;

void register_keys(const char* key1, int size1, const char* key2, int size2, const char* key3, int size3, int pnum){
	char k1[size1+1];
	char k2[size2+1];
	char k3[size3+1];
	memcpy(k1, key1, size1);
	memcpy(k2, key2, size2);
	memcpy(k3, key3, size3);
	k1[size1] = '\0';
	k2[size2] = '\0';
	k3[size3] = '\0';
	
	std::string str_key1(k1);
	std::string str_key2(k2);
	std::string str_key3(k3);

	if(pnum == 0){
		keyA = str_key1;
		keyAB = str_key2;
		keyD = str_key3;
	}
	else if(pnum == 1){
		keyB = str_key1;
		keyAB = str_key2;
		keyD = str_key3;

	}
	else if(pnum == 2){
		keyC = str_key1;
		keyCD = str_key2;
		keyD = str_key3;
	}

}

void ocall_print_string(const char* buf){
	ocall_print_string_out(buf);
}

struct cmd_pack{
	int argc;
	char** argv;
};

char* cmd_a;
char** argvpack;

cmd_pack* cmd_parse(std::string input){
	struct cmd_pack* pack = new cmd_pack;
	std::vector<std::string> sliced = sgx_slice(input);
	pack->argc = sliced.size();
	argvpack = (char**)malloc(pack->argc+1);
	for(int i=0; i<=pack->argc; i++){
		if(i==0){
			cmd_a = "0"; // argv[0] does not matter
			argvpack[i] = cmd_a;
		}
		else{
			cmd_a = (char*)malloc(sliced[i-1].length()+1);
			const char* b = sliced[i-1].c_str();
			memcpy(cmd_a, b, sliced[i-1].length());
			cmd_a[sliced[i-1].length()] = '\0';
			argvpack[i] = cmd_a;
			pack->argv = argvpack;
		}
	}
	return pack;
}

void sgx_main(int pnum, const char *fmt, ...)
{
	party_num = pnum;
	struct cmd_pack* pack;
	struct cmd_pack* packsh;
	main_aramis(pnum);
}

