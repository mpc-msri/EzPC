/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h> 
#include <iostream>
#include <fstream>

#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>
#include <curl/curl.h>
#include "sgx_urts.h"

#include "truce_client.h"
#include "quote_creator/quote_creator.h"

//Extract Quote from IAS Report
#include "IAS_report_verifier.h"

#include "sgx_urts.h"
#include "truce_u.h"
#include "Enclave_u.h"

#include "truce_addresses.h"
#include "print_prepends.h"
#include "mac_key_utils_app.h"

void ocall_print_string(const char *str);

void print_string(const char* str);

bool truce_client_init(const char* truce_server_address);
    
void print_buffer(uint8_t* buf, 
		int len);

void pretty_print_buffer(uint8_t* buf, 
		int len);

bool truce_client_recv_enclave_record(const truce_id_t &t_id, 
		truce_record_t &t_rec);

bool truce_client_extract_quote_from_record(const truce_record_t &t_rec, 
		sgx_quote_t &quote);

bool truce_client_verify_enclave_record(sgx_enclave_id_t enclave_id, 
		sgx_status_t &status, 
		const truce_id_t &t_id, 
		uint32_t t_id_size, 
		const truce_record_t &t_rec, 
		uint32_t t_rec_size, 
		const sgx_measurement_t &expected_mrenclave, 
		uint32_t mrenclave_size, 
		const sgx_measurement_t &expected_mrsigner, 
		uint32_t mrsigner_size, 
		int serverid);

bool truce_client_encrypt_secret(const truce_record_t &t_rec, 
		const uint8_t *secret, 
		uint32_t secret_len, 
		uint8_t *&output, 
		uint32_t &output_size);

int attest_main_alice(sgx_enclave_id_t &enclave_id);

int attest_main_bob(sgx_enclave_id_t &enclave_id);

int attest_main_charlie(sgx_enclave_id_t &enclave_id);

