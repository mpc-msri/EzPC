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

/*
     Modified by Mayank, Microsoft Research India.
     mayankrathee.japan@gmail.com
 */


#ifndef TRUCE_UNTRUSTED_QUOTE_CREATOR_H_
#define TRUCE_UNTRUSTED_QUOTE_CREATOR_H_

#include <sgx_eid.h>
#include <sgx_quote.h>
#include <sgx_error.h>

void print_buffer(int nesting_level, uint8_t *buffer, size_t size);
void print_report_body(int nesting_level, sgx_report_body_t *p_report_body);
void print_quote(int nesting_level, sgx_quote_t *p_quote);

// note: report_data = sgx_sha256_msg(public_keys)

typedef sgx_status_t (*ecall_create_report_t)(
							sgx_enclave_id_t eid, //in
							sgx_status_t *retval, //in
							const sgx_target_info_t *target_info, //in
							sgx_report_t *report); //out


sgx_status_t create_quote_for_enclave(
								sgx_enclave_id_t eid, //in
								ecall_create_report_t ecall_create_report, //in
								sgx_quote_sign_type_t sign_type, //in
								const sgx_spid_t &spid, //in
								const sgx_quote_nonce_t &nonce, //in
								const uint8_t *p_sig_rl, //in
								uint32_t sig_rl_size, //in
								sgx_quote_t *&p_quote, //out
								uint32_t &quote_size); //out




#endif /* TRUCE_UNTRUSTED_QUOTE_CREATOR_H_ */
