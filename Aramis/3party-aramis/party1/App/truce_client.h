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

#ifndef _TRUCE_CLIENT_H
#define _TRUCE_CLIENT_H


#include <stdint.h>
#include "../truce_headers/IAS_report.h"
#include "../truce_headers/truce_record.h"
#include "../truce_headers/defs.h"
#include "../aux_lib/aux_funcs.h"
#include "sgx_quote.h"

bool truce_client_init(const char* truce_server_address);

bool truce_client_recv_enclave_record(
        const truce_id_t &t_id,
        truce_record_t &t_rec);


// TODO: Should be removed in the future.
// We should implement a way of calculating mrenclave and mrsigner of a given SO.
bool truce_client_extract_quote_from_record(
        const truce_record_t &t_rec,
        sgx_quote_t &quote);

bool truce_client_verify_enclave_record(
        const truce_id_t &t_id,
        const truce_record_t &t_rec,
        const sgx_measurement_t &expected_mrenclave,
        const sgx_measurement_t &expected_mrsigner);


bool truce_client_encrypt_secret(
        const truce_record_t &t_rec,
        const uint8_t *secret,
        uint32_t secret_len,
        uint8_t *&output, // output - should be freed outside.
        uint32_t &output_size); // output

#endif

