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

#ifndef _TRUCE_U_H
#define _TRUCE_U_H

#include "sgx_eid.h"
#include <openssl/sha.h>
#include "../truce_headers/defs.h"
#include "../aux_lib/aux_funcs.h"

#ifdef  __cplusplus
extern "C" {
#endif

const int TRUCE_ID_LENGTH = SHA256_DIGEST_LENGTH;

typedef struct {
    sgx_enclave_id_t enclave_id;
    uint8_t truce_id[SHA256_DIGEST_LENGTH];
} truce_session_t;

typedef struct {
    char *truce_server_address;
    char *seal_path = nullptr;
} truce_config_t;

bool truce_session(sgx_enclave_id_t enclave_id,
        const truce_config_t &config,
        truce_session_t &truce_session);

bool truce_add_secret(const truce_session_t &t_session,
        const uint8_t* secret_buf,
        uint32_t secret_buf_size);



#ifdef  __cplusplus
}
#endif

#endif
