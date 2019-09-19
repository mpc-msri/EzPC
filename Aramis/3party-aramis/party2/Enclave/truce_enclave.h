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
 * Modified by Mayank Rathee, Microsoft Research India.
 * mayankrathee.japan@gmail.com
 */

#include <assert.h>
#include "Enclave_t.h"
#include "sgx_tcrypto.h"
#include "sgx_tseal.h"
#include "sgx_utils.h"
#include "string.h"
#include "truce_t.h"
#include "truce_public_keys.h"
#include "truce_private_keys.h"
#include "IAS_report_verifier.h"
#include <stdio.h>  // for snprintf
#include <string.h>
#include <algorithm>    // std::min
#include "sgx_trts.h" // sgxssl aes gcm


#include <openssl/crypto.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/rsa.h>


