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

#ifndef TRUCE_DEFS_H_
#define TRUCE_DEFS_H_

#include "sgx_quote.h"

#define SP_CERT                "cert_and_key.pem"
//#define SPID                   {{0x4F, 0x37, 0xE6, 0xF5, 0x4A, 0x5C, 0x69, 0xCC, 0x73, 0xCF, 0x78, 0x74, 0x31, 0xA7, 0x9B, 0xDA}}                 
#define SPID                   {{0xE0, 0xE0, 0x63, 0x72, 0x00, 0xAA, 0xAD, 0x5E, 0x1C, 0x3B, 0x1B, 0xBA, 0xF2, 0x74, 0xE3, 0xC2}}                 

#define QUOTE_SIGN_TYPE        SGX_UNLINKABLE_SIGNATURE

#define IAS_URL     "https://api.trustedservices.intel.com/sgx/dev/attestation/v3/"

//#define IAS_URL     "https://test-as.sgx.trustedservices.intel.com:443/attestation/sgx/v2/"

#define SP_AS_PORT_DEFAULT    48023
#define SP_RS_PORT_DEFAULT    48033


#endif /* TRUCE_DEFS_H_ */
