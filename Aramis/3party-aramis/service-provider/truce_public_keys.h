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

#ifndef TRUCE_TRUCE_PUBLIC_KEYS_H_
#define TRUCE_TRUCE_PUBLIC_KEYS_H_

#include "/home/mayank/Downloads/sgxsdk/include/sgx_tcrypto.h"

typedef struct {
    sgx_ec256_public_t     ec256_public_key;
    uint32_t             rsa4096_public_key_size;
    uint8_t             rsa4096_public_key[];
} truce_public_keys_t;

#endif /* TRUCE_TRUCE_PUBLIC_KEYS_H_ */
