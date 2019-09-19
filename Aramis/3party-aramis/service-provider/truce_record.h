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

#ifndef TRUCE_TRUCE_RECORD_H_
#define TRUCE_TRUCE_RECORD_H_

#include <openssl/sha.h>
#include "IAS_report.h"
#include "truce_public_keys.h"

typedef struct {
    uint8_t id[SHA256_DIGEST_LENGTH];
} truce_id_t;


typedef struct {
    IAS_report_t         ias_report;
    uint32_t            public_keys_size;
    truce_public_keys_t    *p_public_keys;
} truce_record_t;


#endif /* TRUCE_TRUCE_RECORD_H_ */
