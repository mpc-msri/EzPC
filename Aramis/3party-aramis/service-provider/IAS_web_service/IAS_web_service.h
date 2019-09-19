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

#ifndef TRUCE_IAS_WEB_SERVICE_IAS_WEB_SERVICE_H_
#define TRUCE_IAS_WEB_SERVICE_IAS_WEB_SERVICE_H_

#include <string>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <stdio.h>
#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <iostream>

#include "../IAS_report.h"

enum IAS {
    sigrl,
    report
};

struct attestation_verification_report_t {
    char* report_id;
    char* isv_enclave_quote_status;
    char* timestamp;
};

struct attestation_evidence_payload_t {
    char* isv_enclave_quote;
};

struct ias_report_header_t {
    int report_status;
    int content_length;
    char* request_id;
    char* iasreport_signature;
    char* iasreport_signing_certificate;
};

struct ias_report_container_t {
    char *p_report;
    size_t size;
};


typedef struct {
    CURL *curl;
    std::string url;
} ias_web_service_t ;

bool init_ias_web_service(ias_web_service_t &ias_web_service,
        const std::string &client_cert_path,
        const std::string &ias_url);

bool get_sig_rl(const ias_web_service_t &ias_web_service, //in
        const uint8_t (&epid_group_id)[4], //in
        uint8_t *&p_sig_rl, //out
        uint32_t &sig_rl_size //out
);

bool get_ias_report(const ias_web_service_t &ias_web_service, //in
        const uint8_t *p_quote, //in
        uint32_t quote_size,
        const uint8_t *p_pse_manifest, //in (optional)
        uint32_t pse_manifest_size, // in (should be 0 if pseManifest = NULL)
        const uint8_t *p_nonce, //in (optional)
        uint32_t nonce_size, // in (should be 0 if nonce = NULL)
        IAS_report &ias_report //out
);



#endif



