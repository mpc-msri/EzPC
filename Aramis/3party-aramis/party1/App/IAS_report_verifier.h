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

#ifndef TRUCE_CLIENT_IAS_REPORT_VERIFIER_H_
#define TRUCE_CLIENT_IAS_REPORT_VERIFIER_H_

#include "../truce_headers/IAS_report.h"
#include "sgx_quote.h"

typedef void (*debug_print_func)(const char* str);

bool extract_quote_from_IAS_report(const IAS_report &ias_report,
                sgx_quote_t &quote,
                debug_print_func debug_print);


bool extract_quote_from_IAS_report(const IAS_report &ias_report,
                sgx_quote_t &quote);


#endif /* TRUCE_CLIENT_IAS_REPORT_VERIFIER_H_ */
