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

#include <string.h>
#include <sgx_uae_service.h>
#include <stdio.h>
#include <stdlib.h>

#include "quote_creator.h"

#define MAX_NUMBER_IN_LINE    10

void print_buffer(int nesting_level, uint8_t *buffer, size_t size) {
    char *nesting_string = (char *) calloc(4*nesting_level+1, 1);
    memset(nesting_string, ' ', 4*nesting_level);

    if (size > 0) {
        printf("%s 0x%x", nesting_string, buffer[0]);
        for (int i=1; i<size; ++i) {
            if (i%MAX_NUMBER_IN_LINE == 0) {
                printf(",\n%s 0x%x", nesting_string, buffer[i]);
            }
            else {
                printf(",0x%x", buffer[i]);
            }
        }
        printf("\n");
    }

    free(nesting_string);
}



void print_report_body(int nesting_level, sgx_report_body_t *p_report_body) {

    char *nesting_string = (char *) calloc(4*nesting_level+1, 1);
    memset(nesting_string, ' ', 4*nesting_level);

    printf("%s cpu_svn =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->cpu_svn.svn, sizeof(p_report_body->cpu_svn.svn));
    printf("%s misc_select = 0x%x\n", nesting_string, p_report_body->misc_select);
    printf("%s reserved1 =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->reserved1, sizeof(p_report_body->reserved1));
    printf("%s attributes =\n", nesting_string);
    printf("%s     flags = 0x%lx\n", nesting_string, p_report_body->attributes.flags);
    printf("%s     xfrm = 0x%lx\n", nesting_string, p_report_body->attributes.xfrm);
    printf("%s mr_enclave =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->mr_enclave.m, sizeof(p_report_body->mr_enclave));
    printf("%s reserved2 =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->reserved2, sizeof(p_report_body->reserved2));
    printf("%s mr_signer =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->mr_signer.m, sizeof(p_report_body->mr_signer));
    printf("%s reserved3 =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->reserved3, sizeof(p_report_body->reserved3));
    printf("%s isv_prod_id = 0x%x\n", nesting_string, p_report_body->isv_prod_id);
    printf("%s isv_svn = 0x%x\n", nesting_string, p_report_body->isv_svn);
    printf("%s reserved4 =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->reserved4, sizeof(p_report_body->reserved4));
    printf("%s report_data =\n", nesting_string);
    print_buffer(nesting_level+1, p_report_body->report_data.d, sizeof(p_report_body->report_data));
}

void print_quote(int nesting_level, sgx_quote_t *p_quote) {

    char *nesting_string = (char *) calloc(4*nesting_level+1, 1);
    memset(nesting_string, ' ', 4*nesting_level);

    printf("%s version = 0x%x\n", nesting_string, p_quote->version);
    printf("%s sign_type = 0x%x\n", nesting_string, p_quote->sign_type);
    printf("%s epid_group_id = 0x%x,0x%x,0x%x,0x%x\n", nesting_string,
                                                    p_quote->epid_group_id[0],
                                                    p_quote->epid_group_id[1],
                                                    p_quote->epid_group_id[2],
                                                    p_quote->epid_group_id[3]);
    printf("%s qe_svn = 0x%x\n", nesting_string, p_quote->qe_svn);
    printf("%s pce_svn = 0x%x\n", nesting_string, p_quote->pce_svn);
    printf("%s xeid = 0x%x\n", nesting_string, p_quote->xeid);
    printf("%s basename =\n", nesting_string);
    print_buffer(nesting_level+1, p_quote->basename.name, 32);
    printf("%s report =\n", nesting_string);
    print_report_body(nesting_level+1, &p_quote->report_body);
    printf("%s signature_len = %d\n", nesting_string, p_quote->signature_len);
    printf("%s signature =\n", nesting_string);
    print_buffer(nesting_level+1, p_quote->signature, p_quote->signature_len);

}


sgx_status_t create_quote_for_enclave(
                                sgx_enclave_id_t eid,
                                ecall_create_report_t ecall_create_report,
                                sgx_quote_sign_type_t sign_type,
                                const sgx_spid_t &spid,
                                const sgx_quote_nonce_t &nonce,
                                const uint8_t *p_sig_rl,
                                uint32_t sig_rl_size,
                                sgx_quote_t *&p_quote,
                                uint32_t &quote_size) {

    sgx_status_t retval = SGX_SUCCESS;
    sgx_status_t ret = SGX_SUCCESS;
    sgx_epid_group_id_t gid = {0};
    sgx_target_info_t qe_target_info;
    sgx_report_t qe_report;
    sgx_report_t report;

    memset(&report, 0, sizeof(sgx_report_t));
    memset(&qe_report, 0, sizeof(sgx_report_t));
    memset(&qe_target_info, 0, sizeof(sgx_target_info_t));

    ret = sgx_init_quote(&qe_target_info, &gid);
    if (ret != SGX_SUCCESS) {
        printf("ERROR: sgx_init_quote has failed. error code %d\n", ret);
        goto cleanup;
    }

    // Create Report(eid) signed (using CMAC) by the key of the quoting enclave.
    ret = ecall_create_report(eid,
            &retval,
            &qe_target_info,
            &report);


    if (ret != SGX_SUCCESS) {
        printf("ERROR: ecall_create_report has failed. error code %d\n", ret);
        goto cleanup;
    }


    ret = sgx_calc_quote_size(p_sig_rl, sig_rl_size, &quote_size);
    if (ret != SGX_SUCCESS) {
        printf("ERROR: sgx_calc_quote_size has failed. error code %d\n", ret);
        goto cleanup;
    }

    p_quote = (sgx_quote_t *) calloc(1, quote_size);
    if (NULL == p_quote) {
        printf("ERROR: failed to allocate %d byte for quote\n", quote_size);
        ret = SGX_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    ret = sgx_get_quote(&report,
            sign_type,
            &spid,
            &nonce,
            p_sig_rl,
            sig_rl_size,
            &qe_report,
            p_quote,
            quote_size);

    if (ret != SGX_SUCCESS) {
        printf("ERROR: sgx_get_quote has failed. error code %d\n", ret);
        goto cleanup;
    }

cleanup:
    return ret;
}
