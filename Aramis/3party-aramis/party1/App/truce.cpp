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

#include "truce_u.h"
#include "quote_creator.h"

#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include <errno.h>
#include "truce_app.h"

#include "Enclave_u.h"

// Needed to create enclave and do ecall.
#include "sgx_urts.h"

#ifndef SAFE_FREE
#define SAFE_FREE(ptr) {if (NULL != (ptr)) {free(ptr); (ptr) = NULL;}}
#endif


#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h> 
#include <dirent.h> 
#include <openssl/sha.h>


bool truce_session(sgx_enclave_id_t enclave_id,
        const truce_config_t &config,
        truce_session_t &truce_session)
{
    bool ret = false;
    sgx_status_t sgx_ret = SGX_SUCCESS;
    sgx_status_t status = SGX_SUCCESS;
    FILE* OUTPUT = stdout;
    FILE *seal_file = NULL;
    const char *seal_file_name = config.seal_path;
    uint8_t *sealed_keys = NULL;
    uint32_t sealed_keys_size = 0;
    sgx_quote_t *p_quote = NULL;
    uint32_t quote_size = 0;
    uint8_t *p_sig_rl = NULL;
    uint32_t sig_rl_size = 0;
    sgx_quote_nonce_t nonce = {{0}};
    int sockfd = 0;
    sgx_spid_t spid = {{0}};
    uint8_t *p_public_keys = NULL;
    uint32_t public_keys_size = 0;
    uint8_t attestation_result = 0;
    bool init_with_seal_file = false;

    const char* service_prov_addr = "127.0.0.1";
    char sp_address[100];
    int sp_port;

    const char *port_pos = strchr(config.truce_server_address, ':');
    if (port_pos) {
        size_t addr_len = port_pos - config.truce_server_address;
        memcpy(sp_address, config.truce_server_address, addr_len);
        sp_address[addr_len] = '\0';
        sp_port = atoi(port_pos + 1);
    }
    else {
        memcpy(sp_address, config.truce_server_address, strlen(config.truce_server_address));
        sp_port = SP_AS_PORT_DEFAULT;
    }


    if (seal_file_name != NULL) {
        seal_file = fopen(seal_file_name, "rb");
    }

    if (seal_file != NULL) {
        // The Enclave's keys were sealed on disk. Restore them.

        fprintf(OUTPUT, "Reading keys from seal file: %s\n", seal_file_name);
        if (fread(&sealed_keys_size, 1, 4, seal_file) != 4) {
            fprintf(OUTPUT, "ERROR: failed read 4 bytes of sealed_keys_size\n");
            goto cleanup;
        }
        sealed_keys= (uint8_t *) malloc(sealed_keys_size);
        if (NULL == sealed_keys) {
            fprintf(OUTPUT, "ERROR: failed to allocate %d bytes for sealed_keys\n",sealed_keys_size);
            goto cleanup;
        }
        if (fread(sealed_keys, 1, sealed_keys_size, seal_file) != sealed_keys_size) {
            fprintf(OUTPUT, "ERROR: failed read %d bytes of sealed_keys_size\n", sealed_keys_size);
            goto cleanup;
        }

        printf("(1) sealed_keys_size = %u\n", sealed_keys_size);

	sgx_ret = ECALL_unseal_keys(enclave_id,
		&status,
		sealed_keys,
		sealed_keys_size);

        if (sgx_ret != SGX_SUCCESS || status != SGX_SUCCESS) {
            fprintf(OUTPUT, "ERROR: ECALL_unseal_keys has failed: (ret,status) = (0x%x,0x%x)\n", ret, status);
            goto cleanup;
        }
        init_with_seal_file = true;

    }
    else {
        // No seal file found for Enclave's keys. Generate new ones.
#ifdef VERBOSE_PRINT
        fprintf(OUTPUT, "No available seal file. Generating new keys\n");
#endif
	sgx_ret = ECALL_generate_keys(enclave_id, &status);
        if (sgx_ret != SGX_SUCCESS || status != SGX_SUCCESS) {
            fprintf(OUTPUT, "ERROR: ECALL_generate_keys has failed: (ret,status) = (0x%x,0x%x)\n", ret, status);
            goto cleanup;
        }

        if (seal_file_name != NULL) {
            // Seal the new keys to seal_path.

            fprintf(OUTPUT, "Sealing the new keys to %s\n", seal_file_name);
            errno = 0;
            seal_file = fopen(seal_file_name, "wb");
            if (NULL == seal_file) {
                fprintf(OUTPUT, "ERROR: fopen for writing has failed. errno = %d, file = %s\n", errno, seal_file_name);
                goto cleanup;
            }

	    ret = ECALL_get_sealed_keys_size(enclave_id,
		    &status,
		    &sealed_keys_size);

            if (sgx_ret != SGX_SUCCESS || status != SGX_SUCCESS) {
                fprintf(OUTPUT, "ERROR: ECALL_get_sealed_keys_size has failed: (ret,status) = (0x%x,0x%x)\n", ret, status);
                goto cleanup;
            }

            sealed_keys = (uint8_t *) malloc(sealed_keys_size);
            if (NULL == sealed_keys) {
                fprintf(OUTPUT, "ERROR: failed to allocate %u bytes for sealed_keys\n", sealed_keys_size);
                goto cleanup;
            }

	    sgx_ret = ECALL_seal_keys(enclave_id,
		    &status,
		    sealed_keys,
		    sealed_keys_size);

            if (sgx_ret != SGX_SUCCESS || status != SGX_SUCCESS) {
                fprintf(OUTPUT, "ERROR: ECALL_seal_keys has failed: (ret,status) = (0x%x,0x%x)\n", ret, status);
                goto cleanup;
            }

            if (4 != fwrite(&sealed_keys_size, 1, 4, seal_file)) {
                fprintf(OUTPUT, "ERROR: failed to write 4 bytes to %s\n", seal_file_name);
                goto cleanup;
            }

            printf("(2) sealed_keys_size = %u\n", sealed_keys_size);
            if (sealed_keys_size !=
                    fwrite(sealed_keys, 1, sealed_keys_size, seal_file)) {
                fprintf(OUTPUT, "ERROR: failed to write %u bytes to %s\n", sealed_keys_size, seal_file_name);
                goto cleanup;
            }

        }

    }

    // get public_keys_size
    sgx_ret = ECALL_get_public_keys_size(enclave_id,
	    &status,
	    &public_keys_size);
    if (sgx_ret != SGX_SUCCESS || status != SGX_SUCCESS) {
        fprintf(OUTPUT, "ERROR: ECALL_get_public_keys_size has failed\n");
        goto cleanup;
    }

    // get enclave's public_keys
    p_public_keys = (uint8_t *) malloc(public_keys_size);
    if (NULL == p_public_keys) {
        fprintf(OUTPUT, "ERROR: failed to allocate %u bytes for public_keys\n", public_keys_size);
        goto cleanup;
    }
    sgx_ret = ECALL_get_public_keys(enclave_id,
	    &status,
	    p_public_keys,
	    public_keys_size);

    if (sgx_ret != SGX_SUCCESS || status != SGX_SUCCESS) {
        fprintf(OUTPUT, "ERROR: ECALL_get_public_keys has failed\n");
        goto cleanup;
    }

    // At that point, the attestation process with the SP has succeeded.
    // Create appropriate truce_session:
    truce_session.enclave_id = enclave_id;
    SHA256(p_public_keys, public_keys_size, truce_session.truce_id);


    if (init_with_seal_file) {
        // No need to connect to SP. Can exit successfully.
        ret = true;
        goto cleanup;
    }


    // Connecting to the Service Provider
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Connecting to SP at address %s : %d\n", sp_address, sp_port);
#endif    
    
    if (!inet_connect(sockfd, service_prov_addr, sp_port)) {
        fprintf(OUTPUT, "ERROR: failed to connect to SP at address %s : %d\n", sp_address, sp_port);
        goto cleanup;
    }

    // Receive SPID
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Receiving SPID...\n");
#endif
    if (!read_all(sockfd, (uint8_t *) &spid, sizeof(spid))) {
        fprintf(OUTPUT, "ERROR: failed to recv spid\n");
        goto cleanup;
    }


    // Receive nonce
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Receiving nonce...\n");
#endif    
    if (!read_all(sockfd, (uint8_t *) &nonce, sizeof(nonce))) {
        fprintf(OUTPUT, "ERROR: failed to recv nonce\n");
        goto cleanup;
    }

    // Creating quote assuming that sigrl is empty, in order to extract epid_group_id.
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Creating quote with empty SigRL...\n");
#endif   
    sgx_ret = create_quote_for_enclave(
		enclave_id,
		ECALL_create_enclave_report,
		QUOTE_SIGN_TYPE,
		spid,
		nonce,
		NULL,
		0,
		p_quote,
		quote_size);


    if (sgx_ret != SGX_SUCCESS) {
        fprintf(OUTPUT, "ERROR: create_quote_for_enclave has failed.\n");
        goto cleanup;
    }

    // Send epid_group_id
#ifdef VERBOSE_PRINT    
    fprintf(OUTPUT, "Sending epid_group_id...\n");
#endif
    if (!write_all(sockfd, (uint8_t *) &p_quote->epid_group_id, 4)) {
        fprintf(OUTPUT, "ERROR: failed to write epid_group_id\n");
        goto cleanup;
    }


    // Receive SigRL size
    if (!read_all(sockfd, (uint8_t *) &sig_rl_size, 4)) {
        fprintf(OUTPUT, "ERROR: failed to recv sigrl_size\n");
        goto cleanup;
    }
    if (sig_rl_size != 0) {

        // Receive SigRL
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Receiving %u bytes of SigRL\n", sig_rl_size);
#endif
	p_sig_rl = (uint8_t *) malloc(sig_rl_size);
        if (NULL == p_sig_rl) {
            fprintf(OUTPUT, "ERROR: failed allocate %u bytes for SigRL\n", sig_rl_size);
            goto cleanup;
        }
        if (!read_all(sockfd, p_sig_rl, sig_rl_size)) {
            fprintf(OUTPUT, "ERROR: failed to recv sigrl_size\n");
            goto cleanup;
        }

        free(p_quote);
        p_quote = NULL;

        // Create New Quote with the current sigrl
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Creating new quote with the current SigRL\n");
#endif
	sgx_ret = create_quote_for_enclave(
		enclave_id,
		ECALL_create_enclave_report,
		SGX_LINKABLE_SIGNATURE,
		spid,
		nonce,
		p_sig_rl,
		sig_rl_size,
		p_quote,
		quote_size);

        if (sgx_ret != SGX_SUCCESS) {
            printf("ERROR: create_quote_for_agent_enclave has failed. ret = %d\n", ret);
            goto cleanup;
        }
    }
    else {
#ifdef VERBOSE_PRINT    
	    fprintf(OUTPUT, "SigRL size = 0. Using the same Quote. \n");
#endif    
    }

    // Send Quote size
    if (!write_all(sockfd, (uint8_t *) &quote_size, 4)) {
        fprintf(OUTPUT, "ERROR: failed to send quote size\n");
        goto cleanup;
    }
    // Send Quote
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Sending %u bytes of Quote...\n", quote_size);
#endif    
    
    if (!write_all(sockfd, (uint8_t *) p_quote, quote_size)) {
        fprintf(OUTPUT, "ERROR: failed to send %d bytes of quote\n", quote_size);
        goto cleanup;
    }

    // Send public_keys_size
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Sending the size of the enclave's public keys\n");
#endif
    if (!write_all(sockfd, (uint8_t *) &public_keys_size, 4)) {
        fprintf(OUTPUT, "ERROR: failed to send public keys size\n");
        goto cleanup;
    }
    // Send public_keys
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Sending %u bytes of enclave's public keys...\n", public_keys_size);
    fprintf(OUTPUT, "This is the public key that app sends to the server - %s\n", p_public_keys);
#endif  
    if (!write_all(sockfd, p_public_keys, public_keys_size)) {
        fprintf(OUTPUT, "ERROR: failed to send %d bytes of public_keys\n", public_keys_size);
        goto cleanup;
    }

    // Receive attestation result
#ifdef VERBOSE_PRINT
    fprintf(OUTPUT, "Receiving attestation result...\n");
#endif    
    if (read(sockfd, &attestation_result, 1) != 1) {
        fprintf(OUTPUT, "ERROR: failed to receive attestation result\n");
        goto cleanup;
    }

    if (attestation_result == 0) {
        fprintf(OUTPUT, "ERROR: bad attestation_result value\n");
        goto cleanup;
    }
    std::cout<<(success+"Succesfully sent attestation proof!\n");

    ret = true;

cleanup:

    if (seal_file != NULL) {
        fclose(seal_file);
    }
    if (sealed_keys != NULL) {
        free(sealed_keys);
    }
    if (p_public_keys != NULL) {
        free(p_public_keys);
    }
    if (p_sig_rl != NULL) {
        free(p_sig_rl);
    }
    if (p_quote != NULL) {
        free(p_quote);
    }

    return ret;

}


