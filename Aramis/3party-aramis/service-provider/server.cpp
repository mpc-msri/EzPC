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

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h> 
#include <limits.h>
#include <unistd.h>
#include <pthread.h>

#include <openssl/sha.h>
#include "/home/mayank/Downloads/sgxsdk/include/sgx_quote.h"

#include "defs.h"
#include "truce_record.h"
#include "IAS_web_service.h"
#include "aux_funcs.h"
#include "base64.h"

//#define SIMULATE_IAS 0
#define SIZEOF_QUOTE_WITHOUT_SIGNATURE 432


using namespace std;

pthread_t tid[2];

typedef struct {
    bool operator()(const truce_id_t &id1, const truce_id_t &id2) {
        return (memcmp((void *) &id1,(void *) &id2, sizeof(truce_id_t)) < 0);
    }
} truce_cmp_ids_t;

map<truce_id_t,truce_record_t,truce_cmp_ids_t> g_truce_map;

int attestation_service_port;
int report_service_port;


bool handle_attestation_service_connection(int connfd) {
    sgx_quote_nonce_t nonce = {{0}};// TODO: should be chosen at random
    uint8_t epid_group_id[4] = {0};
    uint8_t *p_sig_rl = NULL;
    uint32_t sig_rl_size = 0;
    uint32_t quote_size = 0;
    uint8_t *p_quote = NULL;
    uint32_t public_keys_size = 0;
    uint8_t *p_public_keys = NULL;
    truce_record_t t_rec;
    truce_id_t t_id = {{0}};
    uint8_t att_result = 0;
    sgx_spid_t spid = SPID;
#ifndef SIMULATE_IAS
    ias_web_service_t ias_web_service;
    if (!init_ias_web_service(ias_web_service,
            SP_CERT,
            IAS_URL)) {
        printf("ERROR: IAS web service initialization has failed\n");
        goto cleanup;
    }
    printf("SP_CERT value - %s\n", SP_CERT);
#endif
    fprintf(stdout, " Attestation Service: Sending SPID...\n");
    if (!write_all(connfd, (uint8_t *) &spid, sizeof(spid))) {
         fprintf(stdout, "ERROR: failed to send spid\n");
         goto cleanup;
    }

    fprintf(stdout, " Attestation Service: Sending nonce...\n");
    if (!write_all(connfd, (uint8_t *) &nonce, sizeof(nonce))) {
         fprintf(stdout, "ERROR: failed to send nonce\n");
         goto cleanup;
    }

    fprintf(stdout, " Attestation Service: Receiving epid_group_id...\n");
    if (!read_all(connfd, (uint8_t *) &epid_group_id, sizeof(epid_group_id))) {
         fprintf(stdout, "ERROR: failed to recv epid_group_id\n");
         goto cleanup;
    }
#ifdef SIMULATE_IAS
    fprintf(stdout, "******* Warning: Running in IAS simulation mode. Sending empty SigRL\n");
#else
    fprintf(stdout, " Attestation Service: Retrieving SigRL from IAS...\n");
    if (!get_sig_rl(ias_web_service,
                    epid_group_id,
                    p_sig_rl,
                    sig_rl_size)) {
            printf("Failed to get SigRL from IAS\n");
            goto cleanup;
    }
#endif

    //fprintf(stdout, " Attestation Service: Sending SigRL size (= %u)...\n", sig_rl_size);
    if (!write_all(connfd, (uint8_t *) &sig_rl_size, sizeof(sig_rl_size))) {
         fprintf(stdout, "ERROR: failed to send sig_rl_size\n");
         goto cleanup;
    }
    if (sig_rl_size != 0) {
        fprintf(stdout, " Attestation Service: Sending %u bytes of SigRL\n", sig_rl_size);
        if (!write_all(connfd, (uint8_t *) p_sig_rl, sig_rl_size)) {
             fprintf(stdout, "ERROR: failed to send p_sig_rl\n");
             goto cleanup;
        }
    }

    // receiving quote
    fprintf(stdout, " Attestation Service: Receiving Quote size...\n");
    if (!read_all(connfd, (uint8_t *) &quote_size, sizeof(quote_size))) {
         fprintf(stdout, "ERROR: failed to recv quote_size\n");
         goto cleanup;
    }
    p_quote = (uint8_t *) malloc(quote_size);
    if (NULL == p_quote) {
         fprintf(stdout, "ERROR: failed to allocate %u bytes for quote\n", quote_size);
         goto cleanup;
    }
    fprintf(stdout, " Attestation Service: Receiving %u bytes of Quote...\n", quote_size);
    if (!read_all(connfd, p_quote, quote_size)) {
         fprintf(stdout, "ERROR: failed to recv %u bytes for quote\n", quote_size);
         goto cleanup;
    }

    // receiving public_keys
    fprintf(stdout, " Attestation Service: Receiving the size of Enclave's Public Keys...\n");
    if (!read_all(connfd, (uint8_t *) &public_keys_size, sizeof(public_keys_size))) {
         fprintf(stdout, "ERROR: failed to recv public_keys_size\n");
         goto cleanup;
    }
    p_public_keys = (uint8_t *) malloc(public_keys_size);
    if (NULL == p_public_keys) {
         fprintf(stdout, "ERROR: failed to allocate %u bytes for public_keys\n", public_keys_size);
         goto cleanup;
    }
    fprintf(stdout, " Attestation Service: Receiving %u bytes of Enclave's Public Keys...\n", public_keys_size);
    if (!read_all(connfd, p_public_keys, public_keys_size)) {
         fprintf(stdout, "ERROR: failed to recv %u bytes for public_keys\n", public_keys_size);
         goto cleanup;
    }

    fprintf(stdout, " Attestation Service: Computing t_id = SHA256(public_keys)...\n");
    SHA256(p_public_keys,public_keys_size,(uint8_t *) &t_id);

    if (g_truce_map.find(t_id) != g_truce_map.end()) {
        fprintf(stdout, "WARNING: truce_id already exists!\n");
        goto cleanup;
    }


#ifdef SIMULATE_IAS
    printf("******* Warning: Running in IAS simulation mode. Generating fake report from IAS\n");
    t_rec.ias_report.report_body = "{\"simulated_report\":\"1\",\"isvEnclaveQuoteBody\":\"";
    t_rec.ias_report.report_body.append(base64_encode(p_quote, SIZEOF_QUOTE_WITHOUT_SIGNATURE));
    t_rec.ias_report.report_body.append("\"}");
    t_rec.ias_report.report_cert_chain_urlsafe_pem = "";
    t_rec.ias_report.report_signature_base64 = "";

    t_rec.public_keys_size = public_keys_size;
    t_rec.p_public_keys = (truce_public_keys_t *) p_public_keys;
    g_truce_map[t_id] = t_rec;
    p_public_keys = NULL;

    printf(" Attestation Service: New record has been successfully added to the records map!\n");
    //printf("t_id:\n");
    //print_buffer((uint8_t *) &t_id, sizeof(t_id));
    att_result = 1;
#else
    printf(" Attestation Service: Getting IAS report...\n");
    if (!get_ias_report(
                    ias_web_service,
                    (uint8_t *) p_quote,
                    quote_size,
                    NULL,
                    0,
                    NULL,
                    0,
                    t_rec.ias_report)) {
        printf(" Attestation Service: Failed to get Report from IAS\n");
    }
    else {
        //printf("\nIAS_report_body = %s\n", t_rec.ias_report.report_body.c_str());
        //printf("\nIAS_report_signature_base64 = %s\n", t_rec.ias_report.report_signature_base64.c_str());
        //printf("\nIAS_report_cert_chain = %s\n\n", t_rec.ias_report.report_cert_chain_urlsafe_pem.c_str());

        t_rec.public_keys_size = public_keys_size;
        t_rec.p_public_keys = (truce_public_keys_t *) p_public_keys;

        g_truce_map[t_id] = t_rec;
        p_public_keys = NULL;

        printf(" Attestation Service: New record has been successfully added to the records map!\n");
        //printf("t_id:\n");
        //print_buffer((uint8_t *) &t_id, sizeof(t_id));

        att_result = 1;
    }
#endif
    write(connfd, &att_result, 1);

cleanup:

    if (p_sig_rl != NULL) {
        free(p_sig_rl);
    }
    if (p_quote != NULL) {
        free(p_quote);
    }
    if (p_public_keys != NULL) {
        free(p_public_keys);
    }

    return (att_result == 1);
}


bool handle_certificate_service_connection(int connfd) {
    uint32_t public_keys_size = 0;
    uint8_t *p_public_keys = NULL;
    truce_id_t t_id = {{0}};
    uint8_t match_result = 0;
    truce_record_t t_rec;
    bool retval = false;
    uint32_t len = 0;

    fprintf(stdout, " Report Service: Receiving Truce ID...\n");
    if (!read_all(connfd, (uint8_t *) &t_id, sizeof(t_id))) {
        printf("ERROR: failed to read tid\n");
        goto cleanup;
    }

    //printf(" Report Service: Received t_id:\n");
    //print_buffer((uint8_t *) &t_id, sizeof(t_id));

    if (g_truce_map.find(t_id) == g_truce_map.end()) {
        fprintf(stdout, "Warning: Truce ID wasn't found in SP map\n");
        write(connfd, &match_result, 1);
        goto cleanup;
    }
    match_result = 1;
    fprintf(stdout, " Report Service: found Truce ID! Sending match_result...\n");
    write(connfd, &match_result, 1);

    t_rec = g_truce_map[t_id];


    // Sending IAS report body length
    fprintf(stdout, " Report Service: Sending IAS_report_body length (=%lu)...\n", strlen(t_rec.ias_report.report_body));
    len = htonl(strlen(t_rec.ias_report.report_body));
    if (!write_all(connfd, (uint8_t *) &len, sizeof(len))) {
        printf("ERROR: failed to send ias report body length");
        goto cleanup;
    }
    // Sending IAS report body
    fprintf(stdout, " Report Service: Sending %lu bytes of IAS_report_body...\n", strlen(t_rec.ias_report.report_body));
    if (!write_all(connfd,
            (uint8_t *) t_rec.ias_report.report_body,
            strlen(t_rec.ias_report.report_body))) {
        printf("ERROR: failed to send ias report body\n");
        goto cleanup;
    }

    // Sending IAS report signature length
    fprintf(stdout, " Report Service: Sending IAS_report_signature length (=%lu)...\n", strlen(t_rec.ias_report.report_signature_base64));
    len = htonl(strlen(t_rec.ias_report.report_signature_base64));
    if (!write_all(connfd,(uint8_t *) &len,    sizeof(len))) {
        printf("ERROR: failed to send ias report signature length\n");
        goto cleanup;
    }
    // Sending IAS report signature
    fprintf(stdout, " Report Service: Sending %lu bytes of IAS_report_signature...\n", strlen(t_rec.ias_report.report_signature_base64));
    if (!write_all(connfd,
            (uint8_t *) t_rec.ias_report.report_signature_base64,
            strlen(t_rec.ias_report.report_signature_base64))) {
        printf("ERROR: failed to send ias report signature\n");
        goto cleanup;
    }


    // Sending IAS report cert_chain length
    fprintf(stdout, " Report Service: Sending IAS_cert_chain length (=%lu)...\n", strlen(t_rec.ias_report.report_cert_chain_urlsafe_pem));
    len = htonl(strlen(t_rec.ias_report.report_cert_chain_urlsafe_pem));
    if (!write_all(connfd,(uint8_t *) &len,    sizeof(len))) {
        printf("ERROR: failed to send ias report cert_chain length\n");
        goto cleanup;
    }
    // Sending IAS report cert_chain
    fprintf(stdout, " Report Service: Sending %lu bytes of IAS_cert_chain...\n", strlen(t_rec.ias_report.report_cert_chain_urlsafe_pem));
    if (!write_all(connfd,
            (uint8_t *) t_rec.ias_report.report_cert_chain_urlsafe_pem,
            strlen(t_rec.ias_report.report_cert_chain_urlsafe_pem))) {
        printf("ERROR: failed to send ias report cert_chain\n");
        goto cleanup;
    }

    // Sending public_keys_size
    //fprintf(stdout, " Report Service: Sending Public Keys size (=%u)...\n", t_rec.public_keys_size);
    len = htonl(t_rec.public_keys_size);
    if (!write_all(connfd,(uint8_t *) &len,    sizeof(len))) {
        printf("ERROR: failed to send public_keys_size\n");
        goto cleanup;
    }
    // Sending public_keys
    fprintf(stdout, " Report Service: Sending %u bytes of Public Keys...\n", t_rec.public_keys_size);
    if (!write_all(connfd,
            (uint8_t *) t_rec.p_public_keys,
            t_rec.public_keys_size)) {
        printf("ERROR: failed to send public_keys\n");
        goto cleanup;
    }

    retval = true;

cleanup:

    return retval;

}


void* certificate_service(void *arg)
{
    int port = report_service_port;
    int listenfd = -1, connfd = -1;

    if (!inet_listen(listenfd, port)) {
        fprintf(stdout, "ERROR: Failed to listen on port %d\n", port);
        goto cleanup;
    }

    while (true)
    {
        fprintf(stdout, "Report Service: Waiting for incoming connections on port %d\n", port);
        if (!inet_accept(connfd, listenfd)) {
            fprintf(stdout, "ERROR: inet_accept has failed (port %d)\n", port);
            goto cleanup;
        }

        if (!handle_certificate_service_connection(connfd)) {
            fprintf(stdout, " Report Service: Certification for connection %d has failed\n", connfd);
            goto cleanup;
        }
        close(connfd);
        connfd = -1;
     }

cleanup:
    if (listenfd >= 0) {
        close(listenfd);
    }
    if (connfd >= 0) {
        close(connfd);
    }
    return NULL;
}

void* attestation_service(void *arg) 
{
    int port = attestation_service_port;
    int listenfd = 0, connfd = 0;

    if (!inet_listen(listenfd, port)) {
        fprintf(stdout, "ERROR: Failed to listen on port %d\n", port);
        return NULL;
    }


    while(true)
    {
        fprintf(stdout, "Attestation Service: Waiting for incoming TCP connections on port %d\n", port);

        if (!inet_accept(connfd, listenfd)) {
            fprintf(stdout, "ERROR: inet_accept has failed (port %d)\n", port);
            goto cleanup;
        }

        if (!handle_attestation_service_connection(connfd)) {
            fprintf(stdout, " Attestation Service: Attestation for connection %d has failed\n", connfd);
            goto cleanup;
        }

        close(connfd);
        connfd = -1;
     }

cleanup:
    if (listenfd >= 0) {
        close(listenfd);
    }
    if (connfd >= 0) {
        close(connfd);
    }
    return NULL;
}

int main(int argc, char* argv[])
{

#ifdef SIMULATE_IAS
    printf("******* Warning: Server Running in IAS Simulation Mode ******\n");
#endif

    attestation_service_port = SP_AS_PORT_DEFAULT;
    report_service_port = SP_RS_PORT_DEFAULT;

    // Read server config, if available: 
    FILE *config_file = fopen("server.config","r");
    if (NULL != config_file) {
        char as_port[10];
        char rs_port[10];

        int res = fscanf(config_file," AS_PORT=%s",as_port);
        if (1 == res) {
            attestation_service_port = stoi(as_port);
        }

        res = fscanf(config_file," RS_PORT=%s",rs_port);
        if (1 == res) {
            report_service_port = stoi(rs_port);
        }

        fclose (config_file);
    }

    int err = pthread_create(&(tid[0]), NULL, &attestation_service, NULL);
    if (err != 0)
        printf("\ncan't create AS thread :[%s]", strerror(err));
    else
        printf("Attestation Service Thread created successfully\n");

    err = pthread_create(&(tid[1]), NULL, &certificate_service, NULL);
    if (err != 0)
        printf("\ncan't create Report Service thread :[%s]", strerror(err));
    else
        printf("Report Service Thread created successfully\n");


    while (true) {sleep(1000);}
}


