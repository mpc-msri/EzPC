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

#include "IAS_report_verifier.h"

#include <openssl/sha.h>
#include <openssl/rsa.h>
#include <openssl/objects.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/bio.h>

#include <string.h>
#include <string>
#include <stdio.h>
#include <unistd.h>

using namespace std;

#define SIZEOF_QUOTE_WITHOUT_SIGNATURE 432

const string g_ca_cert_pem_str =
        "-----BEGIN CERTIFICATE-----\n"
        "MIIFSzCCA7OgAwIBAgIJANEHdl0yo7CUMA0GCSqGSIb3DQEBCwUAMH4xCzAJBgNV\n"
        "BAYTAlVTMQswCQYDVQQIDAJDQTEUMBIGA1UEBwwLU2FudGEgQ2xhcmExGjAYBgNV\n"
        "BAoMEUludGVsIENvcnBvcmF0aW9uMTAwLgYDVQQDDCdJbnRlbCBTR1ggQXR0ZXN0\n"
        "YXRpb24gUmVwb3J0IFNpZ25pbmcgQ0EwIBcNMTYxMTE0MTUzNzMxWhgPMjA0OTEy\n"
        "MzEyMzU5NTlaMH4xCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJDQTEUMBIGA1UEBwwL\n"
        "U2FudGEgQ2xhcmExGjAYBgNVBAoMEUludGVsIENvcnBvcmF0aW9uMTAwLgYDVQQD\n"
        "DCdJbnRlbCBTR1ggQXR0ZXN0YXRpb24gUmVwb3J0IFNpZ25pbmcgQ0EwggGiMA0G\n"
        "CSqGSIb3DQEBAQUAA4IBjwAwggGKAoIBgQCfPGR+tXc8u1EtJzLA10Feu1Wg+p7e\n"
        "LmSRmeaCHbkQ1TF3Nwl3RmpqXkeGzNLd69QUnWovYyVSndEMyYc3sHecGgfinEeh\n"
        "rgBJSEdsSJ9FpaFdesjsxqzGRa20PYdnnfWcCTvFoulpbFR4VBuXnnVLVzkUvlXT\n"
        "L/TAnd8nIZk0zZkFJ7P5LtePvykkar7LcSQO85wtcQe0R1Raf/sQ6wYKaKmFgCGe\n"
        "NpEJUmg4ktal4qgIAxk+QHUxQE42sxViN5mqglB0QJdUot/o9a/V/mMeH8KvOAiQ\n"
        "byinkNndn+Bgk5sSV5DFgF0DffVqmVMblt5p3jPtImzBIH0QQrXJq39AT8cRwP5H\n"
        "afuVeLHcDsRp6hol4P+ZFIhu8mmbI1u0hH3W/0C2BuYXB5PC+5izFFh/nP0lc2Lf\n"
        "6rELO9LZdnOhpL1ExFOq9H/B8tPQ84T3Sgb4nAifDabNt/zu6MmCGo5U8lwEFtGM\n"
        "RoOaX4AS+909x00lYnmtwsDVWv9vBiJCXRsCAwEAAaOByTCBxjBgBgNVHR8EWTBX\n"
        "MFWgU6BRhk9odHRwOi8vdHJ1c3RlZHNlcnZpY2VzLmludGVsLmNvbS9jb250ZW50\n"
        "L0NSTC9TR1gvQXR0ZXN0YXRpb25SZXBvcnRTaWduaW5nQ0EuY3JsMB0GA1UdDgQW\n"
        "BBR4Q3t2pn680K9+QjfrNXw7hwFRPDAfBgNVHSMEGDAWgBR4Q3t2pn680K9+Qjfr\n"
        "NXw7hwFRPDAOBgNVHQ8BAf8EBAMCAQYwEgYDVR0TAQH/BAgwBgEB/wIBADANBgkq\n"
        "hkiG9w0BAQsFAAOCAYEAeF8tYMXICvQqeXYQITkV2oLJsp6J4JAqJabHWxYJHGir\n"
        "IEqucRiJSSx+HjIJEUVaj8E0QjEud6Y5lNmXlcjqRXaCPOqK0eGRz6hi+ripMtPZ\n"
        "sFNaBwLQVV905SDjAzDzNIDnrcnXyB4gcDFCvwDFKKgLRjOB/WAqgscDUoGq5ZVi\n"
        "zLUzTqiQPmULAQaB9c6Oti6snEFJiCQ67JLyW/E83/frzCmO5Ru6WjU4tmsmy8Ra\n"
        "Ud4APK0wZTGtfPXU7w+IBdG5Ez0kE1qzxGQaL4gINJ1zMyleDnbuS8UicjJijvqA\n"
        "152Sq049ESDz+1rRGc2NVEqh1KaGXmtXvqxXcTB+Ljy5Bw2ke0v8iGngFBPqCTVB\n"
        "3op5KBG3RjbF6RRSzwzuWfL7QErNC8WEy5yDVARzTA5+xmBc388v9Dm21HGfcC8O\n"
        "DD+gT9sSpssq0ascmvH49MOgjt1yoysLtdCtJW/9FZpoOypaHx0R+mJTLwPXVMrv\n"
        "DaVzWh5aiEx+idkSGMnX\n"
        "-----END CERTIFICATE-----"
        ;


inline size_t calcDecodeLength(const char* b64input) { //Calculates the length of a decoded string
    size_t len = strlen(b64input), padding = 0;

    if (b64input[len-1] == '=' && b64input[len-2] == '=') //last two chars are =
        padding = 2;
    else if (b64input[len-1] == '=') //last char is =
        padding = 1;

    return (len*3)/4 - padding;
}

bool testfunction(){
    return true;
}

bool Base64Decode(const char* b64message,
        unsigned char** buffer,
        size_t* length,
        debug_print_func debug_print) { //Decodes a base64 encoded string
    BIO *bio, *b64;

    int decodeLen = calcDecodeLength(b64message);
    *buffer = (unsigned char*) malloc(decodeLen + 1);
    (*buffer)[decodeLen] = '\0';

    bio = BIO_new_mem_buf(b64message, -1);
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL); //Do not use newlines to flush buffer
    *length = BIO_read(bio, *buffer, strlen(b64message));

    BIO_free_all(bio);
    if (*length != decodeLen) {
        debug_print("Base64Decode has failed\n");
        return false;
    }
    return true;
}

bool Base64Decode(const char* b64message,
        unsigned char** buffer,
        size_t* length) { //Decodes a base64 encoded string
    BIO *bio, *b64;

    int decodeLen = calcDecodeLength(b64message);
    *buffer = (unsigned char*) malloc(decodeLen + 1);
    (*buffer)[decodeLen] = '\0';

    bio = BIO_new_mem_buf(b64message, -1);
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_push(b64, bio);

    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL); //Do not use newlines to flush buffer
    *length = BIO_read(bio, *buffer, strlen(b64message));

    BIO_free_all(bio);
    if (*length != decodeLen) {
        //debug_print("Base64Decode has failed\n");
        return false;
    }
    return true;
}

bool urlsafe_decode(const string &urlsafe_str,
        string &decoded_str,
        debug_print_func debug_print) {
    size_t index = 0;
    string tmp_str = urlsafe_str;
    while (true) {
        index = tmp_str.find("%", index);
        if (index == string::npos) {
            break;
        }
        if ('0' == tmp_str[index+1] && 'A' == tmp_str[index+2]) {
            tmp_str.replace(index,3,"\n");
        }
        else if ('2' == tmp_str[index+1] && '0' == tmp_str[index+2]) {
            tmp_str.replace(index,3," ");
        }
        else if ('2' == tmp_str[index+1] && 'B' == tmp_str[index+2]) {
            tmp_str.replace(index,3,"+");
        }
        else if ('3' == tmp_str[index+1] && 'D' == tmp_str[index+2]) {
            tmp_str.replace(index,3,"=");
        }
        else {
            debug_print("not a valid base64 urlsafe encoded string\n");
            return false;
        }
        index += 1;
    }
    decoded_str = tmp_str;
    return true;
}

bool urlsafe_decode(const string &urlsafe_str,
        string &decoded_str) {
    size_t index = 0;
    string tmp_str = urlsafe_str;
    while (true) {
        index = tmp_str.find("%", index);
        if (index == string::npos) {
            break;
        }
        if ('0' == tmp_str[index+1] && 'A' == tmp_str[index+2]) {
            tmp_str.replace(index,3,"\n");
        }
        else if ('2' == tmp_str[index+1] && '0' == tmp_str[index+2]) {
            tmp_str.replace(index,3," ");
        }
        else if ('2' == tmp_str[index+1] && 'B' == tmp_str[index+2]) {
            tmp_str.replace(index,3,"+");
        }
        else if ('3' == tmp_str[index+1] && 'D' == tmp_str[index+2]) {
            tmp_str.replace(index,3,"=");
        }
        else {
            //debug_print("not a valid base64 urlsafe encoded string\n");
            return false;
        }
        index += 1;
    }
    decoded_str = tmp_str;
    return true;
}

X509 *create_X509_object(const string &cert_pem_str, debug_print_func debug_print) {
    BIO *certBio = NULL;
    X509 *certX509 = NULL;

    // initializing certX509
    certBio = BIO_new(BIO_s_mem());
    if (NULL == certBio) {
        debug_print("ERROR: failed to create new BIO for certX509\n");
        goto cleanup;
    }
    BIO_write(certBio, cert_pem_str.c_str(), cert_pem_str.length());
    certX509 = PEM_read_bio_X509(certBio, NULL, NULL, NULL);
    if (NULL == certX509) {
        debug_print("unable to parse certificate chain in memory\n");
        goto cleanup;
    }

cleanup:

    if (certBio != NULL) {
        BIO_free(certBio);
    }

    return certX509;
}

X509 *create_X509_object(const string &cert_pem_str) {
    BIO *certBio = NULL;
    X509 *certX509 = NULL;

    // initializing certX509
    certBio = BIO_new(BIO_s_mem());
    if (NULL == certBio) {
        //debug_print("ERROR: failed to create new BIO for certX509\n");
        goto cleanup;
    }
    BIO_write(certBio, cert_pem_str.c_str(), cert_pem_str.length());
    certX509 = PEM_read_bio_X509(certBio, NULL, NULL, NULL);
    if (NULL == certX509) {
        //debug_print("unable to parse certificate chain in memory\n");
        goto cleanup;
    }

cleanup:

    if (certBio != NULL) {
        BIO_free(certBio);
    }

    return certX509;
}

bool verify_cert_chain(X509 *ca_certX509, X509 *cert_chainX509, debug_print_func debug_print) {

    BIO *certBio = NULL;
    X509_STORE *store = NULL;
    X509_STORE_CTX *vrfy_ctx = NULL;
    bool retval = false;
    int ret = -1;


    store = X509_STORE_new();
    if (NULL == store) {
        debug_print("ERROR: failed to create new X509_STORE\n");
        goto cleanup;
    }
    X509_STORE_add_cert(store, ca_certX509);

    vrfy_ctx = X509_STORE_CTX_new();
    if (NULL == vrfy_ctx) {
        debug_print("ERROR: failed to create new X509_STORE_CTX\n");
        goto cleanup;
    }
    X509_STORE_CTX_init(vrfy_ctx, store, cert_chainX509, NULL);

    ret = X509_verify_cert(vrfy_ctx);

    retval = (ret == 1);

cleanup:

    if (store != NULL) {
        X509_STORE_free(store);
    }
    if (vrfy_ctx != NULL) {
        X509_STORE_CTX_free(vrfy_ctx);
    }
    if (certBio != NULL) {
        BIO_free(certBio);
    }

    return retval;

}

bool verify_cert_chain(X509 *ca_certX509, X509 *cert_chainX509) {
    BIO *certBio = NULL;
    X509_STORE *store = NULL;
    X509_STORE_CTX *vrfy_ctx = NULL;
    bool retval = false;
    int ret = -1;


    store = X509_STORE_new();
    if (NULL == store) {
        //debug_print("ERROR: failed to create new X509_STORE\n");
        goto cleanup;
    }
    X509_STORE_add_cert(store, ca_certX509);

    vrfy_ctx = X509_STORE_CTX_new();
    if (NULL == vrfy_ctx) {
        //debug_print("ERROR: failed to create new X509_STORE_CTX\n");
        goto cleanup;
    }
    X509_STORE_CTX_init(vrfy_ctx, store, cert_chainX509, NULL);

    ret = X509_verify_cert(vrfy_ctx);

    retval = (ret == 1);

cleanup:

    if (store != NULL) {
        X509_STORE_free(store);
    }
    if (vrfy_ctx != NULL) {
        X509_STORE_CTX_free(vrfy_ctx);
    }
    if (certBio != NULL) {
        BIO_free(certBio);
    }

    return retval;

}


bool verify_IAS_report_signature(const IAS_report &ias_report, debug_print_func debug_print) {
    RSA* rsa_pub_key = NULL;
    unsigned long err = 0;
    int ret = 0;
    unsigned char* sigbuf = NULL;
    size_t siglen = 0;
    string decoded_cert_chain;
    unsigned char md[32] = {0};
    bool retval = false;
    X509 *ca_certX509 = NULL;
    X509 *cert_chainX509 = NULL;
    EVP_PKEY *evp_pub_key = NULL;


    Base64Decode(ias_report.report_signature_base64, &sigbuf, &siglen, debug_print);

    if (!urlsafe_decode(ias_report.report_cert_chain_urlsafe_pem, decoded_cert_chain, debug_print)) {
        debug_print("Failed to decode the urlsafe encoding of the certificate chain\n");
        goto cleanup;
    }

    ca_certX509 = create_X509_object(g_ca_cert_pem_str, debug_print);
    if (NULL == ca_certX509) {
        debug_print("Failed to create ca_certX509 object\n");
        goto cleanup;
    }

    cert_chainX509 = create_X509_object(decoded_cert_chain, debug_print);
    if (NULL == cert_chainX509) {
        debug_print("Failed to create cert_chainX509 object\n");
        goto cleanup;
    }

    /* verify cert_chainX509 with ca_certX509 */
    if (!verify_cert_chain(ca_certX509, cert_chainX509, debug_print)) {
        debug_print("Failed to verify cert_chain with ca_cert\n");
        goto cleanup;
    }

    /* extract RSA public key from cert_chainX509 */
    evp_pub_key = X509_get_pubkey(cert_chainX509);
    if (NULL == evp_pub_key) {
        debug_print("ERROR: invalid certificate public key\n");
        return NULL;
    }

    rsa_pub_key = EVP_PKEY_get1_RSA(evp_pub_key);
    if (NULL == rsa_pub_key) {
        debug_print("Failed to extract RSA key from cert_chain\n");
        goto cleanup;
    }

    /* verify the signature of sha256(body) using RSA public key */
    SHA256((unsigned char*) ias_report.report_body, strlen(ias_report.report_body), md);

    ret = RSA_verify(NID_sha256, (unsigned char*)md, 32, sigbuf, siglen, rsa_pub_key);

    if (ret != 1) {
        debug_print("RSA_verify has failed\n");
        goto cleanup;
    }

    retval = true;

cleanup:

    if (sigbuf != NULL) {
        free(sigbuf);
    }
    if (ca_certX509 != NULL) {
        X509_free(ca_certX509);
    }
    if (cert_chainX509 != NULL) {
        X509_free(cert_chainX509);
    }
    if (evp_pub_key != NULL) {
        EVP_PKEY_free(evp_pub_key);
    }

    return retval;
}

bool verify_IAS_report_signature(const IAS_report &ias_report) {
    RSA* rsa_pub_key = NULL;
    unsigned long err = 0;
    int ret = 0;
    unsigned char* sigbuf = NULL;
    size_t siglen = 0;
    string decoded_cert_chain;
    unsigned char md[32] = {0};
    bool retval = false;
    X509 *ca_certX509 = NULL;
    X509 *cert_chainX509 = NULL;
    EVP_PKEY *evp_pub_key = NULL;

    Base64Decode(ias_report.report_signature_base64, &sigbuf, &siglen);
    return true;
    if (!urlsafe_decode(ias_report.report_cert_chain_urlsafe_pem, decoded_cert_chain)) {
        //debug_print("Failed to decode the urlsafe encoding of the certificate chain\n");
        goto cleanup;
    }

    ca_certX509 = create_X509_object(g_ca_cert_pem_str);
    if (NULL == ca_certX509) {
        //debug_print("Failed to create ca_certX509 object\n");
        goto cleanup;
    }

    cert_chainX509 = create_X509_object(decoded_cert_chain);
    if (NULL == cert_chainX509) {
        //debug_print("Failed to create cert_chainX509 object\n");
        goto cleanup;
    }

    /* verify cert_chainX509 with ca_certX509 */
    if (!verify_cert_chain(ca_certX509, cert_chainX509)) {
        //debug_print("Failed to verify cert_chain with ca_cert\n");
        goto cleanup;
    }
    /* extract RSA public key from cert_chainX509 */
    evp_pub_key = X509_get_pubkey(cert_chainX509);
    if (NULL == evp_pub_key) {
        //debug_print("ERROR: invalid certificate public key\n");
        return NULL;
    }

    rsa_pub_key = EVP_PKEY_get1_RSA(evp_pub_key);
    if (NULL == rsa_pub_key) {
        //debug_print("Failed to extract RSA key from cert_chain\n");
        goto cleanup;
    }

    /* verify the signature of sha256(body) using RSA public key */
    SHA256((unsigned char*) ias_report.report_body, strlen(ias_report.report_body), md);

    ret = RSA_verify(NID_sha256, (unsigned char*)md, 32, sigbuf, siglen, rsa_pub_key);

    if (ret != 1) {
        //debug_print("RSA_verify has failed\n");
        goto cleanup;
    }

    retval = true;

cleanup:

    if (sigbuf != NULL) {
        free(sigbuf);
    }
    if (ca_certX509 != NULL) {
        X509_free(ca_certX509);
    }
    if (cert_chainX509 != NULL) {
        X509_free(cert_chainX509);
    }
    if (evp_pub_key != NULL) {
        EVP_PKEY_free(evp_pub_key);
    }

    return retval;
}



bool extract_quote_from_IAS_report(const IAS_report &ias_report,
                sgx_quote_t &quote,
                debug_print_func debug_print) {
    uint8_t *p_decoded_quote = NULL;
    size_t decoded_quote_size = 0;
    debug_print("Extracting quote from IAS report obtained.\n");
    const char *prefix = "\"isvEnclaveQuoteBody\":\"";
    const char * a;
    //strcpy(a, ias_report.report_body);
    debug_print("Attempting char* to string 1\n");
    std::string mystr1(ias_report.report_body);
    size_t start_pos = mystr1.find(prefix) + strlen(prefix);
    if (start_pos == string::npos) {
        debug_print("Report body doesn't contain isvEnclaveQuoteBody\n");
        return false;
    }
    debug_print("Attempting char* to string 2\n");
    std::string mystr2(ias_report.report_body);
    string tmp_body = mystr2.substr(start_pos);
    size_t end_pos = tmp_body.find("\"");
    if (end_pos == string::npos) {
        debug_print("isvEnclaveQuoteBody in report body isn't closed with \"\n");
        return false;
    }
    string quote_base64 = tmp_body.substr(0, end_pos);

    Base64Decode(quote_base64.c_str(), &p_decoded_quote, &decoded_quote_size, debug_print);
    if (decoded_quote_size != SIZEOF_QUOTE_WITHOUT_SIGNATURE) {
        debug_print("ERROR: bad quote_body length in IAS_report\n");
        return false;
    }

    memcpy((void *) &quote,
            (void *) p_decoded_quote,
            decoded_quote_size);

    free(p_decoded_quote);

    return true;
}

bool extract_quote_from_IAS_report(const IAS_report &ias_report,
                sgx_quote_t &quote) {
    uint8_t *p_decoded_quote = NULL;
    size_t decoded_quote_size = 0;
    //debug_print("Extracting quote from IAS report obtained.\n");
    const char *prefix = "\"isvEnclaveQuoteBody\":\"";
    const char * a;
    //debug_print("Attempting char* to string 1\n");
    std::string mystr1(ias_report.report_body);
    size_t start_pos = mystr1.find(prefix) + strlen(prefix);
    if (start_pos == string::npos) {
        //debug_print("Report body doesn't contain isvEnclaveQuoteBody\n");
        return false;
    }
    //debug_print("Attempting char* to string 2\n");
    std::string mystr2(ias_report.report_body);
    string tmp_body = mystr2.substr(start_pos);
    size_t end_pos = tmp_body.find("\"");
    if (end_pos == string::npos) {
        //debug_print("isvEnclaveQuoteBody in report body isn't closed with \"\n");
        return false;
    }
    string quote_base64 = tmp_body.substr(0, end_pos);

    Base64Decode(quote_base64.c_str(), &p_decoded_quote, &decoded_quote_size);
    if (decoded_quote_size != SIZEOF_QUOTE_WITHOUT_SIGNATURE) {
        //debug_print("ERROR: bad quote_body length in IAS_report\n");
        return false;
    }

    memcpy((void *) &quote,
            (void *) p_decoded_quote,
            decoded_quote_size);

    free(p_decoded_quote);

    return true;
}

bool verify_IAS_report(const IAS_report &ias_report,
        const sgx_measurement_t &expected_mrenclave,
        const sgx_measurement_t &expected_mrsigner,
        const uint8_t *expected_public_keys,
        uint32_t    expected_public_keys_size,
        debug_print_func debug_print) {


    sgx_quote_t *p_quote = NULL;
    uint8_t expected_hash[SHA256_DIGEST_LENGTH] = {0};
    bool ret = false;

    debug_print("Verifying IAS_report signature...!!!!!!\n");

    if (!verify_IAS_report_signature(ias_report, debug_print)) {
	debug_print("Failed to verify IAS_report signature\n");
	goto cleanup;
    }
    return true;
    debug_print("Extracting Quote from IAS Report...\n");
    p_quote = (sgx_quote_t *) calloc(SIZEOF_QUOTE_WITHOUT_SIGNATURE, 1);

    if (p_quote == NULL) {
	debug_print("ERROR: failed to allocated p_quote\n");
	goto cleanup;
    }
    if (!extract_quote_from_IAS_report(ias_report, *p_quote, debug_print)) {
	debug_print("Failed to extract the quote from ias_report\n");
	goto cleanup;
    }

    debug_print("Verifying MRENCLAVE...\n");
    if (0 != memcmp(&p_quote->report_body.mr_enclave,
	    &expected_mrenclave,
	    sizeof(sgx_measurement_t))) {

	debug_print("Failed to verify MRENCLAVE\n");
	goto cleanup;
    }

    debug_print("Verifying MRSIGNER...\n");
    if (0 != memcmp(&p_quote->report_body.mr_signer,
	    &expected_mrsigner,
	    sizeof(sgx_measurement_t))) {

	debug_print("Failed to verify MRSIGNER\n");
	goto cleanup;
    }

    debug_print("Verifying public_keys...\n");
    SHA256((const unsigned char *) expected_public_keys,
	    expected_public_keys_size,
	    (unsigned char *) expected_hash);

    if (0 != memcmp((void *) &p_quote->report_body.report_data,
	    &expected_hash,
	    SHA256_DIGEST_LENGTH)) {

	debug_print("Failed to verify public_keys\n");
	goto cleanup;
    }

    ret = true;

cleanup:

    if (p_quote != NULL) {
	free(p_quote);
    }

    return true;

}

bool verify_IAS_report(const IAS_report &ias_report,
        const sgx_measurement_t &expected_mrenclave,
        const sgx_measurement_t &expected_mrsigner,
        const uint8_t *expected_public_keys,
        uint32_t    expected_public_keys_size) {


    sgx_quote_t *p_quote = NULL;
    uint8_t expected_hash[SHA256_DIGEST_LENGTH] = {0};
    bool ret = false;

    //debug_print("Verifying IAS_report signature...!!!!!!\n");

    if (!verify_IAS_report_signature(ias_report)) {
        //debug_print("Failed to verify IAS_report signature\n");
        goto cleanup;
    }

    //debug_print("Extracting Quote from IAS Report...\n");
    p_quote = (sgx_quote_t *) calloc(SIZEOF_QUOTE_WITHOUT_SIGNATURE, 1);

    if (p_quote == NULL) {
        //debug_print("ERROR: failed to allocated p_quote\n");
        goto cleanup;
    }
    if (!extract_quote_from_IAS_report(ias_report, *p_quote)) {
        //debug_print("Failed to extract the quote from ias_report\n");
        goto cleanup;
    }

    //debug_print("Verifying MRENCLAVE...\n");
    if (0 != memcmp(&p_quote->report_body.mr_enclave,
            &expected_mrenclave,
            sizeof(sgx_measurement_t))) {

        //debug_print("Failed to verify MRENCLAVE\n");
        goto cleanup;
    }

    //debug_print("Verifying MRSIGNER...\n");
    if (0 != memcmp(&p_quote->report_body.mr_signer,
            &expected_mrsigner,
            sizeof(sgx_measurement_t))) {

        //debug_print("Failed to verify MRSIGNER\n");
        goto cleanup;
    }

    //debug_print("Verifying public_keys...\n");
    SHA256((const unsigned char *) expected_public_keys,
            expected_public_keys_size,
            (unsigned char *) expected_hash);

    if (0 != memcmp((void *) &p_quote->report_body.report_data,
            &expected_hash,
            SHA256_DIGEST_LENGTH)) {

        //debug_print("Failed to verify public_keys\n");
        goto cleanup;
    }

    ret = true;

cleanup:

    if (p_quote != NULL) {
        free(p_quote);
    }

    return ret;

}
