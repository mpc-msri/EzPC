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

#include "truce_enclave.h"
#include "utils_ported_sgx.h"
#include "sigcounts.h"

#define  ENCLAVE_DEBUG_PRINT_PREFIX        "[SGX_ENCLAVE] "
#define  ECDSA_SIG_SIZE 64

// TODO: Remove this hardcoded key and generate it using random gen.

static sgx_aes_gcm_128bit_key_t aeskeyab = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xe };

static sgx_aes_gcm_128bit_key_t aeskeyac = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xe };

static sgx_aes_gcm_128bit_key_t aeskeybc = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xe };

static sgx_aes_gcm_128bit_key_t aeskey = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xe };

int partynumberself = 0;
int aeskeyabflag = 0;
int aeskeyacflag = 0;
int aeskeybcflag = 0;

int whichaeskey = 0;
static sgx_aes_gcm_128bit_key_t aeskey_other = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xc, 0xe, 0xe };
static sgx_aes_gcm_128bit_key_t aeskey_otherab = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xc, 0xe, 0xe };
static sgx_aes_gcm_128bit_key_t aeskey_otherac = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xc, 0xe, 0xe };
static sgx_aes_gcm_128bit_key_t aeskey_otherbc = { 0x0, 0x5, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x7, 0x9, 0xa, 0xb, 0xc, 0xc, 0xe, 0xe };

static truce_public_keys_t     *g_public_keys     =     NULL;
static uint32_t                g_public_keys_size = 0;
static truce_private_keys_t *g_private_keys =     NULL;
static uint32_t                g_private_keys_size = 0;
static bool                 g_after_generate_keys    =    false;

static RSA *g_rsa = NULL;

static truce_public_keys_t *other_party_keyab = NULL;
static truce_public_keys_t *other_party_keyac = NULL;
static truce_public_keys_t *other_party_keybc = NULL;

static uint8_t* complement_party_publickeys = NULL;
static uint32_t complement_party_publickeyssize = 0;

static truce_private_keys_t *own_private_key = NULL;

static uint8_t *server1_publickeys = NULL;
static uint32_t server1_publickeyssize = 0;
static uint8_t *server2_publickeys = NULL;
static uint32_t server2_publickeyssize = 0;
static uint8_t *server1_rsapublickeys = NULL;
static uint32_t server1_rsapublickeyssize = 0;
static uint8_t *server2_rsapublickeys = NULL;
static uint32_t server2_rsapublickeyssize = 0;
static uint32_t ec256_pubkey_size = 64;
int RSA_cipher_size = 0;

int xshare, yshare, eshare, fshare, cshare, ashare, bshare;

void enclave_debug_print(const char *str) {
	ocall_print_string(ENCLAVE_DEBUG_PRINT_PREFIX);
	ocall_print_string(str);
}


truce_secret_t *Secret_head = NULL;
truce_secret_t *Secret_tail = NULL;
RSA *pubkey_rsa = NULL;
RSA *privkey_rsa = NULL;
uint8_t *privkey_rsa_grand = NULL;


void print_buffer(uint8_t* buf, int len)
{
	char out[20];
	snprintf(out, 20, "Length: %d \n", len);
	ocall_print_string(out);
	for (int i=0; i < len; i++) {
		snprintf(out, 20, "0x%c ", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}

void pretty_print_buffer(uint8_t* buf, int len)
{
	char out[20];
	snprintf(out, 20, "Length: %d \n", len);
	ocall_print_string(out);
	for (int i=0; i < len; i++) {
		snprintf(out, 20, "%d ", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}

void pretty_print_buffer(const uint8_t* buf, int len)
{
	char out[20];
	snprintf(out, 20, "Length: %d \n", len);
	ocall_print_string(out);
	for (int i=0; i < len; i++) {
		snprintf(out, 20, "%c ", buf[i]);
		ocall_print_string(out);
	}
	ocall_print_string("\n");
}

void enclave_print_integer(int len)
{
	char out[20];
	snprintf(out, 20, "Integer: %u \n", len);
	ocall_print_string(out);
}

void print_sgx_error(sgx_status_t status){
	if(status == SGX_ERROR_INVALID_PARAMETER){
		ocall_print_string("ERROR is: SGX_ERROR_INVALID_PARAMETER\n");
		return;
	}
	if(status == SGX_ERROR_OUT_OF_MEMORY){
		ocall_print_string("ERROR is: SGX_ERROR_OUT_OF_MEMORY\n");
		return;
	}
	else if(status == SGX_ERROR_UNEXPECTED){
		ocall_print_string("ERROR is: SGX_ERROR_UNEXPECTED\n");
		return;
	}
	else{
		ocall_print_string("ERROR is: UNKNOWN UNEXPECTED ERROR\n");
		return;
	}

}


sgx_status_t ECALL_generate_keys() {

	sgx_status_t ret = SGX_SUCCESS;
	sgx_ecc_state_handle_t ecc_state = NULL;
	sgx_ecc_state_handle_t ecc_state2 = NULL;
	sgx_ecc_state_handle_t ecc_state_verifier_side = NULL;
	int retval = 0;
	BIGNUM    *bne = NULL;
	EVP_PKEY *evp_pkey = NULL;
	int rsa_pub_key_size = 0;
	int rsa_priv_key_size = 0;
	uint8_t *tmp_buf = NULL;
	unsigned long errorcode;
	uint8_t *output;
	output = (uint8_t *) calloc(1, 50);
	const char *pdataraw = "Use encryption result here.";
	const uint8_t *pdata = (const uint8_t*)output;
	uint32_t data_size = strlen(pdataraw);
	sgx_ec256_signature_t *p_signature;
	sgx_ec256_signature_t *p_signature_communicated;
	uint8_t *result_of_signature = NULL;

	uint8_t *pubkey_rsa_tmp = NULL;
	uint32_t rsa_pubkey_len = 0;
	uint32_t rsa_privkey_len = 0;
	uint8_t* privkey_rsa_tmp = NULL;

	if (g_after_generate_keys) {
		enclave_debug_print("ERROR: Keys have been already generated\n");
		return SGX_ERROR_INVALID_STATE;
	}

	/////////////////////////////   Creating RSA 4096 key pair with openssl.
	bne = BN_new();
	if (bne == NULL) {
		enclave_debug_print("ERROR: BN_new has failed\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}
	retval = BN_set_word(bne,RSA_F4);
	if (retval != 1) {
		enclave_debug_print("ERROR: BN_set_word has failed\n");
		ret = SGX_ERROR_UNEXPECTED;
		goto cleanup;
	}
	g_rsa = RSA_new();
	if (NULL == g_rsa) {
		enclave_debug_print("ERROR: RSA_new has failed\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}
	retval = RSA_generate_key_ex(g_rsa, 4096, bne, NULL);

	BN_free(bne);

	if (retval != 1) {
		enclave_debug_print("ERROR: RSA_generate_key_ex has failed\n");
		ret = SGX_ERROR_UNEXPECTED;
		goto cleanup;
	}
	evp_pkey = EVP_PKEY_new();
	if (evp_pkey == NULL) {
		enclave_debug_print("ERROR: EVP_PKEY_new has failed\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}
	EVP_PKEY_assign_RSA(evp_pkey, g_rsa);

	// extract RSA 4096 public key
	rsa_pub_key_size = i2d_PublicKey(evp_pkey, NULL);
	rsa_priv_key_size = i2d_PrivateKey(evp_pkey, NULL);

	g_public_keys_size = sizeof(truce_public_keys_t) + rsa_pub_key_size;
	g_public_keys = (truce_public_keys_t *) calloc(1, g_public_keys_size);
	g_private_keys_size = sizeof(truce_private_keys_t) + rsa_priv_key_size;
	g_private_keys = (truce_private_keys_t *) calloc(1, g_private_keys_size);

	g_public_keys->rsa4096_public_key_size = rsa_pub_key_size;
	tmp_buf = (uint8_t *) &g_public_keys->rsa4096_public_key;

	i2d_PublicKey(evp_pkey, &tmp_buf);

	if (g_private_keys == NULL) {
		enclave_debug_print("ERROR: calloc for g_private_keys has failed\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}
	g_private_keys->rsa4096_private_key_size = rsa_priv_key_size;
	tmp_buf = (uint8_t *) &g_private_keys->rsa4096_private_key;
	i2d_PrivateKey(evp_pkey, &tmp_buf);

	rsa_privkey_len = g_private_keys->rsa4096_private_key_size;
	privkey_rsa_grand = (uint8_t *) calloc(1, rsa_privkey_len);
	memcpy(privkey_rsa_grand, g_private_keys->rsa4096_private_key, rsa_privkey_len);
	privkey_rsa_tmp = (uint8_t* )calloc(1, rsa_privkey_len);
	memcpy(privkey_rsa_tmp, privkey_rsa_grand, rsa_privkey_len);
	privkey_rsa = d2i_RSAPrivateKey(0, (const unsigned char**) &privkey_rsa_tmp, g_private_keys->rsa4096_private_key_size);
	rsa_pubkey_len = g_public_keys->rsa4096_public_key_size;
	pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
	memcpy(pubkey_rsa_tmp, g_public_keys->rsa4096_public_key, rsa_pubkey_len);

	pubkey_rsa = d2i_RSAPublicKey(0, (const unsigned char**)&pubkey_rsa_tmp, rsa_pubkey_len);
	RSA_cipher_size = RSA_size(pubkey_rsa);

	///////////////////////////////////  Creating ec256 key pair with sgxsdk
	ret = sgx_ecc256_open_context(&ecc_state);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_open_context has failed.\n");
		goto cleanup;
	}

	ret = sgx_ecc256_create_key_pair(&g_private_keys->ec256_private_key, // a
			&g_public_keys->ec256_public_key, // g^a
			ecc_state);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_create_key_pair has failed.\n");
		goto cleanup;
	}
	ret = sgx_ecc256_close_context(ecc_state);

#ifdef ENCLAVE_VERBOSE
	pretty_print_buffer((uint8_t*)&g_public_keys->ec256_public_key, sizeof(sgx_ec256_public_t));
#endif

	// Testing the usage of sgx_ecdsa_sign and sgx_ecdsa_verify functions here.
	p_signature = (sgx_ec256_signature_t *)calloc(1, sizeof(sgx_ec256_signature_t));
	ret = sgx_ecc256_open_context(&ecc_state2);
	ret = sgx_ecdsa_sign(pdata, data_size, &g_private_keys->ec256_private_key, p_signature, ecc_state2);
	if(SGX_SUCCESS != ret){
		ocall_print_string("\n\nERROR: Signing with ECDSA has failed\n\n");
		goto cleanup;
	}
#ifdef ENCLAVE_VERBOSE
	//ocall_print_string("Successfully signed the data using ECDSA\n");
#endif
	ret = sgx_ecc256_close_context(ecc_state2);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_close_context has failed.\n");
		goto cleanup;
	}
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_open_context has failed.\n");
		goto cleanup;
	}

	// Trying to convert the signature produced to unit8_t* to be able to send it over socket.
	uint8_t* signature_communicated;
	signature_communicated = (uint8_t*) calloc(1, sizeof(sgx_ec256_signature_t));
	memcpy(signature_communicated, (uint8_t*)p_signature, sizeof(sgx_ec256_signature_t));
	p_signature_communicated = (sgx_ec256_signature_t*)signature_communicated;


#ifdef ENCLAVE_VERBOSE
	ocall_print_string("Size of the generated signature is: ");
	enclave_print_integer(sizeof(p_signature));
	if(p_signature == NULL){
		ocall_print_string("ERROR: Signature pointer is NULL\n");
	}
	if(pdata == NULL){
		ocall_print_string("ERROR: Data pointer is NULL\n");
	}
#endif

	result_of_signature = (uint8_t*)calloc(1, sizeof(SGX_EC_VALID));
	// You probably need to generate the context again everytime signing is done. Verification won't work without regenerating the context.
	ret = sgx_ecc256_open_context(&ecc_state_verifier_side);
	// Now test verifying the signature that was just generated.
	ret = sgx_ecdsa_verify(pdata, data_size, &g_public_keys->ec256_public_key, p_signature_communicated, result_of_signature, ecc_state_verifier_side);
	if(SGX_SUCCESS != ret){
		ocall_print_string("\n\nERROR: Verification of the signature with ECDSA has failed\n\n");
		print_sgx_error(ret);
		goto cleanup;
	}
	if(result_of_signature == NULL){
		ocall_print_string("ERROR: Result of signature verification pointer is NULL\n");
	}
	//ocall_print_string("The result of the verification process of ECDSA signature is: \n");
	//pretty_print_buffer(result_of_signature, 10);
	if(SGX_EC_VALID == *result_of_signature){
#ifdef ENCLAVE_VERBOSE
		ocall_print_string("SUCCESS\n");
#endif
	}
	ret = sgx_ecc256_close_context(ecc_state_verifier_side);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_close_context has failed.\n");
		goto cleanup;
	}
	g_after_generate_keys = true;
#ifdef ENCLAVE_VERBOSE
	ocall_print_string("Size of a ECDSA signature: ");
	enclave_print_integer(sizeof(sgx_ec256_signature_t));
	ocall_print_string("The ecdsa public key size is: ");
	enclave_print_integer(sizeof(g_public_keys->ec256_public_key));
#endif

	uint8_t* keypub;
	keypub = (uint8_t*)calloc(1, 64);
	memcpy(keypub, &g_public_keys->ec256_public_key, 64);
#ifdef ENCLAVE_VERBOSE
	pretty_print_buffer(keypub, 64);
#endif
cleanup:

	if (bne != NULL) {
		BN_free(bne);
	}
	if (evp_pkey != NULL) {
		/* TBD: EVP_PKEY_free causes segfault.
		   Probably because it tries to free g_rsa again.
		   Verify that this indeed the case and we don't need to free evp_pkey.
		   Maybe we need to just call free in this case, and not EVP_PKEY_free. */
		/*EVP_PKEY_free(evp_pkey);*/
	}
	if (!g_after_generate_keys) {
		//if (g_rsa != NULL) {
		//RSA_free(g_rsa);
		//g_rsa = NULL;
		//}
		if (g_public_keys != NULL) {
			free(g_public_keys);
			g_public_keys = NULL;
		}
		if (g_private_keys != NULL) {
			free(g_private_keys);
			g_private_keys = NULL;
		}
	}
	return ret;
}


sgx_status_t ECALL_get_public_keys_size(uint32_t *pub_keys_size) {
	if (!g_after_generate_keys) {
		enclave_debug_print("Warning: ECALL_get_public_keys_size has failed since there is no public key.\n");
		return SGX_ERROR_INVALID_STATE;
	}

	*pub_keys_size = g_public_keys_size;
	return SGX_SUCCESS;
}

sgx_status_t ECALL_get_public_keys(uint8_t *p_public_keys,
		uint32_t pub_keys_size) {
	if (!g_after_generate_keys) {
		enclave_debug_print("Warning: ECALL_get_public_keys has failed since there is no public key.\n");
		return SGX_ERROR_INVALID_STATE;
	}
	if (pub_keys_size < g_public_keys_size) {
		enclave_debug_print("Warning: ECALL_get_public_keys has failed since pub_keys_size is too small.\n");
		return SGX_ERROR_INVALID_PARAMETER;
	}
	memcpy(p_public_keys, (void *) g_public_keys, g_public_keys_size);
	return SGX_SUCCESS;
}


sgx_status_t ECALL_create_enclave_report(
		const sgx_target_info_t *p_target_info, //in
		sgx_report_t *p_report) { //out

	sgx_status_t ret = SGX_SUCCESS;
	sgx_report_data_t report_data = {{0}};

	if (!g_after_generate_keys) {
		enclave_debug_print("Warning: ECALL_create_enclave_report has failed since there is no public keys.\n");
		return SGX_ERROR_INVALID_STATE;
	}
	if (sizeof(sgx_report_data_t) < sizeof(sgx_sha256_hash_t)) {
		return SGX_ERROR_UNEXPECTED;
	}

	ret = sgx_sha256_msg((uint8_t *) g_public_keys,
			g_public_keys_size,
			(sgx_sha256_hash_t *) &report_data);

	if (ret != SGX_SUCCESS) {
		enclave_debug_print("ERROR: sgx_sha256_msg has failed.\n");
		return ret;
	}

	ret = sgx_create_report(p_target_info, &report_data, p_report);
	if (ret != SGX_SUCCESS) {
		enclave_debug_print("ERROR: sgx_create_report has failed.\n");
		return ret;
	}

	return ret;
}


bool get_seals_sizes(uint32_t &first_seal_size, uint32_t &second_seal_size) {

	if (!g_after_generate_keys) {
		enclave_debug_print("Warning: ECALL_get_sealed_keys_size has failed since there is no public keys.\n");
		return false;
	}
	// First Seal using MRSIGNER. The data is: (pub_key_size, pub_key, priv_key_size, priv_key)
	first_seal_size = sgx_calc_sealed_data_size(0,
			sizeof(uint32_t) + g_public_keys_size + sizeof(uint32_t) + g_private_keys_size);
	if (UINT32_MAX == first_seal_size) {
		enclave_debug_print("ERROR: first sgx_calc_sealed_data_size has failed.\n");
		return false;
	}
	second_seal_size = sgx_calc_sealed_data_size(0, first_seal_size); // Second seal using MRENCLAVE
	if (UINT32_MAX == second_seal_size) {
		enclave_debug_print("ERROR: first sgx_calc_sealed_data_size has failed.\n");
		return false;
	}
	return true;
}


sgx_status_t ECALL_get_sealed_keys_size(uint32_t *sealed_keys_size) {
	uint32_t first_seal_size = 0;
	uint32_t second_seal_size = 0;

	if (!get_seals_sizes(first_seal_size, second_seal_size)) {
		enclave_debug_print("ERROR: get_seals_sizes has failed.\n");
		return SGX_ERROR_INVALID_STATE;
	}

	*sealed_keys_size = first_seal_size;
	return SGX_SUCCESS;
}

sgx_status_t ECALL_seal_keys(uint8_t *sealed_keys,
		uint32_t sealed_keys_size) {

	sgx_status_t ret = SGX_SUCCESS;
	uint32_t first_seal_size = 0;
	uint32_t second_seal_size = 0;
	uint8_t *data_to_seal = NULL;
	uint32_t data_to_seal_size = 0;
	uint8_t *tmp_ptr = NULL;
	uint8_t *first_seal = NULL;
	sgx_attributes_t attribute_mask;
	sgx_misc_select_t misc_mask = 0xF0000000;

	if (!g_after_generate_keys) {
		//print_string_cout("Warning: ECALL_get_sealed_keys has failed since there is no public keys.\n");
		return SGX_ERROR_INVALID_STATE;
	}

	data_to_seal_size = sizeof(uint32_t) + g_public_keys_size +
		sizeof(uint32_t) + g_private_keys_size;
	data_to_seal = (uint8_t *) malloc(data_to_seal_size);

	if (NULL == data_to_seal) {
		//print_string_cout("ERROR: failed to allocate data_to_seal.\n");
		goto cleanup;
	}
	tmp_ptr = data_to_seal;

	memcpy(tmp_ptr, (void *) &g_public_keys_size, sizeof(uint32_t));
	tmp_ptr += sizeof(uint32_t);
	memcpy(tmp_ptr, (void *) g_public_keys, g_public_keys_size);
	tmp_ptr += g_public_keys_size;
	memcpy(tmp_ptr, (void *) &g_private_keys_size, sizeof(uint32_t));
	tmp_ptr += sizeof(uint32_t);
	memcpy(tmp_ptr, (void *) g_private_keys, g_private_keys_size);

	first_seal = (uint8_t *) malloc(first_seal_size);
	if (NULL == first_seal) {
		//print_string_cout("ERROR: failed to allocate first_seal.\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}

	// First, seal with key derived from MR_SIGNER
	ret = sgx_seal_data(0,NULL,data_to_seal_size, (uint8_t *) data_to_seal,
			sealed_keys_size, (sgx_sealed_data_t *) sealed_keys);

	if(ret != SGX_SUCCESS) {
		//print_string_cout("ERROR: Failed to mr_signer-seal enclave keys\n");
		goto cleanup;
	}

cleanup:

	if (data_to_seal != NULL) {
		free(data_to_seal);
	}
	if (first_seal != NULL) {
		free(first_seal);
	}

	return ret;
}


sgx_status_t ECALL_unseal_keys(
		const uint8_t *sealed_keys,
		uint32_t sealed_keys_size) {

	sgx_status_t ret = SGX_SUCCESS;
	uint32_t first_seal_size = 0;
	uint8_t *first_seal = NULL;
	uint32_t data_size = 0;
	uint8_t *data = NULL;
	uint8_t *tmp_buf = NULL;
	g_private_keys = (truce_private_keys_t *) malloc(8000);
	if (g_after_generate_keys) {
		//print_string_cout("ERROR: Keys have been generated already\n");
		return SGX_ERROR_INVALID_STATE;
	}

	//// First, unseal the mr-enclave sealing
	data_size = sgx_get_encrypt_txt_len((const _sealed_data_t *) sealed_keys);
	if (UINT32_MAX == data_size) {
		//print_string_cout("ERROR: first sgx_get_encrypt_txt_len has failed.\n");
		ret = SGX_ERROR_UNEXPECTED;
		goto cleanup;
	}
	data = (uint8_t *) malloc(data_size);
	if (NULL == data) {
		//print_string_cout("ERROR: failed to allocate first_seal\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}
	ret = sgx_unseal_data((sgx_sealed_data_t *) sealed_keys, NULL, NULL,
			data, &data_size);

	if (ret != SGX_SUCCESS) {
		//print_string_cout("ERROR: Failed to unseal enclave key (step 1)\n");
		goto cleanup;
	}

	//// Finally, Extract the keys from the data
	// Extract public keys
	tmp_buf = data;
	memcpy(&g_public_keys_size, (void *) tmp_buf, sizeof(uint32_t));
	tmp_buf += sizeof(uint32_t);
	tmp_buf += g_public_keys_size;
	// Extract private keys
	memcpy(&g_private_keys_size, (void *) tmp_buf, sizeof(uint32_t));
	tmp_buf += sizeof(uint32_t);
	if (NULL == g_private_keys) {
		//print_string_cout("ERROR: failed to allocate g_private_keys\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}
	memcpy(g_private_keys, (void *) tmp_buf, g_private_keys_size);

cleanup:

	free(data);
	if (first_seal != NULL) {
		free(first_seal);
	}

	if (!g_after_generate_keys) {
		if (g_public_keys != NULL) {
			//free(g_public_keys);
			//g_public_keys = NULL;
		}
		if (g_private_keys != NULL) {
			//free(g_private_keys);
			//g_private_keys = NULL;
		}
		//if (g_rsa != NULL) {
		//RSA_free(g_rsa);
		//g_rsa = NULL;
		//}
	}
	return ret;
}


sgx_status_t ECALL_add_secret(
		const uint8_t* secret_buf,
		uint32_t secret_buf_size, uint8_t* ptback, uint32_t ptbacksize,
		uint32_t* ptactualsize)
{
	enclave_print_integer(ptbacksize);
	sgx_status_t ret = SGX_SUCCESS;
	uint32_t ptextsize = 1000;
	unsigned char ptext[ptextsize]; // TODO
	uint32_t rsa_privkey_len = g_private_keys->rsa4096_private_key_size;
	uint8_t* privkey_rsa_tmp;// = (uint8_t* )calloc(1, rsa_privkey_len);

	privkey_rsa_tmp = (uint8_t* )malloc(rsa_privkey_len);
	memcpy(privkey_rsa_tmp, privkey_rsa_grand, rsa_privkey_len);
	privkey_rsa = d2i_RSAPrivateKey(0, (const unsigned char**) &privkey_rsa_tmp, g_private_keys->rsa4096_private_key_size);
	int num = RSA_private_decrypt(secret_buf_size, secret_buf, ptext,  privkey_rsa, RSA_PKCS1_PADDING);

	if(num > ptextsize || num > ptbacksize){
		ocall_print_string("ERROR: The ptext variable has lesser space allocated than the decrypted plaintext\n");
		enclave_print_integer(num);
		enclave_print_integer(ptbacksize);
		return SGX_ERROR_UNEXPECTED;
	}

	ocall_print_string("Enclave: got secret\n");

	// Copying the plaintext decrypted back to the ptback
	memcpy(ptback, ptext, num);
	memcpy(ptactualsize, &num, sizeof(uint32_t));

	if(num < 0)
		ocall_print_string("\n\n\nError in the code.\n\n\n");

	truce_secret_t *ptr = (truce_secret_t*)malloc(sizeof(truce_secret_t));
	ptr->secret = (unsigned char *) malloc (num);
	memcpy(ptr->secret, ptext, num);
	ptr->secret_len = num;
	ptr->next = NULL;

	if (NULL == Secret_head) {
		Secret_head = ptr;
		Secret_tail = ptr;
	}
	else {
		Secret_tail->next = ptr;
		Secret_tail = ptr;
	}

	return ret;
}

truce_secret_t *truce_get_secrets() {
	return Secret_head;
}

sgx_status_t ECALL_integer_arithmetic(uint32_t *input){
	sgx_status_t ret = SGX_SUCCESS;
	*input *= 2;

	return ret;
}

sgx_status_t ECALL_generate_signature(uint8_t* data, uint32_t data_size, uint8_t* signature){
	sgx_status_t ret = SGX_SUCCESS;

	static sgx_ecc_state_handle_t ecc_state = NULL;
	static sgx_ec256_signature_t *p_signature;

	p_signature = (sgx_ec256_signature_t *)calloc(1, sizeof(sgx_ec256_signature_t));
	ret = sgx_ecc256_open_context(&ecc_state);
	if (SGX_SUCCESS != ret) {
		//enclave_debug_print("ERROR: sgx_ecc256_open_context has failed.\n");
		return SGX_ERROR_UNEXPECTED;
	}
	ret = sgx_ecdsa_sign(data, data_size, &g_private_keys->ec256_private_key, p_signature, ecc_state);
	if(SGX_SUCCESS != ret){
		//print_string_cout("\n\nERROR: Signing with ECDSA has failed\n\n");
		return SGX_ERROR_UNEXPECTED;
	}

	ret = sgx_ecc256_close_context(ecc_state);
	if (SGX_SUCCESS != ret) {
		//enclave_debug_print("ERROR: sgx_ecc256_close_context has failed.\n");
		return SGX_ERROR_UNEXPECTED;
	}
	memcpy(signature, (uint8_t*)p_signature, sizeof(sgx_ec256_signature_t));
	free(p_signature);

	return ret;
}

sgx_status_t verify_signature_and_data(uint8_t* pdata, uint32_t data_size, uint8_t* sig, uint32_t serverid){


	sgx_ecc_state_handle_t ecc_state_verifier_side = NULL;
	sgx_status_t ret = SGX_SUCCESS;
	uint8_t *result_of_signature = NULL;
	result_of_signature = (uint8_t*)calloc(1, sizeof(SGX_EC_VALID));
	truce_public_keys_t* pubkeyset = (truce_public_keys_t*)complement_party_publickeys;
	ret = sgx_ecc256_open_context(&ecc_state_verifier_side);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_close_context has failed.\n");
		goto cleanup;
	}

	ret = sgx_ecc256_close_context(ecc_state_verifier_side);

	if(SGX_SUCCESS != ret){
		//print_string_cout("\n\nERROR: Verification of the signature with ECDSA has failed\n\n");
		enclave_print_integer(ret);
		goto cleanup;
	}
	if(result_of_signature == NULL){
		ocall_print_string("ERROR: Result of signature verification pointer is NULL\n");
	}

	if(SGX_EC_VALID == *result_of_signature){
		sigcorrect++;
		//ocall_print_string("SUCCESS: Signature is completely VALID\n");
	}
	else{
		ocall_print_string("ERROR: Signature INVALID\n");
	}
	//assert(sigcorrect == sigrecvd);
	free(result_of_signature);
cleanup:
	return ret;

}

// Only after the IAS report has been verified can this enclave extract the ECDSA verification public keys from the report.
sgx_status_t verify_IAS_report_inside_enclave(truce_id_t* t_id, uint32_t t_id_size, truce_record_t* t_rec, uint32_t t_rec_size, sgx_measurement_t* expected_mrenclave, uint32_t mrenclave_size, sgx_measurement_t* expected_mrsigner, uint32_t mrsigner_size, uint8_t* publickeys, uint32_t keysize, uint32_t serverid, int sequence){

	truce_record_t record;
	memcpy(&record, t_rec, t_rec_size);
	sgx_measurement_t mrenclavefull;
	memcpy(&mrenclavefull, expected_mrenclave, mrenclave_size);
	sgx_measurement_t mrsignerfull;
	memcpy(&mrsignerfull, expected_mrsigner, mrsigner_size);

	bool result = verify_IAS_report(
			record.ias_report,
			mrenclavefull,
			mrsignerfull,
			publickeys,
			keysize);
	if(result){
		//save keys based on the sequence and serverid (self id).
		ECALL_save_keys_from_report(t_id, t_id_size, t_rec, t_rec_size, expected_mrenclave, mrenclave_size, expected_mrsigner, mrsigner_size, publickeys, keysize, serverid, sequence);

	}
	else{
		ocall_print_string("\n\nERROR [IN ENCLAVE]: IAS report verification unsuccessful.\n");
		return SGX_ERROR_UNEXPECTED;
	}
	return SGX_SUCCESS;
}

sgx_status_t encrypt_and_sign_data(uint8_t* data, uint32_t datalen, uint32_t serverid, uint8_t* encdata, uint8_t* signature){
	sgx_status_t ret = SGX_SUCCESS;

	// First sign data.
	sgx_ecc_state_handle_t ecc_state = NULL;
	sgx_ec256_signature_t *p_signature;

	ret = sgx_ecc256_open_context(&ecc_state);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_open_context has failed.\n");
		return SGX_ERROR_UNEXPECTED;
	}

	p_signature = (sgx_ec256_signature_t *)calloc(1, sizeof(sgx_ec256_signature_t));
	ret = sgx_ecdsa_sign(data, datalen, &g_private_keys->ec256_private_key, p_signature, ecc_state);
	if(SGX_SUCCESS != ret){
		ocall_print_string("\n\nERROR: Signing with ECDSA has failed\n\n");
		return SGX_ERROR_UNEXPECTED;
	}
	//ocall_print_string("Successfully signed the data using ECDSA\n");

	ret = sgx_ecc256_close_context(ecc_state);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_close_context has failed.\n");
		return SGX_ERROR_UNEXPECTED;
	}
	memcpy(signature, (uint8_t*)p_signature, sizeof(sgx_ec256_signature_t));

	ocall_print_string("Starting to encrypt data with RSA\n");
	// Now encrypt data with RSA public key of server with given server id.
	RSA *pubkey_rsa = NULL;
	uint8_t *pubkey_rsa_tmp = NULL;
	uint32_t rsa_pubkey_len;
	if(serverid == 1){
		rsa_pubkey_len = server1_rsapublickeyssize;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, server1_rsapublickeys, rsa_pubkey_len);
	}
	else if(serverid == 2){
		rsa_pubkey_len = server2_rsapublickeyssize;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, server2_rsapublickeys, rsa_pubkey_len);

	}



	if (NULL == pubkey_rsa_tmp) {
		ocall_print_string("ERROR: calloc has failed for pubkey_rsa_tmp\n");
		return SGX_ERROR_UNEXPECTED;
	}


	pubkey_rsa = d2i_RSAPublicKey(0, (const unsigned char**)&pubkey_rsa_tmp, rsa_pubkey_len);
	if (NULL == pubkey_rsa) {
		ocall_print_string("ERROR: d2i_RSAPublicKey has failed\n");
		return SGX_ERROR_UNEXPECTED;
	}
	int output_size;
	output_size = RSA_size(pubkey_rsa);
	enclave_print_integer(output_size);

	if (output_size != RSA_public_encrypt(datalen, data, encdata, pubkey_rsa,
				RSA_PKCS1_PADDING)) {

		ocall_print_string("ERROR: RSA_public_encrypt has failed\n");
		return SGX_ERROR_UNEXPECTED;
	}

	return ret;
}

sgx_status_t aes_decrypt_message_mac(uint8_t *encMessageIn, size_t len, uint8_t *decMessageOut, size_t lenOut, int targetparty)
{
	sgx_status_t ret = SGX_SUCCESS;
	uint8_t *encMessage = encMessageIn;
	uint8_t p_dst1[lenOut+1] = {0};

	if((partynumberself==0 && targetparty==1) || (partynumberself==1 && targetparty == 0)){
		if(aeskeyabflag == 0){
			ret = sgx_rijndael128GCM_decrypt(
					&aeskeyab,
					encMessage + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					lenOut,
					p_dst1,
					encMessage + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) encMessage);

		}
		else{
			ret = sgx_rijndael128GCM_decrypt(
					&aeskey_otherab,
					encMessage + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					lenOut,
					p_dst1,
					encMessage + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) encMessage);

		}
	}
	else if((partynumberself==0 && targetparty==2) || (partynumberself==2 && targetparty == 0)){
		if(aeskeyacflag == 0){
			ret = sgx_rijndael128GCM_decrypt(
					&aeskeyac,
					encMessage + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					lenOut,
					p_dst1,
					encMessage + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) encMessage);

		}
		else{
			ret = sgx_rijndael128GCM_decrypt(
					&aeskey_otherac,
					encMessage + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					lenOut,
					p_dst1,
					encMessage + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) encMessage);

		}
	}
	else if((partynumberself==1 && targetparty==2) || (partynumberself==2 && targetparty == 1)){
		if(aeskeybcflag == 0){
			ret = sgx_rijndael128GCM_decrypt(
					&aeskeybc,
					encMessage + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					lenOut,
					p_dst1,
					encMessage + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) encMessage);

		}
		else{
			ret = sgx_rijndael128GCM_decrypt(
					&aeskey_otherbc,
					encMessage + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					lenOut,
					p_dst1,
					encMessage + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) encMessage);

		}
	}

	memcpy(decMessageOut, p_dst1, lenOut);


	return ret;
}

sgx_status_t aes_encrypt_message_mac(uint8_t *decMessageIn, size_t len, uint8_t *encMessageOut, size_t lenOut, int targetparty)
{
	sgx_status_t ret = SGX_SUCCESS;
	uint8_t *origMessage = decMessageIn;
	uint8_t p_dst1[lenOut+1] = {0};
	// Generate the IV (nonce)
	sgx_read_rand(p_dst1 + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE);

	if((partynumberself==0 && targetparty==1) || (partynumberself==1 && targetparty == 0)){
		if(aeskeyabflag == 0){
			ret = sgx_rijndael128GCM_encrypt(
					&aeskeyab,
					origMessage, len,
					p_dst1 + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					p_dst1 + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) (p_dst1));

		}
		else{
			ret = sgx_rijndael128GCM_encrypt(
					&aeskey_otherab,
					origMessage, len,
					p_dst1 + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					p_dst1 + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) (p_dst1));

		}
	}
	else if((partynumberself==0 && targetparty==2) || (partynumberself==2 && targetparty == 0)){
		if(aeskeyacflag == 0){
			ret = sgx_rijndael128GCM_encrypt(
					&aeskeyac,
					origMessage, len,
					p_dst1 + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					p_dst1 + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) (p_dst1));

		}
		else{
			ret = sgx_rijndael128GCM_encrypt(
					&aeskey_otherac,
					origMessage, len,
					p_dst1 + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					p_dst1 + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) (p_dst1));

		}
	}
	else if((partynumberself==1 && targetparty==2) || (partynumberself==2 && targetparty == 1)){
		if(aeskeybcflag == 0){
			ret = sgx_rijndael128GCM_encrypt(
					&aeskeybc,
					origMessage, len,
					p_dst1 + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					p_dst1 + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) (p_dst1));

		}
		else{
			ret = sgx_rijndael128GCM_encrypt(
					&aeskey_otherbc,
					origMessage, len,
					p_dst1 + SGX_AESGCM_MAC_SIZE + SGX_AESGCM_IV_SIZE,
					p_dst1 + SGX_AESGCM_MAC_SIZE, SGX_AESGCM_IV_SIZE,
					NULL, 0,
					(sgx_aes_gcm_128bit_tag_t *) (p_dst1));

		}
	}


	memcpy(encMessageOut,p_dst1,lenOut);

	return ret;
}

sgx_status_t encrypt_and_sign_aes_key(uint8_t* encdata, uint8_t* signature, uint32_t encdatalen, uint32_t partyid, int sequence){
	sgx_status_t ret = SGX_SUCCESS;

	// First sign keys.
	sgx_ecc_state_handle_t ecc_state = NULL;
	sgx_ec256_signature_t *p_signature;

	ret = sgx_ecc256_open_context(&ecc_state);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_open_context has failed.\n");
		return SGX_ERROR_UNEXPECTED;
	}

	sgx_aes_gcm_128bit_key_t *aeskeycopy = (sgx_aes_gcm_128bit_key_t*) calloc(0, sizeof(sgx_aes_gcm_128bit_key_t));
	if(partyid == 0 && sequence == 0){
		memcpy((uint8_t*)aeskeycopy, (uint8_t*)&aeskeyab, sizeof(sgx_aes_gcm_128bit_key_t));

	}
	else if(partyid == 0 && sequence == 1){
		memcpy((uint8_t*)aeskeycopy, (uint8_t*)&aeskeyac, sizeof(sgx_aes_gcm_128bit_key_t));

	}
	else if(partyid == 1 && sequence == 1){
		memcpy((uint8_t*)aeskeycopy, (uint8_t*)&aeskeybc, sizeof(sgx_aes_gcm_128bit_key_t));

	}
	//memcpy((uint8_t*)aeskeycopy, (uint8_t*)&aeskeyab, sizeof(sgx_aes_gcm_128bit_key_t));
	uint8_t* aeskeypointer = (uint8_t*) aeskeycopy;

	pretty_print_buffer(aeskeypointer, sizeof(sgx_aes_gcm_128bit_key_t));
	enclave_print_integer(sizeof(sgx_aes_gcm_128bit_key_t));

	p_signature = (sgx_ec256_signature_t *)calloc(1, sizeof(sgx_ec256_signature_t));
	ret = sgx_ecdsa_sign(aeskeypointer, sizeof(sgx_aes_gcm_128bit_key_t), &g_private_keys->ec256_private_key, p_signature, ecc_state);
	if(SGX_SUCCESS != ret){
		ocall_print_string("\n\nERROR: Signing with ECDSA has failed\n\n");
		return SGX_ERROR_UNEXPECTED;
	}
	ocall_print_string("Successfully signed the data using ECDSA\n");

	ret = sgx_ecc256_close_context(ecc_state);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_close_context has failed.\n");
		return SGX_ERROR_UNEXPECTED;
	}
	memcpy((uint8_t*)signature, (uint8_t*)p_signature, sizeof(sgx_ec256_signature_t));


	ocall_print_string("Starting to encrypt data with RSA\n");
	// Now encrypt data with RSA public key of server with given server id.
	RSA *pubkey_rsa = NULL;
	uint8_t *pubkey_rsa_tmp = NULL;
	uint32_t rsa_pubkey_len;
	if(partyid == 0 && sequence == 0){
		rsa_pubkey_len = other_party_keyab->rsa4096_public_key_size;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, other_party_keyab->rsa4096_public_key, rsa_pubkey_len);

	}
	else if(partyid == 0 && sequence == 1){
		rsa_pubkey_len = other_party_keyac->rsa4096_public_key_size;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, other_party_keyac->rsa4096_public_key, rsa_pubkey_len);

	}
	else if(partyid == 1 && sequence == 1){
		rsa_pubkey_len = other_party_keybc->rsa4096_public_key_size;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, other_party_keybc->rsa4096_public_key, rsa_pubkey_len);

	}


	if (NULL == pubkey_rsa_tmp) {
		ocall_print_string("ERROR: calloc has failed for pubkey_rsa_tmp\n");
		return SGX_ERROR_UNEXPECTED;
	}



	pubkey_rsa = d2i_RSAPublicKey(0, (const unsigned char**)&pubkey_rsa_tmp, rsa_pubkey_len);
	if (NULL == pubkey_rsa) {
		ocall_print_string("ERROR: d2i_RSAPublicKey has failed\n");
		return SGX_ERROR_UNEXPECTED;
	}
	int output_size;
	output_size = RSA_size(pubkey_rsa);
	enclave_print_integer(output_size);

	if (output_size != RSA_public_encrypt(sizeof(sgx_aes_gcm_128bit_key_t), aeskeypointer, encdata, pubkey_rsa,
				RSA_PKCS1_PADDING)) {

		ocall_print_string("ERROR: RSA_public_encrypt has failed\n");
		return SGX_ERROR_UNEXPECTED;
	}

	return ret;
}


sgx_status_t ECALL_save_keys_from_report(truce_id_t* t_id, uint32_t t_id_size, truce_record_t* t_rec, uint32_t t_rec_size, sgx_measurement_t* expected_mrenclave, uint32_t mrenclave_size, sgx_measurement_t* expected_mrsigner, uint32_t mrsigner_size, uint8_t* publickeys, uint32_t keysize, uint32_t partyid, int sequence){

	partynumberself = partyid;
	if((partyid==0 && sequence==0)||(partyid==1 && sequence==0)){
		//keyab
		other_party_keyab = (truce_public_keys_t*)malloc(g_public_keys_size);
		complement_party_publickeys = (uint8_t*)malloc(g_public_keys_size);
		complement_party_publickeyssize = keysize;
		memcpy(complement_party_publickeys, publickeys, g_public_keys_size);
		memcpy(other_party_keyab, complement_party_publickeys, g_public_keys_size);

#ifdef ENCLAVE_VERBOSE
		//print_string_cout("\n\n\n\nTHE KEY IS:");
		pretty_print_buffer((uint8_t*)&other_party_keyab->rsa4096_public_key, other_party_keyab->rsa4096_public_key_size);
		pretty_print_buffer((uint8_t*)&other_party_keyab->ec256_public_key, sizeof(sgx_ec256_public_t));
		enclave_print_integer(keysize);
		enclave_debug_print("======>>>>> Saved keys of the other party\n");
#endif

	}
	else if((partyid==0 && sequence==1)||(partyid==2 && sequence==0)){
		//keyac
		other_party_keyac = (truce_public_keys_t*)malloc(g_public_keys_size);
		complement_party_publickeys = (uint8_t*)malloc(g_public_keys_size);
		complement_party_publickeyssize = keysize;
		memcpy(complement_party_publickeys, publickeys, g_public_keys_size);
		memcpy(other_party_keyac, complement_party_publickeys, g_public_keys_size);

#ifdef ENCLAVE_VERBOSE
		//print_string_cout("\n\n\n\nTHE KEY IS:");
		pretty_print_buffer((uint8_t*)&other_party_keyac->rsa4096_public_key, other_party_keyac->rsa4096_public_key_size);
		pretty_print_buffer((uint8_t*)&other_party_keyac->ec256_public_key, sizeof(sgx_ec256_public_t));
		enclave_print_integer(keysize);
		enclave_debug_print("======>>>>> Saved keys of the other party\n");
#endif

	}
	else if((partyid==1 && sequence==1)||(partyid==2 && sequence==1)){
		//keybc
		other_party_keybc = (truce_public_keys_t*)malloc(g_public_keys_size);
		complement_party_publickeys = (uint8_t*)malloc(g_public_keys_size);
		complement_party_publickeyssize = keysize;
		memcpy(complement_party_publickeys, publickeys, g_public_keys_size);
		memcpy(other_party_keybc, complement_party_publickeys, g_public_keys_size);

#ifdef ENCLAVE_VERBOSE
		//print_string_cout("\n\n\n\nTHE KEY IS:");
		pretty_print_buffer((uint8_t*)&other_party_keybc->rsa4096_public_key, other_party_keybc->rsa4096_public_key_size);
		pretty_print_buffer((uint8_t*)&other_party_keybc->ec256_public_key, sizeof(sgx_ec256_public_t));
		enclave_print_integer(keysize);
		enclave_debug_print("======>>>>> Saved keys of the other party\n");
#endif

	}
	free(complement_party_publickeys);
	return SGX_SUCCESS;
}

/****************************************************
 * Seal the ECDSA verification keys of the other party
 * extracted from the report.
 * ***************************************************
 */

bool get_seals_sizes_other(uint32_t &first_seal_size, uint32_t &second_seal_size) {

	if (!g_after_generate_keys) {
		enclave_debug_print("Warning: ECALL_get_sealed_keys_size has failed since there is no public keys.\n");
		return false;
	}
	// First Seal using MRSIGNER. The data is: (pub_key_size, pub_key)
	first_seal_size = sgx_calc_sealed_data_size(0,
			sizeof(uint32_t) + g_public_keys_size);
	if (UINT32_MAX == first_seal_size) {
		enclave_debug_print("ERROR: first sgx_calc_sealed_data_size has failed.\n");
		return false;
	}
	second_seal_size = sgx_calc_sealed_data_size(0, first_seal_size); // Second seal using MRENCLAVE
	if (UINT32_MAX == second_seal_size) {
		enclave_debug_print("ERROR: first sgx_calc_sealed_data_size has failed.\n");
		return false;
	}
	return true;
}

sgx_status_t ECALL_get_sealed_keys_size_other(uint32_t *sealed_keys_size) {
	uint32_t first_seal_size = 0;
	uint32_t second_seal_size = 0;

	if (!get_seals_sizes_other(first_seal_size, second_seal_size)) {
		enclave_debug_print("ERROR: get_seals_sizes has failed.\n");
		return SGX_ERROR_INVALID_STATE;
	}

	*sealed_keys_size = first_seal_size;
	return SGX_SUCCESS;
}

sgx_status_t ECALL_seal_keys_other(uint8_t *sealed_keys,
		uint32_t sealed_keys_size) {

	sgx_status_t ret = SGX_SUCCESS;
	uint32_t first_seal_size = 0;
	uint32_t second_seal_size = 0;
	uint8_t *data_to_seal = NULL;
	uint32_t data_to_seal_size = 0;
	uint8_t *tmp_ptr = NULL;
	uint8_t *first_seal = NULL;
	sgx_attributes_t attribute_mask;
	sgx_misc_select_t misc_mask = 0xF0000000;


	data_to_seal_size = sizeof(uint32_t) + g_public_keys_size;
	data_to_seal = (uint8_t *) malloc(data_to_seal_size);

	if (NULL == data_to_seal) {
		//print_string_cout("ERROR: failed to allocate data_to_seal.\n");
		goto cleanup;
	}
	tmp_ptr = data_to_seal;

	memcpy(tmp_ptr, (void *) &g_public_keys_size, sizeof(uint32_t));
	tmp_ptr += sizeof(uint32_t);
	memcpy(tmp_ptr, (void *) other_party_keyab, g_public_keys_size);
	tmp_ptr += g_public_keys_size;

	if (!get_seals_sizes_other(first_seal_size, second_seal_size)) {
		//print_string_cout("ERROR: get_seals_sizes has failed.\n");
		ret = SGX_ERROR_UNEXPECTED;
		goto cleanup;
	}


	first_seal = (uint8_t *) malloc(first_seal_size);
	if (NULL == first_seal) {
		//print_string_cout("ERROR: failed to allocate first_seal.\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}

	// First, seal with key derived from MR_SIGNER
	ret = sgx_seal_data(0,NULL,data_to_seal_size, (uint8_t *) data_to_seal,
			sealed_keys_size, (sgx_sealed_data_t *) sealed_keys);

	if(ret != SGX_SUCCESS) {
		//print_string_cout("ERROR: Failed to mr_signer-seal enclave keys\n");
		goto cleanup;
	}

cleanup:

	if (data_to_seal != NULL) {
		free(data_to_seal);
	}
	if (first_seal != NULL) {
		free(first_seal);
	}

	return ret;
}


sgx_status_t ECALL_unseal_keys_other(
		const uint8_t *sealed_keys,
		uint32_t sealed_keys_size) {

	sgx_status_t ret = SGX_SUCCESS;
	uint32_t first_seal_size = 0;
	uint8_t *first_seal = NULL;
	uint32_t data_size = 0;
	uint8_t *data = NULL;
	uint8_t *tmp_buf = NULL;
	uint8_t* complement_party_publickeys;
	other_party_keyab = (truce_public_keys_t *) malloc(g_public_keys_size);
	complement_party_publickeys = (uint8_t*)malloc(g_public_keys_size);

	//// First, unseal the mr-enclave sealing
	data_size = sgx_get_encrypt_txt_len((const _sealed_data_t *) sealed_keys);
	if (UINT32_MAX == data_size) {
		//print_string_cout("ERROR: first sgx_get_encrypt_txt_len has failed.\n");
		ret = SGX_ERROR_UNEXPECTED;
		goto cleanup;
	}
	data = (uint8_t *) malloc(data_size);
	if (NULL == data) {
		//print_string_cout("ERROR: failed to allocate first_seal\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}

	ret = sgx_unseal_data((sgx_sealed_data_t *) sealed_keys, NULL, NULL,
			data, &data_size);

	if (ret != SGX_SUCCESS) {
		//print_string_cout("ERROR: Failed to unseal enclave key (step 1)\n");
		goto cleanup;
	}

	//// Finally, Extract the keys from the data
	// Extract public keys
	tmp_buf = data;
	memcpy(&g_public_keys_size, (void *) tmp_buf, sizeof(uint32_t));
	tmp_buf += sizeof(uint32_t);
	if (NULL == other_party_keyab) {
		//print_string_cout("ERROR: failed to allocate other_party_key\n");
		ret = SGX_ERROR_OUT_OF_MEMORY;
		goto cleanup;
	}
	memcpy(other_party_keyab, (void *) tmp_buf, g_public_keys_size);


cleanup:

	free(data);
	if (first_seal != NULL) {
		free(first_seal);
	}

	return ret;
}

sgx_status_t ECALL_get_RSA_cipher_size(uint32_t* rsasize, int partyid, int sequence){
	enclave_print_integer(1234);
	RSA *pubkey_rsa = NULL;
	uint8_t *pubkey_rsa_tmp = NULL;
	uint32_t rsa_pubkey_len;
	if(partyid == 0 && sequence == 0){
		rsa_pubkey_len = other_party_keyab->rsa4096_public_key_size;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, other_party_keyab->rsa4096_public_key, rsa_pubkey_len);

	}
	else if(partyid == 0 && sequence == 1){
		rsa_pubkey_len = other_party_keyac->rsa4096_public_key_size;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, other_party_keyac->rsa4096_public_key, rsa_pubkey_len);

	}
	else if(partyid == 1 && sequence == 1){
		rsa_pubkey_len = other_party_keybc->rsa4096_public_key_size;
		pubkey_rsa_tmp = (uint8_t *) calloc(1, rsa_pubkey_len);
		memcpy(pubkey_rsa_tmp, other_party_keybc->rsa4096_public_key, rsa_pubkey_len);

	}
	if (NULL == pubkey_rsa_tmp) {
		enclave_print_integer(11111);
		ocall_print_string("ERROR: calloc has failed for pubkey_rsa_tmp\n");
		return SGX_ERROR_UNEXPECTED;
	}
	pubkey_rsa = d2i_RSAPublicKey(0, (const unsigned char**)&pubkey_rsa_tmp, rsa_pubkey_len);
	if (NULL == pubkey_rsa) {
		enclave_print_integer(22222);
		ocall_print_string("ERROR: d2i_RSAPublicKey has failed\n");
		return SGX_ERROR_UNEXPECTED;
	}
	enclave_print_integer(33333);
	int res = RSA_size(pubkey_rsa);
	memcpy(rsasize, &res, sizeof(uint32_t));
	enclave_print_integer(RSA_size(pubkey_rsa));
	return SGX_SUCCESS;
}

sgx_status_t decrypt_verify_aes_key_and_save(uint8_t* pdata, uint32_t data_size, uint8_t* sig, uint32_t partyid, int sequence){

	sgx_status_t ret = SGX_SUCCESS;
	partynumberself = partyid;
	unsigned char* ptback;
	uint32_t ptbacksize = 1000;
	ptback = (unsigned char*)calloc(1, ptbacksize);
	uint32_t* ptactualsize;
	ptactualsize = (uint32_t*)calloc(1, sizeof(uint32_t));
	// a share
	ret = ECALL_add_secret(pdata, data_size, ptback, ptbacksize, ptactualsize);
	if(ret != SGX_SUCCESS){
		return SGX_ERROR_UNEXPECTED;
	}

	//ocall_print_string("Verfying the signature attached with the data just received from sender enclave.\n");
	sgx_ecc_state_handle_t ecc_state_verifier_side = NULL;

	uint8_t *result_of_signature = NULL;
	result_of_signature = (uint8_t*)calloc(1, sizeof(SGX_EC_VALID));
	//sgx_ec256_signature_t *t_signature;
	ret = sgx_ecc256_open_context(&ecc_state_verifier_side);
	if (SGX_SUCCESS != ret) {
		enclave_debug_print("ERROR: sgx_ecc256_close_context has failed.\n");
		goto cleanup;
	}
	result_of_signature = (uint8_t*)calloc(1, sizeof(SGX_EC_VALID));
	truce_public_keys_t pubkeyset;

	if(partyid == 1 && sequence == 0){
		memcpy(&pubkeyset, other_party_keyab, ec256_pubkey_size);
	}
	else if(partyid == 2 && sequence == 0){
		memcpy(&pubkeyset, other_party_keyac, ec256_pubkey_size);
	}
	else if(partyid == 2 && sequence == 1){
		memcpy(&pubkeyset, other_party_keybc, ec256_pubkey_size);
	}

	ret = sgx_ecdsa_verify(ptback, *ptactualsize,(const sgx_ec256_public_t*) &pubkeyset.ec256_public_key, (sgx_ec256_signature_t *)sig, result_of_signature, ecc_state_verifier_side);

	if(SGX_SUCCESS != ret){
		ocall_print_string("\n\nERROR: Verification of the signature with ECDSA has failed\n\n");
		goto cleanup;
	}
	if(result_of_signature == NULL){
		ocall_print_string("ERROR: Result of signature verification pointer is NULL\n");
	}

	if(SGX_EC_VALID == *result_of_signature){
		ocall_print_string("SUCCESS: Signature is completely VALID\n");
		sgx_aes_gcm_128bit_key_t* aeskeypointer = (sgx_aes_gcm_128bit_key_t*) ptback;

		if(partyid == 1 && sequence == 0){
			memcpy((uint8_t*)&aeskey_otherab, (uint8_t*) aeskeypointer, sizeof(sgx_aes_gcm_128bit_key_t));
			aeskeyabflag = 1; // This means user aeskeyotherab and not aeskeyab
		}
		else if(partyid == 2 && sequence == 0){
			memcpy((uint8_t*)&aeskey_otherac, (uint8_t*) aeskeypointer, sizeof(sgx_aes_gcm_128bit_key_t));
			aeskeyacflag = 1;
		}
		else if(partyid == 2 && sequence == 1){
			memcpy((uint8_t*)&aeskey_otherbc, (uint8_t*) aeskeypointer, sizeof(sgx_aes_gcm_128bit_key_t));
			aeskeybcflag = 1;
		}

		ocall_print_string("[SUCCESS]: Saved the AES Key from SERVER0");
	}
	else{
		ocall_print_string("ERROR: Signature INVALID\n");
	}
	// Use the aes key received from the other party.
	whichaeskey = 1;

cleanup:
	return ret;

}

void set_party_number(int pnum){
	partynumberself = pnum;
}
