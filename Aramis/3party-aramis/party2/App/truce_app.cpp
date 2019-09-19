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

#include "truce_app.h"

#define ERROR_SEND_SIGNED_TO_CLIENT 0
#define SUCCESS_SEND_SIGNED_TO_CLIENT 1
#define ENCLAVE_PATH "truce_enclave.signed.so"
#define SERVER1 1
#define ASSISTING_SERVER 2
#define ECDSA_SIG_SIZE 64
#define BUFLEN 2048
#define SGX_AESGCM_MAC_SIZE 16
#define SGX_AESGCM_IV_SIZE 12
#define VERBOSE_PRINT

using namespace std;

uint32_t ECDSA_signature_size = 64;
uint32_t signature_key_size = 0;
uint8_t* signature_key;
FILE* OUTPUT =  stdout;

char truceServerAddress[100]; //TODO
int truceServerPort = -1;

/* OCall functions */
void ocall_print_string(const char *str)
{
	/* Proxy/Bridge will check the length and null-terminate
	 * the input string to prevent buffer overflow.
	 */
	printf("%s", str);
}


void print_string(const char* str)
{
	fprintf(OUTPUT, "%s", str);
}


bool truce_client_init(const char* truce_server_address)
{
	const char *port_pos = strchr(truce_server_address, ':');
	if (port_pos) {
		size_t addr_len = port_pos - truce_server_address;
		memcpy(truceServerAddress, truce_server_address, addr_len);
		truceServerAddress[addr_len] = '\0';
		truceServerPort = atoi(port_pos + 1);
	}
	else {
		memcpy(truceServerAddress, truce_server_address, strlen(truce_server_address));
		truceServerPort = SP_RS_PORT_DEFAULT;
	}

#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "TruCE server address: %s, port %d\n", truceServerAddress, truceServerPort);
#endif

	// TODO: we call curl_global_init since for some reason, it makes verify_cert_chain to work.
	// It might be cause due to some memory leak, or for some good reason. We should investigate it.
	curl_global_init(CURL_GLOBAL_DEFAULT);

	return true;
}

void print_buffer(uint8_t* buf, 
		int len)
{
	char out[20];
	snprintf(out, 20, "Length: %d \n", len);
	fprintf(OUTPUT, "%s", out);
	for (int i=0; i < len; i++) {
		snprintf(out, 20, "0x%c ", buf[i]);
		fprintf(OUTPUT, "%s", out);
	}
	fprintf(OUTPUT, "\n");
}

void pretty_print_buffer(uint8_t* buf, 
		int len)
{
	char out[20];
	snprintf(out, 20, "Length: %d \n", len);
	fprintf(OUTPUT, "%s", out);
	for (int i=0; i < len; i++) {
		snprintf(out, 20, "%3d ", buf[i]);
		fprintf(OUTPUT, "%s", out);
	}
	fprintf(OUTPUT, "\n");
}


bool truce_client_recv_enclave_record(
		const truce_id_t &t_id,
		truce_record_t &t_rec)
{

	if (NULL == truceServerAddress) {
		fprintf(OUTPUT, "ERROR: client not initialized with TruCE server address\n");
		return false;
	}

	// Create connection to TruCE server
	int sockfd = -1;
	char *ias_report_body = NULL;
	char *ias_report_signature_base64 = NULL;
	char *ias_report_cert_chain_urlsafe_pem = NULL;
	uint8_t *public_keys = NULL;
	uint32_t public_keys_size = 0;
	int len = 0;
	bool ret = false;
	int tmp_int = 0;
	uint8_t match_result = 0;

	if (!inet_connect(sockfd, truceServerAddress, truceServerPort)) {
		fprintf(OUTPUT, "ERROR: connecting to TruCE server (%s:%d) has failed\n",
				truceServerAddress, truceServerPort);
		return false;
	}

#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Connected to TruCE server\n");
#endif
	// Sending t_id
	if (!write_all(sockfd, (uint8_t *) &t_id, sizeof(t_id))) {
		fprintf( OUTPUT, "ERROR: failed to send t_id\n");
		goto cleanup;
	}

	// Receiving search result
	if (1 != read(sockfd, &match_result, 1)) {
		fprintf( OUTPUT, "ERROR: failed to read match_result\n");
		goto cleanup;
	}

	if (match_result != 1) {
		fprintf( OUTPUT, "Warning: No enclave was found\n");
		goto cleanup;
	}

	// Receiving IAS_report_body length
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Receiving the size of IAS_report_body...\n");
#endif
	if (!read_all(sockfd, (uint8_t *) &tmp_int, 4)) {
		fprintf( OUTPUT, "ERROR: missing bytes for IAS_report_body length\n");
		goto cleanup;
	}
	len = ntohl(tmp_int);

	// Receiving IAS_report_body
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Receiving %u bytes of IAS_report_body...\n", len);
#endif
	ias_report_body = (char *) calloc(1,len+1);
	if (NULL == ias_report_body) {
		fprintf( OUTPUT, "ERROR: failed to allocate %d byte for ias_report_body\n", len);
		goto cleanup;
	}
	if (!read_all(sockfd, (uint8_t *) ias_report_body, len)) {
		fprintf( OUTPUT, "ERROR: missing bytes for ias_report_body\n");
		goto cleanup;
	}


	// Receiving IAS_report_signature_base64 length
	if (!read_all(sockfd, (uint8_t *) &tmp_int, 4)) {
		fprintf( OUTPUT, "ERROR: missing bytes for IAS_report_signature_base64 length\n");
		goto cleanup;
	}
	len = ntohl(tmp_int);
	// Receiving IAS_report_signature_base64
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Receiving %u bytes of ias_report_signature_base64...\n", len);
#endif
	ias_report_signature_base64 = (char *) calloc(1,len+1);
	if (NULL == ias_report_signature_base64) {
		fprintf( OUTPUT, "ERROR: failed to allocated %d byte for ias_report_signature_base64\n", len+1);
		goto cleanup;
	}
	if (!read_all(sockfd, (uint8_t *) ias_report_signature_base64, len)) {
		fprintf( OUTPUT, "ERROR: missing bytes for ias_report_signature_base64\n");
		goto cleanup;
	}


	// Receiving IAS_report_cert_chain_urlsafe_pem length
	if (!read_all(sockfd, (uint8_t *) &tmp_int, 4)) {
		fprintf( OUTPUT, "ERROR: missing bytes for IAS_report_cert_chain_urlsafe_pem length\n");
		goto cleanup;
	}
	len = ntohl(tmp_int);
	// Receiving IAS_report_cert_chain_urlsafe_pem
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Receiving %u bytes of ias_report_cert_chain_urlsafe_pem...\n", len);
#endif

	ias_report_cert_chain_urlsafe_pem = (char *) calloc(1,len+1);
	if (NULL == ias_report_cert_chain_urlsafe_pem) {
		fprintf( OUTPUT, "ERROR: failed to allocated %d byte for ias_report_cert_chain_urlsafe_pem\n", len+1);
		goto cleanup;
	}
	if (!read_all(sockfd, (uint8_t *) ias_report_cert_chain_urlsafe_pem, len)) {
		fprintf( OUTPUT, "ERROR: missing bytes for ias_report_cert_chain_urlsafe_pem\n");
		goto cleanup;
	}

	// Receiving public_keys_size length
	if (!read_all(sockfd, (uint8_t *) &tmp_int, 4)) {
		fprintf( OUTPUT, "ERROR: missing bytes for public_keys_size\n");
		goto cleanup;
	}
	public_keys_size = ntohl(tmp_int);

	// Receiving public_keys
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Receiving %u bytes of Enclave's Public Keys...\n", public_keys_size);
#endif

	public_keys = (uint8_t *) calloc(1,public_keys_size);
	if (NULL == public_keys) {
		fprintf( OUTPUT, "ERROR: failed to allocated %d byte for public_keys\n", public_keys_size);
		goto cleanup;
	}
	if (!read_all(sockfd, public_keys, public_keys_size)) {
		fprintf( OUTPUT, "ERROR: missing bytes for public_keys\n");
		goto cleanup;
	}

	t_rec.ias_report.report_body = ias_report_body;
	t_rec.ias_report.report_cert_chain_urlsafe_pem = ias_report_cert_chain_urlsafe_pem;
	t_rec.ias_report.report_signature_base64 = ias_report_signature_base64;
	t_rec.p_public_keys = (truce_public_keys_t *) public_keys;
	t_rec.public_keys_size = public_keys_size;

	ret = true;
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Record has been received successfully!\n");
#endif

cleanup:

	if (sockfd != 0) {
		close(sockfd);
	}
	if (!ret) {
		if (ias_report_body != NULL) {
			free(ias_report_body);
		}
		if (ias_report_cert_chain_urlsafe_pem != NULL) {
			free(ias_report_cert_chain_urlsafe_pem);
		}
		if (ias_report_signature_base64 != NULL) {
			free(ias_report_signature_base64);
		}
		if (public_keys != NULL) {
			free(public_keys);
		}
	}

	return ret;
}


bool truce_client_extract_quote_from_record(
		const truce_record_t &t_rec,
		sgx_quote_t &quote)
{
#ifdef VERBOSE_PRINT
	printf("\nCalling quote extractor.\n");
#endif
	return extract_quote_from_IAS_report(
			t_rec.ias_report,
			quote,
			print_string);
}

bool truce_client_verify_enclave_record(
		sgx_enclave_id_t enclave_id, 
		sgx_status_t &status,
		const truce_id_t &t_id, 
		uint32_t t_id_size,
		const truce_record_t &t_rec, 
		uint32_t t_rec_size,
		const sgx_measurement_t &expected_mrenclave, 
		uint32_t mrenclave_size,
		const sgx_measurement_t &expected_mrsigner, 
		uint32_t mrsigner_size,
		int serverid, 
		int sequence)
{

	uint8_t sha_result[SHA256_DIGEST_LENGTH];


	// Verify that t_id == sha256(public_keys)
	SHA256((uint8_t *) t_rec.p_public_keys,
			t_rec.public_keys_size,
			(uint8_t *) &sha_result);

	if (memcmp((void *) &t_id, (void *) &sha_result, SHA256_DIGEST_LENGTH) != 0) {
		fprintf(OUTPUT, "t_id is different from sha256(public_keys).\n");
		return false;
	}

	truce_id_t *t_idtemp = (truce_id_t*) calloc(1, t_id_size);
	memcpy(t_idtemp, &t_id, t_id_size);
	truce_record_t *t_rectemp = (truce_record_t*) calloc(1, t_rec_size);
	memcpy(t_rectemp, &t_rec, t_rec_size);
	sgx_measurement_t *mrenclavetemp = (sgx_measurement_t*) calloc(1, mrenclave_size);
	memcpy(mrenclavetemp, &expected_mrenclave, mrenclave_size );
	sgx_measurement_t *mrsignertemp = (sgx_measurement_t*) calloc(1, mrsigner_size);
	memcpy(mrsignertemp, &expected_mrsigner, mrsigner_size);

	//Preparing the variables to be sent to the enclave.
	sgx_status_t statustemp = SGX_SUCCESS;
#ifdef VERBOSE_PRINT
	printf("\nCalling IAS report verification inside enclave.\n");
#endif

	verify_IAS_report_inside_enclave(enclave_id, &statustemp, t_idtemp, t_id_size, t_rectemp, t_rec_size, mrenclavetemp, mrenclave_size, mrsignertemp, mrsigner_size, (uint8_t *) t_rec.p_public_keys, t_rec.public_keys_size, serverid, sequence);

	cout<<(success+"Verified IAS report inside application.\n");

	if(statustemp == SGX_SUCCESS){
		return true;
	}
	else{
		return false;
	}

}

bool verify_measurement_values(sgx_quote_t &p_quote, int targetpartynum){
	//First extract the mrenclave and mrsigner values
	//from the file given by the other party
	//in a string
	std::string filepath = "../party"+std::to_string(targetpartynum)+"/dumpsign.txt";
	std::ifstream infile(filepath);
	std::string line;
	std::string mrenclavereal = "";
	std::string mrsignerreal = "";
	int linenum = 1;
	while(std::getline(infile, line)){
		if(linenum == 81 || linenum == 82){
			mrenclavereal += line;
		}
		if(linenum == 415 || linenum == 416){
			mrsignerreal += line;
		}
		linenum++;
	}
#ifdef VERBOSE_PRINT
	std::cout<<mrenclavereal<<std::endl;
	std::cout<<mrsignerreal<<std::endl;
#endif
	//Now compare these real values with the ones received
	//from the report
	std::string mrenclaverecvd = "";
	std::string mrsignerrecvd = "";
	char* temp = (char*)malloc(6);
	std::string stringtemp;
	for(int i=0; i<32; i++){
		if(i==0){
			snprintf(temp, 5, "0x%02x", p_quote.report_body.mr_enclave.m[i]);
			stringtemp.assign(temp, 4);
		}
		else{
			snprintf(temp, 6, " 0x%02x", p_quote.report_body.mr_enclave.m[i]);
			stringtemp.assign(temp, 5);
		}
		mrenclaverecvd += stringtemp;
	}
	for(int i=0; i<32; i++){
		if(i==0){
			snprintf(temp, 5, "0x%02x", p_quote.report_body.mr_signer.m[i]);
			stringtemp.assign(temp, 4);
		}
		else{
			snprintf(temp, 6, " 0x%02x", p_quote.report_body.mr_signer.m[i]);
			stringtemp.assign(temp, 5);
		}
		mrsignerrecvd += stringtemp;
	}

	mrsignerreal = mrsignerreal.substr(0, 159);
	mrenclavereal = mrenclavereal.substr(0, 159);

#ifdef VERBOSE_PRINT
	std::cout<<mrenclaverecvd<<std::endl;
	std::cout<<mrsignerrecvd<<std::endl;
#endif
	if(mrsignerrecvd.compare(mrsignerreal)==0 && mrenclaverecvd.compare(mrenclavereal)==0){
#ifdef VERBOSE_PRINT
		printf("===>>>Measurement correct.\n");
#endif
		return true;
	}
	return false;
}


/* ********************************************* */
// main control functions for alice and bob sequence.
// Party 0
int attest_main_alice(sgx_enclave_id_t &enclave_id)
{
	sgx_status_t ret = SGX_SUCCESS;
	int listenfd = -1;
	int listenfd2 = -1;

	int connfd = -1;
	int connfd2 = -1;

	sgx_status_t status = SGX_SUCCESS;

	FILE* OUTPUT = stdout;

	char* sp_address = truce_server_address;

	truce_config_t t_config;
	t_config.truce_server_address = sp_address;

	truce_session_t t_session;
	if (!truce_session(enclave_id, t_config, t_session)) {
		cout<<(failure+"Failed to create truce_session.\n");
		return 1;
	}
	cout<<(success+"Successfully created truce_session.\n");

#ifdef VERBOSE_PRINT
	printf("Received t_id:\n");
	print_buffer((uint8_t *) &t_session.truce_id, sizeof(t_session.truce_id));
#endif

	// ---------------For connection with party 1----------

	// Now, the server will receive the client's report from TruCE server and IAS.
	const char* agent_address = party1_address;
	const char* agent_address2 = party2_address;

	truce_id_t t_id = {{0}};
	truce_id_t t_id2 = {{0}};

	truce_record_t t_rec;
	truce_record_t t_rec2;

	sgx_measurement_t expected_mrenclave = {{0}}; // Should be the real value
	sgx_measurement_t expected_mrenclave2 = {{0}};
	sgx_measurement_t expected_mrsigner = {{0}}; // Should be the real value
	sgx_measurement_t expected_mrsigner2 = {{0}};

	sgx_quote_t quote = {0};
	sgx_quote_t quote2 = {0};

	int sockfd = -1;
	int sockfd2 = -1;

	// Creating a listening socket. Waiting for clients connections.
	if (!inet_listen(listenfd, port01)) {
		cout<<(failure+"Failed to listen on port %d.\n", 0);
		return 1;
	}


	if (!inet_accept(connfd, listenfd)) {
		cout<<(failure+"inet_accept has failed.\n");
		return 1;
	}

	// Party should send its truce ID to the other party
	if (!write_all(connfd, t_session.truce_id, sizeof(t_session.truce_id))) {
		cout<<(failure+"failed to write ")<<sizeof(t_session.truce_id)<<(" bytes of truce_id.\n");
		return 1;
	}

	// Party received the truce id of the other party
	if (!read_all(connfd, (uint8_t *) &t_id, sizeof(t_id))) {
		cout<<(failure+"Failed to read ")<<sizeof(t_id)<<(" bytes of t_id\n");
		goto cleanup;
	}

#ifdef VERBOSE_PRINT
	printf("Received t_id from client:\n");
	print_buffer((uint8_t *) &t_id, sizeof(t_id));
#endif

	// Receive the REPORT of the other party from the truce_server
	truce_client_init(truce_server_address);

	if (!truce_client_recv_enclave_record(t_id, t_rec)) {
		cout<<(failure+"Failed to receive truce record from truce server\n");
		goto cleanup;
	}
	cout<<(success+"Received enclave record from TruCE server\n");

	if (!truce_client_extract_quote_from_record(
				t_rec,
				quote)) {
		cout<<(failure+"Failed to extract quote from record\n");
		goto cleanup;
	}

#ifdef VERBOSE_PRINT
	printf("Extracted the quote from the enclave record.\n");
#endif

	memcpy((void *) &expected_mrenclave, (void *) &quote.report_body.mr_enclave, sizeof(sgx_measurement_t));
	memcpy((void *) &expected_mrsigner, (void *) &quote.report_body.mr_signer, sizeof(sgx_measurement_t));

#ifdef VERBOSE_PRINT
	printf("MRSIGNER and MRENCLAVE values extracted.\n");
	print_report_body(1, &quote.report_body);
#endif

	if (!truce_client_verify_enclave_record(
				enclave_id, status,
				t_id, sizeof(t_id),
				t_rec, sizeof(t_rec),
				expected_mrenclave, sizeof(expected_mrenclave),
				expected_mrsigner, sizeof(expected_mrsigner),
				0, 0)) {

		cout<<(failure+"Failed to verify enclave's record\n");
		goto cleanup;
	}
	if(!verify_measurement_values(quote, 1)){
		cout<<(failure+"MEASUREMENT ERROR: Measurement values do not match\n");
		goto cleanup;
	}

	cout<<(success+"SUCCESS: Successfully verified enclave's record inside the server's enclave.\n");

	// -------------------Connection with party 2-----------

	// Creating a listening socket. Waiting for clients connections.
	if (!inet_listen(listenfd2, port02)) {
		cout<<(failure+"Failed to listen on port %d.\n", 0);
		return 1;
	}


	if (!inet_accept(connfd2, listenfd2)) {
		cout<<(failure+"inet_accept has failed.\n");
		return 1;
	}

	// Party should send its truce ID to the other party
	if (!write_all(connfd2, t_session.truce_id, sizeof(t_session.truce_id))) {
		cout<<(failure+"failed to write ")<<sizeof(t_session.truce_id)<<(" bytes of truce_id.\n");
		return 1;
	}

	// Party received the truce id of the other party
	if (!read_all(connfd2, (uint8_t *) &t_id2, sizeof(t_id2))) {
		cout<<(failure+"Failed to read ")<<sizeof(t_id2)<<(" bytes of t_id\n");
		goto cleanup;
	}

#ifdef VERBOSE_PRINT
	printf("Received t_id from client:\n");
	print_buffer((uint8_t *) &t_id2, sizeof(t_id2));
#endif

	// Receive the REPORT of the other party from the truce_server

	if (!truce_client_recv_enclave_record(t_id2, t_rec2)) {
		cout<<(failure+"Failed to receive truce record from truce server\n");
		goto cleanup;
	}
	cout<<(success+"Received enclave record from TruCE server\n");

	if (!truce_client_extract_quote_from_record(
				t_rec2,
				quote2)) {
		cout<<(failure+"Failed to extract quote from record\n");
		goto cleanup;
	}

#ifdef VERBOSE_PRINT
	printf("Extracted the quote from the enclave record.\n");
#endif

	memcpy((void *) &expected_mrenclave2, (void *) &quote2.report_body.mr_enclave, sizeof(sgx_measurement_t));
	memcpy((void *) &expected_mrsigner2, (void *) &quote2.report_body.mr_signer, sizeof(sgx_measurement_t));

#ifdef VERBOSE_PRINT
	printf("MRSIGNER and MRENCLAVE values extracted.\n");
	print_report_body(1, &quote2.report_body);
#endif

	if (!truce_client_verify_enclave_record(
				enclave_id, status,
				t_id2, sizeof(t_id2),
				t_rec2, sizeof(t_rec2),
				expected_mrenclave2, sizeof(expected_mrenclave2),
				expected_mrsigner2, sizeof(expected_mrsigner2),
				0, 1)) {

		cout<<(failure+"Failed to verify enclave's record\n");
		goto cleanup;
	}
	if(!verify_measurement_values(quote2, 2)){
		cout<<(failure+"MEASUREMENT ERROR: Measurement values do not match\n");
		goto cleanup;
	}

	cout<<(success+"SUCCESS: Successfully verified enclave's record inside the server's enclave.\n");

	cout<<"AES GCM Addition: Calling AES GCM sequence with P1\n";
	mac_key_sequence(connfd, 0, 0, enclave_id);

	cout<<"AES GCM Addition: Calling AES GCM sequence with P2\n";
	mac_key_sequence(connfd2, 0, 1, enclave_id);



cleanup:

	if (sockfd >= 0) {
		close(sockfd);
	}

	// Enclave will be destroyed after the MPC protocol finishes.

	return ret;
}

// Party 1
int attest_main_bob(sgx_enclave_id_t &enclave_id)
{
	sgx_status_t ret = SGX_SUCCESS;
	int listenfd = -1;
	int listenfd2 = -1;

	int connfd = -1;
	int connfd2 = -1;

	sgx_status_t status = SGX_SUCCESS;

	FILE* OUTPUT = stdout;

	char* sp_address = truce_server_address;

	truce_config_t t_config;
	t_config.truce_server_address = sp_address;

	truce_session_t t_session;
	if (!truce_session(enclave_id, t_config, t_session)) {
		cout<<(failure+"Failed to create truce_session.\n");
		return 1;
	}
	cout<<(success+"Successfully created truce_session.\n");
#ifdef VERBOSE_PRINT
	printf("Received t_id:\n");
	print_buffer((uint8_t *) &t_session.truce_id, sizeof(t_session.truce_id));
#endif

	//-----------------------connection to party0-------------

	// Now, the server will receive the client's report from TruCE server and IAS.
	const char* agent_address = party0_address;
	const char* agent_address2 = party2_address;

	truce_id_t t_id = {{0}};
	truce_id_t t_id2 = {{0}};

	truce_record_t t_rec;
	truce_record_t t_rec2;

	sgx_measurement_t expected_mrenclave = {{0}}; // Should be the real value
	sgx_measurement_t expected_mrenclave2 = {{0}};

	sgx_measurement_t expected_mrsigner = {{0}}; // Should be the real value
	sgx_measurement_t expected_mrsigner2 = {{0}};

	sgx_quote_t quote = {0};
	sgx_quote_t quote2 = {0};

	int sockfd = -1;
	int sockfd2 = -1;

	// Connecting to the alice party.
	if (!inet_connect(sockfd, agent_address, port01)) {
		cout<<(failure+"Failed to connect to alice party on port %d.\n", 6000);
		return 1;
	}

	// First receive alice's t_id
	if(!read_all(sockfd, (uint8_t*)&t_id, sizeof(t_id))) {
		cout<<(failure+"Failed to read ")<<sizeof(t_id)<<(" bytes of t_id\n");
		goto cleanup;
	}
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Received t_id of the other party\n");
#endif
	// Now send bob's t_id to alice.
	if(!write_all(sockfd, t_session.truce_id, sizeof(t_session.truce_id))){
		cout<<(failure+"Failed to write ")<<sizeof(t_session.truce_id)<<(" bytes of truce_id\n");
		return 0;
	}
#ifdef VERBOSE_PRINT
	printf("Received t_id from alice party:\n");
	print_buffer((uint8_t *) &t_id, sizeof(t_id));
#endif

	// Receive the REPORT of the other party from the truce_server
	truce_client_init(truce_server_address);

	if (!truce_client_recv_enclave_record(t_id, t_rec)) {
		cout<<(failure+"Failed to receive truce record from truce server\n");
		goto cleanup;
	}

	cout<<(success+"Received enclave record from TruCE server\n");

	if (!truce_client_extract_quote_from_record(
				t_rec,
				quote)) {
		cout<<(failure+"Failed to extract quote from record\n");
		goto cleanup;
	}
#ifdef VERBOSE_PRINT
	printf("Extracted the quote from the enclave record.\n");
#endif
	memcpy((void *) &expected_mrenclave, (void *) &quote.report_body.mr_enclave, sizeof(sgx_measurement_t));
	memcpy((void *) &expected_mrsigner, (void *) &quote.report_body.mr_signer, sizeof(sgx_measurement_t));
#ifdef VERBOSE_PRINT
	print_report_body(1, &quote.report_body);
	printf("MRSIGNER and MRENCLAVE values extracted.\n");
#endif

	if (!truce_client_verify_enclave_record(
				enclave_id, status,
				t_id, sizeof(t_id),
				t_rec, sizeof(t_rec),
				expected_mrenclave, sizeof(expected_mrenclave),
				expected_mrsigner, sizeof(expected_mrsigner),
				1, 0)) {

		cout<<(failure+"Failed to verify enclave's record\n");
		goto cleanup;
	}

	if(!verify_measurement_values(quote, 0)){
		cout<<(failure+"MEASUREMENT ERROR: Measurement values do not match\n");
		goto cleanup;
	}

	cout<<(success+"Successfully verified enclave's record inside the server's enclave.\n");

	//---------------------connection party 2-------------
	// Creating a listening socket. Waiting for clients connections.
	if (!inet_listen(listenfd2, port12)) {
		cout<<(failure+"Failed to listen on port %d.\n", 0);
		return 1;
	}


	if (!inet_accept(connfd2, listenfd2)) {
		cout<<(failure+"inet_accept has failed.\n");
		return 1;
	}

	// Party should send its truce ID to the other party
	if (!write_all(connfd2, t_session.truce_id, sizeof(t_session.truce_id))) {
		cout<<(failure+"failed to write ")<<sizeof(t_session.truce_id)<<(" bytes of truce_id.\n");
		return 1;
	}

	// Party received the truce id of the other party
	if (!read_all(connfd2, (uint8_t *) &t_id2, sizeof(t_id2))) {
		cout<<(failure+"Failed to read ")<<sizeof(t_id2)<<(" bytes of t_id\n");
		goto cleanup;
	}

#ifdef VERBOSE_PRINT
	printf("Received t_id from client:\n");
	print_buffer((uint8_t *) &t_id2, sizeof(t_id2));
#endif

	// Receive the REPORT of the other party from the truce_server

	if (!truce_client_recv_enclave_record(t_id2, t_rec2)) {
		cout<<(failure+"Failed to receive truce record from truce server\n");
		goto cleanup;
	}
	cout<<(success+"Received enclave record from TruCE server\n");

	if (!truce_client_extract_quote_from_record(
				t_rec2,
				quote2)) {
		cout<<(failure+"Failed to extract quote from record\n");
		goto cleanup;
	}

#ifdef VERBOSE_PRINT
	printf("Extracted the quote from the enclave record.\n");
#endif

	memcpy((void *) &expected_mrenclave2, (void *) &quote2.report_body.mr_enclave, sizeof(sgx_measurement_t));
	memcpy((void *) &expected_mrsigner2, (void *) &quote2.report_body.mr_signer, sizeof(sgx_measurement_t));

#ifdef VERBOSE_PRINT
	printf("MRSIGNER and MRENCLAVE values extracted.\n");
	print_report_body(1, &quote2.report_body);
#endif

	if (!truce_client_verify_enclave_record(
				enclave_id, status,
				t_id2, sizeof(t_id2),
				t_rec2, sizeof(t_rec2),
				expected_mrenclave2, sizeof(expected_mrenclave2),
				expected_mrsigner2, sizeof(expected_mrsigner2),
				1, 1)) {

		cout<<(failure+"Failed to verify enclave's record\n");
		goto cleanup;
	}
	if(!verify_measurement_values(quote2, 2)){
		cout<<(failure+"MEASUREMENT ERROR: Measurement values do not match\n");
		goto cleanup;
	}

	cout<<(success+"SUCCESS: Successfully verified enclave's record inside the server's enclave.\n");

	cout<<"AES GCM Addition: Calling AES GCM sequence with P0\n";
	mac_key_sequence(sockfd, 1, 0, enclave_id);

	cout<<"AES GCM Addition: Calling AES GCM sequence with P2\n";
	mac_key_sequence(connfd2, 1, 1, enclave_id);

cleanup:

	if (sockfd >= 0) {
		close(sockfd);
	}

	// Enclave will be destroyed after the MPC protocol finishes

	return ret;

}

int attest_main_charlie(sgx_enclave_id_t &enclave_id)
{
	sgx_status_t ret = SGX_SUCCESS;
	int listenfd = -1;
	int listenfd2 = -1;

	int connfd = -1;
	int connfd2 = -1;

	sgx_status_t status = SGX_SUCCESS;

	FILE* OUTPUT = stdout;

	char* sp_address = truce_server_address;

	truce_config_t t_config;
	t_config.truce_server_address = sp_address;

	truce_session_t t_session;
	if (!truce_session(enclave_id, t_config, t_session)) {
		cout<<(failure+"Failed to create truce_session.\n");
		return 1;
	}
	cout<<(success+"Successfully created truce_session.\n");
#ifdef VERBOSE_PRINT
	printf("Received t_id:\n");
	print_buffer((uint8_t *) &t_session.truce_id, sizeof(t_session.truce_id));
#endif

	//---------------connection party0-------------------

	// Now, the server will receive the client's report from TruCE server and IAS.
	const char* agent_address = party0_address;
	const char* agent_address2 = party1_address;

	truce_id_t t_id = {{0}};
	truce_id_t t_id2 = {{0}};

	truce_record_t t_rec;
	truce_record_t t_rec2;

	sgx_measurement_t expected_mrenclave = {{0}}; // Should be the real value
	sgx_measurement_t expected_mrenclave2 = {{0}};

	sgx_measurement_t expected_mrsigner = {{0}}; // Should be the real value
	sgx_measurement_t expected_mrsigner2 = {{0}};

	sgx_quote_t quote = {0};
	sgx_quote_t quote2 = {0};

	int sockfd = -1;
	int sockfd2 = -1;

	// Connecting to the alice party.
	if (!inet_connect(sockfd, agent_address, port02)) {
		cout<<(failure+"Failed to connect to alice party on port %d.\n", 6000);
		return 1;
	}

	// First receive alice's t_id
	if(!read_all(sockfd, (uint8_t*)&t_id, sizeof(t_id))) {
		cout<<(failure+"Failed to read ")<<sizeof(t_id)<<(" bytes of t_id\n");
		goto cleanup;
	}
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Received t_id of the other party\n");
#endif
	// Now send bob's t_id to alice.
	if(!write_all(sockfd, t_session.truce_id, sizeof(t_session.truce_id))){
		cout<<(failure+"Failed to write ")<<sizeof(t_session.truce_id)<<(" bytes of truce_id\n");
		return 0;
	}
#ifdef VERBOSE_PRINT
	printf("Received t_id from alice party:\n");
	print_buffer((uint8_t *) &t_id, sizeof(t_id));
#endif

	// Receive the REPORT of the other party from the truce_server
	truce_client_init(truce_server_address);

	if (!truce_client_recv_enclave_record(t_id, t_rec)) {
		cout<<(failure+"Failed to receive truce record from truce server\n");
		goto cleanup;
	}

	cout<<(success+"Received enclave record from TruCE server\n");

	if (!truce_client_extract_quote_from_record(
				t_rec,
				quote)) {
		cout<<(failure+"Failed to extract quote from record\n");
		goto cleanup;
	}
#ifdef VERBOSE_PRINT
	printf("Extracted the quote from the enclave record.\n");
#endif
	memcpy((void *) &expected_mrenclave, (void *) &quote.report_body.mr_enclave, sizeof(sgx_measurement_t));
	memcpy((void *) &expected_mrsigner, (void *) &quote.report_body.mr_signer, sizeof(sgx_measurement_t));
#ifdef VERBOSE_PRINT
	print_report_body(1, &quote.report_body);
	printf("MRSIGNER and MRENCLAVE values extracted.\n");
#endif

	if (!truce_client_verify_enclave_record(
				enclave_id, status,
				t_id, sizeof(t_id),
				t_rec, sizeof(t_rec),
				expected_mrenclave, sizeof(expected_mrenclave),
				expected_mrsigner, sizeof(expected_mrsigner),
				2, 0)) {

		cout<<(failure+"Failed to verify enclave's record\n");
		goto cleanup;
	}

	if(!verify_measurement_values(quote, 0)){
		cout<<(failure+"MEASUREMENT ERROR: Measurement values do not match\n");
		goto cleanup;
	}

	cout<<(success+"Successfully verified enclave's record inside the server's enclave.\n");

	// ----------------connection party 1----------------
	//
	// Connecting to the bob party.
	if (!inet_connect(sockfd2, agent_address2, port12)) {
		cout<<(failure+"Failed to connect to alice party on port %d.\n", 6000);
		return 1;
	}

	// First receive bob's t_id
	if(!read_all(sockfd2, (uint8_t*)&t_id2, sizeof(t_id2))) {
		cout<<(failure+"Failed to read ")<<sizeof(t_id2)<<(" bytes of t_id\n");
		goto cleanup;
	}
#ifdef VERBOSE_PRINT
	fprintf(OUTPUT, "Received t_id of the other party\n");
#endif
	// Now send charlie's t_id to alice.
	if(!write_all(sockfd2, t_session.truce_id, sizeof(t_session.truce_id))){
		cout<<(failure+"Failed to write ")<<sizeof(t_session.truce_id)<<(" bytes of truce_id\n");
		return 0;
	}
#ifdef VERBOSE_PRINT
	printf("Received t_id from bob party:\n");
	print_buffer((uint8_t *) &t_id2, sizeof(t_id2));
#endif

	// Receive the REPORT of the other party from the truce_server
	truce_client_init(truce_server_address);

	if (!truce_client_recv_enclave_record(t_id2, t_rec2)) {
		cout<<(failure+"Failed to receive truce record from truce server\n");
		goto cleanup;
	}

	cout<<(success+"Received enclave record from TruCE server\n");

	if (!truce_client_extract_quote_from_record(
				t_rec2,
				quote2)) {
		cout<<(failure+"Failed to extract quote from record\n");
		goto cleanup;
	}
#ifdef VERBOSE_PRINT
	printf("Extracted the quote from the enclave record.\n");
#endif
	memcpy((void *) &expected_mrenclave2, (void *) &quote2.report_body.mr_enclave, sizeof(sgx_measurement_t));
	memcpy((void *) &expected_mrsigner2, (void *) &quote2.report_body.mr_signer, sizeof(sgx_measurement_t));
#ifdef VERBOSE_PRINT
	print_report_body(1, &quote2.report_body);
	printf("MRSIGNER and MRENCLAVE values extracted.\n");
#endif

	if (!truce_client_verify_enclave_record(
				enclave_id, status,
				t_id2, sizeof(t_id2),
				t_rec2, sizeof(t_rec2),
				expected_mrenclave2, sizeof(expected_mrenclave2),
				expected_mrsigner2, sizeof(expected_mrsigner2),
				2, 1)) {

		cout<<(failure+"Failed to verify enclave's record\n");
		goto cleanup;
	}

	if(!verify_measurement_values(quote2, 1 /*1 for BOB*/)){
		cout<<(failure+"MEASUREMENT ERROR: Measurement values do not match\n");
		goto cleanup;
	}

	cout<<(success+"Successfully verified enclave's record inside the server's enclave.\n");

	cout<<"AES GCM Addition: Calling AES GCM sequence with P0\n";
	mac_key_sequence(sockfd, 2, 0, enclave_id);

	cout<<"AES GCM Addition: Calling AES GCM sequence with P1\n";
	mac_key_sequence(sockfd2, 2, 1, enclave_id);

cleanup:

	if (sockfd >= 0) {
		close(sockfd);
	}

	// Enclave will be destroyed after the MPC protocol finishes

	return ret;

}

