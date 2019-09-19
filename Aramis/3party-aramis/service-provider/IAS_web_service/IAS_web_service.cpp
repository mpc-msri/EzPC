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

#include <stdio.h>
#include <string.h>
#include <string>
#include <assert.h>
#include <arpa/inet.h>
#include "IAS_web_service.h"
#include "base64.h"

using namespace std;


static int REQUEST_ID_MAX_LEN = 32;
static int IASREPORT_SIGNATURE_MAX_LEN = 2000;
static int IASREPORT_SIGNING_CERT_MAX_LEN = 4000;


size_t ias_report_body_handler( void *ptr, size_t size, size_t nmemb, void *userdata ) {
	size_t realsize = size * nmemb;
	ias_report_container_t *ias_report_container = ( ias_report_container_t * ) userdata;
	ias_report_container->p_report = (char *) realloc(ias_report_container->p_report,
			ias_report_container->size + realsize + 1);

	if (NULL == ias_report_container->p_report) {
		printf("ERROR: Unable to allocate extra memory for report\n");
		return 0;
	}

	memcpy( &( ias_report_container->p_report[ias_report_container->size]), ptr, realsize );
	ias_report_container->size += realsize;
	ias_report_container->p_report[ias_report_container->size] = 0;

	return realsize;
}


size_t ias_report_header_parser(void *ptr, size_t size, size_t nmemb, void *userdata) {
	int parsed_fields = 0;
	int report_status = 0;
	int content_length = 0;
	int ret = size * nmemb;

	char *tmp_str = (char*) calloc(size+1, nmemb);
	if (NULL == tmp_str) {
		printf("ERROR: calloc(%lu,%lu) has failed\n", size+1, nmemb);
		return 0;
	}

	memcpy(tmp_str, ptr, size * nmemb);
	parsed_fields = sscanf(tmp_str, "HTTP/1.1 %d", &report_status );

	if (parsed_fields == 1) {
		((ias_report_header_t *) userdata)->report_status = report_status;
		return ret;
	}

	parsed_fields = sscanf(tmp_str, "content-length: %d", &content_length );
	if (parsed_fields == 1) {
		((ias_report_header_t *) userdata)->content_length = content_length;
		return ret;
	}

	char *p_request_id = (char*) calloc(1, REQUEST_ID_MAX_LEN);
	parsed_fields = sscanf(tmp_str, "request-id: %s", p_request_id );
	if (parsed_fields == 1) {
		char* request_id_str( p_request_id );
		( ( ias_report_header_t * ) userdata )->request_id = request_id_str;
		return ret;
	}

	char *p_iasreport_signature = (char *) calloc(1, IASREPORT_SIGNATURE_MAX_LEN);
	parsed_fields = sscanf(tmp_str, "x-iasreport-signature: %s", p_iasreport_signature );
	if (parsed_fields == 1) {
		char* iasreport_signature_str( p_iasreport_signature );
		( ( ias_report_header_t * ) userdata )->iasreport_signature = iasreport_signature_str;
		return ret;
	}

	char *p_iasreport_signing_certificate = (char *) calloc(1, IASREPORT_SIGNING_CERT_MAX_LEN);
	parsed_fields = sscanf(tmp_str, "x-iasreport-signing-certificate: %s", p_iasreport_signing_certificate );
	if (parsed_fields == 1) {
		char* p_iasreport_signing_certificate_str ( p_iasreport_signing_certificate );
		( ( ias_report_header_t * ) userdata )->iasreport_signing_certificate = p_iasreport_signing_certificate_str;
		return ret;
	}

	return ret;
}




bool init_ias_web_service(ias_web_service_t &ias_web_service,
		const std::string &client_cert_path,
		const std::string &ias_url) {

	ias_web_service.url =  ias_url;
	curl_global_init(CURL_GLOBAL_DEFAULT);
	printf("This is the certificate path this is being used here - %s\n", client_cert_path.c_str());
	ias_web_service.curl = curl_easy_init();

	if (!ias_web_service.curl) {
		printf("ERROR: Curl init error\n");
		return false;
	}

	printf("    Curl initialized successfully\n");
	//curl_easy_setopt( curl, CURLOPT_VERBOSE, 1L );
	curl_easy_setopt( ias_web_service.curl, CURLOPT_SSLCERTTYPE, "PEM");
	curl_easy_setopt( ias_web_service.curl, CURLOPT_SSLCERT, client_cert_path.c_str());
	curl_easy_setopt( ias_web_service.curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
	curl_easy_setopt( ias_web_service.curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
	curl_easy_setopt( ias_web_service.curl, CURLOPT_NOPROGRESS, 1L);


	return true;
}

bool send_to_ias_sig_rl(CURL *curl,
		string url,
		IAS type,
		string payload,
		struct curl_slist *headers,
		ias_report_container_t *ias_report_container,
		ias_report_header_t *report_header) {

	CURLcode res = CURLE_OK;

	string subscription_key_p = "0bfece7f85cc4a3abea5b1699831cb6c";
	string subscription_key_s = "5655dcf103354bfd8e4fe07d53678108";

	string subscriptionKeyHeader = "Ocp-Apim-Subscription-Key: ";
	subscriptionKeyHeader.append(subscription_key_s);
	
	if((headers = curl_slist_append(headers, subscriptionKeyHeader.c_str())) == NULL)
		return 0;


	if (headers) {
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		//curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
	}

	ias_report_container->p_report = (char*) malloc(1);
	if (NULL == ias_report_container->p_report) {
		printf("ERROR: failed to allocate 1 byte when sending to IAS\n");
		return false;
	}
	ias_report_container->size = 0;

	curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ias_report_header_parser);
	curl_easy_setopt(curl, CURLOPT_HEADERDATA, report_header);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ias_report_body_handler);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, ias_report_container);
	
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

	res = curl_easy_perform(curl);
	if (res != 0) {
		printf("ERROR: curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		free(ias_report_container->p_report);
		ias_report_container->p_report = NULL;
		return false;
	}

	return true;
}


bool send_to_ias(CURL *curl,
		string url,
		IAS type,
		string payload,
		struct curl_slist *headers,
		ias_report_container_t *ias_report_container,
		ias_report_header_t *report_header) {

	CURLcode res = CURLE_OK;

	
	string subscription_key_p = "0bfece7f85cc4a3abea5b1699831cb6c";
	string subscription_key_s = "5655dcf103354bfd8e4fe07d53678108";

	string subscriptionKeyHeader = "Ocp-Apim-Subscription-Key: ";
	subscriptionKeyHeader.append(subscription_key_s);
	
	if((headers = curl_slist_append(headers, subscriptionKeyHeader.c_str())) == NULL)
		return 0;

	if (headers) {
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
	}

	ias_report_container->p_report = (char*) malloc(1);
	if (NULL == ias_report_container->p_report) {
		printf("ERROR: failed to allocate 1 byte when sending to IAS\n");
		return false;
	}
	ias_report_container->size = 0;

	curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, ias_report_header_parser);
	curl_easy_setopt(curl, CURLOPT_HEADERDATA, report_header);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, ias_report_body_handler);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, ias_report_container);
	
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

	res = curl_easy_perform(curl);
	if (res != 0) {
		printf("ERROR: curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		free(ias_report_container->p_report);
		ias_report_container->p_report = NULL;
		return false;
	}

	return true;
}

int new_ias_spec_get_sig_rl(string url,
		string sresponse)
{
	string subscription_key_p = "0bfece7f85cc4a3abea5b1699831cb6c";
	string subscription_key_s = "5655dcf103354bfd8e4fe07d53678108";

	string subscriptionKeyHeader = "Ocp-Apim-Subscription-Key: ";
	subscriptionKeyHeader.append(subscription_key_p);

	CURL *curl;
	curl = curl_easy_init();
	curl_slist *slist= NULL;
	if((slist = curl_slist_append(slist, subscriptionKeyHeader.c_str())) == NULL)
		return 0;

	if ( curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist)
			!= CURLE_OK ) return 0;

	if ( curl_easy_setopt(curl, CURLOPT_URL, url.c_str()) != CURLE_OK )
		return 0;

	if ( curl_easy_perform(curl) != 0 ) {
		return 0;
	}
	return 1;
}

bool get_sig_rl(const ias_web_service_t &ias_web_service,
		const uint8_t (&epid_group_id)[4], //in
		uint8_t *&p_sig_rl, //out
		uint32_t &sig_rl_size //out
	       )
{
	string gid_str = "";
	char tmp_str[10] = {0};

	/* The gid should be encoded as a Big Endian integer. */
	uint32_t gid_uint = htonl(*((uint32_t *)&epid_group_id));
	uint8_t *gid_byte_arr = (uint8_t *) &gid_uint;
	for (int i=0; i<sizeof(uint32_t); ++i) {
		if (gid_byte_arr[i] < 0x10) {
			sprintf(tmp_str, "0%x", gid_byte_arr[i]);
		}
		else {
			sprintf(tmp_str, "%x", gid_byte_arr[i]);
		}
		gid_str += tmp_str;
	}

	ias_report_container_t ias_report_container;
	ias_report_header_t report_header;

	string url = ias_web_service.url + "sigrl/" + gid_str;
	string resp = "";

	if (!send_to_ias_sig_rl(ias_web_service.curl,
				url,
				IAS::sigrl,
				"",
				NULL,
				&ias_report_container,
				&report_header)) {

		printf("ERROR: send_to_ias has failed\n");
		return false;
	}

	/*printf("\tResponse status is: %d\n" , report_header.report_status);
	  printf("\tContent-Length: %d\n", report_header.content_length);*/

	if (report_header.report_status != 200) {
		printf("ERROR: Failed to retrieve SigRL from IAS. report status = %d\n", report_header.report_status);
		return false;
	}
	if (report_header.content_length < 0) {
		printf("ERROR: Failed to retrieve SigRL from IAS. Bad content length = %d\n", report_header.content_length);
		return false;
	}
	if (report_header.content_length == 0) {
		sig_rl_size = 0;
		return true;
	}

	string report(ias_report_container.p_report);
	string sigrl_str = base64_decode(report);
	sig_rl_size = sigrl_str.size();
	memcpy(p_sig_rl, sigrl_str.c_str(), sig_rl_size);

	return true;
}



string create_json_for_ias(
		const uint8_t *p_quote,
		uint32_t quote_size,
		const uint8_t *p_pseManifest,
		uint32_t pseManifest_size,
		const uint8_t *p_nonce,
		uint32_t nonce_size) {
	Json::Value request;

	request["isvEnclaveQuote"] = base64_encode(p_quote, quote_size);
	if (p_pseManifest != NULL) {
		request["pseManifest"] = base64_encode(p_pseManifest, pseManifest_size);
	}
	if (p_nonce != NULL) {
		request["nonce"] = string((char *) p_nonce, nonce_size);
	}

	Json::FastWriter fastWriter;
	string output = fastWriter.write(request);

	return output;
}


bool get_ias_report(const ias_web_service_t &ias_web_service,
		const uint8_t *p_quote, //in
		uint32_t quote_size,
		const uint8_t *p_pseManifest, //in (optional)
		uint32_t pseManifest_size, // in (should be 0 if pseManifest = NULL)
		const uint8_t *p_nonce, //in (optional)
		uint32_t nonce_size, // in (should be 0 if nonce = NULL)
		IAS_report &ias_report //out
		)
{
	string request_payload = create_json_for_ias(
			p_quote,
			quote_size,
			p_pseManifest,
			pseManifest_size,
			p_nonce,
			nonce_size);

	/*printf("encoded_quote = %s\n", encoded_quote.c_str());*/

	ias_report_container_t ias_report_container;
	ias_report_header_t report_header;

	struct curl_slist *headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");

	string url = ias_web_service.url + "report";
	if (!send_to_ias(ias_web_service.curl,
				url,
				IAS::report,
				request_payload,
				headers,
				&ias_report_container,
				&report_header)) {

		printf("ERROR: send_to_ias has failed\n");
		return false;
	}

	if (report_header.report_status != 200) {
		printf("Quote attestation has failed. Returned status: %d\n", report_header.report_status);
		return false;
	}

	printf("    New IAS report has been created\n");
	ias_report.report_body = ias_report_container.p_report;
	ias_report.report_signature_base64 = report_header.iasreport_signature;
	ias_report.report_cert_chain_urlsafe_pem = report_header.iasreport_signing_certificate;

	return true;
}




