/*
Authors: Mayank Rathee.
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "mac_key_utils_app.h"

bool mac_key_sequence(int connfd, int pnum, int sequence, sgx_enclave_id_t &enclave_id){
	if(pnum == 0){
		// Send the MAC key signed and encrypted.
		// First make an ecall to get RSA_cipher_size
		sgx_status_t status;
		uint32_t* RSA_cipher_size;
		RSA_cipher_size = (uint32_t*)malloc(sizeof(uint32_t));
		*RSA_cipher_size = 1500;
		// Make an ECALL to fill RSA_cipher_size with
		// correct value
		ECALL_get_RSA_cipher_size(enclave_id, &status, RSA_cipher_size, 0, sequence);
		uint8_t* aeskeysignature;
		uint8_t* aeskeyencrypted;
		aeskeysignature = (uint8_t*)malloc(ECDSA_SIG_SIZE);
		aeskeyencrypted = (uint8_t*)malloc(*RSA_cipher_size);

		sgx_status_t ret = encrypt_and_sign_aes_key(enclave_id, &status, aeskeyencrypted, aeskeysignature, *RSA_cipher_size, 0, sequence);

		// Send the key to the other party
		if(!write_all(connfd, (uint8_t*)RSA_cipher_size, sizeof(uint32_t))){
			printf("ERROR: Failed to send RSA_cipher_size\n");
			close(connfd);
			return false;
		}
		if(!write_all(connfd, aeskeyencrypted, *RSA_cipher_size)){
			printf("ERROR: Failed to send encrypted aes key\n");
			close(connfd);
			return false;
		}
		if(!write_all(connfd, aeskeysignature, ECDSA_SIG_SIZE)){
			printf("ERROR: Failed to send signature of aes key\n");
			close(connfd);
			return false;
		}
		printf("SUCCESS: Sent the AES GCM key, signed and encrypted, to the other party\n");
		return true;


	}
	else if(pnum ==1){
		if(sequence == 0){
			// Receive the signed and encrypted MAC key
			// Verify the signature on the key
			// before accepting it.
			sgx_status_t status;
			uint32_t* aeskeysize;
			aeskeysize = (uint32_t*)malloc(sizeof(uint32_t));
			uint8_t* aeskeyencrypted;
			uint8_t* aeskeysignature;
			//Receive the aes key and signature
			if(!read_all(connfd, (uint8_t*)aeskeysize, sizeof(uint32_t))){
				printf("ERROR: Failed to receieve aeskeysize\n");
				close(connfd);
				return false;
			}
			aeskeyencrypted = (uint8_t*)malloc(*aeskeysize);
			if(!read_all(connfd, aeskeyencrypted, *aeskeysize)){
				printf("ERROR: Failed to receive encrypted aes key\n");
				close(connfd);
				return false;
			}
			aeskeysignature = (uint8_t*)malloc(ECDSA_SIG_SIZE);
			if(!read_all(connfd, aeskeysignature, ECDSA_SIG_SIZE)){
				printf("ERROR: Failed to receive aes key signature\n");
				close(connfd);
				return false;
			}
			decrypt_verify_aes_key_and_save(enclave_id, &status, aeskeyencrypted, *aeskeysize, aeskeysignature, 1, sequence);
			printf("SUCCESS: Received and saved the aes key\n");
			return true;

		}
		else{
			// Send the MAC key signed and encrypted.
			// First make an ecall to get RSA_cipher_size
			sgx_status_t status;
			uint32_t* RSA_cipher_size;
			RSA_cipher_size = (uint32_t*)malloc(sizeof(uint32_t));
			*RSA_cipher_size = 1500;
			// Make an ECALL to fill RSA_cipher_size with
			// correct value
			ECALL_get_RSA_cipher_size(enclave_id, &status, RSA_cipher_size, 1, sequence);
			uint8_t* aeskeysignature;
			uint8_t* aeskeyencrypted;
			aeskeysignature = (uint8_t*)malloc(ECDSA_SIG_SIZE);
			aeskeyencrypted = (uint8_t*)malloc(*RSA_cipher_size);

			sgx_status_t ret = encrypt_and_sign_aes_key(enclave_id, &status, aeskeyencrypted, aeskeysignature, *RSA_cipher_size, 1, sequence);

			// Send the key to the other party
			if(!write_all(connfd, (uint8_t*)RSA_cipher_size, sizeof(uint32_t))){
				printf("ERROR: Failed to send RSA_cipher_size\n");
				close(connfd);
				return false;
			}
			if(!write_all(connfd, aeskeyencrypted, *RSA_cipher_size)){
				printf("ERROR: Failed to send encrypted aes key\n");
				close(connfd);
				return false;
			}
			if(!write_all(connfd, aeskeysignature, ECDSA_SIG_SIZE)){
				printf("ERROR: Failed to send signature of aes key\n");
				close(connfd);
				return false;
			}
			printf("SUCCESS: Sent the AES GCM key, signed and encrypted, to the other party\n");
			return true;

		}

	}
	else{
		//Charlie
		// Receive the signed and encrypted MAC key
		// Verify the signature on the key
		// before accepting it.
		sgx_status_t status;
		uint32_t* aeskeysize;
		aeskeysize = (uint32_t*)malloc(sizeof(uint32_t));
		uint8_t* aeskeyencrypted;
		uint8_t* aeskeysignature;
		//Receive the aes key and signature
		if(!read_all(connfd, (uint8_t*)aeskeysize, sizeof(uint32_t))){
			printf("ERROR: Failed to receieve aeskeysize\n");
			close(connfd);
			return false;
		}
		aeskeyencrypted = (uint8_t*)malloc(*aeskeysize);
		if(!read_all(connfd, aeskeyencrypted, *aeskeysize)){
			printf("ERROR: Failed to receive encrypted aes key\n");
			close(connfd);
			return false;
		}
		aeskeysignature = (uint8_t*)malloc(ECDSA_SIG_SIZE);
		if(!read_all(connfd, aeskeysignature, ECDSA_SIG_SIZE)){
			printf("ERROR: Failed to receive aes key signature\n");
			close(connfd);
			return false;
		}
		decrypt_verify_aes_key_and_save(enclave_id, &status, aeskeyencrypted, *aeskeysize, aeskeysignature, 2, sequence);
		printf("SUCCESS: Received and saved the aes key\n");
		return true;

	}
}
