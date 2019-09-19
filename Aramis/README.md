crypTFlow::Aramis: Efficient Malicious Secure Neural Network Inference.

NOTE: The project comes with sample testing public key (not white-listed with Intel) that has already been registered with Intel for development use. The project runs in Debug Hardware mode by default (this can be changed to Pre-Release Hardware mode by changing flags in Makefile). This can be changed to Pre-Release Hardware mode as well for full enclave security, but the private key is acessible openly to the OS and this can be used by the OS to run malicious enclaves by itself. In Debug HW and Pre-Release HW modes, enclave security is not fully enforced. To get a fully secure enclave, Release mode must be used, which requires a multi-step license registering process along with a more complex enclave signing procedure. Release mode should only be used when you want to commercially deploy your enclave application with the full security guarantees that SGX enclaves provide. To run an enclave with Release mode, it is mandatory to get your enclave signing key white-listed with Intel. Note that white-listed keys will not be allowed to run in Debug or Pre-Release mode. Once white-listed, this key can be used to sign production Release enclaves. The only method that allows singing of production Release enclaves is the 2-step signing process which requires the signing key to be stored in a separate Hardware Security Module or another enclave.

NOTE: Pre-Release mode doesn't work yet. There is some issue with estabilishing socket connections with this mode. Use Debug mode for the time being.

For bugs, issues or clarifications, contact at mayankrathee.japan@gmail.com.

* Makefile
Edit the variables SGX_SDK and SSL_SGX in the Makefile with the installation locations of your local SGX SDK and SGX SSL. 
