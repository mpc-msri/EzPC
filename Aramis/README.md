#Aramis: Efficient Malicious Secure Neural Network Inference.

#Introduction
This directory contains code for Aramis - a novel technique that uses hardware with integrity guarantees to convert any semi-honest MPC protocol into an MPC protocol that provides malicious security. 

#Setup
(Ubuntu):
* sudo apt-get install libssl-dev
* sudo apt-get install g++
* sudo apt-get install make
* git clone https://github.com/mayank0403/SGX-Installation-Scripts (Follow the instructions there to install SGX SDK, SGX SSL and related libraries on your system)
* Go to Makefiles in `party0/`, `party1/`, `party2/` and `service-provider` and update the variables `SGX_SDK` and `SSL_SGX` with the locations where SGX SDK and SGX SLL are installed on your system.
* Make sure that you have an SGX compatible machine and check that sgx support is enabled in BIOS.

#Running Aramis
(Ubuntu):
* Open up 4 terminal sessions for party0, party1, party2 and service provider and enter their respective directories on each terminal window.
* Run `make clean && make` in each window and this should compile Aramis.
* Now, to run, in the window for party0, run `./aramis < <Path to input image file>`, on windows for party1 and party2, run `./aramis` and on the service-provider window, run `./truce-server`.

#Some Notes Regarding SGX Configuration and Aramis
* The project comes with sample testing public key (not white-listed with Intel) that has already been registered with Intel for development use and also comes with hardcoded subscription keys registered with Intel. 
* The project runs in Debug Hardware mode by default (this can be changed to Pre-Release Hardware mode by changing flags in Makefile). This can be changed to Pre-Release Hardware mode as well, but the private key is acessible openly to the OS and this can be used by the OS to run malicious enclaves by itself. In Debug HW and Pre-Release HW modes, enclave security is not fully enforced. To get a fully secure enclave, Release mode must be used, which requires a multi-step license registering process along with a more complex enclave signing procedure. Release mode should only be used when you want to commercially deploy your enclave application with the full security guarantees that SGX enclaves provide. To run an enclave with Release mode, it is mandatory to get your enclave signing key white-listed with Intel. Note that white-listed keys will not be allowed to run in Debug or Pre-Release mode. Once white-listed, this key can be used to sign production Release enclaves. The only method that allows singing of production Release enclaves is the 2-step signing process which requires the signing key to be stored in a separate Hardware Security Module or another enclave.
* Pre-Release mode doesn't work yet. There is some issue with estabilishing socket connections with this mode. Use Debug mode for the time being.

For bugs, issues or clarifications, contact at mayankrathee.japan@gmail.com.

