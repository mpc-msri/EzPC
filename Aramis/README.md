# Aramis: Efficient Malicious Secure Neural Network Inference.

# Introduction
This directory contains code for Aramis - a novel technique that uses hardware with integrity guarantees to convert any semi-honest MPC protocol into an MPC protocol that provides malicious security. 

# Setup
(Ubuntu):
* sudo apt-get install libssl-dev
* sudo apt-get install g++
* sudo apt-get install make
* sudo apt-get install libjsoncpp-dev
* sudo apt-get install libcurl4-openssl-dev
* Download cpp-base64 from the git repository https://github.com/ReneNyffenegger/cpp-base64, and put the cpp-base64 folder under the aux_lib folder inside each of `party0/`, `party1/` and `party2/` directories.
* git clone https://github.com/mayank0403/SGX-Installation-Scripts (Follow the instructions there to install SGX SDK, SGX SSL and related libraries on your system)
* Go to Makefiles in `party0/`, `party1/`, `party2/` and `service-provider` and update the variables `SGX_SDK` and `SSL_SGX` with the locations where SGX SDK and SGX SLL are installed on your system.
* Make sure that you have an SGX compatible machine and check that sgx support is enabled in BIOS.

# Running Aramis
(Ubuntu):
* Open up 4 terminal sessions for party0, party1, party2 and service provider and enter their respective directories on each terminal window.
* Run `make clean && make CC=gcc-4.9 CXX=g++-5` in each window and this should compile Aramis.
* Now, to run, in the window for party0, run `./aramis < <Path to input image file>`, on windows for party1 and party2, run `./aramis` and on the service-provider window, run `./truce-server`.
* Note that you will need an input image file to run a NN. For this, refer to Athos and Porthos documentation on how to generate input image file. Use the path of that image file in the previous step.

# Running Aramis on Athos compiled NNs
* Go to `party0/` directory. Note that this should be your main working directory. You should make all changes here and then run `./update-all.sh` in this directory to update `party1/` and `party2/` automatically.
* Athos compiled files can be found in Porthos directory. You can take any Porthos NN file and compile it to Aramis. Suppose that you want to run ResNet50 in Aramis. Porthos (Athos compiled) code for ResNet50 resides in `Porthos/src/example_neural_nets/mainResNet50.cpp`. Open the file `Aramis/3party-aramis/party0/compile_porthos_to_aramis.py` and update the path of `porthos_file` object to `../../../Porthos/src/example_neural_nets/mainResNet50.cpp`.
* Now in the same `party0/` directory, run `./compile_to_aramis.sh`. This will compile the NN to Aramis and also automatically update `party1/` and `party2/` directories.
* After this, you can follow the instructions as mentioned in the previous section of this README. (Running Aramis section).

# Included Examples
Aramis already has the following example compiled NNs in `party0/src/example_neural_nets/` directory. To run these examples, copy the code of the respective example file (cpp file) to `party0/src/main.cpp` and overwrite it. Now in `party0/src/main.cpp` change the header from `#include "res_net_mem_opti.h"` to `#include "main.h"`. Also, in the same file, change `start_m()` to `StartComputation()` and `end_m()` to `EndComputation()`. Also, go to `party0/src/network_config.h` file and change the preprocessor flag to the network that you want to run. Finally, run `./update_all` to update code of the other 2 parties as well.
* ResNet50 and other ResNet variants.
* DenseNet121.
* SqueezeNet for ImageNet data.
* The 4 benchmarks from SecureNN paper.

# Programming in SGX
* You can debug SGX applications using a tool that comes with SGX SDK called `sgx-gdb`. When running Aramis with it, you might face a SIGILL interrupt at the beginning itself, enter `c` and continue. You will need to do this 4 times. Don't worry, this will not have any effect on your debugging. This fault comes because of absence of access to CPU_ID to SGX SSL lib.
* If you want to benchmark the peak memory usage of your SGX application, then you can use `sgx_emmt` tool. To use it, run your application with `sgx-gdb` and enter `enable sgx_emmt` before running you application.
* You might face a SIGILL or SEGFAULT error while running a big NN in Aramis. This occurs when you try to use more memory during runtime than what is specified in the file `party0/Enclave/Enclave.config.xml`. There are 2 lines there, one for `StackMaxSize` and the other for `HeapMaxSize`. To fix SIGLL or SEGFAULT, you can increase these values. Try increasing only the `HeakMaxSize` value first and it the faults are still there, then increase `StackMaxSize` slowly and try to run. These values are in Hex representation of the number of Bytes the program is expected to use during runtime. Remember to run `./update_all.sh` after any change that you make to `party0/` directory to update `party1/` and `party2/` directories automatically.

# Some Notes Regarding SGX Configuration and Aramis
* The project's Remote Attestation handling routine is totally compatible with Intel IAS API version 3, revision 5.
* The project comes with sample testing public key (not white-listed with Intel) that has already been registered with Intel for development use and also comes with hardcoded subscription keys registered with Intel. 
* The project runs in Debug Hardware mode by default (this can be changed to Pre-Release Hardware mode by changing flags in Makefile). This can be changed to Pre-Release Hardware mode as well, but the private key is acessible openly to the OS and this can be used by the OS to run malicious enclaves by itself. In Debug HW and Pre-Release HW modes, enclave security is not fully enforced. To get a fully secure enclave, Release mode must be used, which requires a multi-step license registering process along with a more complex enclave signing procedure. Release mode should only be used when you want to commercially deploy your enclave application with the full security guarantees that SGX enclaves provide. To run an enclave with Release mode, it is mandatory to get your enclave signing key white-listed with Intel. Note that white-listed keys will not be allowed to run in Debug or Pre-Release mode. Once white-listed, this key can be used to sign production Release enclaves. The only method that allows singing of production Release enclaves is the 2-step signing process which requires the signing key to be stored in a separate Hardware Security Module or another enclave.
* Pre-Release mode doesn't work yet. There is some issue with estabilishing socket connections with this mode. Use Debug mode for the time being.

# Registering with Intel with Custom Keys
* Create an Intel developer account to register custom keys. After the registration with a certificate (can be self-signed for development purposes - Look up on Google for how to generate self-signed certificates using OpenSSL), Intel will respond with a SPID, certificate and quote sign type.
* In the latest version of IAS API, Intel now requires all SigRL requests to have a header with registered subscription keys. You can get yours at https://api.portal.trustedservices.intel.com/.

# Adding Custom Intel-registered Keys to Aramis
If you have registered you own keys with Intel, then you can follow these steps to use them instead of the preincluded ones.
* Go to `service-provider/defs.h` and update your Intel assigned SPID and QUOTE_SIGN_TYPE in the file.
* Create an empty pem file with the name - "cert_and_key.pem" and put in the certificate that you used to register to Intel developer account and in a new line of this file (cert_and_key.pem) after the certificate, put the key that you used while generating the certificate. If you used openssl to generate the certificate, then the certificate.pem and key.pem files are both created in the same directory. Put this file in the `service-provider/cert_and_key.pem` overwriting the previous certificate and private key.
* Overwrite the files `party0/Enclave/Enclave_private.pem`, `party1/Enclave/Enclave_private.pem` and `party2/Enclave/Enclave_private.pem` with you self-signed private key that you created for registering with Intel. (Previous section of README, point 1).
* Go to file `service-provider/IAS_web_service/IAS_web_service.cpp` in the funtion `send_to_ias_sig_rl()` and update the values `subscription_key_p` and `subscription_key_s` with your primary and secondary subscription keys.

# External Code
This project contains/relies on some components of the following external code.
* https://github.com/ReneNyffenegger/cpp-base64 for utility cpp functions.
* https://github.com/IBM/sgx-trust-management/ for basic attestation framework.
* A very early fork of https://github.com/snwagh/securenn-public.
* `util/` directory contains code taken from SCAPI project.
* `basicSockets` files contain code writen by Roi Inbar. We have modified this code for crypTFlow.
* `tools.cpp` contains some code written by Anner Ben-Efraim and Satyanarayana. This code is majorly modified for crypTFlow.
