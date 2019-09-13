Packages required 
(Ubuntu): 
* sudo apt-get install libssl-dev
* sudo apt-get install g++
* sudo apt-get install make

(Mac, not working):
* brew install libssl
* cd /usr/local/include
* ln -s ../opt/openssl/include/openssl . 
Also, use the second build command for Mac to be safe. Read the Caveats part of brew install instructions, since Apple has deprecated the use of OpenSSL.

To-Do: AESObjects are used as global variables. They need to be passed through all the functions involved.  

Note: libmiracl.a is compiled locally, if it does not work try the libmiracl_old.a or download the source files from https://github.com/miracl/MIRACL.git and compile miracl.a yourself (and rename to libmriacl.a when copying into this repo)

Note: when extracting bits out of \_\_m128i, val[0] corresponds to the LSB. The set functions of \_\_m128i take the first argument as val[0].

Note: If porthosSecretType != uint64_t, porthosSecretType multiplication function won't work.

Note: Matrix multiplication assembly code only works for Intel C/C++ compiler. The non-assembly code has correctness issues when both the multiplicands are large uint64_t's

For bugs contact at mayankrathee.japan@gmail.com or nishant.kr10@gmail.com

