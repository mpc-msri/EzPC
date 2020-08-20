# Secure and Correct Inference (SCI) Library

## Introduction
This directory contains the code for the Secure and Correct Inference (SCI) library from ["CrypTFlow2: Practical 2-Party Secure Inference"](https://eprint.iacr.org/2020/1002).

## Required Packages
 - g++ (version >= 8)
 - cmake
 - make
 - libgmp-dev
 - libssl-dev  
 - SEAL 3.3.2
 - Eigen 3.3

SEAL and Eigen are included in `extern/` and are automatically compiled and installed. The other packages can be installed directly using `sudo apt-get install <package>` on Linux.

## Compilation

To compile the library:

```
mkdir build && cd build
cmake .. [-DBUILD_TESTS=ON] [-DBUILD_NETWORKS=ON]
make
// or make -j for faster compilation
```

## Running Tests & Networks

On successful compilation, the test and network binaries will be created in `build/bin/`.

Run the tests as follows to make sure everything works as intended:

`./<test> r=1 [p=port] & ./<test> r=2 [p=port]`

To run secure inference on networks:

```
./<network> r=1 [p=port] < <model_file> // Server
./<network> r=2 [ip=server_address] [p=port] < <image_file> // Client
```

# Acknowledgements

This library includes code from the following external repositories:

 - [emp-toolkit/emp-tool](https://github.com/emp-toolkit/emp-tool/tree/c44566f40690d2f499aba4660f80223dc238eb03/emp-tool) for cryptographic tools and network I/O.
 - [emp-toolkit/emp-ot](https://github.com/emp-toolkit/emp-ot/tree/0f4a1e41a25cf1a034b5796752fde903a241f482/emp-ot) for Naor-Pinkas (base) OT and IKNP OT extension implementation.
 - [mc2-project/delphi](https://github.com/mc2-project/delphi/tree/de77cd7b896a2314fec205a8f67b257df46dd75c/rust/protocols-sys/c++/src/lib) for implementation of [Gazelle's](https://eprint.iacr.org/2018/073.pdf) algorithms for convolution and fully connected layers, which was majorly modified for better efficiency. 
 - [homenc/HElib](https://github.com/homenc/HElib/blob/6397b23e64c32fd6eab76bd7a08b95d8399503f4/src/NumbTh.h) for command-line argument parsing.
