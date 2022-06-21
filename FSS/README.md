# LLAMA: A Low Latency Math Library for Secure Inference

## Setup

These steps are tested on Ubuntu 18 and 20. To quickly test and try LLAMA out, you can use the `kanav99/ezpc-llama` docker image, which comes with all the dependencies pre-installed. To create and start a container using this image, run -

```bash
docker run -it kanav99/ezpc-llama:latest
```


### Steps
1. Clone EzPC repository

```bash
git clone http://github.com/mpc-msri/EzPC/
cd EzPC
```

2. (not required if running on the `kanav99/ezpc-llama` docker image) Install dependencies. This installs required tools like `g++`, `cmake`, `boost` and `ocaml`. This takes a long time.

```
./setup_env_and_build.sh quick
```

3. Recompile EzPC compiler

```bash
cd EzPC/EzPC/
eval `opam env`
make
cd ../../
```

3. Make a symlink for the FSS compiler wrapper

```bash
sudo ln -s `pwd`/EzPC/EzPC/fssc /usr/local/bin/fssc
```

4. Compile FSS backend

```bash
cd FSS/
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_BUILD_TYPE=Release ../
make install
cd ../../
```

## Tests

The directory `tests` contains a folder for each correctness test. For example, for a test `google30` there is a directory `tests/google30`. To run the test, run the following command while being inside the `tests` directory:

```bash
./run.sh google30
```

To run all tests, run the following command:

```bash
./runall.sh
```

## General Usage

1. To compile an EzPC file `program.ezpc` to `program.out`, run the following command:

```bash
fssc --bitlen <bitlen> program.ezpc
```

> For varied bitlen benchmarks like Google-30, bitlen is the max bitlen possible during the computation. For Google-30, use 64.

2. To run the dealer and generate key files, run the following command:

```bash
./program.out r=1 file=1
```

This generates two files - `server.dat` and `client.dat`. The `server.dat` file contains the server's key and the `client.dat` file contains the client's key in binary format. Copy these two files to their respective owner's folder/machine.

3. First the server must be run using the command:

```bash
cat input.txt | ./program.out r=2 file=1 port=<port> nt=<number of threads to use>
```

Ports `port` and `port+1` are used for the server. `input.txt` should contain space/endline separated input in fixed-point representation. For benchmarking purposes (not for verifying correctness), you can just replace `cat input.txt` with `yes 3`.

4. Then the client must be run using the command:

```bash
cat input.txt | ./program.out r=3 file=1 server=<ip-address-of-server> port=<port-of-server> nt=<same-number-of-threads-as-server>
```

If `server` is ommited, the client assumes that the server ip is `127.0.0.1`, that is, it is being run locally.

## Usage with Athos

Follow the instructions in the [Athos demo](https://github.com/mpc-msri/EzPC/tree/master/Athos/demos/onnx) with `target` set to `FSS` in `config.json`. The generated binary can be then used as described in the "General Usage" section.

> Note that FSS compiler currently performs Three-Address-Code (TAC) optimization and hence compilation takes much longer than other backends.

## Microbenchmarks

The directory `microbenchmarks` contains the microbenchmarks considered in the paper.

1. Sign Extension
2. Truncate-Reduce
3. Sigmoid
4. Tanh
5. Reciprocal Squareroot
6. MatMul (10x200 and 200x1000)
7. MatMul (10x2000 and 2000x100)
8. MatMul (200x200 and 200x200)

Respective source files can be compiled using - 

```bash
fssc --bitlen 64 <filename>.ezpc
```

> Some of these microbenchmarks might need re-configuration. Refer to point 2 of Notes section.

## Benchmarks

The directory `benchmarks` contains the benchmarks considered in the paper.

1. Google-30 - `fssc --bitlen 64 google30.ezpc`
2. DeepSecure - `fssc --bitlen 64 deepsecure.ezpc`
3. Industrial-72 (only Sigmoid and Tanh) - `fssc --bitlen 64 industrial.ezpc`
4. MiniONN LSTM (only Sigmoid and Tanh) - `fssc --bitlen 64 lstm.ezpc`
5. Heads (only Reciprocal Squarte Root) - `fssc --bitlen 64 heads1/2/3.ezpc`
6. MiniONN CNN - `fssc --bitlen 41 minionn-cnn.ezpc`
7. ResNet-50 - `fssc --bitlen 37 resnet50.ezpc`
8. ResNet-18 - `fssc --bitlen 32 resnet18.ezpc`

> Some of these benchmarks might need re-configuration. Refer to point 2 of Notes section.

## Notes

1. The dealer run (with `r=1`) reports the computation time only. The total offline should be this value plus the time taken by the dealer to transfer the files to their respective owners.

2. The code currently doesn't support multiple (bitlength, scale) in single program for math functions (Sigmoid, Tanh and InvSqrt). To enable a particular configuration, edit `src/config.h`, recompile the backend and then use `fssc` to compile an EzPC file. If you use a wrong configuration, you will see errors like this while running the dealer:

```
Assertion failed: shift_in == 9 in void TanH(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, GroupElement*, GroupElement*, GroupElement*, GroupElement*) at /EzPC/FSS/src/api_varied.cpp:1542
```

3. The server-client run (with `r=2` or `r=3`) reports offline communication and time. This is the communication and time required to send the masked model weights to client and generate maskes using the PRG key.
