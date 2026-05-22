# Introduction

This folder contains the following

## Beacon Frontend

This folder contains the frontend to translate a pytorch network into training code in either Secfloat/Beacon. The frontend also dumps the networks weights, and sample input/label files to be used for training.

The `compile_networks.py` folder contains 5 networks. Please check the class definitions for information as to how to specify the network model.
- Logistic
- FFNN
- Relevance
- LeNet
- HiNet

## Microbenchmarks

The scripts to reproduce the microbenchmark results on the paper [Secure Floating-Point Training](https://eprint.iacr.org/2023/467)

## Benchmarks

The scripts to reproduce the benchmark results on the paper [Secure Floating-Point Training](https://eprint.iacr.org/2023/467)

# Usage

## Using the beacon frontend

To run, please use the following command

`python3 compile_networks.py <network> <batch size> <training iterations> <learning rate> <loss> <momentum>`

For loss specify either "CE" or "MSE"
For momentum specify either "yes" or "no"

After running the script, on the server side run

`./<network>_secfloat r=1 < <network>_weights.inp`

On the client side run

`cat <network>_input<batch size>.inp <network>_labels<batch size>.inp | ./<network>_secfloat r=2 add=<ip of server>`

## Reproducing microbenchmark results

On the server side, run the following command

`./micro_server.sh`

On the client side, run the following command

`./micro_client.sh <ip address of server>`

## Reproducing benchmark results

On the server side, run the following command

`./bench_server.sh`

On the client side, run the following command

`./bench_client.sh <ip address of server>`
