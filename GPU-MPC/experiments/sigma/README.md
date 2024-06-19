
# SIGMA: Secure GPT Inference with Function Secret Sharing

Implementation of protocols from the paper [SIGMA](https://eprint.iacr.org/2023/1269).

**Warning**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Build

This project requires NVIDIA GPUs and assumes that GPU drivers and the [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/) are already installed. The following has been tested on Ubuntu 20.04 with CUDA 11.7, CMake 3.27.2 and g++-9. 

Please note that Sytorch requires CMake version >= 3.17 and the build will fail if this dependency is not met. 

The code uses CUTLASS version 2.11 by default, so if you change the CUDA version, please make sure that the CUTLASS version being built is compatible with the new CUDA version. To change the version of CUTLASS being built, add `git checkout <branch>;` after line 31 (`cd ext/cutlass;`) of setup.sh.

The last line of `setup.sh` tries to install `matplotlib`, which is needed for generating Figure 10. In our experience, the installation fails if the versions of Python and `pip` do not match. In case the installation fails, please install `matplotlib` manually before running `run_experiment.py`.

This artifact requires a large amount of resources to run. To produce the numbers reported in the paper, we used two machines connected via LAN with 9.4 Gbps bandwidth and 0.05 ms ping time. Each machine has 1 TB RAM, an A6000 GPU with 46GB GPU memory, and an AMD Epyc 7742 processor.

1. Export environment variables

```
export CUDA_VERSION=11.7
export GPU_ARCH=86
```

2. Set up the environment

```
sh setup.sh
```
_Note:_ The above script compiles CUTLASS with the maximum possible number of threads. As this step requires a large amount of memory, you can change line 34 from `make -j` to `make -j<n>` to compile on `n` threads and avoid out-of-memory errors. 

3. Make SIGMA

```
make sigma
```

4. Switch to the experiments directory

```
cd experiments/sigma
```

## Run SIGMA

### Prerequisites and caveats

1. Since FSS generates large keys, please ensure that you have a writeable disk with at least 500GB of free space. This is only required by our largest model (Llama2-13B). Other models require less space, and an idea of how much free space is needed per model can be estimated from the key size reported in Table 9 of the paper.

2. In the online phase, SIGMA loads the entire key from the disk into CPU memory. Thus, the CPU must have (free) memory that is at least as large as the key that will be read from the disk.

3. Currently, we only support sequence lengths that are powers-of-2.


### Run standalone

Make produces the `sigma` executable which is in `experiments/sigma`.

Each party (the server and the client) needs to run two processes in sequence: the dealer and the evaluator. In addition to other arguments, the dealer requires the user to specify the directory in which it will store keys (see prerequisites and caveats). The evaluator requires the user to specify the directory to read keys from, the IP address of its peer, and the number of CPU threads to use for computation.

The syntax for running the dealer is 
```javascript
./sigma <model name> <sequence length> <role=0 for dealer> <party=0/1 (server/client)> <key directory>
```

The syntax for running the evaluator is 
```javascript
./sigma <model name> <sequence length> <role=1 for evaluator> <party=0/1 (server/client)> <key directory> <peer IP> <CPU threads>`
```

We currently support the following models: `bert-tiny, bert-base, bert-large, gpt2, llama-7b, llama-13b`.

**Example:** To run GPT2, the server will run (in sequence):
```javascript
./sigma gpt2 128 0 0 /tmp/
./sigma gpt2 128 1 0 /tmp/ <client IP> 64
```

The client will run (_on a different machine_):
```javascript
./sigma gpt2 128 0 1 /tmp/
./sigma gpt2 128 1 1 /tmp/ <server IP> 64
```

Results are stored in the `output/P<party number>/models/<model name>-<sequence length>/` folder.

### Running the artifact

Before the artifact can be run, we need to specify the dealer and evaluator configurations in `config.json`. 

For the server(=P0), `config.json` looks like:
```javascript
{
    "P0": {
        "dealer": {
            "gpu": <The ID of the GPU to use>,
            "key_dir": <The directory in which the dealer will store keys>
        },
        "evaluator": {
            "gpu": <The ID of the GPU to use>,
            "peer": <The IP address of the remote peer>,
            "cpu_threads": <The number of CPU threads to use for computation>
        }
    }
}
```

For the client(=P1), `config.json` looks exactly the same, only the arguments are specified under the key "P1".

A sample `config.json` file can be found in the `experiments/sigma` folder.

Once `config.json` has been filled, the script `run_experiment.py` can be used to reproduce the tables and figures in the paper. Here are the relevant options:

```
usage: python run_experiment.py [-h] [--perf true] [--n_seq true] [--all true] --party 0/1

optional arguments:
  --perf true      Generate Tables 3, 5, 9, and Figure 10.
  --n_seq true     Generate Table 8.
  --all true       Run all the experiments.
```

Table 7 can be reproduced by throttling the network bandwidth (with `tc`, for example) and re-running `python run_experiment.py --perf true` to generate Table 5. 

Results are stored in `output/P<party-number>/Table<table-number>.json` or `output/P<party-number>/Fig<figure-number>.json`. 

Log files (which might help with debugging) can be found in the `output/P<party number>/models/<model name>-<sequence length>/logs/` folder.

### Generating CPU numbers

To generate CPU performance numbers in Table 3 and Figures 9,10, follow these steps:

1. On both machines, run `setup.sh` script, as described in previous sections.

2. On the first machine, change to the build directory and run the Python script with its IP address

```
cd ext/sytorch/build/
python ../scripts/all-cpu-benchmarks-remote.py <ip> 0
```

3. On the second machine, use the IP address of the first machine.

```
cd ext/sytorch/build/
python ../scripts/all-cpu-benchmarks-remote.py <ip> 1
```

If you'd like to run on a single machine, use the local script without any arguments. Note that since it runs 2 processes in parallel, it requires double the hardware.

```
python ../scripts/all-cpu-benchmarks-local.py
```

At the end of this script, which could take several hours, you get a `results.csv` file containing all the required time and communication numbers required to generate Table 3 and Figure 9,10.

## Citation

You can cite the paper using the following BibTeX entry:

```
@misc{cryptoeprint:2023/1269,
      author = {Kanav Gupta and Neha Jawalkar and Ananta Mukherjee and Nishanth Chandran and Divya Gupta and Ashish Panwar and Rahul Sharma},
      title = {SIGMA: Secure GPT Inference with Function Secret Sharing},
      howpublished = {Cryptology ePrint Archive, Paper 2023/1269},
      year = {2023},
      note = {\url{https://eprint.iacr.org/2023/1269}},
      url = {https://eprint.iacr.org/2023/1269}
}
```

