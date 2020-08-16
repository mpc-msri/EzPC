/*
Authors: Mayank Rathee, Deevashwer Rathee
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

#define BITLEN_32
#define SCI_OT
#include <iostream>
#include <fstream>
#include <thread>
#include "globals.h"
#include "NonLinear/relu-ring.h"

#define LAN_EXEC

using namespace sci;
using namespace std;

int num_relu = 1<<20, port = 32000;
int num_relu_orig = 0;
int l = 32, b = 4;
int batch_size = 0;
string address;
bool localhost = true;
string network = "none";
vector<int> network_layer_sizes;

const std::map<std::string, std::vector<int>> layer_sizes {
    {"sq",
        vector<int>({ 200704, 50176, 200704, 200704, 50176, 200704, 200704, 23328, 93312, 93312, 23328, 93312, 93312, 8112, 32448, 32448, 8112, 32448, 32448, 10816, 43264, 43264, 10816, 43264, 43264, 169000 })
    },
    {"res",
        vector<int>({ 200704, 200704, 200704, 802816, 200704, 200704, 802816, 200704, 200704, 802816, 401408, 100352, 401408, 100352, 100352, 401408, 100352, 100352, 401408, 100352, 100352, 401408, 200704, 50176, 200704, 50176, 50176, 200704, 50176, 50176, 200704, 50176, 50176, 200704, 50176, 50176, 200704, 50176, 50176, 200704, 100352, 25088, 100352, 25088, 25088, 100352, 25088, 25088, 100352 })
    },
    {"dense",
        vector<int>({ 200704, 200704, 401408, 301056, 401408, 401408, 401408, 501760, 401408, 602112, 401408, 702464, 401408, 802816, 100352, 100352, 125440, 100352, 150528, 100352, 175616, 100352, 200704, 100352, 225792, 100352, 250880, 100352, 275968, 100352, 301056, 100352, 326144, 100352, 351232, 100352, 376320, 100352, 401408, 50176, 25088, 56448, 25088, 62720, 25088, 68992, 25088, 75264, 25088, 81536, 25088, 87808, 25088, 94080, 25088, 100352, 25088, 106624, 25088, 112896, 25088, 119168, 25088, 125440, 25088, 131712, 25088, 137984, 25088, 144256, 25088, 150528, 25088, 156800, 25088, 163072, 25088, 169344, 25088, 175616, 25088, 181888, 25088, 188160, 25088, 194432, 25088, 200704, 25088, 6272, 26656, 6272, 28224, 6272, 29792, 6272, 31360, 6272, 32928, 6272, 34496, 6272, 36064, 6272, 37632, 6272, 39200, 6272, 40768, 6272, 42336, 6272, 43904, 6272, 45472, 6272, 47040, 6272, 48608, 6272, 50176 })
    },
};

void ring_relu_thread(int tid, uint64_t* z, uint64_t* x, int lnum_relu) {
    ReLURingProtocol<NetIO, uint64_t>* relu_oracle;
    if(tid & 1) {
        relu_oracle = new ReLURingProtocol<NetIO, uint64_t>(3-party, RING, ioArr[tid], l, b, otpackArr[tid]);
    } else {
        relu_oracle = new ReLURingProtocol<NetIO, uint64_t>(party, RING, ioArr[tid], l, b, otpackArr[tid]);
    }
    if (batch_size) {
        for (int j = 0; j < lnum_relu; j += batch_size) {
            if (batch_size <= lnum_relu - j) {
                relu_oracle->relu(z+j, x+j, batch_size);
            } else {
                relu_oracle->relu(z+j, x+j, lnum_relu-j);
            }
        }
    } else {
        relu_oracle->relu(z, x, lnum_relu);
    }

    delete relu_oracle;
    return;
}

int main(int argc, char** argv) {
    /************* Argument Parsing  ************/
    /********************************************/

    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("l", l, "Bitlength of inputs");
    amap.arg("N", num_relu, "Number of ReLUs");
    amap.arg("b", b, "Radix base");
    amap.arg("lo", localhost, "Localhost Run?");
    amap.arg("bt", batch_size, "Batch size as a power of 2 (No batching = 0)");
    amap.arg("network", network, "Network Type: sq - SqNet, res - ResNet50, dense - DenseNet121");

    amap.parse(argc, argv);

    if (batch_size > 0) {
        batch_size = 1 << batch_size;
    }
    if(not localhost) {
#if defined(LAN_EXEC)
        address = "40.118.124.169"; // SGX4
#elif defined(WAN_EXEC)
        address = "40.117.40.111"; // SGX2
#endif
    } else {
        address = "127.0.0.1";
    }
    if (network != "none") {
        num_relu = 0;
        network_layer_sizes = layer_sizes.at(network);
        for(size_t i = 0; i < network_layer_sizes.size(); i++) {
            num_relu_orig += network_layer_sizes[i];
            num_relu += ((network_layer_sizes[i] + 7)/8)*8;
        }
        if (network == "res") l = 37;
        else l = 32;
    } else {
        num_relu_orig = num_relu;
        num_relu = ((num_relu + 7)/8)*8;
    }

    cout << "========================================================" << endl;
    cout << "Role: " << party << " - Bitlength: " << l << " - Radix Base: " << b
        << "\n# ReLUs: " << num_relu_orig << " - Batch Size: "
        << batch_size << " - # Threads: " << numThreads << endl;
    cout << "========================================================" << endl;

    /************ Generate Test Data ************/
    /********************************************/

    sci::PRG128 prg;
    uint64_t mask_l;
    if (l == 64) mask_l = -1;
    else mask_l = (1ULL << l) - 1;
    uint64_t *x = new uint64_t[num_relu];
    uint64_t *z = new uint64_t[num_relu];
    prg.random_data(x, sizeof(uint64_t)*num_relu);
    for(int i = 0; i < num_relu; i++) {
        x[i] = x[i] & mask_l;
    }

    /********** Setup IO and Base OTs ***********/
    /********************************************/

    for(int i = 0; i < numThreads; i++) {
        ioArr[i] = new NetIO(party==1 ? nullptr:address.c_str(), port+i);
        if (i == 0) {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], party, b, l);
        } else if (i == 1) {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], 3-party, b, l);
        } else if (i & 1) {
            // otpackArr[i] = new OTPack<NetIO>(ioArr[i], 3-party, b, l);
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], 3-party, b, l, false);
            // *otpackArr[i] = otpackArr[1];
            otpackArr[i]->copy(otpackArr[1]);
        } else {
            // otpackArr[i] = new OTPack<NetIO>(ioArr[i], party, b, l);
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], party, b, l, false);
            // *otpackArr[i] = otpackArr[0];
            otpackArr[i]->copy(otpackArr[0]);
        }
    }
    std::cout << "All Base OTs Done" << std::endl;
	
    /************** Fork Threads ****************/
    /********************************************/

    uint64_t comm_sent = 0;
	uint64_t multiThreadedIOStart[numThreads];
	for(int i=0;i<numThreads;i++){
		multiThreadedIOStart[i] = ioArr[i]->counter;
	}
    auto start = clock_start();
    if (network != "none") {
        int layer_offset = 0;
        for(size_t layer_idx = 0; layer_idx < network_layer_sizes.size(); layer_idx++) {
            std::thread relu_threads[numThreads];
            int layer_size = ((network_layer_sizes[layer_idx] + 7)/8)*8;
            int chunk_size = (layer_size/(8*numThreads))*8;
            cout << "Layer_idx: " << layer_idx << "; Layer_size: " << layer_size << endl;
            for (int i = 0; i < numThreads; ++i) {
                int offset = i*chunk_size + layer_offset;
                int lnum_relu;
                if (i == (numThreads - 1)) {
                    lnum_relu = (layer_offset + layer_size) - offset;
                } else {
                    lnum_relu = chunk_size;
                }
                relu_threads[i] = std::thread(ring_relu_thread, i, z+offset, x+offset, lnum_relu);
            }
            for (int i = 0; i < numThreads; ++i) {
              relu_threads[i].join();
            }
            layer_offset += layer_size;
        }
    } else {
        std::thread relu_threads[numThreads];
        int chunk_size = (num_relu/(8*numThreads))*8;
        for (int i = 0; i < numThreads; ++i) {
            int offset = i*chunk_size;
            int lnum_relu;
            if (i == (numThreads - 1)) {
                lnum_relu = num_relu - offset;
            } else {
                lnum_relu = chunk_size;
            }
            relu_threads[i] = std::thread(ring_relu_thread, i, z+offset, x+offset, lnum_relu);
        }
        for (int i = 0; i < numThreads; ++i) {
          relu_threads[i].join();
        }
    }
    long long t = time_from(start);
	for(int i=0;i<numThreads;i++){
		auto curComm = (ioArr[i]->counter) - multiThreadedIOStart[i];
		comm_sent += curComm;
	}
    std::cout << "Comm. Sent/ell: " << double(comm_sent*8)/(l*num_relu) << std::endl;

    /************** Verification ****************/
    /********************************************/

    switch (party) {
        case sci::ALICE: {
            ioArr[0]->send_data(x, sizeof(uint64_t)*num_relu);
            ioArr[0]->send_data(z, sizeof(uint64_t)*num_relu);
            break;
        }
        case sci::BOB: {
            uint64_t *xi = new uint64_t[num_relu];
            uint64_t *zi = new uint64_t[num_relu];
            ioArr[0]->recv_data(xi, sizeof(uint64_t)*num_relu);
            ioArr[0]->recv_data(zi, sizeof(uint64_t)*num_relu);
            // relu_oracle.drelu_ring_ideal_func(drelu_output_share, input_share1, input_share2, num_relu);

            for(int i=0; i<num_relu; i++){
                xi[i] = (xi[i] + x[i]) & mask_l;
                zi[i] = (zi[i] + z[i]) & mask_l;
                // cout << zi[i] << "\t" << xi[i] << "\t" << (xi[i] < (1ULL<<(l-1))) << "\t" << ((xi[i] < (1ULL<<(l-1))) * xi[i]) << endl;
                assert((zi[i] == ((xi[i] < (1ULL<<(l-1))) * xi[i]))
                        && "ReLU protocol's answer is incorrect!");
            }
            delete[] xi;
            delete[] zi;
            break;
        }
    }
    delete[] x;
    delete[] z;

    /**** Process & Write Benchmarking Data *****/
    /********************************************/

    cout <<"Number of ReLU/s:\t" <<(double(num_relu)/t)*1e6<< std::endl;
    cout <<"ReLU Time (l=" << l << "; b=" << b << ")\t" << t <<" mus"<< endl;

    string file_addr;
    switch (party) {
        case 1: {
            file_addr = "our-ring-relu-P0.csv";
            break;
        }
        case 2: {
            file_addr = "our-ring-relu-P1.csv";
            break;
        }
    }
    bool write_title = true; {
        fstream result(file_addr.c_str(), fstream::in);
        if(result.is_open())
            write_title = false;
        result.close();
    }
    fstream result(file_addr.c_str(), fstream::out|fstream::app);
    if(write_title){
        result << "Bitlen,Base,Batch Size,#Threads,#ReLU,Time (mus),Throughput/sec,Data Sent (MiB)" << endl;
    }
    result << l << "," << b << "," << batch_size << "," << numThreads << "," << num_relu_orig
        << "," << t << "," << (double(num_relu)/t)*1e6 << "," << comm_sent/(1.0*(1ULL<<20)) << endl;
    result.close();

    /******************* Cleanup ****************/
    /********************************************/

    for (int i = 0; i < numThreads; i++) {
        delete ioArr[i];
        delete otpackArr[i];
    }
	return 0;
}
