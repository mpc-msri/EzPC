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

#define BITLEN_41
#define SCI_OT
#include "globals.h"
#include "NonLinear/relu-field.h"
#include "NonLinear/relu-ring.h"
#include "NonLinear/maxpool.h"
#include <fstream>
#include <thread>

using namespace std;
using namespace sci;

int port = 32000;
int num_rows = 1<<15, num_cols = 1<<7;
int l = 41, b = 4;
int batch_size = 0;
string address;
bool localhost = true;

void ring_maxpool_thread(int tid, uint64_t* z, uint64_t* x, int lnum_rows, int lnum_cols) {
    MaxPoolProtocol<NetIO, uint64_t>* maxpool_oracle;
    if(tid & 1) {
        maxpool_oracle = new MaxPoolProtocol<NetIO, uint64_t>(3-party, RING, ioArr[tid], l, b, 0, otpackArr[tid]);
    } else {
        maxpool_oracle = new MaxPoolProtocol<NetIO, uint64_t>(party, RING, ioArr[tid], l, b, 0, otpackArr[tid]);
    }
    if (batch_size) {
        for (int j = 0; j < lnum_rows; j += batch_size) {
            if (batch_size <= lnum_rows - j) {
                maxpool_oracle->funcMaxMPC(batch_size, lnum_cols, x+j, z+j, nullptr);
            } else {
                maxpool_oracle->funcMaxMPC(lnum_rows-j, lnum_cols, x+j, z+j, nullptr);
            }
        }
    } else {
        maxpool_oracle->funcMaxMPC(lnum_rows, lnum_cols, x, z, nullptr);
    }

    delete maxpool_oracle;
    return;
}

int main(int argc, char** argv){
    /************* Argument Parsing  ************/
    /********************************************/

    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("l", l, "Bitlength of inputs");
    amap.arg("b", b, "Radix base");
    amap.arg("Nr", num_rows, "Number of rows");
    amap.arg("Nc", num_cols, "Number of cols");
    amap.arg("lo", localhost, "Localhost Run?");
    amap.arg("bt", batch_size, "Batch size as a power of 2 (No batching = 0)");

    amap.parse(argc, argv);
    if (batch_size > 0) {
        batch_size = 1 << batch_size;
    }
    if(not localhost) {
        address = "40.118.124.169";
    } else {
        address = "127.0.0.1";
    }
    // assert(l == 32);
		num_rows = ((num_rows + 7)/8)*8;

    cout << "========================================================" << endl;
    cout << "Role: " << party << " - Bitlength: " << l << " - Radix Base: " << b
        << "\n#rows: " << num_rows << " - #cols: " << num_cols << " - Batch Size: "
        << batch_size << " - # Threads: " << numThreads << endl;
    cout << "========================================================" << endl;

    /************ Generate Test Data ************/
    /********************************************/
    uint64_t mask_l;
    if (l == 64) mask_l = -1;
    else mask_l = (1ULL << l) - 1;
    PRG128 prg;
    uint64_t *x = new uint64_t[num_rows*num_cols];
    // uint8_t *x_sign = new uint8_t[num_rows*num_cols];
    uint64_t *z = new uint64_t[num_rows];
    prg.random_data(x, sizeof(uint64_t)*num_rows*num_cols);
    // prg.random_bool((bool*) x_sign, num_rows*num_cols);
    for(int i = 0; i < num_rows*num_cols; i++) {
        x[i] = x[i] & mask_l;//(magnitude_bound-1);
        // if (x_sign[i]) {
        //     x[i] = (-1 * x[i]) & mask_l;
        // }
    }
    // delete[] x_sign;

    /********** Setup IO and Base OTs ***********/
    /********************************************/

    for(int i = 0; i < numThreads; i++) {
        ioArr[i] = new NetIO(party==1 ? nullptr:address.c_str(), port+i);
        if (i == 0) {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], party, b, l);
        } else if (i == 1) {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], 3-party, b, l);
        } else if (i & 1) {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], 3-party, b, l, false);
            otpackArr[i]->copy(otpackArr[1]);
        } else {
            otpackArr[i] = new OTPack<NetIO>(ioArr[i], party, b, l, false);
            otpackArr[i]->copy(otpackArr[0]);
        }
    }
    std::cout << "All Base OTs Done" << std::endl;

    /************** Fork Threads ****************/
    /********************************************/

    auto start = clock_start();
    std::thread maxpool_threads[numThreads];
    int chunk_size = (num_rows/(8*numThreads))*8;
    for (int i = 0; i < numThreads; ++i) {
        int offset = i*chunk_size;
        int lnum_rows;
        if (i == (numThreads - 1)) {
            lnum_rows = num_rows - offset;
        } else {
            lnum_rows = chunk_size;
        }
        maxpool_threads[i] = std::thread(ring_maxpool_thread, i, z+offset, x+offset*num_cols, lnum_rows, num_cols);
    }
    for (int i = 0; i < numThreads; ++i) {
      maxpool_threads[i].join();
    }
    long long t = time_from(start);

    /************** Verification ****************/
    /********************************************/

    switch (party) {
        case sci::ALICE: {
            ioArr[0]->send_data(x, sizeof(uint64_t)*num_rows*num_cols);
            ioArr[0]->send_data(z, sizeof(uint64_t)*num_rows);
            break;
        }
        case sci::BOB: {
            uint64_t *xi = new uint64_t[num_rows*num_cols];
            uint64_t *zi = new uint64_t[num_rows];
            ioArr[0]->recv_data(xi, sizeof(uint64_t)*num_rows*num_cols);
            ioArr[0]->recv_data(zi, sizeof(uint64_t)*num_rows);

            for(int i=0; i<num_rows; i++){
                zi[i] = (zi[i] + z[i]) & mask_l;
                for(int c=0; c<num_cols; c++){
                    xi[i*num_cols + c] = (xi[i*num_cols + c] + x[i*num_cols + c]) & mask_l;
                }
                uint64_t maxpool_output = xi[i*num_cols];
                // cout << xi[i*num_cols] << "\t";
                for(int c=1; c<num_cols; c++){
                    maxpool_output = ((maxpool_output - xi[i*num_cols + c]) & mask_l) >= (1ULL << (l-1)) ?
                        xi[i*num_cols + c]:maxpool_output;
                    // cout << xi[i*num_cols + c] << "\t";
                }
                // cout << maxpool_output << "\t";
                // cout << zi[i] << endl;
                assert((zi[i] == maxpool_output) && "MaxPool output is incorrect");
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

    cout << "Number of Maxpool rows (num_cols=" << num_cols << ")/s:\t"
        << (double(num_rows)/t)*1e6<< std::endl;
    cout <<"Maxpool Time (l=" << l << "; b=" << b << ")\t" << t <<" mus"<< endl;
    string file_addr;
    switch (party) {
        case 1: {
            file_addr = "our-ring-maxpool-P0.csv";
            break;
        }
        case 2: {
            file_addr = "our-ring-maxpool-P1.csv";
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
        result << "Bitlen,Base,Batch Size,#Threads,Maxpool Size,Time (mus),Throughput/sec" << endl;
    }
    result << l << "," << b << "," << batch_size << "," << numThreads << "," << num_rows << "x" << num_cols
        << "," << t << "," << (double(num_rows)/t)*1e6 << endl;
    result.close();

    /******************* Cleanup ****************/
    /********************************************/

    for (int i = 0; i < numThreads; i++) {
        delete ioArr[i];
        delete otpackArr[i];
    }

	return 0;
}
