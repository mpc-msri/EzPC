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

#include "NonLinear/relu-field.h"
#include "library_fixed.h"
#include <fstream>
#include <iostream>
#include <thread>

#define MAX_THREADS 4

using namespace sci;
using namespace std;

int party = 0;
int num_relu = 35, port = 32000;
int b = 4;
int batch_size = 0;
string address = "127.0.0.1";
int num_threads = 1;
string network = "none";
vector<int> network_layer_sizes;
int32_t bitlength = 32;

const std::map<std::string, std::vector<int>> layer_sizes{
    {"sq", vector<int>({200704, 50176, 200704, 200704, 50176, 200704, 200704,
                        23328,  93312, 93312,  23328,  93312, 93312,  8112,
                        32448,  32448, 8112,   32448,  32448, 10816,  43264,
                        43264,  10816, 43264,  43264,  169000})},
    {"res",
     vector<int>({200704, 200704, 200704, 802816, 200704, 200704, 802816,
                  200704, 200704, 802816, 401408, 100352, 401408, 100352,
                  100352, 401408, 100352, 100352, 401408, 100352, 100352,
                  401408, 200704, 50176,  200704, 50176,  50176,  200704,
                  50176,  50176,  200704, 50176,  50176,  200704, 50176,
                  50176,  200704, 50176,  50176,  200704, 100352, 25088,
                  100352, 25088,  25088,  100352, 25088,  25088,  100352})},
    {"dense",
     vector<int>(
         {200704, 200704, 401408, 301056, 401408, 401408, 401408, 501760,
          401408, 602112, 401408, 702464, 401408, 802816, 100352, 100352,
          125440, 100352, 150528, 100352, 175616, 100352, 200704, 100352,
          225792, 100352, 250880, 100352, 275968, 100352, 301056, 100352,
          326144, 100352, 351232, 100352, 376320, 100352, 401408, 50176,
          25088,  56448,  25088,  62720,  25088,  68992,  25088,  75264,
          25088,  81536,  25088,  87808,  25088,  94080,  25088,  100352,
          25088,  106624, 25088,  112896, 25088,  119168, 25088,  125440,
          25088,  131712, 25088,  137984, 25088,  144256, 25088,  150528,
          25088,  156800, 25088,  163072, 25088,  169344, 25088,  175616,
          25088,  181888, 25088,  188160, 25088,  194432, 25088,  200704,
          25088,  6272,   26656,  6272,   28224,  6272,   29792,  6272,
          31360,  6272,   32928,  6272,   34496,  6272,   36064,  6272,
          37632,  6272,   39200,  6272,   40768,  6272,   42336,  6272,
          43904,  6272,   45472,  6272,   47040,  6272,   48608,  6272,
          50176})},
};

void field_relu_thread(int tid, uint64_t *z, uint64_t *x, int lnum_relu) {
  ReLUFieldProtocol<uint64_t> *relu_oracle;
  if (tid & 1) {
    relu_oracle = new ReLUFieldProtocol<uint64_t>(3 - party, FIELD,
                                                  iopackArr[tid], bitlength, b,
                                                  prime_mod, otpackArr[tid]);
  } else {
    relu_oracle = new ReLUFieldProtocol<uint64_t>(
        party, FIELD, iopackArr[tid], bitlength, b, prime_mod, otpackArr[tid]);
  }
  if (batch_size) {
    for (int j = 0; j < lnum_relu; j += batch_size) {
      if (batch_size <= lnum_relu - j) {
        relu_oracle->relu(z + j, x + j, batch_size);
      } else {
        relu_oracle->relu(z + j, x + j, lnum_relu - j);
      }
    }
  } else {
    relu_oracle->relu(z, x, lnum_relu);
  }

  delete relu_oracle;
  return;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/

  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", num_relu, "Number of ReLUs");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("b", b, "Radix base");
  amap.arg("bt", batch_size, "Batch size as a power of 2 (No batching = 0)");
  amap.arg("network", network,
           "Network Type: sq - SqNet, res - ResNet50, dense - DenseNet121");
  amap.arg("l", bitlength, "Bitlength of inputs");
  amap.parse(argc, argv);
  prime_mod = sci::default_prime_mod.at(bitlength);

  if (batch_size > 0) {
    batch_size = 1 << batch_size;
  }
  if (network != "none") {
    num_relu = 0;
    network_layer_sizes = layer_sizes.at(network);
    for (size_t i = 0; i < network_layer_sizes.size(); i++) {
      num_relu += network_layer_sizes[i];
    }
    if (network == "res")
      bitlength = 37;
    else
      bitlength = 32;
  }

  cout << "========================================================" << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength
       << " - Radix Base: " << b << "\n# ReLUs: " << num_relu
       << " - Batch Size: " << batch_size << " - # Threads: " << num_threads
       << endl;
  cout << "========================================================" << endl;

  /************ Generate Test Data ************/
  /********************************************/

  sci::PRG128 prg;
  uint64_t mask_l;
  if (bitlength == 64)
    mask_l = -1;
  else
    mask_l = (1ULL << bitlength) - 1;
  uint64_t *x = new uint64_t[num_relu];
  uint64_t *z = new uint64_t[num_relu];
  prg.random_mod_p<uint64_t>(x, num_relu, prime_mod);

  /********** Setup IO and Base OTs ***********/
  /********************************************/

  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }
  std::cout << "All Base OTs Done" << std::endl;

  /************** Fork Threads ****************/
  /********************************************/

  uint64_t comm_sent = 0;
  uint64_t multiThreadedIOStart[num_threads];
  for (int i = 0; i < num_threads; i++) {
    multiThreadedIOStart[i] = iopackArr[i]->get_comm();
  }
  auto start = clock_start();
  if (network != "none") {
    int layer_offset = 0;
    for (size_t layer_idx = 0; layer_idx < network_layer_sizes.size();
         layer_idx++) {
      std::thread relu_threads[num_threads];
      int layer_size = network_layer_sizes[layer_idx];
      int chunk_size = layer_size / num_threads;
      cout << "Layer_idx: " << layer_idx << "; Layer_size: " << layer_size
           << endl;
      for (int i = 0; i < num_threads; ++i) {
        int offset = i * chunk_size + layer_offset;
        int lnum_relu;
        if (i == (num_threads - 1)) {
          lnum_relu = (layer_offset + layer_size) - offset;
        } else {
          lnum_relu = chunk_size;
        }
        relu_threads[i] = std::thread(field_relu_thread, i, z + offset,
                                      x + offset, lnum_relu);
      }
      for (int i = 0; i < num_threads; ++i) {
        relu_threads[i].join();
      }
      layer_offset += layer_size;
    }
  } else {
    std::thread relu_threads[num_threads];
    int chunk_size = num_relu / num_threads;
    for (int i = 0; i < num_threads; ++i) {
      int offset = i * chunk_size;
      int lnum_relu;
      if (i == (num_threads - 1)) {
        lnum_relu = num_relu - offset;
      } else {
        lnum_relu = chunk_size;
      }
      relu_threads[i] =
          std::thread(field_relu_thread, i, z + offset, x + offset, lnum_relu);
    }
    for (int i = 0; i < num_threads; ++i) {
      relu_threads[i].join();
    }
  }
  long long t = time_from(start);
  for (int i = 0; i < num_threads; i++) {
    auto curComm = (iopackArr[i]->get_comm()) - multiThreadedIOStart[i];
    comm_sent += curComm;
  }
  std::cout << "Comm. Sent/ell: "
            << double(comm_sent * 8) / (bitlength * num_relu) << std::endl;

  /************** Verification ****************/
  /********************************************/

  switch (party) {
  case sci::ALICE: {
    iopackArr[0]->io->send_data(x, sizeof(uint64_t) * num_relu);
    iopackArr[0]->io->send_data(z, sizeof(uint64_t) * num_relu);
    break;
  }
  case sci::BOB: {
    uint64_t *xi = new uint64_t[num_relu];
    uint64_t *zi = new uint64_t[num_relu];
    iopackArr[0]->io->recv_data(xi, sizeof(uint64_t) * num_relu);
    iopackArr[0]->io->recv_data(zi, sizeof(uint64_t) * num_relu);

    for (int i = 0; i < num_relu; i++) {
      xi[i] = (xi[i] + x[i]) % prime_mod;
      zi[i] = (zi[i] + z[i]) % prime_mod;
      assert((zi[i] == ((xi[i] <= prime_mod / 2) * xi[i])) &&
             "ReLU protocol's answer is incorrect!");
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

  cout << "Number of ReLU/s:\t" << (double(num_relu) / t) * 1e6 << std::endl;
  cout << "ReLU Time (bitlength=" << bitlength << "; b=" << b << ")\t" << t
       << " mus" << endl;

  /******************* Cleanup ****************/
  /********************************************/

  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }
  return 0;
}
