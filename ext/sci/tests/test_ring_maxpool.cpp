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

#include "NonLinear/maxpool.h"
#include <fstream>
#include <thread>

using namespace std;
using namespace sci;

#define MAX_THREADS 4

int party = 0;
int port = 32000;
int num_rows = 35, num_cols = 1 << 6;
int bitlength = 32, b = 4;
int batch_size = 0;
string address = "127.0.0.1";
int num_threads = 1;

IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];

void ring_maxpool_thread(int tid, uint64_t *z, uint64_t *x, int lnum_rows,
                         int lnum_cols) {
  MaxPoolProtocol<uint64_t> *maxpool_oracle;
  if (tid & 1) {
    maxpool_oracle = new MaxPoolProtocol<uint64_t>(
        3 - party, RING, iopackArr[tid], bitlength, b, 0, otpackArr[tid]);
  } else {
    maxpool_oracle = new MaxPoolProtocol<uint64_t>(party, RING, iopackArr[tid],
                                                   bitlength, b, 0, otpackArr[tid]);
  }
  if (batch_size) {
    for (int j = 0; j < lnum_rows; j += batch_size) {
      if (batch_size <= lnum_rows - j) {
        maxpool_oracle->funcMaxMPC(batch_size, lnum_cols, x + j, z + j,
                                   nullptr);
      } else {
        maxpool_oracle->funcMaxMPC(lnum_rows - j, lnum_cols, x + j, z + j,
                                   nullptr);
      }
    }
  } else {
    maxpool_oracle->funcMaxMPC(lnum_rows, lnum_cols, x, z, nullptr);
  }

  delete maxpool_oracle;
  return;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/

  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("l", bitlength, "Bitlength of inputs");
  amap.arg("b", b, "Radix base");
  amap.arg("Nr", num_rows, "Number of rows");
  amap.arg("Nc", num_cols, "Number of cols");
  amap.arg("bt", batch_size, "Batch size as a power of 2 (No batching = 0)");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);
  if (batch_size > 0) {
    batch_size = 1 << batch_size;
  }

  cout << "========================================================" << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength << " - Radix Base: " << b
       << "\n#rows: " << num_rows << " - #cols: " << num_cols
       << " - Batch Size: " << batch_size << " - # Threads: " << num_threads
       << endl;
  cout << "========================================================" << endl;

  /************ Generate Test Data ************/
  /********************************************/
  uint64_t mask_l;
  if (bitlength == 64)
    mask_l = -1;
  else
    mask_l = (1ULL << bitlength) - 1;
  PRG128 prg;
  uint64_t *x = new uint64_t[num_rows * num_cols];
  uint64_t *z = new uint64_t[num_rows];
  prg.random_data(x, sizeof(uint64_t) * num_rows * num_cols);
  for (int i = 0; i < num_rows * num_cols; i++) {
    x[i] = x[i] & mask_l;
  }

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

  auto start = clock_start();
  std::thread maxpool_threads[num_threads];
  int chunk_size = num_rows / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_rows;
    if (i == (num_threads - 1)) {
      lnum_rows = num_rows - offset;
    } else {
      lnum_rows = chunk_size;
    }
    maxpool_threads[i] =
        std::thread(ring_maxpool_thread, i, z + offset, x + offset * num_cols,
                    lnum_rows, num_cols);
  }
  for (int i = 0; i < num_threads; ++i) {
    maxpool_threads[i].join();
  }
  long long t = time_from(start);

  /************** Verification ****************/
  /********************************************/

  switch (party) {
  case sci::ALICE: {
    iopackArr[0]->io->send_data(x, sizeof(uint64_t) * num_rows * num_cols);
    iopackArr[0]->io->send_data(z, sizeof(uint64_t) * num_rows);
    break;
  }
  case sci::BOB: {
    uint64_t *xi = new uint64_t[num_rows * num_cols];
    uint64_t *zi = new uint64_t[num_rows];
    iopackArr[0]->io->recv_data(xi, sizeof(uint64_t) * num_rows * num_cols);
    iopackArr[0]->io->recv_data(zi, sizeof(uint64_t) * num_rows);

    for (int i = 0; i < num_rows; i++) {
      zi[i] = (zi[i] + z[i]) & mask_l;
      for (int c = 0; c < num_cols; c++) {
        xi[i * num_cols + c] =
            (xi[i * num_cols + c] + x[i * num_cols + c]) & mask_l;
      }
      uint64_t maxpool_output = xi[i * num_cols];
      for (int c = 1; c < num_cols; c++) {
        maxpool_output = ((maxpool_output - xi[i * num_cols + c]) & mask_l) >=
                                 (1ULL << (bitlength - 1))
                             ? xi[i * num_cols + c]
                             : maxpool_output;
      }
      assert((zi[i] == maxpool_output) && "MaxPool output is incorrect");
    }
    delete[] xi;
    delete[] zi;
    cout << "Maxpool Tests Passed" << endl;
    break;
  }
  }
  delete[] x;
  delete[] z;

  cout << "Number of Maxpool rows (num_cols=" << num_cols << ")/s:\t"
       << (double(num_rows) / t) * 1e6 << std::endl;
  cout << "Maxpool Time (bitlength=" << bitlength << "; b=" << b << ")\t" << t << " mus"
       << endl;

  /******************* Cleanup ****************/
  /********************************************/

  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }

  return 0;
}
