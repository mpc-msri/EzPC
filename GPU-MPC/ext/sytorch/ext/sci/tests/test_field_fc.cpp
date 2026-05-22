/*
Authors: Deevashwer Rathee
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
#include "LinearHE/fc-field.h"

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int num_rows = 1001;
int common_dim = 512;
int filter_precision = 15;

void MatMul(FCField &he_fc, int32_t num_rows, int32_t common_dim) {
  int num_cols = 1;
  vector<vector<uint64_t>> A(num_rows);   // Weights
  vector<vector<uint64_t>> B(common_dim); // Image
  vector<vector<uint64_t>> C(num_rows);
  PRG128 prg;
  for (int i = 0; i < num_rows; i++) {
    A[i].resize(common_dim);
    C[i].resize(num_cols);
    if (party == ALICE) {
      prg.random_data(A[i].data(), common_dim * sizeof(uint64_t));
      for (int j = 0; j < common_dim; j++) {
        A[i][j] = ((int64_t)A[i][j]) >> (64 - filter_precision);
      }
    }
  }
  for (int i = 0; i < common_dim; i++) {
    B[i].resize(1);
    prg.random_mod_p<uint64_t>(B[i].data(), num_cols, prime_mod);
  }
  INIT_TIMER;
  START_TIMER;
  he_fc.matrix_multiplication(num_rows, common_dim, num_cols, A, B, C, true,
                              true);
  STOP_TIMER("Total Time for FC");
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("n", num_rows, "Rows in Weight Matrix");
  amap.arg("c", common_dim, "Image Length / Columns in Weight Matrix");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("l", bitlength, "Bitlength of inputs");
  amap.parse(argc, argv);
  prime_mod = sci::default_prime_mod.at(bitlength);

  cout << "===================================================================="
       << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength
       << " - Mod: " << prime_mod << " - Rows: " << num_rows
       << " - Cols: " << common_dim << " - # Threads: " << num_threads << endl;
  cout << "===================================================================="
       << endl;

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

  FCField he_fc(party, io);

  MatMul(he_fc, num_rows, common_dim);

  io->flush();
  return 0;
}
