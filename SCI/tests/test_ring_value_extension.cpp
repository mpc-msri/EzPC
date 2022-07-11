/*
Authors: Mayank Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
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

#include "BuildingBlocks/value-extension.h"
#include <iostream>

using namespace sci;
using namespace std;

int dim = 1024;
int bwA = 32;
int bwB = 64;
bool precomputed_MSBs = false;

uint64_t maskA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
uint64_t maskB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));

// vars
int party, port = 32000;
IOPack *iopack;
OTPack *otpack;
XTProtocol *ext;
PRG128 prg;

void z_ext() {
  uint64_t *inA = new uint64_t[dim];
  uint64_t *outB = new uint64_t[dim];

  prg.random_data(inA, dim * sizeof(uint64_t));
  prg.random_data(outB, dim * sizeof(uint64_t));

  for (int i = 0; i < dim; i++) {
    inA[i] &= maskA;
    outB[i] = 0;
  }

  uint8_t *msbA = nullptr;
  if (precomputed_MSBs) {
    msbA = new uint8_t[dim];
    ext->aux->MSB(inA, msbA, dim, bwA);
  }
  uint64_t num_rounds = iopack->get_rounds();
  ext->z_extend(dim, inA, outB, bwA, bwB, msbA);
  num_rounds = iopack->get_rounds() - num_rounds;
  cout << "Num rounds (Zero-Extension): " << num_rounds << endl;

  if (party == ALICE) {
    uint64_t *inA_bob = new uint64_t[dim];
    uint64_t *outB_bob = new uint64_t[dim];
    iopack->io->recv_data(inA_bob, sizeof(uint64_t) * dim);
    iopack->io->recv_data(outB_bob, sizeof(uint64_t) * dim);
    for (int i = 0; i < dim; i++) {
      inA[i] = (inA[i] + inA_bob[i]) & maskA;
      outB[i] = (outB[i] + outB_bob[i]) & maskB;
    }
    cout << "Testing for correctness..." << endl;
    for (int i = 0; i < dim; i++) {
      // cout << inA[i] << " " << outB[i] << endl;
      assert(inA[i] == outB[i]);
    }
    cout << "Correct!" << endl;
  } else { // BOB
    iopack->io->send_data(inA, sizeof(uint64_t) * dim);
    iopack->io->send_data(outB, sizeof(uint64_t) * dim);
  }
}

void s_ext() {
  uint64_t *inA = new uint64_t[dim];
  uint64_t *outB = new uint64_t[dim];

  prg.random_data(inA, dim * sizeof(uint64_t));
  prg.random_data(outB, dim * sizeof(uint64_t));

  for (int i = 0; i < dim; i++) {
    inA[i] &= maskA;
    outB[i] = 0;
  }

  uint8_t *msbA = nullptr;
  if (precomputed_MSBs) {
    msbA = new uint8_t[dim];
    ext->aux->MSB(inA, msbA, dim, bwA);
  }
  uint64_t num_rounds = iopack->get_rounds();
  ext->s_extend(dim, inA, outB, bwA, bwB, msbA);
  num_rounds = iopack->get_rounds() - num_rounds;
  cout << "Num rounds (Signed-Extension): " << num_rounds << endl;

  if (party == ALICE) {
    uint64_t *inA_bob = new uint64_t[dim];
    uint64_t *outB_bob = new uint64_t[dim];
    iopack->io->recv_data(inA_bob, sizeof(uint64_t) * dim);
    iopack->io->recv_data(outB_bob, sizeof(uint64_t) * dim);
    for (int i = 0; i < dim; i++) {
      inA[i] = (inA[i] + inA_bob[i]) & maskA;
      outB[i] = (outB[i] + outB_bob[i]) & maskB;
    }
    cout << "Testing for correctness..." << endl;
    for (int i = 0; i < dim; i++) {
      // cout << inA[i] << " " << outB[i] << endl;
      assert(signed_val(inA[i], bwA) == signed_val(outB[i], bwB));
    }
    cout << "Correct!" << endl;
  } else { // BOB
    iopack->io->send_data(inA, sizeof(uint64_t) * dim);
    iopack->io->send_data(outB, sizeof(uint64_t) * dim);
  }
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("m", precomputed_MSBs, "MSB_to_Wrap Optimization?");
  amap.arg("d", dim, "Size of vector");
  amap.parse(argc, argv);

  iopack = new IOPack(party, port, "127.0.0.1");
  otpack = new OTPack(iopack, party);
  ext = new XTProtocol(party, iopack, otpack);

  cout << "<><><><> Zero Extension <><><><>" << endl;
  z_ext();
  cout << "<><><><> Signed Extension <><><><>" << endl;
  s_ext();
}
