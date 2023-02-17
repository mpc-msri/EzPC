#include "LinearOT/linear-ot.h"
#include "utils/emp-tool.h"
#include <iostream>

using namespace sci;
using namespace std;

int party, port = 32000;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
LinearOT *prod;

int dim = 1 << 16;
int bwA = 32;
int bwB = 32;
int bwC = 32;

uint64_t maskA = (bwA == 64 ? -1 : ((1ULL << bwA) - 1));
uint64_t maskB = (bwB == 64 ? -1 : ((1ULL << bwB) - 1));
uint64_t maskC = (bwC == 64 ? -1 : ((1ULL << bwC) - 1));

void test_hadamard_product(uint64_t *inA, uint64_t *inB,
                           bool signed_arithmetic = true) {
  uint64_t *outC = new uint64_t[dim];

  prod->hadamard_product(dim, inA, inB, outC, bwA, bwB, bwC, signed_arithmetic);

  if (party == ALICE) {
    iopack->io->send_data(inA, dim * sizeof(uint64_t));
    iopack->io->send_data(inB, dim * sizeof(uint64_t));
    iopack->io->send_data(outC, dim * sizeof(uint64_t));
  } else { // party == BOB
    uint64_t *inA0 = new uint64_t[dim];
    uint64_t *inB0 = new uint64_t[dim];
    uint64_t *outC0 = new uint64_t[dim];
    iopack->io->recv_data(inA0, dim * sizeof(uint64_t));
    iopack->io->recv_data(inB0, dim * sizeof(uint64_t));
    iopack->io->recv_data(outC0, dim * sizeof(uint64_t));

    for (int i = 0; i < dim; i++) {
      if (signed_arithmetic) {
        assert(signed_val(outC[i] + outC0[i], bwC) ==
               (signed_val(signed_val(inA[i] + inA0[i], bwA) *
                               signed_val(inB[i] + inB0[i], bwB),
                           bwC)));
      } else {
        assert(unsigned_val(outC[i] + outC0[i], bwC) ==
               (unsigned_val(unsigned_val(inA[i] + inA0[i], bwA) *
                                 unsigned_val(inB[i] + inB0[i], bwB),
                             bwC)));
      }
    }
    if (signed_arithmetic)
      cout << "SMult Tests Passed" << endl;
    else
      cout << "UMult Tests Passed" << endl;

    delete[] inA0;
    delete[] inB0;
    delete[] outC0;
  }

  delete[] outC;
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);

  iopack = new IOPack(party, port, address);
  otpack = new OTPack(iopack, party);

  prod = new LinearOT(party, iopack, otpack);

  PRG128 prg;

  uint64_t *inA = new uint64_t[dim];
  uint64_t *inB = new uint64_t[dim];

  prg.random_data(inA, dim * sizeof(uint64_t));
  prg.random_data(inB, dim * sizeof(uint64_t));

  for (int i = 0; i < dim; i++) {
    inA[i] &= maskA;
    inB[i] &= maskB;
  }

  test_hadamard_product(inA, inB, false);
  test_hadamard_product(inA, inB, true);

  delete[] inA;
  delete[] inB;
  delete prod;
}
