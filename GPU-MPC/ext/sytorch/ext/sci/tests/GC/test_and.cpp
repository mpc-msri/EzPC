#include "GC/emp-sh2pc.h"
#include <iostream>

using namespace sci;
using namespace std;

int party, port = 8000, iters = 512;
int bw_x = 64;
NetIO *io_gc;

void test_and() {
  PRG128 prg(fix_key);

  uint64_t *x = new uint64_t[iters];
  uint64_t *y = new uint64_t[iters];
  uint64_t *z = new uint64_t[iters];

  prg.random_data(x, iters * sizeof(uint64_t));
  prg.random_data(y, iters * sizeof(uint64_t));

  for (int i = 0; i < iters; i++) {
    Integer a(bw_x, x[i], ALICE);
    Integer b(bw_x, y[i], BOB);
    Integer c = a & b;
    z[i] = c.reveal<uint64_t>(PUBLIC);

    if (party == BOB) {
      if (z[i] != (x[i] & y[i])) {
        cout << i << "\t" << x[i] << "\t" << y[i] << "\t" << (x[i] & y[i])
             << "\t" << z[i] << endl;
        cout << ((SemiHonestParty<NetIO> *)prot_exec)->top << endl;
      }
      assert(z[i] == (x[i] & y[i]));
    }
  }

  delete[] x;
  delete[] y;
  delete[] z;
}

int main(int argc, char **argv) {
  party = atoi(argv[1]);
  io_gc = new NetIO(party == ALICE ? nullptr : "127.0.0.1",
                    port + GC_PORT_OFFSET, true);

  setup_semi_honest(io_gc, party);
  test_and();
}
