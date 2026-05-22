#include "GC/emp-sh2pc.h"
using namespace sci;
using namespace std;
int party;
int num_threads = 1;
NetIO *io_gc;

void test_bit() {
  bool b[] = {true, false};
  int p[] = {PUBLIC, ALICE, BOB};

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 2; ++k)
        for (int l = 0; l < 3; ++l) {
          {
            Bit b1(b[i], p[j]);
            Bit b2(b[k], p[l]);
            bool res = (b1 & b2).reveal(PUBLIC);
            if (res != (b[i] and b[k])) {
              cout << "AND" << i << " " << j << " " << k << " " << l << " "
                   << res << endl;
              error("test bit error!");
            }
            res = (b1 & b1).reveal(PUBLIC);
            if (res != b[i]) {
              cout << "AND" << i << " " << j << res << endl;
              error("test bit error!");
            }

            res = (b1 & (!b1)).reveal(PUBLIC);
            if (res) {
              cout << "AND" << i << " " << j << res << endl;
              error("test bit error!");
            }
          }
          {
            Bit b1(b[i], p[j]);
            Bit b2(b[k], p[l]);
            bool res = (b1 ^ b2).reveal(PUBLIC);
            if (res != (b[i] xor b[k])) {
              cout << "XOR" << i << " " << j << " " << k << " " << l << " "
                   << res << endl;
              error("test bit error!");
            }

            res = (b1 ^ b1).reveal(PUBLIC);
            if (res) {
              cout << "XOR" << i << " " << j << res << endl;
              error("test bit error!");
            }

            res = (b1 ^ (!b1)).reveal(PUBLIC);
            if (!res) {
              cout << "XOR" << i << " " << j << res << endl;
              error("test bit error!");
            }
          }
          {
            Bit b1(b[i], p[j]);
            Bit b2(b[k], p[l]);
            bool res = (b1 ^ b2).reveal(XOR);
            if (party == ALICE) {
              io_gc->send_data(&res, 1);
            } else {
              bool tmp;
              io_gc->recv_data(&tmp, 1);
              res = res != tmp;
              if (res != (b[i] xor b[k])) {
                cout << "XOR" << i << " " << j << " " << k << " " << l << " "
                     << res << endl;
                error("test bit error!");
              }
            }
          }
        }
  io_gc->flush();
  cout << "success!" << endl;
}

int main(int argc, char **argv) {
  int port = 8000;
  party = atoi(argv[1]);
  io_gc = new NetIO(party == ALICE ? nullptr : "127.0.0.1",
                    port + GC_PORT_OFFSET, true);

  setup_semi_honest(io_gc, party);
  test_bit();
}
