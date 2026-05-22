/*
Authors: Deevashwer Rathee, Mayank Rathee
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

#include "BuildingBlocks/aux-protocols.h"
#include "utils/emp-tool.h"
#include <iostream>
using namespace sci;
using namespace std;

int party, port = 8000, dim = 35;
string address = "127.0.0.1";
IOPack *iopack;
OTPack *otpack;
AuxProtocols *aux;

void test_wrap_computation() {
  int bw_x = 32;
  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *x = new uint64_t[dim];
  uint8_t *y = new uint8_t[dim];

  prg.random_data(x, dim * sizeof(uint64_t));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & mask_x;
  }

  aux->wrap_computation(x, y, dim, bw_x);

  if (party == ALICE) {
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(y, dim * sizeof(uint8_t));
  } else {
    uint64_t *x0 = new uint64_t[dim];
    uint8_t *y0 = new uint8_t[dim];
    iopack->io->recv_data(x0, dim * sizeof(uint64_t));
    iopack->io->recv_data(y0, dim * sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
      assert((x0[i] > (mask_x - x[i])) == (y0[i] ^ y[i]));
    }
    cout << "Wrap Computation Tests passed" << endl;

    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}

void test_mux() {
  int bw_x = 32, bw_y = 32;
  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

  uint8_t *sel = new uint8_t[dim];
  uint64_t *x = new uint64_t[dim];
  uint64_t *y = new uint64_t[dim];

  prg.random_data(sel, dim * sizeof(uint8_t));
  prg.random_data(x, dim * sizeof(uint64_t));
  for (int i = 0; i < dim; i++) {
    sel[i] = sel[i] & 1;
    x[i] = x[i] & mask_x;
  }

  aux->multiplexer(sel, x, y, dim, bw_x, bw_y);

  if (party == ALICE) {
    iopack->io->send_data(sel, dim * sizeof(uint8_t));
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(y, dim * sizeof(uint64_t));
  } else {
    uint8_t *sel0 = new uint8_t[dim];
    uint64_t *x0 = new uint64_t[dim];
    uint64_t *y0 = new uint64_t[dim];
    iopack->io->recv_data(sel0, dim * sizeof(uint8_t));
    iopack->io->recv_data(x0, dim * sizeof(uint64_t));
    iopack->io->recv_data(y0, dim * sizeof(uint64_t));

    for (int i = 0; i < dim; i++) {
      assert(((uint64_t(sel0[i] ^ sel[i]) * (x0[i] + x[i])) & mask_y) ==
             ((y0[i] + y[i]) & mask_y));
    }
    cout << "MUX Tests passed" << endl;

    delete[] sel0;
    delete[] x0;
    delete[] y0;
  }
  delete[] sel;
  delete[] x;
  delete[] y;
}

void test_B2A() {
  int bw_y = 32;
  PRG128 prg;
  uint64_t mask_y = (bw_y == 64 ? -1 : ((1ULL << bw_y) - 1));

  uint8_t *x = new uint8_t[dim];
  uint64_t *y = new uint64_t[dim];

  prg.random_data(x, dim * sizeof(uint8_t));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & 1;
  }

  aux->B2A(x, y, dim, bw_y);

  if (party == ALICE) {
    iopack->io->send_data(x, dim * sizeof(uint8_t));
    iopack->io->send_data(y, dim * sizeof(uint64_t));
  } else {
    uint8_t *x0 = new uint8_t[dim];
    uint64_t *y0 = new uint64_t[dim];
    iopack->io->recv_data(x0, dim * sizeof(uint8_t));
    iopack->io->recv_data(y0, dim * sizeof(uint64_t));

    for (int i = 0; i < dim; i++) {
      assert(((uint64_t(x0[i] ^ x[i])) & mask_y) == ((y0[i] + y[i]) & mask_y));
    }
    cout << "B2A Tests passed" << endl;

    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}

template <typename T> void test_lookup_table() {
  int32_t T_size = sizeof(T) * 8;
  int bw_x = 8;
  int bw_y;
  if (T_size == 8)
    bw_y = 7;
  else
    bw_y = 29;
  PRG128 prg;
  uint64_t N = 1ULL << bw_x;
  T mask_x = (bw_x == T_size ? -1 : ((1ULL << bw_x) - 1));
  T mask_y = (bw_y == T_size ? -1 : ((1ULL << bw_y) - 1));

  T **spec = new T *[dim];
  T *x = new T[dim];
  T *y = new T[dim];

  for (int i = 0; i < dim; i++) {
    spec[i] = new T[N];
    prg.random_data(spec[i], N * sizeof(T));
  }
  prg.random_data(x, dim * sizeof(T));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & mask_x;
    for (int j = 0; j < N; j++) {
      spec[i][j] = spec[i][j] & mask_y;
    }
  }

  if (party == ALICE) {
    aux->lookup_table<T>(spec, nullptr, nullptr, dim, bw_x, bw_y);
  } else { // party == BOB
    aux->lookup_table<T>(nullptr, x, y, dim, bw_x, bw_y);
  }

  if (party == BOB) {
    iopack->io->send_data(x, dim * sizeof(T));
    iopack->io->send_data(y, dim * sizeof(T));
  } else { // ALICE knows the correct spec
    T *x0 = new T[dim];
    T *y0 = new T[dim];
    iopack->io->recv_data(x0, dim * sizeof(T));
    iopack->io->recv_data(y0, dim * sizeof(T));

    for (int i = 0; i < dim; i++) {
      assert((spec[i][x0[i] & mask_x]) == (y0[i] & mask_y));
    }
    if (T_size == 8)
      cout << "Lookup Table <uint8_t> Tests passed" << endl;
    else
      cout << "Lookup Table <uint64_t> Tests passed" << endl;

    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
  for (int i = 0; i < dim; i++)
    delete[] spec[i];
  delete[] spec;
}

void test_MSB_computation() {
  int bw_x = 32;
  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *x = new uint64_t[dim];
  uint8_t *y = new uint8_t[dim];

  prg.random_data(x, dim * sizeof(uint64_t));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & mask_x;
  }

  aux->MSB(x, y, dim, bw_x);

  if (party == ALICE) {
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(y, dim * sizeof(uint8_t));
  } else {
    uint64_t *x0 = new uint64_t[dim];
    uint8_t *y0 = new uint8_t[dim];
    iopack->io->recv_data(x0, dim * sizeof(uint64_t));
    iopack->io->recv_data(y0, dim * sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
      assert((((x0[i] + x[i]) & mask_x) >= (1ULL << (bw_x - 1))) ==
             (y0[i] ^ y[i]));
    }
    cout << "MSB Computation Tests passed" << endl;

    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}

void test_MSB_to_Wrap() {
  int bw_x = 32;
  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *x = new uint64_t[dim];
  uint8_t *msb_x = new uint8_t[dim];
  uint8_t *y = new uint8_t[dim];

  prg.random_data(x, dim * sizeof(uint64_t));
  prg.random_data(msb_x, dim * sizeof(uint8_t));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & mask_x;
    msb_x[i] = msb_x[i] & 1;
  }
  if (party == ALICE) {
    uint64_t *x_bob = new uint64_t[dim];
    uint8_t *msb_x_bob = new uint8_t[dim];
    iopack->io->recv_data(x_bob, dim * sizeof(uint64_t));
    iopack->io->recv_data(msb_x_bob, dim * sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
      msb_x[i] =
          (((x_bob[i] + x[i]) & mask_x) >= (1ULL << (bw_x - 1))) ^ msb_x_bob[i];
    }

    delete[] x_bob;
    delete[] msb_x_bob;
  } else {
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(msb_x, dim * sizeof(uint8_t));
  }

  aux->MSB_to_Wrap(x, msb_x, y, dim, bw_x);

  if (party == ALICE) {
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(y, dim * sizeof(uint8_t));
  } else {
    uint64_t *x0 = new uint64_t[dim];
    uint8_t *y0 = new uint8_t[dim];
    iopack->io->recv_data(x0, dim * sizeof(uint64_t));
    iopack->io->recv_data(y0, dim * sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
      assert((x0[i] > (mask_x - x[i])) == (y0[i] ^ y[i]));
    }
    cout << "MSB to Wrap Tests passed" << endl;

    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}

void test_AND() {
  int bw_in = 32;
  PRG128 prg;
  uint64_t mask_in = (bw_in == 64 ? -1 : ((1ULL << bw_in) - 1));

  uint8_t *x = new uint8_t[dim];
  uint8_t *y = new uint8_t[dim];
  uint8_t *z = new uint8_t[dim];

  prg.random_data(x, dim * sizeof(uint8_t));
  prg.random_data(y, dim * sizeof(uint8_t));

  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & 1;
    y[i] = y[i] & 1;
  }

  aux->AND(x, y, z, dim);

  if (party == ALICE) {
    uint8_t *x_bob = new uint8_t[dim];
    uint8_t *y_bob = new uint8_t[dim];
    uint8_t *z_bob = new uint8_t[dim];
    iopack->io->recv_data(x_bob, dim * sizeof(uint8_t));
    iopack->io->recv_data(y_bob, dim * sizeof(uint8_t));
    iopack->io->recv_data(z_bob, dim * sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
      x_bob[i] ^= x[i];
      y_bob[i] ^= y[i];
      z_bob[i] ^= z[i];
      assert((x_bob[i] & y_bob[i]) == z_bob[i]);
    }
    cout << "AND Computation Tests passed" << endl;

  } else {
    iopack->io->send_data(x, dim * sizeof(uint8_t));
    iopack->io->send_data(y, dim * sizeof(uint8_t));
    iopack->io->send_data(z, dim * sizeof(uint8_t));
  }

  delete[] x;
  delete[] y;
  delete[] z;
}

void test_digit_decomposition() {
  int bw_x = 32;
  int digit_size = 8;
  int num_digits = ceil((1.0 * bw_x) / digit_size);
  int last_digit_size = bw_x - (num_digits - 1) * digit_size;
  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
  uint64_t digit_mask = (digit_size == 64 ? -1 : (1ULL << digit_size) - 1);
  uint64_t last_digit_mask =
      (last_digit_size == 64 ? -1 : (1ULL << last_digit_size) - 1);

  uint64_t *x = new uint64_t[dim];
  uint64_t *xprime = new uint64_t[dim];
  uint64_t *y = new uint64_t[dim * num_digits];

  prg.random_data(x, dim * sizeof(uint64_t));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & mask_x;
  }

  aux->digit_decomposition_sci(dim, x, y, bw_x, digit_size);

  if (party == ALICE) {
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(y, dim * num_digits * sizeof(uint64_t));
  } else {
    uint64_t *x0 = new uint64_t[dim];
    uint64_t *y0 = new uint64_t[dim * num_digits];
    iopack->io->recv_data(x0, dim * sizeof(uint64_t));
    iopack->io->recv_data(y0, dim * num_digits * sizeof(uint64_t));
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < num_digits - 1; j++) {
        y0[j * dim + i] += y[j * dim + i];
        y0[j * dim + i] &= digit_mask;
      }
      y0[(num_digits - 1) * dim + i] += y[(num_digits - 1) * dim + i];
      y0[(num_digits - 1) * dim + i] &= last_digit_mask;
      xprime[i] = 0;
      for (int j = 0; j < num_digits; j++) {
        xprime[i] += y0[j * dim + i] * (1ULL << (j * digit_size));
      }
      xprime[i] &= mask_x;
      x0[i] += x[i];
      x0[i] &= mask_x;
      for (int j = 0; j < num_digits; j++) {
        uint64_t temp_mask =
            (j == (num_digits - 1)) ? last_digit_mask : digit_mask;
      }
      assert(xprime[i] == x0[i]);
    }
    cout << "Digit Decomposition Tests Passed" << endl;

    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}

void test_msnzb_one_hot() {
  int bw_x = 32;
  int digit_size = 8;
  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *x = new uint64_t[dim];
  uint8_t *y = new uint8_t[dim * bw_x];

  prg.random_data(x, dim * sizeof(uint64_t));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & mask_x;
  }

  aux->msnzb_one_hot(x, y, bw_x, dim, digit_size);

  if (party == ALICE) {
    iopack->io->send_data(x, dim * sizeof(uint64_t));
    iopack->io->send_data(y, dim * bw_x * sizeof(uint8_t));
  } else {
    uint64_t *x0 = new uint64_t[dim];
    uint8_t *y0 = new uint8_t[dim * bw_x];
    iopack->io->recv_data(x0, dim * sizeof(uint64_t));
    iopack->io->recv_data(y0, dim * bw_x * sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
      uint64_t secure_val = 0ULL;
      for (int j = 0; j < bw_x; j++) {
        secure_val += (1ULL << j) * int(y[i * bw_x + j] ^ y0[i * bw_x + j]);
      }
      if (unsigned_val(x0[i] + x[i], bw_x) == 0) {
        continue;
      }
      assert((unsigned_val(x0[i] + x[i], bw_x) >=
              unsigned_val(secure_val, bw_x)) &&
             (unsigned_val(x0[i] + x[i], bw_x) <
              2 * unsigned_val(secure_val, bw_x)));
    }
    std::cout << "MSNZB One Hot Tests Passed" << std::endl;
    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("d", dim, "Size of vector");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);

  iopack = new IOPack(party, port, "127.0.0.1");
  otpack = new OTPack(iopack, party);
  uint64_t num_rounds;

  aux = new AuxProtocols(party, iopack, otpack);

  test_MSB_computation();
  test_wrap_computation();
  test_mux();
  test_B2A();
  test_lookup_table<uint8_t>();
  test_lookup_table<uint64_t>();
  test_MSB_to_Wrap();
  test_AND();
  test_digit_decomposition();
  test_msnzb_one_hot();

  return 0;
}
