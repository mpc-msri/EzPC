#include "BuildingBlocks/aux-protocols.h"
#include <iostream>

using namespace sci;
using namespace std;

#define MAX_THREADS 4

int party, port = 8000, dim = 1 << 16;
string address = "127.0.0.1";
int op = 1;
IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];
int bw_x = 32;
int num_threads = 4;

void test_msnzb(int tid, uint64_t *x, uint8_t *y, int32_t ldim) {
  if (ldim == 0)
    return;
  int lparty = (tid & 1 ? 3 - party : party);
  AuxProtocols *aux = new AuxProtocols(lparty, iopackArr[tid], otpackArr[tid]);
  aux->msnzb_GC(x, y, bw_x, ldim);
  delete aux;
}

void test_msnzb_one_hot(int tid, uint64_t *x, uint8_t *y, int32_t ldim) {
  if (ldim == 0)
    return;
  int lparty = (tid & 1 ? 3 - party : party);
  int digit_size = 8;
  AuxProtocols *aux = new AuxProtocols(lparty, iopackArr[tid], otpackArr[tid]);
  aux->msnzb_one_hot(x, y, bw_x, ldim, digit_size);
  delete aux;
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("o", op, "MSNZB Kind? 1 (GC)/2 (SIRNN)");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("N", dim, "Batch dim");
  amap.parse(argc, argv);

  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }

  PRG128 prg;
  uint64_t mask_x = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));

  uint64_t *x = new uint64_t[dim];
  uint8_t *y = new uint8_t[dim * bw_x];

  prg.random_data(x, dim * sizeof(uint64_t));
  for (int i = 0; i < dim; i++) {
    x[i] = x[i] & mask_x;
  }

  auto start = clock_start();
  std::thread bench_threads[num_threads];
  int chunk_size = dim / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lsize;
    if (i == (num_threads - 1)) {
      lsize = dim - offset;
    } else {
      lsize = chunk_size;
    }
    if (op == 1) {
      bench_threads[i] =
          std::thread(test_msnzb, i, x + offset, y + bw_x * offset, lsize);
    } else {
      bench_threads[i] = std::thread(test_msnzb_one_hot, i, x + offset,
                                     y + (bw_x * offset), lsize);
    }
  }
  for (int i = 0; i < num_threads; ++i) {
    bench_threads[i].join();
  }
  long long t = time_from(start);
  cout << "Time MSNZB GC: " << t / 1000.0 << " ms" << endl;

  if (party == ALICE) {
    iopackArr[0]->io->send_data(x, dim * sizeof(uint64_t));
    iopackArr[0]->io->send_data(y, dim * bw_x * sizeof(uint8_t));
  } else {
    uint64_t *x0 = new uint64_t[dim];
    uint8_t *y0 = new uint8_t[dim * bw_x];
    iopackArr[0]->io->recv_data(x0, dim * sizeof(uint64_t));
    iopackArr[0]->io->recv_data(y0, dim * bw_x * sizeof(uint8_t));

    for (int i = 0; i < dim; i++) {
      uint64_t actual_val = unsigned_val(x0[i] + x[i], bw_x);
      uint64_t secure_val = 0ULL;
      for (int j = 0; j < bw_x; j++) {
        secure_val +=
            (uint64_t(y[i * bw_x + j] ^ y0[i * bw_x + j]) * (1ULL << j));
      }
      if (actual_val == 0) {
        assert(secure_val == 0);
        continue;
      }
      if (!((actual_val >= secure_val) && (actual_val < 2 * secure_val))) {
        cout << i << "\t" << actual_val << "\t" << secure_val << "\t";
        for (int j = 0; j < bw_x; j++) {
          cout << uint64_t(y[i * bw_x + j] ^ y0[i * bw_x + j]);
        }
        cout << endl;
      }
      assert((actual_val >= secure_val) && (actual_val < 2 * secure_val));
    }
    std::cout << "Correct!" << std::endl;
    delete[] x0;
    delete[] y0;
  }
  delete[] x;
  delete[] y;
}
