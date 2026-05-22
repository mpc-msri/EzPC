/*
Authors: Deevashwer Rathee
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

#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include <cfenv>
#include <fstream>
#include <random>
#include <thread>

using namespace sci;
using namespace std;

enum class Op {
  GT,
  ADD,
  MUL,
  DIV,
  SQRT,
  SINPI,
  COSPI,
  TANPI,
  LOG2,
  EXP2,
  EXP,
  LN,
  ERF
};

#define MAX_THREADS 4

Op op = Op::ADD;
string op_name = "ADD";
bool verbose = true;
IOPack *iopackArr[MAX_THREADS];
OTPack *otpackArr[MAX_THREADS];
int dim = 8192;
int party = 1;
string address = "127.0.0.1";
int port = 32000;
int num_threads = 4;
std::random_device rand_div;
// std::mt19937 generator(rand_div());
std::mt19937 generator(0);

void bench_op(int tid, int sz) {
  if (sz == 0)
    return;
  int lparty = (tid & 1 ? 3 - party : party);
  FPOp *fp_op;
  FPMath *fp_math;
  fp_op = new FPOp(lparty, iopackArr[tid], otpackArr[tid]);
  fp_math = new FPMath(lparty, iopackArr[tid], otpackArr[tid]);
  assert(party == ALICE || party == BOB);
  BoolArray bp;
  FPArray fp;
  float *f_1 = new float[sz];
  float *f_2 = new float[sz];
  for (int i = 0; i < sz; i++) {
    uint32_t fint = generator();
    f_1[i] = *((float *)&fint);
    fint = generator();
    f_2[i] = *((float *)&fint);
  }
  FPArray fp_1 = fp_op->input<float>(ALICE, sz, f_1);
  FPArray fp_2 = fp_op->input<float>(BOB, sz, f_2);
  switch (op) {
  case Op::GT:
    bp = fp_op->GT(fp_1, fp_2);
    op_name = "GT";
    break;
  case Op::ADD:
    fp = fp_op->add(fp_1, fp_2);
    op_name = "ADD";
    break;
  case Op::MUL:
    fp = fp_op->mul(fp_1, fp_2);
    op_name = "MUL";
    break;
  case Op::DIV:
    fp = fp_op->div(fp_1, fp_2);
    op_name = "DIV";
    break;
  case Op::SQRT:
    fp = fp_op->sqrt(fp_1);
    op_name = "SQRT";
    break;
  case Op::SINPI:
    fp = fp_math->sinpi(fp_1);
    op_name = "SINPI";
    break;
  case Op::COSPI:
    fp = fp_math->cospi(fp_1);
    op_name = "COSPI";
    break;
  case Op::TANPI:
    fp = fp_math->tanpi(fp_1);
    op_name = "TANPI";
    break;
  case Op::EXP2:
    fp = fp_math->exp2(fp_1);
    op_name = "EXP2";
    break;
  case Op::LOG2:
    fp = fp_math->log2(fp_1);
    op_name = "LOG2";
    break;
  case Op::EXP:
    fp = fp_math->exp(fp_1);
    op_name = "EXP";
    break;
  case Op::LN:
    fp = fp_math->ln(fp_1);
    op_name = "LN";
    break;
  case Op::ERF:
    fp = fp_math->erf(fp_1);
    op_name = "ERF";
    break;
  default:
    assert(false);
  }
  delete[] f_1;
  delete[] f_2;
  delete fp_op;
  delete fp_math;
}

int main(int argc, char **argv) {
  ArgMapping amap;

  int int_op = static_cast<int>(op);
  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("o", int_op, "FP Operation");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("N", dim, "Batch dim");
  amap.parse(argc, argv);
  op = static_cast<Op>(int_op);

  assert(num_threads <= MAX_THREADS);

  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new sci::IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }
  std::cout << "All Base OTs Done" << std::endl;

  uint64_t num_rounds = iopackArr[0]->get_rounds();
  uint64_t total_comm = 0;
  uint64_t thread_comm[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm();
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
    bench_threads[i] = std::thread(bench_op, i, lsize);
  }
  for (int i = 0; i < num_threads; ++i) {
    bench_threads[i].join();
  }
  long long t = time_from(start);

  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm() - thread_comm[i];
    total_comm += thread_comm[i];
  }
  num_rounds = iopackArr[0]->get_rounds() - num_rounds;

  if (party == BOB) {
    uint64_t total_comm_ALICE = 0;
    iopackArr[0]->io->recv_data(&total_comm_ALICE, sizeof(uint64_t));
    total_comm += total_comm_ALICE;
    cout << "Total Communication (ALICE + BOB)\t" << total_comm << " bytes"
         << endl;
    string file_addr;
    file_addr = "FP-op.csv";
    bool write_title = true;
    {
      fstream result(file_addr.c_str(), fstream::in);
      if (result.is_open())
        write_title = false;
      result.close();
    }
    fstream result(file_addr.c_str(), fstream::out | fstream::app);
    if (write_title) {
      result << "Operator,Using GC?,#Operations,#Threads,Time "
                "(ms),Communication (Bytes),Total Rounds"
             << endl;
    }
    result << op_name << "," << dim << "," << num_threads << "," << t / (1000.0)
           << "," << total_comm << "," << num_rounds << endl;
    result.close();
  } else {
    iopackArr[0]->io->send_data(&total_comm, sizeof(uint64_t));
  }
  cout << "Ops/s:\t" << (double(dim) / t) * 1e6 << std::endl;
  cout << "Total Time\t" << t / (1000.0) << " ms" << endl;

  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }
}
