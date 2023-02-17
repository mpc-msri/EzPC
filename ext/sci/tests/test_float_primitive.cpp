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
#include <random>
#include <limits>
#include "float_utils.h"

using namespace sci;
using namespace std;

enum class Op { ADD, MUL, DIV, SQRT, CHEAP_ADD, CHEAP_DIV };

enum class CmpOp { LT, GT, GE, LE };

Op op = Op::DIV;
CmpOp cmp_op = CmpOp::LT;
bool verbose = true;
IOPack *iopack = nullptr;
OTPack *otpack = nullptr;
FPOp *fp_op = nullptr;
FPMath *fp_math = nullptr;
int sz = 10000;
int party = 1;
string address = "127.0.0.1";
int port = 8000;
uint8_t m_bits = 23, e_bits = 8;
std::random_device rand_div;
// std::mt19937 generator(rand_div());
std::mt19937 generator(0);
float lb;
float ub;

#define f32_get_e(f) ((f & 0x7F800000) >> 23)
#define f32_get_m(f) (f & 0x007FFFFF)
#define f32_get_s(f) (f >> 31)
#define f32_is_denormal_number(f) (f32_get_e(f) == 0 && f32_get_m(f) != 0)
#define f32_is_nan(f) (f32_get_e(f) == 0xff && f32_get_m(f) != 0)
#define f32_is_inf(f) (f32_get_e(f) == 255 && f32_get_m(f) == 0)

bool check_limit(float x_1, float x_2, int32_t ULP_limit = 1) {
  if ((x_1 == INFINITY) && (x_2 == -INFINITY))
    return true;
  if ((x_1 == -INFINITY) && (x_2 == INFINITY))
    return true;
  if (x_1 == x_2)
    return true;
  int32_t x_1_int, x_2_int;
  x_1_int = *((int32_t *)&x_1);
  x_2_int = *((int32_t *)&x_2);
  int32_t diff = x_1_int - x_2_int;
  if (diff > ULP_limit || diff < -1 * ULP_limit)
    return false;
  else
    return true;
}

void test_assignment() {
  FPArray fp;
  float *f = new float[sz];
  for (int i = 0; i < sz; i++) {
    uint32_t fint = generator();
    f[i] = *((float *)&fint);
  }
  if (party == PUBLIC) {
    fp = fp_op->input<float>(PUBLIC, sz, f, m_bits, e_bits);
  } else {
    fp = fp_op->input<float>(ALICE, sz, f, m_bits, e_bits);
  }
  FPArray fp_ = fp_op->output(PUBLIC, fp);
  vector<float> f_ = fp_.get_native_type<float>();
  for (int i = 0; i < sz; i++) {
    uint32_t f_int = *((uint32_t *)&f[i]);
    FPArray fp_i = fp_.subset(i, i + 1);
    cout << i << "\t" << f[i] << "\t" << f_[i] << "\t" << fp_i << endl;
    if (!(f32_is_nan(f_int) || f32_is_denormal_number(f_int))) {
      assert(f[i] == f_[i]);
    }
  }
  delete[] f;
}

void test_op() {
  assert(party == ALICE || party == BOB);
  switch (op) {
  case Op::ADD:
  case Op::MUL:
  case Op::DIV:
  case Op::CHEAP_ADD:
  case Op::CHEAP_DIV:
    lb = -1*std::numeric_limits<float>::max();
    ub = std::numeric_limits<float>::max();
    break;
  case Op::SQRT:
    lb = std::numeric_limits<float>::min();
    ub = std::numeric_limits<float>::max();
    break;
  default:
    assert(false);
  }
  FPArray fp;
  float *f_1 = new float[sz];
  float *f_2 = new float[sz];
  float *f = new float[sz];
  for (int i = 0; i < sz; i++) {
    f_1[i] = sample_float(generator, lb, ub);
    f_2[i] = sample_float(generator, lb, ub);
  }
  FPArray fp_1 = fp_op->input<float>(ALICE, sz, f_1, m_bits, e_bits);
  FPArray fp_2 = fp_op->input<float>(BOB, sz, f_2, m_bits, e_bits);
  switch (op) {
  case Op::ADD:
    cout << "ADD" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = f_1[i] + f_2[i];
    }
    fp = fp_op->add(fp_1, fp_2);
    break;
  case Op::MUL:
    cout << "MUL" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = f_1[i] * f_2[i];
    }
    fp = fp_op->mul(fp_1, fp_2);
    break;
  case Op::DIV:
    cout << "DIV" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = f_1[i] / f_2[i];
    }
    fp = fp_op->div(fp_1, fp_2);
    break;
  case Op::SQRT:
    cout << "SQRT" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = sqrtf(f_1[i]);
    }
    fp = fp_op->sqrt(fp_1);
    break;
  case Op::CHEAP_ADD:
    cout << "CHEAP ADD" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = f_1[i] + f_2[i];
    }
    fp = fp_op->add(fp_1, fp_2, true);
    break;
  case Op::CHEAP_DIV:
    cout << "CHEAP DIV" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = f_1[i] / f_2[i];
    }
    fp = fp_op->div(fp_1, fp_2, true);
    break;
  default:
    assert(false);
  }
  FPArray fp_pub = fp_op->output(PUBLIC, fp);
  vector<float> f_ = fp_pub.get_native_type<float>();
  for (int i = 0; i < sz; i++) {
    uint32_t f_int_1 = *((uint32_t *)&f_1[i]);
    uint32_t f_int_2 = *((uint32_t *)&f_2[i]);
    uint32_t f_int = *((uint32_t *)&f[i]);
    if (verbose) {
      FPArray fp_i = fp_pub.subset(i, i + 1);
      cout << i << "\t" << f_1[i] << "\t" << f_2[i] << "\t" << f[i] << "\t"
           << f_[i] << "\t" << fp_i << endl;
    }
    if (op == Op::SQRT && (f_1[i] < 0))
      continue;
    if ((op == Op::DIV) && (f_2[i] == 0.0))
      continue;
    if ((f32_is_nan(f_int_1) || f32_is_denormal_number(f_int_1) ||
         f32_is_nan(f_int_2) || f32_is_denormal_number(f_int_2)))
      continue;
    if (f32_is_denormal_number(f_int))
      assert(f_[i] == 0.0);
    else {
      if ((op != Op::CHEAP_ADD && op != Op::CHEAP_DIV) && (m_bits == 23 && e_bits == 8)) {
        assert(f[i] == f_[i]);
      }
    }
  }
  delete[] f;
  delete[] f_1;
  delete[] f_2;
}

void test_cmp_op() {
  assert(party == ALICE || party == BOB);
  lb = -1*std::numeric_limits<float>::max();
  ub = std::numeric_limits<float>::max();
  BoolArray bp;
  float *f_1 = new float[sz];
  float *f_2 = new float[sz];
  uint8_t *b = new uint8_t[sz];
  uint8_t *b_ = new uint8_t[sz];
  for (int i = 0; i < sz; i++) {
    f_1[i] = sample_float(generator, lb, ub);
    f_2[i] = sample_float(generator, lb, ub);
  }
  FPArray fp_1 = fp_op->input<float>(ALICE, sz, f_1, m_bits, e_bits);
  FPArray fp_2 = fp_op->input<float>(BOB, sz, f_2, m_bits, e_bits);
  switch (cmp_op) {
  case CmpOp::LT:
    cout << "LT" << endl;
    for (int i = 0; i < sz; i++) {
      b[i] = f_1[i] < f_2[i];
    }
    bp = fp_op->LT(fp_1, fp_2);
    break;
  case CmpOp::GT:
    cout << "GT" << endl;
    for (int i = 0; i < sz; i++) {
      b[i] = f_1[i] > f_2[i];
    }
    bp = fp_op->GT(fp_1, fp_2);
    break;
  case CmpOp::LE:
    cout << "LE" << endl;
    for (int i = 0; i < sz; i++) {
      b[i] = f_1[i] <= f_2[i];
    }
    bp = fp_op->LE(fp_1, fp_2);
    break;
  case CmpOp::GE:
    cout << "GE" << endl;
    for (int i = 0; i < sz; i++) {
      b[i] = f_1[i] >= f_2[i];
    }
    bp = fp_op->GE(fp_1, fp_2);
    break;
  default:
    assert(false);
  }
  BoolArray bp_pub = fp_op->bool_op->output(PUBLIC, bp);
  memcpy(b_, bp_pub.data, sz * sizeof(uint8_t));
  for (int i = 0; i < sz; i++) {
    uint32_t f_int_1 = *((uint32_t *)&f_1[i]);
    uint32_t f_int_2 = *((uint32_t *)&f_2[i]);
    if (verbose) {
      cout << i << "\t" << f_1[i] << "\t" << f_2[i] << "\t" << int(b[i]) << "\t"
           << int(b_[i]) << endl;
    }
    if (f32_is_nan(f_int_1) || f32_is_nan(f_int_2))
      continue;
    assert(b[i] == b_[i]);
  }
  delete[] b;
  delete[] b_;
  delete[] f_1;
  delete[] f_2;
}

void test_int_to_float() {
  bool signed_ = true;
  FPArray fp;
  uint64_t *f_1 = new uint64_t[sz];
  float *f = new float[sz];
  for (int i = 0; i < sz; i++) {
    uint32_t fint = generator();
    f_1[i] = fint;
  }
  FixArray fx = fp_op->fix->input(ALICE, sz, f_1, signed_, 32, 0);

  cout << "INT TO FLOAT" << endl;
  fp = fp_op->int_to_float(fx, 23, 8);

  FPArray fp_pub = fp_op->output(PUBLIC, fp);
  vector<float> f_ = fp_pub.get_native_type<float>();
  for (int i = 0; i < sz; i++) {
    if (signed_) {
      f[i] = float(int32_t(f_1[i]));
    } else {
      f[i] = float(f_1[i]);
    }
    if (verbose) {
      FPArray fp_i = fp_pub.subset(i, i + 1);
      cout << i << "\t" << f_1[i] << "\t" << f[i] << "\t" << f_[i] << "\t"
           << fp_i << endl;
    }
    assert(f[i] == f_[i]);
  }
  delete[] f;
  delete[] f_1;
}

int main(int argc, char **argv) {
  cout.precision(15);

  ArgMapping amap;

  int int_op = static_cast<int>(op);
  int int_cmp_op = static_cast<int>(cmp_op);
  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("o", int_op, "FP Primitve Operation");
  amap.arg("c", int_cmp_op, "FP Comparison Operation");
  amap.arg("v", verbose, "Print test inputs/outputs?");
  amap.parse(argc, argv);
  op = static_cast<Op>(int_op);
  cmp_op = static_cast<CmpOp>(int_cmp_op);

  iopack = new IOPack(party, port, address);
  otpack = new OTPack(iopack, party);

  fp_op = new FPOp(party, iopack, otpack);
  fp_math = new FPMath(party, iopack, otpack);

  auto start = clock_start();
  uint64_t comm_start = iopack->get_comm();
  uint64_t initial_rounds = iopack->get_rounds();
  // test_int_to_float();
  test_op();
  // test_cmp_op();
  uint64_t comm_end = iopack->get_comm();
  long long t = time_from(start);
  cout << "Comm. per operations: " << 8 * (comm_end - comm_start) / sz
       << " bits" << endl;
  cout << "Number of FP ops/s:\t" << (double(sz) / t) * 1e6 << std::endl;
  cout << "Total Time:\t" << t / (1000.0) << " ms" << endl;
  cout << "Num_rounds: " << (iopack->get_rounds() - initial_rounds) << endl;
  cout << "lb: " << lb << endl;
  cout << "ub: " << ub << endl;

  delete iopack;
  delete otpack;
  delete fp_op;
  delete fp_math;
}
