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

#include "FloatingPoint/fp-math.h"
#include "mpfr.h"
#include <random>
#include <limits>
#include "float_utils.h"

#define MPFR_PREC 300

using namespace sci;
using namespace std;

mpfr_t mval;

enum class Op { SINPI, COSPI, TANPI, EXP2, LOG2, EXP, LN, ERF };

Op op = Op::EXP;
bool verbose = true;
IOPack *iopack = nullptr;
OTPack *otpack = nullptr;
FPMath *fp_math = nullptr;
int m_bits = 23, e_bits = 8;
int sz = 8192;
int party = 1;
string address = "127.0.0.1";
int port = 8000;
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

int64_t get_exponent(double x) {
  int64_t x_int = *((int64_t *)&x);
  return ((x_int >> 52) & 2047) - 1023;
}

double ULP_error(float actual, double expected) {
  if (abs(expected) < pow(2, -126)) {
    if (actual == 0.0)
      expected = 0;
  }
  if (abs(expected) >= pow(2, 128)) {
    if (expected < 0)
      expected = -INFINITY;
    else
      expected = INFINITY;
  }
  if ((actual == INFINITY) && (expected == -INFINITY))
    return 0.0;
  if ((actual == -INFINITY) && (expected == INFINITY))
    return 0.0;
  if (actual == expected)
    return 0.0;
  double abs_err = abs(double(actual) - expected);
  int64_t expected_exp = get_exponent(expected);
  double ulp;
  ulp = exp2(expected_exp - 23.0);
  return abs_err / ulp;
}

bool check_limit(float x_1, double x_2, int32_t ULP_limit = 1) {
  double ulp = ULP_error(x_1, x_2);
  if (ulp > ULP_limit)
    return false;
  else
    return true;
}

// x <= 2^-14, tanpi(x) = pi*x in float representation
double precise_tanpi(float x) {
  mpfr_const_pi(mval, MPFR_RNDN);
  mpfr_mul_d(mval, mval, (double)x, MPFR_RNDN);
  mpfr_tan(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

// x <= 2^-14, sinpi(x) = pi*x in float representation
double precise_sinpi(float x) {
  mpfr_const_pi(mval, MPFR_RNDN);
  mpfr_mul_d(mval, mval, (double)x, MPFR_RNDN);
  mpfr_sin(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

double precise_cospi(float x) {
  mpfr_const_pi(mval, MPFR_RNDN);
  mpfr_mul_d(mval, mval, (double)x, MPFR_RNDN);
  mpfr_cos(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

double precise_exp2(float x) {
  mpfr_set_flt(mval, x, MPFR_RNDN);
  mpfr_exp2(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

double precise_log2(float x) {
  mpfr_set_flt(mval, x, MPFR_RNDN);
  mpfr_log2(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

double precise_ln(float x) {
  mpfr_set_flt(mval, x, MPFR_RNDN);
  mpfr_log(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

double precise_exp(float x) {
  mpfr_set_flt(mval, x, MPFR_RNDN);
  mpfr_exp(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

double precise_erf(float x) {
  mpfr_set_flt(mval, x, MPFR_RNDN);
  mpfr_erf(mval, mval, MPFR_RNDN);
  return mpfr_get_d(mval, MPFR_RNDN);
}

void test_op() {
  assert(party == ALICE || party == BOB);
  switch (op) {
  case Op::SINPI:
  case Op::COSPI:
  case Op::TANPI:
    lb = pow(2.0, -14);
    ub = pow(2.0, 23);
    break;
  case Op::EXP2:
    lb = -126;
    ub = pow(2.0, 7);
    break;
  case Op::EXP:
    lb = -87.33654022216796875;
    ub = 88.72283172607421875;
    break;
  case Op::LOG2:
  case Op::LN:
    lb = std::numeric_limits<float>::min();
    ub = std::numeric_limits<float>::max();
    break;
  case Op::ERF:
    lb = pow(2.0, -12);
    ub = 3.875;
    break;
  default:
    assert(false);
  }
  FPArray fp;
  float *f_1 = new float[sz];
  double *f = new double[sz];
  for (int i = 0; i < sz; i++) {
    f_1[i] = sample_float(generator, lb, ub);
  }
  FPArray fp_1 = fp_math->fp_op->input<float>(ALICE, sz, f_1, m_bits, e_bits);
  switch (op) {
  case Op::SINPI:
    cout << "SINPI" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_sinpi(f_1[i]);
    }
    fp = fp_math->sinpi(fp_1);
    break;
  case Op::COSPI:
    cout << "COSPI" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_cospi(f_1[i]);
    }
    fp = fp_math->cospi(fp_1);
    break;
  case Op::TANPI:
    cout << "TANPI" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_tanpi(f_1[i]);
    }
    fp = fp_math->tanpi(fp_1);
    break;
  case Op::EXP2:
    cout << "EXP2" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_exp2(f_1[i]);
    }
    fp = fp_math->exp2(fp_1);
    break;
  case Op::LOG2:
    cout << "LOG2" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_log2(f_1[i]);
    }
    fp = fp_math->log2(fp_1);
    break;
  case Op::LN:
    cout << "LN" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_ln(f_1[i]);
    }
    fp = fp_math->ln(fp_1);
    break;
  case Op::EXP:
    cout << "EXP" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_exp(f_1[i]);
    }
    fp = fp_math->exp(fp_1);
    break;
  case Op::ERF:
    cout << "ERF" << endl;
    for (int i = 0; i < sz; i++) {
      f[i] = precise_erf(f_1[i]);
    }
    fp = fp_math->erf(fp_1);
    break;
  default:
    assert(false);
  }
  FPArray fp_pub = fp_math->fp_op->output(PUBLIC, fp);
  vector<float> f_ = fp_pub.get_native_type<float>();
  for (int i = 0; i < sz; i++) {
    uint32_t f_int_1 = *((uint32_t *)&f_1[i]);
    uint32_t f_int = *((uint32_t *)&f[i]);
    double ulp_err = ULP_error(f_[i], f[i]);
    if (verbose) {
      FPArray fp_i = fp_pub.subset(i, i + 1);
      cout << i << "\t" << f_1[i] << "\t" << f[i] << "\t" << f_[i] << "\t"
           << fp_i << "\t" << ulp_err << endl;
    }
    if (op == Op::TANPI &&
        (float(f[i]) == INFINITY || float(f[i]) == -INFINITY))
      continue;
    if (op == Op::LOG2 && f_1[i] == 0.0)
      continue;
    if (f32_is_nan(f_int_1) || f32_is_denormal_number(f_int_1))
      continue;
    else {
      assert(check_limit(f_[i], f[i]));
    }
  }
  delete[] f;
  delete[] f_1;
}

int main(int argc, char **argv) {
  cout.precision(15);
  mpfr_init2(mval, MPFR_PREC);

  ArgMapping amap;

  int int_op = static_cast<int>(op);
  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("o", int_op, "FP Operation");
  amap.arg("v", verbose, "Print test inputs/outputs?");
  amap.parse(argc, argv);
  op = static_cast<Op>(int_op);

  iopack = new IOPack(party, port, address);
  otpack = new OTPack(iopack, party);

  fp_math = new FPMath(party, iopack, otpack);

  auto start = clock_start();
  uint64_t comm_start = iopack->get_comm();
  uint64_t initial_rounds = iopack->get_rounds();
  test_op();
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
  mpfr_clear(mval);
}
