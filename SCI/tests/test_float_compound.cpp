/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2022 Microsoft Research
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

enum class Op { DOT_PRODUCT, MATMUL, SOFTMAX };

Op op = Op::MATMUL;
bool verbose = true;
IOPack *iopack = nullptr;
OTPack *otpack = nullptr;
FPOp *fp_op = nullptr;
FPMath *fp_math = nullptr;
int n1 = 10;
int n2 = 128;
int n3 = 10;
int sz = n1*n3;
int party = 1;
string address = "127.0.0.1";
int port = 8000;
// uint8_t m_bits = BFLOAT16_M_BITS, e_bits = BFLOAT16_E_BITS;
uint8_t m_bits = FP32_M_BITS, e_bits = FP32_E_BITS;
std::mt19937 generator(0);
float lb;
float ub;
double total_ULP_err = 0.0;

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

void test_op() {
  assert(party == ALICE || party == BOB);
  switch (op) {
  case Op::DOT_PRODUCT:
  case Op::MATMUL:
    /* lb = -1*std::numeric_limits<float>::max(); */
    /* ub = std::numeric_limits<float>::max(); */
    lb = -10000;
    ub = 10000;
    break;
  case Op::SOFTMAX:
    /* lb = -1*std::numeric_limits<float>::max(); */
    /* ub = std::numeric_limits<float>::max(); */
    lb = pow(2.0, -9);
    ub = 10;
    break;
  default:
    assert(false);
  }
  FPArray fp;
  FPMatrix fpm;
  float *f_1, *f_2, *f;
  double *f_dbl;
  vector<FPArray> fp_1;
  vector<FPArray> fp_2;
  FPMatrix fpm_1;
  FPMatrix fpm_2;
  FPArray fp_pub;
  FPMatrix fpm_pub;
  vector<float> f_;
  switch (op) {
  case Op::DOT_PRODUCT:
    f_1 = new float[sz*n2];
    f_2 = new float[sz*n2];
    f = new float[sz*n2];
    f_dbl = new double[sz*n2];
    for (int i = 0; i < sz*n2; i++) {
      f_1[i] = sample_float(generator, lb, ub);
      f_2[i] = sample_float(generator, lb, ub);
    }
    fp_1.resize(sz); fp_2.resize(sz);
    for (int i = 0; i < sz; i++) {
        fp_1[i] = fp_op->input<float>(ALICE, n2, f_1 + i*n2, m_bits, e_bits);
        fp_2[i] = fp_op->input<float>(BOB, n2, f_2 + i*n2, m_bits, e_bits);
        FPArray fpm_1_pub = fp_op->output(PUBLIC, fp_1[i]);
        FPArray fpm_2_pub = fp_op->output(PUBLIC, fp_2[i]);
        vector<float> f_1_vec = fpm_1_pub.get_native_type<float>();
        vector<float> f_2_vec = fpm_2_pub.get_native_type<float>();
        for (int j = 0; j < n2; j++) {
          f_1[i*n2 + j] = f_1_vec[j];
          f_2[i*n2 + j] = f_2_vec[j];
        }
    }
    for (int i = 0; i < sz; i++) {
        f[i] = 0;
        f_dbl[i] = 0;
        for (int j = 0; j < n2; j++) {
          f[i] += (f_1[i*n2 + j] * f_2[i*n2 + j]);
          f_dbl[i] += (double(f_1[i*n2 + j]) * double(f_2[i*n2 + j]));
        }
    }
    fp = fp_op->dot_product(fp_1, fp_2);
    fp_pub = fp_op->output(PUBLIC, fp);
    f_ = fp_pub.get_native_type<float>();
    for (int i = 0; i < sz; i++) {
      uint32_t f_int = *((uint32_t *)&f[i]);
      if (verbose) {
        FPArray fp_i = fp_pub.subset(i, i + 1);
        cout << i << "\t" << f_dbl[i] << "\t" << f[i] << "\t" << f_[i] << "\t" << fp_i << "\t" << ULP_error(f[i], f_dbl[i]) << "\t" << ULP_error(f_[i], f_dbl[i]) << endl;
      }
      total_ULP_err += ULP_error(f_[i], f_dbl[i]);
    }
    break;
  case Op::MATMUL:
    f_1 = new float[n1*n2];
    f_2 = new float[n2*n3];
    f = new float[n1*n3];
    f_dbl = new double[n1*n3];
    for (int i = 0; i < n1*n2; i++) {
      f_1[i] = sample_float(generator, lb, ub);
    }
    for (int i = 0; i < n2*n3; i++) {
      f_2[i] = sample_float(generator, lb, ub);
    }
    fpm_1 = fp_op->input<float>(ALICE, n1, n2, f_1, m_bits, e_bits);
    fpm_2 = fp_op->input<float>(BOB, n2, n3, f_2, m_bits, e_bits);
    {
      FPArray fpm_1_pub = fp_op->output(PUBLIC, fpm_1);
      FPArray fpm_2_pub = fp_op->output(PUBLIC, fpm_2);
      vector<float> f_1_vec = fpm_1_pub.get_native_type<float>();
      vector<float> f_2_vec = fpm_2_pub.get_native_type<float>();
      for (int i = 0; i < n1*n2; i++) {
        f_1[i] = f_1_vec[i];
      }
      for (int i = 0; i < n2*n3; i++) {
        f_2[i] = f_2_vec[i];
      }
    }
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n3; j++) {
        f[i*n3 + j] = 0;
        f_dbl[i*n3 + j] = 0;
        double tmp[n2];
        for (int k = 0; k < n2; k++) {
          f[i*n3 + j] += (f_1[i*n2 + k] * f_2[k*n3 + j]);
          tmp[k] = double(f_1[i*n2 + k]) * double(f_2[k*n3 + j]);
        }
        for (int k = 1; k < n2; k *= 2) {
          for (int l = 0; l < n2 and l + k < n2; l += 2*k) {
            tmp[l] = tmp[l] + tmp[l+k];
          }
        }
        f_dbl[i*n3+j] = tmp[0];
      }
    }
    fpm = fp_op->matrix_multiplication_beacon(fpm_1, fpm_2);
    fpm_pub = fp_op->output(PUBLIC, fpm);
    f_ = fpm_pub.get_native_type<float>();
    for (int i = 0; i < n1*n3; i++) {
      uint32_t f_int = *((uint32_t *)&f[i]);
      if (verbose) {
        FPArray fp_i = static_cast<FPArray>(fpm_pub).subset(i, i + 1);
        cout << i << "\t" << f_dbl[i] << "\t" << f[i] << "\t" << f_[i] << "\t" << fp_i << "\t" << ULP_error(f[i], f_dbl[i]) << "\t" << ULP_error(f_[i], f_dbl[i]) << endl;
      }
      total_ULP_err += ULP_error(f_[i], f_dbl[i]);
    }
    break;
  case Op::SOFTMAX:
    f_1 = new float[sz*n2];
    f_2 = new float[sz*n2];
    f = new float[sz*n2];
    f_dbl = new double[sz*n2];
    for (int i = 0; i < sz*n2; i++) {
      f_1[i] = sample_float(generator, lb, ub);
    }
    fp_1.resize(sz);
    for (int i = 0; i < sz; i++) {
        fp_1[i] = fp_op->input<float>(ALICE, n2, f_1 + i*n2, m_bits, e_bits);
        FPArray fpm_1_pub = fp_op->output(PUBLIC, fp_1[i]);
        vector<float> f_1_vec = fpm_1_pub.get_native_type<float>();
        for (int j = 0; j < n2; j++) {
          f_1[i*n2 + j] = f_1_vec[j];
        }
    }
    for (int i = 0; i < sz; i++) {
        float f_max = f_1[i*n2];
        for (int j = 0; j < n2; j++) {
          f_max = max(f_1[i*n2 + j], f_max);
        }
        float f_shifted[n2];
        double f_shifted_dbl[n2];
        float sum = 0.0;
        float sum_dbl = 0.0;
        for (int j = 0; j < n2; j++) {
          f_shifted[j] = f_1[i*n2 + j] - f_max;
          f_shifted_dbl[j] = f_1[i*n2 + j] - double(f_max);
          f_shifted[j] = expf(f_shifted[j]);
          f_shifted_dbl[j] = exp(f_shifted_dbl[j]);
          sum += f_shifted[j];
          sum_dbl += f_shifted_dbl[j];
        }
        for (int j = 0; j < n2; j++) {
          f[i*n2 + j] = f_shifted[j]/sum;
          f_dbl[i*n2 + j] = f_shifted_dbl[j]/sum_dbl;
        }
    }
    {
      vector<FPArray> fp_vec = fp_math->softmax_beacon(fp_1);
      FPArray fp_vec_concat = concat(fp_vec);
      fp_pub = fp_op->output(PUBLIC, fp_vec_concat);
    }
    f_ = fp_pub.get_native_type<float>();
    for (int i = 0; i < sz*n2; i++) {
      uint32_t f_int = *((uint32_t *)&f[i]);
      if (verbose) {
        FPArray fp_i = fp_pub.subset(i, i + 1);
        cout << i << "\t" << f_1[i] << "\t" << f_dbl[i] << "\t" << f[i] << "\t" << f_[i] << "\t" << fp_i << "\t" << ULP_error(f[i], f_dbl[i]) << "\t" << ULP_error(f_[i], f_dbl[i]) << endl;
      }
      total_ULP_err += ULP_error(f_[i], f_dbl[i]);
    }
    break;
  default:
    assert(false);
  }
  delete[] f;
  delete[] f_dbl;
  delete[] f_1;
  delete[] f_2;
}

int main(int argc, char **argv) {
  cout.precision(15);

  ArgMapping amap;

  int int_op = static_cast<int>(op);
  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("o", int_op, "FP Primitve Operation");
  amap.parse(argc, argv);
  op = static_cast<Op>(int_op);

  iopack = new IOPack(party, port, address);
  otpack = new OTPack(iopack, party);

  fp_op = new FPOp(party, iopack, otpack);
  fp_math = new FPMath(party, iopack, otpack);

  auto start = clock_start();
  uint64_t comm_start = iopack->get_comm();
  uint64_t initial_rounds = iopack->get_rounds();
  test_op();
  uint64_t comm_end = iopack->get_comm();
  long long t = time_from(start);
  cout << "Comm. per operations: " << 8 * (comm_end - comm_start) / (sz)
       << " bits" << endl;
  cout << "Number of FP ops/s:\t" << (double(sz) / t) * 1e6 << std::endl;
  cout << "Total Time:\t" << t / (1000.0) << " ms" << endl;
  cout << "Num_rounds: " << (iopack->get_rounds() - initial_rounds) << endl;
  cout << "lb: " << lb << endl;
  cout << "ub: " << ub << endl;
  cout << "Total ULP Error: " << total_ULP_err << endl;

  delete iopack;
  delete otpack;
  delete fp_op;
}
