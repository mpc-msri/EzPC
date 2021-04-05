/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
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

#include "library_fixed.h"
#include <fstream>
#include <iostream>
#include <random>
#include <thread>

using namespace std;

#define MAX_THREADS 4

int party = 1;
string address = "127.0.0.1";
int port = 8000;
int num_threads = 4;
int bitlength = 32;

void test_matmul() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int I = 1024;
  int K = 4;
  int J = 16;
  int shrA = 4;
  int shrB = 8;
  int H1 = 2;
  int H2 = 0;
  int demote = 1;
  int64_t *A = new int64_t[I * K];
  for (int i = 0; i < I * K; i++) {
    A[i] = (int16_t)generator();
  }
  int64_t *B = new int64_t[K * J];
  for (int i = 0; i < K * J; i++) {
    B[i] = (int16_t)(party == 1 ? generator() : 0);
  }
  int64_t *C = new int64_t[I * J];
  initialize();
  // int64_t* tmp = new int64_t[K];

  MatMul(I, K, J, shrA, shrB, H1, H2, demote, 16, 16, 32, 16, A, B, C, nullptr);

  finalize();
}
//

void test_convolution() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int N = 1;
  int H = 230;
  int W = 230;
  int CIN = 3;
  int HF = 7;
  int WF = 7;
  int CINF = 3;
  int COUTF = 64;
  int HOUT = 112;
  int WOUT = 112;
  int HPADL = 0;
  int HPADR = 0;
  int WPADL = 0;
  int WPADR = 0;
  int HSTR = 2;
  int WSTR = 2;
  int HDL = 1;
  int WDL = 1;
  int G = 1;
  int shrA = 1;
  int shrB = 1;
  int H1 = 18;
  int H2 = 0;
  int demote = 1;
  int64_t *A = new int64_t[N * H * W * CIN];
  for (int i = 0; i < N * H * W * CIN; i++) {
    A[i] = (int16_t)generator();
  }
  int64_t *B = new int64_t[G * HF * WF * CINF * COUTF];
  for (int i = 0; i < G * HF * WF * CINF * COUTF; i++) {
    B[i] = (party == 1 ? (int16_t)generator() : 0);
  }
  int64_t *C = new int64_t[N * HOUT * WOUT * COUTF * G];
  // for (int i = 0; i < N*HOUT*WOUT*COUTF*G; i++) {
  //     C[i] = (int16_t)generator();
  // }
  // int64_t* tmp = new int64_t[HF*WF*CINF + 100];
  initialize();
  Convolution(N, H, W, CIN, HF, WF, CINF, COUTF, HOUT, WOUT, HPADL, HPADR,
              WPADL, WPADR, HSTR, WSTR, HDL, WDL, G, shrA, shrB, H1, H2, demote,
              16, 16, 32, 16, A, B, C, nullptr);

  finalize();
}
//

void test_BNorm() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int32_t I = 1 * 30 * 40;
  int32_t J = 128;
  int32_t shA = 1;
  int32_t shBNB = 0;
  int32_t shB = 14;
  int64_t *A = new int64_t[I * J];
  for (int i = 0; i < I * J; i++) {
    A[i] = (int16_t)generator();
  }
  int64_t *BNW = new int64_t[J];
  int64_t *BNB = new int64_t[J];
  for (int i = 0; i < J; i++) {
    BNW[i] = party == 1 ? (int16_t)generator() : 0;
    BNB[i] = party == 1 ? (int16_t)generator() : 0;
  }
  int64_t *B = new int64_t[I * J];

  initialize();

  BNorm(I, J, shA, shBNB, shB, 16, 16, 16, 32, 16, A, BNW, BNB, B);

  finalize();

  delete[] A;
  delete[] BNW;
  delete[] BNB;
  delete[] B;
}

// MBConv<int16_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t,
// int8_t, int8_t, int16_t, int16_t, int16_t, int32_t, int32_t, int32_t,
// int32_t>(&tmp378_16[0][0][0][0], &L9F1[0][0][0][0][0], &L9W1[0], &L9B1[0],
// &L9F2[0][0][0][0][0], &L9W2[0], &L9B2[0], &L9F3[0][0][0][0][0], &L9W3[0],
// &L9B3[0], &tmp391_16[0][0][0][0], &tmp392_16[0][0][0], &tmp393_16[0],
// &tmp390_32[0], 1, 15, 20, 96, 192, 3, 3, 96, 15, 20, 1, 1, 1, 1, 1, 1, 7, 4,
// 8, 1572864L, 393216L, 1, 1, 64, 8, 1, 16, 1, 1, 64, 1, 8, 1, 1, 16, 1, 2, 64,
// 1);

void test_MBConv() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int N = 1;
  int H = 15;
  int W = 20;
  int Cin = 96;
  int Ct = 192;
  int HF = 3;
  int WF = 3;
  int Cout = 96;
  int Hout = 15;
  int Wout = 20;
  int HPADL = 1;
  int HPADR = 1;
  int WPADL = 1;
  int WPADR = 1;
  int HSTR = 1;
  int WSTR = 1;
  int D1 = 7;
  int D2 = 4;
  int D3 = 8;
  int32_t SIX_1 = 1572864L;
  int32_t SIX_2 = 393216L;
  int shr1 = 1;
  int shr2 = 1;
  int shr3 = 64;
  int shr4 = 8;
  int shr5 = 1;
  int shr6 = 16;
  int shr7 = 1;
  int shr8 = 1;
  int shr9 = 64;
  int shl1 = 1;
  int shl2 = 8;
  int shl3 = 1;
  int shl4 = 1;
  int shl5 = 16;
  int shl6 = 1;
  int shl7 = 2;
  int shl8 = 64;
  int shl9 = 1;

  int64_t *A = new int64_t[N * H * W * Cin];
  int64_t *F1 = new int64_t[Cin * Ct];
  int64_t *BN1W = new int64_t[Ct];
  int64_t *BN1B = new int64_t[Ct];
  int64_t *F2 = new int64_t[Ct * HF * WF];
  int64_t *BN2W = new int64_t[Ct];
  int64_t *BN2B = new int64_t[Ct];
  int64_t *F3 = new int64_t[Ct * Cout];
  int64_t *BN3W = new int64_t[Cout];
  int64_t *BN3B = new int64_t[Cout];

  for (int i = 0; i < N * H * W * Cin; i++) {
    A[i] = (int16_t)generator();
  }
  for (int i = 0; i < Cin * Ct; i++) {
    F1[i] = (int8_t)(party == 1 ? generator() : 0);
  }
  for (int i = 0; i < Ct * HF * WF; i++) {
    F2[i] = (int8_t)(party == 1 ? generator() : 0);
  }
  for (int i = 0; i < Ct * Cout; i++) {
    F3[i] = (int8_t)(party == 1 ? generator() : 0);
  }
  for (int i = 0; i < Ct; i++) {
    BN1W[i] = (int8_t)(party == 1 ? generator() : 0);
    BN1B[i] = (int8_t)(party == 1 ? generator() : 0);
    BN2W[i] = (int8_t)(party == 1 ? generator() : 0);
    BN2B[i] = (int8_t)(party == 1 ? generator() : 0);
  }
  for (int i = 0; i < Cout; i++) {
    BN3W[i] = (int8_t)(party == 1 ? generator() : 0);
    BN3B[i] = (int8_t)(party == 1 ? generator() : 0);
  }

  int64_t *C = new int64_t[N * Hout * Wout * Cout];
  // int64_t* C_layerwise = new int64_t[N*Hout*Wout*Cout];
  // int64_t* C_seedot = new int64_t[N*Hout*Wout*Cout];

  initialize();

  MBConv(N, H, W, Cin, Ct, HF, WF, Cout, Hout, Wout, HPADL, HPADR, WPADL, WPADR,
         HSTR, WSTR, D1, D2, D3, SIX_1, SIX_2, shr1, shr2, shr3, shr4, shr5,
         shr6, shr7, shr8, shr9, shl1, shl2, shl3, shl4, shl5, shl6, shl7, shl8,
         shl9, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, A, F1,
         BN1W, BN1B, F2, BN2W, BN2B, F3, BN3W, BN3B, C, nullptr, nullptr,
         nullptr);
  // cleartext_MBConv<int16_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t,
  // int8_t, int8_t, int8_t, int16_t, int16_t, int16_t, int16_t, int32_t,
  // int32_t, int32_t>(A, F1, BN1W, BN1B, F2, BN2W, BN2B, F3, BN3W, BN3B, C,
  // nullptr, nullptr, nullptr, N, H, W, Cin, Ct, HF, WF, Cout, Hout, Wout,
  // HPADL, HPADR, WPADL, WPADR, HSTR, WSTR, D1, D2, D3, SIX_1, SIX_2, shr1,
  // shr2, shr3, shr4, shr5, shr6, shr7, shr8, shr9, shl1, shl2, shl3, shl4,
  // shl5, shl6, shl7, shl8, shl9); cleartext_MBConv_layerwise<int16_t, int8_t,
  // int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int8_t, int16_t,
  // int16_t, int16_t, int16_t, int32_t, int32_t, int32_t>(A, F1, BN1W, BN1B,
  // F2, BN2W, BN2B, F3, BN3W, BN3B, C_layerwise, nullptr, nullptr, nullptr, N,
  // H, W, Cin, Ct, HF, WF, Cout, Hout, Wout, HPADL, HPADR, WPADL, WPADR, HSTR,
  // WSTR, D1, D2, D3, SIX_1, SIX_2, shr1, shr2, shr3, shr4, shr5, shr6, shr7,
  // shr8, shr9, shl1, shl2, shl3, shl4, shl5, shl6, shl7, shl8, shl9);
  // cleartext_MBConv_seedot<int16_t, int8_t, int8_t, int8_t, int8_t, int8_t,
  // int8_t, int8_t, int8_t, int8_t, int16_t, int16_t, int16_t, int16_t,
  // int32_t, int32_t, int32_t>(A, F1, BN1W, BN1B, F2, BN2W, BN2B, F3, BN3W,
  // BN3B, C_seedot, nullptr, nullptr, nullptr, N, H, W, Cin, Ct, HF, WF, Cout,
  // Hout, Wout, HPADL, HPADR, WPADL, WPADR, HSTR, WSTR, D1, D2, D3, SIX_1,
  // SIX_2, shr1, shr2, shr3, shr4, shr5, shr6, shr7, shr8, shr9, shl1, shl2,
  // shl3, shl4, shl5, shl6, shl7, shl8, shl9);

  finalize();

  delete[] A;
  delete[] F1;
  delete[] BN1W;
  delete[] BN1B;
  delete[] F2;
  delete[] BN2W;
  delete[] BN2B;
  delete[] F3;
  delete[] BN3W;
  delete[] BN3B;

  delete[] C;
  // delete[] C_layerwise;
  // delete[] C_seedot;
}
//

//

void test_NormaliseL2() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int32_t N = 1;
  int32_t H = 30;
  int32_t W = 40;
  int32_t C = 32;
  int32_t scaleA = -12;
  int32_t shrA = 8;
  int32_t bwA = 16;
  int64_t *A = new int64_t[N * H * W * C];
  // int64_t* calcB = new int64_t[N*H*W*C];
  // int64_t* calcCorrectB = new int64_t[N*H*W*C];
  // ifstream infile("norminfile");
  for (int i = 0; i < N * H * W * C; i++) {
    // cin>>A[i];
    // infile>>calcB[i];
    // infile>>calcB[i];
    // infile>> calcCorrectB[i];
    A[i] = party == 1 ? (int16_t)generator() : 0;
  }
  int64_t *B = new int64_t[N * H * W * C];

  initialize();

  NormaliseL2(N, H, W, C, scaleA, shrA, 16, A, B);

  finalize();

  delete[] A;
  delete[] B;
}

void test_exp() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int32_t I = 1;
  int32_t J = 16;
  int32_t scale_in = 1LL << 26;
  int32_t scale_out = 1LL << 26;
  int32_t bwA = sizeof(int16_t) * 8;
  int64_t *A = new int64_t[I * J];
  for (int i = 0; i < I * J; i++) {
    A[i] = (int16_t)(party == 1 ? (generator() | (1LL << (bwA - 1))) : 0);
  }
  int64_t *B = new int64_t[I * J];

  initialize();

  Exp(I, J, scale_in, scale_out, 32, A, B);

  finalize();

  delete[] A;
  delete[] B;
}
//

void test_div() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int32_t I = 1;
  int32_t J = 16;
  int32_t scale_in = 1LL << 26;
  int32_t scale_out = 1LL << 26;
  int32_t bwA = sizeof(int16_t) * 8;
  int64_t *A = new int64_t[I * J];
  int64_t *B = new int64_t[I * J];
  for (int i = 0; i < I * J; i++) {
    A[i] = (int32_t)generator();
    B[i] =
        (int32_t)(party == 1 ? ((generator() & (scale_in - 1)) | scale_in) : 0);
  }
  int64_t *C = new int64_t[I * J];

  initialize();

  Div(I, J, scale_in, scale_in, scale_out, 32, A, B, C);

  finalize();

  delete[] A;
  delete[] B;
  delete[] C;
}
//

void test_sigmoid() {
  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int32_t I = 1131;
  int32_t J = 16;
  int32_t scale_in = 1LL << 11;
  int32_t scale_out = 1LL << 14;
  int64_t *A = new int64_t[I * J];
  for (int i = 0; i < I * J; i++) {
    A[i] = (int16_t)generator();
  }
  int64_t *B = new int64_t[I * J];

  initialize();

  Sigmoid(I, J, scale_in, scale_out, 16, 16, A, B);

  finalize();

  delete[] A;
  delete[] B;
}
//

void test_TanH() {

  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);
  int32_t I = 1131;
  int32_t J = 16;
  int32_t scale_in = 1LL << 11;
  int32_t scale_out = 1LL << 14;

  int64_t *A = new int64_t[I * J];
  for (int i = 0; i < I * J; i++) {
    A[i] = (int16_t)generator();
  }

  int64_t *B = new int64_t[I * J];

  initialize();

  TanH(I, J, scale_in, scale_out, 16, 16, A, B);

  finalize();

  delete[] A;
  delete[] B;
}
//

void test_sqrt() {

  std::random_device rand_div;
  std::mt19937 generator(rand_div());
  // std::mt19937 generator(0);

  int32_t I = 1131;
  int32_t J = 16;

  int32_t scale_in = 12;
  int32_t scale_out = 14;

  uint64_t *A = new uint64_t[I * J];

  for (int i = 0; i < I * J; i++) {
    A[i] = party == 1 ? (int16_t)generator() : 0;
    A[i] = A[i] < 0 ? -1 * A[i] : 0;
    assert(A[i] >= 0);
  }

  uint64_t *B1 = new uint64_t[I * J];
  uint64_t *B2 = new uint64_t[I * J];

  initialize();

  Sqrt(I, J, scale_in, scale_out, 16, 16, true, A, B1);

  finalize();

  initialize();

  Sqrt(I, J, scale_in, scale_out, 16, 16, false, A, B2);

  finalize();

  delete[] A;
  delete[] B1;
  delete[] B2;
}
//

void test_AddOrSubCir4D() {
  random_device rand_div;
  mt19937 generator(rand_div());
  // mt19937 generator(0);

  int N, H, W, C, shrA, shrB, shrC, demote;
  bool add = true;

  N = 4;
  H = 10; // Height of Array
  W = 2;  // Width of Array
  C = 5;

  shrA = 4; // Scale down 1
  shrB = 2; // Scale down 2
  shrC = 8; // Scale down final
  demote = 2;

  int64_t *A = new int64_t[N * C * H * W];
  int64_t *B = new int64_t[C];
  int64_t *X1 = new int64_t[N * C * H * W];
  int64_t *X2 = new int64_t[N * C * H * W];

  for (int i = 0; i < N * C * H * W; i++) {
    A[i] = (int16_t)generator();
  }
  for (int i = 0; i < C; i++) {
    B[i] = (int16_t)(party == 2 ? generator() : 0);
  }

  initialize();
  AddOrSubCir4D(N, H, W, C, shrA, shrB, shrC, add, demote, 16, 16, 32, 16, A, B,
                X1);
  finalize();
  add = false;

  // initialize();
  // AddOrSubCir4D(N, H, W, C, shrA, shrB, shrC, add, demote, 16, 16, 32, 16, A,
  // B, X2); finalize();

  delete[] A;
  delete[] B;
  delete[] X1;
  delete[] X2;
}

void test_MatAdd() {
  random_device rand_div;
  mt19937 generator(rand_div());

  // mt19937 generator(0);

  int I, J, shrA, shrB, shrC, demote;

  I = 10;
  J = 5;
  shrA = 4;
  shrB = 8;
  shrC = 16;
  demote = 2;

  int64_t *A = new int64_t[I * J];
  int64_t *B = new int64_t[I * J];
  int64_t *C1 = new int64_t[I * J];
  int64_t *C2 = new int64_t[I * J];
  int64_t *C3 = new int64_t[I * J];

  for (int i = 0; i < I * J; i++) {
    A[i] = (int16_t)generator();
    B[i] = (int16_t)(party == 1 ? generator() : 0);
  }

  initialize();

  MatAdd(I, J, shrA, shrB, shrC, demote, 16, 16, 32, 16, A, B, C1);
  finalize();

  initialize();
  MatAddBroadCastA(I, J, shrA, shrB, shrC, demote, 16, 16, 32, 16, int64_t(10),
                   B, C2);
  finalize();

  initialize();
  MatAddBroadCastB(I, J, shrA, shrB, shrC, demote, 16, 16, 32, 16, A,
                   int64_t(11), C3);
  finalize();

  int64_t N, H, W, X;
  N = 2;
  H = 4;
  W = 3;
  X = 4;

  int64_t *A4_s = new int64_t[N * H * W * X];
  int64_t *B4_s = new int64_t[N * H * W * X];
  int64_t *C4_s = new int64_t[N * H * W * X];

  for (int i = 0; i < N * H * W * X; i++) {
    A4_s[i] = (int8_t)generator();
    B4_s[i] = (int8_t)(party == 1 ? generator() : 0);
  }

  initialize();

  MatAdd4(N, H, W, X, shrA, shrB, shrC, demote, 8, 8, 16, 8, A4_s, B4_s, C4_s);

  finalize();

  delete[] A;
  delete[] B;
  delete[] C1;
  delete[] C2;
  delete[] C3;
  delete[] A4_s;
  delete[] B4_s;
  delete[] C4_s;
}

void test_MatSub() {
  random_device rand_div;
  mt19937 generator(rand_div());
  // mt19937 generator(0);

  int I, J, shrA, shrB, shrC, demote;

  I = 10;
  J = 5;
  shrA = 4;
  shrB = 8;
  shrC = 16;
  demote = 2;

  int64_t *A = new int64_t[I * J];
  int64_t *B = new int64_t[I * J];
  int64_t *C1 = new int64_t[I * J];
  int64_t *C2 = new int64_t[I * J];
  int64_t *C3 = new int64_t[I * J];

  for (int i = 0; i < I * J; i++) {
    A[i] = (int16_t)generator();
    B[i] = (int16_t)(party == 1 ? generator() : 0);
  }

  initialize();
  MatSub(I, J, shrA, shrB, shrC, demote, 16, 16, 32, 16, A, B, C1);
  finalize();

  initialize();
  MatSubBroadCastA(I, J, shrA, shrB, shrC, demote, 16, 16, 32, 16, 10, B, C2);
  finalize();

  initialize();
  MatSubBroadCastB(I, J, shrA, shrB, shrC, demote, 16, 16, 32, 16, A, 11, C3);
  finalize();

  delete[] A;
  delete[] B;
  delete[] C1;
  delete[] C2;
  delete[] C3;
}

void test_MulCir() {
  random_device rand_div;
  mt19937 generator(rand_div());
  // mt19937 generator(0);

  int I, J, shrA, shrB, demote;

  I = 10;
  J = 5;

  shrA = 4;
  shrB = 8;
  demote = 2;

  int64_t *A = new int64_t[I * J];
  int64_t *B = new int64_t[I * J];
  int64_t *C = new int64_t[I * J];

  for (int i = 0; i < I * J; i++) {
    A[i] = (int16_t)generator();
  }
  for (int i = 0; i < I * J; i++) {
    B[i] = (int16_t)(party == 1 ? generator() : 0);
  }

  initialize();
  MulCir(I, J, shrA, shrB, demote, 16, 16, 32, 16, A, B, C);
  finalize();

  delete[] A;
  delete[] B;
  delete[] C;
}

void test_ScalarMul() {
  random_device rand_div;
  mt19937 generator(rand_div());
  // mt19937 generator(0);

  int I, J, shrA, shrB, demote;

  I = 10;
  J = 5;

  shrA = 4;
  shrB = 8;
  demote = 2;

  int64_t A = 777;
  int64_t *B = new int64_t[I * J];
  int64_t *C = new int64_t[I * J];

  for (int i = 0; i < I * J; i++) {
    B[i] = (int16_t)(party == 1 ? generator() : 0);
  }

  initialize();
  ScalarMul(I, J, shrA, shrB, demote, 16, 16, 32, 16, A, B, C);
  finalize();

  delete[] B;
  delete[] C;
}

int main(int argc, char **argv) {
  if (argc >= 2) {
    party = atoi(argv[1]);
  }
  if (argc >= 3) {
    address = argv[2];
  }
  if (argc >= 4) {
    port = atoi(argv[3]);
  }
  if (argc >= 5) {
    num_threads = atoi(argv[4]);
  }
  assert(party == 1 || party == 2);

  cout << "Party: " << party << endl;

  // test_convolution();
  // test_BNorm();
  // test_MBConv();
  // test_NormaliseL2();
  // test_AddOrSubCir4D();

  // test_matmul();
  // test_sigmoid();
  // test_TanH();
  // test_exp();
  // test_div();
  // test_MatAdd();
  // test_MatSub();
  // test_MulCir();
  // test_ScalarMul();
  // test_sqrt();
}
