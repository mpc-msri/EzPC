/*
Authors: Mayank Rathee, Nishant Kumar.
Copyright:
Copyright (c) 2018 Microsoft Research
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

#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "main.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef INC_NN

sgx_instream cin = sgx_instream();

using namespace std;

extern int partyNum;
vector<uint64_t*> toFreeMemoryLaterArr;
int NUM_OF_PARTIES;	

AESObject* aes_common;
AESObject* aes_indep;

AESObject* m_g_aes_indep_p0 = new AESObject("KeyA");
AESObject* m_g_aes_common_p0 = new AESObject("KeyAB");

AESObject* m_g_aes_indep_p1 = new AESObject("KeyB");
AESObject* m_g_aes_common_p1 = new AESObject("KeyAB");

AESObject* m_g_aes_indep_p2 = new AESObject("KeyC");
AESObject* m_g_aes_common_p2 = new AESObject("KeyCD");

AESObject* aes_a_1 = new AESObject("KeyD");
AESObject* aes_a_2 = new AESObject("KeyD");
AESObject* aes_b_1 = new AESObject("KeyD");
AESObject* aes_b_2 = new AESObject("KeyD");
AESObject* aes_c_1 = new AESObject("KeyD");

AESObject* aes_share_conv_bit_shares_p0_p2 = new AESObject("KeyD");
AESObject* aes_share_conv_bit_shares_p1_p2 = new AESObject("KeyD");
AESObject* aes_share_conv_shares_mod_odd_p0_p2 = new AESObject("KeyD");
AESObject* aes_share_conv_shares_mod_odd_p1_p2 = new AESObject("KeyD");
AESObject* aes_comp_msb_shares_lsb_p0_p2 = new AESObject("KeyD");
AESObject* aes_comp_msb_shares_lsb_p1_p2 = new AESObject("KeyD");
AESObject* aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject("KeyD");
AESObject* aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject("KeyD");
AESObject* aes_conv_opti_a_1 = new AESObject("KeyD");
AESObject* aes_conv_opti_a_2 = new AESObject("KeyD");
AESObject* aes_conv_opti_b_1 = new AESObject("KeyD");
AESObject* aes_conv_opti_b_2 = new AESObject("KeyD");
AESObject* aes_conv_opti_c_1 = new AESObject("KeyD");

//Arr to keep thread specific AES objects.
AESObject* threaded_aes_indep[NO_CORES];
AESObject* threaded_aes_common[NO_CORES];
AESObject* threaded_aes_a_1[NO_CORES];
AESObject* threaded_aes_a_2[NO_CORES];
AESObject* threaded_aes_b_1[NO_CORES];
AESObject* threaded_aes_b_2[NO_CORES];
AESObject* threaded_aes_c_1[NO_CORES];
AESObject* threaded_aes_share_conv_bit_shares_p0_p2[NO_CORES];
AESObject* threaded_aes_share_conv_bit_shares_p1_p2[NO_CORES];
AESObject* threaded_aes_share_conv_shares_mod_odd_p0_p2[NO_CORES];
AESObject* threaded_aes_share_conv_shares_mod_odd_p1_p2[NO_CORES];
AESObject* threaded_aes_comp_msb_shares_lsb_p0_p2[NO_CORES];
AESObject* threaded_aes_comp_msb_shares_lsb_p1_p2[NO_CORES];
AESObject* threaded_aes_comp_msb_shares_bit_vec_p0_p2[NO_CORES];
AESObject* threaded_aes_comp_msb_shares_bit_vec_p1_p2[NO_CORES];
AESObject* threaded_aes_conv_opti_a_1[NO_CORES];
AESObject* threaded_aes_conv_opti_a_2[NO_CORES];
AESObject* threaded_aes_conv_opti_b_1[NO_CORES];
AESObject* threaded_aes_conv_opti_b_2[NO_CORES];
AESObject* threaded_aes_conv_opti_c_1[NO_CORES];

//For thread 0
AESObject* a_m_g_aes_indep_p0 = new AESObject("KeyA");
AESObject* a_m_g_aes_common_p0 = new AESObject("KeyAB");

AESObject* a_m_g_aes_indep_p1 = new AESObject("KeyB");
AESObject* a_m_g_aes_common_p1 = new AESObject("KeyAB");

AESObject* a_m_g_aes_indep_p2 = new AESObject("KeyC");
AESObject* a_m_g_aes_common_p2 = new AESObject("KeyCD");

AESObject* a_aes_a_1 = new AESObject("KeyD");
AESObject* a_aes_a_2 = new AESObject("KeyD");
AESObject* a_aes_b_1 = new AESObject("KeyD");
AESObject* a_aes_b_2 = new AESObject("KeyD");
AESObject* a_aes_c_1 = new AESObject("KeyD");
AESObject* a_aes_share_conv_bit_shares_p0_p2 = new AESObject("KeyD");
AESObject* a_aes_share_conv_bit_shares_p1_p2 = new AESObject("KeyD");
AESObject* a_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject("KeyD");
AESObject* a_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject("KeyD");
AESObject* a_aes_comp_msb_shares_lsb_p0_p2 = new AESObject("KeyD");
AESObject* a_aes_comp_msb_shares_lsb_p1_p2 = new AESObject("KeyD");
AESObject* a_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject("KeyD");
AESObject* a_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject("KeyD");
AESObject* a_aes_conv_opti_a_1 = new AESObject("KeyD");
AESObject* a_aes_conv_opti_a_2 = new AESObject("KeyD");
AESObject* a_aes_conv_opti_b_1 = new AESObject("KeyD");
AESObject* a_aes_conv_opti_b_2 = new AESObject("KeyD");
AESObject* a_aes_conv_opti_c_1 = new AESObject("KeyD");

//For thread 1
AESObject* b_m_g_aes_indep_p0 = new AESObject("KeyA");
AESObject* b_m_g_aes_common_p0 = new AESObject("KeyAB");

AESObject* b_m_g_aes_indep_p1 = new AESObject("KeyB");
AESObject* b_m_g_aes_common_p1 = new AESObject("KeyAB");

AESObject* b_m_g_aes_indep_p2 = new AESObject("KeyC");
AESObject* b_m_g_aes_common_p2 = new AESObject("KeyCD");

AESObject* b_aes_a_1 = new AESObject("KeyD");
AESObject* b_aes_a_2 = new AESObject("KeyD");
AESObject* b_aes_b_1 = new AESObject("KeyD");
AESObject* b_aes_b_2 = new AESObject("KeyD");
AESObject* b_aes_c_1 = new AESObject("KeyD");
AESObject* b_aes_share_conv_bit_shares_p0_p2 = new AESObject("KeyD");
AESObject* b_aes_share_conv_bit_shares_p1_p2 = new AESObject("KeyD");
AESObject* b_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject("KeyD");
AESObject* b_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject("KeyD");
AESObject* b_aes_comp_msb_shares_lsb_p0_p2 = new AESObject("KeyD");
AESObject* b_aes_comp_msb_shares_lsb_p1_p2 = new AESObject("KeyD");
AESObject* b_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject("KeyD");
AESObject* b_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject("KeyD");
AESObject* b_aes_conv_opti_a_1 = new AESObject("KeyD");
AESObject* b_aes_conv_opti_a_2 = new AESObject("KeyD");
AESObject* b_aes_conv_opti_b_1 = new AESObject("KeyD");
AESObject* b_aes_conv_opti_b_2 = new AESObject("KeyD");
AESObject* b_aes_conv_opti_c_1 = new AESObject("KeyD");

//For thread 2
AESObject* c_m_g_aes_indep_p0 = new AESObject("KeyA");
AESObject* c_m_g_aes_common_p0 = new AESObject("KeyAB");

AESObject* c_m_g_aes_indep_p1 = new AESObject("KeyB");
AESObject* c_m_g_aes_common_p1 = new AESObject("KeyAB");

AESObject* c_m_g_aes_indep_p2 = new AESObject("KeyC");
AESObject* c_m_g_aes_common_p2 = new AESObject("KeyCD");

AESObject* c_aes_a_1 = new AESObject("KeyD");
AESObject* c_aes_a_2 = new AESObject("KeyD");
AESObject* c_aes_b_1 = new AESObject("KeyD");
AESObject* c_aes_b_2 = new AESObject("KeyD");
AESObject* c_aes_c_1 = new AESObject("KeyD");
AESObject* c_aes_share_conv_bit_shares_p0_p2 = new AESObject("KeyD");
AESObject* c_aes_share_conv_bit_shares_p1_p2 = new AESObject("KeyD");
AESObject* c_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject("KeyD");
AESObject* c_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject("KeyD");
AESObject* c_aes_comp_msb_shares_lsb_p0_p2 = new AESObject("KeyD");
AESObject* c_aes_comp_msb_shares_lsb_p1_p2 = new AESObject("KeyD");
AESObject* c_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject("KeyD");
AESObject* c_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject("KeyD");
AESObject* c_aes_conv_opti_a_1 = new AESObject("KeyD");
AESObject* c_aes_conv_opti_a_2 = new AESObject("KeyD");
AESObject* c_aes_conv_opti_b_1 = new AESObject("KeyD");
AESObject* c_aes_conv_opti_b_2 = new AESObject("KeyD");
AESObject* c_aes_conv_opti_c_1 = new AESObject("KeyD");

//For thread 3
AESObject* d_m_g_aes_indep_p0 = new AESObject("KeyA");
AESObject* d_m_g_aes_common_p0 = new AESObject("KeyAB");

AESObject* d_m_g_aes_indep_p1 = new AESObject("KeyB");
AESObject* d_m_g_aes_common_p1 = new AESObject("KeyAB");

AESObject* d_m_g_aes_indep_p2 = new AESObject("KeyC");
AESObject* d_m_g_aes_common_p2 = new AESObject("KeyCD");

AESObject* d_aes_a_1 = new AESObject("KeyD");
AESObject* d_aes_a_2 = new AESObject("KeyD");
AESObject* d_aes_b_1 = new AESObject("KeyD");
AESObject* d_aes_b_2 = new AESObject("KeyD");
AESObject* d_aes_c_1 = new AESObject("KeyD");
AESObject* d_aes_share_conv_bit_shares_p0_p2 = new AESObject("KeyD");
AESObject* d_aes_share_conv_bit_shares_p1_p2 = new AESObject("KeyD");
AESObject* d_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject("KeyD");
AESObject* d_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject("KeyD");
AESObject* d_aes_comp_msb_shares_lsb_p0_p2 = new AESObject("KeyD");
AESObject* d_aes_comp_msb_shares_lsb_p1_p2 = new AESObject("KeyD");
AESObject* d_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject("KeyD");
AESObject* d_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject("KeyD");
AESObject* d_aes_conv_opti_a_1 = new AESObject("KeyD");
AESObject* d_aes_conv_opti_a_2 = new AESObject("KeyD");
AESObject* d_aes_conv_opti_b_1 = new AESObject("KeyD");
AESObject* d_aes_conv_opti_b_2 = new AESObject("KeyD");
AESObject* d_aes_conv_opti_c_1 = new AESObject("KeyD");

ParallelAESObject* aes_parallel = new ParallelAESObject("");

int run_sequence = 0;

uint32_t public_lrshift(uint32_t x, uint32_t y){
return (x >> y);
}

int32_t public_lrshift(int32_t x, uint32_t y){
return ((int32_t)(((uint32_t)x) >> y));
}

uint64_t public_lrshift(uint64_t x, uint64_t y){
return (x >> y);
}

int64_t public_lrshift(int64_t x, uint64_t y){
return ((int64_t)(((uint64_t)x) >> y));
}

template<typename T>
vector<T> make_vector(size_t size) {
return std::vector<T>(size);
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
auto inner = make_vector<T>(sizes...);
return vector<decltype(inner)>(first, inner);
}

//template<typename T>
//ostream& operator<< (ostream &os, const vector<T> &v)
//{
//for(auto it = v.begin (); it != v.end (); ++it) {
//os << *it << endl;
//}
//return os;
//}


#include "ezpc.h"

//extern int partyNum;
//int NUM_OF_PARTIES;

//AESObject* aes_common;
//AESObject* aes_indep;
//AESObject* aes_a_1;
//AESObject* aes_a_2;
//AESObject* aes_b_1;
//AESObject* aes_b_2;
//AESObject* aes_c_1;
//AESObject* aes_share_conv_bit_shares_p0_p2;
//AESObject* aes_share_conv_bit_shares_p1_p2;
//AESObject* aes_share_conv_shares_mod_odd_p0_p2;
//AESObject* aes_share_conv_shares_mod_odd_p1_p2;
//AESObject* aes_comp_msb_shares_lsb_p0_p2;
//AESObject* aes_comp_msb_shares_lsb_p1_p2;
//AESObject* aes_comp_msb_shares_bit_vec_p0_p2;
//AESObject* aes_comp_msb_shares_bit_vec_p1_p2;
//AESObject* aes_conv_opti_a_1;
//AESObject* aes_conv_opti_a_2;
//AESObject* aes_conv_opti_b_1;
//AESObject* aes_conv_opti_b_2;
//AESObject* aes_conv_opti_c_1;
//ParallelAESObject* aes_parallel;

//output_queue out_q;










void MatAddBroadCast2(int32_t s1, int32_t s2, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = A[i1][i2]+B[i2];
}
}
}

void MatAdd2(int32_t s1, int32_t s2, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = A[i1][i2]+B[i1][i2];
}
}
}

void MatAddBroadCast4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = A[i1][i2][i3][i4]+B[i4];
}
}
}
}
}

void MatAdd4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& A, auto& B, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = A[i1][i2][i3][i4]+B[i1][i2][i3][i4];
}
}
}
}
}

void CreateTensor1(int32_t s1, int64_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
arr[i1] = val;
}
}

void CreateTensor2(int32_t s1, int32_t s2, int64_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
arr[i1][i2] = val;
}
}
}

void CreateTensor4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int64_t val, auto& arr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
arr[i1][i2][i3][i4] = val;
}
}
}
}
}

void CopyTensor1(int32_t s1, auto& targetArr, auto& fromArr, auto& ignore){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
targetArr[i1] = fromArr[i1];
}
}

void CopyTensor2(int32_t s1, int32_t s2, auto& targetArr, auto& fromArr, auto& ignore){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
targetArr[i1][i2] = fromArr[i1][i2];
}
}
}

void CopyTensor4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& targetArr, auto& fromArr, auto& ignore){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
targetArr[i1][i2][i3][i4] = fromArr[i1][i2][i3][i4];
}
}
}
}
}

void CreateIdentity11(int32_t s1, auto& fromArr, auto& newArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
newArr[i1] = fromArr[i1];
}
}

void CreateIdentity22(int32_t s1, int32_t s2, auto& fromArr, auto& newArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
newArr[i1][i2] = fromArr[i1][i2];
}
}
}

void CreateIdentity44(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& fromArr, auto& newArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
newArr[i1][i2][i3][i4] = fromArr[i1][i2][i3][i4];
}
}
}
}
}

void Concat2T444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto& inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto& inp2, int32_t axis, auto& outp){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
for (uint32_t i3 =  (int32_t)0; i3 < s3; i3++){
for (uint32_t i4 =  (int32_t)0; i4 < s4; i4++){
if ((axis ==  (int32_t)0)) {
if ((i1 < inp1s1)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
}
} else {
if ((axis ==  (int32_t)1)) {
if ((i2 < inp1s2)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
}
} else {
if ((axis ==  (int32_t)2)) {
if ((i3 < inp1s3)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
}
} else {
if ((i4 < inp1s4)) {
outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
} else {
outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
}
}
}
}
}
}
}
}
}

void RandomUniform2(int32_t s1, int32_t s2, int64_t dataType, auto& outArr){
for (uint32_t i1 =  (int32_t)0; i1 < s1; i1++){
for (uint32_t i2 =  (int32_t)0; i2 < s2; i2++){
outArr[i1][i2] = funcSSCons( (int64_t)100);
}
}
}

void Conv2DReshapeFilter(int32_t FH, int32_t FW, int32_t CI, int32_t CO, auto& inputArr, auto& outputArr){
for (uint32_t co =  (int32_t)0; co < CO; co++){
for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
for (uint32_t fw =  (int32_t)0; fw < FW; fw++){
for (uint32_t ci =  (int32_t)0; ci < CI; ci++){

int32_t linIdx = ((((fh * FW) * CI) + (fw * CI)) + ci);
outputArr[co][linIdx] = inputArr[fh][fw][ci][co];
}
}
}
}
}

void Conv2DReshapeMatMulOP(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, auto& inputArr, auto& outputArr){
for (uint32_t co =  (int32_t)0; co < CO; co++){
for (uint32_t n =  (int32_t)0; n < N; n++){
for (uint32_t h =  (int32_t)0; h < finalH; h++){
for (uint32_t w =  (int32_t)0; w < finalW; w++){
outputArr[n][h][w][co] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)];
}
}
}
}
}

void Conv2DReshapeInput(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zeroPadH, int32_t zeroPadW, int32_t strideH, int32_t strideW, int32_t RRows, int32_t RCols, auto& inputArr, auto& outputArr){

int32_t linIdxFilterMult =  (int32_t)0;
for (uint32_t n =  (int32_t)0; n < N; n++){

int32_t leftTopCornerH = ( (int32_t)0 - zeroPadH);

int32_t extremeRightBottomCornerH = ((H -  (int32_t)1) + zeroPadH);
while ((((leftTopCornerH + FH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

int32_t leftTopCornerW = ( (int32_t)0 - zeroPadW);

int32_t extremeRightBottomCornerW = ((W -  (int32_t)1) + zeroPadW);
while ((((leftTopCornerW + FW) -  (int32_t)1) <= extremeRightBottomCornerW)) {
for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
for (uint32_t fw =  (int32_t)0; fw < FW; fw++){

int32_t curPosH = (leftTopCornerH + fh);

int32_t curPosW = (leftTopCornerW + fw);

uint64_t val = 0;
for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
val = 0;
} else {
val = inputArr[n][curPosH][curPosW][ci];
}
outputArr[((((fh * FW) * CI) + (fw * CI)) + ci)][linIdxFilterMult] = val;
}
}
}
linIdxFilterMult = (linIdxFilterMult +  (int32_t)1);
leftTopCornerW = (leftTopCornerW + strideW);
}

leftTopCornerH = (leftTopCornerH + strideH);
}

}
}

void Conv2DCSFOld(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadH, int32_t zPadW, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, auto& outArr, int64_t consSF){

int32_t reshapedFilterRows = CO;

int32_t reshapedFilterCols = ((FH * FW) * CI);

int32_t reshapedIPRows = ((FH * FW) * CI);

int32_t newH = ((((H + ( (int32_t)2 * zPadH)) - FH) / strideH) +  (int32_t)1);

int32_t newW = ((((W + ( (int32_t)2 * zPadW)) - FW) / strideW) +  (int32_t)1);

int32_t reshapedIPCols = ((N * newH) * newW);

auto filterReshaped = make_vector<uint64_t>(reshapedFilterRows, reshapedFilterCols);

auto inputReshaped = make_vector<uint64_t>(reshapedIPRows, reshapedIPCols);

auto matmulOP = make_vector<uint64_t>(reshapedFilterRows, reshapedIPCols);
Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, filterReshaped);
Conv2DReshapeInput(N, H, W, CI, FH, FW, zPadH, zPadW, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
MatMulCSF2D(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP, consSF);
Conv2DReshapeMatMulOP(N, newH, newW, CO, matmulOP, outArr);
}

void Conv2DCSF(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadH, int32_t zPadW, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, auto& outArr, int64_t consSF){
#ifdef CONV_OPTI
	if((FH>=5) || (FW>=5)){
		funcConv2DCSF(N, H, W, CI, FH, FW, CO, zPadH, zPadH, zPadW, zPadW, strideH, strideW, inputArr, filterArr, consSF, outArr);
	}
	else{
		funcConv2DCSFSplit(N, H, W, CI, FH, FW, CO, zPadH, zPadH, zPadW, zPadW, strideH, strideW, inputArr, filterArr, consSF, outArr);
		//Conv2DCSFOld(N, H, W, CI, FH, FW, CO, zPadH, zPadW, strideH, strideW, inputArr, filterArr, outArr, consSF);
	}
#else	
		Conv2DCSFOld(N, H, W, CI, FH, FW, CO, zPadH, zPadW, strideH, strideW, inputArr, filterArr, outArr, consSF);
#endif
}
void Transpose2(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
for (uint32_t i =  (int32_t)0; i < s1; i++){
for (uint32_t j =  (int32_t)0; j < s2; j++){
outArr[i][j] = inArr[j][i];
}
}
}

void Pad442(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inps1, int32_t inps2, int32_t inps3, int32_t inps4, auto& inpArr, int32_t pads1, int32_t pads2, auto& paddings, auto& outArr){

int32_t lbounds1 = paddings[ (int32_t)0][ (int32_t)0];

int32_t rbounds1excl = (s1 - paddings[ (int32_t)0][ (int32_t)1]);

int32_t lbounds2 = paddings[ (int32_t)1][ (int32_t)0];

int32_t rbounds2excl = (s2 - paddings[ (int32_t)1][ (int32_t)1]);

int32_t lbounds3 = paddings[ (int32_t)2][ (int32_t)0];

int32_t rbounds3excl = (s3 - paddings[ (int32_t)2][ (int32_t)1]);

int32_t lbounds4 = paddings[ (int32_t)3][ (int32_t)0];

int32_t rbounds4excl = (s4 - paddings[ (int32_t)3][ (int32_t)1]);
for (uint32_t i =  (int32_t)0; i < s1; i++){
for (uint32_t j =  (int32_t)0; j < s2; j++){
for (uint32_t k =  (int32_t)0; k < s3; k++){
for (uint32_t l =  (int32_t)0; l < s4; l++){
if (((((((((i >= lbounds1) && (i < rbounds1excl)) && (j >= lbounds2)) && (j < rbounds2excl)) && (k >= lbounds3)) && (k < rbounds3excl)) && (l >= lbounds4)) && (l < rbounds4excl))) {
outArr[i][j][k][l] = inpArr[(i - paddings[ (int32_t)0][ (int32_t)0])][(j - paddings[ (int32_t)1][ (int32_t)0])][(k - paddings[ (int32_t)2][ (int32_t)0])][(l - paddings[ (int32_t)3][ (int32_t)0])];
} else {
//outArr[i][j][k][l] = funcSSCons( (int64_t)0);
outArr[i][j][k][l] = 0;
}
}
}
}
}
}

void Squeeze24(int32_t s1, int32_t s2, int32_t dim1, int32_t dim2, int32_t ins1, int32_t ins2, int32_t ins3, int32_t ins4, auto& inArr, auto& outArr){
for (uint32_t i =  (int32_t)0; i < ins1; i++){
for (uint32_t j =  (int32_t)0; j < ins2; j++){
for (uint32_t k =  (int32_t)0; k < ins3; k++){
for (uint32_t l =  (int32_t)0; l < ins4; l++){

int32_t linIdx = ((((((i * ins2) * ins3) * ins4) + ((j * ins3) * ins4)) + (k * ins4)) + l);

int32_t outIdx1 = (linIdx / s2);

int32_t outIdx2 = (linIdx % s2);
outArr[outIdx1][outIdx2] = inArr[i][j][k][l];
}
}
}
}
}


void main_securenn(int pnum)
{
show_aramis_mode();
touch_time();
parseInputs(pnum);
string whichNetwork = "Resnet";
if(run_sequence != 0){

	if(pnum == 0){
		m_g_aes_indep_p0->ResetKey("KeyA");
		m_g_aes_common_p0->ResetKey("KeyAB");
		a_m_g_aes_indep_p0->ResetKey("KeyA");
		a_m_g_aes_common_p0->ResetKey("KeyAB");
		b_m_g_aes_indep_p0->ResetKey("KeyA");
		b_m_g_aes_common_p0->ResetKey("KeyAB");
		c_m_g_aes_indep_p0->ResetKey("KeyA");
		c_m_g_aes_common_p0->ResetKey("KeyAB");
		d_m_g_aes_indep_p0->ResetKey("KeyA");
		d_m_g_aes_common_p0->ResetKey("KeyAB");
		
		aes_indep = m_g_aes_indep_p0;
		aes_common = m_g_aes_common_p0;
		
		threaded_aes_indep[0] = a_m_g_aes_indep_p0;
		threaded_aes_indep[1] = b_m_g_aes_indep_p0;
		threaded_aes_indep[2] = c_m_g_aes_indep_p0;
		threaded_aes_indep[3] = d_m_g_aes_indep_p0;
		
		threaded_aes_common[0] = a_m_g_aes_common_p0;		
		threaded_aes_common[1] = b_m_g_aes_common_p0;		
		threaded_aes_common[2] = c_m_g_aes_common_p0;		
		threaded_aes_common[3] = d_m_g_aes_common_p0;		
	
	}
	else if(pnum == 1){
		m_g_aes_indep_p1->ResetKey("KeyB");
		m_g_aes_common_p1->ResetKey("KeyAB");
		a_m_g_aes_indep_p1->ResetKey("KeyB");
		a_m_g_aes_common_p1->ResetKey("KeyAB");
		b_m_g_aes_indep_p1->ResetKey("KeyB");
		b_m_g_aes_common_p1->ResetKey("KeyAB");
		c_m_g_aes_indep_p1->ResetKey("KeyB");
		c_m_g_aes_common_p1->ResetKey("KeyAB");
		d_m_g_aes_indep_p1->ResetKey("KeyB");
		d_m_g_aes_common_p1->ResetKey("KeyAB");
		
		aes_indep = m_g_aes_indep_p1;
		aes_common = m_g_aes_common_p1;
		
		threaded_aes_indep[0] = a_m_g_aes_indep_p1;
		threaded_aes_indep[1] = b_m_g_aes_indep_p1;
		threaded_aes_indep[2] = c_m_g_aes_indep_p1;
		threaded_aes_indep[3] = d_m_g_aes_indep_p1;
		
		threaded_aes_common[0] = a_m_g_aes_common_p1;		
		threaded_aes_common[1] = b_m_g_aes_common_p1;		
		threaded_aes_common[2] = c_m_g_aes_common_p1;		
		threaded_aes_common[3] = d_m_g_aes_common_p1;		
	}
	else if(pnum == 2){
		m_g_aes_indep_p2->ResetKey("KeyB");
		m_g_aes_common_p2->ResetKey("KeyAB");
		a_m_g_aes_indep_p2->ResetKey("KeyB");
		a_m_g_aes_common_p2->ResetKey("KeyAB");
		b_m_g_aes_indep_p2->ResetKey("KeyB");
		b_m_g_aes_common_p2->ResetKey("KeyAB");
		c_m_g_aes_indep_p2->ResetKey("KeyB");
		c_m_g_aes_common_p2->ResetKey("KeyAB");
		d_m_g_aes_indep_p2->ResetKey("KeyB");
		d_m_g_aes_common_p2->ResetKey("KeyAB");
		
		aes_indep = m_g_aes_indep_p2;
		aes_common = m_g_aes_common_p2;
		
		threaded_aes_indep[0] = a_m_g_aes_indep_p2;
		threaded_aes_indep[1] = b_m_g_aes_indep_p2;
		threaded_aes_indep[2] = c_m_g_aes_indep_p2;
		threaded_aes_indep[3] = d_m_g_aes_indep_p2;
		
		threaded_aes_common[0] = a_m_g_aes_common_p2;		
		threaded_aes_common[1] = b_m_g_aes_common_p2;		
		threaded_aes_common[2] = c_m_g_aes_common_p2;		
		threaded_aes_common[3] = d_m_g_aes_common_p2;		
	}
	
	aes_a_1->ResetKey("KeyD");
	a_aes_a_1->ResetKey("KeyD");
	b_aes_a_1->ResetKey("KeyD");
	c_aes_a_1->ResetKey("KeyD");
	d_aes_a_1->ResetKey("KeyD");
	threaded_aes_a_1[0] = a_aes_a_1;
	threaded_aes_a_1[1] = b_aes_a_1;
	threaded_aes_a_1[2] = c_aes_a_1;
	threaded_aes_a_1[3] = d_aes_a_1;

	aes_a_2->ResetKey("KeyD");
	a_aes_a_2->ResetKey("KeyD");
	b_aes_a_2->ResetKey("KeyD");
	c_aes_a_2->ResetKey("KeyD");
	d_aes_a_2->ResetKey("KeyD");
	threaded_aes_a_2[0] = a_aes_a_2;
	threaded_aes_a_2[1] = b_aes_a_2;
	threaded_aes_a_2[2] = c_aes_a_2;
	threaded_aes_a_2[3] = d_aes_a_2;

	aes_b_1->ResetKey("KeyD");
	a_aes_b_1->ResetKey("KeyD");
	b_aes_b_1->ResetKey("KeyD");
	c_aes_b_1->ResetKey("KeyD");
	d_aes_b_1->ResetKey("KeyD");
	threaded_aes_b_1[0] = a_aes_b_1;
	threaded_aes_b_1[1] = b_aes_b_1;
	threaded_aes_b_1[2] = c_aes_b_1;
	threaded_aes_b_1[3] = d_aes_b_1;

	aes_b_2->ResetKey("KeyD");
	a_aes_b_2->ResetKey("KeyD");
	b_aes_b_2->ResetKey("KeyD");
	c_aes_b_2->ResetKey("KeyD");
	d_aes_b_2->ResetKey("KeyD");
	threaded_aes_b_2[0] = a_aes_b_2;
	threaded_aes_b_2[1] = b_aes_b_2;
	threaded_aes_b_2[2] = c_aes_b_2;
	threaded_aes_b_2[3] = d_aes_b_2;

	aes_c_1->ResetKey("KeyD");
	a_aes_c_1->ResetKey("KeyD");
	b_aes_c_1->ResetKey("KeyD");
	c_aes_c_1->ResetKey("KeyD");
	d_aes_c_1->ResetKey("KeyD");
	threaded_aes_c_1[0] = a_aes_c_1;
	threaded_aes_c_1[1] = b_aes_c_1;
	threaded_aes_c_1[2] = c_aes_c_1;
	threaded_aes_c_1[3] = d_aes_c_1;

	aes_share_conv_bit_shares_p0_p2->ResetKey("KeyD");
	a_aes_share_conv_bit_shares_p0_p2->ResetKey("KeyD");
	b_aes_share_conv_bit_shares_p0_p2->ResetKey("KeyD");
	c_aes_share_conv_bit_shares_p0_p2->ResetKey("KeyD");
	d_aes_share_conv_bit_shares_p0_p2->ResetKey("KeyD");
	threaded_aes_share_conv_bit_shares_p0_p2[0] = a_aes_share_conv_bit_shares_p0_p2;
	threaded_aes_share_conv_bit_shares_p0_p2[1] = b_aes_share_conv_bit_shares_p0_p2;
	threaded_aes_share_conv_bit_shares_p0_p2[2] = c_aes_share_conv_bit_shares_p0_p2;
	threaded_aes_share_conv_bit_shares_p0_p2[3] = d_aes_share_conv_bit_shares_p0_p2;
	
	aes_share_conv_bit_shares_p1_p2->ResetKey("KeyD");
	a_aes_share_conv_bit_shares_p1_p2->ResetKey("KeyD");
	b_aes_share_conv_bit_shares_p1_p2->ResetKey("KeyD");
	c_aes_share_conv_bit_shares_p1_p2->ResetKey("KeyD");
	d_aes_share_conv_bit_shares_p1_p2->ResetKey("KeyD");
	threaded_aes_share_conv_bit_shares_p1_p2[0] = a_aes_share_conv_bit_shares_p1_p2;
	threaded_aes_share_conv_bit_shares_p1_p2[1] = b_aes_share_conv_bit_shares_p1_p2;
	threaded_aes_share_conv_bit_shares_p1_p2[2] = c_aes_share_conv_bit_shares_p1_p2;
	threaded_aes_share_conv_bit_shares_p1_p2[3] = d_aes_share_conv_bit_shares_p1_p2;
	

	aes_share_conv_shares_mod_odd_p0_p2->ResetKey("KeyD");
	a_aes_share_conv_shares_mod_odd_p0_p2->ResetKey("KeyD");
	b_aes_share_conv_shares_mod_odd_p0_p2->ResetKey("KeyD");
	c_aes_share_conv_shares_mod_odd_p0_p2->ResetKey("KeyD");
	d_aes_share_conv_shares_mod_odd_p0_p2->ResetKey("KeyD");
	threaded_aes_share_conv_shares_mod_odd_p0_p2[0] = a_aes_share_conv_shares_mod_odd_p0_p2;
	threaded_aes_share_conv_shares_mod_odd_p0_p2[1] = b_aes_share_conv_shares_mod_odd_p0_p2;
	threaded_aes_share_conv_shares_mod_odd_p0_p2[2] = c_aes_share_conv_shares_mod_odd_p0_p2;
	threaded_aes_share_conv_shares_mod_odd_p0_p2[3] = d_aes_share_conv_shares_mod_odd_p0_p2;
	
	
	aes_share_conv_shares_mod_odd_p1_p2->ResetKey("KeyD");
	a_aes_share_conv_shares_mod_odd_p1_p2->ResetKey("KeyD");
	b_aes_share_conv_shares_mod_odd_p1_p2->ResetKey("KeyD");
	c_aes_share_conv_shares_mod_odd_p1_p2->ResetKey("KeyD");
	d_aes_share_conv_shares_mod_odd_p1_p2->ResetKey("KeyD");
	threaded_aes_share_conv_shares_mod_odd_p1_p2[0] = a_aes_share_conv_shares_mod_odd_p1_p2;
	threaded_aes_share_conv_shares_mod_odd_p1_p2[1] = b_aes_share_conv_shares_mod_odd_p1_p2;
	threaded_aes_share_conv_shares_mod_odd_p1_p2[2] = c_aes_share_conv_shares_mod_odd_p1_p2;
	threaded_aes_share_conv_shares_mod_odd_p1_p2[3] = d_aes_share_conv_shares_mod_odd_p1_p2;
	
	aes_comp_msb_shares_lsb_p0_p2->ResetKey("KeyD");
	a_aes_comp_msb_shares_lsb_p0_p2->ResetKey("KeyD");
	b_aes_comp_msb_shares_lsb_p0_p2->ResetKey("KeyD");
	c_aes_comp_msb_shares_lsb_p0_p2->ResetKey("KeyD");
	d_aes_comp_msb_shares_lsb_p0_p2->ResetKey("KeyD");
	threaded_aes_comp_msb_shares_lsb_p0_p2[0] = a_aes_comp_msb_shares_lsb_p0_p2;
	threaded_aes_comp_msb_shares_lsb_p0_p2[1] = b_aes_comp_msb_shares_lsb_p0_p2;
	threaded_aes_comp_msb_shares_lsb_p0_p2[2] = c_aes_comp_msb_shares_lsb_p0_p2;
	threaded_aes_comp_msb_shares_lsb_p0_p2[3] = d_aes_comp_msb_shares_lsb_p0_p2;
	
	
	aes_comp_msb_shares_lsb_p1_p2->ResetKey("KeyD");
	a_aes_comp_msb_shares_lsb_p1_p2->ResetKey("KeyD");
	b_aes_comp_msb_shares_lsb_p1_p2->ResetKey("KeyD");
	c_aes_comp_msb_shares_lsb_p1_p2->ResetKey("KeyD");
	d_aes_comp_msb_shares_lsb_p1_p2->ResetKey("KeyD");
	threaded_aes_comp_msb_shares_lsb_p1_p2[0] = a_aes_comp_msb_shares_lsb_p1_p2;
	threaded_aes_comp_msb_shares_lsb_p1_p2[1] = b_aes_comp_msb_shares_lsb_p1_p2;
	threaded_aes_comp_msb_shares_lsb_p1_p2[2] = c_aes_comp_msb_shares_lsb_p1_p2;
	threaded_aes_comp_msb_shares_lsb_p1_p2[3] = d_aes_comp_msb_shares_lsb_p1_p2;
	
	
	aes_comp_msb_shares_bit_vec_p0_p2->ResetKey("KeyD");
	a_aes_comp_msb_shares_bit_vec_p0_p2->ResetKey("KeyD");
	b_aes_comp_msb_shares_bit_vec_p0_p2->ResetKey("KeyD");
	c_aes_comp_msb_shares_bit_vec_p0_p2->ResetKey("KeyD");
	d_aes_comp_msb_shares_bit_vec_p0_p2->ResetKey("KeyD");
	threaded_aes_comp_msb_shares_bit_vec_p0_p2[0] = a_aes_comp_msb_shares_bit_vec_p0_p2;
	threaded_aes_comp_msb_shares_bit_vec_p0_p2[1] = b_aes_comp_msb_shares_bit_vec_p0_p2;
	threaded_aes_comp_msb_shares_bit_vec_p0_p2[2] = c_aes_comp_msb_shares_bit_vec_p0_p2;
	threaded_aes_comp_msb_shares_bit_vec_p0_p2[3] = d_aes_comp_msb_shares_bit_vec_p0_p2;
	
	
	aes_comp_msb_shares_bit_vec_p1_p2->ResetKey("KeyD");
	a_aes_comp_msb_shares_bit_vec_p1_p2->ResetKey("KeyD");
	b_aes_comp_msb_shares_bit_vec_p1_p2->ResetKey("KeyD");
	c_aes_comp_msb_shares_bit_vec_p1_p2->ResetKey("KeyD");
	d_aes_comp_msb_shares_bit_vec_p1_p2->ResetKey("KeyD");
	threaded_aes_comp_msb_shares_bit_vec_p1_p2[0] = a_aes_comp_msb_shares_bit_vec_p1_p2;
	threaded_aes_comp_msb_shares_bit_vec_p1_p2[1] = b_aes_comp_msb_shares_bit_vec_p1_p2;
	threaded_aes_comp_msb_shares_bit_vec_p1_p2[2] = c_aes_comp_msb_shares_bit_vec_p1_p2;
	threaded_aes_comp_msb_shares_bit_vec_p1_p2[3] = d_aes_comp_msb_shares_bit_vec_p1_p2;
	
	
	aes_conv_opti_a_1->ResetKey("KeyD");
	a_aes_conv_opti_a_1->ResetKey("KeyD");
	b_aes_conv_opti_a_1->ResetKey("KeyD");
	c_aes_conv_opti_a_1->ResetKey("KeyD");
	d_aes_conv_opti_a_1->ResetKey("KeyD");
	threaded_aes_conv_opti_a_1[0] = a_aes_conv_opti_a_1;
	threaded_aes_conv_opti_a_1[1] = b_aes_conv_opti_a_1;
	threaded_aes_conv_opti_a_1[2] = c_aes_conv_opti_a_1;
	threaded_aes_conv_opti_a_1[3] = d_aes_conv_opti_a_1;
	
	
	aes_conv_opti_a_2->ResetKey("KeyD");
	a_aes_conv_opti_a_2->ResetKey("KeyD");
	b_aes_conv_opti_a_2->ResetKey("KeyD");
	c_aes_conv_opti_a_2->ResetKey("KeyD");
	d_aes_conv_opti_a_2->ResetKey("KeyD");
	threaded_aes_conv_opti_a_2[0] = a_aes_conv_opti_a_2;
	threaded_aes_conv_opti_a_2[1] = b_aes_conv_opti_a_2;
	threaded_aes_conv_opti_a_2[2] = c_aes_conv_opti_a_2;
	threaded_aes_conv_opti_a_2[3] = d_aes_conv_opti_a_2;
	
	
	aes_conv_opti_b_1->ResetKey("KeyD");
	a_aes_conv_opti_b_1->ResetKey("KeyD");
	b_aes_conv_opti_b_1->ResetKey("KeyD");
	c_aes_conv_opti_b_1->ResetKey("KeyD");
	d_aes_conv_opti_b_1->ResetKey("KeyD");
	threaded_aes_conv_opti_b_1[0] = a_aes_conv_opti_b_1;
	threaded_aes_conv_opti_b_1[1] = b_aes_conv_opti_b_1;
	threaded_aes_conv_opti_b_1[2] = c_aes_conv_opti_b_1;
	threaded_aes_conv_opti_b_1[3] = d_aes_conv_opti_b_1;
	
	
	aes_conv_opti_b_2->ResetKey("KeyD");
	a_aes_conv_opti_b_2->ResetKey("KeyD");
	b_aes_conv_opti_b_2->ResetKey("KeyD");
	c_aes_conv_opti_b_2->ResetKey("KeyD");
	d_aes_conv_opti_b_2->ResetKey("KeyD");
	threaded_aes_conv_opti_b_2[0] = a_aes_conv_opti_b_2;
	threaded_aes_conv_opti_b_2[1] = b_aes_conv_opti_b_2;
	threaded_aes_conv_opti_b_2[2] = c_aes_conv_opti_b_2;
	threaded_aes_conv_opti_b_2[3] = d_aes_conv_opti_b_2;
	
	
	aes_conv_opti_c_1->ResetKey("KeyD");
	a_aes_conv_opti_c_1->ResetKey("KeyD");
	b_aes_conv_opti_c_1->ResetKey("KeyD");
	c_aes_conv_opti_c_1->ResetKey("KeyD");
	d_aes_conv_opti_c_1->ResetKey("KeyD");
	threaded_aes_conv_opti_c_1[0] = a_aes_conv_opti_c_1;
	threaded_aes_conv_opti_c_1[1] = b_aes_conv_opti_c_1;
	threaded_aes_conv_opti_c_1[2] = c_aes_conv_opti_c_1;
	threaded_aes_conv_opti_c_1[3] = d_aes_conv_opti_c_1;

}
if(run_sequence == 0){
		if (!STANDALONE)
		{
			initializeCommunication("", partyNum);
			synchronize(1000000);	
		}
		print_string("AramisNet has been successfully created.");
		run_sequence++;
		return;
}

initializeMPC();

//if (!STANDALONE)
//{
//initializeMPC();
//initializeCommunication(argv[3], partyNum);
//synchronize(2000000); 
//}

if (PARALLEL)
aes_parallel->precompute();

e_role role;
if(partyNum == 0){
	role = CLIENT;
}
else if(partyNum == 1){
	role = SERVER;
}
else{
	role = ALL;
}

auto tmp252 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp253 = make_vector<uint64_t>( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3);

auto tmp254 = make_vector<int64_t>( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64);

auto tmp255 = make_vector<uint64_t>( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64);

auto tmp256 = make_vector<int32_t>( (int32_t)2);

auto tmp257 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);

auto tmp258 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);

auto tmp259 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp260 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp261 = make_vector<int64_t>( (int32_t)64);

auto tmp262 = make_vector<uint64_t>( (int32_t)64);

auto tmp263 = make_vector<int64_t>( (int32_t)64);

auto tmp264 = make_vector<uint64_t>( (int32_t)64);

auto tmp265 = make_vector<int64_t>( (int32_t)64);

auto tmp266 = make_vector<uint64_t>( (int32_t)64);

auto tmp267 = make_vector<int64_t>( (int32_t)64);

auto tmp268 = make_vector<uint64_t>( (int32_t)64);

auto tmp269 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp270 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp271 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp272 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp273 = make_vector<int32_t>( (int32_t)2);

auto tmp274 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp275 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64);

auto tmp276 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64);

auto tmp277 = make_vector<int32_t>( (int32_t)2);

auto tmp278 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp279 = make_vector<int64_t>( (int32_t)64);

auto tmp280 = make_vector<uint64_t>( (int32_t)64);

auto tmp281 = make_vector<int64_t>( (int32_t)64);

auto tmp282 = make_vector<uint64_t>( (int32_t)64);

auto tmp283 = make_vector<int64_t>( (int32_t)64);

auto tmp284 = make_vector<uint64_t>( (int32_t)64);

auto tmp285 = make_vector<int64_t>( (int32_t)64);

auto tmp286 = make_vector<uint64_t>( (int32_t)64);

auto tmp287 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp288 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp289 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);

auto tmp290 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);

auto tmp291 = make_vector<int32_t>( (int32_t)2);

auto tmp292 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp293 = make_vector<int64_t>( (int32_t)64);

auto tmp294 = make_vector<uint64_t>( (int32_t)64);

auto tmp295 = make_vector<int64_t>( (int32_t)64);

auto tmp296 = make_vector<uint64_t>( (int32_t)64);

auto tmp297 = make_vector<int64_t>( (int32_t)64);

auto tmp298 = make_vector<uint64_t>( (int32_t)64);

auto tmp299 = make_vector<int64_t>( (int32_t)64);

auto tmp300 = make_vector<uint64_t>( (int32_t)64);

auto tmp301 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp302 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp303 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp304 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp305 = make_vector<int32_t>( (int32_t)2);

auto tmp306 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp307 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp308 = make_vector<int64_t>( (int32_t)256);

auto tmp309 = make_vector<uint64_t>( (int32_t)256);

auto tmp310 = make_vector<int64_t>( (int32_t)256);

auto tmp311 = make_vector<uint64_t>( (int32_t)256);

auto tmp312 = make_vector<int64_t>( (int32_t)256);

auto tmp313 = make_vector<uint64_t>( (int32_t)256);

auto tmp314 = make_vector<int64_t>( (int32_t)256);

auto tmp315 = make_vector<uint64_t>( (int32_t)256);

auto tmp316 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp317 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp318 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);

auto tmp319 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);

auto tmp320 = make_vector<int32_t>( (int32_t)2);

auto tmp321 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp322 = make_vector<int64_t>( (int32_t)64);

auto tmp323 = make_vector<uint64_t>( (int32_t)64);

auto tmp324 = make_vector<int64_t>( (int32_t)64);

auto tmp325 = make_vector<uint64_t>( (int32_t)64);

auto tmp326 = make_vector<int64_t>( (int32_t)64);

auto tmp327 = make_vector<uint64_t>( (int32_t)64);

auto tmp328 = make_vector<int64_t>( (int32_t)64);

auto tmp329 = make_vector<uint64_t>( (int32_t)64);

auto tmp330 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp331 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp332 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);

auto tmp333 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);

auto tmp334 = make_vector<int32_t>( (int32_t)2);

auto tmp335 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp336 = make_vector<int64_t>( (int32_t)64);

auto tmp337 = make_vector<uint64_t>( (int32_t)64);

auto tmp338 = make_vector<int64_t>( (int32_t)64);

auto tmp339 = make_vector<uint64_t>( (int32_t)64);

auto tmp340 = make_vector<int64_t>( (int32_t)64);

auto tmp341 = make_vector<uint64_t>( (int32_t)64);

auto tmp342 = make_vector<int64_t>( (int32_t)64);

auto tmp343 = make_vector<uint64_t>( (int32_t)64);

auto tmp344 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp345 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp346 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp347 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp348 = make_vector<int32_t>( (int32_t)2);

auto tmp349 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp350 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp351 = make_vector<int64_t>( (int32_t)256);

auto tmp352 = make_vector<uint64_t>( (int32_t)256);

auto tmp353 = make_vector<int64_t>( (int32_t)256);

auto tmp354 = make_vector<uint64_t>( (int32_t)256);

auto tmp355 = make_vector<int64_t>( (int32_t)256);

auto tmp356 = make_vector<uint64_t>( (int32_t)256);

auto tmp357 = make_vector<int64_t>( (int32_t)256);

auto tmp358 = make_vector<uint64_t>( (int32_t)256);

auto tmp359 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp360 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp361 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);

auto tmp362 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);

auto tmp363 = make_vector<int32_t>( (int32_t)2);

auto tmp364 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp365 = make_vector<int64_t>( (int32_t)64);

auto tmp366 = make_vector<uint64_t>( (int32_t)64);

auto tmp367 = make_vector<int64_t>( (int32_t)64);

auto tmp368 = make_vector<uint64_t>( (int32_t)64);

auto tmp369 = make_vector<int64_t>( (int32_t)64);

auto tmp370 = make_vector<uint64_t>( (int32_t)64);

auto tmp371 = make_vector<int64_t>( (int32_t)64);

auto tmp372 = make_vector<uint64_t>( (int32_t)64);

auto tmp373 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp374 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp375 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);

auto tmp376 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);

auto tmp377 = make_vector<int32_t>( (int32_t)2);

auto tmp378 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp379 = make_vector<int64_t>( (int32_t)64);

auto tmp380 = make_vector<uint64_t>( (int32_t)64);

auto tmp381 = make_vector<int64_t>( (int32_t)64);

auto tmp382 = make_vector<uint64_t>( (int32_t)64);

auto tmp383 = make_vector<int64_t>( (int32_t)64);

auto tmp384 = make_vector<uint64_t>( (int32_t)64);

auto tmp385 = make_vector<int64_t>( (int32_t)64);

auto tmp386 = make_vector<uint64_t>( (int32_t)64);

auto tmp387 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp388 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);

auto tmp389 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp390 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);

auto tmp391 = make_vector<int32_t>( (int32_t)2);

auto tmp392 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp393 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp394 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp395 = make_vector<int64_t>( (int32_t)256);

auto tmp396 = make_vector<uint64_t>( (int32_t)256);

auto tmp397 = make_vector<int64_t>( (int32_t)256);

auto tmp398 = make_vector<uint64_t>( (int32_t)256);

auto tmp399 = make_vector<int64_t>( (int32_t)256);

auto tmp400 = make_vector<uint64_t>( (int32_t)256);

auto tmp401 = make_vector<int64_t>( (int32_t)256);

auto tmp402 = make_vector<uint64_t>( (int32_t)256);

auto tmp403 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp404 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp405 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp406 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);

auto tmp407 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512);

auto tmp408 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512);

auto tmp409 = make_vector<int32_t>( (int32_t)2);

auto tmp410 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp411 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128);

auto tmp412 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128);

auto tmp413 = make_vector<int32_t>( (int32_t)2);

auto tmp414 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp415 = make_vector<int64_t>( (int32_t)128);

auto tmp416 = make_vector<uint64_t>( (int32_t)128);

auto tmp417 = make_vector<int64_t>( (int32_t)128);

auto tmp418 = make_vector<uint64_t>( (int32_t)128);

auto tmp419 = make_vector<int64_t>( (int32_t)128);

auto tmp420 = make_vector<uint64_t>( (int32_t)128);

auto tmp421 = make_vector<int64_t>( (int32_t)128);

auto tmp422 = make_vector<uint64_t>( (int32_t)128);

auto tmp423 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp424 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);

auto tmp425 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp426 = make_vector<uint64_t>( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128);

auto tmp427 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp428 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp429 = make_vector<int32_t>( (int32_t)2);

auto tmp430 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp431 = make_vector<int64_t>( (int32_t)128);

auto tmp432 = make_vector<uint64_t>( (int32_t)128);

auto tmp433 = make_vector<int64_t>( (int32_t)128);

auto tmp434 = make_vector<uint64_t>( (int32_t)128);

auto tmp435 = make_vector<int64_t>( (int32_t)128);

auto tmp436 = make_vector<uint64_t>( (int32_t)128);

auto tmp437 = make_vector<int64_t>( (int32_t)128);

auto tmp438 = make_vector<uint64_t>( (int32_t)128);

auto tmp439 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp440 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp441 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp442 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp443 = make_vector<int32_t>( (int32_t)2);

auto tmp444 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp445 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp446 = make_vector<int64_t>( (int32_t)512);

auto tmp447 = make_vector<uint64_t>( (int32_t)512);

auto tmp448 = make_vector<int64_t>( (int32_t)512);

auto tmp449 = make_vector<uint64_t>( (int32_t)512);

auto tmp450 = make_vector<int64_t>( (int32_t)512);

auto tmp451 = make_vector<uint64_t>( (int32_t)512);

auto tmp452 = make_vector<int64_t>( (int32_t)512);

auto tmp453 = make_vector<uint64_t>( (int32_t)512);

auto tmp454 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp455 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp456 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);

auto tmp457 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);

auto tmp458 = make_vector<int32_t>( (int32_t)2);

auto tmp459 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp460 = make_vector<int64_t>( (int32_t)128);

auto tmp461 = make_vector<uint64_t>( (int32_t)128);

auto tmp462 = make_vector<int64_t>( (int32_t)128);

auto tmp463 = make_vector<uint64_t>( (int32_t)128);

auto tmp464 = make_vector<int64_t>( (int32_t)128);

auto tmp465 = make_vector<uint64_t>( (int32_t)128);

auto tmp466 = make_vector<int64_t>( (int32_t)128);

auto tmp467 = make_vector<uint64_t>( (int32_t)128);

auto tmp468 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp469 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp470 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp471 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp472 = make_vector<int32_t>( (int32_t)2);

auto tmp473 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp474 = make_vector<int64_t>( (int32_t)128);

auto tmp475 = make_vector<uint64_t>( (int32_t)128);

auto tmp476 = make_vector<int64_t>( (int32_t)128);

auto tmp477 = make_vector<uint64_t>( (int32_t)128);

auto tmp478 = make_vector<int64_t>( (int32_t)128);

auto tmp479 = make_vector<uint64_t>( (int32_t)128);

auto tmp480 = make_vector<int64_t>( (int32_t)128);

auto tmp481 = make_vector<uint64_t>( (int32_t)128);

auto tmp482 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp483 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp484 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp485 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp486 = make_vector<int32_t>( (int32_t)2);

auto tmp487 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp488 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp489 = make_vector<int64_t>( (int32_t)512);

auto tmp490 = make_vector<uint64_t>( (int32_t)512);

auto tmp491 = make_vector<int64_t>( (int32_t)512);

auto tmp492 = make_vector<uint64_t>( (int32_t)512);

auto tmp493 = make_vector<int64_t>( (int32_t)512);

auto tmp494 = make_vector<uint64_t>( (int32_t)512);

auto tmp495 = make_vector<int64_t>( (int32_t)512);

auto tmp496 = make_vector<uint64_t>( (int32_t)512);

auto tmp497 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp498 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp499 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);

auto tmp500 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);

auto tmp501 = make_vector<int32_t>( (int32_t)2);

auto tmp502 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp503 = make_vector<int64_t>( (int32_t)128);

auto tmp504 = make_vector<uint64_t>( (int32_t)128);

auto tmp505 = make_vector<int64_t>( (int32_t)128);

auto tmp506 = make_vector<uint64_t>( (int32_t)128);

auto tmp507 = make_vector<int64_t>( (int32_t)128);

auto tmp508 = make_vector<uint64_t>( (int32_t)128);

auto tmp509 = make_vector<int64_t>( (int32_t)128);

auto tmp510 = make_vector<uint64_t>( (int32_t)128);

auto tmp511 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp512 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp513 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp514 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp515 = make_vector<int32_t>( (int32_t)2);

auto tmp516 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp517 = make_vector<int64_t>( (int32_t)128);

auto tmp518 = make_vector<uint64_t>( (int32_t)128);

auto tmp519 = make_vector<int64_t>( (int32_t)128);

auto tmp520 = make_vector<uint64_t>( (int32_t)128);

auto tmp521 = make_vector<int64_t>( (int32_t)128);

auto tmp522 = make_vector<uint64_t>( (int32_t)128);

auto tmp523 = make_vector<int64_t>( (int32_t)128);

auto tmp524 = make_vector<uint64_t>( (int32_t)128);

auto tmp525 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp526 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp527 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp528 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp529 = make_vector<int32_t>( (int32_t)2);

auto tmp530 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp531 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp532 = make_vector<int64_t>( (int32_t)512);

auto tmp533 = make_vector<uint64_t>( (int32_t)512);

auto tmp534 = make_vector<int64_t>( (int32_t)512);

auto tmp535 = make_vector<uint64_t>( (int32_t)512);

auto tmp536 = make_vector<int64_t>( (int32_t)512);

auto tmp537 = make_vector<uint64_t>( (int32_t)512);

auto tmp538 = make_vector<int64_t>( (int32_t)512);

auto tmp539 = make_vector<uint64_t>( (int32_t)512);

auto tmp540 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp541 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp542 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);

auto tmp543 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);

auto tmp544 = make_vector<int32_t>( (int32_t)2);

auto tmp545 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp546 = make_vector<int64_t>( (int32_t)128);

auto tmp547 = make_vector<uint64_t>( (int32_t)128);

auto tmp548 = make_vector<int64_t>( (int32_t)128);

auto tmp549 = make_vector<uint64_t>( (int32_t)128);

auto tmp550 = make_vector<int64_t>( (int32_t)128);

auto tmp551 = make_vector<uint64_t>( (int32_t)128);

auto tmp552 = make_vector<int64_t>( (int32_t)128);

auto tmp553 = make_vector<uint64_t>( (int32_t)128);

auto tmp554 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp555 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp556 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp557 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);

auto tmp558 = make_vector<int32_t>( (int32_t)2);

auto tmp559 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp560 = make_vector<int64_t>( (int32_t)128);

auto tmp561 = make_vector<uint64_t>( (int32_t)128);

auto tmp562 = make_vector<int64_t>( (int32_t)128);

auto tmp563 = make_vector<uint64_t>( (int32_t)128);

auto tmp564 = make_vector<int64_t>( (int32_t)128);

auto tmp565 = make_vector<uint64_t>( (int32_t)128);

auto tmp566 = make_vector<int64_t>( (int32_t)128);

auto tmp567 = make_vector<uint64_t>( (int32_t)128);

auto tmp568 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp569 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);

auto tmp570 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp571 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);

auto tmp572 = make_vector<int32_t>( (int32_t)2);

auto tmp573 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp574 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp575 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp576 = make_vector<int64_t>( (int32_t)512);

auto tmp577 = make_vector<uint64_t>( (int32_t)512);

auto tmp578 = make_vector<int64_t>( (int32_t)512);

auto tmp579 = make_vector<uint64_t>( (int32_t)512);

auto tmp580 = make_vector<int64_t>( (int32_t)512);

auto tmp581 = make_vector<uint64_t>( (int32_t)512);

auto tmp582 = make_vector<int64_t>( (int32_t)512);

auto tmp583 = make_vector<uint64_t>( (int32_t)512);

auto tmp584 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp585 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp586 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp587 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);

auto tmp588 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024);

auto tmp589 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024);

auto tmp590 = make_vector<int32_t>( (int32_t)2);

auto tmp591 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp592 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256);

auto tmp593 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256);

auto tmp594 = make_vector<int32_t>( (int32_t)2);

auto tmp595 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp596 = make_vector<int64_t>( (int32_t)256);

auto tmp597 = make_vector<uint64_t>( (int32_t)256);

auto tmp598 = make_vector<int64_t>( (int32_t)256);

auto tmp599 = make_vector<uint64_t>( (int32_t)256);

auto tmp600 = make_vector<int64_t>( (int32_t)256);

auto tmp601 = make_vector<uint64_t>( (int32_t)256);

auto tmp602 = make_vector<int64_t>( (int32_t)256);

auto tmp603 = make_vector<uint64_t>( (int32_t)256);

auto tmp604 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp605 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);

auto tmp606 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp607 = make_vector<uint64_t>( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256);

auto tmp608 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp609 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp610 = make_vector<int32_t>( (int32_t)2);

auto tmp611 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp612 = make_vector<int64_t>( (int32_t)256);

auto tmp613 = make_vector<uint64_t>( (int32_t)256);

auto tmp614 = make_vector<int64_t>( (int32_t)256);

auto tmp615 = make_vector<uint64_t>( (int32_t)256);

auto tmp616 = make_vector<int64_t>( (int32_t)256);

auto tmp617 = make_vector<uint64_t>( (int32_t)256);

auto tmp618 = make_vector<int64_t>( (int32_t)256);

auto tmp619 = make_vector<uint64_t>( (int32_t)256);

auto tmp620 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp621 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp622 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp623 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp624 = make_vector<int32_t>( (int32_t)2);

auto tmp625 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp626 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp627 = make_vector<int64_t>( (int32_t)1);

auto tmp628 = make_vector<int64_t>( (int32_t)1024);

auto tmp629 = make_vector<uint64_t>( (int32_t)1024);

auto tmp630 = make_vector<int64_t>( (int32_t)1);

auto tmp631 = make_vector<int64_t>( (int32_t)1024);

auto tmp632 = make_vector<uint64_t>( (int32_t)1024);

auto tmp633 = make_vector<int64_t>( (int32_t)1);

auto tmp634 = make_vector<int64_t>( (int32_t)1024);

auto tmp635 = make_vector<uint64_t>( (int32_t)1024);

auto tmp636 = make_vector<int64_t>( (int32_t)1);

auto tmp637 = make_vector<int64_t>( (int32_t)1024);

auto tmp638 = make_vector<uint64_t>( (int32_t)1024);

auto tmp639 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp640 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp641 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp642 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp643 = make_vector<int32_t>( (int32_t)2);

auto tmp644 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp645 = make_vector<int64_t>( (int32_t)256);

auto tmp646 = make_vector<uint64_t>( (int32_t)256);

auto tmp647 = make_vector<int64_t>( (int32_t)256);

auto tmp648 = make_vector<uint64_t>( (int32_t)256);

auto tmp649 = make_vector<int64_t>( (int32_t)256);

auto tmp650 = make_vector<uint64_t>( (int32_t)256);

auto tmp651 = make_vector<int64_t>( (int32_t)256);

auto tmp652 = make_vector<uint64_t>( (int32_t)256);

auto tmp653 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp654 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp655 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp656 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp657 = make_vector<int32_t>( (int32_t)2);

auto tmp658 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp659 = make_vector<int64_t>( (int32_t)256);

auto tmp660 = make_vector<uint64_t>( (int32_t)256);

auto tmp661 = make_vector<int64_t>( (int32_t)256);

auto tmp662 = make_vector<uint64_t>( (int32_t)256);

auto tmp663 = make_vector<int64_t>( (int32_t)256);

auto tmp664 = make_vector<uint64_t>( (int32_t)256);

auto tmp665 = make_vector<int64_t>( (int32_t)256);

auto tmp666 = make_vector<uint64_t>( (int32_t)256);

auto tmp667 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp668 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp669 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp670 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp671 = make_vector<int32_t>( (int32_t)2);

auto tmp672 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp673 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp674 = make_vector<int64_t>( (int32_t)1);

auto tmp675 = make_vector<int64_t>( (int32_t)1024);

auto tmp676 = make_vector<uint64_t>( (int32_t)1024);

auto tmp677 = make_vector<int64_t>( (int32_t)1);

auto tmp678 = make_vector<int64_t>( (int32_t)1024);

auto tmp679 = make_vector<uint64_t>( (int32_t)1024);

auto tmp680 = make_vector<int64_t>( (int32_t)1);

auto tmp681 = make_vector<int64_t>( (int32_t)1024);

auto tmp682 = make_vector<uint64_t>( (int32_t)1024);

auto tmp683 = make_vector<int64_t>( (int32_t)1);

auto tmp684 = make_vector<int64_t>( (int32_t)1024);

auto tmp685 = make_vector<uint64_t>( (int32_t)1024);

auto tmp686 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp687 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp688 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp689 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp690 = make_vector<int32_t>( (int32_t)2);

auto tmp691 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp692 = make_vector<int64_t>( (int32_t)256);

auto tmp693 = make_vector<uint64_t>( (int32_t)256);

auto tmp694 = make_vector<int64_t>( (int32_t)256);

auto tmp695 = make_vector<uint64_t>( (int32_t)256);

auto tmp696 = make_vector<int64_t>( (int32_t)256);

auto tmp697 = make_vector<uint64_t>( (int32_t)256);

auto tmp698 = make_vector<int64_t>( (int32_t)256);

auto tmp699 = make_vector<uint64_t>( (int32_t)256);

auto tmp700 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp701 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp702 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp703 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp704 = make_vector<int32_t>( (int32_t)2);

auto tmp705 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp706 = make_vector<int64_t>( (int32_t)256);

auto tmp707 = make_vector<uint64_t>( (int32_t)256);

auto tmp708 = make_vector<int64_t>( (int32_t)256);

auto tmp709 = make_vector<uint64_t>( (int32_t)256);

auto tmp710 = make_vector<int64_t>( (int32_t)256);

auto tmp711 = make_vector<uint64_t>( (int32_t)256);

auto tmp712 = make_vector<int64_t>( (int32_t)256);

auto tmp713 = make_vector<uint64_t>( (int32_t)256);

auto tmp714 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp715 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp716 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp717 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp718 = make_vector<int32_t>( (int32_t)2);

auto tmp719 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp720 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp721 = make_vector<int64_t>( (int32_t)1);

auto tmp722 = make_vector<int64_t>( (int32_t)1024);

auto tmp723 = make_vector<uint64_t>( (int32_t)1024);

auto tmp724 = make_vector<int64_t>( (int32_t)1);

auto tmp725 = make_vector<int64_t>( (int32_t)1024);

auto tmp726 = make_vector<uint64_t>( (int32_t)1024);

auto tmp727 = make_vector<int64_t>( (int32_t)1);

auto tmp728 = make_vector<int64_t>( (int32_t)1024);

auto tmp729 = make_vector<uint64_t>( (int32_t)1024);

auto tmp730 = make_vector<int64_t>( (int32_t)1);

auto tmp731 = make_vector<int64_t>( (int32_t)1024);

auto tmp732 = make_vector<uint64_t>( (int32_t)1024);

auto tmp733 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp734 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp735 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp736 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp737 = make_vector<int32_t>( (int32_t)2);

auto tmp738 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp739 = make_vector<int64_t>( (int32_t)256);

auto tmp740 = make_vector<uint64_t>( (int32_t)256);

auto tmp741 = make_vector<int64_t>( (int32_t)256);

auto tmp742 = make_vector<uint64_t>( (int32_t)256);

auto tmp743 = make_vector<int64_t>( (int32_t)256);

auto tmp744 = make_vector<uint64_t>( (int32_t)256);

auto tmp745 = make_vector<int64_t>( (int32_t)256);

auto tmp746 = make_vector<uint64_t>( (int32_t)256);

auto tmp747 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp748 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp749 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp750 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp751 = make_vector<int32_t>( (int32_t)2);

auto tmp752 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp753 = make_vector<int64_t>( (int32_t)256);

auto tmp754 = make_vector<uint64_t>( (int32_t)256);

auto tmp755 = make_vector<int64_t>( (int32_t)256);

auto tmp756 = make_vector<uint64_t>( (int32_t)256);

auto tmp757 = make_vector<int64_t>( (int32_t)256);

auto tmp758 = make_vector<uint64_t>( (int32_t)256);

auto tmp759 = make_vector<int64_t>( (int32_t)256);

auto tmp760 = make_vector<uint64_t>( (int32_t)256);

auto tmp761 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp762 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp763 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp764 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp765 = make_vector<int32_t>( (int32_t)2);

auto tmp766 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp767 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp768 = make_vector<int64_t>( (int32_t)1);

auto tmp769 = make_vector<int64_t>( (int32_t)1024);

auto tmp770 = make_vector<uint64_t>( (int32_t)1024);

auto tmp771 = make_vector<int64_t>( (int32_t)1);

auto tmp772 = make_vector<int64_t>( (int32_t)1024);

auto tmp773 = make_vector<uint64_t>( (int32_t)1024);

auto tmp774 = make_vector<int64_t>( (int32_t)1);

auto tmp775 = make_vector<int64_t>( (int32_t)1024);

auto tmp776 = make_vector<uint64_t>( (int32_t)1024);

auto tmp777 = make_vector<int64_t>( (int32_t)1);

auto tmp778 = make_vector<int64_t>( (int32_t)1024);

auto tmp779 = make_vector<uint64_t>( (int32_t)1024);

auto tmp780 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp781 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp782 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp783 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp784 = make_vector<int32_t>( (int32_t)2);

auto tmp785 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp786 = make_vector<int64_t>( (int32_t)256);

auto tmp787 = make_vector<uint64_t>( (int32_t)256);

auto tmp788 = make_vector<int64_t>( (int32_t)256);

auto tmp789 = make_vector<uint64_t>( (int32_t)256);

auto tmp790 = make_vector<int64_t>( (int32_t)256);

auto tmp791 = make_vector<uint64_t>( (int32_t)256);

auto tmp792 = make_vector<int64_t>( (int32_t)256);

auto tmp793 = make_vector<uint64_t>( (int32_t)256);

auto tmp794 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp795 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp796 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp797 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp798 = make_vector<int32_t>( (int32_t)2);

auto tmp799 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp800 = make_vector<int64_t>( (int32_t)256);

auto tmp801 = make_vector<uint64_t>( (int32_t)256);

auto tmp802 = make_vector<int64_t>( (int32_t)256);

auto tmp803 = make_vector<uint64_t>( (int32_t)256);

auto tmp804 = make_vector<int64_t>( (int32_t)256);

auto tmp805 = make_vector<uint64_t>( (int32_t)256);

auto tmp806 = make_vector<int64_t>( (int32_t)256);

auto tmp807 = make_vector<uint64_t>( (int32_t)256);

auto tmp808 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp809 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp810 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp811 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp812 = make_vector<int32_t>( (int32_t)2);

auto tmp813 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp814 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp815 = make_vector<int64_t>( (int32_t)1);

auto tmp816 = make_vector<int64_t>( (int32_t)1024);

auto tmp817 = make_vector<uint64_t>( (int32_t)1024);

auto tmp818 = make_vector<int64_t>( (int32_t)1);

auto tmp819 = make_vector<int64_t>( (int32_t)1024);

auto tmp820 = make_vector<uint64_t>( (int32_t)1024);

auto tmp821 = make_vector<int64_t>( (int32_t)1);

auto tmp822 = make_vector<int64_t>( (int32_t)1024);

auto tmp823 = make_vector<uint64_t>( (int32_t)1024);

auto tmp824 = make_vector<int64_t>( (int32_t)1);

auto tmp825 = make_vector<int64_t>( (int32_t)1024);

auto tmp826 = make_vector<uint64_t>( (int32_t)1024);

auto tmp827 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp828 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp829 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp830 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);

auto tmp831 = make_vector<int32_t>( (int32_t)2);

auto tmp832 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp833 = make_vector<int64_t>( (int32_t)256);

auto tmp834 = make_vector<uint64_t>( (int32_t)256);

auto tmp835 = make_vector<int64_t>( (int32_t)256);

auto tmp836 = make_vector<uint64_t>( (int32_t)256);

auto tmp837 = make_vector<int64_t>( (int32_t)256);

auto tmp838 = make_vector<uint64_t>( (int32_t)256);

auto tmp839 = make_vector<int64_t>( (int32_t)256);

auto tmp840 = make_vector<uint64_t>( (int32_t)256);

auto tmp841 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp842 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp843 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp844 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);

auto tmp845 = make_vector<int32_t>( (int32_t)2);

auto tmp846 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp847 = make_vector<int64_t>( (int32_t)256);

auto tmp848 = make_vector<uint64_t>( (int32_t)256);

auto tmp849 = make_vector<int64_t>( (int32_t)256);

auto tmp850 = make_vector<uint64_t>( (int32_t)256);

auto tmp851 = make_vector<int64_t>( (int32_t)256);

auto tmp852 = make_vector<uint64_t>( (int32_t)256);

auto tmp853 = make_vector<int64_t>( (int32_t)256);

auto tmp854 = make_vector<uint64_t>( (int32_t)256);

auto tmp855 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp856 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);

auto tmp857 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp858 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);

auto tmp859 = make_vector<int32_t>( (int32_t)2);

auto tmp860 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp861 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp862 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp863 = make_vector<int64_t>( (int32_t)1);

auto tmp864 = make_vector<int64_t>( (int32_t)1024);

auto tmp865 = make_vector<uint64_t>( (int32_t)1024);

auto tmp866 = make_vector<int64_t>( (int32_t)1);

auto tmp867 = make_vector<int64_t>( (int32_t)1024);

auto tmp868 = make_vector<uint64_t>( (int32_t)1024);

auto tmp869 = make_vector<int64_t>( (int32_t)1);

auto tmp870 = make_vector<int64_t>( (int32_t)1024);

auto tmp871 = make_vector<uint64_t>( (int32_t)1024);

auto tmp872 = make_vector<int64_t>( (int32_t)1);

auto tmp873 = make_vector<int64_t>( (int32_t)1024);

auto tmp874 = make_vector<uint64_t>( (int32_t)1024);

auto tmp875 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp876 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp877 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp878 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);

auto tmp879 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048);

auto tmp880 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048);

auto tmp881 = make_vector<int32_t>( (int32_t)2);

auto tmp882 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp883 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512);

auto tmp884 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512);

auto tmp885 = make_vector<int32_t>( (int32_t)2);

auto tmp886 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp887 = make_vector<int64_t>( (int32_t)512);

auto tmp888 = make_vector<uint64_t>( (int32_t)512);

auto tmp889 = make_vector<int64_t>( (int32_t)512);

auto tmp890 = make_vector<uint64_t>( (int32_t)512);

auto tmp891 = make_vector<int64_t>( (int32_t)512);

auto tmp892 = make_vector<uint64_t>( (int32_t)512);

auto tmp893 = make_vector<int64_t>( (int32_t)512);

auto tmp894 = make_vector<uint64_t>( (int32_t)512);

auto tmp895 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp896 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);

auto tmp897 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);

auto tmp898 = make_vector<uint64_t>( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512);

auto tmp899 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);

auto tmp900 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);

auto tmp901 = make_vector<int32_t>( (int32_t)2);

auto tmp902 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp903 = make_vector<int64_t>( (int32_t)512);

auto tmp904 = make_vector<uint64_t>( (int32_t)512);

auto tmp905 = make_vector<int64_t>( (int32_t)512);

auto tmp906 = make_vector<uint64_t>( (int32_t)512);

auto tmp907 = make_vector<int64_t>( (int32_t)512);

auto tmp908 = make_vector<uint64_t>( (int32_t)512);

auto tmp909 = make_vector<int64_t>( (int32_t)512);

auto tmp910 = make_vector<uint64_t>( (int32_t)512);

auto tmp911 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp912 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp913 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);

auto tmp914 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);

auto tmp915 = make_vector<int32_t>( (int32_t)2);

auto tmp916 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp917 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp918 = make_vector<int64_t>( (int32_t)1);

auto tmp919 = make_vector<int64_t>( (int32_t)2048);

auto tmp920 = make_vector<uint64_t>( (int32_t)2048);

auto tmp921 = make_vector<int64_t>( (int32_t)1);

auto tmp922 = make_vector<int64_t>( (int32_t)2048);

auto tmp923 = make_vector<uint64_t>( (int32_t)2048);

auto tmp924 = make_vector<int64_t>( (int32_t)1);

auto tmp925 = make_vector<int64_t>( (int32_t)2048);

auto tmp926 = make_vector<uint64_t>( (int32_t)2048);

auto tmp927 = make_vector<int64_t>( (int32_t)1);

auto tmp928 = make_vector<int64_t>( (int32_t)2048);

auto tmp929 = make_vector<uint64_t>( (int32_t)2048);

auto tmp930 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp931 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp932 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);

auto tmp933 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);

auto tmp934 = make_vector<int32_t>( (int32_t)2);

auto tmp935 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp936 = make_vector<int64_t>( (int32_t)512);

auto tmp937 = make_vector<uint64_t>( (int32_t)512);

auto tmp938 = make_vector<int64_t>( (int32_t)512);

auto tmp939 = make_vector<uint64_t>( (int32_t)512);

auto tmp940 = make_vector<int64_t>( (int32_t)512);

auto tmp941 = make_vector<uint64_t>( (int32_t)512);

auto tmp942 = make_vector<int64_t>( (int32_t)512);

auto tmp943 = make_vector<uint64_t>( (int32_t)512);

auto tmp944 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp945 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp946 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);

auto tmp947 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);

auto tmp948 = make_vector<int32_t>( (int32_t)2);

auto tmp949 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp950 = make_vector<int64_t>( (int32_t)512);

auto tmp951 = make_vector<uint64_t>( (int32_t)512);

auto tmp952 = make_vector<int64_t>( (int32_t)512);

auto tmp953 = make_vector<uint64_t>( (int32_t)512);

auto tmp954 = make_vector<int64_t>( (int32_t)512);

auto tmp955 = make_vector<uint64_t>( (int32_t)512);

auto tmp956 = make_vector<int64_t>( (int32_t)512);

auto tmp957 = make_vector<uint64_t>( (int32_t)512);

auto tmp958 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp959 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp960 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);

auto tmp961 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);

auto tmp962 = make_vector<int32_t>( (int32_t)2);

auto tmp963 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp964 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp965 = make_vector<int64_t>( (int32_t)1);

auto tmp966 = make_vector<int64_t>( (int32_t)2048);

auto tmp967 = make_vector<uint64_t>( (int32_t)2048);

auto tmp968 = make_vector<int64_t>( (int32_t)1);

auto tmp969 = make_vector<int64_t>( (int32_t)2048);

auto tmp970 = make_vector<uint64_t>( (int32_t)2048);

auto tmp971 = make_vector<int64_t>( (int32_t)1);

auto tmp972 = make_vector<int64_t>( (int32_t)2048);

auto tmp973 = make_vector<uint64_t>( (int32_t)2048);

auto tmp974 = make_vector<int64_t>( (int32_t)1);

auto tmp975 = make_vector<int64_t>( (int32_t)2048);

auto tmp976 = make_vector<uint64_t>( (int32_t)2048);

auto tmp977 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp978 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp979 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);

auto tmp980 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);

auto tmp981 = make_vector<int32_t>( (int32_t)2);

auto tmp982 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp983 = make_vector<int64_t>( (int32_t)512);

auto tmp984 = make_vector<uint64_t>( (int32_t)512);

auto tmp985 = make_vector<int64_t>( (int32_t)512);

auto tmp986 = make_vector<uint64_t>( (int32_t)512);

auto tmp987 = make_vector<int64_t>( (int32_t)512);

auto tmp988 = make_vector<uint64_t>( (int32_t)512);

auto tmp989 = make_vector<int64_t>( (int32_t)512);

auto tmp990 = make_vector<uint64_t>( (int32_t)512);

auto tmp991 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp992 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp993 = make_vector<int64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);

auto tmp994 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);

auto tmp995 = make_vector<int32_t>( (int32_t)2);

auto tmp996 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp997 = make_vector<int64_t>( (int32_t)512);

auto tmp998 = make_vector<uint64_t>( (int32_t)512);

auto tmp999 = make_vector<int64_t>( (int32_t)512);

auto tmp1000 = make_vector<uint64_t>( (int32_t)512);

auto tmp1001 = make_vector<int64_t>( (int32_t)512);

auto tmp1002 = make_vector<uint64_t>( (int32_t)512);

auto tmp1003 = make_vector<int64_t>( (int32_t)512);

auto tmp1004 = make_vector<uint64_t>( (int32_t)512);

auto tmp1005 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp1006 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);

auto tmp1007 = make_vector<int64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);

auto tmp1008 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);

auto tmp1009 = make_vector<int32_t>( (int32_t)2);

auto tmp1010 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp1011 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp1012 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp1013 = make_vector<int64_t>( (int32_t)1);

auto tmp1014 = make_vector<int64_t>( (int32_t)2048);

auto tmp1015 = make_vector<uint64_t>( (int32_t)2048);

auto tmp1016 = make_vector<int64_t>( (int32_t)1);

auto tmp1017 = make_vector<int64_t>( (int32_t)2048);

auto tmp1018 = make_vector<uint64_t>( (int32_t)2048);

auto tmp1019 = make_vector<int64_t>( (int32_t)1);

auto tmp1020 = make_vector<int64_t>( (int32_t)2048);

auto tmp1021 = make_vector<uint64_t>( (int32_t)2048);

auto tmp1022 = make_vector<int64_t>( (int32_t)1);

auto tmp1023 = make_vector<int64_t>( (int32_t)2048);

auto tmp1024 = make_vector<uint64_t>( (int32_t)2048);

auto tmp1025 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp1026 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);

auto tmp1027 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048);

auto tmp1028 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048);

auto tmp1029 = make_vector<uint64_t>( (int32_t)1,  (int32_t)2048);

auto tmp1030 = make_vector<int64_t>( (int32_t)2048,  (int32_t)1001);

auto tmp1031 = make_vector<uint64_t>( (int32_t)2048,  (int32_t)1001);

auto tmp1032 = make_vector<int64_t>( (int32_t)1);

auto tmp1033 = make_vector<int64_t>( (int32_t)1001);

auto tmp1034 = make_vector<uint64_t>( (int32_t)1001);

auto tmp1035 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);

auto tmp1036 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);

auto tmp1037 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);

auto tmp1038 = make_vector<uint64_t>( (int32_t)1);


auto tmp0 = make_vector<uint64_t>( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3);
/* Variable to read the clear value corresponding to the input variable tmp0 at (1169,1-1169,47) */
uint64_t __tmp_in_tmp0;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)224; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)224; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)3; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp0;
}
tmp0[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp0 : 0;
}
}
}
}

auto tmp1 = make_vector<uint64_t>( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp1 at (1172,1-1172,44) */
uint64_t __tmp_in_tmp1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)7; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)7; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)3; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp1;
}
tmp1[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp1 : 0;
}
}
}
}

auto tmp2 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp2 at (1175,1-1175,35) */
uint64_t __tmp_in_tmp2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp2;
}
tmp2[i0] = (role == CLIENT) ? __tmp_in_tmp2 : 0;
}

auto tmp3 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp3 at (1178,1-1178,35) */
uint64_t __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp3;
}
tmp3[i0] = (role == CLIENT) ? __tmp_in_tmp3 : 0;
}

auto tmp4 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp4 at (1181,1-1181,35) */
uint64_t __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp4;
}
tmp4[i0] = (role == CLIENT) ? __tmp_in_tmp4 : 0;
}

auto tmp5 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp5 at (1184,1-1184,35) */
uint64_t __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp5;
}
tmp5[i0] = (role == CLIENT) ? __tmp_in_tmp5 : 0;
}

auto tmp6 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp6 at (1187,1-1187,46) */
uint64_t __tmp_in_tmp6;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp6;
}
tmp6[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp6 : 0;
}
}
}
}

auto tmp7 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp7 at (1190,1-1190,45) */
uint64_t __tmp_in_tmp7;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp7;
}
tmp7[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp7 : 0;
}
}
}
}

auto tmp8 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp8 at (1193,1-1193,35) */
uint64_t __tmp_in_tmp8;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp8;
}
tmp8[i0] = (role == CLIENT) ? __tmp_in_tmp8 : 0;
}

auto tmp9 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp9 at (1196,1-1196,35) */
uint64_t __tmp_in_tmp9;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp9;
}
tmp9[i0] = (role == CLIENT) ? __tmp_in_tmp9 : 0;
}

auto tmp10 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp10 at (1199,1-1199,36) */
uint64_t __tmp_in_tmp10;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp10;
}
tmp10[i0] = (role == CLIENT) ? __tmp_in_tmp10 : 0;
}

auto tmp11 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp11 at (1202,1-1202,36) */
uint64_t __tmp_in_tmp11;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp11;
}
tmp11[i0] = (role == CLIENT) ? __tmp_in_tmp11 : 0;
}

auto tmp12 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp12 at (1205,1-1205,46) */
uint64_t __tmp_in_tmp12;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp12;
}
tmp12[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp12 : 0;
}
}
}
}

auto tmp13 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp13 at (1208,1-1208,36) */
uint64_t __tmp_in_tmp13;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp13;
}
tmp13[i0] = (role == CLIENT) ? __tmp_in_tmp13 : 0;
}

auto tmp14 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp14 at (1211,1-1211,36) */
uint64_t __tmp_in_tmp14;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp14;
}
tmp14[i0] = (role == CLIENT) ? __tmp_in_tmp14 : 0;
}

auto tmp15 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp15 at (1214,1-1214,36) */
uint64_t __tmp_in_tmp15;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp15;
}
tmp15[i0] = (role == CLIENT) ? __tmp_in_tmp15 : 0;
}

auto tmp16 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp16 at (1217,1-1217,36) */
uint64_t __tmp_in_tmp16;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp16;
}
tmp16[i0] = (role == CLIENT) ? __tmp_in_tmp16 : 0;
}

auto tmp17 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp17 at (1220,1-1220,47) */
uint64_t __tmp_in_tmp17;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp17;
}
tmp17[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp17 : 0;
}
}
}
}

auto tmp18 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp18 at (1223,1-1223,37) */
uint64_t __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp18;
}
tmp18[i0] = (role == CLIENT) ? __tmp_in_tmp18 : 0;
}

auto tmp19 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp19 at (1226,1-1226,37) */
uint64_t __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp19;
}
tmp19[i0] = (role == CLIENT) ? __tmp_in_tmp19 : 0;
}

auto tmp20 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp20 at (1229,1-1229,37) */
uint64_t __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp20;
}
tmp20[i0] = (role == CLIENT) ? __tmp_in_tmp20 : 0;
}

auto tmp21 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp21 at (1232,1-1232,37) */
uint64_t __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp21;
}
tmp21[i0] = (role == CLIENT) ? __tmp_in_tmp21 : 0;
}

auto tmp22 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp22 at (1235,1-1235,47) */
uint64_t __tmp_in_tmp22;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp22;
}
tmp22[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp22 : 0;
}
}
}
}

auto tmp23 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp23 at (1238,1-1238,36) */
uint64_t __tmp_in_tmp23;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp23;
}
tmp23[i0] = (role == CLIENT) ? __tmp_in_tmp23 : 0;
}

auto tmp24 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp24 at (1241,1-1241,36) */
uint64_t __tmp_in_tmp24;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp24;
}
tmp24[i0] = (role == CLIENT) ? __tmp_in_tmp24 : 0;
}

auto tmp25 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp25 at (1244,1-1244,36) */
uint64_t __tmp_in_tmp25;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp25;
}
tmp25[i0] = (role == CLIENT) ? __tmp_in_tmp25 : 0;
}

auto tmp26 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp26 at (1247,1-1247,36) */
uint64_t __tmp_in_tmp26;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp26;
}
tmp26[i0] = (role == CLIENT) ? __tmp_in_tmp26 : 0;
}

auto tmp27 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp27 at (1250,1-1250,46) */
uint64_t __tmp_in_tmp27;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp27;
}
tmp27[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp27 : 0;
}
}
}
}

auto tmp28 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp28 at (1253,1-1253,36) */
uint64_t __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp28;
}
tmp28[i0] = (role == CLIENT) ? __tmp_in_tmp28 : 0;
}

auto tmp29 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp29 at (1256,1-1256,36) */
uint64_t __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp29;
}
tmp29[i0] = (role == CLIENT) ? __tmp_in_tmp29 : 0;
}

auto tmp30 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp30 at (1259,1-1259,36) */
uint64_t __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp30;
}
tmp30[i0] = (role == CLIENT) ? __tmp_in_tmp30 : 0;
}

auto tmp31 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp31 at (1262,1-1262,36) */
uint64_t __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp31;
}
tmp31[i0] = (role == CLIENT) ? __tmp_in_tmp31 : 0;
}

auto tmp32 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp32 at (1265,1-1265,47) */
uint64_t __tmp_in_tmp32;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp32;
}
tmp32[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp32 : 0;
}
}
}
}

auto tmp33 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp33 at (1268,1-1268,37) */
uint64_t __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp33;
}
tmp33[i0] = (role == CLIENT) ? __tmp_in_tmp33 : 0;
}

auto tmp34 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp34 at (1271,1-1271,37) */
uint64_t __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp34;
}
tmp34[i0] = (role == CLIENT) ? __tmp_in_tmp34 : 0;
}

auto tmp35 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp35 at (1274,1-1274,37) */
uint64_t __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp35;
}
tmp35[i0] = (role == CLIENT) ? __tmp_in_tmp35 : 0;
}

auto tmp36 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp36 at (1277,1-1277,37) */
uint64_t __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp36;
}
tmp36[i0] = (role == CLIENT) ? __tmp_in_tmp36 : 0;
}

auto tmp37 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp37 at (1280,1-1280,47) */
uint64_t __tmp_in_tmp37;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp37;
}
tmp37[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp37 : 0;
}
}
}
}

auto tmp38 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp38 at (1283,1-1283,36) */
uint64_t __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp38;
}
tmp38[i0] = (role == CLIENT) ? __tmp_in_tmp38 : 0;
}

auto tmp39 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp39 at (1286,1-1286,36) */
uint64_t __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp39;
}
tmp39[i0] = (role == CLIENT) ? __tmp_in_tmp39 : 0;
}

auto tmp40 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp40 at (1289,1-1289,36) */
uint64_t __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp40;
}
tmp40[i0] = (role == CLIENT) ? __tmp_in_tmp40 : 0;
}

auto tmp41 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp41 at (1292,1-1292,36) */
uint64_t __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp41;
}
tmp41[i0] = (role == CLIENT) ? __tmp_in_tmp41 : 0;
}

auto tmp42 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp42 at (1295,1-1295,46) */
uint64_t __tmp_in_tmp42;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp42;
}
tmp42[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp42 : 0;
}
}
}
}

auto tmp43 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp43 at (1298,1-1298,36) */
uint64_t __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp43;
}
tmp43[i0] = (role == CLIENT) ? __tmp_in_tmp43 : 0;
}

auto tmp44 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp44 at (1301,1-1301,36) */
uint64_t __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp44;
}
tmp44[i0] = (role == CLIENT) ? __tmp_in_tmp44 : 0;
}

auto tmp45 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp45 at (1304,1-1304,36) */
uint64_t __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp45;
}
tmp45[i0] = (role == CLIENT) ? __tmp_in_tmp45 : 0;
}

auto tmp46 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp46 at (1307,1-1307,36) */
uint64_t __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp46;
}
tmp46[i0] = (role == CLIENT) ? __tmp_in_tmp46 : 0;
}

auto tmp47 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp47 at (1310,1-1310,47) */
uint64_t __tmp_in_tmp47;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp47;
}
tmp47[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp47 : 0;
}
}
}
}

auto tmp48 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp48 at (1313,1-1313,37) */
uint64_t __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp48;
}
tmp48[i0] = (role == CLIENT) ? __tmp_in_tmp48 : 0;
}

auto tmp49 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp49 at (1316,1-1316,37) */
uint64_t __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp49;
}
tmp49[i0] = (role == CLIENT) ? __tmp_in_tmp49 : 0;
}

auto tmp50 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp50 at (1319,1-1319,37) */
uint64_t __tmp_in_tmp50;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp50;
}
tmp50[i0] = (role == CLIENT) ? __tmp_in_tmp50 : 0;
}

auto tmp51 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp51 at (1322,1-1322,37) */
uint64_t __tmp_in_tmp51;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp51;
}
tmp51[i0] = (role == CLIENT) ? __tmp_in_tmp51 : 0;
}

auto tmp52 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp52 at (1325,1-1325,48) */
uint64_t __tmp_in_tmp52;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp52;
}
tmp52[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp52 : 0;
}
}
}
}

auto tmp53 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp53 at (1328,1-1328,48) */
uint64_t __tmp_in_tmp53;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp53;
}
tmp53[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp53 : 0;
}
}
}
}

auto tmp54 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp54 at (1331,1-1331,37) */
uint64_t __tmp_in_tmp54;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp54;
}
tmp54[i0] = (role == CLIENT) ? __tmp_in_tmp54 : 0;
}

auto tmp55 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp55 at (1334,1-1334,37) */
uint64_t __tmp_in_tmp55;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp55;
}
tmp55[i0] = (role == CLIENT) ? __tmp_in_tmp55 : 0;
}

auto tmp56 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp56 at (1337,1-1337,37) */
uint64_t __tmp_in_tmp56;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp56;
}
tmp56[i0] = (role == CLIENT) ? __tmp_in_tmp56 : 0;
}

auto tmp57 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp57 at (1340,1-1340,37) */
uint64_t __tmp_in_tmp57;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp57;
}
tmp57[i0] = (role == CLIENT) ? __tmp_in_tmp57 : 0;
}

auto tmp58 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp58 at (1343,1-1343,48) */
uint64_t __tmp_in_tmp58;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp58;
}
tmp58[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp58 : 0;
}
}
}
}

auto tmp59 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp59 at (1346,1-1346,37) */
uint64_t __tmp_in_tmp59;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp59;
}
tmp59[i0] = (role == CLIENT) ? __tmp_in_tmp59 : 0;
}

auto tmp60 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp60 at (1349,1-1349,37) */
uint64_t __tmp_in_tmp60;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp60;
}
tmp60[i0] = (role == CLIENT) ? __tmp_in_tmp60 : 0;
}

auto tmp61 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp61 at (1352,1-1352,37) */
uint64_t __tmp_in_tmp61;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp61;
}
tmp61[i0] = (role == CLIENT) ? __tmp_in_tmp61 : 0;
}

auto tmp62 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp62 at (1355,1-1355,37) */
uint64_t __tmp_in_tmp62;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp62;
}
tmp62[i0] = (role == CLIENT) ? __tmp_in_tmp62 : 0;
}

auto tmp63 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp63 at (1358,1-1358,48) */
uint64_t __tmp_in_tmp63;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp63;
}
tmp63[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp63 : 0;
}
}
}
}

auto tmp64 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp64 at (1361,1-1361,37) */
uint64_t __tmp_in_tmp64;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp64;
}
tmp64[i0] = (role == CLIENT) ? __tmp_in_tmp64 : 0;
}

auto tmp65 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp65 at (1364,1-1364,37) */
uint64_t __tmp_in_tmp65;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp65;
}
tmp65[i0] = (role == CLIENT) ? __tmp_in_tmp65 : 0;
}

auto tmp66 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp66 at (1367,1-1367,37) */
uint64_t __tmp_in_tmp66;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp66;
}
tmp66[i0] = (role == CLIENT) ? __tmp_in_tmp66 : 0;
}

auto tmp67 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp67 at (1370,1-1370,37) */
uint64_t __tmp_in_tmp67;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp67;
}
tmp67[i0] = (role == CLIENT) ? __tmp_in_tmp67 : 0;
}

auto tmp68 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp68 at (1373,1-1373,48) */
uint64_t __tmp_in_tmp68;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp68;
}
tmp68[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp68 : 0;
}
}
}
}

auto tmp69 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp69 at (1376,1-1376,37) */
uint64_t __tmp_in_tmp69;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp69;
}
tmp69[i0] = (role == CLIENT) ? __tmp_in_tmp69 : 0;
}

auto tmp70 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp70 at (1379,1-1379,37) */
uint64_t __tmp_in_tmp70;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp70;
}
tmp70[i0] = (role == CLIENT) ? __tmp_in_tmp70 : 0;
}

auto tmp71 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp71 at (1382,1-1382,37) */
uint64_t __tmp_in_tmp71;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp71;
}
tmp71[i0] = (role == CLIENT) ? __tmp_in_tmp71 : 0;
}

auto tmp72 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp72 at (1385,1-1385,37) */
uint64_t __tmp_in_tmp72;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp72;
}
tmp72[i0] = (role == CLIENT) ? __tmp_in_tmp72 : 0;
}

auto tmp73 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp73 at (1388,1-1388,48) */
uint64_t __tmp_in_tmp73;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp73;
}
tmp73[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp73 : 0;
}
}
}
}

auto tmp74 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp74 at (1391,1-1391,37) */
uint64_t __tmp_in_tmp74;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp74;
}
tmp74[i0] = (role == CLIENT) ? __tmp_in_tmp74 : 0;
}

auto tmp75 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp75 at (1394,1-1394,37) */
uint64_t __tmp_in_tmp75;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp75;
}
tmp75[i0] = (role == CLIENT) ? __tmp_in_tmp75 : 0;
}

auto tmp76 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp76 at (1397,1-1397,37) */
uint64_t __tmp_in_tmp76;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp76;
}
tmp76[i0] = (role == CLIENT) ? __tmp_in_tmp76 : 0;
}

auto tmp77 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp77 at (1400,1-1400,37) */
uint64_t __tmp_in_tmp77;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp77;
}
tmp77[i0] = (role == CLIENT) ? __tmp_in_tmp77 : 0;
}

auto tmp78 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp78 at (1403,1-1403,48) */
uint64_t __tmp_in_tmp78;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp78;
}
tmp78[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp78 : 0;
}
}
}
}

auto tmp79 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp79 at (1406,1-1406,37) */
uint64_t __tmp_in_tmp79;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp79;
}
tmp79[i0] = (role == CLIENT) ? __tmp_in_tmp79 : 0;
}

auto tmp80 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp80 at (1409,1-1409,37) */
uint64_t __tmp_in_tmp80;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp80;
}
tmp80[i0] = (role == CLIENT) ? __tmp_in_tmp80 : 0;
}

auto tmp81 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp81 at (1412,1-1412,37) */
uint64_t __tmp_in_tmp81;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp81;
}
tmp81[i0] = (role == CLIENT) ? __tmp_in_tmp81 : 0;
}

auto tmp82 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp82 at (1415,1-1415,37) */
uint64_t __tmp_in_tmp82;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp82;
}
tmp82[i0] = (role == CLIENT) ? __tmp_in_tmp82 : 0;
}

auto tmp83 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp83 at (1418,1-1418,48) */
uint64_t __tmp_in_tmp83;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp83;
}
tmp83[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp83 : 0;
}
}
}
}

auto tmp84 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp84 at (1421,1-1421,37) */
uint64_t __tmp_in_tmp84;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp84;
}
tmp84[i0] = (role == CLIENT) ? __tmp_in_tmp84 : 0;
}

auto tmp85 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp85 at (1424,1-1424,37) */
uint64_t __tmp_in_tmp85;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp85;
}
tmp85[i0] = (role == CLIENT) ? __tmp_in_tmp85 : 0;
}

auto tmp86 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp86 at (1427,1-1427,37) */
uint64_t __tmp_in_tmp86;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp86;
}
tmp86[i0] = (role == CLIENT) ? __tmp_in_tmp86 : 0;
}

auto tmp87 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp87 at (1430,1-1430,37) */
uint64_t __tmp_in_tmp87;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp87;
}
tmp87[i0] = (role == CLIENT) ? __tmp_in_tmp87 : 0;
}

auto tmp88 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp88 at (1433,1-1433,48) */
uint64_t __tmp_in_tmp88;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp88;
}
tmp88[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp88 : 0;
}
}
}
}

auto tmp89 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp89 at (1436,1-1436,37) */
uint64_t __tmp_in_tmp89;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp89;
}
tmp89[i0] = (role == CLIENT) ? __tmp_in_tmp89 : 0;
}

auto tmp90 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp90 at (1439,1-1439,37) */
uint64_t __tmp_in_tmp90;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp90;
}
tmp90[i0] = (role == CLIENT) ? __tmp_in_tmp90 : 0;
}

auto tmp91 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp91 at (1442,1-1442,37) */
uint64_t __tmp_in_tmp91;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp91;
}
tmp91[i0] = (role == CLIENT) ? __tmp_in_tmp91 : 0;
}

auto tmp92 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp92 at (1445,1-1445,37) */
uint64_t __tmp_in_tmp92;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp92;
}
tmp92[i0] = (role == CLIENT) ? __tmp_in_tmp92 : 0;
}

auto tmp93 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp93 at (1448,1-1448,48) */
uint64_t __tmp_in_tmp93;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp93;
}
tmp93[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp93 : 0;
}
}
}
}

auto tmp94 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp94 at (1451,1-1451,37) */
uint64_t __tmp_in_tmp94;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp94;
}
tmp94[i0] = (role == CLIENT) ? __tmp_in_tmp94 : 0;
}

auto tmp95 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp95 at (1454,1-1454,37) */
uint64_t __tmp_in_tmp95;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp95;
}
tmp95[i0] = (role == CLIENT) ? __tmp_in_tmp95 : 0;
}

auto tmp96 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp96 at (1457,1-1457,37) */
uint64_t __tmp_in_tmp96;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp96;
}
tmp96[i0] = (role == CLIENT) ? __tmp_in_tmp96 : 0;
}

auto tmp97 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp97 at (1460,1-1460,37) */
uint64_t __tmp_in_tmp97;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp97;
}
tmp97[i0] = (role == CLIENT) ? __tmp_in_tmp97 : 0;
}

auto tmp98 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp98 at (1463,1-1463,48) */
uint64_t __tmp_in_tmp98;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp98;
}
tmp98[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp98 : 0;
}
}
}
}

auto tmp99 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp99 at (1466,1-1466,37) */
uint64_t __tmp_in_tmp99;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp99;
}
tmp99[i0] = (role == CLIENT) ? __tmp_in_tmp99 : 0;
}

auto tmp100 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp100 at (1469,1-1469,38) */
uint64_t __tmp_in_tmp100;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp100;
}
tmp100[i0] = (role == CLIENT) ? __tmp_in_tmp100 : 0;
}

auto tmp101 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp101 at (1472,1-1472,38) */
uint64_t __tmp_in_tmp101;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp101;
}
tmp101[i0] = (role == CLIENT) ? __tmp_in_tmp101 : 0;
}

auto tmp102 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp102 at (1475,1-1475,38) */
uint64_t __tmp_in_tmp102;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp102;
}
tmp102[i0] = (role == CLIENT) ? __tmp_in_tmp102 : 0;
}

auto tmp103 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp103 at (1478,1-1478,49) */
uint64_t __tmp_in_tmp103;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp103;
}
tmp103[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp103 : 0;
}
}
}
}

auto tmp104 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp104 at (1481,1-1481,38) */
uint64_t __tmp_in_tmp104;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp104;
}
tmp104[i0] = (role == CLIENT) ? __tmp_in_tmp104 : 0;
}

auto tmp105 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp105 at (1484,1-1484,38) */
uint64_t __tmp_in_tmp105;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp105;
}
tmp105[i0] = (role == CLIENT) ? __tmp_in_tmp105 : 0;
}

auto tmp106 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp106 at (1487,1-1487,38) */
uint64_t __tmp_in_tmp106;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp106;
}
tmp106[i0] = (role == CLIENT) ? __tmp_in_tmp106 : 0;
}

auto tmp107 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp107 at (1490,1-1490,38) */
uint64_t __tmp_in_tmp107;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp107;
}
tmp107[i0] = (role == CLIENT) ? __tmp_in_tmp107 : 0;
}

auto tmp108 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp108 at (1493,1-1493,49) */
uint64_t __tmp_in_tmp108;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp108;
}
tmp108[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp108 : 0;
}
}
}
}

auto tmp109 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp109 at (1496,1-1496,38) */
uint64_t __tmp_in_tmp109;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp109;
}
tmp109[i0] = (role == CLIENT) ? __tmp_in_tmp109 : 0;
}

auto tmp110 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp110 at (1499,1-1499,38) */
uint64_t __tmp_in_tmp110;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp110;
}
tmp110[i0] = (role == CLIENT) ? __tmp_in_tmp110 : 0;
}

auto tmp111 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp111 at (1502,1-1502,38) */
uint64_t __tmp_in_tmp111;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp111;
}
tmp111[i0] = (role == CLIENT) ? __tmp_in_tmp111 : 0;
}

auto tmp112 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp112 at (1505,1-1505,38) */
uint64_t __tmp_in_tmp112;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp112;
}
tmp112[i0] = (role == CLIENT) ? __tmp_in_tmp112 : 0;
}

auto tmp113 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp113 at (1508,1-1508,50) */
uint64_t __tmp_in_tmp113;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp113;
}
tmp113[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp113 : 0;
}
}
}
}

auto tmp114 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp114 at (1511,1-1511,49) */
uint64_t __tmp_in_tmp114;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp114;
}
tmp114[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp114 : 0;
}
}
}
}

auto tmp115 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp115 at (1514,1-1514,38) */
uint64_t __tmp_in_tmp115;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp115;
}
tmp115[i0] = (role == CLIENT) ? __tmp_in_tmp115 : 0;
}

auto tmp116 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp116 at (1517,1-1517,38) */
uint64_t __tmp_in_tmp116;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp116;
}
tmp116[i0] = (role == CLIENT) ? __tmp_in_tmp116 : 0;
}

auto tmp117 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp117 at (1520,1-1520,38) */
uint64_t __tmp_in_tmp117;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp117;
}
tmp117[i0] = (role == CLIENT) ? __tmp_in_tmp117 : 0;
}

auto tmp118 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp118 at (1523,1-1523,38) */
uint64_t __tmp_in_tmp118;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp118;
}
tmp118[i0] = (role == CLIENT) ? __tmp_in_tmp118 : 0;
}

auto tmp119 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp119 at (1526,1-1526,49) */
uint64_t __tmp_in_tmp119;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp119;
}
tmp119[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp119 : 0;
}
}
}
}

auto tmp120 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp120 at (1529,1-1529,38) */
uint64_t __tmp_in_tmp120;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp120;
}
tmp120[i0] = (role == CLIENT) ? __tmp_in_tmp120 : 0;
}

auto tmp121 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp121 at (1532,1-1532,38) */
uint64_t __tmp_in_tmp121;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp121;
}
tmp121[i0] = (role == CLIENT) ? __tmp_in_tmp121 : 0;
}

auto tmp122 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp122 at (1535,1-1535,38) */
uint64_t __tmp_in_tmp122;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp122;
}
tmp122[i0] = (role == CLIENT) ? __tmp_in_tmp122 : 0;
}

auto tmp123 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp123 at (1538,1-1538,38) */
uint64_t __tmp_in_tmp123;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp123;
}
tmp123[i0] = (role == CLIENT) ? __tmp_in_tmp123 : 0;
}

auto tmp124 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp124 at (1541,1-1541,50) */
uint64_t __tmp_in_tmp124;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp124;
}
tmp124[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp124 : 0;
}
}
}
}

auto tmp125 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp125 at (1544,1-1544,39) */
uint64_t __tmp_in_tmp125;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp125;
}
tmp125[i0] = (role == CLIENT) ? __tmp_in_tmp125 : 0;
}

auto tmp126 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp126 at (1547,1-1547,39) */
uint64_t __tmp_in_tmp126;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp126;
}
tmp126[i0] = (role == CLIENT) ? __tmp_in_tmp126 : 0;
}

auto tmp127 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp127 at (1550,1-1550,39) */
uint64_t __tmp_in_tmp127;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp127;
}
tmp127[i0] = (role == CLIENT) ? __tmp_in_tmp127 : 0;
}

auto tmp128 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp128 at (1553,1-1553,39) */
uint64_t __tmp_in_tmp128;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp128;
}
tmp128[i0] = (role == CLIENT) ? __tmp_in_tmp128 : 0;
}

auto tmp129 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp129 at (1556,1-1556,50) */
uint64_t __tmp_in_tmp129;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp129;
}
tmp129[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp129 : 0;
}
}
}
}

auto tmp130 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp130 at (1559,1-1559,38) */
uint64_t __tmp_in_tmp130;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp130;
}
tmp130[i0] = (role == CLIENT) ? __tmp_in_tmp130 : 0;
}

auto tmp131 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp131 at (1562,1-1562,38) */
uint64_t __tmp_in_tmp131;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp131;
}
tmp131[i0] = (role == CLIENT) ? __tmp_in_tmp131 : 0;
}

auto tmp132 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp132 at (1565,1-1565,38) */
uint64_t __tmp_in_tmp132;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp132;
}
tmp132[i0] = (role == CLIENT) ? __tmp_in_tmp132 : 0;
}

auto tmp133 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp133 at (1568,1-1568,38) */
uint64_t __tmp_in_tmp133;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp133;
}
tmp133[i0] = (role == CLIENT) ? __tmp_in_tmp133 : 0;
}

auto tmp134 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp134 at (1571,1-1571,49) */
uint64_t __tmp_in_tmp134;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp134;
}
tmp134[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp134 : 0;
}
}
}
}

auto tmp135 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp135 at (1574,1-1574,38) */
uint64_t __tmp_in_tmp135;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp135;
}
tmp135[i0] = (role == CLIENT) ? __tmp_in_tmp135 : 0;
}

auto tmp136 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp136 at (1577,1-1577,38) */
uint64_t __tmp_in_tmp136;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp136;
}
tmp136[i0] = (role == CLIENT) ? __tmp_in_tmp136 : 0;
}

auto tmp137 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp137 at (1580,1-1580,38) */
uint64_t __tmp_in_tmp137;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp137;
}
tmp137[i0] = (role == CLIENT) ? __tmp_in_tmp137 : 0;
}

auto tmp138 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp138 at (1583,1-1583,38) */
uint64_t __tmp_in_tmp138;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp138;
}
tmp138[i0] = (role == CLIENT) ? __tmp_in_tmp138 : 0;
}

auto tmp139 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp139 at (1586,1-1586,50) */
uint64_t __tmp_in_tmp139;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp139;
}
tmp139[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp139 : 0;
}
}
}
}

auto tmp140 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp140 at (1589,1-1589,39) */
uint64_t __tmp_in_tmp140;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp140;
}
tmp140[i0] = (role == CLIENT) ? __tmp_in_tmp140 : 0;
}

auto tmp141 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp141 at (1592,1-1592,39) */
uint64_t __tmp_in_tmp141;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp141;
}
tmp141[i0] = (role == CLIENT) ? __tmp_in_tmp141 : 0;
}

auto tmp142 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp142 at (1595,1-1595,39) */
uint64_t __tmp_in_tmp142;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp142;
}
tmp142[i0] = (role == CLIENT) ? __tmp_in_tmp142 : 0;
}

auto tmp143 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp143 at (1598,1-1598,39) */
uint64_t __tmp_in_tmp143;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp143;
}
tmp143[i0] = (role == CLIENT) ? __tmp_in_tmp143 : 0;
}

auto tmp144 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp144 at (1601,1-1601,50) */
uint64_t __tmp_in_tmp144;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp144;
}
tmp144[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp144 : 0;
}
}
}
}

auto tmp145 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp145 at (1604,1-1604,38) */
uint64_t __tmp_in_tmp145;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp145;
}
tmp145[i0] = (role == CLIENT) ? __tmp_in_tmp145 : 0;
}

auto tmp146 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp146 at (1607,1-1607,38) */
uint64_t __tmp_in_tmp146;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp146;
}
tmp146[i0] = (role == CLIENT) ? __tmp_in_tmp146 : 0;
}

auto tmp147 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp147 at (1610,1-1610,38) */
uint64_t __tmp_in_tmp147;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp147;
}
tmp147[i0] = (role == CLIENT) ? __tmp_in_tmp147 : 0;
}

auto tmp148 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp148 at (1613,1-1613,38) */
uint64_t __tmp_in_tmp148;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp148;
}
tmp148[i0] = (role == CLIENT) ? __tmp_in_tmp148 : 0;
}

auto tmp149 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp149 at (1616,1-1616,49) */
uint64_t __tmp_in_tmp149;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp149;
}
tmp149[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp149 : 0;
}
}
}
}

auto tmp150 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp150 at (1619,1-1619,38) */
uint64_t __tmp_in_tmp150;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp150;
}
tmp150[i0] = (role == CLIENT) ? __tmp_in_tmp150 : 0;
}

auto tmp151 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp151 at (1622,1-1622,38) */
uint64_t __tmp_in_tmp151;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp151;
}
tmp151[i0] = (role == CLIENT) ? __tmp_in_tmp151 : 0;
}

auto tmp152 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp152 at (1625,1-1625,38) */
uint64_t __tmp_in_tmp152;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp152;
}
tmp152[i0] = (role == CLIENT) ? __tmp_in_tmp152 : 0;
}

auto tmp153 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp153 at (1628,1-1628,38) */
uint64_t __tmp_in_tmp153;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp153;
}
tmp153[i0] = (role == CLIENT) ? __tmp_in_tmp153 : 0;
}

auto tmp154 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp154 at (1631,1-1631,50) */
uint64_t __tmp_in_tmp154;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp154;
}
tmp154[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp154 : 0;
}
}
}
}

auto tmp155 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp155 at (1634,1-1634,39) */
uint64_t __tmp_in_tmp155;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp155;
}
tmp155[i0] = (role == CLIENT) ? __tmp_in_tmp155 : 0;
}

auto tmp156 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp156 at (1637,1-1637,39) */
uint64_t __tmp_in_tmp156;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp156;
}
tmp156[i0] = (role == CLIENT) ? __tmp_in_tmp156 : 0;
}

auto tmp157 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp157 at (1640,1-1640,39) */
uint64_t __tmp_in_tmp157;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp157;
}
tmp157[i0] = (role == CLIENT) ? __tmp_in_tmp157 : 0;
}

auto tmp158 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp158 at (1643,1-1643,39) */
uint64_t __tmp_in_tmp158;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp158;
}
tmp158[i0] = (role == CLIENT) ? __tmp_in_tmp158 : 0;
}

auto tmp159 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp159 at (1646,1-1646,50) */
uint64_t __tmp_in_tmp159;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp159;
}
tmp159[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp159 : 0;
}
}
}
}

auto tmp160 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp160 at (1649,1-1649,38) */
uint64_t __tmp_in_tmp160;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp160;
}
tmp160[i0] = (role == CLIENT) ? __tmp_in_tmp160 : 0;
}

auto tmp161 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp161 at (1652,1-1652,38) */
uint64_t __tmp_in_tmp161;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp161;
}
tmp161[i0] = (role == CLIENT) ? __tmp_in_tmp161 : 0;
}

auto tmp162 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp162 at (1655,1-1655,38) */
uint64_t __tmp_in_tmp162;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp162;
}
tmp162[i0] = (role == CLIENT) ? __tmp_in_tmp162 : 0;
}

auto tmp163 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp163 at (1658,1-1658,38) */
uint64_t __tmp_in_tmp163;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp163;
}
tmp163[i0] = (role == CLIENT) ? __tmp_in_tmp163 : 0;
}

auto tmp164 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp164 at (1661,1-1661,49) */
uint64_t __tmp_in_tmp164;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp164;
}
tmp164[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp164 : 0;
}
}
}
}

auto tmp165 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp165 at (1664,1-1664,38) */
uint64_t __tmp_in_tmp165;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp165;
}
tmp165[i0] = (role == CLIENT) ? __tmp_in_tmp165 : 0;
}

auto tmp166 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp166 at (1667,1-1667,38) */
uint64_t __tmp_in_tmp166;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp166;
}
tmp166[i0] = (role == CLIENT) ? __tmp_in_tmp166 : 0;
}

auto tmp167 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp167 at (1670,1-1670,38) */
uint64_t __tmp_in_tmp167;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp167;
}
tmp167[i0] = (role == CLIENT) ? __tmp_in_tmp167 : 0;
}

auto tmp168 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp168 at (1673,1-1673,38) */
uint64_t __tmp_in_tmp168;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp168;
}
tmp168[i0] = (role == CLIENT) ? __tmp_in_tmp168 : 0;
}

auto tmp169 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp169 at (1676,1-1676,50) */
uint64_t __tmp_in_tmp169;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp169;
}
tmp169[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp169 : 0;
}
}
}
}

auto tmp170 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp170 at (1679,1-1679,39) */
uint64_t __tmp_in_tmp170;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp170;
}
tmp170[i0] = (role == CLIENT) ? __tmp_in_tmp170 : 0;
}

auto tmp171 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp171 at (1682,1-1682,39) */
uint64_t __tmp_in_tmp171;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp171;
}
tmp171[i0] = (role == CLIENT) ? __tmp_in_tmp171 : 0;
}

auto tmp172 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp172 at (1685,1-1685,39) */
uint64_t __tmp_in_tmp172;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp172;
}
tmp172[i0] = (role == CLIENT) ? __tmp_in_tmp172 : 0;
}

auto tmp173 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp173 at (1688,1-1688,39) */
uint64_t __tmp_in_tmp173;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp173;
}
tmp173[i0] = (role == CLIENT) ? __tmp_in_tmp173 : 0;
}

auto tmp174 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp174 at (1691,1-1691,50) */
uint64_t __tmp_in_tmp174;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp174;
}
tmp174[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp174 : 0;
}
}
}
}

auto tmp175 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp175 at (1694,1-1694,38) */
uint64_t __tmp_in_tmp175;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp175;
}
tmp175[i0] = (role == CLIENT) ? __tmp_in_tmp175 : 0;
}

auto tmp176 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp176 at (1697,1-1697,38) */
uint64_t __tmp_in_tmp176;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp176;
}
tmp176[i0] = (role == CLIENT) ? __tmp_in_tmp176 : 0;
}

auto tmp177 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp177 at (1700,1-1700,38) */
uint64_t __tmp_in_tmp177;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp177;
}
tmp177[i0] = (role == CLIENT) ? __tmp_in_tmp177 : 0;
}

auto tmp178 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp178 at (1703,1-1703,38) */
uint64_t __tmp_in_tmp178;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp178;
}
tmp178[i0] = (role == CLIENT) ? __tmp_in_tmp178 : 0;
}

auto tmp179 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp179 at (1706,1-1706,49) */
uint64_t __tmp_in_tmp179;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp179;
}
tmp179[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp179 : 0;
}
}
}
}

auto tmp180 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp180 at (1709,1-1709,38) */
uint64_t __tmp_in_tmp180;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp180;
}
tmp180[i0] = (role == CLIENT) ? __tmp_in_tmp180 : 0;
}

auto tmp181 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp181 at (1712,1-1712,38) */
uint64_t __tmp_in_tmp181;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp181;
}
tmp181[i0] = (role == CLIENT) ? __tmp_in_tmp181 : 0;
}

auto tmp182 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp182 at (1715,1-1715,38) */
uint64_t __tmp_in_tmp182;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp182;
}
tmp182[i0] = (role == CLIENT) ? __tmp_in_tmp182 : 0;
}

auto tmp183 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp183 at (1718,1-1718,38) */
uint64_t __tmp_in_tmp183;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp183;
}
tmp183[i0] = (role == CLIENT) ? __tmp_in_tmp183 : 0;
}

auto tmp184 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp184 at (1721,1-1721,50) */
uint64_t __tmp_in_tmp184;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp184;
}
tmp184[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp184 : 0;
}
}
}
}

auto tmp185 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp185 at (1724,1-1724,39) */
uint64_t __tmp_in_tmp185;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp185;
}
tmp185[i0] = (role == CLIENT) ? __tmp_in_tmp185 : 0;
}

auto tmp186 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp186 at (1727,1-1727,39) */
uint64_t __tmp_in_tmp186;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp186;
}
tmp186[i0] = (role == CLIENT) ? __tmp_in_tmp186 : 0;
}

auto tmp187 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp187 at (1730,1-1730,39) */
uint64_t __tmp_in_tmp187;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp187;
}
tmp187[i0] = (role == CLIENT) ? __tmp_in_tmp187 : 0;
}

auto tmp188 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp188 at (1733,1-1733,39) */
uint64_t __tmp_in_tmp188;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp188;
}
tmp188[i0] = (role == CLIENT) ? __tmp_in_tmp188 : 0;
}

auto tmp189 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp189 at (1736,1-1736,50) */
uint64_t __tmp_in_tmp189;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp189;
}
tmp189[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp189 : 0;
}
}
}
}

auto tmp190 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp190 at (1739,1-1739,38) */
uint64_t __tmp_in_tmp190;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp190;
}
tmp190[i0] = (role == CLIENT) ? __tmp_in_tmp190 : 0;
}

auto tmp191 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp191 at (1742,1-1742,38) */
uint64_t __tmp_in_tmp191;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp191;
}
tmp191[i0] = (role == CLIENT) ? __tmp_in_tmp191 : 0;
}

auto tmp192 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp192 at (1745,1-1745,38) */
uint64_t __tmp_in_tmp192;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp192;
}
tmp192[i0] = (role == CLIENT) ? __tmp_in_tmp192 : 0;
}

auto tmp193 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp193 at (1748,1-1748,38) */
uint64_t __tmp_in_tmp193;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp193;
}
tmp193[i0] = (role == CLIENT) ? __tmp_in_tmp193 : 0;
}

auto tmp194 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp194 at (1751,1-1751,49) */
uint64_t __tmp_in_tmp194;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp194;
}
tmp194[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp194 : 0;
}
}
}
}

auto tmp195 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp195 at (1754,1-1754,38) */
uint64_t __tmp_in_tmp195;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp195;
}
tmp195[i0] = (role == CLIENT) ? __tmp_in_tmp195 : 0;
}

auto tmp196 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp196 at (1757,1-1757,38) */
uint64_t __tmp_in_tmp196;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp196;
}
tmp196[i0] = (role == CLIENT) ? __tmp_in_tmp196 : 0;
}

auto tmp197 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp197 at (1760,1-1760,38) */
uint64_t __tmp_in_tmp197;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp197;
}
tmp197[i0] = (role == CLIENT) ? __tmp_in_tmp197 : 0;
}

auto tmp198 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp198 at (1763,1-1763,38) */
uint64_t __tmp_in_tmp198;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp198;
}
tmp198[i0] = (role == CLIENT) ? __tmp_in_tmp198 : 0;
}

auto tmp199 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp199 at (1766,1-1766,50) */
uint64_t __tmp_in_tmp199;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp199;
}
tmp199[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp199 : 0;
}
}
}
}

auto tmp200 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp200 at (1769,1-1769,39) */
uint64_t __tmp_in_tmp200;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp200;
}
tmp200[i0] = (role == CLIENT) ? __tmp_in_tmp200 : 0;
}

auto tmp201 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp201 at (1772,1-1772,39) */
uint64_t __tmp_in_tmp201;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp201;
}
tmp201[i0] = (role == CLIENT) ? __tmp_in_tmp201 : 0;
}

auto tmp202 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp202 at (1775,1-1775,39) */
uint64_t __tmp_in_tmp202;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp202;
}
tmp202[i0] = (role == CLIENT) ? __tmp_in_tmp202 : 0;
}

auto tmp203 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp203 at (1778,1-1778,39) */
uint64_t __tmp_in_tmp203;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp203;
}
tmp203[i0] = (role == CLIENT) ? __tmp_in_tmp203 : 0;
}

auto tmp204 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp204 at (1781,1-1781,51) */
uint64_t __tmp_in_tmp204;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp204;
}
tmp204[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp204 : 0;
}
}
}
}

auto tmp205 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp205 at (1784,1-1784,50) */
uint64_t __tmp_in_tmp205;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp205;
}
tmp205[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp205 : 0;
}
}
}
}

auto tmp206 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp206 at (1787,1-1787,38) */
uint64_t __tmp_in_tmp206;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp206;
}
tmp206[i0] = (role == CLIENT) ? __tmp_in_tmp206 : 0;
}

auto tmp207 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp207 at (1790,1-1790,38) */
uint64_t __tmp_in_tmp207;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp207;
}
tmp207[i0] = (role == CLIENT) ? __tmp_in_tmp207 : 0;
}

auto tmp208 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp208 at (1793,1-1793,38) */
uint64_t __tmp_in_tmp208;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp208;
}
tmp208[i0] = (role == CLIENT) ? __tmp_in_tmp208 : 0;
}

auto tmp209 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp209 at (1796,1-1796,38) */
uint64_t __tmp_in_tmp209;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp209;
}
tmp209[i0] = (role == CLIENT) ? __tmp_in_tmp209 : 0;
}

auto tmp210 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp210 at (1799,1-1799,49) */
uint64_t __tmp_in_tmp210;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp210;
}
tmp210[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp210 : 0;
}
}
}
}

auto tmp211 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp211 at (1802,1-1802,38) */
uint64_t __tmp_in_tmp211;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp211;
}
tmp211[i0] = (role == CLIENT) ? __tmp_in_tmp211 : 0;
}

auto tmp212 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp212 at (1805,1-1805,38) */
uint64_t __tmp_in_tmp212;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp212;
}
tmp212[i0] = (role == CLIENT) ? __tmp_in_tmp212 : 0;
}

auto tmp213 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp213 at (1808,1-1808,38) */
uint64_t __tmp_in_tmp213;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp213;
}
tmp213[i0] = (role == CLIENT) ? __tmp_in_tmp213 : 0;
}

auto tmp214 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp214 at (1811,1-1811,38) */
uint64_t __tmp_in_tmp214;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp214;
}
tmp214[i0] = (role == CLIENT) ? __tmp_in_tmp214 : 0;
}

auto tmp215 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp215 at (1814,1-1814,50) */
uint64_t __tmp_in_tmp215;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp215;
}
tmp215[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp215 : 0;
}
}
}
}

auto tmp216 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp216 at (1817,1-1817,39) */
uint64_t __tmp_in_tmp216;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp216;
}
tmp216[i0] = (role == CLIENT) ? __tmp_in_tmp216 : 0;
}

auto tmp217 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp217 at (1820,1-1820,39) */
uint64_t __tmp_in_tmp217;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp217;
}
tmp217[i0] = (role == CLIENT) ? __tmp_in_tmp217 : 0;
}

auto tmp218 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp218 at (1823,1-1823,39) */
uint64_t __tmp_in_tmp218;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp218;
}
tmp218[i0] = (role == CLIENT) ? __tmp_in_tmp218 : 0;
}

auto tmp219 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp219 at (1826,1-1826,39) */
uint64_t __tmp_in_tmp219;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp219;
}
tmp219[i0] = (role == CLIENT) ? __tmp_in_tmp219 : 0;
}

auto tmp220 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp220 at (1829,1-1829,50) */
uint64_t __tmp_in_tmp220;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp220;
}
tmp220[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp220 : 0;
}
}
}
}

auto tmp221 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp221 at (1832,1-1832,38) */
uint64_t __tmp_in_tmp221;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp221;
}
tmp221[i0] = (role == CLIENT) ? __tmp_in_tmp221 : 0;
}

auto tmp222 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp222 at (1835,1-1835,38) */
uint64_t __tmp_in_tmp222;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp222;
}
tmp222[i0] = (role == CLIENT) ? __tmp_in_tmp222 : 0;
}

auto tmp223 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp223 at (1838,1-1838,38) */
uint64_t __tmp_in_tmp223;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp223;
}
tmp223[i0] = (role == CLIENT) ? __tmp_in_tmp223 : 0;
}

auto tmp224 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp224 at (1841,1-1841,38) */
uint64_t __tmp_in_tmp224;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp224;
}
tmp224[i0] = (role == CLIENT) ? __tmp_in_tmp224 : 0;
}

auto tmp225 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp225 at (1844,1-1844,49) */
uint64_t __tmp_in_tmp225;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp225;
}
tmp225[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp225 : 0;
}
}
}
}

auto tmp226 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp226 at (1847,1-1847,38) */
uint64_t __tmp_in_tmp226;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp226;
}
tmp226[i0] = (role == CLIENT) ? __tmp_in_tmp226 : 0;
}

auto tmp227 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp227 at (1850,1-1850,38) */
uint64_t __tmp_in_tmp227;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp227;
}
tmp227[i0] = (role == CLIENT) ? __tmp_in_tmp227 : 0;
}

auto tmp228 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp228 at (1853,1-1853,38) */
uint64_t __tmp_in_tmp228;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp228;
}
tmp228[i0] = (role == CLIENT) ? __tmp_in_tmp228 : 0;
}

auto tmp229 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp229 at (1856,1-1856,38) */
uint64_t __tmp_in_tmp229;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp229;
}
tmp229[i0] = (role == CLIENT) ? __tmp_in_tmp229 : 0;
}

auto tmp230 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp230 at (1859,1-1859,50) */
uint64_t __tmp_in_tmp230;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp230;
}
tmp230[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp230 : 0;
}
}
}
}

auto tmp231 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp231 at (1862,1-1862,39) */
uint64_t __tmp_in_tmp231;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp231;
}
tmp231[i0] = (role == CLIENT) ? __tmp_in_tmp231 : 0;
}

auto tmp232 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp232 at (1865,1-1865,39) */
uint64_t __tmp_in_tmp232;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp232;
}
tmp232[i0] = (role == CLIENT) ? __tmp_in_tmp232 : 0;
}

auto tmp233 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp233 at (1868,1-1868,39) */
uint64_t __tmp_in_tmp233;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp233;
}
tmp233[i0] = (role == CLIENT) ? __tmp_in_tmp233 : 0;
}

auto tmp234 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp234 at (1871,1-1871,39) */
uint64_t __tmp_in_tmp234;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp234;
}
tmp234[i0] = (role == CLIENT) ? __tmp_in_tmp234 : 0;
}

auto tmp235 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp235 at (1874,1-1874,50) */
uint64_t __tmp_in_tmp235;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp235;
}
tmp235[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp235 : 0;
}
}
}
}

auto tmp236 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp236 at (1877,1-1877,38) */
uint64_t __tmp_in_tmp236;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp236;
}
tmp236[i0] = (role == CLIENT) ? __tmp_in_tmp236 : 0;
}

auto tmp237 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp237 at (1880,1-1880,38) */
uint64_t __tmp_in_tmp237;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp237;
}
tmp237[i0] = (role == CLIENT) ? __tmp_in_tmp237 : 0;
}

auto tmp238 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp238 at (1883,1-1883,38) */
uint64_t __tmp_in_tmp238;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp238;
}
tmp238[i0] = (role == CLIENT) ? __tmp_in_tmp238 : 0;
}

auto tmp239 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp239 at (1886,1-1886,38) */
uint64_t __tmp_in_tmp239;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp239;
}
tmp239[i0] = (role == CLIENT) ? __tmp_in_tmp239 : 0;
}

auto tmp240 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp240 at (1889,1-1889,49) */
uint64_t __tmp_in_tmp240;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp240;
}
tmp240[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp240 : 0;
}
}
}
}

auto tmp241 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp241 at (1892,1-1892,38) */
uint64_t __tmp_in_tmp241;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp241;
}
tmp241[i0] = (role == CLIENT) ? __tmp_in_tmp241 : 0;
}

auto tmp242 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp242 at (1895,1-1895,38) */
uint64_t __tmp_in_tmp242;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp242;
}
tmp242[i0] = (role == CLIENT) ? __tmp_in_tmp242 : 0;
}

auto tmp243 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp243 at (1898,1-1898,38) */
uint64_t __tmp_in_tmp243;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp243;
}
tmp243[i0] = (role == CLIENT) ? __tmp_in_tmp243 : 0;
}

auto tmp244 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp244 at (1901,1-1901,38) */
uint64_t __tmp_in_tmp244;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp244;
}
tmp244[i0] = (role == CLIENT) ? __tmp_in_tmp244 : 0;
}

auto tmp245 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp245 at (1904,1-1904,50) */
uint64_t __tmp_in_tmp245;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp245;
}
tmp245[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp245 : 0;
}
}
}
}

auto tmp246 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp246 at (1907,1-1907,39) */
uint64_t __tmp_in_tmp246;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp246;
}
tmp246[i0] = (role == CLIENT) ? __tmp_in_tmp246 : 0;
}

auto tmp247 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp247 at (1910,1-1910,39) */
uint64_t __tmp_in_tmp247;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp247;
}
tmp247[i0] = (role == CLIENT) ? __tmp_in_tmp247 : 0;
}

auto tmp248 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp248 at (1913,1-1913,39) */
uint64_t __tmp_in_tmp248;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp248;
}
tmp248[i0] = (role == CLIENT) ? __tmp_in_tmp248 : 0;
}

auto tmp249 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp249 at (1916,1-1916,39) */
uint64_t __tmp_in_tmp249;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp249;
}
tmp249[i0] = (role == CLIENT) ? __tmp_in_tmp249 : 0;
}

auto tmp250 = make_vector<uint64_t>( (int32_t)2048,  (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp250 at (1919,1-1919,45) */
uint64_t __tmp_in_tmp250;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1001; i1++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp250;
}
tmp250[i0][i1] = (role == CLIENT) ? __tmp_in_tmp250 : 0;
}
}

auto tmp251 = make_vector<uint64_t>( (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp251 at (1922,1-1922,39) */
uint64_t __tmp_in_tmp251;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1001; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp251;
}
tmp251[i0] = (role == CLIENT) ? __tmp_in_tmp251 : 0;
}

//Main Point

leave_time();
print_string("[ARAMIS STATUS]: Aramis initilization completed. Bootstrapping main protocol now...");
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("[ARAMIS STATUS]: Starting main protocol...");
touch_time();
#ifdef PRECOMPUTEAES
auto t1 = high_resolution_clock::now();

#ifndef RUN_SHARECONV_OPTI //If shareconv is off

#ifndef RUN_MSB_OPTI  //If both shareConv and computeMSB are off


#else //If shareConv is off, computeMSB is on


#endif

#else //If share convert opti is ON.

#ifndef RUN_MSB_OPTI //If share conv is on, and msb is off


#else //If share conv is on, and so is msb ; UPDATE : this is when all opti are on
cout<<"DOING PRECOMPUTATION\n";
if (partyNum == PARTY_A)
{
	aes_common->PreComputeKeys(16085667 + 10, NO_CORES);
	aes_a_1->PreComputeKeys(15118043 + 10, NO_CORES);
	aes_b_1->PreComputeKeys(16142043 + 10, NO_CORES);
	aes_c_1->PreComputeKeys(15117520 + 10, NO_CORES);
	aes_share_conv_bit_shares_p0_p2->PreComputeKeys(43294340 + 10, NO_CORES);
	aes_share_conv_shares_mod_odd_p0_p2->PreComputeKeys(7959667 + 10, NO_CORES);
	aes_comp_msb_shares_lsb_p0_p2->PreComputeKeys(2653055 + 10, NO_CORES);
	aes_comp_msb_shares_bit_vec_p0_p2->PreComputeKeys(2653055 + 10, NO_CORES);
	aes_conv_opti_a_1->PreComputeKeys(11727455 + 10, NO_CORES);
	aes_conv_opti_b_1->PreComputeKeys(5380085 + 10, NO_CORES);
	aes_conv_opti_c_1->PreComputeKeys(5556991 + 10, NO_CORES);
}
else if (partyNum == PARTY_B)
{
	aes_common->PreComputeKeys(16085667 + 10, NO_CORES);
	aes_a_2->PreComputeKeys(15118043 + 10, NO_CORES);
	aes_b_2->PreComputeKeys(16142043 + 10, NO_CORES);
	aes_share_conv_bit_shares_p1_p2->PreComputeKeys(43302503 + 10, NO_CORES);
	aes_share_conv_shares_mod_odd_p1_p2->PreComputeKeys(7960167 + 10, NO_CORES);
	aes_comp_msb_shares_lsb_p1_p2->PreComputeKeys(2653555 + 10, NO_CORES);
	aes_comp_msb_shares_bit_vec_p1_p2->PreComputeKeys(2653555 + 10, NO_CORES);
	aes_conv_opti_a_2->PreComputeKeys(11727455 + 10, NO_CORES);
	aes_conv_opti_b_2->PreComputeKeys(5380085 + 10, NO_CORES);
}
else
{
	aes_indep->PreComputeKeys(5306611 + 10, NO_CORES);
	aes_a_1->PreComputeKeys(15118043 + 10, NO_CORES);
	aes_a_2->PreComputeKeys(15118043 + 10, NO_CORES);
	aes_b_1->PreComputeKeys(16142043 + 10, NO_CORES);
	aes_b_2->PreComputeKeys(16142043 + 10, NO_CORES);
	aes_c_1->PreComputeKeys(15117520 + 10, NO_CORES);
	aes_share_conv_bit_shares_p0_p2->PreComputeKeys(43294340 + 10, NO_CORES);
	aes_share_conv_bit_shares_p1_p2->PreComputeKeys(43302503 + 10, NO_CORES);
	aes_share_conv_shares_mod_odd_p0_p2->PreComputeKeys(7959667 + 10, NO_CORES);
	aes_share_conv_shares_mod_odd_p1_p2->PreComputeKeys(7960167 + 10, NO_CORES);
	aes_comp_msb_shares_lsb_p0_p2->PreComputeKeys(2653055 + 10, NO_CORES);
	aes_comp_msb_shares_lsb_p1_p2->PreComputeKeys(2653555 + 10, NO_CORES);
	aes_comp_msb_shares_bit_vec_p0_p2->PreComputeKeys(2653055 + 10, NO_CORES);
	aes_comp_msb_shares_bit_vec_p1_p2->PreComputeKeys(2653555 + 10, NO_CORES);
	aes_conv_opti_a_1->PreComputeKeys(11727455 + 10, NO_CORES);
	aes_conv_opti_a_2->PreComputeKeys(11727455 + 10, NO_CORES);
	aes_conv_opti_b_1->PreComputeKeys(5380085 + 10, NO_CORES);
	aes_conv_opti_b_2->PreComputeKeys(5380085 + 10, NO_CORES);
	aes_conv_opti_c_1->PreComputeKeys(5556991 + 10, NO_CORES);
}

#endif

#endif

auto t2 = high_resolution_clock::now();
auto tt = (duration_cast<duration<double>>(t2 - t1)).count();
cout<<"Time for precomputation = "<<tt<<endl;
#endif

tmp252[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp252[ (int64_t)1][ (int64_t)0] =  (int32_t)3;
tmp252[ (int64_t)1][ (int64_t)1] =  (int32_t)3;
tmp252[ (int64_t)2][ (int64_t)0] =  (int32_t)3;
tmp252[ (int64_t)2][ (int64_t)1] =  (int32_t)3;
tmp252[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)3][ (int64_t)1] =  (int32_t)0;
Pad442( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0,  (int32_t)4,  (int32_t)2, tmp252, tmp253);
CreateTensor4( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64,  (int64_t)32, tmp254);
CreateIdentity44( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64, tmp1, tmp255);
tmp256[ (int64_t)0] =  (int32_t)1;
tmp256[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp253, tmp255, tmp257,  (int64_t)15);
CreateIdentity44( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp257, tmp258);
MaxPool44Hook( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)1,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp258, tmp259);
CreateIdentity44( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp259, tmp260);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp261);
CreateIdentity11( (int32_t)64, tmp2, tmp262);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp263);
CreateIdentity11( (int32_t)64, tmp3, tmp264);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp265);
CreateIdentity11( (int32_t)64, tmp4, tmp266);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp267);
CreateIdentity11( (int32_t)64, tmp5, tmp268);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp260,  (int32_t)64, tmp262, tmp264, tmp269);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp269, tmp270);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256,  (int64_t)32, tmp271);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp6, tmp272);
tmp273[ (int64_t)0] =  (int32_t)1;
tmp273[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp270, tmp272, tmp274,  (int64_t)15);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64,  (int64_t)32, tmp275);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64, tmp7, tmp276);
tmp277[ (int64_t)0] =  (int32_t)1;
tmp277[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp270, tmp276, tmp278,  (int64_t)15);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp279);
CreateIdentity11( (int32_t)64, tmp8, tmp280);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp281);
CreateIdentity11( (int32_t)64, tmp9, tmp282);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp283);
CreateIdentity11( (int32_t)64, tmp10, tmp284);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp285);
CreateIdentity11( (int32_t)64, tmp11, tmp286);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp278,  (int32_t)64, tmp280, tmp282, tmp287);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp287, tmp288);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64,  (int64_t)32, tmp289);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp12, tmp290);
tmp291[ (int64_t)0] =  (int32_t)1;
tmp291[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp288, tmp290, tmp292,  (int64_t)15);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp293);
CreateIdentity11( (int32_t)64, tmp13, tmp294);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp295);
CreateIdentity11( (int32_t)64, tmp14, tmp296);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp297);
CreateIdentity11( (int32_t)64, tmp15, tmp298);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp299);
CreateIdentity11( (int32_t)64, tmp16, tmp300);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp292,  (int32_t)64, tmp294, tmp296, tmp301);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp301, tmp302);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256,  (int64_t)32, tmp303);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp17, tmp304);
tmp305[ (int64_t)0] =  (int32_t)1;
tmp305[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp302, tmp304, tmp306,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp306, tmp274, tmp307);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp308);
CreateIdentity11( (int32_t)256, tmp18, tmp309);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp310);
CreateIdentity11( (int32_t)256, tmp19, tmp311);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp312);
CreateIdentity11( (int32_t)256, tmp20, tmp313);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp314);
CreateIdentity11( (int32_t)256, tmp21, tmp315);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp307,  (int32_t)256, tmp309, tmp311, tmp316);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp316, tmp317);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64,  (int64_t)32, tmp318);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp22, tmp319);
tmp320[ (int64_t)0] =  (int32_t)1;
tmp320[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp317, tmp319, tmp321,  (int64_t)15);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp322);
CreateIdentity11( (int32_t)64, tmp23, tmp323);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp324);
CreateIdentity11( (int32_t)64, tmp24, tmp325);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp326);
CreateIdentity11( (int32_t)64, tmp25, tmp327);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp328);
CreateIdentity11( (int32_t)64, tmp26, tmp329);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp321,  (int32_t)64, tmp323, tmp325, tmp330);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp330, tmp331);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64,  (int64_t)32, tmp332);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp27, tmp333);
tmp334[ (int64_t)0] =  (int32_t)1;
tmp334[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp331, tmp333, tmp335,  (int64_t)15);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp336);
CreateIdentity11( (int32_t)64, tmp28, tmp337);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp338);
CreateIdentity11( (int32_t)64, tmp29, tmp339);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp340);
CreateIdentity11( (int32_t)64, tmp30, tmp341);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp342);
CreateIdentity11( (int32_t)64, tmp31, tmp343);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp335,  (int32_t)64, tmp337, tmp339, tmp344);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp344, tmp345);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256,  (int64_t)32, tmp346);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp32, tmp347);
tmp348[ (int64_t)0] =  (int32_t)1;
tmp348[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp345, tmp347, tmp349,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp349, tmp307, tmp350);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp351);
CreateIdentity11( (int32_t)256, tmp33, tmp352);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp353);
CreateIdentity11( (int32_t)256, tmp34, tmp354);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp355);
CreateIdentity11( (int32_t)256, tmp35, tmp356);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp357);
CreateIdentity11( (int32_t)256, tmp36, tmp358);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp350,  (int32_t)256, tmp352, tmp354, tmp359);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp359, tmp360);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64,  (int64_t)32, tmp361);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp37, tmp362);
tmp363[ (int64_t)0] =  (int32_t)1;
tmp363[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp360, tmp362, tmp364,  (int64_t)15);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp365);
CreateIdentity11( (int32_t)64, tmp38, tmp366);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp367);
CreateIdentity11( (int32_t)64, tmp39, tmp368);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp369);
CreateIdentity11( (int32_t)64, tmp40, tmp370);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp371);
CreateIdentity11( (int32_t)64, tmp41, tmp372);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp364,  (int32_t)64, tmp366, tmp368, tmp373);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp373, tmp374);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64,  (int64_t)32, tmp375);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp42, tmp376);
tmp377[ (int64_t)0] =  (int32_t)1;
tmp377[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp374, tmp376, tmp378,  (int64_t)15);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp379);
CreateIdentity11( (int32_t)64, tmp43, tmp380);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp381);
CreateIdentity11( (int32_t)64, tmp44, tmp382);
CreateTensor1( (int32_t)64,  (int64_t)0, tmp383);
CreateIdentity11( (int32_t)64, tmp45, tmp384);
CreateTensor1( (int32_t)64,  (int64_t)32768, tmp385);
CreateIdentity11( (int32_t)64, tmp46, tmp386);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp378,  (int32_t)64, tmp380, tmp382, tmp387);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp387, tmp388);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256,  (int64_t)32, tmp389);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp47, tmp390);
tmp391[ (int64_t)0] =  (int32_t)1;
tmp391[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp388, tmp390, tmp392,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp392, tmp350, tmp393);
CreateIdentity44( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp393, tmp394);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp395);
CreateIdentity11( (int32_t)256, tmp48, tmp396);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp397);
CreateIdentity11( (int32_t)256, tmp49, tmp398);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp399);
CreateIdentity11( (int32_t)256, tmp50, tmp400);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp401);
CreateIdentity11( (int32_t)256, tmp51, tmp402);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp394,  (int32_t)256, tmp396, tmp398, tmp403);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp403, tmp404);
tmp405[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp405[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp405[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp405[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp405[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp405[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp405[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp405[ (int64_t)3][ (int64_t)1] =  (int32_t)0;
Pad442( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp404,  (int32_t)4,  (int32_t)2, tmp405, tmp406);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512,  (int64_t)32, tmp407);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512, tmp52, tmp408);
tmp409[ (int64_t)0] =  (int32_t)1;
tmp409[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp406, tmp408, tmp410,  (int64_t)15);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128,  (int64_t)32, tmp411);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128, tmp53, tmp412);
tmp413[ (int64_t)0] =  (int32_t)1;
tmp413[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp404, tmp412, tmp414,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp415);
CreateIdentity11( (int32_t)128, tmp54, tmp416);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp417);
CreateIdentity11( (int32_t)128, tmp55, tmp418);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp419);
CreateIdentity11( (int32_t)128, tmp56, tmp420);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp421);
CreateIdentity11( (int32_t)128, tmp57, tmp422);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp414,  (int32_t)128, tmp416, tmp418, tmp423);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp423, tmp424);
tmp425[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp425[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp425[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp425[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp425[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp425[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp425[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp425[ (int64_t)3][ (int64_t)1] =  (int32_t)0;
Pad442( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp424,  (int32_t)4,  (int32_t)2, tmp425, tmp426);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128,  (int64_t)32, tmp427);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp58, tmp428);
tmp429[ (int64_t)0] =  (int32_t)1;
tmp429[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp426, tmp428, tmp430,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp431);
CreateIdentity11( (int32_t)128, tmp59, tmp432);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp433);
CreateIdentity11( (int32_t)128, tmp60, tmp434);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp435);
CreateIdentity11( (int32_t)128, tmp61, tmp436);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp437);
CreateIdentity11( (int32_t)128, tmp62, tmp438);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp430,  (int32_t)128, tmp432, tmp434, tmp439);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp439, tmp440);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512,  (int64_t)32, tmp441);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp63, tmp442);
tmp443[ (int64_t)0] =  (int32_t)1;
tmp443[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp440, tmp442, tmp444,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp444, tmp410, tmp445);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp446);
CreateIdentity11( (int32_t)512, tmp64, tmp447);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp448);
CreateIdentity11( (int32_t)512, tmp65, tmp449);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp450);
CreateIdentity11( (int32_t)512, tmp66, tmp451);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp452);
CreateIdentity11( (int32_t)512, tmp67, tmp453);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp445,  (int32_t)512, tmp447, tmp449, tmp454);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp454, tmp455);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128,  (int64_t)32, tmp456);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp68, tmp457);
tmp458[ (int64_t)0] =  (int32_t)1;
tmp458[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp455, tmp457, tmp459,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp460);
CreateIdentity11( (int32_t)128, tmp69, tmp461);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp462);
CreateIdentity11( (int32_t)128, tmp70, tmp463);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp464);
CreateIdentity11( (int32_t)128, tmp71, tmp465);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp466);
CreateIdentity11( (int32_t)128, tmp72, tmp467);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp459,  (int32_t)128, tmp461, tmp463, tmp468);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp468, tmp469);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128,  (int64_t)32, tmp470);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp73, tmp471);
tmp472[ (int64_t)0] =  (int32_t)1;
tmp472[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp469, tmp471, tmp473,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp474);
CreateIdentity11( (int32_t)128, tmp74, tmp475);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp476);
CreateIdentity11( (int32_t)128, tmp75, tmp477);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp478);
CreateIdentity11( (int32_t)128, tmp76, tmp479);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp480);
CreateIdentity11( (int32_t)128, tmp77, tmp481);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp473,  (int32_t)128, tmp475, tmp477, tmp482);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp482, tmp483);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512,  (int64_t)32, tmp484);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp78, tmp485);
tmp486[ (int64_t)0] =  (int32_t)1;
tmp486[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp483, tmp485, tmp487,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp487, tmp445, tmp488);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp489);
CreateIdentity11( (int32_t)512, tmp79, tmp490);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp491);
CreateIdentity11( (int32_t)512, tmp80, tmp492);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp493);
CreateIdentity11( (int32_t)512, tmp81, tmp494);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp495);
CreateIdentity11( (int32_t)512, tmp82, tmp496);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp488,  (int32_t)512, tmp490, tmp492, tmp497);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp497, tmp498);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128,  (int64_t)32, tmp499);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp83, tmp500);
tmp501[ (int64_t)0] =  (int32_t)1;
tmp501[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp498, tmp500, tmp502,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp503);
CreateIdentity11( (int32_t)128, tmp84, tmp504);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp505);
CreateIdentity11( (int32_t)128, tmp85, tmp506);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp507);
CreateIdentity11( (int32_t)128, tmp86, tmp508);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp509);
CreateIdentity11( (int32_t)128, tmp87, tmp510);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp502,  (int32_t)128, tmp504, tmp506, tmp511);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp511, tmp512);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128,  (int64_t)32, tmp513);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp88, tmp514);
tmp515[ (int64_t)0] =  (int32_t)1;
tmp515[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp512, tmp514, tmp516,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp517);
CreateIdentity11( (int32_t)128, tmp89, tmp518);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp519);
CreateIdentity11( (int32_t)128, tmp90, tmp520);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp521);
CreateIdentity11( (int32_t)128, tmp91, tmp522);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp523);
CreateIdentity11( (int32_t)128, tmp92, tmp524);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp516,  (int32_t)128, tmp518, tmp520, tmp525);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp525, tmp526);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512,  (int64_t)32, tmp527);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp93, tmp528);
tmp529[ (int64_t)0] =  (int32_t)1;
tmp529[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp526, tmp528, tmp530,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp530, tmp488, tmp531);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp532);
CreateIdentity11( (int32_t)512, tmp94, tmp533);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp534);
CreateIdentity11( (int32_t)512, tmp95, tmp535);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp536);
CreateIdentity11( (int32_t)512, tmp96, tmp537);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp538);
CreateIdentity11( (int32_t)512, tmp97, tmp539);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp531,  (int32_t)512, tmp533, tmp535, tmp540);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp540, tmp541);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128,  (int64_t)32, tmp542);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp98, tmp543);
tmp544[ (int64_t)0] =  (int32_t)1;
tmp544[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp541, tmp543, tmp545,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp546);
CreateIdentity11( (int32_t)128, tmp99, tmp547);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp548);
CreateIdentity11( (int32_t)128, tmp100, tmp549);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp550);
CreateIdentity11( (int32_t)128, tmp101, tmp551);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp552);
CreateIdentity11( (int32_t)128, tmp102, tmp553);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp545,  (int32_t)128, tmp547, tmp549, tmp554);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp554, tmp555);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128,  (int64_t)32, tmp556);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp103, tmp557);
tmp558[ (int64_t)0] =  (int32_t)1;
tmp558[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp555, tmp557, tmp559,  (int64_t)15);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp560);
CreateIdentity11( (int32_t)128, tmp104, tmp561);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp562);
CreateIdentity11( (int32_t)128, tmp105, tmp563);
CreateTensor1( (int32_t)128,  (int64_t)0, tmp564);
CreateIdentity11( (int32_t)128, tmp106, tmp565);
CreateTensor1( (int32_t)128,  (int64_t)32768, tmp566);
CreateIdentity11( (int32_t)128, tmp107, tmp567);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp559,  (int32_t)128, tmp561, tmp563, tmp568);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp568, tmp569);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512,  (int64_t)32, tmp570);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp108, tmp571);
tmp572[ (int64_t)0] =  (int32_t)1;
tmp572[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp569, tmp571, tmp573,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp573, tmp531, tmp574);
CreateIdentity44( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp574, tmp575);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp576);
CreateIdentity11( (int32_t)512, tmp109, tmp577);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp578);
CreateIdentity11( (int32_t)512, tmp110, tmp579);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp580);
CreateIdentity11( (int32_t)512, tmp111, tmp581);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp582);
CreateIdentity11( (int32_t)512, tmp112, tmp583);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp575,  (int32_t)512, tmp577, tmp579, tmp584);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp584, tmp585);
tmp586[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp586[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp586[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp586[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp586[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp586[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp586[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp586[ (int64_t)3][ (int64_t)1] =  (int32_t)0;
Pad442( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp585,  (int32_t)4,  (int32_t)2, tmp586, tmp587);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024,  (int64_t)32, tmp588);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024, tmp113, tmp589);
tmp590[ (int64_t)0] =  (int32_t)1;
tmp590[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp587, tmp589, tmp591,  (int64_t)15);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256,  (int64_t)32, tmp592);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256, tmp114, tmp593);
tmp594[ (int64_t)0] =  (int32_t)1;
tmp594[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp585, tmp593, tmp595,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp596);
CreateIdentity11( (int32_t)256, tmp115, tmp597);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp598);
CreateIdentity11( (int32_t)256, tmp116, tmp599);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp600);
CreateIdentity11( (int32_t)256, tmp117, tmp601);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp602);
CreateIdentity11( (int32_t)256, tmp118, tmp603);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp595,  (int32_t)256, tmp597, tmp599, tmp604);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp604, tmp605);
tmp606[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp606[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp606[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp606[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp606[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp606[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp606[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp606[ (int64_t)3][ (int64_t)1] =  (int32_t)0;
Pad442( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp605,  (int32_t)4,  (int32_t)2, tmp606, tmp607);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256,  (int64_t)32, tmp608);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp119, tmp609);
tmp610[ (int64_t)0] =  (int32_t)1;
tmp610[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp607, tmp609, tmp611,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp612);
CreateIdentity11( (int32_t)256, tmp120, tmp613);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp614);
CreateIdentity11( (int32_t)256, tmp121, tmp615);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp616);
CreateIdentity11( (int32_t)256, tmp122, tmp617);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp618);
CreateIdentity11( (int32_t)256, tmp123, tmp619);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp611,  (int32_t)256, tmp613, tmp615, tmp620);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp620, tmp621);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024,  (int64_t)32, tmp622);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp124, tmp623);
tmp624[ (int64_t)0] =  (int32_t)1;
tmp624[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp621, tmp623, tmp625,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp625, tmp591, tmp626);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp627);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp628);
CreateIdentity11( (int32_t)1024, tmp125, tmp629);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp630);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp631);
CreateIdentity11( (int32_t)1024, tmp126, tmp632);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp633);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp634);
CreateIdentity11( (int32_t)1024, tmp127, tmp635);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp636);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp637);
CreateIdentity11( (int32_t)1024, tmp128, tmp638);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp626,  (int32_t)1024, tmp629, tmp632, tmp639);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp639, tmp640);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256,  (int64_t)32, tmp641);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp129, tmp642);
tmp643[ (int64_t)0] =  (int32_t)1;
tmp643[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp640, tmp642, tmp644,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp645);
CreateIdentity11( (int32_t)256, tmp130, tmp646);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp647);
CreateIdentity11( (int32_t)256, tmp131, tmp648);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp649);
CreateIdentity11( (int32_t)256, tmp132, tmp650);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp651);
CreateIdentity11( (int32_t)256, tmp133, tmp652);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp644,  (int32_t)256, tmp646, tmp648, tmp653);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp653, tmp654);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256,  (int64_t)32, tmp655);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp134, tmp656);
tmp657[ (int64_t)0] =  (int32_t)1;
tmp657[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp654, tmp656, tmp658,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp659);
CreateIdentity11( (int32_t)256, tmp135, tmp660);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp661);
CreateIdentity11( (int32_t)256, tmp136, tmp662);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp663);
CreateIdentity11( (int32_t)256, tmp137, tmp664);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp665);
CreateIdentity11( (int32_t)256, tmp138, tmp666);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp658,  (int32_t)256, tmp660, tmp662, tmp667);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp667, tmp668);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024,  (int64_t)32, tmp669);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp139, tmp670);
tmp671[ (int64_t)0] =  (int32_t)1;
tmp671[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp668, tmp670, tmp672,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp672, tmp626, tmp673);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp674);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp675);
CreateIdentity11( (int32_t)1024, tmp140, tmp676);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp677);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp678);
CreateIdentity11( (int32_t)1024, tmp141, tmp679);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp680);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp681);
CreateIdentity11( (int32_t)1024, tmp142, tmp682);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp683);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp684);
CreateIdentity11( (int32_t)1024, tmp143, tmp685);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp673,  (int32_t)1024, tmp676, tmp679, tmp686);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp686, tmp687);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256,  (int64_t)32, tmp688);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp144, tmp689);
tmp690[ (int64_t)0] =  (int32_t)1;
tmp690[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp687, tmp689, tmp691,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp692);
CreateIdentity11( (int32_t)256, tmp145, tmp693);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp694);
CreateIdentity11( (int32_t)256, tmp146, tmp695);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp696);
CreateIdentity11( (int32_t)256, tmp147, tmp697);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp698);
CreateIdentity11( (int32_t)256, tmp148, tmp699);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp691,  (int32_t)256, tmp693, tmp695, tmp700);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp700, tmp701);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256,  (int64_t)32, tmp702);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp149, tmp703);
tmp704[ (int64_t)0] =  (int32_t)1;
tmp704[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp701, tmp703, tmp705,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp706);
CreateIdentity11( (int32_t)256, tmp150, tmp707);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp708);
CreateIdentity11( (int32_t)256, tmp151, tmp709);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp710);
CreateIdentity11( (int32_t)256, tmp152, tmp711);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp712);
CreateIdentity11( (int32_t)256, tmp153, tmp713);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp705,  (int32_t)256, tmp707, tmp709, tmp714);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp714, tmp715);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024,  (int64_t)32, tmp716);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp154, tmp717);
tmp718[ (int64_t)0] =  (int32_t)1;
tmp718[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp715, tmp717, tmp719,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp719, tmp673, tmp720);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp721);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp722);
CreateIdentity11( (int32_t)1024, tmp155, tmp723);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp724);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp725);
CreateIdentity11( (int32_t)1024, tmp156, tmp726);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp727);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp728);
CreateIdentity11( (int32_t)1024, tmp157, tmp729);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp730);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp731);
CreateIdentity11( (int32_t)1024, tmp158, tmp732);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp720,  (int32_t)1024, tmp723, tmp726, tmp733);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp733, tmp734);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256,  (int64_t)32, tmp735);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp159, tmp736);
tmp737[ (int64_t)0] =  (int32_t)1;
tmp737[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp734, tmp736, tmp738,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp739);
CreateIdentity11( (int32_t)256, tmp160, tmp740);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp741);
CreateIdentity11( (int32_t)256, tmp161, tmp742);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp743);
CreateIdentity11( (int32_t)256, tmp162, tmp744);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp745);
CreateIdentity11( (int32_t)256, tmp163, tmp746);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp738,  (int32_t)256, tmp740, tmp742, tmp747);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp747, tmp748);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256,  (int64_t)32, tmp749);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp164, tmp750);
tmp751[ (int64_t)0] =  (int32_t)1;
tmp751[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp748, tmp750, tmp752,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp753);
CreateIdentity11( (int32_t)256, tmp165, tmp754);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp755);
CreateIdentity11( (int32_t)256, tmp166, tmp756);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp757);
CreateIdentity11( (int32_t)256, tmp167, tmp758);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp759);
CreateIdentity11( (int32_t)256, tmp168, tmp760);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp752,  (int32_t)256, tmp754, tmp756, tmp761);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp761, tmp762);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024,  (int64_t)32, tmp763);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp169, tmp764);
tmp765[ (int64_t)0] =  (int32_t)1;
tmp765[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp762, tmp764, tmp766,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp766, tmp720, tmp767);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp768);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp769);
CreateIdentity11( (int32_t)1024, tmp170, tmp770);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp771);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp772);
CreateIdentity11( (int32_t)1024, tmp171, tmp773);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp774);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp775);
CreateIdentity11( (int32_t)1024, tmp172, tmp776);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp777);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp778);
CreateIdentity11( (int32_t)1024, tmp173, tmp779);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp767,  (int32_t)1024, tmp770, tmp773, tmp780);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp780, tmp781);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256,  (int64_t)32, tmp782);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp174, tmp783);
tmp784[ (int64_t)0] =  (int32_t)1;
tmp784[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp781, tmp783, tmp785,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp786);
CreateIdentity11( (int32_t)256, tmp175, tmp787);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp788);
CreateIdentity11( (int32_t)256, tmp176, tmp789);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp790);
CreateIdentity11( (int32_t)256, tmp177, tmp791);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp792);
CreateIdentity11( (int32_t)256, tmp178, tmp793);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp785,  (int32_t)256, tmp787, tmp789, tmp794);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp794, tmp795);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256,  (int64_t)32, tmp796);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp179, tmp797);
tmp798[ (int64_t)0] =  (int32_t)1;
tmp798[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp795, tmp797, tmp799,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp800);
CreateIdentity11( (int32_t)256, tmp180, tmp801);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp802);
CreateIdentity11( (int32_t)256, tmp181, tmp803);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp804);
CreateIdentity11( (int32_t)256, tmp182, tmp805);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp806);
CreateIdentity11( (int32_t)256, tmp183, tmp807);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp799,  (int32_t)256, tmp801, tmp803, tmp808);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp808, tmp809);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024,  (int64_t)32, tmp810);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp184, tmp811);
tmp812[ (int64_t)0] =  (int32_t)1;
tmp812[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp809, tmp811, tmp813,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp813, tmp767, tmp814);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp815);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp816);
CreateIdentity11( (int32_t)1024, tmp185, tmp817);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp818);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp819);
CreateIdentity11( (int32_t)1024, tmp186, tmp820);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp821);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp822);
CreateIdentity11( (int32_t)1024, tmp187, tmp823);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp824);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp825);
CreateIdentity11( (int32_t)1024, tmp188, tmp826);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp814,  (int32_t)1024, tmp817, tmp820, tmp827);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp827, tmp828);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256,  (int64_t)32, tmp829);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp189, tmp830);
tmp831[ (int64_t)0] =  (int32_t)1;
tmp831[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp828, tmp830, tmp832,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp833);
CreateIdentity11( (int32_t)256, tmp190, tmp834);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp835);
CreateIdentity11( (int32_t)256, tmp191, tmp836);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp837);
CreateIdentity11( (int32_t)256, tmp192, tmp838);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp839);
CreateIdentity11( (int32_t)256, tmp193, tmp840);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp832,  (int32_t)256, tmp834, tmp836, tmp841);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp841, tmp842);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256,  (int64_t)32, tmp843);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp194, tmp844);
tmp845[ (int64_t)0] =  (int32_t)1;
tmp845[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp842, tmp844, tmp846,  (int64_t)15);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp847);
CreateIdentity11( (int32_t)256, tmp195, tmp848);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp849);
CreateIdentity11( (int32_t)256, tmp196, tmp850);
CreateTensor1( (int32_t)256,  (int64_t)0, tmp851);
CreateIdentity11( (int32_t)256, tmp197, tmp852);
CreateTensor1( (int32_t)256,  (int64_t)32768, tmp853);
CreateIdentity11( (int32_t)256, tmp198, tmp854);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp846,  (int32_t)256, tmp848, tmp850, tmp855);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp855, tmp856);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024,  (int64_t)32, tmp857);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp199, tmp858);
tmp859[ (int64_t)0] =  (int32_t)1;
tmp859[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp856, tmp858, tmp860,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp860, tmp814, tmp861);
CreateIdentity44( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp861, tmp862);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp863);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp864);
CreateIdentity11( (int32_t)1024, tmp200, tmp865);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp866);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp867);
CreateIdentity11( (int32_t)1024, tmp201, tmp868);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp869);
CreateTensor1( (int32_t)1024,  (int64_t)0, tmp870);
CreateIdentity11( (int32_t)1024, tmp202, tmp871);
CreateTensor1( (int32_t)1,  (int32_t)1024, tmp872);
CreateTensor1( (int32_t)1024,  (int64_t)32768, tmp873);
CreateIdentity11( (int32_t)1024, tmp203, tmp874);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp862,  (int32_t)1024, tmp865, tmp868, tmp875);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp875, tmp876);
tmp877[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp877[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp877[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp877[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp877[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp877[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp877[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp877[ (int64_t)3][ (int64_t)1] =  (int32_t)0;
Pad442( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp876,  (int32_t)4,  (int32_t)2, tmp877, tmp878);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048,  (int64_t)32, tmp879);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048, tmp204, tmp880);
tmp881[ (int64_t)0] =  (int32_t)1;
tmp881[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp878, tmp880, tmp882,  (int64_t)15);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512,  (int64_t)32, tmp883);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512, tmp205, tmp884);
tmp885[ (int64_t)0] =  (int32_t)1;
tmp885[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp876, tmp884, tmp886,  (int64_t)15);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp887);
CreateIdentity11( (int32_t)512, tmp206, tmp888);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp889);
CreateIdentity11( (int32_t)512, tmp207, tmp890);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp891);
CreateIdentity11( (int32_t)512, tmp208, tmp892);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp893);
CreateIdentity11( (int32_t)512, tmp209, tmp894);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp886,  (int32_t)512, tmp888, tmp890, tmp895);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp895, tmp896);
tmp897[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp897[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp897[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp897[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp897[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp897[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp897[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp897[ (int64_t)3][ (int64_t)1] =  (int32_t)0;
Pad442( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp896,  (int32_t)4,  (int32_t)2, tmp897, tmp898);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512,  (int64_t)32, tmp899);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp210, tmp900);
tmp901[ (int64_t)0] =  (int32_t)1;
tmp901[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp898, tmp900, tmp902,  (int64_t)15);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp903);
CreateIdentity11( (int32_t)512, tmp211, tmp904);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp905);
CreateIdentity11( (int32_t)512, tmp212, tmp906);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp907);
CreateIdentity11( (int32_t)512, tmp213, tmp908);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp909);
CreateIdentity11( (int32_t)512, tmp214, tmp910);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp902,  (int32_t)512, tmp904, tmp906, tmp911);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp911, tmp912);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048,  (int64_t)32, tmp913);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp215, tmp914);
tmp915[ (int64_t)0] =  (int32_t)1;
tmp915[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp912, tmp914, tmp916,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp916, tmp882, tmp917);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp918);
CreateTensor1( (int32_t)2048,  (int64_t)32768, tmp919);
CreateIdentity11( (int32_t)2048, tmp216, tmp920);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp921);
CreateTensor1( (int32_t)2048,  (int64_t)0, tmp922);
CreateIdentity11( (int32_t)2048, tmp217, tmp923);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp924);
CreateTensor1( (int32_t)2048,  (int64_t)0, tmp925);
CreateIdentity11( (int32_t)2048, tmp218, tmp926);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp927);
CreateTensor1( (int32_t)2048,  (int64_t)32768, tmp928);
CreateIdentity11( (int32_t)2048, tmp219, tmp929);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp917,  (int32_t)2048, tmp920, tmp923, tmp930);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp930, tmp931);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512,  (int64_t)32, tmp932);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp220, tmp933);
tmp934[ (int64_t)0] =  (int32_t)1;
tmp934[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp931, tmp933, tmp935,  (int64_t)15);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp936);
CreateIdentity11( (int32_t)512, tmp221, tmp937);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp938);
CreateIdentity11( (int32_t)512, tmp222, tmp939);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp940);
CreateIdentity11( (int32_t)512, tmp223, tmp941);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp942);
CreateIdentity11( (int32_t)512, tmp224, tmp943);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp935,  (int32_t)512, tmp937, tmp939, tmp944);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp944, tmp945);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512,  (int64_t)32, tmp946);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp225, tmp947);
tmp948[ (int64_t)0] =  (int32_t)1;
tmp948[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp945, tmp947, tmp949,  (int64_t)15);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp950);
CreateIdentity11( (int32_t)512, tmp226, tmp951);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp952);
CreateIdentity11( (int32_t)512, tmp227, tmp953);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp954);
CreateIdentity11( (int32_t)512, tmp228, tmp955);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp956);
CreateIdentity11( (int32_t)512, tmp229, tmp957);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp949,  (int32_t)512, tmp951, tmp953, tmp958);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp958, tmp959);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048,  (int64_t)32, tmp960);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp230, tmp961);
tmp962[ (int64_t)0] =  (int32_t)1;
tmp962[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp959, tmp961, tmp963,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp963, tmp917, tmp964);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp965);
CreateTensor1( (int32_t)2048,  (int64_t)32768, tmp966);
CreateIdentity11( (int32_t)2048, tmp231, tmp967);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp968);
CreateTensor1( (int32_t)2048,  (int64_t)0, tmp969);
CreateIdentity11( (int32_t)2048, tmp232, tmp970);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp971);
CreateTensor1( (int32_t)2048,  (int64_t)0, tmp972);
CreateIdentity11( (int32_t)2048, tmp233, tmp973);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp974);
CreateTensor1( (int32_t)2048,  (int64_t)32768, tmp975);
CreateIdentity11( (int32_t)2048, tmp234, tmp976);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp964,  (int32_t)2048, tmp967, tmp970, tmp977);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp977, tmp978);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512,  (int64_t)32, tmp979);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp235, tmp980);
tmp981[ (int64_t)0] =  (int32_t)1;
tmp981[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp978, tmp980, tmp982,  (int64_t)15);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp983);
CreateIdentity11( (int32_t)512, tmp236, tmp984);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp985);
CreateIdentity11( (int32_t)512, tmp237, tmp986);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp987);
CreateIdentity11( (int32_t)512, tmp238, tmp988);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp989);
CreateIdentity11( (int32_t)512, tmp239, tmp990);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp982,  (int32_t)512, tmp984, tmp986, tmp991);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp991, tmp992);
CreateTensor4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512,  (int64_t)32, tmp993);
CreateIdentity44( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp240, tmp994);
tmp995[ (int64_t)0] =  (int32_t)1;
tmp995[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp992, tmp994, tmp996,  (int64_t)15);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp997);
CreateIdentity11( (int32_t)512, tmp241, tmp998);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp999);
CreateIdentity11( (int32_t)512, tmp242, tmp1000);
CreateTensor1( (int32_t)512,  (int64_t)0, tmp1001);
CreateIdentity11( (int32_t)512, tmp243, tmp1002);
CreateTensor1( (int32_t)512,  (int64_t)32768, tmp1003);
CreateIdentity11( (int32_t)512, tmp244, tmp1004);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp996,  (int32_t)512, tmp998, tmp1000, tmp1005);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1005, tmp1006);
CreateTensor4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048,  (int64_t)32, tmp1007);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp245, tmp1008);
tmp1009[ (int64_t)0] =  (int32_t)1;
tmp1009[ (int64_t)1] =  (int32_t)1;
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1006, tmp1008, tmp1010,  (int64_t)15);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1010, tmp964, tmp1011);
CreateIdentity44( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1011, tmp1012);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp1013);
CreateTensor1( (int32_t)2048,  (int64_t)32768, tmp1014);
CreateIdentity11( (int32_t)2048, tmp246, tmp1015);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp1016);
CreateTensor1( (int32_t)2048,  (int64_t)0, tmp1017);
CreateIdentity11( (int32_t)2048, tmp247, tmp1018);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp1019);
CreateTensor1( (int32_t)2048,  (int64_t)0, tmp1020);
CreateIdentity11( (int32_t)2048, tmp248, tmp1021);
CreateTensor1( (int32_t)1,  (int32_t)2048, tmp1022);
CreateTensor1( (int32_t)2048,  (int64_t)32768, tmp1023);
CreateIdentity11( (int32_t)2048, tmp249, tmp1024);
TempFusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1012,  (int32_t)2048, tmp1015, tmp1018, tmp1025);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1025, tmp1026);
AvgPool44Hook( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1026, tmp1027);
CreateIdentity44( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp1027, tmp1028);
Squeeze24( (int32_t)1,  (int32_t)2048,  (int32_t)1,  (int32_t)2,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp1028, tmp1029);
CreateTensor2( (int32_t)2048,  (int32_t)1001,  (int64_t)32, tmp1030);
CreateIdentity22( (int32_t)2048,  (int32_t)1001, tmp250, tmp1031);
CreateTensor1( (int32_t)1,  (int32_t)1001, tmp1032);
CreateTensor1( (int32_t)1001,  (int64_t)0, tmp1033);
CreateIdentity11( (int32_t)1001, tmp251, tmp1034);
MatMulCSF2D( (int32_t)1,  (int32_t)2048,  (int32_t)1001, tmp1029, tmp1031, tmp1035,  (int64_t)15);
MatAddBroadCast2( (int32_t)1,  (int32_t)1001, tmp1035, tmp1034, tmp1036);
CreateIdentity22( (int32_t)1,  (int32_t)1001, tmp1036, tmp1037);
ArgMax1( (int32_t)1,  (int32_t)1,  (int32_t)1001, tmp1037,  (int32_t)1, tmp1038);
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
print_integer((int64_t)funcReconstruct2PCCons(tmp1038[i0], 1));
}
leave_time();
	//cout<<"aes_common rCoutner = "<<aes_common->getRCounter()<<endl;
	 //cout<<"aes_indep rCoutner = "<<aes_indep->getRCounter()<<endl;
	 //cout<<"aes_a_1 rCoutner = "<<aes_a_1->getRCounter()<<endl;
	 //cout<<"aes_a_2 rCoutner = "<<aes_a_2->getRCounter()<<endl;
	 //cout<<"aes_b_1 rCoutner = "<<aes_b_1->getRCounter()<<endl;
	 //cout<<"aes_b_2 rCoutner = "<<aes_b_2->getRCounter()<<endl;
	 //cout<<"aes_c_1 rCoutner = "<<aes_c_1->getRCounter()<<endl;
	 //cout<<"aes_share_conv_bit_shares_p0_p2 rCoutner = "<<aes_share_conv_bit_shares_p0_p2->getRCounter()<<endl;
	 //cout<<"aes_share_conv_bit_shares_p1_p2 rCoutner = "<<aes_share_conv_bit_shares_p1_p2->getRCounter()<<endl;
	 //cout<<"aes_share_conv_shares_mod_odd_p0_p2 rCoutner = "<<aes_share_conv_shares_mod_odd_p0_p2->getRCounter()<<endl;
	 //cout<<"aes_share_conv_shares_mod_odd_p1_p2 rCoutner = "<<aes_share_conv_shares_mod_odd_p1_p2->getRCounter()<<endl;
	 //cout<<"aes_comp_msb_shares_lsb_p0_p2 rCoutner = "<<aes_comp_msb_shares_lsb_p0_p2->getRCounter()<<endl;
	 //cout<<"aes_comp_msb_shares_lsb_p1_p2 rCoutner = "<<aes_comp_msb_shares_lsb_p1_p2->getRCounter()<<endl;
	 //cout<<"aes_comp_msb_shares_bit_vec_p0_p2 rCoutner = "<<aes_comp_msb_shares_bit_vec_p0_p2->getRCounter()<<endl;
	 //cout<<"aes_comp_msb_shares_bit_vec_p1_p2 rCoutner = "<<aes_comp_msb_shares_bit_vec_p1_p2->getRCounter()<<endl;
	 //cout<<"aes_conv_opti_a_1 rCoutner = "<<aes_conv_opti_a_1->getRCounter()<<endl;
	 //cout<<"aes_conv_opti_a_2 rCoutner = "<<aes_conv_opti_a_2->getRCounter()<<endl;
	 //cout<<"aes_conv_opti_b_1 rCoutner = "<<aes_conv_opti_b_1->getRCounter()<<endl;
	 //cout<<"aes_conv_opti_b_2 rCoutner = "<<aes_conv_opti_b_2->getRCounter()<<endl;
	 //cout<<"aes_conv_opti_c_1 rCoutner = "<<aes_conv_opti_c_1->getRCounter()<<endl;

//cout << "----------------------------------" << endl;
//cout << NUM_OF_PARTIES << "PC code, P" << partyNum << endl;
//cout << NUM_ITERATIONS << " iterations, " << whichNetwork << ", batch size " << MINI_BATCH_SIZE << endl;
//cout << "----------------------------------" << endl << endl;


//cout<<"**************RESULTS***************"<<endl;
//flush_output_queue(out_q, role);
//cout<<"************************************"<<endl;

/****************************** CLEAN-UP ******************************/
delete aes_common;
delete aes_indep;
delete aes_a_1;
delete aes_a_2;
delete aes_b_1;
delete aes_b_2;
delete aes_c_1;
delete aes_share_conv_bit_shares_p0_p2;
delete aes_share_conv_bit_shares_p1_p2;
delete aes_share_conv_shares_mod_odd_p0_p2;
delete aes_share_conv_shares_mod_odd_p1_p2;
delete aes_comp_msb_shares_lsb_p0_p2;
delete aes_comp_msb_shares_lsb_p1_p2;
delete aes_comp_msb_shares_bit_vec_p0_p2;
delete aes_comp_msb_shares_bit_vec_p1_p2;
delete aes_conv_opti_a_1;
delete aes_conv_opti_a_2;
delete aes_conv_opti_b_1;
delete aes_conv_opti_b_2;
delete aes_conv_opti_c_1;
delete aes_parallel;
// delete config;
// delete l0;
// delete l1;
// delete l2;
// delete l3;
// delete network;
if (partyNum != PARTY_S)
deleteObjects();

return;

}

#endif
