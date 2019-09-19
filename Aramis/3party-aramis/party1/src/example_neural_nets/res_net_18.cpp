#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "res_net_mem_opti.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef RESNET18

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

void Conv2DReshapeInput(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t RRows, int32_t RCols, auto& inputArr, auto& outputArr){

int32_t linIdxFilterMult =  (int32_t)0;
for (uint32_t n =  (int32_t)0; n < N; n++){

int32_t leftTopCornerH = ( (int32_t)0 - zPadHLeft);

int32_t extremeRightBottomCornerH = ((H -  (int32_t)1) + zPadHRight);
while ((((leftTopCornerH + FH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft);

int32_t extremeRightBottomCornerW = ((W -  (int32_t)1) + zPadWRight);
while ((((leftTopCornerW + FW) -  (int32_t)1) <= extremeRightBottomCornerW)) {
for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
for (uint32_t fw =  (int32_t)0; fw < FW; fw++){

int32_t curPosH = (leftTopCornerH + fh);

int32_t curPosW = (leftTopCornerW + fw);

uint64_t val =  (int64_t)0;
for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
val = ( (int64_t)0);
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

void Conv2DCSFMain(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, int32_t consSF, auto& outArr){

int32_t reshapedFilterRows = CO;

int32_t reshapedFilterCols = ((FH * FW) * CI);

int32_t reshapedIPRows = ((FH * FW) * CI);

int32_t newH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) +  (int32_t)1);

int32_t newW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) +  (int32_t)1);

int32_t reshapedIPCols = ((N * newH) * newW);

auto filterReshaped = make_vector<uint64_t>(reshapedFilterRows, reshapedFilterCols);

auto inputReshaped = make_vector<uint64_t>(reshapedIPRows, reshapedIPCols);

auto matmulOP = make_vector<uint64_t>(reshapedFilterRows, reshapedIPCols);
Conv2DReshapeFilter(FH, FW, CI, CO, filterArr, filterReshaped);
Conv2DReshapeInput(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
MatMulCSF2D(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP, consSF);
Conv2DReshapeMatMulOP(N, newH, newW, CO, matmulOP, outArr);
}

void Conv2DCSF(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, int32_t consSF, auto& outArr)
{
#ifdef CONV_OPTI
	if ((FH>=5) || (FW>=5))
	{
		funcConv2DCSF(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);
	}
	else
	{
		Conv2DCSFMain(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);
	}
#else
	Conv2DCSFMain(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);
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
outArr[i][j][k][l] = ( (int64_t)0);
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

void ClearMemPublic(int32_t x){
return ;
}


void main_aramis(int pnum)
{
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


	print_string(aes_common->keystr);
	print_string(aes_indep->keystr);
	print_string(aes_a_1->keystr);
}
if(run_sequence == 0){
		if (!STANDALONE)
		{
			initializeCommunication("", partyNum);
			synchronize(1000000);	
		}
		print_string("sync ends");
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
//start_m();
if(partyNum == 0){
	role = CLIENT;
}
else if(partyNum == 1){
	role = SERVER;
}
else{
	role = ALL;
}
auto tmp0 = make_vector<uint64_t>( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3);
/* Variable to read the clear value corresponding to the input variable tmp0 at (389,1-389,47) */
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
/* Variable to read the clear value corresponding to the input variable tmp1 at (392,1-392,44) */
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
/* Variable to read the clear value corresponding to the input variable tmp2 at (395,1-395,35) */
uint64_t __tmp_in_tmp2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp2;
}
tmp2[i0] = (role == CLIENT) ? __tmp_in_tmp2 : 0;
}

auto tmp3 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp3 at (398,1-398,35) */
uint64_t __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp3;
}
tmp3[i0] = (role == CLIENT) ? __tmp_in_tmp3 : 0;
}

auto tmp4 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp4 at (401,1-401,35) */
uint64_t __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp4;
}
tmp4[i0] = (role == CLIENT) ? __tmp_in_tmp4 : 0;
}

auto tmp5 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp5 at (404,1-404,35) */
uint64_t __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp5;
}
tmp5[i0] = (role == CLIENT) ? __tmp_in_tmp5 : 0;
}

auto tmp6 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp6 at (407,1-407,45) */
uint64_t __tmp_in_tmp6;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp6;
}
tmp6[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp6 : 0;
}
}
}
}

auto tmp7 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp7 at (410,1-410,45) */
uint64_t __tmp_in_tmp7;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
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
/* Variable to read the clear value corresponding to the input variable tmp8 at (413,1-413,35) */
uint64_t __tmp_in_tmp8;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp8;
}
tmp8[i0] = (role == CLIENT) ? __tmp_in_tmp8 : 0;
}

auto tmp9 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp9 at (416,1-416,35) */
uint64_t __tmp_in_tmp9;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp9;
}
tmp9[i0] = (role == CLIENT) ? __tmp_in_tmp9 : 0;
}

auto tmp10 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp10 at (419,1-419,36) */
uint64_t __tmp_in_tmp10;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp10;
}
tmp10[i0] = (role == CLIENT) ? __tmp_in_tmp10 : 0;
}

auto tmp11 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp11 at (422,1-422,36) */
uint64_t __tmp_in_tmp11;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp11;
}
tmp11[i0] = (role == CLIENT) ? __tmp_in_tmp11 : 0;
}

auto tmp12 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp12 at (425,1-425,46) */
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
/* Variable to read the clear value corresponding to the input variable tmp13 at (428,1-428,36) */
uint64_t __tmp_in_tmp13;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp13;
}
tmp13[i0] = (role == CLIENT) ? __tmp_in_tmp13 : 0;
}

auto tmp14 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp14 at (431,1-431,36) */
uint64_t __tmp_in_tmp14;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp14;
}
tmp14[i0] = (role == CLIENT) ? __tmp_in_tmp14 : 0;
}

auto tmp15 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp15 at (434,1-434,36) */
uint64_t __tmp_in_tmp15;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp15;
}
tmp15[i0] = (role == CLIENT) ? __tmp_in_tmp15 : 0;
}

auto tmp16 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp16 at (437,1-437,36) */
uint64_t __tmp_in_tmp16;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp16;
}
tmp16[i0] = (role == CLIENT) ? __tmp_in_tmp16 : 0;
}

auto tmp17 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp17 at (440,1-440,46) */
uint64_t __tmp_in_tmp17;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp17;
}
tmp17[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp17 : 0;
}
}
}
}

auto tmp18 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp18 at (443,1-443,36) */
uint64_t __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp18;
}
tmp18[i0] = (role == CLIENT) ? __tmp_in_tmp18 : 0;
}

auto tmp19 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp19 at (446,1-446,36) */
uint64_t __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp19;
}
tmp19[i0] = (role == CLIENT) ? __tmp_in_tmp19 : 0;
}

auto tmp20 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp20 at (449,1-449,36) */
uint64_t __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp20;
}
tmp20[i0] = (role == CLIENT) ? __tmp_in_tmp20 : 0;
}

auto tmp21 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp21 at (452,1-452,36) */
uint64_t __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp21;
}
tmp21[i0] = (role == CLIENT) ? __tmp_in_tmp21 : 0;
}

auto tmp22 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp22 at (455,1-455,46) */
uint64_t __tmp_in_tmp22;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
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
/* Variable to read the clear value corresponding to the input variable tmp23 at (458,1-458,36) */
uint64_t __tmp_in_tmp23;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp23;
}
tmp23[i0] = (role == CLIENT) ? __tmp_in_tmp23 : 0;
}

auto tmp24 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp24 at (461,1-461,36) */
uint64_t __tmp_in_tmp24;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp24;
}
tmp24[i0] = (role == CLIENT) ? __tmp_in_tmp24 : 0;
}

auto tmp25 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp25 at (464,1-464,36) */
uint64_t __tmp_in_tmp25;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp25;
}
tmp25[i0] = (role == CLIENT) ? __tmp_in_tmp25 : 0;
}

auto tmp26 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp26 at (467,1-467,36) */
uint64_t __tmp_in_tmp26;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp26;
}
tmp26[i0] = (role == CLIENT) ? __tmp_in_tmp26 : 0;
}

auto tmp27 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp27 at (470,1-470,47) */
uint64_t __tmp_in_tmp27;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp27;
}
tmp27[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp27 : 0;
}
}
}
}

auto tmp28 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp28 at (473,1-473,47) */
uint64_t __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp28;
}
tmp28[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp28 : 0;
}
}
}
}

auto tmp29 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp29 at (476,1-476,37) */
uint64_t __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp29;
}
tmp29[i0] = (role == CLIENT) ? __tmp_in_tmp29 : 0;
}

auto tmp30 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp30 at (479,1-479,37) */
uint64_t __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp30;
}
tmp30[i0] = (role == CLIENT) ? __tmp_in_tmp30 : 0;
}

auto tmp31 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp31 at (482,1-482,37) */
uint64_t __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp31;
}
tmp31[i0] = (role == CLIENT) ? __tmp_in_tmp31 : 0;
}

auto tmp32 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp32 at (485,1-485,37) */
uint64_t __tmp_in_tmp32;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp32;
}
tmp32[i0] = (role == CLIENT) ? __tmp_in_tmp32 : 0;
}

auto tmp33 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp33 at (488,1-488,48) */
uint64_t __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp33;
}
tmp33[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp33 : 0;
}
}
}
}

auto tmp34 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp34 at (491,1-491,37) */
uint64_t __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp34;
}
tmp34[i0] = (role == CLIENT) ? __tmp_in_tmp34 : 0;
}

auto tmp35 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp35 at (494,1-494,37) */
uint64_t __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp35;
}
tmp35[i0] = (role == CLIENT) ? __tmp_in_tmp35 : 0;
}

auto tmp36 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp36 at (497,1-497,37) */
uint64_t __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp36;
}
tmp36[i0] = (role == CLIENT) ? __tmp_in_tmp36 : 0;
}

auto tmp37 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp37 at (500,1-500,37) */
uint64_t __tmp_in_tmp37;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp37;
}
tmp37[i0] = (role == CLIENT) ? __tmp_in_tmp37 : 0;
}

auto tmp38 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp38 at (503,1-503,48) */
uint64_t __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp38;
}
tmp38[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp38 : 0;
}
}
}
}

auto tmp39 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp39 at (506,1-506,37) */
uint64_t __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp39;
}
tmp39[i0] = (role == CLIENT) ? __tmp_in_tmp39 : 0;
}

auto tmp40 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp40 at (509,1-509,37) */
uint64_t __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp40;
}
tmp40[i0] = (role == CLIENT) ? __tmp_in_tmp40 : 0;
}

auto tmp41 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp41 at (512,1-512,37) */
uint64_t __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp41;
}
tmp41[i0] = (role == CLIENT) ? __tmp_in_tmp41 : 0;
}

auto tmp42 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp42 at (515,1-515,37) */
uint64_t __tmp_in_tmp42;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp42;
}
tmp42[i0] = (role == CLIENT) ? __tmp_in_tmp42 : 0;
}

auto tmp43 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp43 at (518,1-518,48) */
uint64_t __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp43;
}
tmp43[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp43 : 0;
}
}
}
}

auto tmp44 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp44 at (521,1-521,37) */
uint64_t __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp44;
}
tmp44[i0] = (role == CLIENT) ? __tmp_in_tmp44 : 0;
}

auto tmp45 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp45 at (524,1-524,37) */
uint64_t __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp45;
}
tmp45[i0] = (role == CLIENT) ? __tmp_in_tmp45 : 0;
}

auto tmp46 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp46 at (527,1-527,37) */
uint64_t __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp46;
}
tmp46[i0] = (role == CLIENT) ? __tmp_in_tmp46 : 0;
}

auto tmp47 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp47 at (530,1-530,37) */
uint64_t __tmp_in_tmp47;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp47;
}
tmp47[i0] = (role == CLIENT) ? __tmp_in_tmp47 : 0;
}

auto tmp48 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp48 at (533,1-533,48) */
uint64_t __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp48;
}
tmp48[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp48 : 0;
}
}
}
}

auto tmp49 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp49 at (536,1-536,48) */
uint64_t __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp49;
}
tmp49[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp49 : 0;
}
}
}
}

auto tmp50 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp50 at (539,1-539,37) */
uint64_t __tmp_in_tmp50;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp50;
}
tmp50[i0] = (role == CLIENT) ? __tmp_in_tmp50 : 0;
}

auto tmp51 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp51 at (542,1-542,37) */
uint64_t __tmp_in_tmp51;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp51;
}
tmp51[i0] = (role == CLIENT) ? __tmp_in_tmp51 : 0;
}

auto tmp52 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp52 at (545,1-545,37) */
uint64_t __tmp_in_tmp52;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp52;
}
tmp52[i0] = (role == CLIENT) ? __tmp_in_tmp52 : 0;
}

auto tmp53 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp53 at (548,1-548,37) */
uint64_t __tmp_in_tmp53;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp53;
}
tmp53[i0] = (role == CLIENT) ? __tmp_in_tmp53 : 0;
}

auto tmp54 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp54 at (551,1-551,48) */
uint64_t __tmp_in_tmp54;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp54;
}
tmp54[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp54 : 0;
}
}
}
}

auto tmp55 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp55 at (554,1-554,37) */
uint64_t __tmp_in_tmp55;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp55;
}
tmp55[i0] = (role == CLIENT) ? __tmp_in_tmp55 : 0;
}

auto tmp56 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp56 at (557,1-557,37) */
uint64_t __tmp_in_tmp56;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp56;
}
tmp56[i0] = (role == CLIENT) ? __tmp_in_tmp56 : 0;
}

auto tmp57 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp57 at (560,1-560,37) */
uint64_t __tmp_in_tmp57;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp57;
}
tmp57[i0] = (role == CLIENT) ? __tmp_in_tmp57 : 0;
}

auto tmp58 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp58 at (563,1-563,37) */
uint64_t __tmp_in_tmp58;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp58;
}
tmp58[i0] = (role == CLIENT) ? __tmp_in_tmp58 : 0;
}

auto tmp59 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp59 at (566,1-566,48) */
uint64_t __tmp_in_tmp59;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp59;
}
tmp59[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp59 : 0;
}
}
}
}

auto tmp60 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp60 at (569,1-569,37) */
uint64_t __tmp_in_tmp60;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp60;
}
tmp60[i0] = (role == CLIENT) ? __tmp_in_tmp60 : 0;
}

auto tmp61 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp61 at (572,1-572,37) */
uint64_t __tmp_in_tmp61;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp61;
}
tmp61[i0] = (role == CLIENT) ? __tmp_in_tmp61 : 0;
}

auto tmp62 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp62 at (575,1-575,37) */
uint64_t __tmp_in_tmp62;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp62;
}
tmp62[i0] = (role == CLIENT) ? __tmp_in_tmp62 : 0;
}

auto tmp63 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp63 at (578,1-578,37) */
uint64_t __tmp_in_tmp63;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp63;
}
tmp63[i0] = (role == CLIENT) ? __tmp_in_tmp63 : 0;
}

auto tmp64 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp64 at (581,1-581,48) */
uint64_t __tmp_in_tmp64;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp64;
}
tmp64[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp64 : 0;
}
}
}
}

auto tmp65 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp65 at (584,1-584,37) */
uint64_t __tmp_in_tmp65;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp65;
}
tmp65[i0] = (role == CLIENT) ? __tmp_in_tmp65 : 0;
}

auto tmp66 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp66 at (587,1-587,37) */
uint64_t __tmp_in_tmp66;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp66;
}
tmp66[i0] = (role == CLIENT) ? __tmp_in_tmp66 : 0;
}

auto tmp67 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp67 at (590,1-590,37) */
uint64_t __tmp_in_tmp67;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp67;
}
tmp67[i0] = (role == CLIENT) ? __tmp_in_tmp67 : 0;
}

auto tmp68 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp68 at (593,1-593,37) */
uint64_t __tmp_in_tmp68;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp68;
}
tmp68[i0] = (role == CLIENT) ? __tmp_in_tmp68 : 0;
}

auto tmp69 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp69 at (596,1-596,48) */
uint64_t __tmp_in_tmp69;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp69;
}
tmp69[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp69 : 0;
}
}
}
}

auto tmp70 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp70 at (599,1-599,48) */
uint64_t __tmp_in_tmp70;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp70;
}
tmp70[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp70 : 0;
}
}
}
}

auto tmp71 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp71 at (602,1-602,37) */
uint64_t __tmp_in_tmp71;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp71;
}
tmp71[i0] = (role == CLIENT) ? __tmp_in_tmp71 : 0;
}

auto tmp72 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp72 at (605,1-605,37) */
uint64_t __tmp_in_tmp72;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp72;
}
tmp72[i0] = (role == CLIENT) ? __tmp_in_tmp72 : 0;
}

auto tmp73 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp73 at (608,1-608,37) */
uint64_t __tmp_in_tmp73;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp73;
}
tmp73[i0] = (role == CLIENT) ? __tmp_in_tmp73 : 0;
}

auto tmp74 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp74 at (611,1-611,37) */
uint64_t __tmp_in_tmp74;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp74;
}
tmp74[i0] = (role == CLIENT) ? __tmp_in_tmp74 : 0;
}

auto tmp75 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp75 at (614,1-614,48) */
uint64_t __tmp_in_tmp75;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp75;
}
tmp75[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp75 : 0;
}
}
}
}

auto tmp76 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp76 at (617,1-617,37) */
uint64_t __tmp_in_tmp76;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp76;
}
tmp76[i0] = (role == CLIENT) ? __tmp_in_tmp76 : 0;
}

auto tmp77 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp77 at (620,1-620,37) */
uint64_t __tmp_in_tmp77;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp77;
}
tmp77[i0] = (role == CLIENT) ? __tmp_in_tmp77 : 0;
}

auto tmp78 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp78 at (623,1-623,37) */
uint64_t __tmp_in_tmp78;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp78;
}
tmp78[i0] = (role == CLIENT) ? __tmp_in_tmp78 : 0;
}

auto tmp79 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp79 at (626,1-626,37) */
uint64_t __tmp_in_tmp79;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp79;
}
tmp79[i0] = (role == CLIENT) ? __tmp_in_tmp79 : 0;
}

auto tmp80 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp80 at (629,1-629,48) */
uint64_t __tmp_in_tmp80;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp80;
}
tmp80[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp80 : 0;
}
}
}
}

auto tmp81 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp81 at (632,1-632,37) */
uint64_t __tmp_in_tmp81;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp81;
}
tmp81[i0] = (role == CLIENT) ? __tmp_in_tmp81 : 0;
}

auto tmp82 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp82 at (635,1-635,37) */
uint64_t __tmp_in_tmp82;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp82;
}
tmp82[i0] = (role == CLIENT) ? __tmp_in_tmp82 : 0;
}

auto tmp83 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp83 at (638,1-638,37) */
uint64_t __tmp_in_tmp83;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp83;
}
tmp83[i0] = (role == CLIENT) ? __tmp_in_tmp83 : 0;
}

auto tmp84 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp84 at (641,1-641,37) */
uint64_t __tmp_in_tmp84;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp84;
}
tmp84[i0] = (role == CLIENT) ? __tmp_in_tmp84 : 0;
}

auto tmp85 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp85 at (644,1-644,48) */
uint64_t __tmp_in_tmp85;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp85;
}
tmp85[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp85 : 0;
}
}
}
}

auto tmp86 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp86 at (647,1-647,37) */
uint64_t __tmp_in_tmp86;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp86;
}
tmp86[i0] = (role == CLIENT) ? __tmp_in_tmp86 : 0;
}

auto tmp87 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp87 at (650,1-650,37) */
uint64_t __tmp_in_tmp87;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp87;
}
tmp87[i0] = (role == CLIENT) ? __tmp_in_tmp87 : 0;
}

auto tmp88 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp88 at (653,1-653,37) */
uint64_t __tmp_in_tmp88;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp88;
}
tmp88[i0] = (role == CLIENT) ? __tmp_in_tmp88 : 0;
}

auto tmp89 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp89 at (656,1-656,37) */
uint64_t __tmp_in_tmp89;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp89;
}
tmp89[i0] = (role == CLIENT) ? __tmp_in_tmp89 : 0;
}

auto tmp90 = make_vector<uint64_t>( (int32_t)512,  (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp90 at (659,1-659,43) */
uint64_t __tmp_in_tmp90;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1001; i1++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp90;
}
tmp90[i0][i1] = (role == CLIENT) ? __tmp_in_tmp90 : 0;
}
}

auto tmp91 = make_vector<uint64_t>( (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp91 at (662,1-662,38) */
uint64_t __tmp_in_tmp91;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1001; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp91;
}
tmp91[i0] = (role == CLIENT) ? __tmp_in_tmp91 : 0;
}

//Main Point

leave_time();
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("Starting main protocol");
start_m();
touch_time();

auto tmp92 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp92[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp92[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp92[ (int64_t)1][ (int64_t)0] =  (int32_t)3;
tmp92[ (int64_t)1][ (int64_t)1] =  (int32_t)3;
tmp92[ (int64_t)2][ (int64_t)0] =  (int32_t)3;
tmp92[ (int64_t)2][ (int64_t)1] =  (int32_t)3;
tmp92[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp92[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp93 = make_vector<uint64_t>( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3);
Pad442( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0,  (int32_t)4,  (int32_t)2, tmp92, tmp93);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp92);
ClearMemSecret4( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0);

auto tmp96 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp93, tmp1,  (int32_t)12, tmp96);
ClearMemSecret4( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3, tmp93);
ClearMemSecret4( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64, tmp1);

auto tmp99 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)1,  (int32_t)0,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp96, tmp99);
ClearMemSecret4( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp96);

auto tmp101 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp99, tmp2, tmp3,  (int32_t)12, tmp101);
ClearMemSecret1( (int32_t)64, tmp3);
ClearMemSecret1( (int32_t)64, tmp2);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp99);

auto tmp105 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp101, tmp105);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp101);

auto tmp107 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp105, tmp6,  (int32_t)12, tmp107);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64, tmp6);

auto tmp109 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp105, tmp7,  (int32_t)12, tmp109);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp7);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp105);

auto tmp112 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp109, tmp8, tmp9,  (int32_t)12, tmp112);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp109);
ClearMemSecret1( (int32_t)64, tmp8);
ClearMemSecret1( (int32_t)64, tmp9);

auto tmp116 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp112, tmp116);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp112);

auto tmp118 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp116, tmp12,  (int32_t)12, tmp118);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp116);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp12);

auto tmp121 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp118, tmp107, tmp121);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp118);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp107);

auto tmp124 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp121, tmp13, tmp14,  (int32_t)12, tmp124);
ClearMemSecret1( (int32_t)64, tmp13);
ClearMemSecret1( (int32_t)64, tmp14);

auto tmp127 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp124, tmp127);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp124);

auto tmp129 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp127, tmp17,  (int32_t)12, tmp129);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp127);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp17);

auto tmp132 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp129, tmp18, tmp19,  (int32_t)12, tmp132);
ClearMemSecret1( (int32_t)64, tmp18);
ClearMemSecret1( (int32_t)64, tmp19);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp129);

auto tmp136 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp132, tmp136);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp132);

auto tmp138 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp136, tmp22,  (int32_t)12, tmp138);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp136);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp22);

auto tmp141 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp138, tmp121, tmp141);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp138);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp121);

auto tmp144 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp141, tmp23, tmp24,  (int32_t)12, tmp144);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp141);
ClearMemSecret1( (int32_t)64, tmp23);
ClearMemSecret1( (int32_t)64, tmp24);

auto tmp148 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp144, tmp148);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp144);

auto tmp150 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp150[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp150[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp150[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp150[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp150[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp150[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp150[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp150[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp151 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Pad442( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp148,  (int32_t)4,  (int32_t)2, tmp150, tmp151);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp150);

auto tmp153 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp151, tmp27,  (int32_t)12, tmp153);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)128, tmp27);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp151);

auto tmp156 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp156[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp156[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp156[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp156[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp156[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp156[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp156[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp156[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp157 = make_vector<uint64_t>( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)64);
Pad442( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)64,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp148,  (int32_t)4,  (int32_t)2, tmp156, tmp157);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp156);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp148);

auto tmp160 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp157, tmp28,  (int32_t)12, tmp160);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)128, tmp28);
ClearMemSecret4( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)64, tmp157);

auto tmp163 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp160, tmp29, tmp30,  (int32_t)12, tmp163);
ClearMemSecret1( (int32_t)128, tmp29);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp160);
ClearMemSecret1( (int32_t)128, tmp30);

auto tmp167 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp163, tmp167);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp163);

auto tmp169 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp167, tmp33,  (int32_t)12, tmp169);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp167);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp33);

auto tmp172 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp169, tmp153, tmp172);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp169);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp153);

auto tmp175 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp172, tmp34, tmp35,  (int32_t)12, tmp175);
ClearMemSecret1( (int32_t)128, tmp34);
ClearMemSecret1( (int32_t)128, tmp35);

auto tmp178 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp175, tmp178);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp175);

auto tmp180 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp178, tmp38,  (int32_t)12, tmp180);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp38);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp178);

auto tmp183 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp180, tmp39, tmp40,  (int32_t)12, tmp183);
ClearMemSecret1( (int32_t)128, tmp39);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp180);
ClearMemSecret1( (int32_t)128, tmp40);

auto tmp187 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp183, tmp187);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp183);

auto tmp189 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp187, tmp43,  (int32_t)12, tmp189);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp187);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp43);

auto tmp192 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp189, tmp172, tmp192);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp189);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp172);

auto tmp195 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp192, tmp44, tmp45,  (int32_t)12, tmp195);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp192);
ClearMemSecret1( (int32_t)128, tmp44);
ClearMemSecret1( (int32_t)128, tmp45);

auto tmp199 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp195, tmp199);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp195);

auto tmp201 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp201[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp201[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp201[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp201[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp201[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp201[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp201[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp201[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp202 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Pad442( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp199,  (int32_t)4,  (int32_t)2, tmp201, tmp202);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp201);

auto tmp204 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp202, tmp48,  (int32_t)12, tmp204);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)256, tmp48);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp202);

auto tmp207 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp207[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp207[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp207[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp207[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp207[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp207[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp207[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp207[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp208 = make_vector<uint64_t>( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)128);
Pad442( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)128,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp199,  (int32_t)4,  (int32_t)2, tmp207, tmp208);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp199);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp207);

auto tmp211 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp208, tmp49,  (int32_t)12, tmp211);
ClearMemSecret4( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)128, tmp208);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)256, tmp49);

auto tmp214 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp211, tmp50, tmp51,  (int32_t)12, tmp214);
ClearMemSecret1( (int32_t)256, tmp51);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp211);
ClearMemSecret1( (int32_t)256, tmp50);

auto tmp218 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp214, tmp218);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp214);

auto tmp220 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp218, tmp54,  (int32_t)12, tmp220);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp54);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp218);

auto tmp223 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp220, tmp204, tmp223);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp220);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp204);

auto tmp226 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp223, tmp55, tmp56,  (int32_t)12, tmp226);
ClearMemSecret1( (int32_t)256, tmp56);
ClearMemSecret1( (int32_t)256, tmp55);

auto tmp229 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp226, tmp229);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp226);

auto tmp231 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp229, tmp59,  (int32_t)12, tmp231);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp229);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp59);

auto tmp234 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp231, tmp60, tmp61,  (int32_t)12, tmp234);
ClearMemSecret1( (int32_t)256, tmp60);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp231);
ClearMemSecret1( (int32_t)256, tmp61);

auto tmp238 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp234, tmp238);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp234);

auto tmp240 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp238, tmp64,  (int32_t)12, tmp240);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp64);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp238);

auto tmp243 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp240, tmp223, tmp243);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp240);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp223);

auto tmp246 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp243, tmp65, tmp66,  (int32_t)12, tmp246);
ClearMemSecret1( (int32_t)256, tmp66);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp243);
ClearMemSecret1( (int32_t)256, tmp65);

auto tmp250 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp246, tmp250);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp246);

auto tmp252 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp252[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp252[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp252[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp252[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp253 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp250,  (int32_t)4,  (int32_t)2, tmp252, tmp253);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp252);

auto tmp255 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp253, tmp69,  (int32_t)12, tmp255);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512, tmp69);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp253);

auto tmp258 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp258[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp258[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp258[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp258[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp258[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp258[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp258[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp258[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp259 = make_vector<uint64_t>( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)256,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp250,  (int32_t)4,  (int32_t)2, tmp258, tmp259);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp258);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp250);

auto tmp262 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp259, tmp70,  (int32_t)12, tmp262);
ClearMemSecret4( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)256, tmp259);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)512, tmp70);

auto tmp265 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp262, tmp71, tmp72,  (int32_t)12, tmp265);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp262);
ClearMemSecret1( (int32_t)512, tmp72);
ClearMemSecret1( (int32_t)512, tmp71);

auto tmp269 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp265, tmp269);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp265);

auto tmp271 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp269, tmp75,  (int32_t)12, tmp271);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp75);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp269);

auto tmp274 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp271, tmp255, tmp274);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp255);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp271);

auto tmp277 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp274, tmp76, tmp77,  (int32_t)12, tmp277);
ClearMemSecret1( (int32_t)512, tmp76);
ClearMemSecret1( (int32_t)512, tmp77);

auto tmp280 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp277, tmp280);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp277);

auto tmp282 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp280, tmp80,  (int32_t)12, tmp282);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp280);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp80);

auto tmp285 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp282, tmp81, tmp82,  (int32_t)12, tmp285);
ClearMemSecret1( (int32_t)512, tmp82);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp282);
ClearMemSecret1( (int32_t)512, tmp81);

auto tmp289 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp285, tmp289);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp285);

auto tmp291 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp289, tmp85,  (int32_t)12, tmp291);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp85);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp289);

auto tmp294 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp291, tmp274, tmp294);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp291);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp274);

auto tmp297 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp294, tmp86, tmp87,  (int32_t)12, tmp297);
ClearMemSecret1( (int32_t)512, tmp87);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp294);
ClearMemSecret1( (int32_t)512, tmp86);

auto tmp301 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp297, tmp301);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp297);

auto tmp303 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)512);
AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp301, tmp303);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp301);

auto tmp305 = make_vector<uint64_t>( (int32_t)1,  (int32_t)512);
Squeeze24( (int32_t)1,  (int32_t)512,  (int32_t)1,  (int32_t)2,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)512, tmp303, tmp305);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)512, tmp303);

auto tmp307 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatMulCSF2D( (int32_t)1,  (int32_t)512,  (int32_t)1001, tmp305, tmp90, tmp307,  (int64_t)12);
ClearMemSecret2( (int32_t)1,  (int32_t)512, tmp305);
ClearMemSecret2( (int32_t)512,  (int32_t)1001, tmp90);

auto tmp310 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatAddBroadCast2( (int32_t)1,  (int32_t)1001, tmp307, tmp91, tmp310);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp307);
ClearMemSecret1( (int32_t)1001, tmp91);

auto tmp313 = make_vector<uint64_t>( (int32_t)1);
ArgMax1( (int32_t)1,  (int32_t)1,  (int32_t)1001, tmp310,  (int32_t)1, tmp313);
ClearMemPublic( (int32_t)1);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp310);
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
print_integer(funcReconstruct2PCCons(tmp313[i0], 1));
}


leave_time();
end_m(whichNetwork);
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
