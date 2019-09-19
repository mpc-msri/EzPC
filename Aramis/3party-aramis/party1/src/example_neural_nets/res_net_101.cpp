#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "res_net_mem_opti.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef RESNET101

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

auto tmp6 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp6 at (407,1-407,46) */
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
/* Variable to read the clear value corresponding to the input variable tmp7 at (410,1-410,45) */
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

auto tmp17 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp17 at (440,1-440,47) */
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
/* Variable to read the clear value corresponding to the input variable tmp18 at (443,1-443,37) */
uint64_t __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp18;
}
tmp18[i0] = (role == CLIENT) ? __tmp_in_tmp18 : 0;
}

auto tmp19 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp19 at (446,1-446,37) */
uint64_t __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp19;
}
tmp19[i0] = (role == CLIENT) ? __tmp_in_tmp19 : 0;
}

auto tmp20 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp20 at (449,1-449,37) */
uint64_t __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp20;
}
tmp20[i0] = (role == CLIENT) ? __tmp_in_tmp20 : 0;
}

auto tmp21 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp21 at (452,1-452,37) */
uint64_t __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp21;
}
tmp21[i0] = (role == CLIENT) ? __tmp_in_tmp21 : 0;
}

auto tmp22 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp22 at (455,1-455,47) */
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

auto tmp27 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp27 at (470,1-470,46) */
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
/* Variable to read the clear value corresponding to the input variable tmp28 at (473,1-473,36) */
uint64_t __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp28;
}
tmp28[i0] = (role == CLIENT) ? __tmp_in_tmp28 : 0;
}

auto tmp29 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp29 at (476,1-476,36) */
uint64_t __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp29;
}
tmp29[i0] = (role == CLIENT) ? __tmp_in_tmp29 : 0;
}

auto tmp30 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp30 at (479,1-479,36) */
uint64_t __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp30;
}
tmp30[i0] = (role == CLIENT) ? __tmp_in_tmp30 : 0;
}

auto tmp31 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp31 at (482,1-482,36) */
uint64_t __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp31;
}
tmp31[i0] = (role == CLIENT) ? __tmp_in_tmp31 : 0;
}

auto tmp32 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp32 at (485,1-485,47) */
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
/* Variable to read the clear value corresponding to the input variable tmp33 at (488,1-488,37) */
uint64_t __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp33;
}
tmp33[i0] = (role == CLIENT) ? __tmp_in_tmp33 : 0;
}

auto tmp34 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp34 at (491,1-491,37) */
uint64_t __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp34;
}
tmp34[i0] = (role == CLIENT) ? __tmp_in_tmp34 : 0;
}

auto tmp35 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp35 at (494,1-494,37) */
uint64_t __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp35;
}
tmp35[i0] = (role == CLIENT) ? __tmp_in_tmp35 : 0;
}

auto tmp36 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp36 at (497,1-497,37) */
uint64_t __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp36;
}
tmp36[i0] = (role == CLIENT) ? __tmp_in_tmp36 : 0;
}

auto tmp37 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp37 at (500,1-500,47) */
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
/* Variable to read the clear value corresponding to the input variable tmp38 at (503,1-503,36) */
uint64_t __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp38;
}
tmp38[i0] = (role == CLIENT) ? __tmp_in_tmp38 : 0;
}

auto tmp39 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp39 at (506,1-506,36) */
uint64_t __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp39;
}
tmp39[i0] = (role == CLIENT) ? __tmp_in_tmp39 : 0;
}

auto tmp40 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp40 at (509,1-509,36) */
uint64_t __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp40;
}
tmp40[i0] = (role == CLIENT) ? __tmp_in_tmp40 : 0;
}

auto tmp41 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp41 at (512,1-512,36) */
uint64_t __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp41;
}
tmp41[i0] = (role == CLIENT) ? __tmp_in_tmp41 : 0;
}

auto tmp42 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp42 at (515,1-515,46) */
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
/* Variable to read the clear value corresponding to the input variable tmp43 at (518,1-518,36) */
uint64_t __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp43;
}
tmp43[i0] = (role == CLIENT) ? __tmp_in_tmp43 : 0;
}

auto tmp44 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp44 at (521,1-521,36) */
uint64_t __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp44;
}
tmp44[i0] = (role == CLIENT) ? __tmp_in_tmp44 : 0;
}

auto tmp45 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp45 at (524,1-524,36) */
uint64_t __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp45;
}
tmp45[i0] = (role == CLIENT) ? __tmp_in_tmp45 : 0;
}

auto tmp46 = make_vector<uint64_t>( (int32_t)64);
/* Variable to read the clear value corresponding to the input variable tmp46 at (527,1-527,36) */
uint64_t __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp46;
}
tmp46[i0] = (role == CLIENT) ? __tmp_in_tmp46 : 0;
}

auto tmp47 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp47 at (530,1-530,47) */
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
/* Variable to read the clear value corresponding to the input variable tmp48 at (533,1-533,37) */
uint64_t __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp48;
}
tmp48[i0] = (role == CLIENT) ? __tmp_in_tmp48 : 0;
}

auto tmp49 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp49 at (536,1-536,37) */
uint64_t __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp49;
}
tmp49[i0] = (role == CLIENT) ? __tmp_in_tmp49 : 0;
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

auto tmp52 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp52 at (545,1-545,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp53 at (548,1-548,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp54 at (551,1-551,37) */
uint64_t __tmp_in_tmp54;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp54;
}
tmp54[i0] = (role == CLIENT) ? __tmp_in_tmp54 : 0;
}

auto tmp55 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp55 at (554,1-554,37) */
uint64_t __tmp_in_tmp55;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp55;
}
tmp55[i0] = (role == CLIENT) ? __tmp_in_tmp55 : 0;
}

auto tmp56 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp56 at (557,1-557,37) */
uint64_t __tmp_in_tmp56;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp56;
}
tmp56[i0] = (role == CLIENT) ? __tmp_in_tmp56 : 0;
}

auto tmp57 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp57 at (560,1-560,37) */
uint64_t __tmp_in_tmp57;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp57;
}
tmp57[i0] = (role == CLIENT) ? __tmp_in_tmp57 : 0;
}

auto tmp58 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp58 at (563,1-563,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp59 at (566,1-566,37) */
uint64_t __tmp_in_tmp59;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp59;
}
tmp59[i0] = (role == CLIENT) ? __tmp_in_tmp59 : 0;
}

auto tmp60 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp60 at (569,1-569,37) */
uint64_t __tmp_in_tmp60;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp60;
}
tmp60[i0] = (role == CLIENT) ? __tmp_in_tmp60 : 0;
}

auto tmp61 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp61 at (572,1-572,37) */
uint64_t __tmp_in_tmp61;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp61;
}
tmp61[i0] = (role == CLIENT) ? __tmp_in_tmp61 : 0;
}

auto tmp62 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp62 at (575,1-575,37) */
uint64_t __tmp_in_tmp62;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp62;
}
tmp62[i0] = (role == CLIENT) ? __tmp_in_tmp62 : 0;
}

auto tmp63 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp63 at (578,1-578,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp64 at (581,1-581,37) */
uint64_t __tmp_in_tmp64;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp64;
}
tmp64[i0] = (role == CLIENT) ? __tmp_in_tmp64 : 0;
}

auto tmp65 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp65 at (584,1-584,37) */
uint64_t __tmp_in_tmp65;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp65;
}
tmp65[i0] = (role == CLIENT) ? __tmp_in_tmp65 : 0;
}

auto tmp66 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp66 at (587,1-587,37) */
uint64_t __tmp_in_tmp66;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp66;
}
tmp66[i0] = (role == CLIENT) ? __tmp_in_tmp66 : 0;
}

auto tmp67 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp67 at (590,1-590,37) */
uint64_t __tmp_in_tmp67;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp67;
}
tmp67[i0] = (role == CLIENT) ? __tmp_in_tmp67 : 0;
}

auto tmp68 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp68 at (593,1-593,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp69 at (596,1-596,37) */
uint64_t __tmp_in_tmp69;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp69;
}
tmp69[i0] = (role == CLIENT) ? __tmp_in_tmp69 : 0;
}

auto tmp70 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp70 at (599,1-599,37) */
uint64_t __tmp_in_tmp70;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp70;
}
tmp70[i0] = (role == CLIENT) ? __tmp_in_tmp70 : 0;
}

auto tmp71 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp71 at (602,1-602,37) */
uint64_t __tmp_in_tmp71;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp71;
}
tmp71[i0] = (role == CLIENT) ? __tmp_in_tmp71 : 0;
}

auto tmp72 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp72 at (605,1-605,37) */
uint64_t __tmp_in_tmp72;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp72;
}
tmp72[i0] = (role == CLIENT) ? __tmp_in_tmp72 : 0;
}

auto tmp73 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp73 at (608,1-608,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp74 at (611,1-611,37) */
uint64_t __tmp_in_tmp74;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp74;
}
tmp74[i0] = (role == CLIENT) ? __tmp_in_tmp74 : 0;
}

auto tmp75 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp75 at (614,1-614,37) */
uint64_t __tmp_in_tmp75;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp75;
}
tmp75[i0] = (role == CLIENT) ? __tmp_in_tmp75 : 0;
}

auto tmp76 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp76 at (617,1-617,37) */
uint64_t __tmp_in_tmp76;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp76;
}
tmp76[i0] = (role == CLIENT) ? __tmp_in_tmp76 : 0;
}

auto tmp77 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp77 at (620,1-620,37) */
uint64_t __tmp_in_tmp77;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp77;
}
tmp77[i0] = (role == CLIENT) ? __tmp_in_tmp77 : 0;
}

auto tmp78 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp78 at (623,1-623,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp79 at (626,1-626,37) */
uint64_t __tmp_in_tmp79;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp79;
}
tmp79[i0] = (role == CLIENT) ? __tmp_in_tmp79 : 0;
}

auto tmp80 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp80 at (629,1-629,37) */
uint64_t __tmp_in_tmp80;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp80;
}
tmp80[i0] = (role == CLIENT) ? __tmp_in_tmp80 : 0;
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

auto tmp83 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp83 at (638,1-638,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp84 at (641,1-641,37) */
uint64_t __tmp_in_tmp84;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp84;
}
tmp84[i0] = (role == CLIENT) ? __tmp_in_tmp84 : 0;
}

auto tmp85 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp85 at (644,1-644,37) */
uint64_t __tmp_in_tmp85;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp85;
}
tmp85[i0] = (role == CLIENT) ? __tmp_in_tmp85 : 0;
}

auto tmp86 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp86 at (647,1-647,37) */
uint64_t __tmp_in_tmp86;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp86;
}
tmp86[i0] = (role == CLIENT) ? __tmp_in_tmp86 : 0;
}

auto tmp87 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp87 at (650,1-650,37) */
uint64_t __tmp_in_tmp87;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp87;
}
tmp87[i0] = (role == CLIENT) ? __tmp_in_tmp87 : 0;
}

auto tmp88 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp88 at (653,1-653,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp89 at (656,1-656,37) */
uint64_t __tmp_in_tmp89;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp89;
}
tmp89[i0] = (role == CLIENT) ? __tmp_in_tmp89 : 0;
}

auto tmp90 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp90 at (659,1-659,37) */
uint64_t __tmp_in_tmp90;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp90;
}
tmp90[i0] = (role == CLIENT) ? __tmp_in_tmp90 : 0;
}

auto tmp91 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp91 at (662,1-662,37) */
uint64_t __tmp_in_tmp91;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp91;
}
tmp91[i0] = (role == CLIENT) ? __tmp_in_tmp91 : 0;
}

auto tmp92 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp92 at (665,1-665,37) */
uint64_t __tmp_in_tmp92;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp92;
}
tmp92[i0] = (role == CLIENT) ? __tmp_in_tmp92 : 0;
}

auto tmp93 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp93 at (668,1-668,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp94 at (671,1-671,37) */
uint64_t __tmp_in_tmp94;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp94;
}
tmp94[i0] = (role == CLIENT) ? __tmp_in_tmp94 : 0;
}

auto tmp95 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp95 at (674,1-674,37) */
uint64_t __tmp_in_tmp95;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp95;
}
tmp95[i0] = (role == CLIENT) ? __tmp_in_tmp95 : 0;
}

auto tmp96 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp96 at (677,1-677,37) */
uint64_t __tmp_in_tmp96;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp96;
}
tmp96[i0] = (role == CLIENT) ? __tmp_in_tmp96 : 0;
}

auto tmp97 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp97 at (680,1-680,37) */
uint64_t __tmp_in_tmp97;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp97;
}
tmp97[i0] = (role == CLIENT) ? __tmp_in_tmp97 : 0;
}

auto tmp98 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp98 at (683,1-683,48) */
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
/* Variable to read the clear value corresponding to the input variable tmp99 at (686,1-686,37) */
uint64_t __tmp_in_tmp99;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp99;
}
tmp99[i0] = (role == CLIENT) ? __tmp_in_tmp99 : 0;
}

auto tmp100 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp100 at (689,1-689,38) */
uint64_t __tmp_in_tmp100;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp100;
}
tmp100[i0] = (role == CLIENT) ? __tmp_in_tmp100 : 0;
}

auto tmp101 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp101 at (692,1-692,38) */
uint64_t __tmp_in_tmp101;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp101;
}
tmp101[i0] = (role == CLIENT) ? __tmp_in_tmp101 : 0;
}

auto tmp102 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp102 at (695,1-695,38) */
uint64_t __tmp_in_tmp102;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp102;
}
tmp102[i0] = (role == CLIENT) ? __tmp_in_tmp102 : 0;
}

auto tmp103 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp103 at (698,1-698,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp104 at (701,1-701,38) */
uint64_t __tmp_in_tmp104;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp104;
}
tmp104[i0] = (role == CLIENT) ? __tmp_in_tmp104 : 0;
}

auto tmp105 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp105 at (704,1-704,38) */
uint64_t __tmp_in_tmp105;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp105;
}
tmp105[i0] = (role == CLIENT) ? __tmp_in_tmp105 : 0;
}

auto tmp106 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp106 at (707,1-707,38) */
uint64_t __tmp_in_tmp106;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp106;
}
tmp106[i0] = (role == CLIENT) ? __tmp_in_tmp106 : 0;
}

auto tmp107 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp107 at (710,1-710,38) */
uint64_t __tmp_in_tmp107;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp107;
}
tmp107[i0] = (role == CLIENT) ? __tmp_in_tmp107 : 0;
}

auto tmp108 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp108 at (713,1-713,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp109 at (716,1-716,38) */
uint64_t __tmp_in_tmp109;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp109;
}
tmp109[i0] = (role == CLIENT) ? __tmp_in_tmp109 : 0;
}

auto tmp110 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp110 at (719,1-719,38) */
uint64_t __tmp_in_tmp110;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp110;
}
tmp110[i0] = (role == CLIENT) ? __tmp_in_tmp110 : 0;
}

auto tmp111 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp111 at (722,1-722,38) */
uint64_t __tmp_in_tmp111;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp111;
}
tmp111[i0] = (role == CLIENT) ? __tmp_in_tmp111 : 0;
}

auto tmp112 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp112 at (725,1-725,38) */
uint64_t __tmp_in_tmp112;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp112;
}
tmp112[i0] = (role == CLIENT) ? __tmp_in_tmp112 : 0;
}

auto tmp113 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp113 at (728,1-728,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp114 at (731,1-731,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp115 at (734,1-734,38) */
uint64_t __tmp_in_tmp115;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp115;
}
tmp115[i0] = (role == CLIENT) ? __tmp_in_tmp115 : 0;
}

auto tmp116 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp116 at (737,1-737,38) */
uint64_t __tmp_in_tmp116;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp116;
}
tmp116[i0] = (role == CLIENT) ? __tmp_in_tmp116 : 0;
}

auto tmp117 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp117 at (740,1-740,38) */
uint64_t __tmp_in_tmp117;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp117;
}
tmp117[i0] = (role == CLIENT) ? __tmp_in_tmp117 : 0;
}

auto tmp118 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp118 at (743,1-743,38) */
uint64_t __tmp_in_tmp118;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp118;
}
tmp118[i0] = (role == CLIENT) ? __tmp_in_tmp118 : 0;
}

auto tmp119 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp119 at (746,1-746,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp120 at (749,1-749,38) */
uint64_t __tmp_in_tmp120;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp120;
}
tmp120[i0] = (role == CLIENT) ? __tmp_in_tmp120 : 0;
}

auto tmp121 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp121 at (752,1-752,38) */
uint64_t __tmp_in_tmp121;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp121;
}
tmp121[i0] = (role == CLIENT) ? __tmp_in_tmp121 : 0;
}

auto tmp122 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp122 at (755,1-755,38) */
uint64_t __tmp_in_tmp122;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp122;
}
tmp122[i0] = (role == CLIENT) ? __tmp_in_tmp122 : 0;
}

auto tmp123 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp123 at (758,1-758,38) */
uint64_t __tmp_in_tmp123;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp123;
}
tmp123[i0] = (role == CLIENT) ? __tmp_in_tmp123 : 0;
}

auto tmp124 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp124 at (761,1-761,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp125 at (764,1-764,39) */
uint64_t __tmp_in_tmp125;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp125;
}
tmp125[i0] = (role == CLIENT) ? __tmp_in_tmp125 : 0;
}

auto tmp126 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp126 at (767,1-767,39) */
uint64_t __tmp_in_tmp126;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp126;
}
tmp126[i0] = (role == CLIENT) ? __tmp_in_tmp126 : 0;
}

auto tmp127 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp127 at (770,1-770,39) */
uint64_t __tmp_in_tmp127;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp127;
}
tmp127[i0] = (role == CLIENT) ? __tmp_in_tmp127 : 0;
}

auto tmp128 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp128 at (773,1-773,39) */
uint64_t __tmp_in_tmp128;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp128;
}
tmp128[i0] = (role == CLIENT) ? __tmp_in_tmp128 : 0;
}

auto tmp129 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp129 at (776,1-776,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp130 at (779,1-779,38) */
uint64_t __tmp_in_tmp130;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp130;
}
tmp130[i0] = (role == CLIENT) ? __tmp_in_tmp130 : 0;
}

auto tmp131 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp131 at (782,1-782,38) */
uint64_t __tmp_in_tmp131;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp131;
}
tmp131[i0] = (role == CLIENT) ? __tmp_in_tmp131 : 0;
}

auto tmp132 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp132 at (785,1-785,38) */
uint64_t __tmp_in_tmp132;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp132;
}
tmp132[i0] = (role == CLIENT) ? __tmp_in_tmp132 : 0;
}

auto tmp133 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp133 at (788,1-788,38) */
uint64_t __tmp_in_tmp133;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp133;
}
tmp133[i0] = (role == CLIENT) ? __tmp_in_tmp133 : 0;
}

auto tmp134 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp134 at (791,1-791,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp135 at (794,1-794,38) */
uint64_t __tmp_in_tmp135;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp135;
}
tmp135[i0] = (role == CLIENT) ? __tmp_in_tmp135 : 0;
}

auto tmp136 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp136 at (797,1-797,38) */
uint64_t __tmp_in_tmp136;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp136;
}
tmp136[i0] = (role == CLIENT) ? __tmp_in_tmp136 : 0;
}

auto tmp137 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp137 at (800,1-800,38) */
uint64_t __tmp_in_tmp137;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp137;
}
tmp137[i0] = (role == CLIENT) ? __tmp_in_tmp137 : 0;
}

auto tmp138 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp138 at (803,1-803,38) */
uint64_t __tmp_in_tmp138;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp138;
}
tmp138[i0] = (role == CLIENT) ? __tmp_in_tmp138 : 0;
}

auto tmp139 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp139 at (806,1-806,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp140 at (809,1-809,39) */
uint64_t __tmp_in_tmp140;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp140;
}
tmp140[i0] = (role == CLIENT) ? __tmp_in_tmp140 : 0;
}

auto tmp141 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp141 at (812,1-812,39) */
uint64_t __tmp_in_tmp141;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp141;
}
tmp141[i0] = (role == CLIENT) ? __tmp_in_tmp141 : 0;
}

auto tmp142 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp142 at (815,1-815,39) */
uint64_t __tmp_in_tmp142;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp142;
}
tmp142[i0] = (role == CLIENT) ? __tmp_in_tmp142 : 0;
}

auto tmp143 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp143 at (818,1-818,39) */
uint64_t __tmp_in_tmp143;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp143;
}
tmp143[i0] = (role == CLIENT) ? __tmp_in_tmp143 : 0;
}

auto tmp144 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp144 at (821,1-821,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp145 at (824,1-824,38) */
uint64_t __tmp_in_tmp145;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp145;
}
tmp145[i0] = (role == CLIENT) ? __tmp_in_tmp145 : 0;
}

auto tmp146 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp146 at (827,1-827,38) */
uint64_t __tmp_in_tmp146;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp146;
}
tmp146[i0] = (role == CLIENT) ? __tmp_in_tmp146 : 0;
}

auto tmp147 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp147 at (830,1-830,38) */
uint64_t __tmp_in_tmp147;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp147;
}
tmp147[i0] = (role == CLIENT) ? __tmp_in_tmp147 : 0;
}

auto tmp148 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp148 at (833,1-833,38) */
uint64_t __tmp_in_tmp148;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp148;
}
tmp148[i0] = (role == CLIENT) ? __tmp_in_tmp148 : 0;
}

auto tmp149 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp149 at (836,1-836,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp150 at (839,1-839,38) */
uint64_t __tmp_in_tmp150;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp150;
}
tmp150[i0] = (role == CLIENT) ? __tmp_in_tmp150 : 0;
}

auto tmp151 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp151 at (842,1-842,38) */
uint64_t __tmp_in_tmp151;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp151;
}
tmp151[i0] = (role == CLIENT) ? __tmp_in_tmp151 : 0;
}

auto tmp152 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp152 at (845,1-845,38) */
uint64_t __tmp_in_tmp152;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp152;
}
tmp152[i0] = (role == CLIENT) ? __tmp_in_tmp152 : 0;
}

auto tmp153 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp153 at (848,1-848,38) */
uint64_t __tmp_in_tmp153;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp153;
}
tmp153[i0] = (role == CLIENT) ? __tmp_in_tmp153 : 0;
}

auto tmp154 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp154 at (851,1-851,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp155 at (854,1-854,39) */
uint64_t __tmp_in_tmp155;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp155;
}
tmp155[i0] = (role == CLIENT) ? __tmp_in_tmp155 : 0;
}

auto tmp156 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp156 at (857,1-857,39) */
uint64_t __tmp_in_tmp156;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp156;
}
tmp156[i0] = (role == CLIENT) ? __tmp_in_tmp156 : 0;
}

auto tmp157 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp157 at (860,1-860,39) */
uint64_t __tmp_in_tmp157;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp157;
}
tmp157[i0] = (role == CLIENT) ? __tmp_in_tmp157 : 0;
}

auto tmp158 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp158 at (863,1-863,39) */
uint64_t __tmp_in_tmp158;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp158;
}
tmp158[i0] = (role == CLIENT) ? __tmp_in_tmp158 : 0;
}

auto tmp159 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp159 at (866,1-866,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp160 at (869,1-869,38) */
uint64_t __tmp_in_tmp160;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp160;
}
tmp160[i0] = (role == CLIENT) ? __tmp_in_tmp160 : 0;
}

auto tmp161 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp161 at (872,1-872,38) */
uint64_t __tmp_in_tmp161;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp161;
}
tmp161[i0] = (role == CLIENT) ? __tmp_in_tmp161 : 0;
}

auto tmp162 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp162 at (875,1-875,38) */
uint64_t __tmp_in_tmp162;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp162;
}
tmp162[i0] = (role == CLIENT) ? __tmp_in_tmp162 : 0;
}

auto tmp163 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp163 at (878,1-878,38) */
uint64_t __tmp_in_tmp163;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp163;
}
tmp163[i0] = (role == CLIENT) ? __tmp_in_tmp163 : 0;
}

auto tmp164 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp164 at (881,1-881,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp165 at (884,1-884,38) */
uint64_t __tmp_in_tmp165;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp165;
}
tmp165[i0] = (role == CLIENT) ? __tmp_in_tmp165 : 0;
}

auto tmp166 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp166 at (887,1-887,38) */
uint64_t __tmp_in_tmp166;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp166;
}
tmp166[i0] = (role == CLIENT) ? __tmp_in_tmp166 : 0;
}

auto tmp167 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp167 at (890,1-890,38) */
uint64_t __tmp_in_tmp167;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp167;
}
tmp167[i0] = (role == CLIENT) ? __tmp_in_tmp167 : 0;
}

auto tmp168 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp168 at (893,1-893,38) */
uint64_t __tmp_in_tmp168;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp168;
}
tmp168[i0] = (role == CLIENT) ? __tmp_in_tmp168 : 0;
}

auto tmp169 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp169 at (896,1-896,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp170 at (899,1-899,39) */
uint64_t __tmp_in_tmp170;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp170;
}
tmp170[i0] = (role == CLIENT) ? __tmp_in_tmp170 : 0;
}

auto tmp171 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp171 at (902,1-902,39) */
uint64_t __tmp_in_tmp171;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp171;
}
tmp171[i0] = (role == CLIENT) ? __tmp_in_tmp171 : 0;
}

auto tmp172 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp172 at (905,1-905,39) */
uint64_t __tmp_in_tmp172;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp172;
}
tmp172[i0] = (role == CLIENT) ? __tmp_in_tmp172 : 0;
}

auto tmp173 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp173 at (908,1-908,39) */
uint64_t __tmp_in_tmp173;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp173;
}
tmp173[i0] = (role == CLIENT) ? __tmp_in_tmp173 : 0;
}

auto tmp174 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp174 at (911,1-911,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp175 at (914,1-914,38) */
uint64_t __tmp_in_tmp175;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp175;
}
tmp175[i0] = (role == CLIENT) ? __tmp_in_tmp175 : 0;
}

auto tmp176 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp176 at (917,1-917,38) */
uint64_t __tmp_in_tmp176;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp176;
}
tmp176[i0] = (role == CLIENT) ? __tmp_in_tmp176 : 0;
}

auto tmp177 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp177 at (920,1-920,38) */
uint64_t __tmp_in_tmp177;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp177;
}
tmp177[i0] = (role == CLIENT) ? __tmp_in_tmp177 : 0;
}

auto tmp178 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp178 at (923,1-923,38) */
uint64_t __tmp_in_tmp178;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp178;
}
tmp178[i0] = (role == CLIENT) ? __tmp_in_tmp178 : 0;
}

auto tmp179 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp179 at (926,1-926,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp180 at (929,1-929,38) */
uint64_t __tmp_in_tmp180;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp180;
}
tmp180[i0] = (role == CLIENT) ? __tmp_in_tmp180 : 0;
}

auto tmp181 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp181 at (932,1-932,38) */
uint64_t __tmp_in_tmp181;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp181;
}
tmp181[i0] = (role == CLIENT) ? __tmp_in_tmp181 : 0;
}

auto tmp182 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp182 at (935,1-935,38) */
uint64_t __tmp_in_tmp182;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp182;
}
tmp182[i0] = (role == CLIENT) ? __tmp_in_tmp182 : 0;
}

auto tmp183 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp183 at (938,1-938,38) */
uint64_t __tmp_in_tmp183;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp183;
}
tmp183[i0] = (role == CLIENT) ? __tmp_in_tmp183 : 0;
}

auto tmp184 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp184 at (941,1-941,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp185 at (944,1-944,39) */
uint64_t __tmp_in_tmp185;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp185;
}
tmp185[i0] = (role == CLIENT) ? __tmp_in_tmp185 : 0;
}

auto tmp186 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp186 at (947,1-947,39) */
uint64_t __tmp_in_tmp186;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp186;
}
tmp186[i0] = (role == CLIENT) ? __tmp_in_tmp186 : 0;
}

auto tmp187 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp187 at (950,1-950,39) */
uint64_t __tmp_in_tmp187;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp187;
}
tmp187[i0] = (role == CLIENT) ? __tmp_in_tmp187 : 0;
}

auto tmp188 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp188 at (953,1-953,39) */
uint64_t __tmp_in_tmp188;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp188;
}
tmp188[i0] = (role == CLIENT) ? __tmp_in_tmp188 : 0;
}

auto tmp189 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp189 at (956,1-956,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp190 at (959,1-959,38) */
uint64_t __tmp_in_tmp190;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp190;
}
tmp190[i0] = (role == CLIENT) ? __tmp_in_tmp190 : 0;
}

auto tmp191 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp191 at (962,1-962,38) */
uint64_t __tmp_in_tmp191;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp191;
}
tmp191[i0] = (role == CLIENT) ? __tmp_in_tmp191 : 0;
}

auto tmp192 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp192 at (965,1-965,38) */
uint64_t __tmp_in_tmp192;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp192;
}
tmp192[i0] = (role == CLIENT) ? __tmp_in_tmp192 : 0;
}

auto tmp193 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp193 at (968,1-968,38) */
uint64_t __tmp_in_tmp193;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp193;
}
tmp193[i0] = (role == CLIENT) ? __tmp_in_tmp193 : 0;
}

auto tmp194 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp194 at (971,1-971,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp195 at (974,1-974,38) */
uint64_t __tmp_in_tmp195;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp195;
}
tmp195[i0] = (role == CLIENT) ? __tmp_in_tmp195 : 0;
}

auto tmp196 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp196 at (977,1-977,38) */
uint64_t __tmp_in_tmp196;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp196;
}
tmp196[i0] = (role == CLIENT) ? __tmp_in_tmp196 : 0;
}

auto tmp197 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp197 at (980,1-980,38) */
uint64_t __tmp_in_tmp197;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp197;
}
tmp197[i0] = (role == CLIENT) ? __tmp_in_tmp197 : 0;
}

auto tmp198 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp198 at (983,1-983,38) */
uint64_t __tmp_in_tmp198;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp198;
}
tmp198[i0] = (role == CLIENT) ? __tmp_in_tmp198 : 0;
}

auto tmp199 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp199 at (986,1-986,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp200 at (989,1-989,39) */
uint64_t __tmp_in_tmp200;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp200;
}
tmp200[i0] = (role == CLIENT) ? __tmp_in_tmp200 : 0;
}

auto tmp201 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp201 at (992,1-992,39) */
uint64_t __tmp_in_tmp201;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp201;
}
tmp201[i0] = (role == CLIENT) ? __tmp_in_tmp201 : 0;
}

auto tmp202 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp202 at (995,1-995,39) */
uint64_t __tmp_in_tmp202;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp202;
}
tmp202[i0] = (role == CLIENT) ? __tmp_in_tmp202 : 0;
}

auto tmp203 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp203 at (998,1-998,39) */
uint64_t __tmp_in_tmp203;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp203;
}
tmp203[i0] = (role == CLIENT) ? __tmp_in_tmp203 : 0;
}

auto tmp204 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp204 at (1001,1-1001,50) */
uint64_t __tmp_in_tmp204;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp204;
}
tmp204[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp204 : 0;
}
}
}
}

auto tmp205 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp205 at (1004,1-1004,38) */
uint64_t __tmp_in_tmp205;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp205;
}
tmp205[i0] = (role == CLIENT) ? __tmp_in_tmp205 : 0;
}

auto tmp206 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp206 at (1007,1-1007,38) */
uint64_t __tmp_in_tmp206;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp206;
}
tmp206[i0] = (role == CLIENT) ? __tmp_in_tmp206 : 0;
}

auto tmp207 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp207 at (1010,1-1010,38) */
uint64_t __tmp_in_tmp207;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp207;
}
tmp207[i0] = (role == CLIENT) ? __tmp_in_tmp207 : 0;
}

auto tmp208 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp208 at (1013,1-1013,38) */
uint64_t __tmp_in_tmp208;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp208;
}
tmp208[i0] = (role == CLIENT) ? __tmp_in_tmp208 : 0;
}

auto tmp209 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp209 at (1016,1-1016,49) */
uint64_t __tmp_in_tmp209;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp209;
}
tmp209[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp209 : 0;
}
}
}
}

auto tmp210 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp210 at (1019,1-1019,38) */
uint64_t __tmp_in_tmp210;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp210;
}
tmp210[i0] = (role == CLIENT) ? __tmp_in_tmp210 : 0;
}

auto tmp211 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp211 at (1022,1-1022,38) */
uint64_t __tmp_in_tmp211;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp211;
}
tmp211[i0] = (role == CLIENT) ? __tmp_in_tmp211 : 0;
}

auto tmp212 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp212 at (1025,1-1025,38) */
uint64_t __tmp_in_tmp212;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp212;
}
tmp212[i0] = (role == CLIENT) ? __tmp_in_tmp212 : 0;
}

auto tmp213 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp213 at (1028,1-1028,38) */
uint64_t __tmp_in_tmp213;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp213;
}
tmp213[i0] = (role == CLIENT) ? __tmp_in_tmp213 : 0;
}

auto tmp214 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp214 at (1031,1-1031,50) */
uint64_t __tmp_in_tmp214;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp214;
}
tmp214[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp214 : 0;
}
}
}
}

auto tmp215 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp215 at (1034,1-1034,39) */
uint64_t __tmp_in_tmp215;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp215;
}
tmp215[i0] = (role == CLIENT) ? __tmp_in_tmp215 : 0;
}

auto tmp216 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp216 at (1037,1-1037,39) */
uint64_t __tmp_in_tmp216;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp216;
}
tmp216[i0] = (role == CLIENT) ? __tmp_in_tmp216 : 0;
}

auto tmp217 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp217 at (1040,1-1040,39) */
uint64_t __tmp_in_tmp217;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp217;
}
tmp217[i0] = (role == CLIENT) ? __tmp_in_tmp217 : 0;
}

auto tmp218 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp218 at (1043,1-1043,39) */
uint64_t __tmp_in_tmp218;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp218;
}
tmp218[i0] = (role == CLIENT) ? __tmp_in_tmp218 : 0;
}

auto tmp219 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp219 at (1046,1-1046,50) */
uint64_t __tmp_in_tmp219;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp219;
}
tmp219[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp219 : 0;
}
}
}
}

auto tmp220 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp220 at (1049,1-1049,38) */
uint64_t __tmp_in_tmp220;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp220;
}
tmp220[i0] = (role == CLIENT) ? __tmp_in_tmp220 : 0;
}

auto tmp221 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp221 at (1052,1-1052,38) */
uint64_t __tmp_in_tmp221;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp221;
}
tmp221[i0] = (role == CLIENT) ? __tmp_in_tmp221 : 0;
}

auto tmp222 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp222 at (1055,1-1055,38) */
uint64_t __tmp_in_tmp222;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp222;
}
tmp222[i0] = (role == CLIENT) ? __tmp_in_tmp222 : 0;
}

auto tmp223 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp223 at (1058,1-1058,38) */
uint64_t __tmp_in_tmp223;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp223;
}
tmp223[i0] = (role == CLIENT) ? __tmp_in_tmp223 : 0;
}

auto tmp224 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp224 at (1061,1-1061,49) */
uint64_t __tmp_in_tmp224;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp224;
}
tmp224[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp224 : 0;
}
}
}
}

auto tmp225 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp225 at (1064,1-1064,38) */
uint64_t __tmp_in_tmp225;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp225;
}
tmp225[i0] = (role == CLIENT) ? __tmp_in_tmp225 : 0;
}

auto tmp226 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp226 at (1067,1-1067,38) */
uint64_t __tmp_in_tmp226;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp226;
}
tmp226[i0] = (role == CLIENT) ? __tmp_in_tmp226 : 0;
}

auto tmp227 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp227 at (1070,1-1070,38) */
uint64_t __tmp_in_tmp227;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp227;
}
tmp227[i0] = (role == CLIENT) ? __tmp_in_tmp227 : 0;
}

auto tmp228 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp228 at (1073,1-1073,38) */
uint64_t __tmp_in_tmp228;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp228;
}
tmp228[i0] = (role == CLIENT) ? __tmp_in_tmp228 : 0;
}

auto tmp229 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp229 at (1076,1-1076,50) */
uint64_t __tmp_in_tmp229;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp229;
}
tmp229[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp229 : 0;
}
}
}
}

auto tmp230 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp230 at (1079,1-1079,39) */
uint64_t __tmp_in_tmp230;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp230;
}
tmp230[i0] = (role == CLIENT) ? __tmp_in_tmp230 : 0;
}

auto tmp231 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp231 at (1082,1-1082,39) */
uint64_t __tmp_in_tmp231;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp231;
}
tmp231[i0] = (role == CLIENT) ? __tmp_in_tmp231 : 0;
}

auto tmp232 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp232 at (1085,1-1085,39) */
uint64_t __tmp_in_tmp232;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp232;
}
tmp232[i0] = (role == CLIENT) ? __tmp_in_tmp232 : 0;
}

auto tmp233 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp233 at (1088,1-1088,39) */
uint64_t __tmp_in_tmp233;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp233;
}
tmp233[i0] = (role == CLIENT) ? __tmp_in_tmp233 : 0;
}

auto tmp234 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp234 at (1091,1-1091,50) */
uint64_t __tmp_in_tmp234;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp234;
}
tmp234[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp234 : 0;
}
}
}
}

auto tmp235 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp235 at (1094,1-1094,38) */
uint64_t __tmp_in_tmp235;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp235;
}
tmp235[i0] = (role == CLIENT) ? __tmp_in_tmp235 : 0;
}

auto tmp236 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp236 at (1097,1-1097,38) */
uint64_t __tmp_in_tmp236;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp236;
}
tmp236[i0] = (role == CLIENT) ? __tmp_in_tmp236 : 0;
}

auto tmp237 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp237 at (1100,1-1100,38) */
uint64_t __tmp_in_tmp237;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp237;
}
tmp237[i0] = (role == CLIENT) ? __tmp_in_tmp237 : 0;
}

auto tmp238 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp238 at (1103,1-1103,38) */
uint64_t __tmp_in_tmp238;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp238;
}
tmp238[i0] = (role == CLIENT) ? __tmp_in_tmp238 : 0;
}

auto tmp239 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp239 at (1106,1-1106,49) */
uint64_t __tmp_in_tmp239;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp239;
}
tmp239[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp239 : 0;
}
}
}
}

auto tmp240 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp240 at (1109,1-1109,38) */
uint64_t __tmp_in_tmp240;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp240;
}
tmp240[i0] = (role == CLIENT) ? __tmp_in_tmp240 : 0;
}

auto tmp241 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp241 at (1112,1-1112,38) */
uint64_t __tmp_in_tmp241;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp241;
}
tmp241[i0] = (role == CLIENT) ? __tmp_in_tmp241 : 0;
}

auto tmp242 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp242 at (1115,1-1115,38) */
uint64_t __tmp_in_tmp242;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp242;
}
tmp242[i0] = (role == CLIENT) ? __tmp_in_tmp242 : 0;
}

auto tmp243 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp243 at (1118,1-1118,38) */
uint64_t __tmp_in_tmp243;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp243;
}
tmp243[i0] = (role == CLIENT) ? __tmp_in_tmp243 : 0;
}

auto tmp244 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp244 at (1121,1-1121,50) */
uint64_t __tmp_in_tmp244;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp244;
}
tmp244[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp244 : 0;
}
}
}
}

auto tmp245 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp245 at (1124,1-1124,39) */
uint64_t __tmp_in_tmp245;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp245;
}
tmp245[i0] = (role == CLIENT) ? __tmp_in_tmp245 : 0;
}

auto tmp246 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp246 at (1127,1-1127,39) */
uint64_t __tmp_in_tmp246;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp246;
}
tmp246[i0] = (role == CLIENT) ? __tmp_in_tmp246 : 0;
}

auto tmp247 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp247 at (1130,1-1130,39) */
uint64_t __tmp_in_tmp247;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp247;
}
tmp247[i0] = (role == CLIENT) ? __tmp_in_tmp247 : 0;
}

auto tmp248 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp248 at (1133,1-1133,39) */
uint64_t __tmp_in_tmp248;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp248;
}
tmp248[i0] = (role == CLIENT) ? __tmp_in_tmp248 : 0;
}

auto tmp249 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp249 at (1136,1-1136,50) */
uint64_t __tmp_in_tmp249;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp249;
}
tmp249[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp249 : 0;
}
}
}
}

auto tmp250 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp250 at (1139,1-1139,38) */
uint64_t __tmp_in_tmp250;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp250;
}
tmp250[i0] = (role == CLIENT) ? __tmp_in_tmp250 : 0;
}

auto tmp251 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp251 at (1142,1-1142,38) */
uint64_t __tmp_in_tmp251;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp251;
}
tmp251[i0] = (role == CLIENT) ? __tmp_in_tmp251 : 0;
}

auto tmp252 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp252 at (1145,1-1145,38) */
uint64_t __tmp_in_tmp252;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp252;
}
tmp252[i0] = (role == CLIENT) ? __tmp_in_tmp252 : 0;
}

auto tmp253 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp253 at (1148,1-1148,38) */
uint64_t __tmp_in_tmp253;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp253;
}
tmp253[i0] = (role == CLIENT) ? __tmp_in_tmp253 : 0;
}

auto tmp254 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp254 at (1151,1-1151,49) */
uint64_t __tmp_in_tmp254;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp254;
}
tmp254[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp254 : 0;
}
}
}
}

auto tmp255 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp255 at (1154,1-1154,38) */
uint64_t __tmp_in_tmp255;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp255;
}
tmp255[i0] = (role == CLIENT) ? __tmp_in_tmp255 : 0;
}

auto tmp256 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp256 at (1157,1-1157,38) */
uint64_t __tmp_in_tmp256;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp256;
}
tmp256[i0] = (role == CLIENT) ? __tmp_in_tmp256 : 0;
}

auto tmp257 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp257 at (1160,1-1160,38) */
uint64_t __tmp_in_tmp257;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp257;
}
tmp257[i0] = (role == CLIENT) ? __tmp_in_tmp257 : 0;
}

auto tmp258 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp258 at (1163,1-1163,38) */
uint64_t __tmp_in_tmp258;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp258;
}
tmp258[i0] = (role == CLIENT) ? __tmp_in_tmp258 : 0;
}

auto tmp259 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp259 at (1166,1-1166,50) */
uint64_t __tmp_in_tmp259;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp259;
}
tmp259[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp259 : 0;
}
}
}
}

auto tmp260 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp260 at (1169,1-1169,39) */
uint64_t __tmp_in_tmp260;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp260;
}
tmp260[i0] = (role == CLIENT) ? __tmp_in_tmp260 : 0;
}

auto tmp261 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp261 at (1172,1-1172,39) */
uint64_t __tmp_in_tmp261;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp261;
}
tmp261[i0] = (role == CLIENT) ? __tmp_in_tmp261 : 0;
}

auto tmp262 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp262 at (1175,1-1175,39) */
uint64_t __tmp_in_tmp262;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp262;
}
tmp262[i0] = (role == CLIENT) ? __tmp_in_tmp262 : 0;
}

auto tmp263 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp263 at (1178,1-1178,39) */
uint64_t __tmp_in_tmp263;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp263;
}
tmp263[i0] = (role == CLIENT) ? __tmp_in_tmp263 : 0;
}

auto tmp264 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp264 at (1181,1-1181,50) */
uint64_t __tmp_in_tmp264;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp264;
}
tmp264[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp264 : 0;
}
}
}
}

auto tmp265 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp265 at (1184,1-1184,38) */
uint64_t __tmp_in_tmp265;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp265;
}
tmp265[i0] = (role == CLIENT) ? __tmp_in_tmp265 : 0;
}

auto tmp266 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp266 at (1187,1-1187,38) */
uint64_t __tmp_in_tmp266;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp266;
}
tmp266[i0] = (role == CLIENT) ? __tmp_in_tmp266 : 0;
}

auto tmp267 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp267 at (1190,1-1190,38) */
uint64_t __tmp_in_tmp267;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp267;
}
tmp267[i0] = (role == CLIENT) ? __tmp_in_tmp267 : 0;
}

auto tmp268 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp268 at (1193,1-1193,38) */
uint64_t __tmp_in_tmp268;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp268;
}
tmp268[i0] = (role == CLIENT) ? __tmp_in_tmp268 : 0;
}

auto tmp269 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp269 at (1196,1-1196,49) */
uint64_t __tmp_in_tmp269;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp269;
}
tmp269[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp269 : 0;
}
}
}
}

auto tmp270 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp270 at (1199,1-1199,38) */
uint64_t __tmp_in_tmp270;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp270;
}
tmp270[i0] = (role == CLIENT) ? __tmp_in_tmp270 : 0;
}

auto tmp271 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp271 at (1202,1-1202,38) */
uint64_t __tmp_in_tmp271;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp271;
}
tmp271[i0] = (role == CLIENT) ? __tmp_in_tmp271 : 0;
}

auto tmp272 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp272 at (1205,1-1205,38) */
uint64_t __tmp_in_tmp272;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp272;
}
tmp272[i0] = (role == CLIENT) ? __tmp_in_tmp272 : 0;
}

auto tmp273 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp273 at (1208,1-1208,38) */
uint64_t __tmp_in_tmp273;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp273;
}
tmp273[i0] = (role == CLIENT) ? __tmp_in_tmp273 : 0;
}

auto tmp274 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp274 at (1211,1-1211,50) */
uint64_t __tmp_in_tmp274;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp274;
}
tmp274[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp274 : 0;
}
}
}
}

auto tmp275 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp275 at (1214,1-1214,39) */
uint64_t __tmp_in_tmp275;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp275;
}
tmp275[i0] = (role == CLIENT) ? __tmp_in_tmp275 : 0;
}

auto tmp276 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp276 at (1217,1-1217,39) */
uint64_t __tmp_in_tmp276;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp276;
}
tmp276[i0] = (role == CLIENT) ? __tmp_in_tmp276 : 0;
}

auto tmp277 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp277 at (1220,1-1220,39) */
uint64_t __tmp_in_tmp277;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp277;
}
tmp277[i0] = (role == CLIENT) ? __tmp_in_tmp277 : 0;
}

auto tmp278 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp278 at (1223,1-1223,39) */
uint64_t __tmp_in_tmp278;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp278;
}
tmp278[i0] = (role == CLIENT) ? __tmp_in_tmp278 : 0;
}

auto tmp279 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp279 at (1226,1-1226,50) */
uint64_t __tmp_in_tmp279;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp279;
}
tmp279[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp279 : 0;
}
}
}
}

auto tmp280 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp280 at (1229,1-1229,38) */
uint64_t __tmp_in_tmp280;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp280;
}
tmp280[i0] = (role == CLIENT) ? __tmp_in_tmp280 : 0;
}

auto tmp281 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp281 at (1232,1-1232,38) */
uint64_t __tmp_in_tmp281;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp281;
}
tmp281[i0] = (role == CLIENT) ? __tmp_in_tmp281 : 0;
}

auto tmp282 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp282 at (1235,1-1235,38) */
uint64_t __tmp_in_tmp282;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp282;
}
tmp282[i0] = (role == CLIENT) ? __tmp_in_tmp282 : 0;
}

auto tmp283 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp283 at (1238,1-1238,38) */
uint64_t __tmp_in_tmp283;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp283;
}
tmp283[i0] = (role == CLIENT) ? __tmp_in_tmp283 : 0;
}

auto tmp284 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp284 at (1241,1-1241,49) */
uint64_t __tmp_in_tmp284;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp284;
}
tmp284[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp284 : 0;
}
}
}
}

auto tmp285 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp285 at (1244,1-1244,38) */
uint64_t __tmp_in_tmp285;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp285;
}
tmp285[i0] = (role == CLIENT) ? __tmp_in_tmp285 : 0;
}

auto tmp286 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp286 at (1247,1-1247,38) */
uint64_t __tmp_in_tmp286;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp286;
}
tmp286[i0] = (role == CLIENT) ? __tmp_in_tmp286 : 0;
}

auto tmp287 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp287 at (1250,1-1250,38) */
uint64_t __tmp_in_tmp287;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp287;
}
tmp287[i0] = (role == CLIENT) ? __tmp_in_tmp287 : 0;
}

auto tmp288 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp288 at (1253,1-1253,38) */
uint64_t __tmp_in_tmp288;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp288;
}
tmp288[i0] = (role == CLIENT) ? __tmp_in_tmp288 : 0;
}

auto tmp289 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp289 at (1256,1-1256,50) */
uint64_t __tmp_in_tmp289;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp289;
}
tmp289[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp289 : 0;
}
}
}
}

auto tmp290 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp290 at (1259,1-1259,39) */
uint64_t __tmp_in_tmp290;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp290;
}
tmp290[i0] = (role == CLIENT) ? __tmp_in_tmp290 : 0;
}

auto tmp291 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp291 at (1262,1-1262,39) */
uint64_t __tmp_in_tmp291;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp291;
}
tmp291[i0] = (role == CLIENT) ? __tmp_in_tmp291 : 0;
}

auto tmp292 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp292 at (1265,1-1265,39) */
uint64_t __tmp_in_tmp292;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp292;
}
tmp292[i0] = (role == CLIENT) ? __tmp_in_tmp292 : 0;
}

auto tmp293 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp293 at (1268,1-1268,39) */
uint64_t __tmp_in_tmp293;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp293;
}
tmp293[i0] = (role == CLIENT) ? __tmp_in_tmp293 : 0;
}

auto tmp294 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp294 at (1271,1-1271,50) */
uint64_t __tmp_in_tmp294;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp294;
}
tmp294[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp294 : 0;
}
}
}
}

auto tmp295 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp295 at (1274,1-1274,38) */
uint64_t __tmp_in_tmp295;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp295;
}
tmp295[i0] = (role == CLIENT) ? __tmp_in_tmp295 : 0;
}

auto tmp296 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp296 at (1277,1-1277,38) */
uint64_t __tmp_in_tmp296;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp296;
}
tmp296[i0] = (role == CLIENT) ? __tmp_in_tmp296 : 0;
}

auto tmp297 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp297 at (1280,1-1280,38) */
uint64_t __tmp_in_tmp297;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp297;
}
tmp297[i0] = (role == CLIENT) ? __tmp_in_tmp297 : 0;
}

auto tmp298 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp298 at (1283,1-1283,38) */
uint64_t __tmp_in_tmp298;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp298;
}
tmp298[i0] = (role == CLIENT) ? __tmp_in_tmp298 : 0;
}

auto tmp299 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp299 at (1286,1-1286,49) */
uint64_t __tmp_in_tmp299;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp299;
}
tmp299[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp299 : 0;
}
}
}
}

auto tmp300 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp300 at (1289,1-1289,38) */
uint64_t __tmp_in_tmp300;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp300;
}
tmp300[i0] = (role == CLIENT) ? __tmp_in_tmp300 : 0;
}

auto tmp301 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp301 at (1292,1-1292,38) */
uint64_t __tmp_in_tmp301;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp301;
}
tmp301[i0] = (role == CLIENT) ? __tmp_in_tmp301 : 0;
}

auto tmp302 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp302 at (1295,1-1295,38) */
uint64_t __tmp_in_tmp302;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp302;
}
tmp302[i0] = (role == CLIENT) ? __tmp_in_tmp302 : 0;
}

auto tmp303 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp303 at (1298,1-1298,38) */
uint64_t __tmp_in_tmp303;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp303;
}
tmp303[i0] = (role == CLIENT) ? __tmp_in_tmp303 : 0;
}

auto tmp304 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp304 at (1301,1-1301,50) */
uint64_t __tmp_in_tmp304;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp304;
}
tmp304[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp304 : 0;
}
}
}
}

auto tmp305 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp305 at (1304,1-1304,39) */
uint64_t __tmp_in_tmp305;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp305;
}
tmp305[i0] = (role == CLIENT) ? __tmp_in_tmp305 : 0;
}

auto tmp306 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp306 at (1307,1-1307,39) */
uint64_t __tmp_in_tmp306;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp306;
}
tmp306[i0] = (role == CLIENT) ? __tmp_in_tmp306 : 0;
}

auto tmp307 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp307 at (1310,1-1310,39) */
uint64_t __tmp_in_tmp307;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp307;
}
tmp307[i0] = (role == CLIENT) ? __tmp_in_tmp307 : 0;
}

auto tmp308 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp308 at (1313,1-1313,39) */
uint64_t __tmp_in_tmp308;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp308;
}
tmp308[i0] = (role == CLIENT) ? __tmp_in_tmp308 : 0;
}

auto tmp309 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp309 at (1316,1-1316,50) */
uint64_t __tmp_in_tmp309;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp309;
}
tmp309[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp309 : 0;
}
}
}
}

auto tmp310 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp310 at (1319,1-1319,38) */
uint64_t __tmp_in_tmp310;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp310;
}
tmp310[i0] = (role == CLIENT) ? __tmp_in_tmp310 : 0;
}

auto tmp311 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp311 at (1322,1-1322,38) */
uint64_t __tmp_in_tmp311;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp311;
}
tmp311[i0] = (role == CLIENT) ? __tmp_in_tmp311 : 0;
}

auto tmp312 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp312 at (1325,1-1325,38) */
uint64_t __tmp_in_tmp312;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp312;
}
tmp312[i0] = (role == CLIENT) ? __tmp_in_tmp312 : 0;
}

auto tmp313 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp313 at (1328,1-1328,38) */
uint64_t __tmp_in_tmp313;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp313;
}
tmp313[i0] = (role == CLIENT) ? __tmp_in_tmp313 : 0;
}

auto tmp314 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp314 at (1331,1-1331,49) */
uint64_t __tmp_in_tmp314;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp314;
}
tmp314[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp314 : 0;
}
}
}
}

auto tmp315 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp315 at (1334,1-1334,38) */
uint64_t __tmp_in_tmp315;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp315;
}
tmp315[i0] = (role == CLIENT) ? __tmp_in_tmp315 : 0;
}

auto tmp316 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp316 at (1337,1-1337,38) */
uint64_t __tmp_in_tmp316;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp316;
}
tmp316[i0] = (role == CLIENT) ? __tmp_in_tmp316 : 0;
}

auto tmp317 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp317 at (1340,1-1340,38) */
uint64_t __tmp_in_tmp317;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp317;
}
tmp317[i0] = (role == CLIENT) ? __tmp_in_tmp317 : 0;
}

auto tmp318 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp318 at (1343,1-1343,38) */
uint64_t __tmp_in_tmp318;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp318;
}
tmp318[i0] = (role == CLIENT) ? __tmp_in_tmp318 : 0;
}

auto tmp319 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp319 at (1346,1-1346,50) */
uint64_t __tmp_in_tmp319;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp319;
}
tmp319[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp319 : 0;
}
}
}
}

auto tmp320 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp320 at (1349,1-1349,39) */
uint64_t __tmp_in_tmp320;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp320;
}
tmp320[i0] = (role == CLIENT) ? __tmp_in_tmp320 : 0;
}

auto tmp321 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp321 at (1352,1-1352,39) */
uint64_t __tmp_in_tmp321;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp321;
}
tmp321[i0] = (role == CLIENT) ? __tmp_in_tmp321 : 0;
}

auto tmp322 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp322 at (1355,1-1355,39) */
uint64_t __tmp_in_tmp322;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp322;
}
tmp322[i0] = (role == CLIENT) ? __tmp_in_tmp322 : 0;
}

auto tmp323 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp323 at (1358,1-1358,39) */
uint64_t __tmp_in_tmp323;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp323;
}
tmp323[i0] = (role == CLIENT) ? __tmp_in_tmp323 : 0;
}

auto tmp324 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp324 at (1361,1-1361,50) */
uint64_t __tmp_in_tmp324;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp324;
}
tmp324[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp324 : 0;
}
}
}
}

auto tmp325 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp325 at (1364,1-1364,38) */
uint64_t __tmp_in_tmp325;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp325;
}
tmp325[i0] = (role == CLIENT) ? __tmp_in_tmp325 : 0;
}

auto tmp326 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp326 at (1367,1-1367,38) */
uint64_t __tmp_in_tmp326;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp326;
}
tmp326[i0] = (role == CLIENT) ? __tmp_in_tmp326 : 0;
}

auto tmp327 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp327 at (1370,1-1370,38) */
uint64_t __tmp_in_tmp327;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp327;
}
tmp327[i0] = (role == CLIENT) ? __tmp_in_tmp327 : 0;
}

auto tmp328 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp328 at (1373,1-1373,38) */
uint64_t __tmp_in_tmp328;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp328;
}
tmp328[i0] = (role == CLIENT) ? __tmp_in_tmp328 : 0;
}

auto tmp329 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp329 at (1376,1-1376,49) */
uint64_t __tmp_in_tmp329;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp329;
}
tmp329[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp329 : 0;
}
}
}
}

auto tmp330 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp330 at (1379,1-1379,38) */
uint64_t __tmp_in_tmp330;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp330;
}
tmp330[i0] = (role == CLIENT) ? __tmp_in_tmp330 : 0;
}

auto tmp331 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp331 at (1382,1-1382,38) */
uint64_t __tmp_in_tmp331;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp331;
}
tmp331[i0] = (role == CLIENT) ? __tmp_in_tmp331 : 0;
}

auto tmp332 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp332 at (1385,1-1385,38) */
uint64_t __tmp_in_tmp332;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp332;
}
tmp332[i0] = (role == CLIENT) ? __tmp_in_tmp332 : 0;
}

auto tmp333 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp333 at (1388,1-1388,38) */
uint64_t __tmp_in_tmp333;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp333;
}
tmp333[i0] = (role == CLIENT) ? __tmp_in_tmp333 : 0;
}

auto tmp334 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp334 at (1391,1-1391,50) */
uint64_t __tmp_in_tmp334;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp334;
}
tmp334[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp334 : 0;
}
}
}
}

auto tmp335 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp335 at (1394,1-1394,39) */
uint64_t __tmp_in_tmp335;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp335;
}
tmp335[i0] = (role == CLIENT) ? __tmp_in_tmp335 : 0;
}

auto tmp336 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp336 at (1397,1-1397,39) */
uint64_t __tmp_in_tmp336;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp336;
}
tmp336[i0] = (role == CLIENT) ? __tmp_in_tmp336 : 0;
}

auto tmp337 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp337 at (1400,1-1400,39) */
uint64_t __tmp_in_tmp337;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp337;
}
tmp337[i0] = (role == CLIENT) ? __tmp_in_tmp337 : 0;
}

auto tmp338 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp338 at (1403,1-1403,39) */
uint64_t __tmp_in_tmp338;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp338;
}
tmp338[i0] = (role == CLIENT) ? __tmp_in_tmp338 : 0;
}

auto tmp339 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp339 at (1406,1-1406,50) */
uint64_t __tmp_in_tmp339;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp339;
}
tmp339[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp339 : 0;
}
}
}
}

auto tmp340 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp340 at (1409,1-1409,38) */
uint64_t __tmp_in_tmp340;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp340;
}
tmp340[i0] = (role == CLIENT) ? __tmp_in_tmp340 : 0;
}

auto tmp341 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp341 at (1412,1-1412,38) */
uint64_t __tmp_in_tmp341;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp341;
}
tmp341[i0] = (role == CLIENT) ? __tmp_in_tmp341 : 0;
}

auto tmp342 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp342 at (1415,1-1415,38) */
uint64_t __tmp_in_tmp342;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp342;
}
tmp342[i0] = (role == CLIENT) ? __tmp_in_tmp342 : 0;
}

auto tmp343 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp343 at (1418,1-1418,38) */
uint64_t __tmp_in_tmp343;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp343;
}
tmp343[i0] = (role == CLIENT) ? __tmp_in_tmp343 : 0;
}

auto tmp344 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp344 at (1421,1-1421,49) */
uint64_t __tmp_in_tmp344;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp344;
}
tmp344[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp344 : 0;
}
}
}
}

auto tmp345 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp345 at (1424,1-1424,38) */
uint64_t __tmp_in_tmp345;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp345;
}
tmp345[i0] = (role == CLIENT) ? __tmp_in_tmp345 : 0;
}

auto tmp346 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp346 at (1427,1-1427,38) */
uint64_t __tmp_in_tmp346;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp346;
}
tmp346[i0] = (role == CLIENT) ? __tmp_in_tmp346 : 0;
}

auto tmp347 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp347 at (1430,1-1430,38) */
uint64_t __tmp_in_tmp347;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp347;
}
tmp347[i0] = (role == CLIENT) ? __tmp_in_tmp347 : 0;
}

auto tmp348 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp348 at (1433,1-1433,38) */
uint64_t __tmp_in_tmp348;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp348;
}
tmp348[i0] = (role == CLIENT) ? __tmp_in_tmp348 : 0;
}

auto tmp349 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp349 at (1436,1-1436,50) */
uint64_t __tmp_in_tmp349;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp349;
}
tmp349[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp349 : 0;
}
}
}
}

auto tmp350 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp350 at (1439,1-1439,39) */
uint64_t __tmp_in_tmp350;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp350;
}
tmp350[i0] = (role == CLIENT) ? __tmp_in_tmp350 : 0;
}

auto tmp351 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp351 at (1442,1-1442,39) */
uint64_t __tmp_in_tmp351;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp351;
}
tmp351[i0] = (role == CLIENT) ? __tmp_in_tmp351 : 0;
}

auto tmp352 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp352 at (1445,1-1445,39) */
uint64_t __tmp_in_tmp352;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp352;
}
tmp352[i0] = (role == CLIENT) ? __tmp_in_tmp352 : 0;
}

auto tmp353 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp353 at (1448,1-1448,39) */
uint64_t __tmp_in_tmp353;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp353;
}
tmp353[i0] = (role == CLIENT) ? __tmp_in_tmp353 : 0;
}

auto tmp354 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp354 at (1451,1-1451,50) */
uint64_t __tmp_in_tmp354;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp354;
}
tmp354[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp354 : 0;
}
}
}
}

auto tmp355 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp355 at (1454,1-1454,38) */
uint64_t __tmp_in_tmp355;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp355;
}
tmp355[i0] = (role == CLIENT) ? __tmp_in_tmp355 : 0;
}

auto tmp356 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp356 at (1457,1-1457,38) */
uint64_t __tmp_in_tmp356;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp356;
}
tmp356[i0] = (role == CLIENT) ? __tmp_in_tmp356 : 0;
}

auto tmp357 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp357 at (1460,1-1460,38) */
uint64_t __tmp_in_tmp357;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp357;
}
tmp357[i0] = (role == CLIENT) ? __tmp_in_tmp357 : 0;
}

auto tmp358 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp358 at (1463,1-1463,38) */
uint64_t __tmp_in_tmp358;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp358;
}
tmp358[i0] = (role == CLIENT) ? __tmp_in_tmp358 : 0;
}

auto tmp359 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp359 at (1466,1-1466,49) */
uint64_t __tmp_in_tmp359;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp359;
}
tmp359[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp359 : 0;
}
}
}
}

auto tmp360 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp360 at (1469,1-1469,38) */
uint64_t __tmp_in_tmp360;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp360;
}
tmp360[i0] = (role == CLIENT) ? __tmp_in_tmp360 : 0;
}

auto tmp361 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp361 at (1472,1-1472,38) */
uint64_t __tmp_in_tmp361;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp361;
}
tmp361[i0] = (role == CLIENT) ? __tmp_in_tmp361 : 0;
}

auto tmp362 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp362 at (1475,1-1475,38) */
uint64_t __tmp_in_tmp362;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp362;
}
tmp362[i0] = (role == CLIENT) ? __tmp_in_tmp362 : 0;
}

auto tmp363 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp363 at (1478,1-1478,38) */
uint64_t __tmp_in_tmp363;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp363;
}
tmp363[i0] = (role == CLIENT) ? __tmp_in_tmp363 : 0;
}

auto tmp364 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp364 at (1481,1-1481,50) */
uint64_t __tmp_in_tmp364;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp364;
}
tmp364[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp364 : 0;
}
}
}
}

auto tmp365 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp365 at (1484,1-1484,39) */
uint64_t __tmp_in_tmp365;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp365;
}
tmp365[i0] = (role == CLIENT) ? __tmp_in_tmp365 : 0;
}

auto tmp366 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp366 at (1487,1-1487,39) */
uint64_t __tmp_in_tmp366;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp366;
}
tmp366[i0] = (role == CLIENT) ? __tmp_in_tmp366 : 0;
}

auto tmp367 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp367 at (1490,1-1490,39) */
uint64_t __tmp_in_tmp367;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp367;
}
tmp367[i0] = (role == CLIENT) ? __tmp_in_tmp367 : 0;
}

auto tmp368 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp368 at (1493,1-1493,39) */
uint64_t __tmp_in_tmp368;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp368;
}
tmp368[i0] = (role == CLIENT) ? __tmp_in_tmp368 : 0;
}

auto tmp369 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp369 at (1496,1-1496,50) */
uint64_t __tmp_in_tmp369;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp369;
}
tmp369[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp369 : 0;
}
}
}
}

auto tmp370 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp370 at (1499,1-1499,38) */
uint64_t __tmp_in_tmp370;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp370;
}
tmp370[i0] = (role == CLIENT) ? __tmp_in_tmp370 : 0;
}

auto tmp371 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp371 at (1502,1-1502,38) */
uint64_t __tmp_in_tmp371;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp371;
}
tmp371[i0] = (role == CLIENT) ? __tmp_in_tmp371 : 0;
}

auto tmp372 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp372 at (1505,1-1505,38) */
uint64_t __tmp_in_tmp372;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp372;
}
tmp372[i0] = (role == CLIENT) ? __tmp_in_tmp372 : 0;
}

auto tmp373 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp373 at (1508,1-1508,38) */
uint64_t __tmp_in_tmp373;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp373;
}
tmp373[i0] = (role == CLIENT) ? __tmp_in_tmp373 : 0;
}

auto tmp374 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp374 at (1511,1-1511,49) */
uint64_t __tmp_in_tmp374;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp374;
}
tmp374[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp374 : 0;
}
}
}
}

auto tmp375 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp375 at (1514,1-1514,38) */
uint64_t __tmp_in_tmp375;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp375;
}
tmp375[i0] = (role == CLIENT) ? __tmp_in_tmp375 : 0;
}

auto tmp376 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp376 at (1517,1-1517,38) */
uint64_t __tmp_in_tmp376;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp376;
}
tmp376[i0] = (role == CLIENT) ? __tmp_in_tmp376 : 0;
}

auto tmp377 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp377 at (1520,1-1520,38) */
uint64_t __tmp_in_tmp377;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp377;
}
tmp377[i0] = (role == CLIENT) ? __tmp_in_tmp377 : 0;
}

auto tmp378 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp378 at (1523,1-1523,38) */
uint64_t __tmp_in_tmp378;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp378;
}
tmp378[i0] = (role == CLIENT) ? __tmp_in_tmp378 : 0;
}

auto tmp379 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp379 at (1526,1-1526,50) */
uint64_t __tmp_in_tmp379;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp379;
}
tmp379[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp379 : 0;
}
}
}
}

auto tmp380 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp380 at (1529,1-1529,39) */
uint64_t __tmp_in_tmp380;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp380;
}
tmp380[i0] = (role == CLIENT) ? __tmp_in_tmp380 : 0;
}

auto tmp381 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp381 at (1532,1-1532,39) */
uint64_t __tmp_in_tmp381;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp381;
}
tmp381[i0] = (role == CLIENT) ? __tmp_in_tmp381 : 0;
}

auto tmp382 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp382 at (1535,1-1535,39) */
uint64_t __tmp_in_tmp382;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp382;
}
tmp382[i0] = (role == CLIENT) ? __tmp_in_tmp382 : 0;
}

auto tmp383 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp383 at (1538,1-1538,39) */
uint64_t __tmp_in_tmp383;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp383;
}
tmp383[i0] = (role == CLIENT) ? __tmp_in_tmp383 : 0;
}

auto tmp384 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp384 at (1541,1-1541,50) */
uint64_t __tmp_in_tmp384;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp384;
}
tmp384[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp384 : 0;
}
}
}
}

auto tmp385 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp385 at (1544,1-1544,38) */
uint64_t __tmp_in_tmp385;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp385;
}
tmp385[i0] = (role == CLIENT) ? __tmp_in_tmp385 : 0;
}

auto tmp386 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp386 at (1547,1-1547,38) */
uint64_t __tmp_in_tmp386;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp386;
}
tmp386[i0] = (role == CLIENT) ? __tmp_in_tmp386 : 0;
}

auto tmp387 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp387 at (1550,1-1550,38) */
uint64_t __tmp_in_tmp387;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp387;
}
tmp387[i0] = (role == CLIENT) ? __tmp_in_tmp387 : 0;
}

auto tmp388 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp388 at (1553,1-1553,38) */
uint64_t __tmp_in_tmp388;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp388;
}
tmp388[i0] = (role == CLIENT) ? __tmp_in_tmp388 : 0;
}

auto tmp389 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp389 at (1556,1-1556,49) */
uint64_t __tmp_in_tmp389;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp389;
}
tmp389[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp389 : 0;
}
}
}
}

auto tmp390 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp390 at (1559,1-1559,38) */
uint64_t __tmp_in_tmp390;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp390;
}
tmp390[i0] = (role == CLIENT) ? __tmp_in_tmp390 : 0;
}

auto tmp391 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp391 at (1562,1-1562,38) */
uint64_t __tmp_in_tmp391;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp391;
}
tmp391[i0] = (role == CLIENT) ? __tmp_in_tmp391 : 0;
}

auto tmp392 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp392 at (1565,1-1565,38) */
uint64_t __tmp_in_tmp392;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp392;
}
tmp392[i0] = (role == CLIENT) ? __tmp_in_tmp392 : 0;
}

auto tmp393 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp393 at (1568,1-1568,38) */
uint64_t __tmp_in_tmp393;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp393;
}
tmp393[i0] = (role == CLIENT) ? __tmp_in_tmp393 : 0;
}

auto tmp394 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp394 at (1571,1-1571,50) */
uint64_t __tmp_in_tmp394;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp394;
}
tmp394[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp394 : 0;
}
}
}
}

auto tmp395 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp395 at (1574,1-1574,39) */
uint64_t __tmp_in_tmp395;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp395;
}
tmp395[i0] = (role == CLIENT) ? __tmp_in_tmp395 : 0;
}

auto tmp396 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp396 at (1577,1-1577,39) */
uint64_t __tmp_in_tmp396;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp396;
}
tmp396[i0] = (role == CLIENT) ? __tmp_in_tmp396 : 0;
}

auto tmp397 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp397 at (1580,1-1580,39) */
uint64_t __tmp_in_tmp397;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp397;
}
tmp397[i0] = (role == CLIENT) ? __tmp_in_tmp397 : 0;
}

auto tmp398 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp398 at (1583,1-1583,39) */
uint64_t __tmp_in_tmp398;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp398;
}
tmp398[i0] = (role == CLIENT) ? __tmp_in_tmp398 : 0;
}

auto tmp399 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp399 at (1586,1-1586,50) */
uint64_t __tmp_in_tmp399;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp399;
}
tmp399[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp399 : 0;
}
}
}
}

auto tmp400 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp400 at (1589,1-1589,38) */
uint64_t __tmp_in_tmp400;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp400;
}
tmp400[i0] = (role == CLIENT) ? __tmp_in_tmp400 : 0;
}

auto tmp401 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp401 at (1592,1-1592,38) */
uint64_t __tmp_in_tmp401;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp401;
}
tmp401[i0] = (role == CLIENT) ? __tmp_in_tmp401 : 0;
}

auto tmp402 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp402 at (1595,1-1595,38) */
uint64_t __tmp_in_tmp402;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp402;
}
tmp402[i0] = (role == CLIENT) ? __tmp_in_tmp402 : 0;
}

auto tmp403 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp403 at (1598,1-1598,38) */
uint64_t __tmp_in_tmp403;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp403;
}
tmp403[i0] = (role == CLIENT) ? __tmp_in_tmp403 : 0;
}

auto tmp404 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp404 at (1601,1-1601,49) */
uint64_t __tmp_in_tmp404;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp404;
}
tmp404[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp404 : 0;
}
}
}
}

auto tmp405 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp405 at (1604,1-1604,38) */
uint64_t __tmp_in_tmp405;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp405;
}
tmp405[i0] = (role == CLIENT) ? __tmp_in_tmp405 : 0;
}

auto tmp406 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp406 at (1607,1-1607,38) */
uint64_t __tmp_in_tmp406;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp406;
}
tmp406[i0] = (role == CLIENT) ? __tmp_in_tmp406 : 0;
}

auto tmp407 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp407 at (1610,1-1610,38) */
uint64_t __tmp_in_tmp407;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp407;
}
tmp407[i0] = (role == CLIENT) ? __tmp_in_tmp407 : 0;
}

auto tmp408 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp408 at (1613,1-1613,38) */
uint64_t __tmp_in_tmp408;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp408;
}
tmp408[i0] = (role == CLIENT) ? __tmp_in_tmp408 : 0;
}

auto tmp409 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp409 at (1616,1-1616,50) */
uint64_t __tmp_in_tmp409;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp409;
}
tmp409[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp409 : 0;
}
}
}
}

auto tmp410 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp410 at (1619,1-1619,39) */
uint64_t __tmp_in_tmp410;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp410;
}
tmp410[i0] = (role == CLIENT) ? __tmp_in_tmp410 : 0;
}

auto tmp411 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp411 at (1622,1-1622,39) */
uint64_t __tmp_in_tmp411;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp411;
}
tmp411[i0] = (role == CLIENT) ? __tmp_in_tmp411 : 0;
}

auto tmp412 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp412 at (1625,1-1625,39) */
uint64_t __tmp_in_tmp412;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp412;
}
tmp412[i0] = (role == CLIENT) ? __tmp_in_tmp412 : 0;
}

auto tmp413 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp413 at (1628,1-1628,39) */
uint64_t __tmp_in_tmp413;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp413;
}
tmp413[i0] = (role == CLIENT) ? __tmp_in_tmp413 : 0;
}

auto tmp414 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp414 at (1631,1-1631,50) */
uint64_t __tmp_in_tmp414;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp414;
}
tmp414[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp414 : 0;
}
}
}
}

auto tmp415 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp415 at (1634,1-1634,38) */
uint64_t __tmp_in_tmp415;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp415;
}
tmp415[i0] = (role == CLIENT) ? __tmp_in_tmp415 : 0;
}

auto tmp416 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp416 at (1637,1-1637,38) */
uint64_t __tmp_in_tmp416;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp416;
}
tmp416[i0] = (role == CLIENT) ? __tmp_in_tmp416 : 0;
}

auto tmp417 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp417 at (1640,1-1640,38) */
uint64_t __tmp_in_tmp417;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp417;
}
tmp417[i0] = (role == CLIENT) ? __tmp_in_tmp417 : 0;
}

auto tmp418 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp418 at (1643,1-1643,38) */
uint64_t __tmp_in_tmp418;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp418;
}
tmp418[i0] = (role == CLIENT) ? __tmp_in_tmp418 : 0;
}

auto tmp419 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp419 at (1646,1-1646,49) */
uint64_t __tmp_in_tmp419;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp419;
}
tmp419[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp419 : 0;
}
}
}
}

auto tmp420 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp420 at (1649,1-1649,38) */
uint64_t __tmp_in_tmp420;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp420;
}
tmp420[i0] = (role == CLIENT) ? __tmp_in_tmp420 : 0;
}

auto tmp421 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp421 at (1652,1-1652,38) */
uint64_t __tmp_in_tmp421;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp421;
}
tmp421[i0] = (role == CLIENT) ? __tmp_in_tmp421 : 0;
}

auto tmp422 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp422 at (1655,1-1655,38) */
uint64_t __tmp_in_tmp422;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp422;
}
tmp422[i0] = (role == CLIENT) ? __tmp_in_tmp422 : 0;
}

auto tmp423 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp423 at (1658,1-1658,38) */
uint64_t __tmp_in_tmp423;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp423;
}
tmp423[i0] = (role == CLIENT) ? __tmp_in_tmp423 : 0;
}

auto tmp424 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp424 at (1661,1-1661,50) */
uint64_t __tmp_in_tmp424;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp424;
}
tmp424[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp424 : 0;
}
}
}
}

auto tmp425 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp425 at (1664,1-1664,39) */
uint64_t __tmp_in_tmp425;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp425;
}
tmp425[i0] = (role == CLIENT) ? __tmp_in_tmp425 : 0;
}

auto tmp426 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp426 at (1667,1-1667,39) */
uint64_t __tmp_in_tmp426;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp426;
}
tmp426[i0] = (role == CLIENT) ? __tmp_in_tmp426 : 0;
}

auto tmp427 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp427 at (1670,1-1670,39) */
uint64_t __tmp_in_tmp427;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp427;
}
tmp427[i0] = (role == CLIENT) ? __tmp_in_tmp427 : 0;
}

auto tmp428 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp428 at (1673,1-1673,39) */
uint64_t __tmp_in_tmp428;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp428;
}
tmp428[i0] = (role == CLIENT) ? __tmp_in_tmp428 : 0;
}

auto tmp429 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp429 at (1676,1-1676,50) */
uint64_t __tmp_in_tmp429;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp429;
}
tmp429[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp429 : 0;
}
}
}
}

auto tmp430 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp430 at (1679,1-1679,38) */
uint64_t __tmp_in_tmp430;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp430;
}
tmp430[i0] = (role == CLIENT) ? __tmp_in_tmp430 : 0;
}

auto tmp431 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp431 at (1682,1-1682,38) */
uint64_t __tmp_in_tmp431;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp431;
}
tmp431[i0] = (role == CLIENT) ? __tmp_in_tmp431 : 0;
}

auto tmp432 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp432 at (1685,1-1685,38) */
uint64_t __tmp_in_tmp432;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp432;
}
tmp432[i0] = (role == CLIENT) ? __tmp_in_tmp432 : 0;
}

auto tmp433 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp433 at (1688,1-1688,38) */
uint64_t __tmp_in_tmp433;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp433;
}
tmp433[i0] = (role == CLIENT) ? __tmp_in_tmp433 : 0;
}

auto tmp434 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp434 at (1691,1-1691,49) */
uint64_t __tmp_in_tmp434;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp434;
}
tmp434[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp434 : 0;
}
}
}
}

auto tmp435 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp435 at (1694,1-1694,38) */
uint64_t __tmp_in_tmp435;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp435;
}
tmp435[i0] = (role == CLIENT) ? __tmp_in_tmp435 : 0;
}

auto tmp436 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp436 at (1697,1-1697,38) */
uint64_t __tmp_in_tmp436;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp436;
}
tmp436[i0] = (role == CLIENT) ? __tmp_in_tmp436 : 0;
}

auto tmp437 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp437 at (1700,1-1700,38) */
uint64_t __tmp_in_tmp437;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp437;
}
tmp437[i0] = (role == CLIENT) ? __tmp_in_tmp437 : 0;
}

auto tmp438 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp438 at (1703,1-1703,38) */
uint64_t __tmp_in_tmp438;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp438;
}
tmp438[i0] = (role == CLIENT) ? __tmp_in_tmp438 : 0;
}

auto tmp439 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp439 at (1706,1-1706,50) */
uint64_t __tmp_in_tmp439;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp439;
}
tmp439[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp439 : 0;
}
}
}
}

auto tmp440 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp440 at (1709,1-1709,39) */
uint64_t __tmp_in_tmp440;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp440;
}
tmp440[i0] = (role == CLIENT) ? __tmp_in_tmp440 : 0;
}

auto tmp441 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp441 at (1712,1-1712,39) */
uint64_t __tmp_in_tmp441;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp441;
}
tmp441[i0] = (role == CLIENT) ? __tmp_in_tmp441 : 0;
}

auto tmp442 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp442 at (1715,1-1715,39) */
uint64_t __tmp_in_tmp442;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp442;
}
tmp442[i0] = (role == CLIENT) ? __tmp_in_tmp442 : 0;
}

auto tmp443 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp443 at (1718,1-1718,39) */
uint64_t __tmp_in_tmp443;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp443;
}
tmp443[i0] = (role == CLIENT) ? __tmp_in_tmp443 : 0;
}

auto tmp444 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp444 at (1721,1-1721,50) */
uint64_t __tmp_in_tmp444;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp444;
}
tmp444[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp444 : 0;
}
}
}
}

auto tmp445 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp445 at (1724,1-1724,38) */
uint64_t __tmp_in_tmp445;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp445;
}
tmp445[i0] = (role == CLIENT) ? __tmp_in_tmp445 : 0;
}

auto tmp446 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp446 at (1727,1-1727,38) */
uint64_t __tmp_in_tmp446;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp446;
}
tmp446[i0] = (role == CLIENT) ? __tmp_in_tmp446 : 0;
}

auto tmp447 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp447 at (1730,1-1730,38) */
uint64_t __tmp_in_tmp447;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp447;
}
tmp447[i0] = (role == CLIENT) ? __tmp_in_tmp447 : 0;
}

auto tmp448 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp448 at (1733,1-1733,38) */
uint64_t __tmp_in_tmp448;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp448;
}
tmp448[i0] = (role == CLIENT) ? __tmp_in_tmp448 : 0;
}

auto tmp449 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp449 at (1736,1-1736,49) */
uint64_t __tmp_in_tmp449;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp449;
}
tmp449[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp449 : 0;
}
}
}
}

auto tmp450 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp450 at (1739,1-1739,38) */
uint64_t __tmp_in_tmp450;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp450;
}
tmp450[i0] = (role == CLIENT) ? __tmp_in_tmp450 : 0;
}

auto tmp451 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp451 at (1742,1-1742,38) */
uint64_t __tmp_in_tmp451;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp451;
}
tmp451[i0] = (role == CLIENT) ? __tmp_in_tmp451 : 0;
}

auto tmp452 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp452 at (1745,1-1745,38) */
uint64_t __tmp_in_tmp452;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp452;
}
tmp452[i0] = (role == CLIENT) ? __tmp_in_tmp452 : 0;
}

auto tmp453 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp453 at (1748,1-1748,38) */
uint64_t __tmp_in_tmp453;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp453;
}
tmp453[i0] = (role == CLIENT) ? __tmp_in_tmp453 : 0;
}

auto tmp454 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp454 at (1751,1-1751,50) */
uint64_t __tmp_in_tmp454;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp454;
}
tmp454[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp454 : 0;
}
}
}
}

auto tmp455 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp455 at (1754,1-1754,39) */
uint64_t __tmp_in_tmp455;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp455;
}
tmp455[i0] = (role == CLIENT) ? __tmp_in_tmp455 : 0;
}

auto tmp456 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp456 at (1757,1-1757,39) */
uint64_t __tmp_in_tmp456;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp456;
}
tmp456[i0] = (role == CLIENT) ? __tmp_in_tmp456 : 0;
}

auto tmp457 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp457 at (1760,1-1760,39) */
uint64_t __tmp_in_tmp457;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp457;
}
tmp457[i0] = (role == CLIENT) ? __tmp_in_tmp457 : 0;
}

auto tmp458 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp458 at (1763,1-1763,39) */
uint64_t __tmp_in_tmp458;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp458;
}
tmp458[i0] = (role == CLIENT) ? __tmp_in_tmp458 : 0;
}

auto tmp459 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp459 at (1766,1-1766,51) */
uint64_t __tmp_in_tmp459;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp459;
}
tmp459[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp459 : 0;
}
}
}
}

auto tmp460 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp460 at (1769,1-1769,50) */
uint64_t __tmp_in_tmp460;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp460;
}
tmp460[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp460 : 0;
}
}
}
}

auto tmp461 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp461 at (1772,1-1772,38) */
uint64_t __tmp_in_tmp461;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp461;
}
tmp461[i0] = (role == CLIENT) ? __tmp_in_tmp461 : 0;
}

auto tmp462 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp462 at (1775,1-1775,38) */
uint64_t __tmp_in_tmp462;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp462;
}
tmp462[i0] = (role == CLIENT) ? __tmp_in_tmp462 : 0;
}

auto tmp463 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp463 at (1778,1-1778,38) */
uint64_t __tmp_in_tmp463;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp463;
}
tmp463[i0] = (role == CLIENT) ? __tmp_in_tmp463 : 0;
}

auto tmp464 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp464 at (1781,1-1781,38) */
uint64_t __tmp_in_tmp464;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp464;
}
tmp464[i0] = (role == CLIENT) ? __tmp_in_tmp464 : 0;
}

auto tmp465 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp465 at (1784,1-1784,49) */
uint64_t __tmp_in_tmp465;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp465;
}
tmp465[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp465 : 0;
}
}
}
}

auto tmp466 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp466 at (1787,1-1787,38) */
uint64_t __tmp_in_tmp466;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp466;
}
tmp466[i0] = (role == CLIENT) ? __tmp_in_tmp466 : 0;
}

auto tmp467 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp467 at (1790,1-1790,38) */
uint64_t __tmp_in_tmp467;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp467;
}
tmp467[i0] = (role == CLIENT) ? __tmp_in_tmp467 : 0;
}

auto tmp468 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp468 at (1793,1-1793,38) */
uint64_t __tmp_in_tmp468;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp468;
}
tmp468[i0] = (role == CLIENT) ? __tmp_in_tmp468 : 0;
}

auto tmp469 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp469 at (1796,1-1796,38) */
uint64_t __tmp_in_tmp469;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp469;
}
tmp469[i0] = (role == CLIENT) ? __tmp_in_tmp469 : 0;
}

auto tmp470 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp470 at (1799,1-1799,50) */
uint64_t __tmp_in_tmp470;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp470;
}
tmp470[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp470 : 0;
}
}
}
}

auto tmp471 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp471 at (1802,1-1802,39) */
uint64_t __tmp_in_tmp471;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp471;
}
tmp471[i0] = (role == CLIENT) ? __tmp_in_tmp471 : 0;
}

auto tmp472 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp472 at (1805,1-1805,39) */
uint64_t __tmp_in_tmp472;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp472;
}
tmp472[i0] = (role == CLIENT) ? __tmp_in_tmp472 : 0;
}

auto tmp473 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp473 at (1808,1-1808,39) */
uint64_t __tmp_in_tmp473;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp473;
}
tmp473[i0] = (role == CLIENT) ? __tmp_in_tmp473 : 0;
}

auto tmp474 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp474 at (1811,1-1811,39) */
uint64_t __tmp_in_tmp474;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp474;
}
tmp474[i0] = (role == CLIENT) ? __tmp_in_tmp474 : 0;
}

auto tmp475 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp475 at (1814,1-1814,50) */
uint64_t __tmp_in_tmp475;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp475;
}
tmp475[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp475 : 0;
}
}
}
}

auto tmp476 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp476 at (1817,1-1817,38) */
uint64_t __tmp_in_tmp476;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp476;
}
tmp476[i0] = (role == CLIENT) ? __tmp_in_tmp476 : 0;
}

auto tmp477 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp477 at (1820,1-1820,38) */
uint64_t __tmp_in_tmp477;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp477;
}
tmp477[i0] = (role == CLIENT) ? __tmp_in_tmp477 : 0;
}

auto tmp478 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp478 at (1823,1-1823,38) */
uint64_t __tmp_in_tmp478;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp478;
}
tmp478[i0] = (role == CLIENT) ? __tmp_in_tmp478 : 0;
}

auto tmp479 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp479 at (1826,1-1826,38) */
uint64_t __tmp_in_tmp479;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp479;
}
tmp479[i0] = (role == CLIENT) ? __tmp_in_tmp479 : 0;
}

auto tmp480 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp480 at (1829,1-1829,49) */
uint64_t __tmp_in_tmp480;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp480;
}
tmp480[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp480 : 0;
}
}
}
}

auto tmp481 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp481 at (1832,1-1832,38) */
uint64_t __tmp_in_tmp481;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp481;
}
tmp481[i0] = (role == CLIENT) ? __tmp_in_tmp481 : 0;
}

auto tmp482 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp482 at (1835,1-1835,38) */
uint64_t __tmp_in_tmp482;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp482;
}
tmp482[i0] = (role == CLIENT) ? __tmp_in_tmp482 : 0;
}

auto tmp483 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp483 at (1838,1-1838,38) */
uint64_t __tmp_in_tmp483;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp483;
}
tmp483[i0] = (role == CLIENT) ? __tmp_in_tmp483 : 0;
}

auto tmp484 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp484 at (1841,1-1841,38) */
uint64_t __tmp_in_tmp484;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp484;
}
tmp484[i0] = (role == CLIENT) ? __tmp_in_tmp484 : 0;
}

auto tmp485 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp485 at (1844,1-1844,50) */
uint64_t __tmp_in_tmp485;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp485;
}
tmp485[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp485 : 0;
}
}
}
}

auto tmp486 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp486 at (1847,1-1847,39) */
uint64_t __tmp_in_tmp486;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp486;
}
tmp486[i0] = (role == CLIENT) ? __tmp_in_tmp486 : 0;
}

auto tmp487 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp487 at (1850,1-1850,39) */
uint64_t __tmp_in_tmp487;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp487;
}
tmp487[i0] = (role == CLIENT) ? __tmp_in_tmp487 : 0;
}

auto tmp488 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp488 at (1853,1-1853,39) */
uint64_t __tmp_in_tmp488;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp488;
}
tmp488[i0] = (role == CLIENT) ? __tmp_in_tmp488 : 0;
}

auto tmp489 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp489 at (1856,1-1856,39) */
uint64_t __tmp_in_tmp489;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp489;
}
tmp489[i0] = (role == CLIENT) ? __tmp_in_tmp489 : 0;
}

auto tmp490 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp490 at (1859,1-1859,50) */
uint64_t __tmp_in_tmp490;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp490;
}
tmp490[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp490 : 0;
}
}
}
}

auto tmp491 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp491 at (1862,1-1862,38) */
uint64_t __tmp_in_tmp491;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp491;
}
tmp491[i0] = (role == CLIENT) ? __tmp_in_tmp491 : 0;
}

auto tmp492 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp492 at (1865,1-1865,38) */
uint64_t __tmp_in_tmp492;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp492;
}
tmp492[i0] = (role == CLIENT) ? __tmp_in_tmp492 : 0;
}

auto tmp493 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp493 at (1868,1-1868,38) */
uint64_t __tmp_in_tmp493;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp493;
}
tmp493[i0] = (role == CLIENT) ? __tmp_in_tmp493 : 0;
}

auto tmp494 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp494 at (1871,1-1871,38) */
uint64_t __tmp_in_tmp494;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp494;
}
tmp494[i0] = (role == CLIENT) ? __tmp_in_tmp494 : 0;
}

auto tmp495 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp495 at (1874,1-1874,49) */
uint64_t __tmp_in_tmp495;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp495;
}
tmp495[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp495 : 0;
}
}
}
}

auto tmp496 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp496 at (1877,1-1877,38) */
uint64_t __tmp_in_tmp496;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp496;
}
tmp496[i0] = (role == CLIENT) ? __tmp_in_tmp496 : 0;
}

auto tmp497 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp497 at (1880,1-1880,38) */
uint64_t __tmp_in_tmp497;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp497;
}
tmp497[i0] = (role == CLIENT) ? __tmp_in_tmp497 : 0;
}

auto tmp498 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp498 at (1883,1-1883,38) */
uint64_t __tmp_in_tmp498;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp498;
}
tmp498[i0] = (role == CLIENT) ? __tmp_in_tmp498 : 0;
}

auto tmp499 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp499 at (1886,1-1886,38) */
uint64_t __tmp_in_tmp499;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp499;
}
tmp499[i0] = (role == CLIENT) ? __tmp_in_tmp499 : 0;
}

auto tmp500 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp500 at (1889,1-1889,50) */
uint64_t __tmp_in_tmp500;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp500;
}
tmp500[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp500 : 0;
}
}
}
}

auto tmp501 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp501 at (1892,1-1892,39) */
uint64_t __tmp_in_tmp501;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp501;
}
tmp501[i0] = (role == CLIENT) ? __tmp_in_tmp501 : 0;
}

auto tmp502 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp502 at (1895,1-1895,39) */
uint64_t __tmp_in_tmp502;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp502;
}
tmp502[i0] = (role == CLIENT) ? __tmp_in_tmp502 : 0;
}

auto tmp503 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp503 at (1898,1-1898,39) */
uint64_t __tmp_in_tmp503;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp503;
}
tmp503[i0] = (role == CLIENT) ? __tmp_in_tmp503 : 0;
}

auto tmp504 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp504 at (1901,1-1901,39) */
uint64_t __tmp_in_tmp504;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp504;
}
tmp504[i0] = (role == CLIENT) ? __tmp_in_tmp504 : 0;
}

auto tmp505 = make_vector<uint64_t>( (int32_t)2048,  (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp505 at (1904,1-1904,45) */
uint64_t __tmp_in_tmp505;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1001; i1++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp505;
}
tmp505[i0][i1] = (role == CLIENT) ? __tmp_in_tmp505 : 0;
}
}

auto tmp506 = make_vector<uint64_t>( (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp506 at (1907,1-1907,39) */
uint64_t __tmp_in_tmp506;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1001; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp506;
}
tmp506[i0] = (role == CLIENT) ? __tmp_in_tmp506 : 0;
}


//Main Point

leave_time();
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("Starting main protocol");
start_m();
touch_time();
auto tmp507 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp507[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp507[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp507[ (int64_t)1][ (int64_t)0] =  (int32_t)3;
tmp507[ (int64_t)1][ (int64_t)1] =  (int32_t)3;
tmp507[ (int64_t)2][ (int64_t)0] =  (int32_t)3;
tmp507[ (int64_t)2][ (int64_t)1] =  (int32_t)3;
tmp507[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp507[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp508 = make_vector<uint64_t>( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3);
Pad442( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0,  (int32_t)4,  (int32_t)2, tmp507, tmp508);
ClearMemSecret4( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp507);

auto tmp511 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp508, tmp1,  (int32_t)12, tmp511);
ClearMemSecret4( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64, tmp1);
ClearMemSecret4( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3, tmp508);

auto tmp514 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)1,  (int32_t)0,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp511, tmp514);
ClearMemSecret4( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp511);

auto tmp516 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp514, tmp2, tmp3,  (int32_t)12, tmp516);
ClearMemSecret1( (int32_t)64, tmp2);
ClearMemSecret1( (int32_t)64, tmp3);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp514);

auto tmp520 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp516, tmp520);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp516);

auto tmp522 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp520, tmp6,  (int32_t)12, tmp522);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp6);

auto tmp524 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp520, tmp7,  (int32_t)12, tmp524);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64, tmp7);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp520);

auto tmp527 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp524, tmp8, tmp9,  (int32_t)12, tmp527);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp524);
ClearMemSecret1( (int32_t)64, tmp8);
ClearMemSecret1( (int32_t)64, tmp9);

auto tmp531 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp527, tmp531);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp527);

auto tmp533 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp531, tmp12,  (int32_t)12, tmp533);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp12);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp531);

auto tmp536 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp533, tmp13, tmp14,  (int32_t)12, tmp536);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp533);
ClearMemSecret1( (int32_t)64, tmp14);
ClearMemSecret1( (int32_t)64, tmp13);

auto tmp540 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp536, tmp540);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp536);

auto tmp542 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp540, tmp17,  (int32_t)12, tmp542);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp540);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp17);

auto tmp545 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp542, tmp522, tmp545);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp522);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp542);

auto tmp548 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp545, tmp18, tmp19,  (int32_t)12, tmp548);
ClearMemSecret1( (int32_t)256, tmp19);
ClearMemSecret1( (int32_t)256, tmp18);

auto tmp551 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp548, tmp551);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp548);

auto tmp553 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp551, tmp22,  (int32_t)12, tmp553);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp22);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp551);

auto tmp556 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp553, tmp23, tmp24,  (int32_t)12, tmp556);
ClearMemSecret1( (int32_t)64, tmp23);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp553);
ClearMemSecret1( (int32_t)64, tmp24);

auto tmp560 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp556, tmp560);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp556);

auto tmp562 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp560, tmp27,  (int32_t)12, tmp562);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp27);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp560);

auto tmp565 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp562, tmp28, tmp29,  (int32_t)12, tmp565);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp562);
ClearMemSecret1( (int32_t)64, tmp29);
ClearMemSecret1( (int32_t)64, tmp28);

auto tmp569 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp565, tmp569);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp565);

auto tmp571 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp569, tmp32,  (int32_t)12, tmp571);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp32);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp569);

auto tmp574 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp571, tmp545, tmp574);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp571);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp545);

auto tmp577 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp574, tmp33, tmp34,  (int32_t)12, tmp577);
ClearMemSecret1( (int32_t)256, tmp34);
ClearMemSecret1( (int32_t)256, tmp33);

auto tmp580 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp577, tmp580);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp577);

auto tmp582 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp580, tmp37,  (int32_t)12, tmp582);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp37);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp580);

auto tmp585 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp582, tmp38, tmp39,  (int32_t)12, tmp585);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp582);
ClearMemSecret1( (int32_t)64, tmp39);
ClearMemSecret1( (int32_t)64, tmp38);

auto tmp589 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp585, tmp589);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp585);

auto tmp591 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp589, tmp42,  (int32_t)12, tmp591);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp42);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp589);

auto tmp594 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp591, tmp43, tmp44,  (int32_t)12, tmp594);
ClearMemSecret1( (int32_t)64, tmp44);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp591);
ClearMemSecret1( (int32_t)64, tmp43);

auto tmp598 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp594, tmp598);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp594);

auto tmp600 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp598, tmp47,  (int32_t)12, tmp600);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp598);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp47);

auto tmp603 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp600, tmp574, tmp603);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp574);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp600);

auto tmp606 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp603, tmp48, tmp49,  (int32_t)12, tmp606);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp603);
ClearMemSecret1( (int32_t)256, tmp48);
ClearMemSecret1( (int32_t)256, tmp49);

auto tmp610 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp606, tmp610);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp606);

auto tmp612 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp612[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp612[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp612[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp612[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp612[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp612[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp612[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp612[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp613 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp610,  (int32_t)4,  (int32_t)2, tmp612, tmp613);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp612);

auto tmp615 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp613, tmp52,  (int32_t)12, tmp615);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512, tmp52);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp613);

auto tmp618 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp610, tmp53,  (int32_t)12, tmp618);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp610);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128, tmp53);

auto tmp621 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp618, tmp54, tmp55,  (int32_t)12, tmp621);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp618);
ClearMemSecret1( (int32_t)128, tmp55);
ClearMemSecret1( (int32_t)128, tmp54);

auto tmp625 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp621, tmp625);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp621);

auto tmp627 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp627[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp627[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp627[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp627[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp627[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp627[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp627[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp627[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp628 = make_vector<uint64_t>( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128);
Pad442( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp625,  (int32_t)4,  (int32_t)2, tmp627, tmp628);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp627);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp625);

auto tmp631 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp628, tmp58,  (int32_t)12, tmp631);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp58);
ClearMemSecret4( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128, tmp628);

auto tmp634 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp631, tmp59, tmp60,  (int32_t)12, tmp634);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp631);
ClearMemSecret1( (int32_t)128, tmp59);
ClearMemSecret1( (int32_t)128, tmp60);

auto tmp638 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp634, tmp638);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp634);

auto tmp640 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp638, tmp63,  (int32_t)12, tmp640);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp638);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp63);

auto tmp643 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp640, tmp615, tmp643);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp615);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp640);

auto tmp646 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp643, tmp64, tmp65,  (int32_t)12, tmp646);
ClearMemSecret1( (int32_t)512, tmp64);
ClearMemSecret1( (int32_t)512, tmp65);

auto tmp649 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp646, tmp649);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp646);

auto tmp651 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp649, tmp68,  (int32_t)12, tmp651);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp649);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp68);

auto tmp654 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp651, tmp69, tmp70,  (int32_t)12, tmp654);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp651);
ClearMemSecret1( (int32_t)128, tmp70);
ClearMemSecret1( (int32_t)128, tmp69);

auto tmp658 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp654, tmp658);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp654);

auto tmp660 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp658, tmp73,  (int32_t)12, tmp660);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp73);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp658);

auto tmp663 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp660, tmp74, tmp75,  (int32_t)12, tmp663);
ClearMemSecret1( (int32_t)128, tmp75);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp660);
ClearMemSecret1( (int32_t)128, tmp74);

auto tmp667 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp663, tmp667);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp663);

auto tmp669 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp667, tmp78,  (int32_t)12, tmp669);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp667);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp78);

auto tmp672 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp669, tmp643, tmp672);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp643);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp669);

auto tmp675 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp672, tmp79, tmp80,  (int32_t)12, tmp675);
ClearMemSecret1( (int32_t)512, tmp79);
ClearMemSecret1( (int32_t)512, tmp80);

auto tmp678 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp675, tmp678);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp675);

auto tmp680 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp678, tmp83,  (int32_t)12, tmp680);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp83);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp678);

auto tmp683 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp680, tmp84, tmp85,  (int32_t)12, tmp683);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp680);
ClearMemSecret1( (int32_t)128, tmp85);
ClearMemSecret1( (int32_t)128, tmp84);

auto tmp687 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp683, tmp687);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp683);

auto tmp689 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp687, tmp88,  (int32_t)12, tmp689);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp88);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp687);

auto tmp692 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp689, tmp89, tmp90,  (int32_t)12, tmp692);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp689);
ClearMemSecret1( (int32_t)128, tmp90);
ClearMemSecret1( (int32_t)128, tmp89);

auto tmp696 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp692, tmp696);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp692);

auto tmp698 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp696, tmp93,  (int32_t)12, tmp698);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp696);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp93);

auto tmp701 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp698, tmp672, tmp701);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp672);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp698);

auto tmp704 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp701, tmp94, tmp95,  (int32_t)12, tmp704);
ClearMemSecret1( (int32_t)512, tmp94);
ClearMemSecret1( (int32_t)512, tmp95);

auto tmp707 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp704, tmp707);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp704);

auto tmp709 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp707, tmp98,  (int32_t)12, tmp709);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp98);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp707);

auto tmp712 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp709, tmp99, tmp100,  (int32_t)12, tmp712);
ClearMemSecret1( (int32_t)128, tmp100);
ClearMemSecret1( (int32_t)128, tmp99);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp709);

auto tmp716 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp712, tmp716);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp712);

auto tmp718 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp716, tmp103,  (int32_t)12, tmp718);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp716);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp103);

auto tmp721 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp718, tmp104, tmp105,  (int32_t)12, tmp721);
ClearMemSecret1( (int32_t)128, tmp104);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp718);
ClearMemSecret1( (int32_t)128, tmp105);

auto tmp725 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp721, tmp725);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp721);

auto tmp727 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp725, tmp108,  (int32_t)12, tmp727);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp108);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp725);

auto tmp730 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp727, tmp701, tmp730);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp701);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp727);

auto tmp733 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp730, tmp109, tmp110,  (int32_t)12, tmp733);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp730);
ClearMemSecret1( (int32_t)512, tmp110);
ClearMemSecret1( (int32_t)512, tmp109);

auto tmp737 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp733, tmp737);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp733);

auto tmp739 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp739[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp739[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp739[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp739[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp739[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp739[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp739[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp739[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp740 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Pad442( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp737,  (int32_t)4,  (int32_t)2, tmp739, tmp740);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp739);

auto tmp742 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp740, tmp113,  (int32_t)12, tmp742);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024, tmp113);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp740);

auto tmp745 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp737, tmp114,  (int32_t)12, tmp745);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp737);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256, tmp114);

auto tmp748 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp745, tmp115, tmp116,  (int32_t)12, tmp748);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp745);
ClearMemSecret1( (int32_t)256, tmp116);
ClearMemSecret1( (int32_t)256, tmp115);

auto tmp752 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp748, tmp752);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp748);

auto tmp754 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp754[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp754[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp754[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp754[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp754[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp754[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp754[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp754[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp755 = make_vector<uint64_t>( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp752,  (int32_t)4,  (int32_t)2, tmp754, tmp755);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp754);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp752);

auto tmp758 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp755, tmp119,  (int32_t)12, tmp758);
ClearMemSecret4( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256, tmp755);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp119);

auto tmp761 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp758, tmp120, tmp121,  (int32_t)12, tmp761);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp758);
ClearMemSecret1( (int32_t)256, tmp120);
ClearMemSecret1( (int32_t)256, tmp121);

auto tmp765 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp761, tmp765);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp761);

auto tmp767 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp765, tmp124,  (int32_t)12, tmp767);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp124);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp765);

auto tmp770 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp767, tmp742, tmp770);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp742);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp767);

auto tmp773 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp770, tmp125, tmp126,  (int32_t)12, tmp773);
ClearMemSecret1( (int32_t)1024, tmp125);
ClearMemSecret1( (int32_t)1024, tmp126);

auto tmp776 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp773, tmp776);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp773);

auto tmp778 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp776, tmp129,  (int32_t)12, tmp778);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp129);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp776);

auto tmp781 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp778, tmp130, tmp131,  (int32_t)12, tmp781);
ClearMemSecret1( (int32_t)256, tmp130);
ClearMemSecret1( (int32_t)256, tmp131);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp778);

auto tmp785 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp781, tmp785);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp781);

auto tmp787 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp785, tmp134,  (int32_t)12, tmp787);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp134);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp785);

auto tmp790 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp787, tmp135, tmp136,  (int32_t)12, tmp790);
ClearMemSecret1( (int32_t)256, tmp135);
ClearMemSecret1( (int32_t)256, tmp136);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp787);

auto tmp794 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp790, tmp794);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp790);

auto tmp796 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp794, tmp139,  (int32_t)12, tmp796);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp794);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp139);

auto tmp799 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp796, tmp770, tmp799);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp796);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp770);

auto tmp802 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp799, tmp140, tmp141,  (int32_t)12, tmp802);
ClearMemSecret1( (int32_t)1024, tmp140);
ClearMemSecret1( (int32_t)1024, tmp141);

auto tmp805 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp802, tmp805);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp802);

auto tmp807 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp805, tmp144,  (int32_t)12, tmp807);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp144);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp805);

auto tmp810 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp807, tmp145, tmp146,  (int32_t)12, tmp810);
ClearMemSecret1( (int32_t)256, tmp145);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp807);
ClearMemSecret1( (int32_t)256, tmp146);

auto tmp814 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp810, tmp814);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp810);

auto tmp816 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp814, tmp149,  (int32_t)12, tmp816);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp814);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp149);

auto tmp819 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp816, tmp150, tmp151,  (int32_t)12, tmp819);
ClearMemSecret1( (int32_t)256, tmp151);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp816);
ClearMemSecret1( (int32_t)256, tmp150);

auto tmp823 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp819, tmp823);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp819);

auto tmp825 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp823, tmp154,  (int32_t)12, tmp825);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp823);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp154);

auto tmp828 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp825, tmp799, tmp828);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp825);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp799);

auto tmp831 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp828, tmp155, tmp156,  (int32_t)12, tmp831);
ClearMemSecret1( (int32_t)1024, tmp156);
ClearMemSecret1( (int32_t)1024, tmp155);

auto tmp834 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp831, tmp834);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp831);

auto tmp836 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp834, tmp159,  (int32_t)12, tmp836);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp834);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp159);

auto tmp839 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp836, tmp160, tmp161,  (int32_t)12, tmp839);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp836);
ClearMemSecret1( (int32_t)256, tmp160);
ClearMemSecret1( (int32_t)256, tmp161);

auto tmp843 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp839, tmp843);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp839);

auto tmp845 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp843, tmp164,  (int32_t)12, tmp845);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp843);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp164);

auto tmp848 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp845, tmp165, tmp166,  (int32_t)12, tmp848);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp845);
ClearMemSecret1( (int32_t)256, tmp166);
ClearMemSecret1( (int32_t)256, tmp165);

auto tmp852 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp848, tmp852);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp848);

auto tmp854 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp852, tmp169,  (int32_t)12, tmp854);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp169);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp852);

auto tmp857 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp854, tmp828, tmp857);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp854);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp828);

auto tmp860 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp857, tmp170, tmp171,  (int32_t)12, tmp860);
ClearMemSecret1( (int32_t)1024, tmp170);
ClearMemSecret1( (int32_t)1024, tmp171);

auto tmp863 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp860, tmp863);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp860);

auto tmp865 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp863, tmp174,  (int32_t)12, tmp865);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp863);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp174);

auto tmp868 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp865, tmp175, tmp176,  (int32_t)12, tmp868);
ClearMemSecret1( (int32_t)256, tmp176);
ClearMemSecret1( (int32_t)256, tmp175);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp865);

auto tmp872 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp868, tmp872);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp868);

auto tmp874 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp872, tmp179,  (int32_t)12, tmp874);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp872);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp179);

auto tmp877 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp874, tmp180, tmp181,  (int32_t)12, tmp877);
ClearMemSecret1( (int32_t)256, tmp180);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp874);
ClearMemSecret1( (int32_t)256, tmp181);

auto tmp881 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp877, tmp881);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp877);

auto tmp883 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp881, tmp184,  (int32_t)12, tmp883);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp881);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp184);

auto tmp886 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp883, tmp857, tmp886);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp883);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp857);

auto tmp889 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp886, tmp185, tmp186,  (int32_t)12, tmp889);
ClearMemSecret1( (int32_t)1024, tmp186);
ClearMemSecret1( (int32_t)1024, tmp185);

auto tmp892 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp889, tmp892);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp889);

auto tmp894 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp892, tmp189,  (int32_t)12, tmp894);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp892);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp189);

auto tmp897 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp894, tmp190, tmp191,  (int32_t)12, tmp897);
ClearMemSecret1( (int32_t)256, tmp190);
ClearMemSecret1( (int32_t)256, tmp191);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp894);

auto tmp901 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp897, tmp901);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp897);

auto tmp903 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp901, tmp194,  (int32_t)12, tmp903);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp901);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp194);

auto tmp906 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp903, tmp195, tmp196,  (int32_t)12, tmp906);
ClearMemSecret1( (int32_t)256, tmp196);
ClearMemSecret1( (int32_t)256, tmp195);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp903);

auto tmp910 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp906, tmp910);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp906);

auto tmp912 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp910, tmp199,  (int32_t)12, tmp912);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp910);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp199);

auto tmp915 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp912, tmp886, tmp915);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp912);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp886);

auto tmp918 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp915, tmp200, tmp201,  (int32_t)12, tmp918);
ClearMemSecret1( (int32_t)1024, tmp201);
ClearMemSecret1( (int32_t)1024, tmp200);

auto tmp921 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp918, tmp921);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp918);

auto tmp923 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp921, tmp204,  (int32_t)12, tmp923);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp921);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp204);

auto tmp926 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp923, tmp205, tmp206,  (int32_t)12, tmp926);
ClearMemSecret1( (int32_t)256, tmp206);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp923);
ClearMemSecret1( (int32_t)256, tmp205);

auto tmp930 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp926, tmp930);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp926);

auto tmp932 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp930, tmp209,  (int32_t)12, tmp932);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp930);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp209);

auto tmp935 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp932, tmp210, tmp211,  (int32_t)12, tmp935);
ClearMemSecret1( (int32_t)256, tmp211);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp932);
ClearMemSecret1( (int32_t)256, tmp210);

auto tmp939 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp935, tmp939);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp935);

auto tmp941 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp939, tmp214,  (int32_t)12, tmp941);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp214);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp939);

auto tmp944 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp941, tmp915, tmp944);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp941);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp915);

auto tmp947 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp944, tmp215, tmp216,  (int32_t)12, tmp947);
ClearMemSecret1( (int32_t)1024, tmp215);
ClearMemSecret1( (int32_t)1024, tmp216);

auto tmp950 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp947, tmp950);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp947);

auto tmp952 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp950, tmp219,  (int32_t)12, tmp952);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp219);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp950);

auto tmp955 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp952, tmp220, tmp221,  (int32_t)12, tmp955);
ClearMemSecret1( (int32_t)256, tmp220);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp952);
ClearMemSecret1( (int32_t)256, tmp221);

auto tmp959 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp955, tmp959);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp955);

auto tmp961 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp959, tmp224,  (int32_t)12, tmp961);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp959);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp224);

auto tmp964 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp961, tmp225, tmp226,  (int32_t)12, tmp964);
ClearMemSecret1( (int32_t)256, tmp226);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp961);
ClearMemSecret1( (int32_t)256, tmp225);

auto tmp968 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp964, tmp968);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp964);

auto tmp970 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp968, tmp229,  (int32_t)12, tmp970);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp968);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp229);

auto tmp973 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp970, tmp944, tmp973);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp944);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp970);

auto tmp976 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp973, tmp230, tmp231,  (int32_t)12, tmp976);
ClearMemSecret1( (int32_t)1024, tmp231);
ClearMemSecret1( (int32_t)1024, tmp230);

auto tmp979 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp976, tmp979);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp976);

auto tmp981 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp979, tmp234,  (int32_t)12, tmp981);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp234);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp979);

auto tmp984 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp981, tmp235, tmp236,  (int32_t)12, tmp984);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp981);
ClearMemSecret1( (int32_t)256, tmp235);
ClearMemSecret1( (int32_t)256, tmp236);

auto tmp988 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp984, tmp988);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp984);

auto tmp990 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp988, tmp239,  (int32_t)12, tmp990);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp239);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp988);

auto tmp993 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp990, tmp240, tmp241,  (int32_t)12, tmp993);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp990);
ClearMemSecret1( (int32_t)256, tmp240);
ClearMemSecret1( (int32_t)256, tmp241);

auto tmp997 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp993, tmp997);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp993);

auto tmp999 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp997, tmp244,  (int32_t)12, tmp999);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp997);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp244);

auto tmp1002 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp999, tmp973, tmp1002);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp999);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp973);

auto tmp1005 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1002, tmp245, tmp246,  (int32_t)12, tmp1005);
ClearMemSecret1( (int32_t)1024, tmp245);
ClearMemSecret1( (int32_t)1024, tmp246);

auto tmp1008 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1005, tmp1008);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1005);

auto tmp1010 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1008, tmp249,  (int32_t)12, tmp1010);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp249);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1008);

auto tmp1013 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1010, tmp250, tmp251,  (int32_t)12, tmp1013);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1010);
ClearMemSecret1( (int32_t)256, tmp251);
ClearMemSecret1( (int32_t)256, tmp250);

auto tmp1017 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1013, tmp1017);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1013);

auto tmp1019 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1017, tmp254,  (int32_t)12, tmp1019);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1017);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp254);

auto tmp1022 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1019, tmp255, tmp256,  (int32_t)12, tmp1022);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1019);
ClearMemSecret1( (int32_t)256, tmp256);
ClearMemSecret1( (int32_t)256, tmp255);

auto tmp1026 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1022, tmp1026);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1022);

auto tmp1028 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1026, tmp259,  (int32_t)12, tmp1028);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp259);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1026);

auto tmp1031 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1028, tmp1002, tmp1031);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1028);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1002);

auto tmp1034 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1031, tmp260, tmp261,  (int32_t)12, tmp1034);
ClearMemSecret1( (int32_t)1024, tmp260);
ClearMemSecret1( (int32_t)1024, tmp261);

auto tmp1037 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1034, tmp1037);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1034);

auto tmp1039 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1037, tmp264,  (int32_t)12, tmp1039);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1037);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp264);

auto tmp1042 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1039, tmp265, tmp266,  (int32_t)12, tmp1042);
ClearMemSecret1( (int32_t)256, tmp266);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1039);
ClearMemSecret1( (int32_t)256, tmp265);

auto tmp1046 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1042, tmp1046);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1042);

auto tmp1048 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1046, tmp269,  (int32_t)12, tmp1048);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1046);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp269);

auto tmp1051 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1048, tmp270, tmp271,  (int32_t)12, tmp1051);
ClearMemSecret1( (int32_t)256, tmp271);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1048);
ClearMemSecret1( (int32_t)256, tmp270);

auto tmp1055 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1051, tmp1055);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1051);

auto tmp1057 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1055, tmp274,  (int32_t)12, tmp1057);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1055);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp274);

auto tmp1060 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1057, tmp1031, tmp1060);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1057);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1031);

auto tmp1063 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1060, tmp275, tmp276,  (int32_t)12, tmp1063);
ClearMemSecret1( (int32_t)1024, tmp276);
ClearMemSecret1( (int32_t)1024, tmp275);

auto tmp1066 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1063, tmp1066);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1063);

auto tmp1068 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1066, tmp279,  (int32_t)12, tmp1068);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1066);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp279);

auto tmp1071 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1068, tmp280, tmp281,  (int32_t)12, tmp1071);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1068);
ClearMemSecret1( (int32_t)256, tmp281);
ClearMemSecret1( (int32_t)256, tmp280);

auto tmp1075 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1071, tmp1075);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1071);

auto tmp1077 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1075, tmp284,  (int32_t)12, tmp1077);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1075);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp284);

auto tmp1080 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1077, tmp285, tmp286,  (int32_t)12, tmp1080);
ClearMemSecret1( (int32_t)256, tmp285);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1077);
ClearMemSecret1( (int32_t)256, tmp286);

auto tmp1084 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1080, tmp1084);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1080);

auto tmp1086 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1084, tmp289,  (int32_t)12, tmp1086);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp289);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1084);

auto tmp1089 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1086, tmp1060, tmp1089);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1060);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1086);

auto tmp1092 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1089, tmp290, tmp291,  (int32_t)12, tmp1092);
ClearMemSecret1( (int32_t)1024, tmp291);
ClearMemSecret1( (int32_t)1024, tmp290);

auto tmp1095 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1092, tmp1095);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1092);

auto tmp1097 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1095, tmp294,  (int32_t)12, tmp1097);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp294);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1095);

auto tmp1100 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1097, tmp295, tmp296,  (int32_t)12, tmp1100);
ClearMemSecret1( (int32_t)256, tmp295);
ClearMemSecret1( (int32_t)256, tmp296);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1097);

auto tmp1104 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1100, tmp1104);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1100);

auto tmp1106 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1104, tmp299,  (int32_t)12, tmp1106);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp299);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1104);

auto tmp1109 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1106, tmp300, tmp301,  (int32_t)12, tmp1109);
ClearMemSecret1( (int32_t)256, tmp300);
ClearMemSecret1( (int32_t)256, tmp301);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1106);

auto tmp1113 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1109, tmp1113);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1109);

auto tmp1115 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1113, tmp304,  (int32_t)12, tmp1115);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp304);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1113);

auto tmp1118 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1115, tmp1089, tmp1118);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1089);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1115);

auto tmp1121 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1118, tmp305, tmp306,  (int32_t)12, tmp1121);
ClearMemSecret1( (int32_t)1024, tmp306);
ClearMemSecret1( (int32_t)1024, tmp305);

auto tmp1124 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1121, tmp1124);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1121);

auto tmp1126 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1124, tmp309,  (int32_t)12, tmp1126);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1124);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp309);

auto tmp1129 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1126, tmp310, tmp311,  (int32_t)12, tmp1129);
ClearMemSecret1( (int32_t)256, tmp310);
ClearMemSecret1( (int32_t)256, tmp311);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1126);

auto tmp1133 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1129, tmp1133);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1129);

auto tmp1135 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1133, tmp314,  (int32_t)12, tmp1135);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp314);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1133);

auto tmp1138 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1135, tmp315, tmp316,  (int32_t)12, tmp1138);
ClearMemSecret1( (int32_t)256, tmp316);
ClearMemSecret1( (int32_t)256, tmp315);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1135);

auto tmp1142 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1138, tmp1142);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1138);

auto tmp1144 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1142, tmp319,  (int32_t)12, tmp1144);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1142);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp319);

auto tmp1147 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1144, tmp1118, tmp1147);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1118);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1144);

auto tmp1150 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1147, tmp320, tmp321,  (int32_t)12, tmp1150);
ClearMemSecret1( (int32_t)1024, tmp321);
ClearMemSecret1( (int32_t)1024, tmp320);

auto tmp1153 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1150, tmp1153);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1150);

auto tmp1155 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1153, tmp324,  (int32_t)12, tmp1155);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp324);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1153);

auto tmp1158 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1155, tmp325, tmp326,  (int32_t)12, tmp1158);
ClearMemSecret1( (int32_t)256, tmp325);
ClearMemSecret1( (int32_t)256, tmp326);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1155);

auto tmp1162 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1158, tmp1162);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1158);

auto tmp1164 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1162, tmp329,  (int32_t)12, tmp1164);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp329);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1162);

auto tmp1167 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1164, tmp330, tmp331,  (int32_t)12, tmp1167);
ClearMemSecret1( (int32_t)256, tmp331);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1164);
ClearMemSecret1( (int32_t)256, tmp330);

auto tmp1171 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1167, tmp1171);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1167);

auto tmp1173 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1171, tmp334,  (int32_t)12, tmp1173);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1171);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp334);

auto tmp1176 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1173, tmp1147, tmp1176);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1173);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1147);

auto tmp1179 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1176, tmp335, tmp336,  (int32_t)12, tmp1179);
ClearMemSecret1( (int32_t)1024, tmp336);
ClearMemSecret1( (int32_t)1024, tmp335);

auto tmp1182 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1179, tmp1182);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1179);

auto tmp1184 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1182, tmp339,  (int32_t)12, tmp1184);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp339);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1182);

auto tmp1187 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1184, tmp340, tmp341,  (int32_t)12, tmp1187);
ClearMemSecret1( (int32_t)256, tmp341);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1184);
ClearMemSecret1( (int32_t)256, tmp340);

auto tmp1191 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1187, tmp1191);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1187);

auto tmp1193 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1191, tmp344,  (int32_t)12, tmp1193);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1191);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp344);

auto tmp1196 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1193, tmp345, tmp346,  (int32_t)12, tmp1196);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1193);
ClearMemSecret1( (int32_t)256, tmp345);
ClearMemSecret1( (int32_t)256, tmp346);

auto tmp1200 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1196, tmp1200);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1196);

auto tmp1202 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1200, tmp349,  (int32_t)12, tmp1202);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1200);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp349);

auto tmp1205 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1202, tmp1176, tmp1205);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1202);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1176);

auto tmp1208 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1205, tmp350, tmp351,  (int32_t)12, tmp1208);
ClearMemSecret1( (int32_t)1024, tmp350);
ClearMemSecret1( (int32_t)1024, tmp351);

auto tmp1211 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1208, tmp1211);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1208);

auto tmp1213 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1211, tmp354,  (int32_t)12, tmp1213);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1211);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp354);

auto tmp1216 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1213, tmp355, tmp356,  (int32_t)12, tmp1216);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1213);
ClearMemSecret1( (int32_t)256, tmp356);
ClearMemSecret1( (int32_t)256, tmp355);

auto tmp1220 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1216, tmp1220);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1216);

auto tmp1222 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1220, tmp359,  (int32_t)12, tmp1222);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp359);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1220);

auto tmp1225 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1222, tmp360, tmp361,  (int32_t)12, tmp1225);
ClearMemSecret1( (int32_t)256, tmp360);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1222);
ClearMemSecret1( (int32_t)256, tmp361);

auto tmp1229 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1225, tmp1229);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1225);

auto tmp1231 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1229, tmp364,  (int32_t)12, tmp1231);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp364);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1229);

auto tmp1234 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1231, tmp1205, tmp1234);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1231);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1205);

auto tmp1237 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1234, tmp365, tmp366,  (int32_t)12, tmp1237);
ClearMemSecret1( (int32_t)1024, tmp365);
ClearMemSecret1( (int32_t)1024, tmp366);

auto tmp1240 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1237, tmp1240);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1237);

auto tmp1242 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1240, tmp369,  (int32_t)12, tmp1242);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1240);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp369);

auto tmp1245 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1242, tmp370, tmp371,  (int32_t)12, tmp1245);
ClearMemSecret1( (int32_t)256, tmp371);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1242);
ClearMemSecret1( (int32_t)256, tmp370);

auto tmp1249 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1245, tmp1249);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1245);

auto tmp1251 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1249, tmp374,  (int32_t)12, tmp1251);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1249);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp374);

auto tmp1254 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1251, tmp375, tmp376,  (int32_t)12, tmp1254);
ClearMemSecret1( (int32_t)256, tmp376);
ClearMemSecret1( (int32_t)256, tmp375);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1251);

auto tmp1258 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1254, tmp1258);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1254);

auto tmp1260 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1258, tmp379,  (int32_t)12, tmp1260);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1258);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp379);

auto tmp1263 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1260, tmp1234, tmp1263);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1260);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1234);

auto tmp1266 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1263, tmp380, tmp381,  (int32_t)12, tmp1266);
ClearMemSecret1( (int32_t)1024, tmp381);
ClearMemSecret1( (int32_t)1024, tmp380);

auto tmp1269 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1266, tmp1269);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1266);

auto tmp1271 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1269, tmp384,  (int32_t)12, tmp1271);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1269);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp384);

auto tmp1274 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1271, tmp385, tmp386,  (int32_t)12, tmp1274);
ClearMemSecret1( (int32_t)256, tmp386);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1271);
ClearMemSecret1( (int32_t)256, tmp385);

auto tmp1278 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1274, tmp1278);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1274);

auto tmp1280 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1278, tmp389,  (int32_t)12, tmp1280);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1278);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp389);

auto tmp1283 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1280, tmp390, tmp391,  (int32_t)12, tmp1283);
ClearMemSecret1( (int32_t)256, tmp391);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1280);
ClearMemSecret1( (int32_t)256, tmp390);

auto tmp1287 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1283, tmp1287);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1283);

auto tmp1289 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1287, tmp394,  (int32_t)12, tmp1289);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1287);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp394);

auto tmp1292 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1289, tmp1263, tmp1292);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1289);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1263);

auto tmp1295 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1292, tmp395, tmp396,  (int32_t)12, tmp1295);
ClearMemSecret1( (int32_t)1024, tmp396);
ClearMemSecret1( (int32_t)1024, tmp395);

auto tmp1298 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1295, tmp1298);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1295);

auto tmp1300 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1298, tmp399,  (int32_t)12, tmp1300);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1298);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp399);

auto tmp1303 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1300, tmp400, tmp401,  (int32_t)12, tmp1303);
ClearMemSecret1( (int32_t)256, tmp400);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1300);
ClearMemSecret1( (int32_t)256, tmp401);

auto tmp1307 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1303, tmp1307);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1303);

auto tmp1309 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1307, tmp404,  (int32_t)12, tmp1309);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp404);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1307);

auto tmp1312 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1309, tmp405, tmp406,  (int32_t)12, tmp1312);
ClearMemSecret1( (int32_t)256, tmp405);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1309);
ClearMemSecret1( (int32_t)256, tmp406);

auto tmp1316 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1312, tmp1316);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1312);

auto tmp1318 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1316, tmp409,  (int32_t)12, tmp1318);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp409);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1316);

auto tmp1321 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1318, tmp1292, tmp1321);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1292);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1318);

auto tmp1324 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1321, tmp410, tmp411,  (int32_t)12, tmp1324);
ClearMemSecret1( (int32_t)1024, tmp411);
ClearMemSecret1( (int32_t)1024, tmp410);

auto tmp1327 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1324, tmp1327);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1324);

auto tmp1329 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1327, tmp414,  (int32_t)12, tmp1329);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp414);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1327);

auto tmp1332 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1329, tmp415, tmp416,  (int32_t)12, tmp1332);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1329);
ClearMemSecret1( (int32_t)256, tmp416);
ClearMemSecret1( (int32_t)256, tmp415);

auto tmp1336 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1332, tmp1336);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1332);

auto tmp1338 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1336, tmp419,  (int32_t)12, tmp1338);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1336);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp419);

auto tmp1341 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1338, tmp420, tmp421,  (int32_t)12, tmp1341);
ClearMemSecret1( (int32_t)256, tmp420);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1338);
ClearMemSecret1( (int32_t)256, tmp421);

auto tmp1345 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1341, tmp1345);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1341);

auto tmp1347 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1345, tmp424,  (int32_t)12, tmp1347);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1345);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp424);

auto tmp1350 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1347, tmp1321, tmp1350);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1321);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1347);

auto tmp1353 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1350, tmp425, tmp426,  (int32_t)12, tmp1353);
ClearMemSecret1( (int32_t)1024, tmp425);
ClearMemSecret1( (int32_t)1024, tmp426);

auto tmp1356 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1353, tmp1356);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1353);

auto tmp1358 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1356, tmp429,  (int32_t)12, tmp1358);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1356);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp429);

auto tmp1361 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1358, tmp430, tmp431,  (int32_t)12, tmp1361);
ClearMemSecret1( (int32_t)256, tmp430);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1358);
ClearMemSecret1( (int32_t)256, tmp431);

auto tmp1365 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1361, tmp1365);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1361);

auto tmp1367 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1365, tmp434,  (int32_t)12, tmp1367);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1365);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp434);

auto tmp1370 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1367, tmp435, tmp436,  (int32_t)12, tmp1370);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1367);
ClearMemSecret1( (int32_t)256, tmp435);
ClearMemSecret1( (int32_t)256, tmp436);

auto tmp1374 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1370, tmp1374);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1370);

auto tmp1376 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1374, tmp439,  (int32_t)12, tmp1376);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp439);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1374);

auto tmp1379 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1376, tmp1350, tmp1379);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1350);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1376);

auto tmp1382 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1379, tmp440, tmp441,  (int32_t)12, tmp1382);
ClearMemSecret1( (int32_t)1024, tmp440);
ClearMemSecret1( (int32_t)1024, tmp441);

auto tmp1385 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1382, tmp1385);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1382);

auto tmp1387 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1385, tmp444,  (int32_t)12, tmp1387);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1385);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp444);

auto tmp1390 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1387, tmp445, tmp446,  (int32_t)12, tmp1390);
ClearMemSecret1( (int32_t)256, tmp446);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1387);
ClearMemSecret1( (int32_t)256, tmp445);

auto tmp1394 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1390, tmp1394);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1390);

auto tmp1396 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1394, tmp449,  (int32_t)12, tmp1396);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1394);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp449);

auto tmp1399 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1396, tmp450, tmp451,  (int32_t)12, tmp1399);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1396);
ClearMemSecret1( (int32_t)256, tmp450);
ClearMemSecret1( (int32_t)256, tmp451);

auto tmp1403 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1399, tmp1403);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1399);

auto tmp1405 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1403, tmp454,  (int32_t)12, tmp1405);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1403);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp454);

auto tmp1408 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1405, tmp1379, tmp1408);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1379);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1405);

auto tmp1411 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1408, tmp455, tmp456,  (int32_t)12, tmp1411);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1408);
ClearMemSecret1( (int32_t)1024, tmp455);
ClearMemSecret1( (int32_t)1024, tmp456);

auto tmp1415 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1411, tmp1415);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1411);

auto tmp1417 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp1417[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp1417[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp1417[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp1417[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp1417[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp1417[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp1417[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp1417[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp1418 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Pad442( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1415,  (int32_t)4,  (int32_t)2, tmp1417, tmp1418);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp1417);

auto tmp1420 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp1418, tmp459,  (int32_t)12, tmp1420);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1418);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048, tmp459);

auto tmp1423 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1415, tmp460,  (int32_t)12, tmp1423);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512, tmp460);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1415);

auto tmp1426 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1423, tmp461, tmp462,  (int32_t)12, tmp1426);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1423);
ClearMemSecret1( (int32_t)512, tmp462);
ClearMemSecret1( (int32_t)512, tmp461);

auto tmp1430 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1426, tmp1430);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1426);

auto tmp1432 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp1432[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp1432[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp1432[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp1432[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp1432[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp1432[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp1432[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp1432[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp1433 = make_vector<uint64_t>( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512);
Pad442( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1430,  (int32_t)4,  (int32_t)2, tmp1432, tmp1433);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1430);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp1432);

auto tmp1436 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp1433, tmp465,  (int32_t)12, tmp1436);
ClearMemSecret4( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512, tmp1433);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp465);

auto tmp1439 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1436, tmp466, tmp467,  (int32_t)12, tmp1439);
ClearMemSecret1( (int32_t)512, tmp467);
ClearMemSecret1( (int32_t)512, tmp466);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1436);

auto tmp1443 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1439, tmp1443);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1439);

auto tmp1445 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1443, tmp470,  (int32_t)12, tmp1445);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1443);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp470);

auto tmp1448 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1445, tmp1420, tmp1448);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1420);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1445);

auto tmp1451 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1448, tmp471, tmp472,  (int32_t)12, tmp1451);
ClearMemSecret1( (int32_t)2048, tmp471);
ClearMemSecret1( (int32_t)2048, tmp472);

auto tmp1454 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1451, tmp1454);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1451);

auto tmp1456 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1454, tmp475,  (int32_t)12, tmp1456);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp475);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1454);

auto tmp1459 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1456, tmp476, tmp477,  (int32_t)12, tmp1459);
ClearMemSecret1( (int32_t)512, tmp476);
ClearMemSecret1( (int32_t)512, tmp477);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1456);

auto tmp1463 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1459, tmp1463);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1459);

auto tmp1465 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1463, tmp480,  (int32_t)12, tmp1465);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1463);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp480);

auto tmp1468 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1465, tmp481, tmp482,  (int32_t)12, tmp1468);
ClearMemSecret1( (int32_t)512, tmp482);
ClearMemSecret1( (int32_t)512, tmp481);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1465);

auto tmp1472 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1468, tmp1472);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1468);

auto tmp1474 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1472, tmp485,  (int32_t)12, tmp1474);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1472);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp485);

auto tmp1477 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1474, tmp1448, tmp1477);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1448);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1474);

auto tmp1480 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1477, tmp486, tmp487,  (int32_t)12, tmp1480);
ClearMemSecret1( (int32_t)2048, tmp486);
ClearMemSecret1( (int32_t)2048, tmp487);

auto tmp1483 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1480, tmp1483);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1480);

auto tmp1485 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1483, tmp490,  (int32_t)12, tmp1485);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp490);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1483);

auto tmp1488 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1485, tmp491, tmp492,  (int32_t)12, tmp1488);
ClearMemSecret1( (int32_t)512, tmp492);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1485);
ClearMemSecret1( (int32_t)512, tmp491);

auto tmp1492 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1488, tmp1492);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1488);

auto tmp1494 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1492, tmp495,  (int32_t)12, tmp1494);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1492);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp495);

auto tmp1497 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1494, tmp496, tmp497,  (int32_t)12, tmp1497);
ClearMemSecret1( (int32_t)512, tmp497);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1494);
ClearMemSecret1( (int32_t)512, tmp496);

auto tmp1501 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1497, tmp1501);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1497);

auto tmp1503 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1501, tmp500,  (int32_t)12, tmp1503);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1501);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp500);

auto tmp1506 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1503, tmp1477, tmp1506);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1503);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1477);

auto tmp1509 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1506, tmp501, tmp502,  (int32_t)12, tmp1509);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1506);
ClearMemSecret1( (int32_t)2048, tmp502);
ClearMemSecret1( (int32_t)2048, tmp501);

auto tmp1513 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1509, tmp1513);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1509);

auto tmp1515 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048);
AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1513, tmp1515);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp1513);

auto tmp1517 = make_vector<uint64_t>( (int32_t)1,  (int32_t)2048);
Squeeze24( (int32_t)1,  (int32_t)2048,  (int32_t)1,  (int32_t)2,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp1515, tmp1517);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp1515);

auto tmp1519 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatMulCSF2D( (int32_t)1,  (int32_t)2048,  (int32_t)1001, tmp1517, tmp505, tmp1519,  (int64_t)12);
ClearMemSecret2( (int32_t)2048,  (int32_t)1001, tmp505);
ClearMemSecret2( (int32_t)1,  (int32_t)2048, tmp1517);

auto tmp1522 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatAddBroadCast2( (int32_t)1,  (int32_t)1001, tmp1519, tmp506, tmp1522);
ClearMemSecret1( (int32_t)1001, tmp506);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp1519);

auto tmp1525 = make_vector<uint64_t>( (int32_t)1);
ArgMax1( (int32_t)1,  (int32_t)1,  (int32_t)1001, tmp1522,  (int32_t)1, tmp1525);
ClearMemPublic( (int32_t)1);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp1522);
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
print_integer(funcReconstruct2PCCons(tmp1525[i0], 1));
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
