#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "res_net_mem_opti.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef RESNET50

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
		funcConv2DCSFSplit(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);
		//Conv2DCSFMain(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);
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
//Main Point
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

auto tmp204 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp204 at (1001,1-1001,51) */
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
/* Variable to read the clear value corresponding to the input variable tmp205 at (1004,1-1004,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp206 at (1007,1-1007,38) */
uint64_t __tmp_in_tmp206;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp206;
}
tmp206[i0] = (role == CLIENT) ? __tmp_in_tmp206 : 0;
}

auto tmp207 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp207 at (1010,1-1010,38) */
uint64_t __tmp_in_tmp207;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp207;
}
tmp207[i0] = (role == CLIENT) ? __tmp_in_tmp207 : 0;
}

auto tmp208 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp208 at (1013,1-1013,38) */
uint64_t __tmp_in_tmp208;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp208;
}
tmp208[i0] = (role == CLIENT) ? __tmp_in_tmp208 : 0;
}

auto tmp209 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp209 at (1016,1-1016,38) */
uint64_t __tmp_in_tmp209;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp209;
}
tmp209[i0] = (role == CLIENT) ? __tmp_in_tmp209 : 0;
}

auto tmp210 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp210 at (1019,1-1019,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp211 at (1022,1-1022,38) */
uint64_t __tmp_in_tmp211;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp211;
}
tmp211[i0] = (role == CLIENT) ? __tmp_in_tmp211 : 0;
}

auto tmp212 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp212 at (1025,1-1025,38) */
uint64_t __tmp_in_tmp212;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp212;
}
tmp212[i0] = (role == CLIENT) ? __tmp_in_tmp212 : 0;
}

auto tmp213 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp213 at (1028,1-1028,38) */
uint64_t __tmp_in_tmp213;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp213;
}
tmp213[i0] = (role == CLIENT) ? __tmp_in_tmp213 : 0;
}

auto tmp214 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp214 at (1031,1-1031,38) */
uint64_t __tmp_in_tmp214;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp214;
}
tmp214[i0] = (role == CLIENT) ? __tmp_in_tmp214 : 0;
}

auto tmp215 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp215 at (1034,1-1034,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp216 at (1037,1-1037,39) */
uint64_t __tmp_in_tmp216;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp216;
}
tmp216[i0] = (role == CLIENT) ? __tmp_in_tmp216 : 0;
}

auto tmp217 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp217 at (1040,1-1040,39) */
uint64_t __tmp_in_tmp217;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp217;
}
tmp217[i0] = (role == CLIENT) ? __tmp_in_tmp217 : 0;
}

auto tmp218 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp218 at (1043,1-1043,39) */
uint64_t __tmp_in_tmp218;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp218;
}
tmp218[i0] = (role == CLIENT) ? __tmp_in_tmp218 : 0;
}

auto tmp219 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp219 at (1046,1-1046,39) */
uint64_t __tmp_in_tmp219;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp219;
}
tmp219[i0] = (role == CLIENT) ? __tmp_in_tmp219 : 0;
}

auto tmp220 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp220 at (1049,1-1049,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp221 at (1052,1-1052,38) */
uint64_t __tmp_in_tmp221;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp221;
}
tmp221[i0] = (role == CLIENT) ? __tmp_in_tmp221 : 0;
}

auto tmp222 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp222 at (1055,1-1055,38) */
uint64_t __tmp_in_tmp222;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp222;
}
tmp222[i0] = (role == CLIENT) ? __tmp_in_tmp222 : 0;
}

auto tmp223 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp223 at (1058,1-1058,38) */
uint64_t __tmp_in_tmp223;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp223;
}
tmp223[i0] = (role == CLIENT) ? __tmp_in_tmp223 : 0;
}

auto tmp224 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp224 at (1061,1-1061,38) */
uint64_t __tmp_in_tmp224;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp224;
}
tmp224[i0] = (role == CLIENT) ? __tmp_in_tmp224 : 0;
}

auto tmp225 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp225 at (1064,1-1064,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp226 at (1067,1-1067,38) */
uint64_t __tmp_in_tmp226;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp226;
}
tmp226[i0] = (role == CLIENT) ? __tmp_in_tmp226 : 0;
}

auto tmp227 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp227 at (1070,1-1070,38) */
uint64_t __tmp_in_tmp227;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp227;
}
tmp227[i0] = (role == CLIENT) ? __tmp_in_tmp227 : 0;
}

auto tmp228 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp228 at (1073,1-1073,38) */
uint64_t __tmp_in_tmp228;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp228;
}
tmp228[i0] = (role == CLIENT) ? __tmp_in_tmp228 : 0;
}

auto tmp229 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp229 at (1076,1-1076,38) */
uint64_t __tmp_in_tmp229;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp229;
}
tmp229[i0] = (role == CLIENT) ? __tmp_in_tmp229 : 0;
}

auto tmp230 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp230 at (1079,1-1079,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp231 at (1082,1-1082,39) */
uint64_t __tmp_in_tmp231;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp231;
}
tmp231[i0] = (role == CLIENT) ? __tmp_in_tmp231 : 0;
}

auto tmp232 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp232 at (1085,1-1085,39) */
uint64_t __tmp_in_tmp232;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp232;
}
tmp232[i0] = (role == CLIENT) ? __tmp_in_tmp232 : 0;
}

auto tmp233 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp233 at (1088,1-1088,39) */
uint64_t __tmp_in_tmp233;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp233;
}
tmp233[i0] = (role == CLIENT) ? __tmp_in_tmp233 : 0;
}

auto tmp234 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp234 at (1091,1-1091,39) */
uint64_t __tmp_in_tmp234;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp234;
}
tmp234[i0] = (role == CLIENT) ? __tmp_in_tmp234 : 0;
}

auto tmp235 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp235 at (1094,1-1094,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp236 at (1097,1-1097,38) */
uint64_t __tmp_in_tmp236;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp236;
}
tmp236[i0] = (role == CLIENT) ? __tmp_in_tmp236 : 0;
}

auto tmp237 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp237 at (1100,1-1100,38) */
uint64_t __tmp_in_tmp237;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp237;
}
tmp237[i0] = (role == CLIENT) ? __tmp_in_tmp237 : 0;
}

auto tmp238 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp238 at (1103,1-1103,38) */
uint64_t __tmp_in_tmp238;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp238;
}
tmp238[i0] = (role == CLIENT) ? __tmp_in_tmp238 : 0;
}

auto tmp239 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp239 at (1106,1-1106,38) */
uint64_t __tmp_in_tmp239;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp239;
}
tmp239[i0] = (role == CLIENT) ? __tmp_in_tmp239 : 0;
}

auto tmp240 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp240 at (1109,1-1109,49) */
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
/* Variable to read the clear value corresponding to the input variable tmp241 at (1112,1-1112,38) */
uint64_t __tmp_in_tmp241;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp241;
}
tmp241[i0] = (role == CLIENT) ? __tmp_in_tmp241 : 0;
}

auto tmp242 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp242 at (1115,1-1115,38) */
uint64_t __tmp_in_tmp242;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp242;
}
tmp242[i0] = (role == CLIENT) ? __tmp_in_tmp242 : 0;
}

auto tmp243 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp243 at (1118,1-1118,38) */
uint64_t __tmp_in_tmp243;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp243;
}
tmp243[i0] = (role == CLIENT) ? __tmp_in_tmp243 : 0;
}

auto tmp244 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp244 at (1121,1-1121,38) */
uint64_t __tmp_in_tmp244;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp244;
}
tmp244[i0] = (role == CLIENT) ? __tmp_in_tmp244 : 0;
}

auto tmp245 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp245 at (1124,1-1124,50) */
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
/* Variable to read the clear value corresponding to the input variable tmp246 at (1127,1-1127,39) */
uint64_t __tmp_in_tmp246;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp246;
}
tmp246[i0] = (role == CLIENT) ? __tmp_in_tmp246 : 0;
}

auto tmp247 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp247 at (1130,1-1130,39) */
uint64_t __tmp_in_tmp247;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp247;
}
tmp247[i0] = (role == CLIENT) ? __tmp_in_tmp247 : 0;
}

auto tmp248 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp248 at (1133,1-1133,39) */
uint64_t __tmp_in_tmp248;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp248;
}
tmp248[i0] = (role == CLIENT) ? __tmp_in_tmp248 : 0;
}

auto tmp249 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp249 at (1136,1-1136,39) */
uint64_t __tmp_in_tmp249;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp249;
}
tmp249[i0] = (role == CLIENT) ? __tmp_in_tmp249 : 0;
}

auto tmp250 = make_vector<uint64_t>( (int32_t)2048,  (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp250 at (1139,1-1139,45) */
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
/* Variable to read the clear value corresponding to the input variable tmp251 at (1142,1-1142,39) */
uint64_t __tmp_in_tmp251;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1001; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp251;
}
tmp251[i0] = (role == CLIENT) ? __tmp_in_tmp251 : 0;
}

leave_time();
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("Starting main protocol");
start_m();
touch_time();
auto tmp252 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp252[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp252[ (int64_t)1][ (int64_t)0] =  (int32_t)3;
tmp252[ (int64_t)1][ (int64_t)1] =  (int32_t)3;
tmp252[ (int64_t)2][ (int64_t)0] =  (int32_t)3;
tmp252[ (int64_t)2][ (int64_t)1] =  (int32_t)3;
tmp252[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp252[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp253 = make_vector<uint64_t>( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3);
Pad442( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0,  (int32_t)4,  (int32_t)2, tmp252, tmp253);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp252);
ClearMemSecret4( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0);

auto tmp256 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp253, tmp1,  (int32_t)12, tmp256);
ClearMemSecret4( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3, tmp253);
ClearMemSecret4( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64, tmp1);

auto tmp259 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)1,  (int32_t)0,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp256, tmp259);
ClearMemSecret4( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp256);

auto tmp261 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp259, tmp2, tmp3,  (int32_t)12, tmp261);
ClearMemSecret1( (int32_t)64, tmp3);
ClearMemSecret1( (int32_t)64, tmp2);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp259);

auto tmp265 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp261, tmp265);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp261);

auto tmp267 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp265, tmp6,  (int32_t)12, tmp267);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp6);

auto tmp269 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp265, tmp7,  (int32_t)12, tmp269);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp265);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64, tmp7);

auto tmp272 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp269, tmp8, tmp9,  (int32_t)12, tmp272);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp269);
ClearMemSecret1( (int32_t)64, tmp8);
ClearMemSecret1( (int32_t)64, tmp9);

auto tmp276 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp272, tmp276);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp272);

auto tmp278 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp276, tmp12,  (int32_t)12, tmp278);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp276);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp12);

auto tmp281 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp278, tmp13, tmp14,  (int32_t)12, tmp281);
ClearMemSecret1( (int32_t)64, tmp14);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp278);
ClearMemSecret1( (int32_t)64, tmp13);

auto tmp285 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp281, tmp285);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp281);

auto tmp287 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp285, tmp17,  (int32_t)12, tmp287);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp285);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp17);

auto tmp290 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp287, tmp267, tmp290);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp287);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp267);

auto tmp293 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp290, tmp18, tmp19,  (int32_t)12, tmp293);
ClearMemSecret1( (int32_t)256, tmp18);
ClearMemSecret1( (int32_t)256, tmp19);

auto tmp296 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp293, tmp296);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp293);

auto tmp298 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp296, tmp22,  (int32_t)12, tmp298);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp22);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp296);

auto tmp301 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp298, tmp23, tmp24,  (int32_t)12, tmp301);
ClearMemSecret1( (int32_t)64, tmp23);
ClearMemSecret1( (int32_t)64, tmp24);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp298);

auto tmp305 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp301, tmp305);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp301);

auto tmp307 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp305, tmp27,  (int32_t)12, tmp307);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp27);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp305);

auto tmp310 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp307, tmp28, tmp29,  (int32_t)12, tmp310);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp307);
ClearMemSecret1( (int32_t)64, tmp29);
ClearMemSecret1( (int32_t)64, tmp28);

auto tmp314 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp310, tmp314);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp310);

auto tmp316 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp314, tmp32,  (int32_t)12, tmp316);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp32);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp314);

auto tmp319 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp316, tmp290, tmp319);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp316);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp290);

auto tmp322 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp319, tmp33, tmp34,  (int32_t)12, tmp322);
ClearMemSecret1( (int32_t)256, tmp34);
ClearMemSecret1( (int32_t)256, tmp33);

auto tmp325 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp322, tmp325);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp322);

auto tmp327 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp325, tmp37,  (int32_t)12, tmp327);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp325);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp37);

auto tmp330 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp327, tmp38, tmp39,  (int32_t)12, tmp330);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp327);
ClearMemSecret1( (int32_t)64, tmp39);
ClearMemSecret1( (int32_t)64, tmp38);

auto tmp334 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp330, tmp334);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp330);

auto tmp336 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp334, tmp42,  (int32_t)12, tmp336);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp42);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp334);

auto tmp339 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp336, tmp43, tmp44,  (int32_t)12, tmp339);
ClearMemSecret1( (int32_t)64, tmp44);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp336);
ClearMemSecret1( (int32_t)64, tmp43);

auto tmp343 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp339, tmp343);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp339);

auto tmp345 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp343, tmp47,  (int32_t)12, tmp345);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp47);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp343);

auto tmp348 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp345, tmp319, tmp348);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp345);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp319);

auto tmp351 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp348, tmp48, tmp49,  (int32_t)12, tmp351);
ClearMemSecret1( (int32_t)256, tmp49);
ClearMemSecret1( (int32_t)256, tmp48);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp348);

auto tmp355 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp351, tmp355);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp351);

auto tmp357 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp357[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp357[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp357[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp357[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp357[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp357[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp357[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp357[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp358 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp355,  (int32_t)4,  (int32_t)2, tmp357, tmp358);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp357);

auto tmp360 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp358, tmp52,  (int32_t)12, tmp360);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp358);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512, tmp52);

auto tmp363 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp355, tmp53,  (int32_t)12, tmp363);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128, tmp53);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp355);

auto tmp366 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp363, tmp54, tmp55,  (int32_t)12, tmp366);
ClearMemSecret1( (int32_t)128, tmp54);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp363);
ClearMemSecret1( (int32_t)128, tmp55);

auto tmp370 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp366, tmp370);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp366);

auto tmp372 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp372[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp372[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp372[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp372[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp372[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp372[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp372[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp372[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp373 = make_vector<uint64_t>( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128);
Pad442( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp370,  (int32_t)4,  (int32_t)2, tmp372, tmp373);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp370);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp372);

auto tmp376 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp373, tmp58,  (int32_t)12, tmp376);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp58);
ClearMemSecret4( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128, tmp373);

auto tmp379 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp376, tmp59, tmp60,  (int32_t)12, tmp379);
ClearMemSecret1( (int32_t)128, tmp59);
ClearMemSecret1( (int32_t)128, tmp60);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp376);

auto tmp383 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp379, tmp383);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp379);

auto tmp385 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp383, tmp63,  (int32_t)12, tmp385);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp63);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp383);

auto tmp388 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp385, tmp360, tmp388);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp360);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp385);

auto tmp391 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp388, tmp64, tmp65,  (int32_t)12, tmp391);
ClearMemSecret1( (int32_t)512, tmp65);
ClearMemSecret1( (int32_t)512, tmp64);

auto tmp394 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp391, tmp394);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp391);

auto tmp396 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp394, tmp68,  (int32_t)12, tmp396);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp68);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp394);

auto tmp399 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp396, tmp69, tmp70,  (int32_t)12, tmp399);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp396);
ClearMemSecret1( (int32_t)128, tmp69);
ClearMemSecret1( (int32_t)128, tmp70);

auto tmp403 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp399, tmp403);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp399);

auto tmp405 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp403, tmp73,  (int32_t)12, tmp405);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp73);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp403);

auto tmp408 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp405, tmp74, tmp75,  (int32_t)12, tmp408);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp405);
ClearMemSecret1( (int32_t)128, tmp75);
ClearMemSecret1( (int32_t)128, tmp74);

auto tmp412 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp408, tmp412);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp408);

auto tmp414 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp412, tmp78,  (int32_t)12, tmp414);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp412);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp78);

auto tmp417 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp414, tmp388, tmp417);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp388);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp414);

auto tmp420 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp417, tmp79, tmp80,  (int32_t)12, tmp420);
ClearMemSecret1( (int32_t)512, tmp79);
ClearMemSecret1( (int32_t)512, tmp80);

auto tmp423 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp420, tmp423);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp420);

auto tmp425 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp423, tmp83,  (int32_t)12, tmp425);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp83);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp423);

auto tmp428 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp425, tmp84, tmp85,  (int32_t)12, tmp428);
ClearMemSecret1( (int32_t)128, tmp84);
ClearMemSecret1( (int32_t)128, tmp85);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp425);

auto tmp432 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp428, tmp432);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp428);

auto tmp434 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp432, tmp88,  (int32_t)12, tmp434);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp88);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp432);

auto tmp437 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp434, tmp89, tmp90,  (int32_t)12, tmp437);
ClearMemSecret1( (int32_t)128, tmp90);
ClearMemSecret1( (int32_t)128, tmp89);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp434);

auto tmp441 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp437, tmp441);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp437);

auto tmp443 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp441, tmp93,  (int32_t)12, tmp443);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp441);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp93);

auto tmp446 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp443, tmp417, tmp446);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp443);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp417);

auto tmp449 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp446, tmp94, tmp95,  (int32_t)12, tmp449);
ClearMemSecret1( (int32_t)512, tmp94);
ClearMemSecret1( (int32_t)512, tmp95);

auto tmp452 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp449, tmp452);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp449);

auto tmp454 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp452, tmp98,  (int32_t)12, tmp454);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp452);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp98);

auto tmp457 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp454, tmp99, tmp100,  (int32_t)12, tmp457);
ClearMemSecret1( (int32_t)128, tmp99);
ClearMemSecret1( (int32_t)128, tmp100);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp454);

auto tmp461 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp457, tmp461);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp457);

auto tmp463 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp461, tmp103,  (int32_t)12, tmp463);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp461);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp103);

auto tmp466 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp463, tmp104, tmp105,  (int32_t)12, tmp466);
ClearMemSecret1( (int32_t)128, tmp104);
ClearMemSecret1( (int32_t)128, tmp105);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp463);

auto tmp470 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp466, tmp470);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp466);

auto tmp472 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp470, tmp108,  (int32_t)12, tmp472);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp108);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp470);

auto tmp475 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp472, tmp446, tmp475);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp446);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp472);

auto tmp478 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp475, tmp109, tmp110,  (int32_t)12, tmp478);
ClearMemSecret1( (int32_t)512, tmp109);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp475);
ClearMemSecret1( (int32_t)512, tmp110);

auto tmp482 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp478, tmp482);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp478);

auto tmp484 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp484[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp484[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp484[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp484[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp484[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp484[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp484[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp484[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp485 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Pad442( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp482,  (int32_t)4,  (int32_t)2, tmp484, tmp485);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp484);

auto tmp487 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp485, tmp113,  (int32_t)12, tmp487);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp485);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024, tmp113);

auto tmp490 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp482, tmp114,  (int32_t)12, tmp490);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp482);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256, tmp114);

auto tmp493 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp490, tmp115, tmp116,  (int32_t)12, tmp493);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp490);
ClearMemSecret1( (int32_t)256, tmp116);
ClearMemSecret1( (int32_t)256, tmp115);

auto tmp497 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp493, tmp497);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp493);

auto tmp499 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp499[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp499[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp499[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp499[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp499[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp499[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp499[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp499[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp500 = make_vector<uint64_t>( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp497,  (int32_t)4,  (int32_t)2, tmp499, tmp500);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp499);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp497);

auto tmp503 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp500, tmp119,  (int32_t)12, tmp503);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp119);
ClearMemSecret4( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256, tmp500);

auto tmp506 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp503, tmp120, tmp121,  (int32_t)12, tmp506);
ClearMemSecret1( (int32_t)256, tmp120);
ClearMemSecret1( (int32_t)256, tmp121);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp503);

auto tmp510 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp506, tmp510);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp506);

auto tmp512 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp510, tmp124,  (int32_t)12, tmp512);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp510);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp124);

auto tmp515 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp512, tmp487, tmp515);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp487);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp512);

auto tmp518 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp515, tmp125, tmp126,  (int32_t)12, tmp518);
ClearMemSecret1( (int32_t)1024, tmp125);
ClearMemSecret1( (int32_t)1024, tmp126);

auto tmp521 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp518, tmp521);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp518);

auto tmp523 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp521, tmp129,  (int32_t)12, tmp523);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp129);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp521);

auto tmp526 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp523, tmp130, tmp131,  (int32_t)12, tmp526);
ClearMemSecret1( (int32_t)256, tmp131);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp523);
ClearMemSecret1( (int32_t)256, tmp130);

auto tmp530 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp526, tmp530);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp526);

auto tmp532 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp530, tmp134,  (int32_t)12, tmp532);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp134);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp530);

auto tmp535 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp532, tmp135, tmp136,  (int32_t)12, tmp535);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp532);
ClearMemSecret1( (int32_t)256, tmp136);
ClearMemSecret1( (int32_t)256, tmp135);

auto tmp539 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp535, tmp539);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp535);

auto tmp541 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp539, tmp139,  (int32_t)12, tmp541);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp139);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp539);

auto tmp544 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp541, tmp515, tmp544);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp541);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp515);

auto tmp547 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp544, tmp140, tmp141,  (int32_t)12, tmp547);
ClearMemSecret1( (int32_t)1024, tmp141);
ClearMemSecret1( (int32_t)1024, tmp140);

auto tmp550 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp547, tmp550);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp547);

auto tmp552 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp550, tmp144,  (int32_t)12, tmp552);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp144);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp550);

auto tmp555 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp552, tmp145, tmp146,  (int32_t)12, tmp555);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp552);
ClearMemSecret1( (int32_t)256, tmp146);
ClearMemSecret1( (int32_t)256, tmp145);

auto tmp559 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp555, tmp559);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp555);

auto tmp561 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp559, tmp149,  (int32_t)12, tmp561);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp559);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp149);

auto tmp564 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp561, tmp150, tmp151,  (int32_t)12, tmp564);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp561);
ClearMemSecret1( (int32_t)256, tmp151);
ClearMemSecret1( (int32_t)256, tmp150);

auto tmp568 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp564, tmp568);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp564);

auto tmp570 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp568, tmp154,  (int32_t)12, tmp570);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp154);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp568);

auto tmp573 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp570, tmp544, tmp573);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp570);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp544);

auto tmp576 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp573, tmp155, tmp156,  (int32_t)12, tmp576);
ClearMemSecret1( (int32_t)1024, tmp155);
ClearMemSecret1( (int32_t)1024, tmp156);

auto tmp579 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp576, tmp579);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp576);

auto tmp581 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp579, tmp159,  (int32_t)12, tmp581);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp159);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp579);

auto tmp584 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp581, tmp160, tmp161,  (int32_t)12, tmp584);
ClearMemSecret1( (int32_t)256, tmp161);
ClearMemSecret1( (int32_t)256, tmp160);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp581);

auto tmp588 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp584, tmp588);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp584);

auto tmp590 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp588, tmp164,  (int32_t)12, tmp590);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp164);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp588);

auto tmp593 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp590, tmp165, tmp166,  (int32_t)12, tmp593);
ClearMemSecret1( (int32_t)256, tmp165);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp590);
ClearMemSecret1( (int32_t)256, tmp166);

auto tmp597 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp593, tmp597);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp593);

auto tmp599 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp597, tmp169,  (int32_t)12, tmp599);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp169);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp597);

auto tmp602 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp599, tmp573, tmp602);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp573);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp599);

auto tmp605 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp602, tmp170, tmp171,  (int32_t)12, tmp605);
ClearMemSecret1( (int32_t)1024, tmp170);
ClearMemSecret1( (int32_t)1024, tmp171);

auto tmp608 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp605, tmp608);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp605);

auto tmp610 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp608, tmp174,  (int32_t)12, tmp610);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp608);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp174);

auto tmp613 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp610, tmp175, tmp176,  (int32_t)12, tmp613);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp610);
ClearMemSecret1( (int32_t)256, tmp175);
ClearMemSecret1( (int32_t)256, tmp176);

auto tmp617 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp613, tmp617);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp613);

auto tmp619 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp617, tmp179,  (int32_t)12, tmp619);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp617);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp179);

auto tmp622 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp619, tmp180, tmp181,  (int32_t)12, tmp622);
ClearMemSecret1( (int32_t)256, tmp181);
ClearMemSecret1( (int32_t)256, tmp180);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp619);

auto tmp626 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp622, tmp626);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp622);

auto tmp628 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp626, tmp184,  (int32_t)12, tmp628);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp626);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp184);

auto tmp631 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp628, tmp602, tmp631);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp628);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp602);

auto tmp634 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp631, tmp185, tmp186,  (int32_t)12, tmp634);
ClearMemSecret1( (int32_t)1024, tmp185);
ClearMemSecret1( (int32_t)1024, tmp186);

auto tmp637 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp634, tmp637);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp634);

auto tmp639 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp637, tmp189,  (int32_t)12, tmp639);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp637);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp189);

auto tmp642 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp639, tmp190, tmp191,  (int32_t)12, tmp642);
ClearMemSecret1( (int32_t)256, tmp190);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp639);
ClearMemSecret1( (int32_t)256, tmp191);

auto tmp646 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp642, tmp646);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp642);

auto tmp648 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp646, tmp194,  (int32_t)12, tmp648);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp194);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp646);

auto tmp651 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp648, tmp195, tmp196,  (int32_t)12, tmp651);
ClearMemSecret1( (int32_t)256, tmp195);
ClearMemSecret1( (int32_t)256, tmp196);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp648);

auto tmp655 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp651, tmp655);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp651);

auto tmp657 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp655, tmp199,  (int32_t)12, tmp657);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp199);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp655);

auto tmp660 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp657, tmp631, tmp660);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp657);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp631);

auto tmp663 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp660, tmp200, tmp201,  (int32_t)12, tmp663);
ClearMemSecret1( (int32_t)1024, tmp201);
ClearMemSecret1( (int32_t)1024, tmp200);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp660);

auto tmp667 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp663, tmp667);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp663);

auto tmp669 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp669[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp669[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp669[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp669[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp669[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp669[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp669[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp669[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp670 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Pad442( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp667,  (int32_t)4,  (int32_t)2, tmp669, tmp670);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp669);

auto tmp672 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp670, tmp204,  (int32_t)12, tmp672);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp670);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048, tmp204);

auto tmp675 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp667, tmp205,  (int32_t)12, tmp675);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512, tmp205);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp667);

auto tmp678 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp675, tmp206, tmp207,  (int32_t)12, tmp678);
ClearMemSecret1( (int32_t)512, tmp207);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp675);
ClearMemSecret1( (int32_t)512, tmp206);

auto tmp682 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp678, tmp682);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp678);

auto tmp684 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp684[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp684[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp684[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp684[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp684[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp684[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp684[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp684[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp685 = make_vector<uint64_t>( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512);
Pad442( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp682,  (int32_t)4,  (int32_t)2, tmp684, tmp685);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp682);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp684);

auto tmp688 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp685, tmp210,  (int32_t)12, tmp688);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp210);
ClearMemSecret4( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512, tmp685);

auto tmp691 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp688, tmp211, tmp212,  (int32_t)12, tmp691);
ClearMemSecret1( (int32_t)512, tmp211);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp688);
ClearMemSecret1( (int32_t)512, tmp212);

auto tmp695 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp691, tmp695);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp691);

auto tmp697 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp695, tmp215,  (int32_t)12, tmp697);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp215);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp695);

auto tmp700 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp697, tmp672, tmp700);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp697);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp672);

auto tmp703 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp700, tmp216, tmp217,  (int32_t)12, tmp703);
ClearMemSecret1( (int32_t)2048, tmp216);
ClearMemSecret1( (int32_t)2048, tmp217);

auto tmp706 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp703, tmp706);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp703);

auto tmp708 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp706, tmp220,  (int32_t)12, tmp708);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp706);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp220);

auto tmp711 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp708, tmp221, tmp222,  (int32_t)12, tmp711);
ClearMemSecret1( (int32_t)512, tmp222);
ClearMemSecret1( (int32_t)512, tmp221);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp708);

auto tmp715 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp711, tmp715);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp711);

auto tmp717 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp715, tmp225,  (int32_t)12, tmp717);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp715);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp225);

auto tmp720 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp717, tmp226, tmp227,  (int32_t)12, tmp720);
ClearMemSecret1( (int32_t)512, tmp227);
ClearMemSecret1( (int32_t)512, tmp226);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp717);

auto tmp724 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp720, tmp724);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp720);

auto tmp726 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp724, tmp230,  (int32_t)12, tmp726);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp724);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp230);

auto tmp729 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp726, tmp700, tmp729);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp700);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp726);

auto tmp732 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp729, tmp231, tmp232,  (int32_t)12, tmp732);
ClearMemSecret1( (int32_t)2048, tmp232);
ClearMemSecret1( (int32_t)2048, tmp231);

auto tmp735 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp732, tmp735);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp732);

auto tmp737 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp735, tmp235,  (int32_t)12, tmp737);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp235);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp735);

auto tmp740 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp737, tmp236, tmp237,  (int32_t)12, tmp740);
ClearMemSecret1( (int32_t)512, tmp236);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp737);
ClearMemSecret1( (int32_t)512, tmp237);

auto tmp744 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp740, tmp744);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp740);

auto tmp746 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp744, tmp240,  (int32_t)12, tmp746);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp240);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp744);

auto tmp749 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp746, tmp241, tmp242,  (int32_t)12, tmp749);
ClearMemSecret1( (int32_t)512, tmp241);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp746);
ClearMemSecret1( (int32_t)512, tmp242);

auto tmp753 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp749, tmp753);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp749);

auto tmp755 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp753, tmp245,  (int32_t)12, tmp755);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp753);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp245);

auto tmp758 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp755, tmp729, tmp758);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp729);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp755);

auto tmp761 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp758, tmp246, tmp247,  (int32_t)12, tmp761);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp758);
ClearMemSecret1( (int32_t)2048, tmp247);
ClearMemSecret1( (int32_t)2048, tmp246);

auto tmp765 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp761, tmp765);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp761);

auto tmp767 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048);
AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp765, tmp767);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp765);

auto tmp769 = make_vector<uint64_t>( (int32_t)1,  (int32_t)2048);
Squeeze24( (int32_t)1,  (int32_t)2048,  (int32_t)1,  (int32_t)2,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp767, tmp769);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp767);

auto tmp771 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatMulCSF2D( (int32_t)1,  (int32_t)2048,  (int32_t)1001, tmp769, tmp250, tmp771,  (int64_t)12);
ClearMemSecret2( (int32_t)2048,  (int32_t)1001, tmp250);
ClearMemSecret2( (int32_t)1,  (int32_t)2048, tmp769);

auto tmp774 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatAddBroadCast2( (int32_t)1,  (int32_t)1001, tmp771, tmp251, tmp774);
ClearMemSecret1( (int32_t)1001, tmp251);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp771);

auto tmp777 = make_vector<uint64_t>( (int32_t)1);
ArgMax1( (int32_t)1,  (int32_t)1,  (int32_t)1001, tmp774,  (int32_t)1, tmp777);
ClearMemPublic( (int32_t)1);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp774);
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
print_integer(funcReconstruct2PCCons(tmp777[i0], 1));
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
