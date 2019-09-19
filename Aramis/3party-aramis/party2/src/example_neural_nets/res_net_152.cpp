#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "res_net_mem_opti.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef RESNET152

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

auto tmp113 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp113 at (728,1-728,49) */
uint64_t __tmp_in_tmp113;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp113;
}
tmp113[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp113 : 0;
}
}
}
}

auto tmp114 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp114 at (731,1-731,38) */
uint64_t __tmp_in_tmp114;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp114;
}
tmp114[i0] = (role == CLIENT) ? __tmp_in_tmp114 : 0;
}

auto tmp115 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp115 at (734,1-734,38) */
uint64_t __tmp_in_tmp115;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp115;
}
tmp115[i0] = (role == CLIENT) ? __tmp_in_tmp115 : 0;
}

auto tmp116 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp116 at (737,1-737,38) */
uint64_t __tmp_in_tmp116;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp116;
}
tmp116[i0] = (role == CLIENT) ? __tmp_in_tmp116 : 0;
}

auto tmp117 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp117 at (740,1-740,38) */
uint64_t __tmp_in_tmp117;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp117;
}
tmp117[i0] = (role == CLIENT) ? __tmp_in_tmp117 : 0;
}

auto tmp118 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp118 at (743,1-743,49) */
uint64_t __tmp_in_tmp118;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp118;
}
tmp118[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp118 : 0;
}
}
}
}

auto tmp119 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp119 at (746,1-746,38) */
uint64_t __tmp_in_tmp119;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp119;
}
tmp119[i0] = (role == CLIENT) ? __tmp_in_tmp119 : 0;
}

auto tmp120 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp120 at (749,1-749,38) */
uint64_t __tmp_in_tmp120;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp120;
}
tmp120[i0] = (role == CLIENT) ? __tmp_in_tmp120 : 0;
}

auto tmp121 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp121 at (752,1-752,38) */
uint64_t __tmp_in_tmp121;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp121;
}
tmp121[i0] = (role == CLIENT) ? __tmp_in_tmp121 : 0;
}

auto tmp122 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp122 at (755,1-755,38) */
uint64_t __tmp_in_tmp122;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp122;
}
tmp122[i0] = (role == CLIENT) ? __tmp_in_tmp122 : 0;
}

auto tmp123 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp123 at (758,1-758,49) */
uint64_t __tmp_in_tmp123;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp123;
}
tmp123[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp123 : 0;
}
}
}
}

auto tmp124 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp124 at (761,1-761,38) */
uint64_t __tmp_in_tmp124;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp124;
}
tmp124[i0] = (role == CLIENT) ? __tmp_in_tmp124 : 0;
}

auto tmp125 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp125 at (764,1-764,38) */
uint64_t __tmp_in_tmp125;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp125;
}
tmp125[i0] = (role == CLIENT) ? __tmp_in_tmp125 : 0;
}

auto tmp126 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp126 at (767,1-767,38) */
uint64_t __tmp_in_tmp126;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp126;
}
tmp126[i0] = (role == CLIENT) ? __tmp_in_tmp126 : 0;
}

auto tmp127 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp127 at (770,1-770,38) */
uint64_t __tmp_in_tmp127;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp127;
}
tmp127[i0] = (role == CLIENT) ? __tmp_in_tmp127 : 0;
}

auto tmp128 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp128 at (773,1-773,49) */
uint64_t __tmp_in_tmp128;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp128;
}
tmp128[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp128 : 0;
}
}
}
}

auto tmp129 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp129 at (776,1-776,38) */
uint64_t __tmp_in_tmp129;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp129;
}
tmp129[i0] = (role == CLIENT) ? __tmp_in_tmp129 : 0;
}

auto tmp130 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp130 at (779,1-779,38) */
uint64_t __tmp_in_tmp130;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp130;
}
tmp130[i0] = (role == CLIENT) ? __tmp_in_tmp130 : 0;
}

auto tmp131 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp131 at (782,1-782,38) */
uint64_t __tmp_in_tmp131;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp131;
}
tmp131[i0] = (role == CLIENT) ? __tmp_in_tmp131 : 0;
}

auto tmp132 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp132 at (785,1-785,38) */
uint64_t __tmp_in_tmp132;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp132;
}
tmp132[i0] = (role == CLIENT) ? __tmp_in_tmp132 : 0;
}

auto tmp133 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp133 at (788,1-788,49) */
uint64_t __tmp_in_tmp133;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp133;
}
tmp133[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp133 : 0;
}
}
}
}

auto tmp134 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp134 at (791,1-791,38) */
uint64_t __tmp_in_tmp134;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp134;
}
tmp134[i0] = (role == CLIENT) ? __tmp_in_tmp134 : 0;
}

auto tmp135 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp135 at (794,1-794,38) */
uint64_t __tmp_in_tmp135;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp135;
}
tmp135[i0] = (role == CLIENT) ? __tmp_in_tmp135 : 0;
}

auto tmp136 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp136 at (797,1-797,38) */
uint64_t __tmp_in_tmp136;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp136;
}
tmp136[i0] = (role == CLIENT) ? __tmp_in_tmp136 : 0;
}

auto tmp137 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp137 at (800,1-800,38) */
uint64_t __tmp_in_tmp137;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp137;
}
tmp137[i0] = (role == CLIENT) ? __tmp_in_tmp137 : 0;
}

auto tmp138 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp138 at (803,1-803,49) */
uint64_t __tmp_in_tmp138;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp138;
}
tmp138[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp138 : 0;
}
}
}
}

auto tmp139 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp139 at (806,1-806,38) */
uint64_t __tmp_in_tmp139;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp139;
}
tmp139[i0] = (role == CLIENT) ? __tmp_in_tmp139 : 0;
}

auto tmp140 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp140 at (809,1-809,38) */
uint64_t __tmp_in_tmp140;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp140;
}
tmp140[i0] = (role == CLIENT) ? __tmp_in_tmp140 : 0;
}

auto tmp141 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp141 at (812,1-812,38) */
uint64_t __tmp_in_tmp141;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp141;
}
tmp141[i0] = (role == CLIENT) ? __tmp_in_tmp141 : 0;
}

auto tmp142 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp142 at (815,1-815,38) */
uint64_t __tmp_in_tmp142;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp142;
}
tmp142[i0] = (role == CLIENT) ? __tmp_in_tmp142 : 0;
}

auto tmp143 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp143 at (818,1-818,49) */
uint64_t __tmp_in_tmp143;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp143;
}
tmp143[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp143 : 0;
}
}
}
}

auto tmp144 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp144 at (821,1-821,38) */
uint64_t __tmp_in_tmp144;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp144;
}
tmp144[i0] = (role == CLIENT) ? __tmp_in_tmp144 : 0;
}

auto tmp145 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp145 at (824,1-824,38) */
uint64_t __tmp_in_tmp145;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp145;
}
tmp145[i0] = (role == CLIENT) ? __tmp_in_tmp145 : 0;
}

auto tmp146 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp146 at (827,1-827,38) */
uint64_t __tmp_in_tmp146;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp146;
}
tmp146[i0] = (role == CLIENT) ? __tmp_in_tmp146 : 0;
}

auto tmp147 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp147 at (830,1-830,38) */
uint64_t __tmp_in_tmp147;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp147;
}
tmp147[i0] = (role == CLIENT) ? __tmp_in_tmp147 : 0;
}

auto tmp148 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp148 at (833,1-833,49) */
uint64_t __tmp_in_tmp148;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp148;
}
tmp148[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp148 : 0;
}
}
}
}

auto tmp149 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp149 at (836,1-836,38) */
uint64_t __tmp_in_tmp149;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp149;
}
tmp149[i0] = (role == CLIENT) ? __tmp_in_tmp149 : 0;
}

auto tmp150 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp150 at (839,1-839,38) */
uint64_t __tmp_in_tmp150;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp150;
}
tmp150[i0] = (role == CLIENT) ? __tmp_in_tmp150 : 0;
}

auto tmp151 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp151 at (842,1-842,38) */
uint64_t __tmp_in_tmp151;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp151;
}
tmp151[i0] = (role == CLIENT) ? __tmp_in_tmp151 : 0;
}

auto tmp152 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp152 at (845,1-845,38) */
uint64_t __tmp_in_tmp152;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp152;
}
tmp152[i0] = (role == CLIENT) ? __tmp_in_tmp152 : 0;
}

auto tmp153 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp153 at (848,1-848,49) */
uint64_t __tmp_in_tmp153;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp153;
}
tmp153[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp153 : 0;
}
}
}
}

auto tmp154 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp154 at (851,1-851,38) */
uint64_t __tmp_in_tmp154;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp154;
}
tmp154[i0] = (role == CLIENT) ? __tmp_in_tmp154 : 0;
}

auto tmp155 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp155 at (854,1-854,38) */
uint64_t __tmp_in_tmp155;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp155;
}
tmp155[i0] = (role == CLIENT) ? __tmp_in_tmp155 : 0;
}

auto tmp156 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp156 at (857,1-857,38) */
uint64_t __tmp_in_tmp156;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp156;
}
tmp156[i0] = (role == CLIENT) ? __tmp_in_tmp156 : 0;
}

auto tmp157 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp157 at (860,1-860,38) */
uint64_t __tmp_in_tmp157;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp157;
}
tmp157[i0] = (role == CLIENT) ? __tmp_in_tmp157 : 0;
}

auto tmp158 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp158 at (863,1-863,49) */
uint64_t __tmp_in_tmp158;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp158;
}
tmp158[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp158 : 0;
}
}
}
}

auto tmp159 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp159 at (866,1-866,38) */
uint64_t __tmp_in_tmp159;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp159;
}
tmp159[i0] = (role == CLIENT) ? __tmp_in_tmp159 : 0;
}

auto tmp160 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp160 at (869,1-869,38) */
uint64_t __tmp_in_tmp160;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp160;
}
tmp160[i0] = (role == CLIENT) ? __tmp_in_tmp160 : 0;
}

auto tmp161 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp161 at (872,1-872,38) */
uint64_t __tmp_in_tmp161;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp161;
}
tmp161[i0] = (role == CLIENT) ? __tmp_in_tmp161 : 0;
}

auto tmp162 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp162 at (875,1-875,38) */
uint64_t __tmp_in_tmp162;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp162;
}
tmp162[i0] = (role == CLIENT) ? __tmp_in_tmp162 : 0;
}

auto tmp163 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp163 at (878,1-878,49) */
uint64_t __tmp_in_tmp163;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp163;
}
tmp163[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp163 : 0;
}
}
}
}

auto tmp164 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp164 at (881,1-881,38) */
uint64_t __tmp_in_tmp164;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp164;
}
tmp164[i0] = (role == CLIENT) ? __tmp_in_tmp164 : 0;
}

auto tmp165 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp165 at (884,1-884,38) */
uint64_t __tmp_in_tmp165;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp165;
}
tmp165[i0] = (role == CLIENT) ? __tmp_in_tmp165 : 0;
}

auto tmp166 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp166 at (887,1-887,38) */
uint64_t __tmp_in_tmp166;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp166;
}
tmp166[i0] = (role == CLIENT) ? __tmp_in_tmp166 : 0;
}

auto tmp167 = make_vector<uint64_t>( (int32_t)128);
/* Variable to read the clear value corresponding to the input variable tmp167 at (890,1-890,38) */
uint64_t __tmp_in_tmp167;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp167;
}
tmp167[i0] = (role == CLIENT) ? __tmp_in_tmp167 : 0;
}

auto tmp168 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp168 at (893,1-893,49) */
uint64_t __tmp_in_tmp168;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp168;
}
tmp168[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp168 : 0;
}
}
}
}

auto tmp169 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp169 at (896,1-896,38) */
uint64_t __tmp_in_tmp169;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp169;
}
tmp169[i0] = (role == CLIENT) ? __tmp_in_tmp169 : 0;
}

auto tmp170 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp170 at (899,1-899,38) */
uint64_t __tmp_in_tmp170;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp170;
}
tmp170[i0] = (role == CLIENT) ? __tmp_in_tmp170 : 0;
}

auto tmp171 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp171 at (902,1-902,38) */
uint64_t __tmp_in_tmp171;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp171;
}
tmp171[i0] = (role == CLIENT) ? __tmp_in_tmp171 : 0;
}

auto tmp172 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp172 at (905,1-905,38) */
uint64_t __tmp_in_tmp172;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp172;
}
tmp172[i0] = (role == CLIENT) ? __tmp_in_tmp172 : 0;
}

auto tmp173 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp173 at (908,1-908,50) */
uint64_t __tmp_in_tmp173;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp173;
}
tmp173[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp173 : 0;
}
}
}
}

auto tmp174 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp174 at (911,1-911,49) */
uint64_t __tmp_in_tmp174;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
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

auto tmp459 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp459 at (1766,1-1766,50) */
uint64_t __tmp_in_tmp459;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp459;
}
tmp459[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp459 : 0;
}
}
}
}

auto tmp460 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp460 at (1769,1-1769,38) */
uint64_t __tmp_in_tmp460;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp460;
}
tmp460[i0] = (role == CLIENT) ? __tmp_in_tmp460 : 0;
}

auto tmp461 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp461 at (1772,1-1772,38) */
uint64_t __tmp_in_tmp461;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp461;
}
tmp461[i0] = (role == CLIENT) ? __tmp_in_tmp461 : 0;
}

auto tmp462 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp462 at (1775,1-1775,38) */
uint64_t __tmp_in_tmp462;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp462;
}
tmp462[i0] = (role == CLIENT) ? __tmp_in_tmp462 : 0;
}

auto tmp463 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp463 at (1778,1-1778,38) */
uint64_t __tmp_in_tmp463;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp463;
}
tmp463[i0] = (role == CLIENT) ? __tmp_in_tmp463 : 0;
}

auto tmp464 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp464 at (1781,1-1781,49) */
uint64_t __tmp_in_tmp464;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp464;
}
tmp464[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp464 : 0;
}
}
}
}

auto tmp465 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp465 at (1784,1-1784,38) */
uint64_t __tmp_in_tmp465;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp465;
}
tmp465[i0] = (role == CLIENT) ? __tmp_in_tmp465 : 0;
}

auto tmp466 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp466 at (1787,1-1787,38) */
uint64_t __tmp_in_tmp466;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp466;
}
tmp466[i0] = (role == CLIENT) ? __tmp_in_tmp466 : 0;
}

auto tmp467 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp467 at (1790,1-1790,38) */
uint64_t __tmp_in_tmp467;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp467;
}
tmp467[i0] = (role == CLIENT) ? __tmp_in_tmp467 : 0;
}

auto tmp468 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp468 at (1793,1-1793,38) */
uint64_t __tmp_in_tmp468;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp468;
}
tmp468[i0] = (role == CLIENT) ? __tmp_in_tmp468 : 0;
}

auto tmp469 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp469 at (1796,1-1796,50) */
uint64_t __tmp_in_tmp469;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp469;
}
tmp469[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp469 : 0;
}
}
}
}

auto tmp470 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp470 at (1799,1-1799,39) */
uint64_t __tmp_in_tmp470;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp470;
}
tmp470[i0] = (role == CLIENT) ? __tmp_in_tmp470 : 0;
}

auto tmp471 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp471 at (1802,1-1802,39) */
uint64_t __tmp_in_tmp471;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp471;
}
tmp471[i0] = (role == CLIENT) ? __tmp_in_tmp471 : 0;
}

auto tmp472 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp472 at (1805,1-1805,39) */
uint64_t __tmp_in_tmp472;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp472;
}
tmp472[i0] = (role == CLIENT) ? __tmp_in_tmp472 : 0;
}

auto tmp473 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp473 at (1808,1-1808,39) */
uint64_t __tmp_in_tmp473;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp473;
}
tmp473[i0] = (role == CLIENT) ? __tmp_in_tmp473 : 0;
}

auto tmp474 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp474 at (1811,1-1811,50) */
uint64_t __tmp_in_tmp474;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp474;
}
tmp474[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp474 : 0;
}
}
}
}

auto tmp475 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp475 at (1814,1-1814,38) */
uint64_t __tmp_in_tmp475;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp475;
}
tmp475[i0] = (role == CLIENT) ? __tmp_in_tmp475 : 0;
}

auto tmp476 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp476 at (1817,1-1817,38) */
uint64_t __tmp_in_tmp476;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp476;
}
tmp476[i0] = (role == CLIENT) ? __tmp_in_tmp476 : 0;
}

auto tmp477 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp477 at (1820,1-1820,38) */
uint64_t __tmp_in_tmp477;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp477;
}
tmp477[i0] = (role == CLIENT) ? __tmp_in_tmp477 : 0;
}

auto tmp478 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp478 at (1823,1-1823,38) */
uint64_t __tmp_in_tmp478;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp478;
}
tmp478[i0] = (role == CLIENT) ? __tmp_in_tmp478 : 0;
}

auto tmp479 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp479 at (1826,1-1826,49) */
uint64_t __tmp_in_tmp479;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp479;
}
tmp479[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp479 : 0;
}
}
}
}

auto tmp480 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp480 at (1829,1-1829,38) */
uint64_t __tmp_in_tmp480;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp480;
}
tmp480[i0] = (role == CLIENT) ? __tmp_in_tmp480 : 0;
}

auto tmp481 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp481 at (1832,1-1832,38) */
uint64_t __tmp_in_tmp481;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp481;
}
tmp481[i0] = (role == CLIENT) ? __tmp_in_tmp481 : 0;
}

auto tmp482 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp482 at (1835,1-1835,38) */
uint64_t __tmp_in_tmp482;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp482;
}
tmp482[i0] = (role == CLIENT) ? __tmp_in_tmp482 : 0;
}

auto tmp483 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp483 at (1838,1-1838,38) */
uint64_t __tmp_in_tmp483;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp483;
}
tmp483[i0] = (role == CLIENT) ? __tmp_in_tmp483 : 0;
}

auto tmp484 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp484 at (1841,1-1841,50) */
uint64_t __tmp_in_tmp484;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp484;
}
tmp484[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp484 : 0;
}
}
}
}

auto tmp485 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp485 at (1844,1-1844,39) */
uint64_t __tmp_in_tmp485;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp485;
}
tmp485[i0] = (role == CLIENT) ? __tmp_in_tmp485 : 0;
}

auto tmp486 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp486 at (1847,1-1847,39) */
uint64_t __tmp_in_tmp486;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp486;
}
tmp486[i0] = (role == CLIENT) ? __tmp_in_tmp486 : 0;
}

auto tmp487 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp487 at (1850,1-1850,39) */
uint64_t __tmp_in_tmp487;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp487;
}
tmp487[i0] = (role == CLIENT) ? __tmp_in_tmp487 : 0;
}

auto tmp488 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp488 at (1853,1-1853,39) */
uint64_t __tmp_in_tmp488;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp488;
}
tmp488[i0] = (role == CLIENT) ? __tmp_in_tmp488 : 0;
}

auto tmp489 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp489 at (1856,1-1856,50) */
uint64_t __tmp_in_tmp489;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp489;
}
tmp489[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp489 : 0;
}
}
}
}

auto tmp490 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp490 at (1859,1-1859,38) */
uint64_t __tmp_in_tmp490;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp490;
}
tmp490[i0] = (role == CLIENT) ? __tmp_in_tmp490 : 0;
}

auto tmp491 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp491 at (1862,1-1862,38) */
uint64_t __tmp_in_tmp491;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp491;
}
tmp491[i0] = (role == CLIENT) ? __tmp_in_tmp491 : 0;
}

auto tmp492 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp492 at (1865,1-1865,38) */
uint64_t __tmp_in_tmp492;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp492;
}
tmp492[i0] = (role == CLIENT) ? __tmp_in_tmp492 : 0;
}

auto tmp493 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp493 at (1868,1-1868,38) */
uint64_t __tmp_in_tmp493;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp493;
}
tmp493[i0] = (role == CLIENT) ? __tmp_in_tmp493 : 0;
}

auto tmp494 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp494 at (1871,1-1871,49) */
uint64_t __tmp_in_tmp494;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp494;
}
tmp494[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp494 : 0;
}
}
}
}

auto tmp495 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp495 at (1874,1-1874,38) */
uint64_t __tmp_in_tmp495;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp495;
}
tmp495[i0] = (role == CLIENT) ? __tmp_in_tmp495 : 0;
}

auto tmp496 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp496 at (1877,1-1877,38) */
uint64_t __tmp_in_tmp496;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp496;
}
tmp496[i0] = (role == CLIENT) ? __tmp_in_tmp496 : 0;
}

auto tmp497 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp497 at (1880,1-1880,38) */
uint64_t __tmp_in_tmp497;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp497;
}
tmp497[i0] = (role == CLIENT) ? __tmp_in_tmp497 : 0;
}

auto tmp498 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp498 at (1883,1-1883,38) */
uint64_t __tmp_in_tmp498;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp498;
}
tmp498[i0] = (role == CLIENT) ? __tmp_in_tmp498 : 0;
}

auto tmp499 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp499 at (1886,1-1886,50) */
uint64_t __tmp_in_tmp499;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp499;
}
tmp499[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp499 : 0;
}
}
}
}

auto tmp500 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp500 at (1889,1-1889,39) */
uint64_t __tmp_in_tmp500;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp500;
}
tmp500[i0] = (role == CLIENT) ? __tmp_in_tmp500 : 0;
}

auto tmp501 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp501 at (1892,1-1892,39) */
uint64_t __tmp_in_tmp501;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp501;
}
tmp501[i0] = (role == CLIENT) ? __tmp_in_tmp501 : 0;
}

auto tmp502 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp502 at (1895,1-1895,39) */
uint64_t __tmp_in_tmp502;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp502;
}
tmp502[i0] = (role == CLIENT) ? __tmp_in_tmp502 : 0;
}

auto tmp503 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp503 at (1898,1-1898,39) */
uint64_t __tmp_in_tmp503;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp503;
}
tmp503[i0] = (role == CLIENT) ? __tmp_in_tmp503 : 0;
}

auto tmp504 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp504 at (1901,1-1901,50) */
uint64_t __tmp_in_tmp504;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp504;
}
tmp504[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp504 : 0;
}
}
}
}

auto tmp505 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp505 at (1904,1-1904,38) */
uint64_t __tmp_in_tmp505;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp505;
}
tmp505[i0] = (role == CLIENT) ? __tmp_in_tmp505 : 0;
}

auto tmp506 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp506 at (1907,1-1907,38) */
uint64_t __tmp_in_tmp506;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp506;
}
tmp506[i0] = (role == CLIENT) ? __tmp_in_tmp506 : 0;
}

auto tmp507 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp507 at (1910,1-1910,38) */
uint64_t __tmp_in_tmp507;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp507;
}
tmp507[i0] = (role == CLIENT) ? __tmp_in_tmp507 : 0;
}

auto tmp508 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp508 at (1913,1-1913,38) */
uint64_t __tmp_in_tmp508;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp508;
}
tmp508[i0] = (role == CLIENT) ? __tmp_in_tmp508 : 0;
}

auto tmp509 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp509 at (1916,1-1916,49) */
uint64_t __tmp_in_tmp509;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp509;
}
tmp509[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp509 : 0;
}
}
}
}

auto tmp510 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp510 at (1919,1-1919,38) */
uint64_t __tmp_in_tmp510;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp510;
}
tmp510[i0] = (role == CLIENT) ? __tmp_in_tmp510 : 0;
}

auto tmp511 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp511 at (1922,1-1922,38) */
uint64_t __tmp_in_tmp511;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp511;
}
tmp511[i0] = (role == CLIENT) ? __tmp_in_tmp511 : 0;
}

auto tmp512 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp512 at (1925,1-1925,38) */
uint64_t __tmp_in_tmp512;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp512;
}
tmp512[i0] = (role == CLIENT) ? __tmp_in_tmp512 : 0;
}

auto tmp513 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp513 at (1928,1-1928,38) */
uint64_t __tmp_in_tmp513;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp513;
}
tmp513[i0] = (role == CLIENT) ? __tmp_in_tmp513 : 0;
}

auto tmp514 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp514 at (1931,1-1931,50) */
uint64_t __tmp_in_tmp514;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp514;
}
tmp514[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp514 : 0;
}
}
}
}

auto tmp515 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp515 at (1934,1-1934,39) */
uint64_t __tmp_in_tmp515;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp515;
}
tmp515[i0] = (role == CLIENT) ? __tmp_in_tmp515 : 0;
}

auto tmp516 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp516 at (1937,1-1937,39) */
uint64_t __tmp_in_tmp516;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp516;
}
tmp516[i0] = (role == CLIENT) ? __tmp_in_tmp516 : 0;
}

auto tmp517 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp517 at (1940,1-1940,39) */
uint64_t __tmp_in_tmp517;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp517;
}
tmp517[i0] = (role == CLIENT) ? __tmp_in_tmp517 : 0;
}

auto tmp518 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp518 at (1943,1-1943,39) */
uint64_t __tmp_in_tmp518;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp518;
}
tmp518[i0] = (role == CLIENT) ? __tmp_in_tmp518 : 0;
}

auto tmp519 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp519 at (1946,1-1946,50) */
uint64_t __tmp_in_tmp519;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp519;
}
tmp519[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp519 : 0;
}
}
}
}

auto tmp520 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp520 at (1949,1-1949,38) */
uint64_t __tmp_in_tmp520;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp520;
}
tmp520[i0] = (role == CLIENT) ? __tmp_in_tmp520 : 0;
}

auto tmp521 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp521 at (1952,1-1952,38) */
uint64_t __tmp_in_tmp521;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp521;
}
tmp521[i0] = (role == CLIENT) ? __tmp_in_tmp521 : 0;
}

auto tmp522 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp522 at (1955,1-1955,38) */
uint64_t __tmp_in_tmp522;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp522;
}
tmp522[i0] = (role == CLIENT) ? __tmp_in_tmp522 : 0;
}

auto tmp523 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp523 at (1958,1-1958,38) */
uint64_t __tmp_in_tmp523;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp523;
}
tmp523[i0] = (role == CLIENT) ? __tmp_in_tmp523 : 0;
}

auto tmp524 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp524 at (1961,1-1961,49) */
uint64_t __tmp_in_tmp524;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp524;
}
tmp524[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp524 : 0;
}
}
}
}

auto tmp525 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp525 at (1964,1-1964,38) */
uint64_t __tmp_in_tmp525;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp525;
}
tmp525[i0] = (role == CLIENT) ? __tmp_in_tmp525 : 0;
}

auto tmp526 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp526 at (1967,1-1967,38) */
uint64_t __tmp_in_tmp526;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp526;
}
tmp526[i0] = (role == CLIENT) ? __tmp_in_tmp526 : 0;
}

auto tmp527 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp527 at (1970,1-1970,38) */
uint64_t __tmp_in_tmp527;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp527;
}
tmp527[i0] = (role == CLIENT) ? __tmp_in_tmp527 : 0;
}

auto tmp528 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp528 at (1973,1-1973,38) */
uint64_t __tmp_in_tmp528;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp528;
}
tmp528[i0] = (role == CLIENT) ? __tmp_in_tmp528 : 0;
}

auto tmp529 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp529 at (1976,1-1976,50) */
uint64_t __tmp_in_tmp529;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp529;
}
tmp529[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp529 : 0;
}
}
}
}

auto tmp530 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp530 at (1979,1-1979,39) */
uint64_t __tmp_in_tmp530;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp530;
}
tmp530[i0] = (role == CLIENT) ? __tmp_in_tmp530 : 0;
}

auto tmp531 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp531 at (1982,1-1982,39) */
uint64_t __tmp_in_tmp531;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp531;
}
tmp531[i0] = (role == CLIENT) ? __tmp_in_tmp531 : 0;
}

auto tmp532 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp532 at (1985,1-1985,39) */
uint64_t __tmp_in_tmp532;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp532;
}
tmp532[i0] = (role == CLIENT) ? __tmp_in_tmp532 : 0;
}

auto tmp533 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp533 at (1988,1-1988,39) */
uint64_t __tmp_in_tmp533;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp533;
}
tmp533[i0] = (role == CLIENT) ? __tmp_in_tmp533 : 0;
}

auto tmp534 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp534 at (1991,1-1991,50) */
uint64_t __tmp_in_tmp534;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp534;
}
tmp534[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp534 : 0;
}
}
}
}

auto tmp535 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp535 at (1994,1-1994,38) */
uint64_t __tmp_in_tmp535;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp535;
}
tmp535[i0] = (role == CLIENT) ? __tmp_in_tmp535 : 0;
}

auto tmp536 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp536 at (1997,1-1997,38) */
uint64_t __tmp_in_tmp536;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp536;
}
tmp536[i0] = (role == CLIENT) ? __tmp_in_tmp536 : 0;
}

auto tmp537 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp537 at (2000,1-2000,38) */
uint64_t __tmp_in_tmp537;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp537;
}
tmp537[i0] = (role == CLIENT) ? __tmp_in_tmp537 : 0;
}

auto tmp538 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp538 at (2003,1-2003,38) */
uint64_t __tmp_in_tmp538;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp538;
}
tmp538[i0] = (role == CLIENT) ? __tmp_in_tmp538 : 0;
}

auto tmp539 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp539 at (2006,1-2006,49) */
uint64_t __tmp_in_tmp539;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp539;
}
tmp539[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp539 : 0;
}
}
}
}

auto tmp540 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp540 at (2009,1-2009,38) */
uint64_t __tmp_in_tmp540;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp540;
}
tmp540[i0] = (role == CLIENT) ? __tmp_in_tmp540 : 0;
}

auto tmp541 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp541 at (2012,1-2012,38) */
uint64_t __tmp_in_tmp541;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp541;
}
tmp541[i0] = (role == CLIENT) ? __tmp_in_tmp541 : 0;
}

auto tmp542 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp542 at (2015,1-2015,38) */
uint64_t __tmp_in_tmp542;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp542;
}
tmp542[i0] = (role == CLIENT) ? __tmp_in_tmp542 : 0;
}

auto tmp543 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp543 at (2018,1-2018,38) */
uint64_t __tmp_in_tmp543;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp543;
}
tmp543[i0] = (role == CLIENT) ? __tmp_in_tmp543 : 0;
}

auto tmp544 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp544 at (2021,1-2021,50) */
uint64_t __tmp_in_tmp544;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp544;
}
tmp544[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp544 : 0;
}
}
}
}

auto tmp545 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp545 at (2024,1-2024,39) */
uint64_t __tmp_in_tmp545;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp545;
}
tmp545[i0] = (role == CLIENT) ? __tmp_in_tmp545 : 0;
}

auto tmp546 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp546 at (2027,1-2027,39) */
uint64_t __tmp_in_tmp546;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp546;
}
tmp546[i0] = (role == CLIENT) ? __tmp_in_tmp546 : 0;
}

auto tmp547 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp547 at (2030,1-2030,39) */
uint64_t __tmp_in_tmp547;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp547;
}
tmp547[i0] = (role == CLIENT) ? __tmp_in_tmp547 : 0;
}

auto tmp548 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp548 at (2033,1-2033,39) */
uint64_t __tmp_in_tmp548;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp548;
}
tmp548[i0] = (role == CLIENT) ? __tmp_in_tmp548 : 0;
}

auto tmp549 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp549 at (2036,1-2036,50) */
uint64_t __tmp_in_tmp549;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp549;
}
tmp549[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp549 : 0;
}
}
}
}

auto tmp550 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp550 at (2039,1-2039,38) */
uint64_t __tmp_in_tmp550;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp550;
}
tmp550[i0] = (role == CLIENT) ? __tmp_in_tmp550 : 0;
}

auto tmp551 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp551 at (2042,1-2042,38) */
uint64_t __tmp_in_tmp551;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp551;
}
tmp551[i0] = (role == CLIENT) ? __tmp_in_tmp551 : 0;
}

auto tmp552 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp552 at (2045,1-2045,38) */
uint64_t __tmp_in_tmp552;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp552;
}
tmp552[i0] = (role == CLIENT) ? __tmp_in_tmp552 : 0;
}

auto tmp553 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp553 at (2048,1-2048,38) */
uint64_t __tmp_in_tmp553;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp553;
}
tmp553[i0] = (role == CLIENT) ? __tmp_in_tmp553 : 0;
}

auto tmp554 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp554 at (2051,1-2051,49) */
uint64_t __tmp_in_tmp554;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp554;
}
tmp554[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp554 : 0;
}
}
}
}

auto tmp555 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp555 at (2054,1-2054,38) */
uint64_t __tmp_in_tmp555;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp555;
}
tmp555[i0] = (role == CLIENT) ? __tmp_in_tmp555 : 0;
}

auto tmp556 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp556 at (2057,1-2057,38) */
uint64_t __tmp_in_tmp556;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp556;
}
tmp556[i0] = (role == CLIENT) ? __tmp_in_tmp556 : 0;
}

auto tmp557 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp557 at (2060,1-2060,38) */
uint64_t __tmp_in_tmp557;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp557;
}
tmp557[i0] = (role == CLIENT) ? __tmp_in_tmp557 : 0;
}

auto tmp558 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp558 at (2063,1-2063,38) */
uint64_t __tmp_in_tmp558;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp558;
}
tmp558[i0] = (role == CLIENT) ? __tmp_in_tmp558 : 0;
}

auto tmp559 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp559 at (2066,1-2066,50) */
uint64_t __tmp_in_tmp559;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp559;
}
tmp559[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp559 : 0;
}
}
}
}

auto tmp560 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp560 at (2069,1-2069,39) */
uint64_t __tmp_in_tmp560;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp560;
}
tmp560[i0] = (role == CLIENT) ? __tmp_in_tmp560 : 0;
}

auto tmp561 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp561 at (2072,1-2072,39) */
uint64_t __tmp_in_tmp561;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp561;
}
tmp561[i0] = (role == CLIENT) ? __tmp_in_tmp561 : 0;
}

auto tmp562 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp562 at (2075,1-2075,39) */
uint64_t __tmp_in_tmp562;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp562;
}
tmp562[i0] = (role == CLIENT) ? __tmp_in_tmp562 : 0;
}

auto tmp563 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp563 at (2078,1-2078,39) */
uint64_t __tmp_in_tmp563;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp563;
}
tmp563[i0] = (role == CLIENT) ? __tmp_in_tmp563 : 0;
}

auto tmp564 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp564 at (2081,1-2081,50) */
uint64_t __tmp_in_tmp564;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp564;
}
tmp564[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp564 : 0;
}
}
}
}

auto tmp565 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp565 at (2084,1-2084,38) */
uint64_t __tmp_in_tmp565;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp565;
}
tmp565[i0] = (role == CLIENT) ? __tmp_in_tmp565 : 0;
}

auto tmp566 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp566 at (2087,1-2087,38) */
uint64_t __tmp_in_tmp566;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp566;
}
tmp566[i0] = (role == CLIENT) ? __tmp_in_tmp566 : 0;
}

auto tmp567 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp567 at (2090,1-2090,38) */
uint64_t __tmp_in_tmp567;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp567;
}
tmp567[i0] = (role == CLIENT) ? __tmp_in_tmp567 : 0;
}

auto tmp568 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp568 at (2093,1-2093,38) */
uint64_t __tmp_in_tmp568;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp568;
}
tmp568[i0] = (role == CLIENT) ? __tmp_in_tmp568 : 0;
}

auto tmp569 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp569 at (2096,1-2096,49) */
uint64_t __tmp_in_tmp569;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp569;
}
tmp569[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp569 : 0;
}
}
}
}

auto tmp570 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp570 at (2099,1-2099,38) */
uint64_t __tmp_in_tmp570;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp570;
}
tmp570[i0] = (role == CLIENT) ? __tmp_in_tmp570 : 0;
}

auto tmp571 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp571 at (2102,1-2102,38) */
uint64_t __tmp_in_tmp571;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp571;
}
tmp571[i0] = (role == CLIENT) ? __tmp_in_tmp571 : 0;
}

auto tmp572 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp572 at (2105,1-2105,38) */
uint64_t __tmp_in_tmp572;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp572;
}
tmp572[i0] = (role == CLIENT) ? __tmp_in_tmp572 : 0;
}

auto tmp573 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp573 at (2108,1-2108,38) */
uint64_t __tmp_in_tmp573;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp573;
}
tmp573[i0] = (role == CLIENT) ? __tmp_in_tmp573 : 0;
}

auto tmp574 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp574 at (2111,1-2111,50) */
uint64_t __tmp_in_tmp574;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp574;
}
tmp574[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp574 : 0;
}
}
}
}

auto tmp575 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp575 at (2114,1-2114,39) */
uint64_t __tmp_in_tmp575;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp575;
}
tmp575[i0] = (role == CLIENT) ? __tmp_in_tmp575 : 0;
}

auto tmp576 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp576 at (2117,1-2117,39) */
uint64_t __tmp_in_tmp576;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp576;
}
tmp576[i0] = (role == CLIENT) ? __tmp_in_tmp576 : 0;
}

auto tmp577 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp577 at (2120,1-2120,39) */
uint64_t __tmp_in_tmp577;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp577;
}
tmp577[i0] = (role == CLIENT) ? __tmp_in_tmp577 : 0;
}

auto tmp578 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp578 at (2123,1-2123,39) */
uint64_t __tmp_in_tmp578;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp578;
}
tmp578[i0] = (role == CLIENT) ? __tmp_in_tmp578 : 0;
}

auto tmp579 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp579 at (2126,1-2126,50) */
uint64_t __tmp_in_tmp579;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp579;
}
tmp579[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp579 : 0;
}
}
}
}

auto tmp580 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp580 at (2129,1-2129,38) */
uint64_t __tmp_in_tmp580;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp580;
}
tmp580[i0] = (role == CLIENT) ? __tmp_in_tmp580 : 0;
}

auto tmp581 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp581 at (2132,1-2132,38) */
uint64_t __tmp_in_tmp581;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp581;
}
tmp581[i0] = (role == CLIENT) ? __tmp_in_tmp581 : 0;
}

auto tmp582 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp582 at (2135,1-2135,38) */
uint64_t __tmp_in_tmp582;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp582;
}
tmp582[i0] = (role == CLIENT) ? __tmp_in_tmp582 : 0;
}

auto tmp583 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp583 at (2138,1-2138,38) */
uint64_t __tmp_in_tmp583;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp583;
}
tmp583[i0] = (role == CLIENT) ? __tmp_in_tmp583 : 0;
}

auto tmp584 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp584 at (2141,1-2141,49) */
uint64_t __tmp_in_tmp584;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp584;
}
tmp584[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp584 : 0;
}
}
}
}

auto tmp585 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp585 at (2144,1-2144,38) */
uint64_t __tmp_in_tmp585;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp585;
}
tmp585[i0] = (role == CLIENT) ? __tmp_in_tmp585 : 0;
}

auto tmp586 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp586 at (2147,1-2147,38) */
uint64_t __tmp_in_tmp586;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp586;
}
tmp586[i0] = (role == CLIENT) ? __tmp_in_tmp586 : 0;
}

auto tmp587 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp587 at (2150,1-2150,38) */
uint64_t __tmp_in_tmp587;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp587;
}
tmp587[i0] = (role == CLIENT) ? __tmp_in_tmp587 : 0;
}

auto tmp588 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp588 at (2153,1-2153,38) */
uint64_t __tmp_in_tmp588;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp588;
}
tmp588[i0] = (role == CLIENT) ? __tmp_in_tmp588 : 0;
}

auto tmp589 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp589 at (2156,1-2156,50) */
uint64_t __tmp_in_tmp589;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp589;
}
tmp589[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp589 : 0;
}
}
}
}

auto tmp590 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp590 at (2159,1-2159,39) */
uint64_t __tmp_in_tmp590;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp590;
}
tmp590[i0] = (role == CLIENT) ? __tmp_in_tmp590 : 0;
}

auto tmp591 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp591 at (2162,1-2162,39) */
uint64_t __tmp_in_tmp591;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp591;
}
tmp591[i0] = (role == CLIENT) ? __tmp_in_tmp591 : 0;
}

auto tmp592 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp592 at (2165,1-2165,39) */
uint64_t __tmp_in_tmp592;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp592;
}
tmp592[i0] = (role == CLIENT) ? __tmp_in_tmp592 : 0;
}

auto tmp593 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp593 at (2168,1-2168,39) */
uint64_t __tmp_in_tmp593;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp593;
}
tmp593[i0] = (role == CLIENT) ? __tmp_in_tmp593 : 0;
}

auto tmp594 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp594 at (2171,1-2171,50) */
uint64_t __tmp_in_tmp594;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp594;
}
tmp594[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp594 : 0;
}
}
}
}

auto tmp595 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp595 at (2174,1-2174,38) */
uint64_t __tmp_in_tmp595;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp595;
}
tmp595[i0] = (role == CLIENT) ? __tmp_in_tmp595 : 0;
}

auto tmp596 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp596 at (2177,1-2177,38) */
uint64_t __tmp_in_tmp596;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp596;
}
tmp596[i0] = (role == CLIENT) ? __tmp_in_tmp596 : 0;
}

auto tmp597 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp597 at (2180,1-2180,38) */
uint64_t __tmp_in_tmp597;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp597;
}
tmp597[i0] = (role == CLIENT) ? __tmp_in_tmp597 : 0;
}

auto tmp598 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp598 at (2183,1-2183,38) */
uint64_t __tmp_in_tmp598;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp598;
}
tmp598[i0] = (role == CLIENT) ? __tmp_in_tmp598 : 0;
}

auto tmp599 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp599 at (2186,1-2186,49) */
uint64_t __tmp_in_tmp599;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp599;
}
tmp599[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp599 : 0;
}
}
}
}

auto tmp600 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp600 at (2189,1-2189,38) */
uint64_t __tmp_in_tmp600;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp600;
}
tmp600[i0] = (role == CLIENT) ? __tmp_in_tmp600 : 0;
}

auto tmp601 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp601 at (2192,1-2192,38) */
uint64_t __tmp_in_tmp601;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp601;
}
tmp601[i0] = (role == CLIENT) ? __tmp_in_tmp601 : 0;
}

auto tmp602 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp602 at (2195,1-2195,38) */
uint64_t __tmp_in_tmp602;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp602;
}
tmp602[i0] = (role == CLIENT) ? __tmp_in_tmp602 : 0;
}

auto tmp603 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp603 at (2198,1-2198,38) */
uint64_t __tmp_in_tmp603;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp603;
}
tmp603[i0] = (role == CLIENT) ? __tmp_in_tmp603 : 0;
}

auto tmp604 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp604 at (2201,1-2201,50) */
uint64_t __tmp_in_tmp604;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp604;
}
tmp604[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp604 : 0;
}
}
}
}

auto tmp605 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp605 at (2204,1-2204,39) */
uint64_t __tmp_in_tmp605;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp605;
}
tmp605[i0] = (role == CLIENT) ? __tmp_in_tmp605 : 0;
}

auto tmp606 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp606 at (2207,1-2207,39) */
uint64_t __tmp_in_tmp606;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp606;
}
tmp606[i0] = (role == CLIENT) ? __tmp_in_tmp606 : 0;
}

auto tmp607 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp607 at (2210,1-2210,39) */
uint64_t __tmp_in_tmp607;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp607;
}
tmp607[i0] = (role == CLIENT) ? __tmp_in_tmp607 : 0;
}

auto tmp608 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp608 at (2213,1-2213,39) */
uint64_t __tmp_in_tmp608;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp608;
}
tmp608[i0] = (role == CLIENT) ? __tmp_in_tmp608 : 0;
}

auto tmp609 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp609 at (2216,1-2216,50) */
uint64_t __tmp_in_tmp609;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp609;
}
tmp609[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp609 : 0;
}
}
}
}

auto tmp610 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp610 at (2219,1-2219,38) */
uint64_t __tmp_in_tmp610;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp610;
}
tmp610[i0] = (role == CLIENT) ? __tmp_in_tmp610 : 0;
}

auto tmp611 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp611 at (2222,1-2222,38) */
uint64_t __tmp_in_tmp611;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp611;
}
tmp611[i0] = (role == CLIENT) ? __tmp_in_tmp611 : 0;
}

auto tmp612 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp612 at (2225,1-2225,38) */
uint64_t __tmp_in_tmp612;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp612;
}
tmp612[i0] = (role == CLIENT) ? __tmp_in_tmp612 : 0;
}

auto tmp613 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp613 at (2228,1-2228,38) */
uint64_t __tmp_in_tmp613;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp613;
}
tmp613[i0] = (role == CLIENT) ? __tmp_in_tmp613 : 0;
}

auto tmp614 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp614 at (2231,1-2231,49) */
uint64_t __tmp_in_tmp614;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp614;
}
tmp614[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp614 : 0;
}
}
}
}

auto tmp615 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp615 at (2234,1-2234,38) */
uint64_t __tmp_in_tmp615;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp615;
}
tmp615[i0] = (role == CLIENT) ? __tmp_in_tmp615 : 0;
}

auto tmp616 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp616 at (2237,1-2237,38) */
uint64_t __tmp_in_tmp616;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp616;
}
tmp616[i0] = (role == CLIENT) ? __tmp_in_tmp616 : 0;
}

auto tmp617 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp617 at (2240,1-2240,38) */
uint64_t __tmp_in_tmp617;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp617;
}
tmp617[i0] = (role == CLIENT) ? __tmp_in_tmp617 : 0;
}

auto tmp618 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp618 at (2243,1-2243,38) */
uint64_t __tmp_in_tmp618;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp618;
}
tmp618[i0] = (role == CLIENT) ? __tmp_in_tmp618 : 0;
}

auto tmp619 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp619 at (2246,1-2246,50) */
uint64_t __tmp_in_tmp619;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp619;
}
tmp619[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp619 : 0;
}
}
}
}

auto tmp620 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp620 at (2249,1-2249,39) */
uint64_t __tmp_in_tmp620;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp620;
}
tmp620[i0] = (role == CLIENT) ? __tmp_in_tmp620 : 0;
}

auto tmp621 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp621 at (2252,1-2252,39) */
uint64_t __tmp_in_tmp621;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp621;
}
tmp621[i0] = (role == CLIENT) ? __tmp_in_tmp621 : 0;
}

auto tmp622 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp622 at (2255,1-2255,39) */
uint64_t __tmp_in_tmp622;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp622;
}
tmp622[i0] = (role == CLIENT) ? __tmp_in_tmp622 : 0;
}

auto tmp623 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp623 at (2258,1-2258,39) */
uint64_t __tmp_in_tmp623;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp623;
}
tmp623[i0] = (role == CLIENT) ? __tmp_in_tmp623 : 0;
}

auto tmp624 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp624 at (2261,1-2261,50) */
uint64_t __tmp_in_tmp624;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp624;
}
tmp624[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp624 : 0;
}
}
}
}

auto tmp625 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp625 at (2264,1-2264,38) */
uint64_t __tmp_in_tmp625;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp625;
}
tmp625[i0] = (role == CLIENT) ? __tmp_in_tmp625 : 0;
}

auto tmp626 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp626 at (2267,1-2267,38) */
uint64_t __tmp_in_tmp626;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp626;
}
tmp626[i0] = (role == CLIENT) ? __tmp_in_tmp626 : 0;
}

auto tmp627 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp627 at (2270,1-2270,38) */
uint64_t __tmp_in_tmp627;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp627;
}
tmp627[i0] = (role == CLIENT) ? __tmp_in_tmp627 : 0;
}

auto tmp628 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp628 at (2273,1-2273,38) */
uint64_t __tmp_in_tmp628;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp628;
}
tmp628[i0] = (role == CLIENT) ? __tmp_in_tmp628 : 0;
}

auto tmp629 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp629 at (2276,1-2276,49) */
uint64_t __tmp_in_tmp629;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp629;
}
tmp629[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp629 : 0;
}
}
}
}

auto tmp630 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp630 at (2279,1-2279,38) */
uint64_t __tmp_in_tmp630;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp630;
}
tmp630[i0] = (role == CLIENT) ? __tmp_in_tmp630 : 0;
}

auto tmp631 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp631 at (2282,1-2282,38) */
uint64_t __tmp_in_tmp631;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp631;
}
tmp631[i0] = (role == CLIENT) ? __tmp_in_tmp631 : 0;
}

auto tmp632 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp632 at (2285,1-2285,38) */
uint64_t __tmp_in_tmp632;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp632;
}
tmp632[i0] = (role == CLIENT) ? __tmp_in_tmp632 : 0;
}

auto tmp633 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp633 at (2288,1-2288,38) */
uint64_t __tmp_in_tmp633;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp633;
}
tmp633[i0] = (role == CLIENT) ? __tmp_in_tmp633 : 0;
}

auto tmp634 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp634 at (2291,1-2291,50) */
uint64_t __tmp_in_tmp634;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp634;
}
tmp634[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp634 : 0;
}
}
}
}

auto tmp635 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp635 at (2294,1-2294,39) */
uint64_t __tmp_in_tmp635;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp635;
}
tmp635[i0] = (role == CLIENT) ? __tmp_in_tmp635 : 0;
}

auto tmp636 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp636 at (2297,1-2297,39) */
uint64_t __tmp_in_tmp636;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp636;
}
tmp636[i0] = (role == CLIENT) ? __tmp_in_tmp636 : 0;
}

auto tmp637 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp637 at (2300,1-2300,39) */
uint64_t __tmp_in_tmp637;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp637;
}
tmp637[i0] = (role == CLIENT) ? __tmp_in_tmp637 : 0;
}

auto tmp638 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp638 at (2303,1-2303,39) */
uint64_t __tmp_in_tmp638;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp638;
}
tmp638[i0] = (role == CLIENT) ? __tmp_in_tmp638 : 0;
}

auto tmp639 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp639 at (2306,1-2306,50) */
uint64_t __tmp_in_tmp639;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp639;
}
tmp639[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp639 : 0;
}
}
}
}

auto tmp640 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp640 at (2309,1-2309,38) */
uint64_t __tmp_in_tmp640;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp640;
}
tmp640[i0] = (role == CLIENT) ? __tmp_in_tmp640 : 0;
}

auto tmp641 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp641 at (2312,1-2312,38) */
uint64_t __tmp_in_tmp641;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp641;
}
tmp641[i0] = (role == CLIENT) ? __tmp_in_tmp641 : 0;
}

auto tmp642 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp642 at (2315,1-2315,38) */
uint64_t __tmp_in_tmp642;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp642;
}
tmp642[i0] = (role == CLIENT) ? __tmp_in_tmp642 : 0;
}

auto tmp643 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp643 at (2318,1-2318,38) */
uint64_t __tmp_in_tmp643;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp643;
}
tmp643[i0] = (role == CLIENT) ? __tmp_in_tmp643 : 0;
}

auto tmp644 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp644 at (2321,1-2321,49) */
uint64_t __tmp_in_tmp644;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp644;
}
tmp644[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp644 : 0;
}
}
}
}

auto tmp645 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp645 at (2324,1-2324,38) */
uint64_t __tmp_in_tmp645;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp645;
}
tmp645[i0] = (role == CLIENT) ? __tmp_in_tmp645 : 0;
}

auto tmp646 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp646 at (2327,1-2327,38) */
uint64_t __tmp_in_tmp646;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp646;
}
tmp646[i0] = (role == CLIENT) ? __tmp_in_tmp646 : 0;
}

auto tmp647 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp647 at (2330,1-2330,38) */
uint64_t __tmp_in_tmp647;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp647;
}
tmp647[i0] = (role == CLIENT) ? __tmp_in_tmp647 : 0;
}

auto tmp648 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp648 at (2333,1-2333,38) */
uint64_t __tmp_in_tmp648;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp648;
}
tmp648[i0] = (role == CLIENT) ? __tmp_in_tmp648 : 0;
}

auto tmp649 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp649 at (2336,1-2336,50) */
uint64_t __tmp_in_tmp649;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp649;
}
tmp649[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp649 : 0;
}
}
}
}

auto tmp650 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp650 at (2339,1-2339,39) */
uint64_t __tmp_in_tmp650;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp650;
}
tmp650[i0] = (role == CLIENT) ? __tmp_in_tmp650 : 0;
}

auto tmp651 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp651 at (2342,1-2342,39) */
uint64_t __tmp_in_tmp651;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp651;
}
tmp651[i0] = (role == CLIENT) ? __tmp_in_tmp651 : 0;
}

auto tmp652 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp652 at (2345,1-2345,39) */
uint64_t __tmp_in_tmp652;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp652;
}
tmp652[i0] = (role == CLIENT) ? __tmp_in_tmp652 : 0;
}

auto tmp653 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp653 at (2348,1-2348,39) */
uint64_t __tmp_in_tmp653;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp653;
}
tmp653[i0] = (role == CLIENT) ? __tmp_in_tmp653 : 0;
}

auto tmp654 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp654 at (2351,1-2351,50) */
uint64_t __tmp_in_tmp654;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp654;
}
tmp654[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp654 : 0;
}
}
}
}

auto tmp655 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp655 at (2354,1-2354,38) */
uint64_t __tmp_in_tmp655;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp655;
}
tmp655[i0] = (role == CLIENT) ? __tmp_in_tmp655 : 0;
}

auto tmp656 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp656 at (2357,1-2357,38) */
uint64_t __tmp_in_tmp656;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp656;
}
tmp656[i0] = (role == CLIENT) ? __tmp_in_tmp656 : 0;
}

auto tmp657 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp657 at (2360,1-2360,38) */
uint64_t __tmp_in_tmp657;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp657;
}
tmp657[i0] = (role == CLIENT) ? __tmp_in_tmp657 : 0;
}

auto tmp658 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp658 at (2363,1-2363,38) */
uint64_t __tmp_in_tmp658;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp658;
}
tmp658[i0] = (role == CLIENT) ? __tmp_in_tmp658 : 0;
}

auto tmp659 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp659 at (2366,1-2366,49) */
uint64_t __tmp_in_tmp659;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp659;
}
tmp659[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp659 : 0;
}
}
}
}

auto tmp660 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp660 at (2369,1-2369,38) */
uint64_t __tmp_in_tmp660;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp660;
}
tmp660[i0] = (role == CLIENT) ? __tmp_in_tmp660 : 0;
}

auto tmp661 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp661 at (2372,1-2372,38) */
uint64_t __tmp_in_tmp661;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp661;
}
tmp661[i0] = (role == CLIENT) ? __tmp_in_tmp661 : 0;
}

auto tmp662 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp662 at (2375,1-2375,38) */
uint64_t __tmp_in_tmp662;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp662;
}
tmp662[i0] = (role == CLIENT) ? __tmp_in_tmp662 : 0;
}

auto tmp663 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp663 at (2378,1-2378,38) */
uint64_t __tmp_in_tmp663;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp663;
}
tmp663[i0] = (role == CLIENT) ? __tmp_in_tmp663 : 0;
}

auto tmp664 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp664 at (2381,1-2381,50) */
uint64_t __tmp_in_tmp664;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp664;
}
tmp664[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp664 : 0;
}
}
}
}

auto tmp665 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp665 at (2384,1-2384,39) */
uint64_t __tmp_in_tmp665;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp665;
}
tmp665[i0] = (role == CLIENT) ? __tmp_in_tmp665 : 0;
}

auto tmp666 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp666 at (2387,1-2387,39) */
uint64_t __tmp_in_tmp666;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp666;
}
tmp666[i0] = (role == CLIENT) ? __tmp_in_tmp666 : 0;
}

auto tmp667 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp667 at (2390,1-2390,39) */
uint64_t __tmp_in_tmp667;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp667;
}
tmp667[i0] = (role == CLIENT) ? __tmp_in_tmp667 : 0;
}

auto tmp668 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp668 at (2393,1-2393,39) */
uint64_t __tmp_in_tmp668;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp668;
}
tmp668[i0] = (role == CLIENT) ? __tmp_in_tmp668 : 0;
}

auto tmp669 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp669 at (2396,1-2396,50) */
uint64_t __tmp_in_tmp669;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp669;
}
tmp669[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp669 : 0;
}
}
}
}

auto tmp670 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp670 at (2399,1-2399,38) */
uint64_t __tmp_in_tmp670;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp670;
}
tmp670[i0] = (role == CLIENT) ? __tmp_in_tmp670 : 0;
}

auto tmp671 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp671 at (2402,1-2402,38) */
uint64_t __tmp_in_tmp671;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp671;
}
tmp671[i0] = (role == CLIENT) ? __tmp_in_tmp671 : 0;
}

auto tmp672 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp672 at (2405,1-2405,38) */
uint64_t __tmp_in_tmp672;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp672;
}
tmp672[i0] = (role == CLIENT) ? __tmp_in_tmp672 : 0;
}

auto tmp673 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp673 at (2408,1-2408,38) */
uint64_t __tmp_in_tmp673;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp673;
}
tmp673[i0] = (role == CLIENT) ? __tmp_in_tmp673 : 0;
}

auto tmp674 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp674 at (2411,1-2411,49) */
uint64_t __tmp_in_tmp674;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp674;
}
tmp674[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp674 : 0;
}
}
}
}

auto tmp675 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp675 at (2414,1-2414,38) */
uint64_t __tmp_in_tmp675;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp675;
}
tmp675[i0] = (role == CLIENT) ? __tmp_in_tmp675 : 0;
}

auto tmp676 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp676 at (2417,1-2417,38) */
uint64_t __tmp_in_tmp676;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp676;
}
tmp676[i0] = (role == CLIENT) ? __tmp_in_tmp676 : 0;
}

auto tmp677 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp677 at (2420,1-2420,38) */
uint64_t __tmp_in_tmp677;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp677;
}
tmp677[i0] = (role == CLIENT) ? __tmp_in_tmp677 : 0;
}

auto tmp678 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp678 at (2423,1-2423,38) */
uint64_t __tmp_in_tmp678;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp678;
}
tmp678[i0] = (role == CLIENT) ? __tmp_in_tmp678 : 0;
}

auto tmp679 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp679 at (2426,1-2426,50) */
uint64_t __tmp_in_tmp679;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp679;
}
tmp679[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp679 : 0;
}
}
}
}

auto tmp680 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp680 at (2429,1-2429,39) */
uint64_t __tmp_in_tmp680;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp680;
}
tmp680[i0] = (role == CLIENT) ? __tmp_in_tmp680 : 0;
}

auto tmp681 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp681 at (2432,1-2432,39) */
uint64_t __tmp_in_tmp681;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp681;
}
tmp681[i0] = (role == CLIENT) ? __tmp_in_tmp681 : 0;
}

auto tmp682 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp682 at (2435,1-2435,39) */
uint64_t __tmp_in_tmp682;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp682;
}
tmp682[i0] = (role == CLIENT) ? __tmp_in_tmp682 : 0;
}

auto tmp683 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp683 at (2438,1-2438,39) */
uint64_t __tmp_in_tmp683;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp683;
}
tmp683[i0] = (role == CLIENT) ? __tmp_in_tmp683 : 0;
}

auto tmp684 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp684 at (2441,1-2441,50) */
uint64_t __tmp_in_tmp684;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp684;
}
tmp684[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp684 : 0;
}
}
}
}

auto tmp685 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp685 at (2444,1-2444,38) */
uint64_t __tmp_in_tmp685;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp685;
}
tmp685[i0] = (role == CLIENT) ? __tmp_in_tmp685 : 0;
}

auto tmp686 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp686 at (2447,1-2447,38) */
uint64_t __tmp_in_tmp686;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp686;
}
tmp686[i0] = (role == CLIENT) ? __tmp_in_tmp686 : 0;
}

auto tmp687 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp687 at (2450,1-2450,38) */
uint64_t __tmp_in_tmp687;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp687;
}
tmp687[i0] = (role == CLIENT) ? __tmp_in_tmp687 : 0;
}

auto tmp688 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp688 at (2453,1-2453,38) */
uint64_t __tmp_in_tmp688;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp688;
}
tmp688[i0] = (role == CLIENT) ? __tmp_in_tmp688 : 0;
}

auto tmp689 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp689 at (2456,1-2456,49) */
uint64_t __tmp_in_tmp689;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp689;
}
tmp689[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp689 : 0;
}
}
}
}

auto tmp690 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp690 at (2459,1-2459,38) */
uint64_t __tmp_in_tmp690;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp690;
}
tmp690[i0] = (role == CLIENT) ? __tmp_in_tmp690 : 0;
}

auto tmp691 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp691 at (2462,1-2462,38) */
uint64_t __tmp_in_tmp691;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp691;
}
tmp691[i0] = (role == CLIENT) ? __tmp_in_tmp691 : 0;
}

auto tmp692 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp692 at (2465,1-2465,38) */
uint64_t __tmp_in_tmp692;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp692;
}
tmp692[i0] = (role == CLIENT) ? __tmp_in_tmp692 : 0;
}

auto tmp693 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp693 at (2468,1-2468,38) */
uint64_t __tmp_in_tmp693;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp693;
}
tmp693[i0] = (role == CLIENT) ? __tmp_in_tmp693 : 0;
}

auto tmp694 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp694 at (2471,1-2471,50) */
uint64_t __tmp_in_tmp694;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp694;
}
tmp694[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp694 : 0;
}
}
}
}

auto tmp695 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp695 at (2474,1-2474,39) */
uint64_t __tmp_in_tmp695;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp695;
}
tmp695[i0] = (role == CLIENT) ? __tmp_in_tmp695 : 0;
}

auto tmp696 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp696 at (2477,1-2477,39) */
uint64_t __tmp_in_tmp696;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp696;
}
tmp696[i0] = (role == CLIENT) ? __tmp_in_tmp696 : 0;
}

auto tmp697 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp697 at (2480,1-2480,39) */
uint64_t __tmp_in_tmp697;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp697;
}
tmp697[i0] = (role == CLIENT) ? __tmp_in_tmp697 : 0;
}

auto tmp698 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp698 at (2483,1-2483,39) */
uint64_t __tmp_in_tmp698;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp698;
}
tmp698[i0] = (role == CLIENT) ? __tmp_in_tmp698 : 0;
}

auto tmp699 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp699 at (2486,1-2486,50) */
uint64_t __tmp_in_tmp699;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp699;
}
tmp699[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp699 : 0;
}
}
}
}

auto tmp700 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp700 at (2489,1-2489,38) */
uint64_t __tmp_in_tmp700;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp700;
}
tmp700[i0] = (role == CLIENT) ? __tmp_in_tmp700 : 0;
}

auto tmp701 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp701 at (2492,1-2492,38) */
uint64_t __tmp_in_tmp701;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp701;
}
tmp701[i0] = (role == CLIENT) ? __tmp_in_tmp701 : 0;
}

auto tmp702 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp702 at (2495,1-2495,38) */
uint64_t __tmp_in_tmp702;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp702;
}
tmp702[i0] = (role == CLIENT) ? __tmp_in_tmp702 : 0;
}

auto tmp703 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp703 at (2498,1-2498,38) */
uint64_t __tmp_in_tmp703;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp703;
}
tmp703[i0] = (role == CLIENT) ? __tmp_in_tmp703 : 0;
}

auto tmp704 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp704 at (2501,1-2501,49) */
uint64_t __tmp_in_tmp704;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp704;
}
tmp704[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp704 : 0;
}
}
}
}

auto tmp705 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp705 at (2504,1-2504,38) */
uint64_t __tmp_in_tmp705;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp705;
}
tmp705[i0] = (role == CLIENT) ? __tmp_in_tmp705 : 0;
}

auto tmp706 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp706 at (2507,1-2507,38) */
uint64_t __tmp_in_tmp706;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp706;
}
tmp706[i0] = (role == CLIENT) ? __tmp_in_tmp706 : 0;
}

auto tmp707 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp707 at (2510,1-2510,38) */
uint64_t __tmp_in_tmp707;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp707;
}
tmp707[i0] = (role == CLIENT) ? __tmp_in_tmp707 : 0;
}

auto tmp708 = make_vector<uint64_t>( (int32_t)256);
/* Variable to read the clear value corresponding to the input variable tmp708 at (2513,1-2513,38) */
uint64_t __tmp_in_tmp708;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp708;
}
tmp708[i0] = (role == CLIENT) ? __tmp_in_tmp708 : 0;
}

auto tmp709 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp709 at (2516,1-2516,50) */
uint64_t __tmp_in_tmp709;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1024; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp709;
}
tmp709[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp709 : 0;
}
}
}
}

auto tmp710 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp710 at (2519,1-2519,39) */
uint64_t __tmp_in_tmp710;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp710;
}
tmp710[i0] = (role == CLIENT) ? __tmp_in_tmp710 : 0;
}

auto tmp711 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp711 at (2522,1-2522,39) */
uint64_t __tmp_in_tmp711;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp711;
}
tmp711[i0] = (role == CLIENT) ? __tmp_in_tmp711 : 0;
}

auto tmp712 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp712 at (2525,1-2525,39) */
uint64_t __tmp_in_tmp712;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp712;
}
tmp712[i0] = (role == CLIENT) ? __tmp_in_tmp712 : 0;
}

auto tmp713 = make_vector<uint64_t>( (int32_t)1024);
/* Variable to read the clear value corresponding to the input variable tmp713 at (2528,1-2528,39) */
uint64_t __tmp_in_tmp713;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp713;
}
tmp713[i0] = (role == CLIENT) ? __tmp_in_tmp713 : 0;
}

auto tmp714 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp714 at (2531,1-2531,51) */
uint64_t __tmp_in_tmp714;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp714;
}
tmp714[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp714 : 0;
}
}
}
}

auto tmp715 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp715 at (2534,1-2534,50) */
uint64_t __tmp_in_tmp715;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp715;
}
tmp715[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp715 : 0;
}
}
}
}

auto tmp716 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp716 at (2537,1-2537,38) */
uint64_t __tmp_in_tmp716;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp716;
}
tmp716[i0] = (role == CLIENT) ? __tmp_in_tmp716 : 0;
}

auto tmp717 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp717 at (2540,1-2540,38) */
uint64_t __tmp_in_tmp717;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp717;
}
tmp717[i0] = (role == CLIENT) ? __tmp_in_tmp717 : 0;
}

auto tmp718 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp718 at (2543,1-2543,38) */
uint64_t __tmp_in_tmp718;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp718;
}
tmp718[i0] = (role == CLIENT) ? __tmp_in_tmp718 : 0;
}

auto tmp719 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp719 at (2546,1-2546,38) */
uint64_t __tmp_in_tmp719;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp719;
}
tmp719[i0] = (role == CLIENT) ? __tmp_in_tmp719 : 0;
}

auto tmp720 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp720 at (2549,1-2549,49) */
uint64_t __tmp_in_tmp720;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp720;
}
tmp720[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp720 : 0;
}
}
}
}

auto tmp721 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp721 at (2552,1-2552,38) */
uint64_t __tmp_in_tmp721;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp721;
}
tmp721[i0] = (role == CLIENT) ? __tmp_in_tmp721 : 0;
}

auto tmp722 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp722 at (2555,1-2555,38) */
uint64_t __tmp_in_tmp722;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp722;
}
tmp722[i0] = (role == CLIENT) ? __tmp_in_tmp722 : 0;
}

auto tmp723 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp723 at (2558,1-2558,38) */
uint64_t __tmp_in_tmp723;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp723;
}
tmp723[i0] = (role == CLIENT) ? __tmp_in_tmp723 : 0;
}

auto tmp724 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp724 at (2561,1-2561,38) */
uint64_t __tmp_in_tmp724;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp724;
}
tmp724[i0] = (role == CLIENT) ? __tmp_in_tmp724 : 0;
}

auto tmp725 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp725 at (2564,1-2564,50) */
uint64_t __tmp_in_tmp725;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp725;
}
tmp725[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp725 : 0;
}
}
}
}

auto tmp726 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp726 at (2567,1-2567,39) */
uint64_t __tmp_in_tmp726;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp726;
}
tmp726[i0] = (role == CLIENT) ? __tmp_in_tmp726 : 0;
}

auto tmp727 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp727 at (2570,1-2570,39) */
uint64_t __tmp_in_tmp727;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp727;
}
tmp727[i0] = (role == CLIENT) ? __tmp_in_tmp727 : 0;
}

auto tmp728 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp728 at (2573,1-2573,39) */
uint64_t __tmp_in_tmp728;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp728;
}
tmp728[i0] = (role == CLIENT) ? __tmp_in_tmp728 : 0;
}

auto tmp729 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp729 at (2576,1-2576,39) */
uint64_t __tmp_in_tmp729;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp729;
}
tmp729[i0] = (role == CLIENT) ? __tmp_in_tmp729 : 0;
}

auto tmp730 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp730 at (2579,1-2579,50) */
uint64_t __tmp_in_tmp730;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp730;
}
tmp730[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp730 : 0;
}
}
}
}

auto tmp731 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp731 at (2582,1-2582,38) */
uint64_t __tmp_in_tmp731;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp731;
}
tmp731[i0] = (role == CLIENT) ? __tmp_in_tmp731 : 0;
}

auto tmp732 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp732 at (2585,1-2585,38) */
uint64_t __tmp_in_tmp732;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp732;
}
tmp732[i0] = (role == CLIENT) ? __tmp_in_tmp732 : 0;
}

auto tmp733 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp733 at (2588,1-2588,38) */
uint64_t __tmp_in_tmp733;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp733;
}
tmp733[i0] = (role == CLIENT) ? __tmp_in_tmp733 : 0;
}

auto tmp734 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp734 at (2591,1-2591,38) */
uint64_t __tmp_in_tmp734;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp734;
}
tmp734[i0] = (role == CLIENT) ? __tmp_in_tmp734 : 0;
}

auto tmp735 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp735 at (2594,1-2594,49) */
uint64_t __tmp_in_tmp735;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp735;
}
tmp735[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp735 : 0;
}
}
}
}

auto tmp736 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp736 at (2597,1-2597,38) */
uint64_t __tmp_in_tmp736;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp736;
}
tmp736[i0] = (role == CLIENT) ? __tmp_in_tmp736 : 0;
}

auto tmp737 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp737 at (2600,1-2600,38) */
uint64_t __tmp_in_tmp737;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp737;
}
tmp737[i0] = (role == CLIENT) ? __tmp_in_tmp737 : 0;
}

auto tmp738 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp738 at (2603,1-2603,38) */
uint64_t __tmp_in_tmp738;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp738;
}
tmp738[i0] = (role == CLIENT) ? __tmp_in_tmp738 : 0;
}

auto tmp739 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp739 at (2606,1-2606,38) */
uint64_t __tmp_in_tmp739;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp739;
}
tmp739[i0] = (role == CLIENT) ? __tmp_in_tmp739 : 0;
}

auto tmp740 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp740 at (2609,1-2609,50) */
uint64_t __tmp_in_tmp740;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp740;
}
tmp740[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp740 : 0;
}
}
}
}

auto tmp741 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp741 at (2612,1-2612,39) */
uint64_t __tmp_in_tmp741;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp741;
}
tmp741[i0] = (role == CLIENT) ? __tmp_in_tmp741 : 0;
}

auto tmp742 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp742 at (2615,1-2615,39) */
uint64_t __tmp_in_tmp742;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp742;
}
tmp742[i0] = (role == CLIENT) ? __tmp_in_tmp742 : 0;
}

auto tmp743 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp743 at (2618,1-2618,39) */
uint64_t __tmp_in_tmp743;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp743;
}
tmp743[i0] = (role == CLIENT) ? __tmp_in_tmp743 : 0;
}

auto tmp744 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp744 at (2621,1-2621,39) */
uint64_t __tmp_in_tmp744;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp744;
}
tmp744[i0] = (role == CLIENT) ? __tmp_in_tmp744 : 0;
}

auto tmp745 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp745 at (2624,1-2624,50) */
uint64_t __tmp_in_tmp745;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)2048; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp745;
}
tmp745[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp745 : 0;
}
}
}
}

auto tmp746 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp746 at (2627,1-2627,38) */
uint64_t __tmp_in_tmp746;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp746;
}
tmp746[i0] = (role == CLIENT) ? __tmp_in_tmp746 : 0;
}

auto tmp747 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp747 at (2630,1-2630,38) */
uint64_t __tmp_in_tmp747;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp747;
}
tmp747[i0] = (role == CLIENT) ? __tmp_in_tmp747 : 0;
}

auto tmp748 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp748 at (2633,1-2633,38) */
uint64_t __tmp_in_tmp748;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp748;
}
tmp748[i0] = (role == CLIENT) ? __tmp_in_tmp748 : 0;
}

auto tmp749 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp749 at (2636,1-2636,38) */
uint64_t __tmp_in_tmp749;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp749;
}
tmp749[i0] = (role == CLIENT) ? __tmp_in_tmp749 : 0;
}

auto tmp750 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp750 at (2639,1-2639,49) */
uint64_t __tmp_in_tmp750;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp750;
}
tmp750[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp750 : 0;
}
}
}
}

auto tmp751 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp751 at (2642,1-2642,38) */
uint64_t __tmp_in_tmp751;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp751;
}
tmp751[i0] = (role == CLIENT) ? __tmp_in_tmp751 : 0;
}

auto tmp752 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp752 at (2645,1-2645,38) */
uint64_t __tmp_in_tmp752;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp752;
}
tmp752[i0] = (role == CLIENT) ? __tmp_in_tmp752 : 0;
}

auto tmp753 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp753 at (2648,1-2648,38) */
uint64_t __tmp_in_tmp753;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp753;
}
tmp753[i0] = (role == CLIENT) ? __tmp_in_tmp753 : 0;
}

auto tmp754 = make_vector<uint64_t>( (int32_t)512);
/* Variable to read the clear value corresponding to the input variable tmp754 at (2651,1-2651,38) */
uint64_t __tmp_in_tmp754;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp754;
}
tmp754[i0] = (role == CLIENT) ? __tmp_in_tmp754 : 0;
}

auto tmp755 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp755 at (2654,1-2654,50) */
uint64_t __tmp_in_tmp755;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)2048; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp755;
}
tmp755[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp755 : 0;
}
}
}
}

auto tmp756 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp756 at (2657,1-2657,39) */
uint64_t __tmp_in_tmp756;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp756;
}
tmp756[i0] = (role == CLIENT) ? __tmp_in_tmp756 : 0;
}

auto tmp757 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp757 at (2660,1-2660,39) */
uint64_t __tmp_in_tmp757;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp757;
}
tmp757[i0] = (role == CLIENT) ? __tmp_in_tmp757 : 0;
}

auto tmp758 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp758 at (2663,1-2663,39) */
uint64_t __tmp_in_tmp758;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp758;
}
tmp758[i0] = (role == CLIENT) ? __tmp_in_tmp758 : 0;
}

auto tmp759 = make_vector<uint64_t>( (int32_t)2048);
/* Variable to read the clear value corresponding to the input variable tmp759 at (2666,1-2666,39) */
uint64_t __tmp_in_tmp759;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp759;
}
tmp759[i0] = (role == CLIENT) ? __tmp_in_tmp759 : 0;
}

auto tmp760 = make_vector<uint64_t>( (int32_t)2048,  (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp760 at (2669,1-2669,45) */
uint64_t __tmp_in_tmp760;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)2048; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1001; i1++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp760;
}
tmp760[i0][i1] = (role == CLIENT) ? __tmp_in_tmp760 : 0;
}
}

auto tmp761 = make_vector<uint64_t>( (int32_t)1001);
/* Variable to read the clear value corresponding to the input variable tmp761 at (2672,1-2672,39) */
uint64_t __tmp_in_tmp761;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1001; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp761;
}
tmp761[i0] = (role == CLIENT) ? __tmp_in_tmp761 : 0;
}

//Main Point

leave_time();
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("Starting main protocol");
start_m();
touch_time();

auto tmp762 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp762[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp762[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp762[ (int64_t)1][ (int64_t)0] =  (int32_t)3;
tmp762[ (int64_t)1][ (int64_t)1] =  (int32_t)3;
tmp762[ (int64_t)2][ (int64_t)0] =  (int32_t)3;
tmp762[ (int64_t)2][ (int64_t)1] =  (int32_t)3;
tmp762[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp762[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp763 = make_vector<uint64_t>( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3);
Pad442( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0,  (int32_t)4,  (int32_t)2, tmp762, tmp763);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp762);
ClearMemSecret4( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0);

auto tmp766 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp763, tmp1,  (int32_t)12, tmp766);
ClearMemSecret4( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64, tmp1);
ClearMemSecret4( (int32_t)1,  (int32_t)230,  (int32_t)230,  (int32_t)3, tmp763);

auto tmp769 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)1,  (int32_t)0,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp766, tmp769);
ClearMemSecret4( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp766);

auto tmp771 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp769, tmp2, tmp3,  (int32_t)12, tmp771);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp769);
ClearMemSecret1( (int32_t)64, tmp3);
ClearMemSecret1( (int32_t)64, tmp2);

auto tmp775 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp771, tmp775);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp771);

auto tmp777 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp775, tmp6,  (int32_t)12, tmp777);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp6);

auto tmp779 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp775, tmp7,  (int32_t)12, tmp779);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)64, tmp7);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp775);

auto tmp782 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp779, tmp8, tmp9,  (int32_t)12, tmp782);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp779);
ClearMemSecret1( (int32_t)64, tmp8);
ClearMemSecret1( (int32_t)64, tmp9);

auto tmp786 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp782, tmp786);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp782);

auto tmp788 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp786, tmp12,  (int32_t)12, tmp788);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp12);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp786);

auto tmp791 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp788, tmp13, tmp14,  (int32_t)12, tmp791);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp788);
ClearMemSecret1( (int32_t)64, tmp14);
ClearMemSecret1( (int32_t)64, tmp13);

auto tmp795 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp791, tmp795);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp791);

auto tmp797 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp795, tmp17,  (int32_t)12, tmp797);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp795);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp17);

auto tmp800 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp797, tmp777, tmp800);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp777);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp797);

auto tmp803 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp800, tmp18, tmp19,  (int32_t)12, tmp803);
ClearMemSecret1( (int32_t)256, tmp18);
ClearMemSecret1( (int32_t)256, tmp19);

auto tmp806 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp803, tmp806);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp803);

auto tmp808 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp806, tmp22,  (int32_t)12, tmp808);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp22);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp806);

auto tmp811 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp808, tmp23, tmp24,  (int32_t)12, tmp811);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp808);
ClearMemSecret1( (int32_t)64, tmp23);
ClearMemSecret1( (int32_t)64, tmp24);

auto tmp815 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp811, tmp815);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp811);

auto tmp817 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp815, tmp27,  (int32_t)12, tmp817);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp27);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp815);

auto tmp820 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp817, tmp28, tmp29,  (int32_t)12, tmp820);
ClearMemSecret1( (int32_t)64, tmp29);
ClearMemSecret1( (int32_t)64, tmp28);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp817);

auto tmp824 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp820, tmp824);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp820);

auto tmp826 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp824, tmp32,  (int32_t)12, tmp826);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp824);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp32);

auto tmp829 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp826, tmp800, tmp829);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp826);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp800);

auto tmp832 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp829, tmp33, tmp34,  (int32_t)12, tmp832);
ClearMemSecret1( (int32_t)256, tmp34);
ClearMemSecret1( (int32_t)256, tmp33);

auto tmp835 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp832, tmp835);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp832);

auto tmp837 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp835, tmp37,  (int32_t)12, tmp837);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)64, tmp37);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp835);

auto tmp840 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp837, tmp38, tmp39,  (int32_t)12, tmp840);
ClearMemSecret1( (int32_t)64, tmp39);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp837);
ClearMemSecret1( (int32_t)64, tmp38);

auto tmp844 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp840, tmp844);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp840);

auto tmp846 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp844, tmp42,  (int32_t)12, tmp846);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)64, tmp42);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp844);

auto tmp849 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp846, tmp43, tmp44,  (int32_t)12, tmp849);
ClearMemSecret1( (int32_t)64, tmp44);
ClearMemSecret1( (int32_t)64, tmp43);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp846);

auto tmp853 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp849, tmp853);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp849);

auto tmp855 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp853, tmp47,  (int32_t)12, tmp855);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp47);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp853);

auto tmp858 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
MatAdd4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp855, tmp829, tmp858);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp855);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp829);

auto tmp861 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp858, tmp48, tmp49,  (int32_t)12, tmp861);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp858);
ClearMemSecret1( (int32_t)256, tmp49);
ClearMemSecret1( (int32_t)256, tmp48);

auto tmp865 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp861, tmp865);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp861);

auto tmp867 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp867[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp867[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp867[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp867[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp867[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp867[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp867[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp867[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp868 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp865,  (int32_t)4,  (int32_t)2, tmp867, tmp868);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp867);

auto tmp870 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp868, tmp52,  (int32_t)12, tmp870);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp868);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)512, tmp52);

auto tmp873 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp865, tmp53,  (int32_t)12, tmp873);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128, tmp53);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp865);

auto tmp876 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp873, tmp54, tmp55,  (int32_t)12, tmp876);
ClearMemSecret1( (int32_t)128, tmp55);
ClearMemSecret1( (int32_t)128, tmp54);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp873);

auto tmp880 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp876, tmp880);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp876);

auto tmp882 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp882[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp882[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp882[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp882[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp882[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp882[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp882[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp882[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp883 = make_vector<uint64_t>( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128);
Pad442( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp880,  (int32_t)4,  (int32_t)2, tmp882, tmp883);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp882);
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp880);

auto tmp886 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp883, tmp58,  (int32_t)12, tmp886);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp58);
ClearMemSecret4( (int32_t)1,  (int32_t)58,  (int32_t)58,  (int32_t)128, tmp883);

auto tmp889 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp886, tmp59, tmp60,  (int32_t)12, tmp889);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp886);
ClearMemSecret1( (int32_t)128, tmp59);
ClearMemSecret1( (int32_t)128, tmp60);

auto tmp893 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp889, tmp893);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp889);

auto tmp895 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp893, tmp63,  (int32_t)12, tmp895);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp63);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp893);

auto tmp898 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp895, tmp870, tmp898);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp870);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp895);

auto tmp901 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp898, tmp64, tmp65,  (int32_t)12, tmp901);
ClearMemSecret1( (int32_t)512, tmp65);
ClearMemSecret1( (int32_t)512, tmp64);

auto tmp904 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp901, tmp904);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp901);

auto tmp906 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp904, tmp68,  (int32_t)12, tmp906);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp904);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp68);

auto tmp909 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp906, tmp69, tmp70,  (int32_t)12, tmp909);
ClearMemSecret1( (int32_t)128, tmp69);
ClearMemSecret1( (int32_t)128, tmp70);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp906);

auto tmp913 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp909, tmp913);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp909);

auto tmp915 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp913, tmp73,  (int32_t)12, tmp915);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp73);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp913);

auto tmp918 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp915, tmp74, tmp75,  (int32_t)12, tmp918);
ClearMemSecret1( (int32_t)128, tmp74);
ClearMemSecret1( (int32_t)128, tmp75);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp915);

auto tmp922 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp918, tmp922);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp918);

auto tmp924 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp922, tmp78,  (int32_t)12, tmp924);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp78);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp922);

auto tmp927 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp924, tmp898, tmp927);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp924);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp898);

auto tmp930 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp927, tmp79, tmp80,  (int32_t)12, tmp930);
ClearMemSecret1( (int32_t)512, tmp79);
ClearMemSecret1( (int32_t)512, tmp80);

auto tmp933 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp930, tmp933);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp930);

auto tmp935 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp933, tmp83,  (int32_t)12, tmp935);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp83);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp933);

auto tmp938 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp935, tmp84, tmp85,  (int32_t)12, tmp938);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp935);
ClearMemSecret1( (int32_t)128, tmp84);
ClearMemSecret1( (int32_t)128, tmp85);

auto tmp942 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp938, tmp942);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp938);

auto tmp944 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp942, tmp88,  (int32_t)12, tmp944);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp88);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp942);

auto tmp947 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp944, tmp89, tmp90,  (int32_t)12, tmp947);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp944);
ClearMemSecret1( (int32_t)128, tmp89);
ClearMemSecret1( (int32_t)128, tmp90);

auto tmp951 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp947, tmp951);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp947);

auto tmp953 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp951, tmp93,  (int32_t)12, tmp953);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp951);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp93);

auto tmp956 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp953, tmp927, tmp956);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp927);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp953);

auto tmp959 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp956, tmp94, tmp95,  (int32_t)12, tmp959);
ClearMemSecret1( (int32_t)512, tmp95);
ClearMemSecret1( (int32_t)512, tmp94);

auto tmp962 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp959, tmp962);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp959);

auto tmp964 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp962, tmp98,  (int32_t)12, tmp964);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp98);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp962);

auto tmp967 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp964, tmp99, tmp100,  (int32_t)12, tmp967);
ClearMemSecret1( (int32_t)128, tmp100);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp964);
ClearMemSecret1( (int32_t)128, tmp99);

auto tmp971 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp967, tmp971);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp967);

auto tmp973 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp971, tmp103,  (int32_t)12, tmp973);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp971);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp103);

auto tmp976 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp973, tmp104, tmp105,  (int32_t)12, tmp976);
ClearMemSecret1( (int32_t)128, tmp104);
ClearMemSecret1( (int32_t)128, tmp105);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp973);

auto tmp980 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp976, tmp980);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp976);

auto tmp982 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp980, tmp108,  (int32_t)12, tmp982);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp108);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp980);

auto tmp985 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp982, tmp956, tmp985);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp956);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp982);

auto tmp988 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp985, tmp109, tmp110,  (int32_t)12, tmp988);
ClearMemSecret1( (int32_t)512, tmp109);
ClearMemSecret1( (int32_t)512, tmp110);

auto tmp991 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp988, tmp991);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp988);

auto tmp993 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp991, tmp113,  (int32_t)12, tmp993);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp991);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp113);

auto tmp996 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp993, tmp114, tmp115,  (int32_t)12, tmp996);
ClearMemSecret1( (int32_t)128, tmp115);
ClearMemSecret1( (int32_t)128, tmp114);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp993);

auto tmp1000 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp996, tmp1000);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp996);

auto tmp1002 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1000, tmp118,  (int32_t)12, tmp1002);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1000);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp118);

auto tmp1005 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1002, tmp119, tmp120,  (int32_t)12, tmp1005);
ClearMemSecret1( (int32_t)128, tmp120);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1002);
ClearMemSecret1( (int32_t)128, tmp119);

auto tmp1009 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1005, tmp1009);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1005);

auto tmp1011 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1009, tmp123,  (int32_t)12, tmp1011);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1009);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp123);

auto tmp1014 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1011, tmp985, tmp1014);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp985);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1011);

auto tmp1017 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1014, tmp124, tmp125,  (int32_t)12, tmp1017);
ClearMemSecret1( (int32_t)512, tmp124);
ClearMemSecret1( (int32_t)512, tmp125);

auto tmp1020 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1017, tmp1020);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1017);

auto tmp1022 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1020, tmp128,  (int32_t)12, tmp1022);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp128);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1020);

auto tmp1025 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1022, tmp129, tmp130,  (int32_t)12, tmp1025);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1022);
ClearMemSecret1( (int32_t)128, tmp130);
ClearMemSecret1( (int32_t)128, tmp129);

auto tmp1029 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1025, tmp1029);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1025);

auto tmp1031 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1029, tmp133,  (int32_t)12, tmp1031);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp133);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1029);

auto tmp1034 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1031, tmp134, tmp135,  (int32_t)12, tmp1034);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1031);
ClearMemSecret1( (int32_t)128, tmp135);
ClearMemSecret1( (int32_t)128, tmp134);

auto tmp1038 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1034, tmp1038);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1034);

auto tmp1040 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1038, tmp138,  (int32_t)12, tmp1040);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1038);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp138);

auto tmp1043 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1040, tmp1014, tmp1043);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1014);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1040);

auto tmp1046 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1043, tmp139, tmp140,  (int32_t)12, tmp1046);
ClearMemSecret1( (int32_t)512, tmp140);
ClearMemSecret1( (int32_t)512, tmp139);

auto tmp1049 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1046, tmp1049);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1046);

auto tmp1051 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1049, tmp143,  (int32_t)12, tmp1051);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1049);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp143);

auto tmp1054 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1051, tmp144, tmp145,  (int32_t)12, tmp1054);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1051);
ClearMemSecret1( (int32_t)128, tmp144);
ClearMemSecret1( (int32_t)128, tmp145);

auto tmp1058 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1054, tmp1058);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1054);

auto tmp1060 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1058, tmp148,  (int32_t)12, tmp1060);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp148);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1058);

auto tmp1063 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1060, tmp149, tmp150,  (int32_t)12, tmp1063);
ClearMemSecret1( (int32_t)128, tmp149);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1060);
ClearMemSecret1( (int32_t)128, tmp150);

auto tmp1067 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1063, tmp1067);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1063);

auto tmp1069 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1067, tmp153,  (int32_t)12, tmp1069);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1067);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp153);

auto tmp1072 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1069, tmp1043, tmp1072);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1069);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1043);

auto tmp1075 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1072, tmp154, tmp155,  (int32_t)12, tmp1075);
ClearMemSecret1( (int32_t)512, tmp155);
ClearMemSecret1( (int32_t)512, tmp154);

auto tmp1078 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1075, tmp1078);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1075);

auto tmp1080 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1078, tmp158,  (int32_t)12, tmp1080);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp158);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1078);

auto tmp1083 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1080, tmp159, tmp160,  (int32_t)12, tmp1083);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1080);
ClearMemSecret1( (int32_t)128, tmp160);
ClearMemSecret1( (int32_t)128, tmp159);

auto tmp1087 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1083, tmp1087);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1083);

auto tmp1089 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1087, tmp163,  (int32_t)12, tmp1089);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)128, tmp163);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1087);

auto tmp1092 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1089, tmp164, tmp165,  (int32_t)12, tmp1092);
ClearMemSecret1( (int32_t)128, tmp165);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1089);
ClearMemSecret1( (int32_t)128, tmp164);

auto tmp1096 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1092, tmp1096);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1092);

auto tmp1098 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1096, tmp168,  (int32_t)12, tmp1098);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp1096);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)512, tmp168);

auto tmp1101 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
MatAdd4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1098, tmp1072, tmp1101);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1072);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1098);

auto tmp1104 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1101, tmp169, tmp170,  (int32_t)12, tmp1104);
ClearMemSecret1( (int32_t)512, tmp169);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1101);
ClearMemSecret1( (int32_t)512, tmp170);

auto tmp1108 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1104, tmp1108);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1104);

auto tmp1110 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp1110[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp1110[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp1110[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp1110[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp1110[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp1110[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp1110[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp1110[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp1111 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512);
Pad442( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1108,  (int32_t)4,  (int32_t)2, tmp1110, tmp1111);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp1110);

auto tmp1113 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp1111, tmp173,  (int32_t)12, tmp1113);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1111);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1024, tmp173);

auto tmp1116 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1108, tmp174,  (int32_t)12, tmp1116);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256, tmp174);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1108);

auto tmp1119 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1116, tmp175, tmp176,  (int32_t)12, tmp1119);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1116);
ClearMemSecret1( (int32_t)256, tmp176);
ClearMemSecret1( (int32_t)256, tmp175);

auto tmp1123 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1119, tmp1123);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1119);

auto tmp1125 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp1125[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp1125[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp1125[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp1125[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp1125[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp1125[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp1125[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp1125[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp1126 = make_vector<uint64_t>( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256);
Pad442( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1123,  (int32_t)4,  (int32_t)2, tmp1125, tmp1126);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1123);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp1125);

auto tmp1129 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp1126, tmp179,  (int32_t)12, tmp1129);
ClearMemSecret4( (int32_t)1,  (int32_t)30,  (int32_t)30,  (int32_t)256, tmp1126);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp179);

auto tmp1132 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1129, tmp180, tmp181,  (int32_t)12, tmp1132);
ClearMemSecret1( (int32_t)256, tmp180);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1129);
ClearMemSecret1( (int32_t)256, tmp181);

auto tmp1136 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1132, tmp1136);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1132);

auto tmp1138 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1136, tmp184,  (int32_t)12, tmp1138);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1136);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp184);

auto tmp1141 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1138, tmp1113, tmp1141);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1138);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1113);

auto tmp1144 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1141, tmp185, tmp186,  (int32_t)12, tmp1144);
ClearMemSecret1( (int32_t)1024, tmp186);
ClearMemSecret1( (int32_t)1024, tmp185);

auto tmp1147 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1144, tmp1147);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1144);

auto tmp1149 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1147, tmp189,  (int32_t)12, tmp1149);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp189);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1147);

auto tmp1152 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1149, tmp190, tmp191,  (int32_t)12, tmp1152);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1149);
ClearMemSecret1( (int32_t)256, tmp190);
ClearMemSecret1( (int32_t)256, tmp191);

auto tmp1156 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1152, tmp1156);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1152);

auto tmp1158 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1156, tmp194,  (int32_t)12, tmp1158);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp194);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1156);

auto tmp1161 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1158, tmp195, tmp196,  (int32_t)12, tmp1161);
ClearMemSecret1( (int32_t)256, tmp195);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1158);
ClearMemSecret1( (int32_t)256, tmp196);

auto tmp1165 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1161, tmp1165);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1161);

auto tmp1167 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1165, tmp199,  (int32_t)12, tmp1167);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1165);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp199);

auto tmp1170 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1167, tmp1141, tmp1170);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1141);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1167);

auto tmp1173 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1170, tmp200, tmp201,  (int32_t)12, tmp1173);
ClearMemSecret1( (int32_t)1024, tmp201);
ClearMemSecret1( (int32_t)1024, tmp200);

auto tmp1176 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1173, tmp1176);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1173);

auto tmp1178 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1176, tmp204,  (int32_t)12, tmp1178);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp204);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1176);

auto tmp1181 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1178, tmp205, tmp206,  (int32_t)12, tmp1181);
ClearMemSecret1( (int32_t)256, tmp205);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1178);
ClearMemSecret1( (int32_t)256, tmp206);

auto tmp1185 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1181, tmp1185);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1181);

auto tmp1187 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1185, tmp209,  (int32_t)12, tmp1187);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1185);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp209);

auto tmp1190 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1187, tmp210, tmp211,  (int32_t)12, tmp1190);
ClearMemSecret1( (int32_t)256, tmp211);
ClearMemSecret1( (int32_t)256, tmp210);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1187);

auto tmp1194 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1190, tmp1194);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1190);

auto tmp1196 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1194, tmp214,  (int32_t)12, tmp1196);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1194);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp214);

auto tmp1199 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1196, tmp1170, tmp1199);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1170);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1196);

auto tmp1202 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1199, tmp215, tmp216,  (int32_t)12, tmp1202);
ClearMemSecret1( (int32_t)1024, tmp216);
ClearMemSecret1( (int32_t)1024, tmp215);

auto tmp1205 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1202, tmp1205);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1202);

auto tmp1207 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1205, tmp219,  (int32_t)12, tmp1207);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1205);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp219);

auto tmp1210 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1207, tmp220, tmp221,  (int32_t)12, tmp1210);
ClearMemSecret1( (int32_t)256, tmp221);
ClearMemSecret1( (int32_t)256, tmp220);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1207);

auto tmp1214 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1210, tmp1214);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1210);

auto tmp1216 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1214, tmp224,  (int32_t)12, tmp1216);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp224);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1214);

auto tmp1219 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1216, tmp225, tmp226,  (int32_t)12, tmp1219);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1216);
ClearMemSecret1( (int32_t)256, tmp225);
ClearMemSecret1( (int32_t)256, tmp226);

auto tmp1223 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1219, tmp1223);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1219);

auto tmp1225 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1223, tmp229,  (int32_t)12, tmp1225);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1223);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp229);

auto tmp1228 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1225, tmp1199, tmp1228);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1225);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1199);

auto tmp1231 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1228, tmp230, tmp231,  (int32_t)12, tmp1231);
ClearMemSecret1( (int32_t)1024, tmp231);
ClearMemSecret1( (int32_t)1024, tmp230);

auto tmp1234 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1231, tmp1234);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1231);

auto tmp1236 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1234, tmp234,  (int32_t)12, tmp1236);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1234);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp234);

auto tmp1239 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1236, tmp235, tmp236,  (int32_t)12, tmp1239);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1236);
ClearMemSecret1( (int32_t)256, tmp236);
ClearMemSecret1( (int32_t)256, tmp235);

auto tmp1243 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1239, tmp1243);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1239);

auto tmp1245 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1243, tmp239,  (int32_t)12, tmp1245);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1243);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp239);

auto tmp1248 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1245, tmp240, tmp241,  (int32_t)12, tmp1248);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1245);
ClearMemSecret1( (int32_t)256, tmp240);
ClearMemSecret1( (int32_t)256, tmp241);

auto tmp1252 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1248, tmp1252);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1248);

auto tmp1254 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1252, tmp244,  (int32_t)12, tmp1254);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp244);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1252);

auto tmp1257 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1254, tmp1228, tmp1257);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1254);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1228);

auto tmp1260 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1257, tmp245, tmp246,  (int32_t)12, tmp1260);
ClearMemSecret1( (int32_t)1024, tmp245);
ClearMemSecret1( (int32_t)1024, tmp246);

auto tmp1263 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1260, tmp1263);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1260);

auto tmp1265 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1263, tmp249,  (int32_t)12, tmp1265);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1263);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp249);

auto tmp1268 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1265, tmp250, tmp251,  (int32_t)12, tmp1268);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1265);
ClearMemSecret1( (int32_t)256, tmp251);
ClearMemSecret1( (int32_t)256, tmp250);

auto tmp1272 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1268, tmp1272);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1268);

auto tmp1274 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1272, tmp254,  (int32_t)12, tmp1274);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1272);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp254);

auto tmp1277 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1274, tmp255, tmp256,  (int32_t)12, tmp1277);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1274);
ClearMemSecret1( (int32_t)256, tmp256);
ClearMemSecret1( (int32_t)256, tmp255);

auto tmp1281 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1277, tmp1281);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1277);

auto tmp1283 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1281, tmp259,  (int32_t)12, tmp1283);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1281);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp259);

auto tmp1286 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1283, tmp1257, tmp1286);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1283);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1257);

auto tmp1289 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1286, tmp260, tmp261,  (int32_t)12, tmp1289);
ClearMemSecret1( (int32_t)1024, tmp260);
ClearMemSecret1( (int32_t)1024, tmp261);

auto tmp1292 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1289, tmp1292);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1289);

auto tmp1294 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1292, tmp264,  (int32_t)12, tmp1294);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1292);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp264);

auto tmp1297 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1294, tmp265, tmp266,  (int32_t)12, tmp1297);
ClearMemSecret1( (int32_t)256, tmp265);
ClearMemSecret1( (int32_t)256, tmp266);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1294);

auto tmp1301 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1297, tmp1301);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1297);

auto tmp1303 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1301, tmp269,  (int32_t)12, tmp1303);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp269);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1301);

auto tmp1306 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1303, tmp270, tmp271,  (int32_t)12, tmp1306);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1303);
ClearMemSecret1( (int32_t)256, tmp270);
ClearMemSecret1( (int32_t)256, tmp271);

auto tmp1310 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1306, tmp1310);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1306);

auto tmp1312 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1310, tmp274,  (int32_t)12, tmp1312);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp274);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1310);

auto tmp1315 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1312, tmp1286, tmp1315);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1312);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1286);

auto tmp1318 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1315, tmp275, tmp276,  (int32_t)12, tmp1318);
ClearMemSecret1( (int32_t)1024, tmp276);
ClearMemSecret1( (int32_t)1024, tmp275);

auto tmp1321 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1318, tmp1321);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1318);

auto tmp1323 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1321, tmp279,  (int32_t)12, tmp1323);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1321);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp279);

auto tmp1326 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1323, tmp280, tmp281,  (int32_t)12, tmp1326);
ClearMemSecret1( (int32_t)256, tmp281);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1323);
ClearMemSecret1( (int32_t)256, tmp280);

auto tmp1330 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1326, tmp1330);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1326);

auto tmp1332 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1330, tmp284,  (int32_t)12, tmp1332);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp284);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1330);

auto tmp1335 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1332, tmp285, tmp286,  (int32_t)12, tmp1335);
ClearMemSecret1( (int32_t)256, tmp286);
ClearMemSecret1( (int32_t)256, tmp285);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1332);

auto tmp1339 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1335, tmp1339);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1335);

auto tmp1341 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1339, tmp289,  (int32_t)12, tmp1341);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1339);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp289);

auto tmp1344 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1341, tmp1315, tmp1344);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1315);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1341);

auto tmp1347 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1344, tmp290, tmp291,  (int32_t)12, tmp1347);
ClearMemSecret1( (int32_t)1024, tmp291);
ClearMemSecret1( (int32_t)1024, tmp290);

auto tmp1350 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1347, tmp1350);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1347);

auto tmp1352 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1350, tmp294,  (int32_t)12, tmp1352);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp294);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1350);

auto tmp1355 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1352, tmp295, tmp296,  (int32_t)12, tmp1355);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1352);
ClearMemSecret1( (int32_t)256, tmp296);
ClearMemSecret1( (int32_t)256, tmp295);

auto tmp1359 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1355, tmp1359);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1355);

auto tmp1361 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1359, tmp299,  (int32_t)12, tmp1361);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp299);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1359);

auto tmp1364 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1361, tmp300, tmp301,  (int32_t)12, tmp1364);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1361);
ClearMemSecret1( (int32_t)256, tmp300);
ClearMemSecret1( (int32_t)256, tmp301);

auto tmp1368 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1364, tmp1368);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1364);

auto tmp1370 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1368, tmp304,  (int32_t)12, tmp1370);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1368);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp304);

auto tmp1373 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1370, tmp1344, tmp1373);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1344);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1370);

auto tmp1376 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1373, tmp305, tmp306,  (int32_t)12, tmp1376);
ClearMemSecret1( (int32_t)1024, tmp306);
ClearMemSecret1( (int32_t)1024, tmp305);

auto tmp1379 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1376, tmp1379);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1376);

auto tmp1381 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1379, tmp309,  (int32_t)12, tmp1381);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1379);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp309);

auto tmp1384 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1381, tmp310, tmp311,  (int32_t)12, tmp1384);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1381);
ClearMemSecret1( (int32_t)256, tmp310);
ClearMemSecret1( (int32_t)256, tmp311);

auto tmp1388 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1384, tmp1388);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1384);

auto tmp1390 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1388, tmp314,  (int32_t)12, tmp1390);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp314);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1388);

auto tmp1393 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1390, tmp315, tmp316,  (int32_t)12, tmp1393);
ClearMemSecret1( (int32_t)256, tmp315);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1390);
ClearMemSecret1( (int32_t)256, tmp316);

auto tmp1397 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1393, tmp1397);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1393);

auto tmp1399 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1397, tmp319,  (int32_t)12, tmp1399);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp319);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1397);

auto tmp1402 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1399, tmp1373, tmp1402);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1399);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1373);

auto tmp1405 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1402, tmp320, tmp321,  (int32_t)12, tmp1405);
ClearMemSecret1( (int32_t)1024, tmp321);
ClearMemSecret1( (int32_t)1024, tmp320);

auto tmp1408 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1405, tmp1408);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1405);

auto tmp1410 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1408, tmp324,  (int32_t)12, tmp1410);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp324);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1408);

auto tmp1413 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1410, tmp325, tmp326,  (int32_t)12, tmp1413);
ClearMemSecret1( (int32_t)256, tmp326);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1410);
ClearMemSecret1( (int32_t)256, tmp325);

auto tmp1417 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1413, tmp1417);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1413);

auto tmp1419 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1417, tmp329,  (int32_t)12, tmp1419);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp329);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1417);

auto tmp1422 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1419, tmp330, tmp331,  (int32_t)12, tmp1422);
ClearMemSecret1( (int32_t)256, tmp331);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1419);
ClearMemSecret1( (int32_t)256, tmp330);

auto tmp1426 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1422, tmp1426);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1422);

auto tmp1428 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1426, tmp334,  (int32_t)12, tmp1428);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1426);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp334);

auto tmp1431 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1428, tmp1402, tmp1431);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1402);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1428);

auto tmp1434 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1431, tmp335, tmp336,  (int32_t)12, tmp1434);
ClearMemSecret1( (int32_t)1024, tmp336);
ClearMemSecret1( (int32_t)1024, tmp335);

auto tmp1437 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1434, tmp1437);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1434);

auto tmp1439 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1437, tmp339,  (int32_t)12, tmp1439);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp339);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1437);

auto tmp1442 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1439, tmp340, tmp341,  (int32_t)12, tmp1442);
ClearMemSecret1( (int32_t)256, tmp340);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1439);
ClearMemSecret1( (int32_t)256, tmp341);

auto tmp1446 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1442, tmp1446);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1442);

auto tmp1448 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1446, tmp344,  (int32_t)12, tmp1448);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp344);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1446);

auto tmp1451 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1448, tmp345, tmp346,  (int32_t)12, tmp1451);
ClearMemSecret1( (int32_t)256, tmp345);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1448);
ClearMemSecret1( (int32_t)256, tmp346);

auto tmp1455 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1451, tmp1455);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1451);

auto tmp1457 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1455, tmp349,  (int32_t)12, tmp1457);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp349);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1455);

auto tmp1460 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1457, tmp1431, tmp1460);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1457);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1431);

auto tmp1463 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1460, tmp350, tmp351,  (int32_t)12, tmp1463);
ClearMemSecret1( (int32_t)1024, tmp351);
ClearMemSecret1( (int32_t)1024, tmp350);

auto tmp1466 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1463, tmp1466);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1463);

auto tmp1468 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1466, tmp354,  (int32_t)12, tmp1468);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1466);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp354);

auto tmp1471 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1468, tmp355, tmp356,  (int32_t)12, tmp1471);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1468);
ClearMemSecret1( (int32_t)256, tmp355);
ClearMemSecret1( (int32_t)256, tmp356);

auto tmp1475 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1471, tmp1475);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1471);

auto tmp1477 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1475, tmp359,  (int32_t)12, tmp1477);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1475);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp359);

auto tmp1480 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1477, tmp360, tmp361,  (int32_t)12, tmp1480);
ClearMemSecret1( (int32_t)256, tmp361);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1477);
ClearMemSecret1( (int32_t)256, tmp360);

auto tmp1484 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1480, tmp1484);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1480);

auto tmp1486 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1484, tmp364,  (int32_t)12, tmp1486);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1484);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp364);

auto tmp1489 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1486, tmp1460, tmp1489);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1460);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1486);

auto tmp1492 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1489, tmp365, tmp366,  (int32_t)12, tmp1492);
ClearMemSecret1( (int32_t)1024, tmp366);
ClearMemSecret1( (int32_t)1024, tmp365);

auto tmp1495 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1492, tmp1495);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1492);

auto tmp1497 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1495, tmp369,  (int32_t)12, tmp1497);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1495);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp369);

auto tmp1500 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1497, tmp370, tmp371,  (int32_t)12, tmp1500);
ClearMemSecret1( (int32_t)256, tmp370);
ClearMemSecret1( (int32_t)256, tmp371);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1497);

auto tmp1504 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1500, tmp1504);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1500);

auto tmp1506 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1504, tmp374,  (int32_t)12, tmp1506);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1504);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp374);

auto tmp1509 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1506, tmp375, tmp376,  (int32_t)12, tmp1509);
ClearMemSecret1( (int32_t)256, tmp375);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1506);
ClearMemSecret1( (int32_t)256, tmp376);

auto tmp1513 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1509, tmp1513);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1509);

auto tmp1515 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1513, tmp379,  (int32_t)12, tmp1515);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp379);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1513);

auto tmp1518 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1515, tmp1489, tmp1518);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1515);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1489);

auto tmp1521 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1518, tmp380, tmp381,  (int32_t)12, tmp1521);
ClearMemSecret1( (int32_t)1024, tmp381);
ClearMemSecret1( (int32_t)1024, tmp380);

auto tmp1524 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1521, tmp1524);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1521);

auto tmp1526 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1524, tmp384,  (int32_t)12, tmp1526);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp384);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1524);

auto tmp1529 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1526, tmp385, tmp386,  (int32_t)12, tmp1529);
ClearMemSecret1( (int32_t)256, tmp385);
ClearMemSecret1( (int32_t)256, tmp386);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1526);

auto tmp1533 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1529, tmp1533);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1529);

auto tmp1535 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1533, tmp389,  (int32_t)12, tmp1535);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1533);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp389);

auto tmp1538 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1535, tmp390, tmp391,  (int32_t)12, tmp1538);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1535);
ClearMemSecret1( (int32_t)256, tmp390);
ClearMemSecret1( (int32_t)256, tmp391);

auto tmp1542 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1538, tmp1542);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1538);

auto tmp1544 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1542, tmp394,  (int32_t)12, tmp1544);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1542);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp394);

auto tmp1547 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1544, tmp1518, tmp1547);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1544);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1518);

auto tmp1550 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1547, tmp395, tmp396,  (int32_t)12, tmp1550);
ClearMemSecret1( (int32_t)1024, tmp396);
ClearMemSecret1( (int32_t)1024, tmp395);

auto tmp1553 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1550, tmp1553);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1550);

auto tmp1555 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1553, tmp399,  (int32_t)12, tmp1555);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp399);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1553);

auto tmp1558 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1555, tmp400, tmp401,  (int32_t)12, tmp1558);
ClearMemSecret1( (int32_t)256, tmp400);
ClearMemSecret1( (int32_t)256, tmp401);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1555);

auto tmp1562 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1558, tmp1562);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1558);

auto tmp1564 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1562, tmp404,  (int32_t)12, tmp1564);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1562);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp404);

auto tmp1567 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1564, tmp405, tmp406,  (int32_t)12, tmp1567);
ClearMemSecret1( (int32_t)256, tmp405);
ClearMemSecret1( (int32_t)256, tmp406);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1564);

auto tmp1571 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1567, tmp1571);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1567);

auto tmp1573 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1571, tmp409,  (int32_t)12, tmp1573);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp409);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1571);

auto tmp1576 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1573, tmp1547, tmp1576);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1547);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1573);

auto tmp1579 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1576, tmp410, tmp411,  (int32_t)12, tmp1579);
ClearMemSecret1( (int32_t)1024, tmp411);
ClearMemSecret1( (int32_t)1024, tmp410);

auto tmp1582 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1579, tmp1582);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1579);

auto tmp1584 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1582, tmp414,  (int32_t)12, tmp1584);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1582);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp414);

auto tmp1587 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1584, tmp415, tmp416,  (int32_t)12, tmp1587);
ClearMemSecret1( (int32_t)256, tmp415);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1584);
ClearMemSecret1( (int32_t)256, tmp416);

auto tmp1591 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1587, tmp1591);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1587);

auto tmp1593 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1591, tmp419,  (int32_t)12, tmp1593);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp419);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1591);

auto tmp1596 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1593, tmp420, tmp421,  (int32_t)12, tmp1596);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1593);
ClearMemSecret1( (int32_t)256, tmp420);
ClearMemSecret1( (int32_t)256, tmp421);

auto tmp1600 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1596, tmp1600);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1596);

auto tmp1602 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1600, tmp424,  (int32_t)12, tmp1602);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp424);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1600);

auto tmp1605 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1602, tmp1576, tmp1605);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1576);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1602);

auto tmp1608 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1605, tmp425, tmp426,  (int32_t)12, tmp1608);
ClearMemSecret1( (int32_t)1024, tmp425);
ClearMemSecret1( (int32_t)1024, tmp426);

auto tmp1611 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1608, tmp1611);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1608);

auto tmp1613 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1611, tmp429,  (int32_t)12, tmp1613);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1611);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp429);

auto tmp1616 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1613, tmp430, tmp431,  (int32_t)12, tmp1616);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1613);
ClearMemSecret1( (int32_t)256, tmp431);
ClearMemSecret1( (int32_t)256, tmp430);

auto tmp1620 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1616, tmp1620);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1616);

auto tmp1622 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1620, tmp434,  (int32_t)12, tmp1622);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp434);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1620);

auto tmp1625 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1622, tmp435, tmp436,  (int32_t)12, tmp1625);
ClearMemSecret1( (int32_t)256, tmp436);
ClearMemSecret1( (int32_t)256, tmp435);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1622);

auto tmp1629 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1625, tmp1629);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1625);

auto tmp1631 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1629, tmp439,  (int32_t)12, tmp1631);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp439);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1629);

auto tmp1634 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1631, tmp1605, tmp1634);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1631);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1605);

auto tmp1637 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1634, tmp440, tmp441,  (int32_t)12, tmp1637);
ClearMemSecret1( (int32_t)1024, tmp441);
ClearMemSecret1( (int32_t)1024, tmp440);

auto tmp1640 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1637, tmp1640);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1637);

auto tmp1642 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1640, tmp444,  (int32_t)12, tmp1642);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1640);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp444);

auto tmp1645 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1642, tmp445, tmp446,  (int32_t)12, tmp1645);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1642);
ClearMemSecret1( (int32_t)256, tmp446);
ClearMemSecret1( (int32_t)256, tmp445);

auto tmp1649 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1645, tmp1649);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1645);

auto tmp1651 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1649, tmp449,  (int32_t)12, tmp1651);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp449);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1649);

auto tmp1654 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1651, tmp450, tmp451,  (int32_t)12, tmp1654);
ClearMemSecret1( (int32_t)256, tmp451);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1651);
ClearMemSecret1( (int32_t)256, tmp450);

auto tmp1658 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1654, tmp1658);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1654);

auto tmp1660 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1658, tmp454,  (int32_t)12, tmp1660);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp454);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1658);

auto tmp1663 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1660, tmp1634, tmp1663);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1634);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1660);

auto tmp1666 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1663, tmp455, tmp456,  (int32_t)12, tmp1666);
ClearMemSecret1( (int32_t)1024, tmp455);
ClearMemSecret1( (int32_t)1024, tmp456);

auto tmp1669 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1666, tmp1669);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1666);

auto tmp1671 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1669, tmp459,  (int32_t)12, tmp1671);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp459);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1669);

auto tmp1674 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1671, tmp460, tmp461,  (int32_t)12, tmp1674);
ClearMemSecret1( (int32_t)256, tmp461);
ClearMemSecret1( (int32_t)256, tmp460);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1671);

auto tmp1678 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1674, tmp1678);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1674);

auto tmp1680 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1678, tmp464,  (int32_t)12, tmp1680);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1678);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp464);

auto tmp1683 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1680, tmp465, tmp466,  (int32_t)12, tmp1683);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1680);
ClearMemSecret1( (int32_t)256, tmp466);
ClearMemSecret1( (int32_t)256, tmp465);

auto tmp1687 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1683, tmp1687);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1683);

auto tmp1689 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1687, tmp469,  (int32_t)12, tmp1689);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp469);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1687);

auto tmp1692 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1689, tmp1663, tmp1692);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1663);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1689);

auto tmp1695 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1692, tmp470, tmp471,  (int32_t)12, tmp1695);
ClearMemSecret1( (int32_t)1024, tmp470);
ClearMemSecret1( (int32_t)1024, tmp471);

auto tmp1698 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1695, tmp1698);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1695);

auto tmp1700 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1698, tmp474,  (int32_t)12, tmp1700);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1698);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp474);

auto tmp1703 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1700, tmp475, tmp476,  (int32_t)12, tmp1703);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1700);
ClearMemSecret1( (int32_t)256, tmp475);
ClearMemSecret1( (int32_t)256, tmp476);

auto tmp1707 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1703, tmp1707);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1703);

auto tmp1709 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1707, tmp479,  (int32_t)12, tmp1709);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1707);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp479);

auto tmp1712 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1709, tmp480, tmp481,  (int32_t)12, tmp1712);
ClearMemSecret1( (int32_t)256, tmp480);
ClearMemSecret1( (int32_t)256, tmp481);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1709);

auto tmp1716 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1712, tmp1716);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1712);

auto tmp1718 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1716, tmp484,  (int32_t)12, tmp1718);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1716);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp484);

auto tmp1721 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1718, tmp1692, tmp1721);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1718);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1692);

auto tmp1724 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1721, tmp485, tmp486,  (int32_t)12, tmp1724);
ClearMemSecret1( (int32_t)1024, tmp486);
ClearMemSecret1( (int32_t)1024, tmp485);

auto tmp1727 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1724, tmp1727);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1724);

auto tmp1729 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1727, tmp489,  (int32_t)12, tmp1729);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1727);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp489);

auto tmp1732 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1729, tmp490, tmp491,  (int32_t)12, tmp1732);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1729);
ClearMemSecret1( (int32_t)256, tmp491);
ClearMemSecret1( (int32_t)256, tmp490);

auto tmp1736 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1732, tmp1736);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1732);

auto tmp1738 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1736, tmp494,  (int32_t)12, tmp1738);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1736);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp494);

auto tmp1741 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1738, tmp495, tmp496,  (int32_t)12, tmp1741);
ClearMemSecret1( (int32_t)256, tmp496);
ClearMemSecret1( (int32_t)256, tmp495);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1738);

auto tmp1745 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1741, tmp1745);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1741);

auto tmp1747 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1745, tmp499,  (int32_t)12, tmp1747);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp499);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1745);

auto tmp1750 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1747, tmp1721, tmp1750);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1721);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1747);

auto tmp1753 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1750, tmp500, tmp501,  (int32_t)12, tmp1753);
ClearMemSecret1( (int32_t)1024, tmp500);
ClearMemSecret1( (int32_t)1024, tmp501);

auto tmp1756 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1753, tmp1756);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1753);

auto tmp1758 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1756, tmp504,  (int32_t)12, tmp1758);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp504);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1756);

auto tmp1761 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1758, tmp505, tmp506,  (int32_t)12, tmp1761);
ClearMemSecret1( (int32_t)256, tmp506);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1758);
ClearMemSecret1( (int32_t)256, tmp505);

auto tmp1765 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1761, tmp1765);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1761);

auto tmp1767 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1765, tmp509,  (int32_t)12, tmp1767);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1765);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp509);

auto tmp1770 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1767, tmp510, tmp511,  (int32_t)12, tmp1770);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1767);
ClearMemSecret1( (int32_t)256, tmp510);
ClearMemSecret1( (int32_t)256, tmp511);

auto tmp1774 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1770, tmp1774);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1770);

auto tmp1776 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1774, tmp514,  (int32_t)12, tmp1776);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1774);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp514);

auto tmp1779 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1776, tmp1750, tmp1779);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1750);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1776);

auto tmp1782 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1779, tmp515, tmp516,  (int32_t)12, tmp1782);
ClearMemSecret1( (int32_t)1024, tmp516);
ClearMemSecret1( (int32_t)1024, tmp515);

auto tmp1785 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1782, tmp1785);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1782);

auto tmp1787 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1785, tmp519,  (int32_t)12, tmp1787);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp519);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1785);

auto tmp1790 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1787, tmp520, tmp521,  (int32_t)12, tmp1790);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1787);
ClearMemSecret1( (int32_t)256, tmp521);
ClearMemSecret1( (int32_t)256, tmp520);

auto tmp1794 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1790, tmp1794);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1790);

auto tmp1796 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1794, tmp524,  (int32_t)12, tmp1796);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp524);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1794);

auto tmp1799 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1796, tmp525, tmp526,  (int32_t)12, tmp1799);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1796);
ClearMemSecret1( (int32_t)256, tmp526);
ClearMemSecret1( (int32_t)256, tmp525);

auto tmp1803 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1799, tmp1803);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1799);

auto tmp1805 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1803, tmp529,  (int32_t)12, tmp1805);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp529);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1803);

auto tmp1808 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1805, tmp1779, tmp1808);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1805);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1779);

auto tmp1811 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1808, tmp530, tmp531,  (int32_t)12, tmp1811);
ClearMemSecret1( (int32_t)1024, tmp530);
ClearMemSecret1( (int32_t)1024, tmp531);

auto tmp1814 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1811, tmp1814);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1811);

auto tmp1816 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1814, tmp534,  (int32_t)12, tmp1816);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1814);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp534);

auto tmp1819 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1816, tmp535, tmp536,  (int32_t)12, tmp1819);
ClearMemSecret1( (int32_t)256, tmp535);
ClearMemSecret1( (int32_t)256, tmp536);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1816);

auto tmp1823 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1819, tmp1823);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1819);

auto tmp1825 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1823, tmp539,  (int32_t)12, tmp1825);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1823);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp539);

auto tmp1828 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1825, tmp540, tmp541,  (int32_t)12, tmp1828);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1825);
ClearMemSecret1( (int32_t)256, tmp540);
ClearMemSecret1( (int32_t)256, tmp541);

auto tmp1832 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1828, tmp1832);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1828);

auto tmp1834 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1832, tmp544,  (int32_t)12, tmp1834);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1832);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp544);

auto tmp1837 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1834, tmp1808, tmp1837);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1834);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1808);

auto tmp1840 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1837, tmp545, tmp546,  (int32_t)12, tmp1840);
ClearMemSecret1( (int32_t)1024, tmp546);
ClearMemSecret1( (int32_t)1024, tmp545);

auto tmp1843 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1840, tmp1843);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1840);

auto tmp1845 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1843, tmp549,  (int32_t)12, tmp1845);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1843);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp549);

auto tmp1848 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1845, tmp550, tmp551,  (int32_t)12, tmp1848);
ClearMemSecret1( (int32_t)256, tmp551);
ClearMemSecret1( (int32_t)256, tmp550);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1845);

auto tmp1852 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1848, tmp1852);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1848);

auto tmp1854 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1852, tmp554,  (int32_t)12, tmp1854);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp554);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1852);

auto tmp1857 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1854, tmp555, tmp556,  (int32_t)12, tmp1857);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1854);
ClearMemSecret1( (int32_t)256, tmp555);
ClearMemSecret1( (int32_t)256, tmp556);

auto tmp1861 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1857, tmp1861);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1857);

auto tmp1863 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1861, tmp559,  (int32_t)12, tmp1863);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1861);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp559);

auto tmp1866 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1863, tmp1837, tmp1866);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1863);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1837);

auto tmp1869 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1866, tmp560, tmp561,  (int32_t)12, tmp1869);
ClearMemSecret1( (int32_t)1024, tmp560);
ClearMemSecret1( (int32_t)1024, tmp561);

auto tmp1872 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1869, tmp1872);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1869);

auto tmp1874 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1872, tmp564,  (int32_t)12, tmp1874);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp564);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1872);

auto tmp1877 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1874, tmp565, tmp566,  (int32_t)12, tmp1877);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1874);
ClearMemSecret1( (int32_t)256, tmp565);
ClearMemSecret1( (int32_t)256, tmp566);

auto tmp1881 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1877, tmp1881);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1877);

auto tmp1883 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1881, tmp569,  (int32_t)12, tmp1883);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp569);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1881);

auto tmp1886 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1883, tmp570, tmp571,  (int32_t)12, tmp1886);
ClearMemSecret1( (int32_t)256, tmp570);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1883);
ClearMemSecret1( (int32_t)256, tmp571);

auto tmp1890 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1886, tmp1890);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1886);

auto tmp1892 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1890, tmp574,  (int32_t)12, tmp1892);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp574);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1890);

auto tmp1895 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1892, tmp1866, tmp1895);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1866);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1892);

auto tmp1898 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1895, tmp575, tmp576,  (int32_t)12, tmp1898);
ClearMemSecret1( (int32_t)1024, tmp575);
ClearMemSecret1( (int32_t)1024, tmp576);

auto tmp1901 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1898, tmp1901);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1898);

auto tmp1903 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1901, tmp579,  (int32_t)12, tmp1903);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1901);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp579);

auto tmp1906 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1903, tmp580, tmp581,  (int32_t)12, tmp1906);
ClearMemSecret1( (int32_t)256, tmp581);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1903);
ClearMemSecret1( (int32_t)256, tmp580);

auto tmp1910 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1906, tmp1910);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1906);

auto tmp1912 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1910, tmp584,  (int32_t)12, tmp1912);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1910);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp584);

auto tmp1915 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1912, tmp585, tmp586,  (int32_t)12, tmp1915);
ClearMemSecret1( (int32_t)256, tmp585);
ClearMemSecret1( (int32_t)256, tmp586);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1912);

auto tmp1919 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1915, tmp1919);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1915);

auto tmp1921 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1919, tmp589,  (int32_t)12, tmp1921);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp589);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1919);

auto tmp1924 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1921, tmp1895, tmp1924);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1921);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1895);

auto tmp1927 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1924, tmp590, tmp591,  (int32_t)12, tmp1927);
ClearMemSecret1( (int32_t)1024, tmp591);
ClearMemSecret1( (int32_t)1024, tmp590);

auto tmp1930 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1927, tmp1930);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1927);

auto tmp1932 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1930, tmp594,  (int32_t)12, tmp1932);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1930);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp594);

auto tmp1935 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1932, tmp595, tmp596,  (int32_t)12, tmp1935);
ClearMemSecret1( (int32_t)256, tmp595);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1932);
ClearMemSecret1( (int32_t)256, tmp596);

auto tmp1939 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1935, tmp1939);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1935);

auto tmp1941 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1939, tmp599,  (int32_t)12, tmp1941);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp599);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1939);

auto tmp1944 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1941, tmp600, tmp601,  (int32_t)12, tmp1944);
ClearMemSecret1( (int32_t)256, tmp601);
ClearMemSecret1( (int32_t)256, tmp600);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1941);

auto tmp1948 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1944, tmp1948);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1944);

auto tmp1950 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1948, tmp604,  (int32_t)12, tmp1950);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp604);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1948);

auto tmp1953 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1950, tmp1924, tmp1953);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1924);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1950);

auto tmp1956 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1953, tmp605, tmp606,  (int32_t)12, tmp1956);
ClearMemSecret1( (int32_t)1024, tmp606);
ClearMemSecret1( (int32_t)1024, tmp605);

auto tmp1959 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1956, tmp1959);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1956);

auto tmp1961 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1959, tmp609,  (int32_t)12, tmp1961);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp609);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1959);

auto tmp1964 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1961, tmp610, tmp611,  (int32_t)12, tmp1964);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1961);
ClearMemSecret1( (int32_t)256, tmp611);
ClearMemSecret1( (int32_t)256, tmp610);

auto tmp1968 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1964, tmp1968);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1964);

auto tmp1970 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1968, tmp614,  (int32_t)12, tmp1970);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp614);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1968);

auto tmp1973 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1970, tmp615, tmp616,  (int32_t)12, tmp1973);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1970);
ClearMemSecret1( (int32_t)256, tmp616);
ClearMemSecret1( (int32_t)256, tmp615);

auto tmp1977 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1973, tmp1977);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1973);

auto tmp1979 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1977, tmp619,  (int32_t)12, tmp1979);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1977);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp619);

auto tmp1982 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1979, tmp1953, tmp1982);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1953);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1979);

auto tmp1985 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1982, tmp620, tmp621,  (int32_t)12, tmp1985);
ClearMemSecret1( (int32_t)1024, tmp620);
ClearMemSecret1( (int32_t)1024, tmp621);

auto tmp1988 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1985, tmp1988);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1985);

auto tmp1990 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1988, tmp624,  (int32_t)12, tmp1990);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp624);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1988);

auto tmp1993 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1990, tmp625, tmp626,  (int32_t)12, tmp1993);
ClearMemSecret1( (int32_t)256, tmp626);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1990);
ClearMemSecret1( (int32_t)256, tmp625);

auto tmp1997 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1993, tmp1997);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1993);

auto tmp1999 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1997, tmp629,  (int32_t)12, tmp1999);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp629);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1997);

auto tmp2002 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1999, tmp630, tmp631,  (int32_t)12, tmp2002);
ClearMemSecret1( (int32_t)256, tmp631);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1999);
ClearMemSecret1( (int32_t)256, tmp630);

auto tmp2006 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2002, tmp2006);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2002);

auto tmp2008 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2006, tmp634,  (int32_t)12, tmp2008);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp634);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2006);

auto tmp2011 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2008, tmp1982, tmp2011);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2008);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1982);

auto tmp2014 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2011, tmp635, tmp636,  (int32_t)12, tmp2014);
ClearMemSecret1( (int32_t)1024, tmp636);
ClearMemSecret1( (int32_t)1024, tmp635);

auto tmp2017 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2014, tmp2017);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2014);

auto tmp2019 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2017, tmp639,  (int32_t)12, tmp2019);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2017);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp639);

auto tmp2022 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2019, tmp640, tmp641,  (int32_t)12, tmp2022);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2019);
ClearMemSecret1( (int32_t)256, tmp641);
ClearMemSecret1( (int32_t)256, tmp640);

auto tmp2026 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2022, tmp2026);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2022);

auto tmp2028 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp2026, tmp644,  (int32_t)12, tmp2028);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2026);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp644);

auto tmp2031 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2028, tmp645, tmp646,  (int32_t)12, tmp2031);
ClearMemSecret1( (int32_t)256, tmp645);
ClearMemSecret1( (int32_t)256, tmp646);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2028);

auto tmp2035 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2031, tmp2035);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2031);

auto tmp2037 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2035, tmp649,  (int32_t)12, tmp2037);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp649);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2035);

auto tmp2040 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2037, tmp2011, tmp2040);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2011);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2037);

auto tmp2043 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2040, tmp650, tmp651,  (int32_t)12, tmp2043);
ClearMemSecret1( (int32_t)1024, tmp651);
ClearMemSecret1( (int32_t)1024, tmp650);

auto tmp2046 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2043, tmp2046);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2043);

auto tmp2048 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2046, tmp654,  (int32_t)12, tmp2048);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp654);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2046);

auto tmp2051 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2048, tmp655, tmp656,  (int32_t)12, tmp2051);
ClearMemSecret1( (int32_t)256, tmp656);
ClearMemSecret1( (int32_t)256, tmp655);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2048);

auto tmp2055 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2051, tmp2055);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2051);

auto tmp2057 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp2055, tmp659,  (int32_t)12, tmp2057);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp659);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2055);

auto tmp2060 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2057, tmp660, tmp661,  (int32_t)12, tmp2060);
ClearMemSecret1( (int32_t)256, tmp660);
ClearMemSecret1( (int32_t)256, tmp661);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2057);

auto tmp2064 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2060, tmp2064);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2060);

auto tmp2066 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2064, tmp664,  (int32_t)12, tmp2066);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2064);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp664);

auto tmp2069 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2066, tmp2040, tmp2069);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2066);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2040);

auto tmp2072 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2069, tmp665, tmp666,  (int32_t)12, tmp2072);
ClearMemSecret1( (int32_t)1024, tmp665);
ClearMemSecret1( (int32_t)1024, tmp666);

auto tmp2075 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2072, tmp2075);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2072);

auto tmp2077 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2075, tmp669,  (int32_t)12, tmp2077);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp669);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2075);

auto tmp2080 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2077, tmp670, tmp671,  (int32_t)12, tmp2080);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2077);
ClearMemSecret1( (int32_t)256, tmp671);
ClearMemSecret1( (int32_t)256, tmp670);

auto tmp2084 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2080, tmp2084);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2080);

auto tmp2086 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp2084, tmp674,  (int32_t)12, tmp2086);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp674);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2084);

auto tmp2089 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2086, tmp675, tmp676,  (int32_t)12, tmp2089);
ClearMemSecret1( (int32_t)256, tmp675);
ClearMemSecret1( (int32_t)256, tmp676);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2086);

auto tmp2093 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2089, tmp2093);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2089);

auto tmp2095 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2093, tmp679,  (int32_t)12, tmp2095);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2093);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp679);

auto tmp2098 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2095, tmp2069, tmp2098);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2069);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2095);

auto tmp2101 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2098, tmp680, tmp681,  (int32_t)12, tmp2101);
ClearMemSecret1( (int32_t)1024, tmp680);
ClearMemSecret1( (int32_t)1024, tmp681);

auto tmp2104 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2101, tmp2104);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2101);

auto tmp2106 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2104, tmp684,  (int32_t)12, tmp2106);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp684);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2104);

auto tmp2109 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2106, tmp685, tmp686,  (int32_t)12, tmp2109);
ClearMemSecret1( (int32_t)256, tmp686);
ClearMemSecret1( (int32_t)256, tmp685);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2106);

auto tmp2113 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2109, tmp2113);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2109);

auto tmp2115 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp2113, tmp689,  (int32_t)12, tmp2115);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp689);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2113);

auto tmp2118 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2115, tmp690, tmp691,  (int32_t)12, tmp2118);
ClearMemSecret1( (int32_t)256, tmp690);
ClearMemSecret1( (int32_t)256, tmp691);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2115);

auto tmp2122 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2118, tmp2122);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2118);

auto tmp2124 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2122, tmp694,  (int32_t)12, tmp2124);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2122);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp694);

auto tmp2127 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2124, tmp2098, tmp2127);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2098);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2124);

auto tmp2130 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2127, tmp695, tmp696,  (int32_t)12, tmp2130);
ClearMemSecret1( (int32_t)1024, tmp696);
ClearMemSecret1( (int32_t)1024, tmp695);

auto tmp2133 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2130, tmp2133);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2130);

auto tmp2135 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2133, tmp699,  (int32_t)12, tmp2135);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)256, tmp699);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2133);

auto tmp2138 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2135, tmp700, tmp701,  (int32_t)12, tmp2138);
ClearMemSecret1( (int32_t)256, tmp700);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2135);
ClearMemSecret1( (int32_t)256, tmp701);

auto tmp2142 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2138, tmp2142);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2138);

auto tmp2144 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp2142, tmp704,  (int32_t)12, tmp2144);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2142);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)256, tmp704);

auto tmp2147 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2144, tmp705, tmp706,  (int32_t)12, tmp2147);
ClearMemSecret1( (int32_t)256, tmp705);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2144);
ClearMemSecret1( (int32_t)256, tmp706);

auto tmp2151 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2147, tmp2151);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2147);

auto tmp2153 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2151, tmp709,  (int32_t)12, tmp2153);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp2151);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)1024, tmp709);

auto tmp2156 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
MatAdd4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2153, tmp2127, tmp2156);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2153);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2127);

auto tmp2159 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2156, tmp710, tmp711,  (int32_t)12, tmp2159);
ClearMemSecret1( (int32_t)1024, tmp711);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2156);
ClearMemSecret1( (int32_t)1024, tmp710);

auto tmp2163 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2159, tmp2163);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2159);

auto tmp2165 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp2165[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp2165[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp2165[ (int64_t)1][ (int64_t)0] =  (int32_t)0;
tmp2165[ (int64_t)1][ (int64_t)1] =  (int32_t)0;
tmp2165[ (int64_t)2][ (int64_t)0] =  (int32_t)0;
tmp2165[ (int64_t)2][ (int64_t)1] =  (int32_t)0;
tmp2165[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp2165[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp2166 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024);
Pad442( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2163,  (int32_t)4,  (int32_t)2, tmp2165, tmp2166);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp2165);

auto tmp2168 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp2166, tmp714,  (int32_t)12, tmp2168);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2166);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)2048, tmp714);

auto tmp2171 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2163, tmp715,  (int32_t)12, tmp2171);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512, tmp715);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp2163);

auto tmp2174 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp2171, tmp716, tmp717,  (int32_t)12, tmp2174);
ClearMemSecret1( (int32_t)512, tmp717);
ClearMemSecret1( (int32_t)512, tmp716);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp2171);

auto tmp2178 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp2174, tmp2178);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp2174);

auto tmp2180 = make_vector<int32_t>( (int32_t)4,  (int32_t)2);
tmp2180[ (int64_t)0][ (int64_t)0] =  (int32_t)0;
tmp2180[ (int64_t)0][ (int64_t)1] =  (int32_t)0;
tmp2180[ (int64_t)1][ (int64_t)0] =  (int32_t)1;
tmp2180[ (int64_t)1][ (int64_t)1] =  (int32_t)1;
tmp2180[ (int64_t)2][ (int64_t)0] =  (int32_t)1;
tmp2180[ (int64_t)2][ (int64_t)1] =  (int32_t)1;
tmp2180[ (int64_t)3][ (int64_t)0] =  (int32_t)0;
tmp2180[ (int64_t)3][ (int64_t)1] =  (int32_t)0;

auto tmp2181 = make_vector<uint64_t>( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512);
Pad442( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp2178,  (int32_t)4,  (int32_t)2, tmp2180, tmp2181);
ClearMemPublic2( (int32_t)4,  (int32_t)2, tmp2180);
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp2178);

auto tmp2184 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp2181, tmp720,  (int32_t)12, tmp2184);
ClearMemSecret4( (int32_t)1,  (int32_t)16,  (int32_t)16,  (int32_t)512, tmp2181);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp720);

auto tmp2187 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2184, tmp721, tmp722,  (int32_t)12, tmp2187);
ClearMemSecret1( (int32_t)512, tmp722);
ClearMemSecret1( (int32_t)512, tmp721);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2184);

auto tmp2191 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2187, tmp2191);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2187);

auto tmp2193 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2191, tmp725,  (int32_t)12, tmp2193);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp725);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2191);

auto tmp2196 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2193, tmp2168, tmp2196);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2193);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2168);

auto tmp2199 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2196, tmp726, tmp727,  (int32_t)12, tmp2199);
ClearMemSecret1( (int32_t)2048, tmp727);
ClearMemSecret1( (int32_t)2048, tmp726);

auto tmp2202 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2199, tmp2202);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2199);

auto tmp2204 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2202, tmp730,  (int32_t)12, tmp2204);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2202);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp730);

auto tmp2207 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2204, tmp731, tmp732,  (int32_t)12, tmp2207);
ClearMemSecret1( (int32_t)512, tmp731);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2204);
ClearMemSecret1( (int32_t)512, tmp732);

auto tmp2211 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2207, tmp2211);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2207);

auto tmp2213 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp2211, tmp735,  (int32_t)12, tmp2213);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2211);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp735);

auto tmp2216 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2213, tmp736, tmp737,  (int32_t)12, tmp2216);
ClearMemSecret1( (int32_t)512, tmp736);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2213);
ClearMemSecret1( (int32_t)512, tmp737);

auto tmp2220 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2216, tmp2220);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2216);

auto tmp2222 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2220, tmp740,  (int32_t)12, tmp2222);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2220);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp740);

auto tmp2225 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2222, tmp2196, tmp2225);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2222);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2196);

auto tmp2228 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2225, tmp741, tmp742,  (int32_t)12, tmp2228);
ClearMemSecret1( (int32_t)2048, tmp742);
ClearMemSecret1( (int32_t)2048, tmp741);

auto tmp2231 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2228, tmp2231);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2228);

auto tmp2233 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2231, tmp745,  (int32_t)12, tmp2233);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2231);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)512, tmp745);

auto tmp2236 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2233, tmp746, tmp747,  (int32_t)12, tmp2236);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2233);
ClearMemSecret1( (int32_t)512, tmp747);
ClearMemSecret1( (int32_t)512, tmp746);

auto tmp2240 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2236, tmp2240);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2236);

auto tmp2242 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp2240, tmp750,  (int32_t)12, tmp2242);
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)512,  (int32_t)512, tmp750);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2240);

auto tmp2245 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2242, tmp751, tmp752,  (int32_t)12, tmp2245);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2242);
ClearMemSecret1( (int32_t)512, tmp751);
ClearMemSecret1( (int32_t)512, tmp752);

auto tmp2249 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2245, tmp2249);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2245);

auto tmp2251 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp2249, tmp755,  (int32_t)12, tmp2251);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp2249);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)2048, tmp755);

auto tmp2254 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
MatAdd4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2251, tmp2225, tmp2254);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2225);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2251);

auto tmp2257 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2254, tmp756, tmp757,  (int32_t)12, tmp2257);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2254);
ClearMemSecret1( (int32_t)2048, tmp757);
ClearMemSecret1( (int32_t)2048, tmp756);

auto tmp2261 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048);
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2257, tmp2261);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2257);

auto tmp2263 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048);
AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2261, tmp2263);
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)2048, tmp2261);

auto tmp2265 = make_vector<uint64_t>( (int32_t)1,  (int32_t)2048);
Squeeze24( (int32_t)1,  (int32_t)2048,  (int32_t)1,  (int32_t)2,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp2263, tmp2265);
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)2048, tmp2263);

auto tmp2267 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatMulCSF2D( (int32_t)1,  (int32_t)2048,  (int32_t)1001, tmp2265, tmp760, tmp2267,  (int64_t)12);
ClearMemSecret2( (int32_t)1,  (int32_t)2048, tmp2265);
ClearMemSecret2( (int32_t)2048,  (int32_t)1001, tmp760);

auto tmp2270 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1001);
MatAddBroadCast2( (int32_t)1,  (int32_t)1001, tmp2267, tmp761, tmp2270);
ClearMemSecret1( (int32_t)1001, tmp761);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp2267);

auto tmp2273 = make_vector<uint64_t>( (int32_t)1);
ArgMax1( (int32_t)1,  (int32_t)1,  (int32_t)1001, tmp2270,  (int32_t)1, tmp2273);
ClearMemSecret2( (int32_t)1,  (int32_t)1001, tmp2270);
ClearMemPublic( (int32_t)1);
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
print_integer(funcReconstruct2PCCons(tmp2273[i0], 1));
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
