#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "res_net_mem_opti.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef DENSE_NET

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

		int32_t leftTopCornerH = ( (int32_t)0 - zPadHLeft );

		int32_t extremeRightBottomCornerH = ((H -  (int32_t)1) + zPadHRight);
		while ((((leftTopCornerH + FH) -  (int32_t)1) <= extremeRightBottomCornerH)) {

			int32_t leftTopCornerW = ( (int32_t)0 - zPadWLeft );

			int32_t extremeRightBottomCornerW = ((W -  (int32_t)1) + zPadWRight);
			while ((((leftTopCornerW + FW) -  (int32_t)1) <= extremeRightBottomCornerW)) {
				for (uint32_t fh =  (int32_t)0; fh < FH; fh++){
					for (uint32_t fw =  (int32_t)0; fw < FW; fw++){

						int32_t curPosH = (leftTopCornerH + fh);

						int32_t curPosW = (leftTopCornerW + fw);

						uint64_t val = ( (int64_t)0 );
						for (uint32_t ci =  (int32_t)0; ci < CI; ci++){
							if ((((curPosH <  (int32_t)0) || (curPosH >= H)) || ((curPosW <  (int32_t)0) || (curPosW >= W)))) {
								val = ( (int64_t)0 );
								
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

void Conv2DCSF(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, auto& inputArr, auto& filterArr, int32_t consSF, auto& outArr){
#ifdef CONV_OPTI
	if ((FH>=5) || (FW>=5)){
				funcConv2DCSF(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, consSF, outArr);
					
	}
	else{
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

	int32_t lbounds1 = paddings[ (int32_t)0 ][ (int32_t)0 ];

	int32_t rbounds1excl = (s1 - paddings[ (int32_t)0 ][ (int32_t)1 ]);

	int32_t lbounds2 = paddings[ (int32_t)1 ][ (int32_t)0 ];

	int32_t rbounds2excl = (s2 - paddings[ (int32_t)1 ][ (int32_t)1 ]);

	int32_t lbounds3 = paddings[ (int32_t)2 ][ (int32_t)0 ];

	int32_t rbounds3excl = (s3 - paddings[ (int32_t)2 ][ (int32_t)1 ]);

	int32_t lbounds4 = paddings[ (int32_t)3 ][ (int32_t)0 ];

	int32_t rbounds4excl = (s4 - paddings[ (int32_t)3 ][ (int32_t)1 ]);
	for (uint32_t i =  (int32_t)0; i < s1; i++){
		for (uint32_t j =  (int32_t)0; j < s2; j++){
			for (uint32_t k =  (int32_t)0; k < s3; k++){
				for (uint32_t l =  (int32_t)0; l < s4; l++){
					if (((((((((i >= lbounds1) && (i < rbounds1excl)) && (j >= lbounds2)) && (j < rbounds2excl)) && (k >= lbounds3)) && (k < rbounds3excl)) && (l >= lbounds4)) && (l < rbounds4excl))) {
						outArr[i][j][k][l] = inpArr[(i - paddings[ (int32_t)0 ][ (int32_t)0 ])][(j - paddings[ (int32_t)1 ][ (int32_t)0 ])][(k - paddings[ (int32_t)2 ][ (int32_t)0 ])][(l - paddings[ (int32_t)3 ][ (int32_t)0 ])];
						
					} else {
						outArr[i][j][k][l] = ( (int64_t)0 );
						
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
auto tmp0 = make_vector<uint64_t>( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3 );
/* Variable to read the clear value corresponding to the input variable tmp0 at (393,1-393,47) */
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

auto tmp1 = make_vector<uint64_t>( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp1 at (396,1-396,44) */
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

auto tmp2 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp2 at (399,1-399,35) */
uint64_t __tmp_in_tmp2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp2;
		
	}
	tmp2[i0] = (role == CLIENT) ? __tmp_in_tmp2 : 0;
	
}

auto tmp3 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp3 at (402,1-402,35) */
uint64_t __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp3;
		
	}
	tmp3[i0] = (role == CLIENT) ? __tmp_in_tmp3 : 0;
	
}

auto tmp4 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp4 at (405,1-405,35) */
uint64_t __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp4;
		
	}
	tmp4[i0] = (role == CLIENT) ? __tmp_in_tmp4 : 0;
	
}

auto tmp5 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp5 at (408,1-408,35) */
uint64_t __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp5;
		
	}
	tmp5[i0] = (role == CLIENT) ? __tmp_in_tmp5 : 0;
	
}

auto tmp6 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp6 at (411,1-411,35) */
uint64_t __tmp_in_tmp6;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp6;
		
	}
	tmp6[i0] = (role == CLIENT) ? __tmp_in_tmp6 : 0;
	
}

auto tmp7 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp7 at (414,1-414,35) */
uint64_t __tmp_in_tmp7;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp7;
		
	}
	tmp7[i0] = (role == CLIENT) ? __tmp_in_tmp7 : 0;
	
}

auto tmp8 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp8 at (417,1-417,35) */
uint64_t __tmp_in_tmp8;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp8;
		
	}
	tmp8[i0] = (role == CLIENT) ? __tmp_in_tmp8 : 0;
	
}

auto tmp9 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp9 at (420,1-420,35) */
uint64_t __tmp_in_tmp9;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp9;
		
	}
	tmp9[i0] = (role == CLIENT) ? __tmp_in_tmp9 : 0;
	
}

auto tmp10 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp10 at (423,1-423,47) */
uint64_t __tmp_in_tmp10;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp10;
					
				}
				tmp10[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp10 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp11 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp11 at (426,1-426,37) */
uint64_t __tmp_in_tmp11;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp11;
		
	}
	tmp11[i0] = (role == CLIENT) ? __tmp_in_tmp11 : 0;
	
}

auto tmp12 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp12 at (429,1-429,37) */
uint64_t __tmp_in_tmp12;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp12;
		
	}
	tmp12[i0] = (role == CLIENT) ? __tmp_in_tmp12 : 0;
	
}

auto tmp13 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp13 at (432,1-432,37) */
uint64_t __tmp_in_tmp13;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp13;
		
	}
	tmp13[i0] = (role == CLIENT) ? __tmp_in_tmp13 : 0;
	
}

auto tmp14 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp14 at (435,1-435,37) */
uint64_t __tmp_in_tmp14;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp14;
		
	}
	tmp14[i0] = (role == CLIENT) ? __tmp_in_tmp14 : 0;
	
}

auto tmp15 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp15 at (438,1-438,47) */
uint64_t __tmp_in_tmp15;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp15;
					
				}
				tmp15[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp15 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp16 = make_vector<uint64_t>( (int32_t)96 );
/* Variable to read the clear value corresponding to the input variable tmp16 at (441,1-441,36) */
uint64_t __tmp_in_tmp16;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp16;
		
	}
	tmp16[i0] = (role == CLIENT) ? __tmp_in_tmp16 : 0;
	
}

auto tmp17 = make_vector<uint64_t>( (int32_t)96 );
/* Variable to read the clear value corresponding to the input variable tmp17 at (444,1-444,36) */
uint64_t __tmp_in_tmp17;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp17;
		
	}
	tmp17[i0] = (role == CLIENT) ? __tmp_in_tmp17 : 0;
	
}

auto tmp18 = make_vector<uint64_t>( (int32_t)96 );
/* Variable to read the clear value corresponding to the input variable tmp18 at (447,1-447,36) */
uint64_t __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp18;
		
	}
	tmp18[i0] = (role == CLIENT) ? __tmp_in_tmp18 : 0;
	
}

auto tmp19 = make_vector<uint64_t>( (int32_t)96 );
/* Variable to read the clear value corresponding to the input variable tmp19 at (450,1-450,36) */
uint64_t __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)96; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp19;
		
	}
	tmp19[i0] = (role == CLIENT) ? __tmp_in_tmp19 : 0;
	
}

auto tmp20 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)96,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp20 at (453,1-453,47) */
uint64_t __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)96; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp20;
					
				}
				tmp20[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp20 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp21 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp21 at (456,1-456,37) */
uint64_t __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp21;
		
	}
	tmp21[i0] = (role == CLIENT) ? __tmp_in_tmp21 : 0;
	
}

auto tmp22 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp22 at (459,1-459,37) */
uint64_t __tmp_in_tmp22;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp22;
		
	}
	tmp22[i0] = (role == CLIENT) ? __tmp_in_tmp22 : 0;
	
}

auto tmp23 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp23 at (462,1-462,37) */
uint64_t __tmp_in_tmp23;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp23;
		
	}
	tmp23[i0] = (role == CLIENT) ? __tmp_in_tmp23 : 0;
	
}

auto tmp24 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp24 at (465,1-465,37) */
uint64_t __tmp_in_tmp24;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp24;
		
	}
	tmp24[i0] = (role == CLIENT) ? __tmp_in_tmp24 : 0;
	
}

auto tmp25 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp25 at (468,1-468,47) */
uint64_t __tmp_in_tmp25;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp25;
					
				}
				tmp25[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp25 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp26 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp26 at (471,1-471,37) */
uint64_t __tmp_in_tmp26;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp26;
		
	}
	tmp26[i0] = (role == CLIENT) ? __tmp_in_tmp26 : 0;
	
}

auto tmp27 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp27 at (474,1-474,37) */
uint64_t __tmp_in_tmp27;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp27;
		
	}
	tmp27[i0] = (role == CLIENT) ? __tmp_in_tmp27 : 0;
	
}

auto tmp28 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp28 at (477,1-477,37) */
uint64_t __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp28;
		
	}
	tmp28[i0] = (role == CLIENT) ? __tmp_in_tmp28 : 0;
	
}

auto tmp29 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp29 at (480,1-480,37) */
uint64_t __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp29;
		
	}
	tmp29[i0] = (role == CLIENT) ? __tmp_in_tmp29 : 0;
	
}

auto tmp30 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp30 at (483,1-483,48) */
uint64_t __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp30;
					
				}
				tmp30[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp30 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp31 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp31 at (486,1-486,37) */
uint64_t __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp31;
		
	}
	tmp31[i0] = (role == CLIENT) ? __tmp_in_tmp31 : 0;
	
}

auto tmp32 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp32 at (489,1-489,37) */
uint64_t __tmp_in_tmp32;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp32;
		
	}
	tmp32[i0] = (role == CLIENT) ? __tmp_in_tmp32 : 0;
	
}

auto tmp33 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp33 at (492,1-492,37) */
uint64_t __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp33;
		
	}
	tmp33[i0] = (role == CLIENT) ? __tmp_in_tmp33 : 0;
	
}

auto tmp34 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp34 at (495,1-495,37) */
uint64_t __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp34;
		
	}
	tmp34[i0] = (role == CLIENT) ? __tmp_in_tmp34 : 0;
	
}

auto tmp35 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp35 at (498,1-498,47) */
uint64_t __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp35;
					
				}
				tmp35[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp35 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp36 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp36 at (501,1-501,37) */
uint64_t __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp36;
		
	}
	tmp36[i0] = (role == CLIENT) ? __tmp_in_tmp36 : 0;
	
}

auto tmp37 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp37 at (504,1-504,37) */
uint64_t __tmp_in_tmp37;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp37;
		
	}
	tmp37[i0] = (role == CLIENT) ? __tmp_in_tmp37 : 0;
	
}

auto tmp38 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp38 at (507,1-507,37) */
uint64_t __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp38;
		
	}
	tmp38[i0] = (role == CLIENT) ? __tmp_in_tmp38 : 0;
	
}

auto tmp39 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp39 at (510,1-510,37) */
uint64_t __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp39;
		
	}
	tmp39[i0] = (role == CLIENT) ? __tmp_in_tmp39 : 0;
	
}

auto tmp40 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)160,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp40 at (513,1-513,48) */
uint64_t __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)160; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp40;
					
				}
				tmp40[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp40 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp41 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp41 at (516,1-516,37) */
uint64_t __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp41;
		
	}
	tmp41[i0] = (role == CLIENT) ? __tmp_in_tmp41 : 0;
	
}

auto tmp42 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp42 at (519,1-519,37) */
uint64_t __tmp_in_tmp42;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp42;
		
	}
	tmp42[i0] = (role == CLIENT) ? __tmp_in_tmp42 : 0;
	
}

auto tmp43 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp43 at (522,1-522,37) */
uint64_t __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp43;
		
	}
	tmp43[i0] = (role == CLIENT) ? __tmp_in_tmp43 : 0;
	
}

auto tmp44 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp44 at (525,1-525,37) */
uint64_t __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp44;
		
	}
	tmp44[i0] = (role == CLIENT) ? __tmp_in_tmp44 : 0;
	
}

auto tmp45 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp45 at (528,1-528,47) */
uint64_t __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp45;
					
				}
				tmp45[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp45 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp46 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp46 at (531,1-531,37) */
uint64_t __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp46;
		
	}
	tmp46[i0] = (role == CLIENT) ? __tmp_in_tmp46 : 0;
	
}

auto tmp47 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp47 at (534,1-534,37) */
uint64_t __tmp_in_tmp47;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp47;
		
	}
	tmp47[i0] = (role == CLIENT) ? __tmp_in_tmp47 : 0;
	
}

auto tmp48 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp48 at (537,1-537,37) */
uint64_t __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp48;
		
	}
	tmp48[i0] = (role == CLIENT) ? __tmp_in_tmp48 : 0;
	
}

auto tmp49 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp49 at (540,1-540,37) */
uint64_t __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp49;
		
	}
	tmp49[i0] = (role == CLIENT) ? __tmp_in_tmp49 : 0;
	
}

auto tmp50 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp50 at (543,1-543,48) */
uint64_t __tmp_in_tmp50;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)192; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp50;
					
				}
				tmp50[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp50 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp51 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp51 at (546,1-546,37) */
uint64_t __tmp_in_tmp51;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp51;
		
	}
	tmp51[i0] = (role == CLIENT) ? __tmp_in_tmp51 : 0;
	
}

auto tmp52 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp52 at (549,1-549,37) */
uint64_t __tmp_in_tmp52;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp52;
		
	}
	tmp52[i0] = (role == CLIENT) ? __tmp_in_tmp52 : 0;
	
}

auto tmp53 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp53 at (552,1-552,37) */
uint64_t __tmp_in_tmp53;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp53;
		
	}
	tmp53[i0] = (role == CLIENT) ? __tmp_in_tmp53 : 0;
	
}

auto tmp54 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp54 at (555,1-555,37) */
uint64_t __tmp_in_tmp54;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp54;
		
	}
	tmp54[i0] = (role == CLIENT) ? __tmp_in_tmp54 : 0;
	
}

auto tmp55 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp55 at (558,1-558,47) */
uint64_t __tmp_in_tmp55;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp55;
					
				}
				tmp55[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp55 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp56 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp56 at (561,1-561,37) */
uint64_t __tmp_in_tmp56;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp56;
		
	}
	tmp56[i0] = (role == CLIENT) ? __tmp_in_tmp56 : 0;
	
}

auto tmp57 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp57 at (564,1-564,37) */
uint64_t __tmp_in_tmp57;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp57;
		
	}
	tmp57[i0] = (role == CLIENT) ? __tmp_in_tmp57 : 0;
	
}

auto tmp58 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp58 at (567,1-567,37) */
uint64_t __tmp_in_tmp58;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp58;
		
	}
	tmp58[i0] = (role == CLIENT) ? __tmp_in_tmp58 : 0;
	
}

auto tmp59 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp59 at (570,1-570,37) */
uint64_t __tmp_in_tmp59;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp59;
		
	}
	tmp59[i0] = (role == CLIENT) ? __tmp_in_tmp59 : 0;
	
}

auto tmp60 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)224,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp60 at (573,1-573,48) */
uint64_t __tmp_in_tmp60;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)224; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp60;
					
				}
				tmp60[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp60 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp61 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp61 at (576,1-576,37) */
uint64_t __tmp_in_tmp61;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp61;
		
	}
	tmp61[i0] = (role == CLIENT) ? __tmp_in_tmp61 : 0;
	
}

auto tmp62 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp62 at (579,1-579,37) */
uint64_t __tmp_in_tmp62;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp62;
		
	}
	tmp62[i0] = (role == CLIENT) ? __tmp_in_tmp62 : 0;
	
}

auto tmp63 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp63 at (582,1-582,37) */
uint64_t __tmp_in_tmp63;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp63;
		
	}
	tmp63[i0] = (role == CLIENT) ? __tmp_in_tmp63 : 0;
	
}

auto tmp64 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp64 at (585,1-585,37) */
uint64_t __tmp_in_tmp64;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp64;
		
	}
	tmp64[i0] = (role == CLIENT) ? __tmp_in_tmp64 : 0;
	
}

auto tmp65 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp65 at (588,1-588,47) */
uint64_t __tmp_in_tmp65;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp65;
					
				}
				tmp65[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp65 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp66 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp66 at (591,1-591,37) */
uint64_t __tmp_in_tmp66;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp66;
		
	}
	tmp66[i0] = (role == CLIENT) ? __tmp_in_tmp66 : 0;
	
}

auto tmp67 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp67 at (594,1-594,37) */
uint64_t __tmp_in_tmp67;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp67;
		
	}
	tmp67[i0] = (role == CLIENT) ? __tmp_in_tmp67 : 0;
	
}

auto tmp68 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp68 at (597,1-597,37) */
uint64_t __tmp_in_tmp68;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp68;
		
	}
	tmp68[i0] = (role == CLIENT) ? __tmp_in_tmp68 : 0;
	
}

auto tmp69 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp69 at (600,1-600,37) */
uint64_t __tmp_in_tmp69;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp69;
		
	}
	tmp69[i0] = (role == CLIENT) ? __tmp_in_tmp69 : 0;
	
}

auto tmp70 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp70 at (603,1-603,48) */
uint64_t __tmp_in_tmp70;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp70;
					
				}
				tmp70[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp70 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp71 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp71 at (606,1-606,37) */
uint64_t __tmp_in_tmp71;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp71;
		
	}
	tmp71[i0] = (role == CLIENT) ? __tmp_in_tmp71 : 0;
	
}

auto tmp72 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp72 at (609,1-609,37) */
uint64_t __tmp_in_tmp72;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp72;
		
	}
	tmp72[i0] = (role == CLIENT) ? __tmp_in_tmp72 : 0;
	
}

auto tmp73 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp73 at (612,1-612,37) */
uint64_t __tmp_in_tmp73;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp73;
		
	}
	tmp73[i0] = (role == CLIENT) ? __tmp_in_tmp73 : 0;
	
}

auto tmp74 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp74 at (615,1-615,37) */
uint64_t __tmp_in_tmp74;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp74;
		
	}
	tmp74[i0] = (role == CLIENT) ? __tmp_in_tmp74 : 0;
	
}

auto tmp75 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp75 at (618,1-618,48) */
uint64_t __tmp_in_tmp75;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp75;
					
				}
				tmp75[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp75 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp76 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp76 at (621,1-621,37) */
uint64_t __tmp_in_tmp76;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp76;
		
	}
	tmp76[i0] = (role == CLIENT) ? __tmp_in_tmp76 : 0;
	
}

auto tmp77 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp77 at (624,1-624,37) */
uint64_t __tmp_in_tmp77;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp77;
		
	}
	tmp77[i0] = (role == CLIENT) ? __tmp_in_tmp77 : 0;
	
}

auto tmp78 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp78 at (627,1-627,37) */
uint64_t __tmp_in_tmp78;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp78;
		
	}
	tmp78[i0] = (role == CLIENT) ? __tmp_in_tmp78 : 0;
	
}

auto tmp79 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp79 at (630,1-630,37) */
uint64_t __tmp_in_tmp79;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp79;
		
	}
	tmp79[i0] = (role == CLIENT) ? __tmp_in_tmp79 : 0;
	
}

auto tmp80 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp80 at (633,1-633,47) */
uint64_t __tmp_in_tmp80;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp80;
					
				}
				tmp80[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp80 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp81 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp81 at (636,1-636,37) */
uint64_t __tmp_in_tmp81;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp81;
		
	}
	tmp81[i0] = (role == CLIENT) ? __tmp_in_tmp81 : 0;
	
}

auto tmp82 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp82 at (639,1-639,37) */
uint64_t __tmp_in_tmp82;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp82;
		
	}
	tmp82[i0] = (role == CLIENT) ? __tmp_in_tmp82 : 0;
	
}

auto tmp83 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp83 at (642,1-642,37) */
uint64_t __tmp_in_tmp83;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp83;
		
	}
	tmp83[i0] = (role == CLIENT) ? __tmp_in_tmp83 : 0;
	
}

auto tmp84 = make_vector<uint64_t>( (int32_t)160 );
/* Variable to read the clear value corresponding to the input variable tmp84 at (645,1-645,37) */
uint64_t __tmp_in_tmp84;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)160; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp84;
		
	}
	tmp84[i0] = (role == CLIENT) ? __tmp_in_tmp84 : 0;
	
}

auto tmp85 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)160,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp85 at (648,1-648,48) */
uint64_t __tmp_in_tmp85;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)160; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp85;
					
				}
				tmp85[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp85 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp86 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp86 at (651,1-651,37) */
uint64_t __tmp_in_tmp86;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp86;
		
	}
	tmp86[i0] = (role == CLIENT) ? __tmp_in_tmp86 : 0;
	
}

auto tmp87 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp87 at (654,1-654,37) */
uint64_t __tmp_in_tmp87;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp87;
		
	}
	tmp87[i0] = (role == CLIENT) ? __tmp_in_tmp87 : 0;
	
}

auto tmp88 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp88 at (657,1-657,37) */
uint64_t __tmp_in_tmp88;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp88;
		
	}
	tmp88[i0] = (role == CLIENT) ? __tmp_in_tmp88 : 0;
	
}

auto tmp89 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp89 at (660,1-660,37) */
uint64_t __tmp_in_tmp89;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp89;
		
	}
	tmp89[i0] = (role == CLIENT) ? __tmp_in_tmp89 : 0;
	
}

auto tmp90 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp90 at (663,1-663,47) */
uint64_t __tmp_in_tmp90;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp90;
					
				}
				tmp90[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp90 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp91 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp91 at (666,1-666,37) */
uint64_t __tmp_in_tmp91;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp91;
		
	}
	tmp91[i0] = (role == CLIENT) ? __tmp_in_tmp91 : 0;
	
}

auto tmp92 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp92 at (669,1-669,37) */
uint64_t __tmp_in_tmp92;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp92;
		
	}
	tmp92[i0] = (role == CLIENT) ? __tmp_in_tmp92 : 0;
	
}

auto tmp93 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp93 at (672,1-672,37) */
uint64_t __tmp_in_tmp93;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp93;
		
	}
	tmp93[i0] = (role == CLIENT) ? __tmp_in_tmp93 : 0;
	
}

auto tmp94 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp94 at (675,1-675,37) */
uint64_t __tmp_in_tmp94;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp94;
		
	}
	tmp94[i0] = (role == CLIENT) ? __tmp_in_tmp94 : 0;
	
}

auto tmp95 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp95 at (678,1-678,48) */
uint64_t __tmp_in_tmp95;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)192; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp95;
					
				}
				tmp95[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp95 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp96 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp96 at (681,1-681,37) */
uint64_t __tmp_in_tmp96;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp96;
		
	}
	tmp96[i0] = (role == CLIENT) ? __tmp_in_tmp96 : 0;
	
}

auto tmp97 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp97 at (684,1-684,37) */
uint64_t __tmp_in_tmp97;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp97;
		
	}
	tmp97[i0] = (role == CLIENT) ? __tmp_in_tmp97 : 0;
	
}

auto tmp98 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp98 at (687,1-687,37) */
uint64_t __tmp_in_tmp98;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp98;
		
	}
	tmp98[i0] = (role == CLIENT) ? __tmp_in_tmp98 : 0;
	
}

auto tmp99 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp99 at (690,1-690,37) */
uint64_t __tmp_in_tmp99;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp99;
		
	}
	tmp99[i0] = (role == CLIENT) ? __tmp_in_tmp99 : 0;
	
}

auto tmp100 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp100 at (693,1-693,48) */
uint64_t __tmp_in_tmp100;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp100;
					
				}
				tmp100[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp100 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp101 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp101 at (696,1-696,38) */
uint64_t __tmp_in_tmp101;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp101;
		
	}
	tmp101[i0] = (role == CLIENT) ? __tmp_in_tmp101 : 0;
	
}

auto tmp102 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp102 at (699,1-699,38) */
uint64_t __tmp_in_tmp102;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp102;
		
	}
	tmp102[i0] = (role == CLIENT) ? __tmp_in_tmp102 : 0;
	
}

auto tmp103 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp103 at (702,1-702,38) */
uint64_t __tmp_in_tmp103;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp103;
		
	}
	tmp103[i0] = (role == CLIENT) ? __tmp_in_tmp103 : 0;
	
}

auto tmp104 = make_vector<uint64_t>( (int32_t)224 );
/* Variable to read the clear value corresponding to the input variable tmp104 at (705,1-705,38) */
uint64_t __tmp_in_tmp104;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)224; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp104;
		
	}
	tmp104[i0] = (role == CLIENT) ? __tmp_in_tmp104 : 0;
	
}

auto tmp105 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)224,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp105 at (708,1-708,49) */
uint64_t __tmp_in_tmp105;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)224; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp105;
					
				}
				tmp105[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp105 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp106 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp106 at (711,1-711,38) */
uint64_t __tmp_in_tmp106;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp106;
		
	}
	tmp106[i0] = (role == CLIENT) ? __tmp_in_tmp106 : 0;
	
}

auto tmp107 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp107 at (714,1-714,38) */
uint64_t __tmp_in_tmp107;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp107;
		
	}
	tmp107[i0] = (role == CLIENT) ? __tmp_in_tmp107 : 0;
	
}

auto tmp108 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp108 at (717,1-717,38) */
uint64_t __tmp_in_tmp108;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp108;
		
	}
	tmp108[i0] = (role == CLIENT) ? __tmp_in_tmp108 : 0;
	
}

auto tmp109 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp109 at (720,1-720,38) */
uint64_t __tmp_in_tmp109;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp109;
		
	}
	tmp109[i0] = (role == CLIENT) ? __tmp_in_tmp109 : 0;
	
}

auto tmp110 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp110 at (723,1-723,48) */
uint64_t __tmp_in_tmp110;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp110;
					
				}
				tmp110[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp110 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp111 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp111 at (726,1-726,38) */
uint64_t __tmp_in_tmp111;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp111;
		
	}
	tmp111[i0] = (role == CLIENT) ? __tmp_in_tmp111 : 0;
	
}

auto tmp112 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp112 at (729,1-729,38) */
uint64_t __tmp_in_tmp112;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp112;
		
	}
	tmp112[i0] = (role == CLIENT) ? __tmp_in_tmp112 : 0;
	
}

auto tmp113 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp113 at (732,1-732,38) */
uint64_t __tmp_in_tmp113;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp113;
		
	}
	tmp113[i0] = (role == CLIENT) ? __tmp_in_tmp113 : 0;
	
}

auto tmp114 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp114 at (735,1-735,38) */
uint64_t __tmp_in_tmp114;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp114;
		
	}
	tmp114[i0] = (role == CLIENT) ? __tmp_in_tmp114 : 0;
	
}

auto tmp115 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp115 at (738,1-738,49) */
uint64_t __tmp_in_tmp115;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp115;
					
				}
				tmp115[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp115 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp116 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp116 at (741,1-741,38) */
uint64_t __tmp_in_tmp116;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp116;
		
	}
	tmp116[i0] = (role == CLIENT) ? __tmp_in_tmp116 : 0;
	
}

auto tmp117 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp117 at (744,1-744,38) */
uint64_t __tmp_in_tmp117;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp117;
		
	}
	tmp117[i0] = (role == CLIENT) ? __tmp_in_tmp117 : 0;
	
}

auto tmp118 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp118 at (747,1-747,38) */
uint64_t __tmp_in_tmp118;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp118;
		
	}
	tmp118[i0] = (role == CLIENT) ? __tmp_in_tmp118 : 0;
	
}

auto tmp119 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp119 at (750,1-750,38) */
uint64_t __tmp_in_tmp119;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp119;
		
	}
	tmp119[i0] = (role == CLIENT) ? __tmp_in_tmp119 : 0;
	
}

auto tmp120 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp120 at (753,1-753,48) */
uint64_t __tmp_in_tmp120;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp120;
					
				}
				tmp120[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp120 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp121 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp121 at (756,1-756,38) */
uint64_t __tmp_in_tmp121;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp121;
		
	}
	tmp121[i0] = (role == CLIENT) ? __tmp_in_tmp121 : 0;
	
}

auto tmp122 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp122 at (759,1-759,38) */
uint64_t __tmp_in_tmp122;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp122;
		
	}
	tmp122[i0] = (role == CLIENT) ? __tmp_in_tmp122 : 0;
	
}

auto tmp123 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp123 at (762,1-762,38) */
uint64_t __tmp_in_tmp123;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp123;
		
	}
	tmp123[i0] = (role == CLIENT) ? __tmp_in_tmp123 : 0;
	
}

auto tmp124 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp124 at (765,1-765,38) */
uint64_t __tmp_in_tmp124;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp124;
		
	}
	tmp124[i0] = (role == CLIENT) ? __tmp_in_tmp124 : 0;
	
}

auto tmp125 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)288,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp125 at (768,1-768,49) */
uint64_t __tmp_in_tmp125;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)288; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp125;
					
				}
				tmp125[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp125 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp126 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp126 at (771,1-771,38) */
uint64_t __tmp_in_tmp126;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp126;
		
	}
	tmp126[i0] = (role == CLIENT) ? __tmp_in_tmp126 : 0;
	
}

auto tmp127 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp127 at (774,1-774,38) */
uint64_t __tmp_in_tmp127;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp127;
		
	}
	tmp127[i0] = (role == CLIENT) ? __tmp_in_tmp127 : 0;
	
}

auto tmp128 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp128 at (777,1-777,38) */
uint64_t __tmp_in_tmp128;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp128;
		
	}
	tmp128[i0] = (role == CLIENT) ? __tmp_in_tmp128 : 0;
	
}

auto tmp129 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp129 at (780,1-780,38) */
uint64_t __tmp_in_tmp129;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp129;
		
	}
	tmp129[i0] = (role == CLIENT) ? __tmp_in_tmp129 : 0;
	
}

auto tmp130 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp130 at (783,1-783,48) */
uint64_t __tmp_in_tmp130;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp130;
					
				}
				tmp130[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp130 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp131 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp131 at (786,1-786,38) */
uint64_t __tmp_in_tmp131;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp131;
		
	}
	tmp131[i0] = (role == CLIENT) ? __tmp_in_tmp131 : 0;
	
}

auto tmp132 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp132 at (789,1-789,38) */
uint64_t __tmp_in_tmp132;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp132;
		
	}
	tmp132[i0] = (role == CLIENT) ? __tmp_in_tmp132 : 0;
	
}

auto tmp133 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp133 at (792,1-792,38) */
uint64_t __tmp_in_tmp133;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp133;
		
	}
	tmp133[i0] = (role == CLIENT) ? __tmp_in_tmp133 : 0;
	
}

auto tmp134 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp134 at (795,1-795,38) */
uint64_t __tmp_in_tmp134;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp134;
		
	}
	tmp134[i0] = (role == CLIENT) ? __tmp_in_tmp134 : 0;
	
}

auto tmp135 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)320,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp135 at (798,1-798,49) */
uint64_t __tmp_in_tmp135;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)320; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp135;
					
				}
				tmp135[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp135 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp136 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp136 at (801,1-801,38) */
uint64_t __tmp_in_tmp136;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp136;
		
	}
	tmp136[i0] = (role == CLIENT) ? __tmp_in_tmp136 : 0;
	
}

auto tmp137 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp137 at (804,1-804,38) */
uint64_t __tmp_in_tmp137;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp137;
		
	}
	tmp137[i0] = (role == CLIENT) ? __tmp_in_tmp137 : 0;
	
}

auto tmp138 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp138 at (807,1-807,38) */
uint64_t __tmp_in_tmp138;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp138;
		
	}
	tmp138[i0] = (role == CLIENT) ? __tmp_in_tmp138 : 0;
	
}

auto tmp139 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp139 at (810,1-810,38) */
uint64_t __tmp_in_tmp139;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp139;
		
	}
	tmp139[i0] = (role == CLIENT) ? __tmp_in_tmp139 : 0;
	
}

auto tmp140 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp140 at (813,1-813,48) */
uint64_t __tmp_in_tmp140;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp140;
					
				}
				tmp140[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp140 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp141 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp141 at (816,1-816,38) */
uint64_t __tmp_in_tmp141;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp141;
		
	}
	tmp141[i0] = (role == CLIENT) ? __tmp_in_tmp141 : 0;
	
}

auto tmp142 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp142 at (819,1-819,38) */
uint64_t __tmp_in_tmp142;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp142;
		
	}
	tmp142[i0] = (role == CLIENT) ? __tmp_in_tmp142 : 0;
	
}

auto tmp143 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp143 at (822,1-822,38) */
uint64_t __tmp_in_tmp143;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp143;
		
	}
	tmp143[i0] = (role == CLIENT) ? __tmp_in_tmp143 : 0;
	
}

auto tmp144 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp144 at (825,1-825,38) */
uint64_t __tmp_in_tmp144;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp144;
		
	}
	tmp144[i0] = (role == CLIENT) ? __tmp_in_tmp144 : 0;
	
}

auto tmp145 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)352,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp145 at (828,1-828,49) */
uint64_t __tmp_in_tmp145;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)352; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp145;
					
				}
				tmp145[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp145 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp146 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp146 at (831,1-831,38) */
uint64_t __tmp_in_tmp146;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp146;
		
	}
	tmp146[i0] = (role == CLIENT) ? __tmp_in_tmp146 : 0;
	
}

auto tmp147 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp147 at (834,1-834,38) */
uint64_t __tmp_in_tmp147;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp147;
		
	}
	tmp147[i0] = (role == CLIENT) ? __tmp_in_tmp147 : 0;
	
}

auto tmp148 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp148 at (837,1-837,38) */
uint64_t __tmp_in_tmp148;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp148;
		
	}
	tmp148[i0] = (role == CLIENT) ? __tmp_in_tmp148 : 0;
	
}

auto tmp149 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp149 at (840,1-840,38) */
uint64_t __tmp_in_tmp149;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp149;
		
	}
	tmp149[i0] = (role == CLIENT) ? __tmp_in_tmp149 : 0;
	
}

auto tmp150 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp150 at (843,1-843,48) */
uint64_t __tmp_in_tmp150;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp150;
					
				}
				tmp150[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp150 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp151 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp151 at (846,1-846,38) */
uint64_t __tmp_in_tmp151;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp151;
		
	}
	tmp151[i0] = (role == CLIENT) ? __tmp_in_tmp151 : 0;
	
}

auto tmp152 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp152 at (849,1-849,38) */
uint64_t __tmp_in_tmp152;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp152;
		
	}
	tmp152[i0] = (role == CLIENT) ? __tmp_in_tmp152 : 0;
	
}

auto tmp153 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp153 at (852,1-852,38) */
uint64_t __tmp_in_tmp153;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp153;
		
	}
	tmp153[i0] = (role == CLIENT) ? __tmp_in_tmp153 : 0;
	
}

auto tmp154 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp154 at (855,1-855,38) */
uint64_t __tmp_in_tmp154;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp154;
		
	}
	tmp154[i0] = (role == CLIENT) ? __tmp_in_tmp154 : 0;
	
}

auto tmp155 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp155 at (858,1-858,49) */
uint64_t __tmp_in_tmp155;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp155;
					
				}
				tmp155[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp155 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp156 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp156 at (861,1-861,38) */
uint64_t __tmp_in_tmp156;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp156;
		
	}
	tmp156[i0] = (role == CLIENT) ? __tmp_in_tmp156 : 0;
	
}

auto tmp157 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp157 at (864,1-864,38) */
uint64_t __tmp_in_tmp157;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp157;
		
	}
	tmp157[i0] = (role == CLIENT) ? __tmp_in_tmp157 : 0;
	
}

auto tmp158 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp158 at (867,1-867,38) */
uint64_t __tmp_in_tmp158;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp158;
		
	}
	tmp158[i0] = (role == CLIENT) ? __tmp_in_tmp158 : 0;
	
}

auto tmp159 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp159 at (870,1-870,38) */
uint64_t __tmp_in_tmp159;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp159;
		
	}
	tmp159[i0] = (role == CLIENT) ? __tmp_in_tmp159 : 0;
	
}

auto tmp160 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp160 at (873,1-873,48) */
uint64_t __tmp_in_tmp160;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp160;
					
				}
				tmp160[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp160 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp161 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp161 at (876,1-876,38) */
uint64_t __tmp_in_tmp161;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp161;
		
	}
	tmp161[i0] = (role == CLIENT) ? __tmp_in_tmp161 : 0;
	
}

auto tmp162 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp162 at (879,1-879,38) */
uint64_t __tmp_in_tmp162;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp162;
		
	}
	tmp162[i0] = (role == CLIENT) ? __tmp_in_tmp162 : 0;
	
}

auto tmp163 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp163 at (882,1-882,38) */
uint64_t __tmp_in_tmp163;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp163;
		
	}
	tmp163[i0] = (role == CLIENT) ? __tmp_in_tmp163 : 0;
	
}

auto tmp164 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp164 at (885,1-885,38) */
uint64_t __tmp_in_tmp164;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp164;
		
	}
	tmp164[i0] = (role == CLIENT) ? __tmp_in_tmp164 : 0;
	
}

auto tmp165 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)416,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp165 at (888,1-888,49) */
uint64_t __tmp_in_tmp165;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)416; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp165;
					
				}
				tmp165[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp165 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp166 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp166 at (891,1-891,38) */
uint64_t __tmp_in_tmp166;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp166;
		
	}
	tmp166[i0] = (role == CLIENT) ? __tmp_in_tmp166 : 0;
	
}

auto tmp167 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp167 at (894,1-894,38) */
uint64_t __tmp_in_tmp167;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp167;
		
	}
	tmp167[i0] = (role == CLIENT) ? __tmp_in_tmp167 : 0;
	
}

auto tmp168 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp168 at (897,1-897,38) */
uint64_t __tmp_in_tmp168;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp168;
		
	}
	tmp168[i0] = (role == CLIENT) ? __tmp_in_tmp168 : 0;
	
}

auto tmp169 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp169 at (900,1-900,38) */
uint64_t __tmp_in_tmp169;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp169;
		
	}
	tmp169[i0] = (role == CLIENT) ? __tmp_in_tmp169 : 0;
	
}

auto tmp170 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp170 at (903,1-903,48) */
uint64_t __tmp_in_tmp170;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp170;
					
				}
				tmp170[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp170 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp171 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp171 at (906,1-906,38) */
uint64_t __tmp_in_tmp171;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp171;
		
	}
	tmp171[i0] = (role == CLIENT) ? __tmp_in_tmp171 : 0;
	
}

auto tmp172 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp172 at (909,1-909,38) */
uint64_t __tmp_in_tmp172;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp172;
		
	}
	tmp172[i0] = (role == CLIENT) ? __tmp_in_tmp172 : 0;
	
}

auto tmp173 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp173 at (912,1-912,38) */
uint64_t __tmp_in_tmp173;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp173;
		
	}
	tmp173[i0] = (role == CLIENT) ? __tmp_in_tmp173 : 0;
	
}

auto tmp174 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp174 at (915,1-915,38) */
uint64_t __tmp_in_tmp174;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp174;
		
	}
	tmp174[i0] = (role == CLIENT) ? __tmp_in_tmp174 : 0;
	
}

auto tmp175 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)448,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp175 at (918,1-918,49) */
uint64_t __tmp_in_tmp175;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)448; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp175;
					
				}
				tmp175[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp175 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp176 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp176 at (921,1-921,38) */
uint64_t __tmp_in_tmp176;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp176;
		
	}
	tmp176[i0] = (role == CLIENT) ? __tmp_in_tmp176 : 0;
	
}

auto tmp177 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp177 at (924,1-924,38) */
uint64_t __tmp_in_tmp177;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp177;
		
	}
	tmp177[i0] = (role == CLIENT) ? __tmp_in_tmp177 : 0;
	
}

auto tmp178 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp178 at (927,1-927,38) */
uint64_t __tmp_in_tmp178;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp178;
		
	}
	tmp178[i0] = (role == CLIENT) ? __tmp_in_tmp178 : 0;
	
}

auto tmp179 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp179 at (930,1-930,38) */
uint64_t __tmp_in_tmp179;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp179;
		
	}
	tmp179[i0] = (role == CLIENT) ? __tmp_in_tmp179 : 0;
	
}

auto tmp180 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp180 at (933,1-933,48) */
uint64_t __tmp_in_tmp180;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp180;
					
				}
				tmp180[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp180 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp181 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp181 at (936,1-936,38) */
uint64_t __tmp_in_tmp181;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp181;
		
	}
	tmp181[i0] = (role == CLIENT) ? __tmp_in_tmp181 : 0;
	
}

auto tmp182 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp182 at (939,1-939,38) */
uint64_t __tmp_in_tmp182;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp182;
		
	}
	tmp182[i0] = (role == CLIENT) ? __tmp_in_tmp182 : 0;
	
}

auto tmp183 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp183 at (942,1-942,38) */
uint64_t __tmp_in_tmp183;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp183;
		
	}
	tmp183[i0] = (role == CLIENT) ? __tmp_in_tmp183 : 0;
	
}

auto tmp184 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp184 at (945,1-945,38) */
uint64_t __tmp_in_tmp184;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp184;
		
	}
	tmp184[i0] = (role == CLIENT) ? __tmp_in_tmp184 : 0;
	
}

auto tmp185 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)480,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp185 at (948,1-948,49) */
uint64_t __tmp_in_tmp185;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)480; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp185;
					
				}
				tmp185[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp185 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp186 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp186 at (951,1-951,38) */
uint64_t __tmp_in_tmp186;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp186;
		
	}
	tmp186[i0] = (role == CLIENT) ? __tmp_in_tmp186 : 0;
	
}

auto tmp187 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp187 at (954,1-954,38) */
uint64_t __tmp_in_tmp187;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp187;
		
	}
	tmp187[i0] = (role == CLIENT) ? __tmp_in_tmp187 : 0;
	
}

auto tmp188 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp188 at (957,1-957,38) */
uint64_t __tmp_in_tmp188;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp188;
		
	}
	tmp188[i0] = (role == CLIENT) ? __tmp_in_tmp188 : 0;
	
}

auto tmp189 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp189 at (960,1-960,38) */
uint64_t __tmp_in_tmp189;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp189;
		
	}
	tmp189[i0] = (role == CLIENT) ? __tmp_in_tmp189 : 0;
	
}

auto tmp190 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp190 at (963,1-963,48) */
uint64_t __tmp_in_tmp190;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp190;
					
				}
				tmp190[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp190 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp191 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp191 at (966,1-966,38) */
uint64_t __tmp_in_tmp191;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp191;
		
	}
	tmp191[i0] = (role == CLIENT) ? __tmp_in_tmp191 : 0;
	
}

auto tmp192 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp192 at (969,1-969,38) */
uint64_t __tmp_in_tmp192;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp192;
		
	}
	tmp192[i0] = (role == CLIENT) ? __tmp_in_tmp192 : 0;
	
}

auto tmp193 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp193 at (972,1-972,38) */
uint64_t __tmp_in_tmp193;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp193;
		
	}
	tmp193[i0] = (role == CLIENT) ? __tmp_in_tmp193 : 0;
	
}

auto tmp194 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp194 at (975,1-975,38) */
uint64_t __tmp_in_tmp194;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp194;
		
	}
	tmp194[i0] = (role == CLIENT) ? __tmp_in_tmp194 : 0;
	
}

auto tmp195 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp195 at (978,1-978,49) */
uint64_t __tmp_in_tmp195;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp195;
					
				}
				tmp195[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp195 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp196 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp196 at (981,1-981,38) */
uint64_t __tmp_in_tmp196;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp196;
		
	}
	tmp196[i0] = (role == CLIENT) ? __tmp_in_tmp196 : 0;
	
}

auto tmp197 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp197 at (984,1-984,38) */
uint64_t __tmp_in_tmp197;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp197;
		
	}
	tmp197[i0] = (role == CLIENT) ? __tmp_in_tmp197 : 0;
	
}

auto tmp198 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp198 at (987,1-987,38) */
uint64_t __tmp_in_tmp198;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp198;
		
	}
	tmp198[i0] = (role == CLIENT) ? __tmp_in_tmp198 : 0;
	
}

auto tmp199 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp199 at (990,1-990,38) */
uint64_t __tmp_in_tmp199;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp199;
		
	}
	tmp199[i0] = (role == CLIENT) ? __tmp_in_tmp199 : 0;
	
}

auto tmp200 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp200 at (993,1-993,49) */
uint64_t __tmp_in_tmp200;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp200;
					
				}
				tmp200[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp200 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp201 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp201 at (996,1-996,38) */
uint64_t __tmp_in_tmp201;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp201;
		
	}
	tmp201[i0] = (role == CLIENT) ? __tmp_in_tmp201 : 0;
	
}

auto tmp202 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp202 at (999,1-999,38) */
uint64_t __tmp_in_tmp202;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp202;
		
	}
	tmp202[i0] = (role == CLIENT) ? __tmp_in_tmp202 : 0;
	
}

auto tmp203 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp203 at (1002,1-1002,38) */
uint64_t __tmp_in_tmp203;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp203;
		
	}
	tmp203[i0] = (role == CLIENT) ? __tmp_in_tmp203 : 0;
	
}

auto tmp204 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp204 at (1005,1-1005,38) */
uint64_t __tmp_in_tmp204;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp204;
		
	}
	tmp204[i0] = (role == CLIENT) ? __tmp_in_tmp204 : 0;
	
}

auto tmp205 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp205 at (1008,1-1008,48) */
uint64_t __tmp_in_tmp205;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp205;
					
				}
				tmp205[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp205 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp206 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp206 at (1011,1-1011,38) */
uint64_t __tmp_in_tmp206;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp206;
		
	}
	tmp206[i0] = (role == CLIENT) ? __tmp_in_tmp206 : 0;
	
}

auto tmp207 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp207 at (1014,1-1014,38) */
uint64_t __tmp_in_tmp207;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp207;
		
	}
	tmp207[i0] = (role == CLIENT) ? __tmp_in_tmp207 : 0;
	
}

auto tmp208 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp208 at (1017,1-1017,38) */
uint64_t __tmp_in_tmp208;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp208;
		
	}
	tmp208[i0] = (role == CLIENT) ? __tmp_in_tmp208 : 0;
	
}

auto tmp209 = make_vector<uint64_t>( (int32_t)288 );
/* Variable to read the clear value corresponding to the input variable tmp209 at (1020,1-1020,38) */
uint64_t __tmp_in_tmp209;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)288; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp209;
		
	}
	tmp209[i0] = (role == CLIENT) ? __tmp_in_tmp209 : 0;
	
}

auto tmp210 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)288,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp210 at (1023,1-1023,49) */
uint64_t __tmp_in_tmp210;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)288; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp210;
					
				}
				tmp210[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp210 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp211 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp211 at (1026,1-1026,38) */
uint64_t __tmp_in_tmp211;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp211;
		
	}
	tmp211[i0] = (role == CLIENT) ? __tmp_in_tmp211 : 0;
	
}

auto tmp212 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp212 at (1029,1-1029,38) */
uint64_t __tmp_in_tmp212;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp212;
		
	}
	tmp212[i0] = (role == CLIENT) ? __tmp_in_tmp212 : 0;
	
}

auto tmp213 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp213 at (1032,1-1032,38) */
uint64_t __tmp_in_tmp213;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp213;
		
	}
	tmp213[i0] = (role == CLIENT) ? __tmp_in_tmp213 : 0;
	
}

auto tmp214 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp214 at (1035,1-1035,38) */
uint64_t __tmp_in_tmp214;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp214;
		
	}
	tmp214[i0] = (role == CLIENT) ? __tmp_in_tmp214 : 0;
	
}

auto tmp215 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp215 at (1038,1-1038,48) */
uint64_t __tmp_in_tmp215;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp215;
					
				}
				tmp215[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp215 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp216 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp216 at (1041,1-1041,38) */
uint64_t __tmp_in_tmp216;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp216;
		
	}
	tmp216[i0] = (role == CLIENT) ? __tmp_in_tmp216 : 0;
	
}

auto tmp217 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp217 at (1044,1-1044,38) */
uint64_t __tmp_in_tmp217;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp217;
		
	}
	tmp217[i0] = (role == CLIENT) ? __tmp_in_tmp217 : 0;
	
}

auto tmp218 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp218 at (1047,1-1047,38) */
uint64_t __tmp_in_tmp218;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp218;
		
	}
	tmp218[i0] = (role == CLIENT) ? __tmp_in_tmp218 : 0;
	
}

auto tmp219 = make_vector<uint64_t>( (int32_t)320 );
/* Variable to read the clear value corresponding to the input variable tmp219 at (1050,1-1050,38) */
uint64_t __tmp_in_tmp219;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)320; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp219;
		
	}
	tmp219[i0] = (role == CLIENT) ? __tmp_in_tmp219 : 0;
	
}

auto tmp220 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)320,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp220 at (1053,1-1053,49) */
uint64_t __tmp_in_tmp220;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)320; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp220;
					
				}
				tmp220[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp220 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp221 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp221 at (1056,1-1056,38) */
uint64_t __tmp_in_tmp221;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp221;
		
	}
	tmp221[i0] = (role == CLIENT) ? __tmp_in_tmp221 : 0;
	
}

auto tmp222 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp222 at (1059,1-1059,38) */
uint64_t __tmp_in_tmp222;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp222;
		
	}
	tmp222[i0] = (role == CLIENT) ? __tmp_in_tmp222 : 0;
	
}

auto tmp223 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp223 at (1062,1-1062,38) */
uint64_t __tmp_in_tmp223;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp223;
		
	}
	tmp223[i0] = (role == CLIENT) ? __tmp_in_tmp223 : 0;
	
}

auto tmp224 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp224 at (1065,1-1065,38) */
uint64_t __tmp_in_tmp224;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp224;
		
	}
	tmp224[i0] = (role == CLIENT) ? __tmp_in_tmp224 : 0;
	
}

auto tmp225 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp225 at (1068,1-1068,48) */
uint64_t __tmp_in_tmp225;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp225;
					
				}
				tmp225[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp225 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp226 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp226 at (1071,1-1071,38) */
uint64_t __tmp_in_tmp226;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp226;
		
	}
	tmp226[i0] = (role == CLIENT) ? __tmp_in_tmp226 : 0;
	
}

auto tmp227 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp227 at (1074,1-1074,38) */
uint64_t __tmp_in_tmp227;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp227;
		
	}
	tmp227[i0] = (role == CLIENT) ? __tmp_in_tmp227 : 0;
	
}

auto tmp228 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp228 at (1077,1-1077,38) */
uint64_t __tmp_in_tmp228;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp228;
		
	}
	tmp228[i0] = (role == CLIENT) ? __tmp_in_tmp228 : 0;
	
}

auto tmp229 = make_vector<uint64_t>( (int32_t)352 );
/* Variable to read the clear value corresponding to the input variable tmp229 at (1080,1-1080,38) */
uint64_t __tmp_in_tmp229;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)352; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp229;
		
	}
	tmp229[i0] = (role == CLIENT) ? __tmp_in_tmp229 : 0;
	
}

auto tmp230 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)352,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp230 at (1083,1-1083,49) */
uint64_t __tmp_in_tmp230;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)352; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp230;
					
				}
				tmp230[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp230 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp231 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp231 at (1086,1-1086,38) */
uint64_t __tmp_in_tmp231;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp231;
		
	}
	tmp231[i0] = (role == CLIENT) ? __tmp_in_tmp231 : 0;
	
}

auto tmp232 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp232 at (1089,1-1089,38) */
uint64_t __tmp_in_tmp232;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp232;
		
	}
	tmp232[i0] = (role == CLIENT) ? __tmp_in_tmp232 : 0;
	
}

auto tmp233 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp233 at (1092,1-1092,38) */
uint64_t __tmp_in_tmp233;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp233;
		
	}
	tmp233[i0] = (role == CLIENT) ? __tmp_in_tmp233 : 0;
	
}

auto tmp234 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp234 at (1095,1-1095,38) */
uint64_t __tmp_in_tmp234;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp234;
		
	}
	tmp234[i0] = (role == CLIENT) ? __tmp_in_tmp234 : 0;
	
}

auto tmp235 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp235 at (1098,1-1098,48) */
uint64_t __tmp_in_tmp235;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp235;
					
				}
				tmp235[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp235 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp236 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp236 at (1101,1-1101,38) */
uint64_t __tmp_in_tmp236;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp236;
		
	}
	tmp236[i0] = (role == CLIENT) ? __tmp_in_tmp236 : 0;
	
}

auto tmp237 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp237 at (1104,1-1104,38) */
uint64_t __tmp_in_tmp237;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp237;
		
	}
	tmp237[i0] = (role == CLIENT) ? __tmp_in_tmp237 : 0;
	
}

auto tmp238 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp238 at (1107,1-1107,38) */
uint64_t __tmp_in_tmp238;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp238;
		
	}
	tmp238[i0] = (role == CLIENT) ? __tmp_in_tmp238 : 0;
	
}

auto tmp239 = make_vector<uint64_t>( (int32_t)384 );
/* Variable to read the clear value corresponding to the input variable tmp239 at (1110,1-1110,38) */
uint64_t __tmp_in_tmp239;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)384; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp239;
		
	}
	tmp239[i0] = (role == CLIENT) ? __tmp_in_tmp239 : 0;
	
}

auto tmp240 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp240 at (1113,1-1113,49) */
uint64_t __tmp_in_tmp240;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp240;
					
				}
				tmp240[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp240 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp241 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp241 at (1116,1-1116,38) */
uint64_t __tmp_in_tmp241;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp241;
		
	}
	tmp241[i0] = (role == CLIENT) ? __tmp_in_tmp241 : 0;
	
}

auto tmp242 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp242 at (1119,1-1119,38) */
uint64_t __tmp_in_tmp242;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp242;
		
	}
	tmp242[i0] = (role == CLIENT) ? __tmp_in_tmp242 : 0;
	
}

auto tmp243 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp243 at (1122,1-1122,38) */
uint64_t __tmp_in_tmp243;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp243;
		
	}
	tmp243[i0] = (role == CLIENT) ? __tmp_in_tmp243 : 0;
	
}

auto tmp244 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp244 at (1125,1-1125,38) */
uint64_t __tmp_in_tmp244;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp244;
		
	}
	tmp244[i0] = (role == CLIENT) ? __tmp_in_tmp244 : 0;
	
}

auto tmp245 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp245 at (1128,1-1128,48) */
uint64_t __tmp_in_tmp245;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp245;
					
				}
				tmp245[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp245 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp246 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp246 at (1131,1-1131,38) */
uint64_t __tmp_in_tmp246;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp246;
		
	}
	tmp246[i0] = (role == CLIENT) ? __tmp_in_tmp246 : 0;
	
}

auto tmp247 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp247 at (1134,1-1134,38) */
uint64_t __tmp_in_tmp247;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp247;
		
	}
	tmp247[i0] = (role == CLIENT) ? __tmp_in_tmp247 : 0;
	
}

auto tmp248 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp248 at (1137,1-1137,38) */
uint64_t __tmp_in_tmp248;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp248;
		
	}
	tmp248[i0] = (role == CLIENT) ? __tmp_in_tmp248 : 0;
	
}

auto tmp249 = make_vector<uint64_t>( (int32_t)416 );
/* Variable to read the clear value corresponding to the input variable tmp249 at (1140,1-1140,38) */
uint64_t __tmp_in_tmp249;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)416; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp249;
		
	}
	tmp249[i0] = (role == CLIENT) ? __tmp_in_tmp249 : 0;
	
}

auto tmp250 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)416,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp250 at (1143,1-1143,49) */
uint64_t __tmp_in_tmp250;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)416; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp250;
					
				}
				tmp250[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp250 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp251 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp251 at (1146,1-1146,38) */
uint64_t __tmp_in_tmp251;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp251;
		
	}
	tmp251[i0] = (role == CLIENT) ? __tmp_in_tmp251 : 0;
	
}

auto tmp252 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp252 at (1149,1-1149,38) */
uint64_t __tmp_in_tmp252;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp252;
		
	}
	tmp252[i0] = (role == CLIENT) ? __tmp_in_tmp252 : 0;
	
}

auto tmp253 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp253 at (1152,1-1152,38) */
uint64_t __tmp_in_tmp253;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp253;
		
	}
	tmp253[i0] = (role == CLIENT) ? __tmp_in_tmp253 : 0;
	
}

auto tmp254 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp254 at (1155,1-1155,38) */
uint64_t __tmp_in_tmp254;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp254;
		
	}
	tmp254[i0] = (role == CLIENT) ? __tmp_in_tmp254 : 0;
	
}

auto tmp255 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp255 at (1158,1-1158,48) */
uint64_t __tmp_in_tmp255;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp255;
					
				}
				tmp255[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp255 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp256 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp256 at (1161,1-1161,38) */
uint64_t __tmp_in_tmp256;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp256;
		
	}
	tmp256[i0] = (role == CLIENT) ? __tmp_in_tmp256 : 0;
	
}

auto tmp257 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp257 at (1164,1-1164,38) */
uint64_t __tmp_in_tmp257;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp257;
		
	}
	tmp257[i0] = (role == CLIENT) ? __tmp_in_tmp257 : 0;
	
}

auto tmp258 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp258 at (1167,1-1167,38) */
uint64_t __tmp_in_tmp258;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp258;
		
	}
	tmp258[i0] = (role == CLIENT) ? __tmp_in_tmp258 : 0;
	
}

auto tmp259 = make_vector<uint64_t>( (int32_t)448 );
/* Variable to read the clear value corresponding to the input variable tmp259 at (1170,1-1170,38) */
uint64_t __tmp_in_tmp259;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)448; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp259;
		
	}
	tmp259[i0] = (role == CLIENT) ? __tmp_in_tmp259 : 0;
	
}

auto tmp260 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)448,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp260 at (1173,1-1173,49) */
uint64_t __tmp_in_tmp260;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)448; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp260;
					
				}
				tmp260[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp260 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp261 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp261 at (1176,1-1176,38) */
uint64_t __tmp_in_tmp261;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp261;
		
	}
	tmp261[i0] = (role == CLIENT) ? __tmp_in_tmp261 : 0;
	
}

auto tmp262 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp262 at (1179,1-1179,38) */
uint64_t __tmp_in_tmp262;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp262;
		
	}
	tmp262[i0] = (role == CLIENT) ? __tmp_in_tmp262 : 0;
	
}

auto tmp263 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp263 at (1182,1-1182,38) */
uint64_t __tmp_in_tmp263;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp263;
		
	}
	tmp263[i0] = (role == CLIENT) ? __tmp_in_tmp263 : 0;
	
}

auto tmp264 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp264 at (1185,1-1185,38) */
uint64_t __tmp_in_tmp264;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp264;
		
	}
	tmp264[i0] = (role == CLIENT) ? __tmp_in_tmp264 : 0;
	
}

auto tmp265 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp265 at (1188,1-1188,48) */
uint64_t __tmp_in_tmp265;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp265;
					
				}
				tmp265[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp265 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp266 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp266 at (1191,1-1191,38) */
uint64_t __tmp_in_tmp266;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp266;
		
	}
	tmp266[i0] = (role == CLIENT) ? __tmp_in_tmp266 : 0;
	
}

auto tmp267 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp267 at (1194,1-1194,38) */
uint64_t __tmp_in_tmp267;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp267;
		
	}
	tmp267[i0] = (role == CLIENT) ? __tmp_in_tmp267 : 0;
	
}

auto tmp268 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp268 at (1197,1-1197,38) */
uint64_t __tmp_in_tmp268;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp268;
		
	}
	tmp268[i0] = (role == CLIENT) ? __tmp_in_tmp268 : 0;
	
}

auto tmp269 = make_vector<uint64_t>( (int32_t)480 );
/* Variable to read the clear value corresponding to the input variable tmp269 at (1200,1-1200,38) */
uint64_t __tmp_in_tmp269;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)480; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp269;
		
	}
	tmp269[i0] = (role == CLIENT) ? __tmp_in_tmp269 : 0;
	
}

auto tmp270 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)480,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp270 at (1203,1-1203,49) */
uint64_t __tmp_in_tmp270;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)480; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp270;
					
				}
				tmp270[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp270 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp271 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp271 at (1206,1-1206,38) */
uint64_t __tmp_in_tmp271;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp271;
		
	}
	tmp271[i0] = (role == CLIENT) ? __tmp_in_tmp271 : 0;
	
}

auto tmp272 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp272 at (1209,1-1209,38) */
uint64_t __tmp_in_tmp272;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp272;
		
	}
	tmp272[i0] = (role == CLIENT) ? __tmp_in_tmp272 : 0;
	
}

auto tmp273 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp273 at (1212,1-1212,38) */
uint64_t __tmp_in_tmp273;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp273;
		
	}
	tmp273[i0] = (role == CLIENT) ? __tmp_in_tmp273 : 0;
	
}

auto tmp274 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp274 at (1215,1-1215,38) */
uint64_t __tmp_in_tmp274;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp274;
		
	}
	tmp274[i0] = (role == CLIENT) ? __tmp_in_tmp274 : 0;
	
}

auto tmp275 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp275 at (1218,1-1218,48) */
uint64_t __tmp_in_tmp275;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp275;
					
				}
				tmp275[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp275 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp276 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp276 at (1221,1-1221,38) */
uint64_t __tmp_in_tmp276;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp276;
		
	}
	tmp276[i0] = (role == CLIENT) ? __tmp_in_tmp276 : 0;
	
}

auto tmp277 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp277 at (1224,1-1224,38) */
uint64_t __tmp_in_tmp277;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp277;
		
	}
	tmp277[i0] = (role == CLIENT) ? __tmp_in_tmp277 : 0;
	
}

auto tmp278 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp278 at (1227,1-1227,38) */
uint64_t __tmp_in_tmp278;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp278;
		
	}
	tmp278[i0] = (role == CLIENT) ? __tmp_in_tmp278 : 0;
	
}

auto tmp279 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp279 at (1230,1-1230,38) */
uint64_t __tmp_in_tmp279;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp279;
		
	}
	tmp279[i0] = (role == CLIENT) ? __tmp_in_tmp279 : 0;
	
}

auto tmp280 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp280 at (1233,1-1233,49) */
uint64_t __tmp_in_tmp280;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp280;
					
				}
				tmp280[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp280 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp281 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp281 at (1236,1-1236,38) */
uint64_t __tmp_in_tmp281;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp281;
		
	}
	tmp281[i0] = (role == CLIENT) ? __tmp_in_tmp281 : 0;
	
}

auto tmp282 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp282 at (1239,1-1239,38) */
uint64_t __tmp_in_tmp282;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp282;
		
	}
	tmp282[i0] = (role == CLIENT) ? __tmp_in_tmp282 : 0;
	
}

auto tmp283 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp283 at (1242,1-1242,38) */
uint64_t __tmp_in_tmp283;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp283;
		
	}
	tmp283[i0] = (role == CLIENT) ? __tmp_in_tmp283 : 0;
	
}

auto tmp284 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp284 at (1245,1-1245,38) */
uint64_t __tmp_in_tmp284;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp284;
		
	}
	tmp284[i0] = (role == CLIENT) ? __tmp_in_tmp284 : 0;
	
}

auto tmp285 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp285 at (1248,1-1248,48) */
uint64_t __tmp_in_tmp285;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp285;
					
				}
				tmp285[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp285 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp286 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp286 at (1251,1-1251,38) */
uint64_t __tmp_in_tmp286;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp286;
		
	}
	tmp286[i0] = (role == CLIENT) ? __tmp_in_tmp286 : 0;
	
}

auto tmp287 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp287 at (1254,1-1254,38) */
uint64_t __tmp_in_tmp287;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp287;
		
	}
	tmp287[i0] = (role == CLIENT) ? __tmp_in_tmp287 : 0;
	
}

auto tmp288 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp288 at (1257,1-1257,38) */
uint64_t __tmp_in_tmp288;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp288;
		
	}
	tmp288[i0] = (role == CLIENT) ? __tmp_in_tmp288 : 0;
	
}

auto tmp289 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp289 at (1260,1-1260,38) */
uint64_t __tmp_in_tmp289;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp289;
		
	}
	tmp289[i0] = (role == CLIENT) ? __tmp_in_tmp289 : 0;
	
}

auto tmp290 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)544,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp290 at (1263,1-1263,49) */
uint64_t __tmp_in_tmp290;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)544; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp290;
					
				}
				tmp290[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp290 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp291 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp291 at (1266,1-1266,38) */
uint64_t __tmp_in_tmp291;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp291;
		
	}
	tmp291[i0] = (role == CLIENT) ? __tmp_in_tmp291 : 0;
	
}

auto tmp292 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp292 at (1269,1-1269,38) */
uint64_t __tmp_in_tmp292;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp292;
		
	}
	tmp292[i0] = (role == CLIENT) ? __tmp_in_tmp292 : 0;
	
}

auto tmp293 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp293 at (1272,1-1272,38) */
uint64_t __tmp_in_tmp293;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp293;
		
	}
	tmp293[i0] = (role == CLIENT) ? __tmp_in_tmp293 : 0;
	
}

auto tmp294 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp294 at (1275,1-1275,38) */
uint64_t __tmp_in_tmp294;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp294;
		
	}
	tmp294[i0] = (role == CLIENT) ? __tmp_in_tmp294 : 0;
	
}

auto tmp295 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp295 at (1278,1-1278,48) */
uint64_t __tmp_in_tmp295;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp295;
					
				}
				tmp295[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp295 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp296 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp296 at (1281,1-1281,38) */
uint64_t __tmp_in_tmp296;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp296;
		
	}
	tmp296[i0] = (role == CLIENT) ? __tmp_in_tmp296 : 0;
	
}

auto tmp297 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp297 at (1284,1-1284,38) */
uint64_t __tmp_in_tmp297;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp297;
		
	}
	tmp297[i0] = (role == CLIENT) ? __tmp_in_tmp297 : 0;
	
}

auto tmp298 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp298 at (1287,1-1287,38) */
uint64_t __tmp_in_tmp298;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp298;
		
	}
	tmp298[i0] = (role == CLIENT) ? __tmp_in_tmp298 : 0;
	
}

auto tmp299 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp299 at (1290,1-1290,38) */
uint64_t __tmp_in_tmp299;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp299;
		
	}
	tmp299[i0] = (role == CLIENT) ? __tmp_in_tmp299 : 0;
	
}

auto tmp300 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)576,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp300 at (1293,1-1293,49) */
uint64_t __tmp_in_tmp300;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)576; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp300;
					
				}
				tmp300[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp300 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp301 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp301 at (1296,1-1296,38) */
uint64_t __tmp_in_tmp301;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp301;
		
	}
	tmp301[i0] = (role == CLIENT) ? __tmp_in_tmp301 : 0;
	
}

auto tmp302 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp302 at (1299,1-1299,38) */
uint64_t __tmp_in_tmp302;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp302;
		
	}
	tmp302[i0] = (role == CLIENT) ? __tmp_in_tmp302 : 0;
	
}

auto tmp303 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp303 at (1302,1-1302,38) */
uint64_t __tmp_in_tmp303;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp303;
		
	}
	tmp303[i0] = (role == CLIENT) ? __tmp_in_tmp303 : 0;
	
}

auto tmp304 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp304 at (1305,1-1305,38) */
uint64_t __tmp_in_tmp304;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp304;
		
	}
	tmp304[i0] = (role == CLIENT) ? __tmp_in_tmp304 : 0;
	
}

auto tmp305 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp305 at (1308,1-1308,48) */
uint64_t __tmp_in_tmp305;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp305;
					
				}
				tmp305[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp305 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp306 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp306 at (1311,1-1311,38) */
uint64_t __tmp_in_tmp306;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp306;
		
	}
	tmp306[i0] = (role == CLIENT) ? __tmp_in_tmp306 : 0;
	
}

auto tmp307 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp307 at (1314,1-1314,38) */
uint64_t __tmp_in_tmp307;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp307;
		
	}
	tmp307[i0] = (role == CLIENT) ? __tmp_in_tmp307 : 0;
	
}

auto tmp308 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp308 at (1317,1-1317,38) */
uint64_t __tmp_in_tmp308;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp308;
		
	}
	tmp308[i0] = (role == CLIENT) ? __tmp_in_tmp308 : 0;
	
}

auto tmp309 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp309 at (1320,1-1320,38) */
uint64_t __tmp_in_tmp309;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp309;
		
	}
	tmp309[i0] = (role == CLIENT) ? __tmp_in_tmp309 : 0;
	
}

auto tmp310 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)608,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp310 at (1323,1-1323,49) */
uint64_t __tmp_in_tmp310;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)608; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp310;
					
				}
				tmp310[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp310 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp311 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp311 at (1326,1-1326,38) */
uint64_t __tmp_in_tmp311;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp311;
		
	}
	tmp311[i0] = (role == CLIENT) ? __tmp_in_tmp311 : 0;
	
}

auto tmp312 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp312 at (1329,1-1329,38) */
uint64_t __tmp_in_tmp312;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp312;
		
	}
	tmp312[i0] = (role == CLIENT) ? __tmp_in_tmp312 : 0;
	
}

auto tmp313 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp313 at (1332,1-1332,38) */
uint64_t __tmp_in_tmp313;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp313;
		
	}
	tmp313[i0] = (role == CLIENT) ? __tmp_in_tmp313 : 0;
	
}

auto tmp314 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp314 at (1335,1-1335,38) */
uint64_t __tmp_in_tmp314;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp314;
		
	}
	tmp314[i0] = (role == CLIENT) ? __tmp_in_tmp314 : 0;
	
}

auto tmp315 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp315 at (1338,1-1338,48) */
uint64_t __tmp_in_tmp315;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp315;
					
				}
				tmp315[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp315 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp316 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp316 at (1341,1-1341,38) */
uint64_t __tmp_in_tmp316;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp316;
		
	}
	tmp316[i0] = (role == CLIENT) ? __tmp_in_tmp316 : 0;
	
}

auto tmp317 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp317 at (1344,1-1344,38) */
uint64_t __tmp_in_tmp317;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp317;
		
	}
	tmp317[i0] = (role == CLIENT) ? __tmp_in_tmp317 : 0;
	
}

auto tmp318 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp318 at (1347,1-1347,38) */
uint64_t __tmp_in_tmp318;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp318;
		
	}
	tmp318[i0] = (role == CLIENT) ? __tmp_in_tmp318 : 0;
	
}

auto tmp319 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp319 at (1350,1-1350,38) */
uint64_t __tmp_in_tmp319;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp319;
		
	}
	tmp319[i0] = (role == CLIENT) ? __tmp_in_tmp319 : 0;
	
}

auto tmp320 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)640,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp320 at (1353,1-1353,49) */
uint64_t __tmp_in_tmp320;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)640; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp320;
					
				}
				tmp320[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp320 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp321 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp321 at (1356,1-1356,38) */
uint64_t __tmp_in_tmp321;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp321;
		
	}
	tmp321[i0] = (role == CLIENT) ? __tmp_in_tmp321 : 0;
	
}

auto tmp322 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp322 at (1359,1-1359,38) */
uint64_t __tmp_in_tmp322;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp322;
		
	}
	tmp322[i0] = (role == CLIENT) ? __tmp_in_tmp322 : 0;
	
}

auto tmp323 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp323 at (1362,1-1362,38) */
uint64_t __tmp_in_tmp323;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp323;
		
	}
	tmp323[i0] = (role == CLIENT) ? __tmp_in_tmp323 : 0;
	
}

auto tmp324 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp324 at (1365,1-1365,38) */
uint64_t __tmp_in_tmp324;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp324;
		
	}
	tmp324[i0] = (role == CLIENT) ? __tmp_in_tmp324 : 0;
	
}

auto tmp325 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp325 at (1368,1-1368,48) */
uint64_t __tmp_in_tmp325;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp325;
					
				}
				tmp325[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp325 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp326 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp326 at (1371,1-1371,38) */
uint64_t __tmp_in_tmp326;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp326;
		
	}
	tmp326[i0] = (role == CLIENT) ? __tmp_in_tmp326 : 0;
	
}

auto tmp327 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp327 at (1374,1-1374,38) */
uint64_t __tmp_in_tmp327;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp327;
		
	}
	tmp327[i0] = (role == CLIENT) ? __tmp_in_tmp327 : 0;
	
}

auto tmp328 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp328 at (1377,1-1377,38) */
uint64_t __tmp_in_tmp328;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp328;
		
	}
	tmp328[i0] = (role == CLIENT) ? __tmp_in_tmp328 : 0;
	
}

auto tmp329 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp329 at (1380,1-1380,38) */
uint64_t __tmp_in_tmp329;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp329;
		
	}
	tmp329[i0] = (role == CLIENT) ? __tmp_in_tmp329 : 0;
	
}

auto tmp330 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)672,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp330 at (1383,1-1383,49) */
uint64_t __tmp_in_tmp330;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)672; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp330;
					
				}
				tmp330[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp330 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp331 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp331 at (1386,1-1386,38) */
uint64_t __tmp_in_tmp331;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp331;
		
	}
	tmp331[i0] = (role == CLIENT) ? __tmp_in_tmp331 : 0;
	
}

auto tmp332 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp332 at (1389,1-1389,38) */
uint64_t __tmp_in_tmp332;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp332;
		
	}
	tmp332[i0] = (role == CLIENT) ? __tmp_in_tmp332 : 0;
	
}

auto tmp333 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp333 at (1392,1-1392,38) */
uint64_t __tmp_in_tmp333;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp333;
		
	}
	tmp333[i0] = (role == CLIENT) ? __tmp_in_tmp333 : 0;
	
}

auto tmp334 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp334 at (1395,1-1395,38) */
uint64_t __tmp_in_tmp334;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp334;
		
	}
	tmp334[i0] = (role == CLIENT) ? __tmp_in_tmp334 : 0;
	
}

auto tmp335 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp335 at (1398,1-1398,48) */
uint64_t __tmp_in_tmp335;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp335;
					
				}
				tmp335[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp335 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp336 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp336 at (1401,1-1401,38) */
uint64_t __tmp_in_tmp336;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp336;
		
	}
	tmp336[i0] = (role == CLIENT) ? __tmp_in_tmp336 : 0;
	
}

auto tmp337 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp337 at (1404,1-1404,38) */
uint64_t __tmp_in_tmp337;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp337;
		
	}
	tmp337[i0] = (role == CLIENT) ? __tmp_in_tmp337 : 0;
	
}

auto tmp338 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp338 at (1407,1-1407,38) */
uint64_t __tmp_in_tmp338;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp338;
		
	}
	tmp338[i0] = (role == CLIENT) ? __tmp_in_tmp338 : 0;
	
}

auto tmp339 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp339 at (1410,1-1410,38) */
uint64_t __tmp_in_tmp339;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp339;
		
	}
	tmp339[i0] = (role == CLIENT) ? __tmp_in_tmp339 : 0;
	
}

auto tmp340 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)704,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp340 at (1413,1-1413,49) */
uint64_t __tmp_in_tmp340;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)704; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp340;
					
				}
				tmp340[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp340 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp341 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp341 at (1416,1-1416,38) */
uint64_t __tmp_in_tmp341;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp341;
		
	}
	tmp341[i0] = (role == CLIENT) ? __tmp_in_tmp341 : 0;
	
}

auto tmp342 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp342 at (1419,1-1419,38) */
uint64_t __tmp_in_tmp342;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp342;
		
	}
	tmp342[i0] = (role == CLIENT) ? __tmp_in_tmp342 : 0;
	
}

auto tmp343 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp343 at (1422,1-1422,38) */
uint64_t __tmp_in_tmp343;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp343;
		
	}
	tmp343[i0] = (role == CLIENT) ? __tmp_in_tmp343 : 0;
	
}

auto tmp344 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp344 at (1425,1-1425,38) */
uint64_t __tmp_in_tmp344;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp344;
		
	}
	tmp344[i0] = (role == CLIENT) ? __tmp_in_tmp344 : 0;
	
}

auto tmp345 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp345 at (1428,1-1428,48) */
uint64_t __tmp_in_tmp345;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp345;
					
				}
				tmp345[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp345 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp346 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp346 at (1431,1-1431,38) */
uint64_t __tmp_in_tmp346;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp346;
		
	}
	tmp346[i0] = (role == CLIENT) ? __tmp_in_tmp346 : 0;
	
}

auto tmp347 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp347 at (1434,1-1434,38) */
uint64_t __tmp_in_tmp347;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp347;
		
	}
	tmp347[i0] = (role == CLIENT) ? __tmp_in_tmp347 : 0;
	
}

auto tmp348 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp348 at (1437,1-1437,38) */
uint64_t __tmp_in_tmp348;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp348;
		
	}
	tmp348[i0] = (role == CLIENT) ? __tmp_in_tmp348 : 0;
	
}

auto tmp349 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp349 at (1440,1-1440,38) */
uint64_t __tmp_in_tmp349;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp349;
		
	}
	tmp349[i0] = (role == CLIENT) ? __tmp_in_tmp349 : 0;
	
}

auto tmp350 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)736,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp350 at (1443,1-1443,49) */
uint64_t __tmp_in_tmp350;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)736; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp350;
					
				}
				tmp350[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp350 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp351 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp351 at (1446,1-1446,38) */
uint64_t __tmp_in_tmp351;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp351;
		
	}
	tmp351[i0] = (role == CLIENT) ? __tmp_in_tmp351 : 0;
	
}

auto tmp352 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp352 at (1449,1-1449,38) */
uint64_t __tmp_in_tmp352;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp352;
		
	}
	tmp352[i0] = (role == CLIENT) ? __tmp_in_tmp352 : 0;
	
}

auto tmp353 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp353 at (1452,1-1452,38) */
uint64_t __tmp_in_tmp353;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp353;
		
	}
	tmp353[i0] = (role == CLIENT) ? __tmp_in_tmp353 : 0;
	
}

auto tmp354 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp354 at (1455,1-1455,38) */
uint64_t __tmp_in_tmp354;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp354;
		
	}
	tmp354[i0] = (role == CLIENT) ? __tmp_in_tmp354 : 0;
	
}

auto tmp355 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp355 at (1458,1-1458,48) */
uint64_t __tmp_in_tmp355;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp355;
					
				}
				tmp355[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp355 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp356 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp356 at (1461,1-1461,38) */
uint64_t __tmp_in_tmp356;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp356;
		
	}
	tmp356[i0] = (role == CLIENT) ? __tmp_in_tmp356 : 0;
	
}

auto tmp357 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp357 at (1464,1-1464,38) */
uint64_t __tmp_in_tmp357;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp357;
		
	}
	tmp357[i0] = (role == CLIENT) ? __tmp_in_tmp357 : 0;
	
}

auto tmp358 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp358 at (1467,1-1467,38) */
uint64_t __tmp_in_tmp358;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp358;
		
	}
	tmp358[i0] = (role == CLIENT) ? __tmp_in_tmp358 : 0;
	
}

auto tmp359 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp359 at (1470,1-1470,38) */
uint64_t __tmp_in_tmp359;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp359;
		
	}
	tmp359[i0] = (role == CLIENT) ? __tmp_in_tmp359 : 0;
	
}

auto tmp360 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)768,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp360 at (1473,1-1473,49) */
uint64_t __tmp_in_tmp360;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)768; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp360;
					
				}
				tmp360[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp360 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp361 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp361 at (1476,1-1476,38) */
uint64_t __tmp_in_tmp361;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp361;
		
	}
	tmp361[i0] = (role == CLIENT) ? __tmp_in_tmp361 : 0;
	
}

auto tmp362 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp362 at (1479,1-1479,38) */
uint64_t __tmp_in_tmp362;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp362;
		
	}
	tmp362[i0] = (role == CLIENT) ? __tmp_in_tmp362 : 0;
	
}

auto tmp363 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp363 at (1482,1-1482,38) */
uint64_t __tmp_in_tmp363;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp363;
		
	}
	tmp363[i0] = (role == CLIENT) ? __tmp_in_tmp363 : 0;
	
}

auto tmp364 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp364 at (1485,1-1485,38) */
uint64_t __tmp_in_tmp364;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp364;
		
	}
	tmp364[i0] = (role == CLIENT) ? __tmp_in_tmp364 : 0;
	
}

auto tmp365 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp365 at (1488,1-1488,48) */
uint64_t __tmp_in_tmp365;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp365;
					
				}
				tmp365[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp365 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp366 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp366 at (1491,1-1491,38) */
uint64_t __tmp_in_tmp366;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp366;
		
	}
	tmp366[i0] = (role == CLIENT) ? __tmp_in_tmp366 : 0;
	
}

auto tmp367 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp367 at (1494,1-1494,38) */
uint64_t __tmp_in_tmp367;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp367;
		
	}
	tmp367[i0] = (role == CLIENT) ? __tmp_in_tmp367 : 0;
	
}

auto tmp368 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp368 at (1497,1-1497,38) */
uint64_t __tmp_in_tmp368;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp368;
		
	}
	tmp368[i0] = (role == CLIENT) ? __tmp_in_tmp368 : 0;
	
}

auto tmp369 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp369 at (1500,1-1500,38) */
uint64_t __tmp_in_tmp369;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp369;
		
	}
	tmp369[i0] = (role == CLIENT) ? __tmp_in_tmp369 : 0;
	
}

auto tmp370 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)800,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp370 at (1503,1-1503,49) */
uint64_t __tmp_in_tmp370;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)800; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp370;
					
				}
				tmp370[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp370 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp371 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp371 at (1506,1-1506,38) */
uint64_t __tmp_in_tmp371;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp371;
		
	}
	tmp371[i0] = (role == CLIENT) ? __tmp_in_tmp371 : 0;
	
}

auto tmp372 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp372 at (1509,1-1509,38) */
uint64_t __tmp_in_tmp372;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp372;
		
	}
	tmp372[i0] = (role == CLIENT) ? __tmp_in_tmp372 : 0;
	
}

auto tmp373 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp373 at (1512,1-1512,38) */
uint64_t __tmp_in_tmp373;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp373;
		
	}
	tmp373[i0] = (role == CLIENT) ? __tmp_in_tmp373 : 0;
	
}

auto tmp374 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp374 at (1515,1-1515,38) */
uint64_t __tmp_in_tmp374;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp374;
		
	}
	tmp374[i0] = (role == CLIENT) ? __tmp_in_tmp374 : 0;
	
}

auto tmp375 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp375 at (1518,1-1518,48) */
uint64_t __tmp_in_tmp375;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp375;
					
				}
				tmp375[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp375 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp376 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp376 at (1521,1-1521,38) */
uint64_t __tmp_in_tmp376;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp376;
		
	}
	tmp376[i0] = (role == CLIENT) ? __tmp_in_tmp376 : 0;
	
}

auto tmp377 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp377 at (1524,1-1524,38) */
uint64_t __tmp_in_tmp377;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp377;
		
	}
	tmp377[i0] = (role == CLIENT) ? __tmp_in_tmp377 : 0;
	
}

auto tmp378 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp378 at (1527,1-1527,38) */
uint64_t __tmp_in_tmp378;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp378;
		
	}
	tmp378[i0] = (role == CLIENT) ? __tmp_in_tmp378 : 0;
	
}

auto tmp379 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp379 at (1530,1-1530,38) */
uint64_t __tmp_in_tmp379;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp379;
		
	}
	tmp379[i0] = (role == CLIENT) ? __tmp_in_tmp379 : 0;
	
}

auto tmp380 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)832,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp380 at (1533,1-1533,49) */
uint64_t __tmp_in_tmp380;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)832; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp380;
					
				}
				tmp380[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp380 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp381 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp381 at (1536,1-1536,38) */
uint64_t __tmp_in_tmp381;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp381;
		
	}
	tmp381[i0] = (role == CLIENT) ? __tmp_in_tmp381 : 0;
	
}

auto tmp382 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp382 at (1539,1-1539,38) */
uint64_t __tmp_in_tmp382;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp382;
		
	}
	tmp382[i0] = (role == CLIENT) ? __tmp_in_tmp382 : 0;
	
}

auto tmp383 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp383 at (1542,1-1542,38) */
uint64_t __tmp_in_tmp383;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp383;
		
	}
	tmp383[i0] = (role == CLIENT) ? __tmp_in_tmp383 : 0;
	
}

auto tmp384 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp384 at (1545,1-1545,38) */
uint64_t __tmp_in_tmp384;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp384;
		
	}
	tmp384[i0] = (role == CLIENT) ? __tmp_in_tmp384 : 0;
	
}

auto tmp385 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp385 at (1548,1-1548,48) */
uint64_t __tmp_in_tmp385;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp385;
					
				}
				tmp385[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp385 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp386 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp386 at (1551,1-1551,38) */
uint64_t __tmp_in_tmp386;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp386;
		
	}
	tmp386[i0] = (role == CLIENT) ? __tmp_in_tmp386 : 0;
	
}

auto tmp387 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp387 at (1554,1-1554,38) */
uint64_t __tmp_in_tmp387;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp387;
		
	}
	tmp387[i0] = (role == CLIENT) ? __tmp_in_tmp387 : 0;
	
}

auto tmp388 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp388 at (1557,1-1557,38) */
uint64_t __tmp_in_tmp388;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp388;
		
	}
	tmp388[i0] = (role == CLIENT) ? __tmp_in_tmp388 : 0;
	
}

auto tmp389 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp389 at (1560,1-1560,38) */
uint64_t __tmp_in_tmp389;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp389;
		
	}
	tmp389[i0] = (role == CLIENT) ? __tmp_in_tmp389 : 0;
	
}

auto tmp390 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)864,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp390 at (1563,1-1563,49) */
uint64_t __tmp_in_tmp390;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)864; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp390;
					
				}
				tmp390[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp390 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp391 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp391 at (1566,1-1566,38) */
uint64_t __tmp_in_tmp391;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp391;
		
	}
	tmp391[i0] = (role == CLIENT) ? __tmp_in_tmp391 : 0;
	
}

auto tmp392 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp392 at (1569,1-1569,38) */
uint64_t __tmp_in_tmp392;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp392;
		
	}
	tmp392[i0] = (role == CLIENT) ? __tmp_in_tmp392 : 0;
	
}

auto tmp393 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp393 at (1572,1-1572,38) */
uint64_t __tmp_in_tmp393;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp393;
		
	}
	tmp393[i0] = (role == CLIENT) ? __tmp_in_tmp393 : 0;
	
}

auto tmp394 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp394 at (1575,1-1575,38) */
uint64_t __tmp_in_tmp394;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp394;
		
	}
	tmp394[i0] = (role == CLIENT) ? __tmp_in_tmp394 : 0;
	
}

auto tmp395 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp395 at (1578,1-1578,48) */
uint64_t __tmp_in_tmp395;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp395;
					
				}
				tmp395[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp395 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp396 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp396 at (1581,1-1581,38) */
uint64_t __tmp_in_tmp396;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp396;
		
	}
	tmp396[i0] = (role == CLIENT) ? __tmp_in_tmp396 : 0;
	
}

auto tmp397 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp397 at (1584,1-1584,38) */
uint64_t __tmp_in_tmp397;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp397;
		
	}
	tmp397[i0] = (role == CLIENT) ? __tmp_in_tmp397 : 0;
	
}

auto tmp398 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp398 at (1587,1-1587,38) */
uint64_t __tmp_in_tmp398;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp398;
		
	}
	tmp398[i0] = (role == CLIENT) ? __tmp_in_tmp398 : 0;
	
}

auto tmp399 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp399 at (1590,1-1590,38) */
uint64_t __tmp_in_tmp399;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp399;
		
	}
	tmp399[i0] = (role == CLIENT) ? __tmp_in_tmp399 : 0;
	
}

auto tmp400 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)896,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp400 at (1593,1-1593,49) */
uint64_t __tmp_in_tmp400;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)896; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp400;
					
				}
				tmp400[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp400 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp401 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp401 at (1596,1-1596,38) */
uint64_t __tmp_in_tmp401;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp401;
		
	}
	tmp401[i0] = (role == CLIENT) ? __tmp_in_tmp401 : 0;
	
}

auto tmp402 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp402 at (1599,1-1599,38) */
uint64_t __tmp_in_tmp402;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp402;
		
	}
	tmp402[i0] = (role == CLIENT) ? __tmp_in_tmp402 : 0;
	
}

auto tmp403 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp403 at (1602,1-1602,38) */
uint64_t __tmp_in_tmp403;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp403;
		
	}
	tmp403[i0] = (role == CLIENT) ? __tmp_in_tmp403 : 0;
	
}

auto tmp404 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp404 at (1605,1-1605,38) */
uint64_t __tmp_in_tmp404;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp404;
		
	}
	tmp404[i0] = (role == CLIENT) ? __tmp_in_tmp404 : 0;
	
}

auto tmp405 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp405 at (1608,1-1608,48) */
uint64_t __tmp_in_tmp405;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp405;
					
				}
				tmp405[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp405 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp406 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp406 at (1611,1-1611,38) */
uint64_t __tmp_in_tmp406;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp406;
		
	}
	tmp406[i0] = (role == CLIENT) ? __tmp_in_tmp406 : 0;
	
}

auto tmp407 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp407 at (1614,1-1614,38) */
uint64_t __tmp_in_tmp407;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp407;
		
	}
	tmp407[i0] = (role == CLIENT) ? __tmp_in_tmp407 : 0;
	
}

auto tmp408 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp408 at (1617,1-1617,38) */
uint64_t __tmp_in_tmp408;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp408;
		
	}
	tmp408[i0] = (role == CLIENT) ? __tmp_in_tmp408 : 0;
	
}

auto tmp409 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp409 at (1620,1-1620,38) */
uint64_t __tmp_in_tmp409;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp409;
		
	}
	tmp409[i0] = (role == CLIENT) ? __tmp_in_tmp409 : 0;
	
}

auto tmp410 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)928,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp410 at (1623,1-1623,49) */
uint64_t __tmp_in_tmp410;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)928; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp410;
					
				}
				tmp410[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp410 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp411 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp411 at (1626,1-1626,38) */
uint64_t __tmp_in_tmp411;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp411;
		
	}
	tmp411[i0] = (role == CLIENT) ? __tmp_in_tmp411 : 0;
	
}

auto tmp412 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp412 at (1629,1-1629,38) */
uint64_t __tmp_in_tmp412;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp412;
		
	}
	tmp412[i0] = (role == CLIENT) ? __tmp_in_tmp412 : 0;
	
}

auto tmp413 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp413 at (1632,1-1632,38) */
uint64_t __tmp_in_tmp413;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp413;
		
	}
	tmp413[i0] = (role == CLIENT) ? __tmp_in_tmp413 : 0;
	
}

auto tmp414 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp414 at (1635,1-1635,38) */
uint64_t __tmp_in_tmp414;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp414;
		
	}
	tmp414[i0] = (role == CLIENT) ? __tmp_in_tmp414 : 0;
	
}

auto tmp415 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp415 at (1638,1-1638,48) */
uint64_t __tmp_in_tmp415;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp415;
					
				}
				tmp415[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp415 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp416 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp416 at (1641,1-1641,38) */
uint64_t __tmp_in_tmp416;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp416;
		
	}
	tmp416[i0] = (role == CLIENT) ? __tmp_in_tmp416 : 0;
	
}

auto tmp417 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp417 at (1644,1-1644,38) */
uint64_t __tmp_in_tmp417;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp417;
		
	}
	tmp417[i0] = (role == CLIENT) ? __tmp_in_tmp417 : 0;
	
}

auto tmp418 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp418 at (1647,1-1647,38) */
uint64_t __tmp_in_tmp418;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp418;
		
	}
	tmp418[i0] = (role == CLIENT) ? __tmp_in_tmp418 : 0;
	
}

auto tmp419 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp419 at (1650,1-1650,38) */
uint64_t __tmp_in_tmp419;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp419;
		
	}
	tmp419[i0] = (role == CLIENT) ? __tmp_in_tmp419 : 0;
	
}

auto tmp420 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)960,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp420 at (1653,1-1653,49) */
uint64_t __tmp_in_tmp420;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)960; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp420;
					
				}
				tmp420[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp420 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp421 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp421 at (1656,1-1656,38) */
uint64_t __tmp_in_tmp421;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp421;
		
	}
	tmp421[i0] = (role == CLIENT) ? __tmp_in_tmp421 : 0;
	
}

auto tmp422 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp422 at (1659,1-1659,38) */
uint64_t __tmp_in_tmp422;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp422;
		
	}
	tmp422[i0] = (role == CLIENT) ? __tmp_in_tmp422 : 0;
	
}

auto tmp423 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp423 at (1662,1-1662,38) */
uint64_t __tmp_in_tmp423;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp423;
		
	}
	tmp423[i0] = (role == CLIENT) ? __tmp_in_tmp423 : 0;
	
}

auto tmp424 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp424 at (1665,1-1665,38) */
uint64_t __tmp_in_tmp424;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp424;
		
	}
	tmp424[i0] = (role == CLIENT) ? __tmp_in_tmp424 : 0;
	
}

auto tmp425 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp425 at (1668,1-1668,48) */
uint64_t __tmp_in_tmp425;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp425;
					
				}
				tmp425[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp425 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp426 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp426 at (1671,1-1671,38) */
uint64_t __tmp_in_tmp426;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp426;
		
	}
	tmp426[i0] = (role == CLIENT) ? __tmp_in_tmp426 : 0;
	
}

auto tmp427 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp427 at (1674,1-1674,38) */
uint64_t __tmp_in_tmp427;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp427;
		
	}
	tmp427[i0] = (role == CLIENT) ? __tmp_in_tmp427 : 0;
	
}

auto tmp428 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp428 at (1677,1-1677,38) */
uint64_t __tmp_in_tmp428;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp428;
		
	}
	tmp428[i0] = (role == CLIENT) ? __tmp_in_tmp428 : 0;
	
}

auto tmp429 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp429 at (1680,1-1680,38) */
uint64_t __tmp_in_tmp429;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp429;
		
	}
	tmp429[i0] = (role == CLIENT) ? __tmp_in_tmp429 : 0;
	
}

auto tmp430 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)992,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp430 at (1683,1-1683,49) */
uint64_t __tmp_in_tmp430;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)992; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp430;
					
				}
				tmp430[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp430 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp431 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp431 at (1686,1-1686,38) */
uint64_t __tmp_in_tmp431;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp431;
		
	}
	tmp431[i0] = (role == CLIENT) ? __tmp_in_tmp431 : 0;
	
}

auto tmp432 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp432 at (1689,1-1689,38) */
uint64_t __tmp_in_tmp432;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp432;
		
	}
	tmp432[i0] = (role == CLIENT) ? __tmp_in_tmp432 : 0;
	
}

auto tmp433 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp433 at (1692,1-1692,38) */
uint64_t __tmp_in_tmp433;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp433;
		
	}
	tmp433[i0] = (role == CLIENT) ? __tmp_in_tmp433 : 0;
	
}

auto tmp434 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp434 at (1695,1-1695,38) */
uint64_t __tmp_in_tmp434;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp434;
		
	}
	tmp434[i0] = (role == CLIENT) ? __tmp_in_tmp434 : 0;
	
}

auto tmp435 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp435 at (1698,1-1698,48) */
uint64_t __tmp_in_tmp435;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp435;
					
				}
				tmp435[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp435 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp436 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp436 at (1701,1-1701,39) */
uint64_t __tmp_in_tmp436;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp436;
		
	}
	tmp436[i0] = (role == CLIENT) ? __tmp_in_tmp436 : 0;
	
}

auto tmp437 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp437 at (1704,1-1704,39) */
uint64_t __tmp_in_tmp437;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp437;
		
	}
	tmp437[i0] = (role == CLIENT) ? __tmp_in_tmp437 : 0;
	
}

auto tmp438 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp438 at (1707,1-1707,39) */
uint64_t __tmp_in_tmp438;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp438;
		
	}
	tmp438[i0] = (role == CLIENT) ? __tmp_in_tmp438 : 0;
	
}

auto tmp439 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp439 at (1710,1-1710,39) */
uint64_t __tmp_in_tmp439;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp439;
		
	}
	tmp439[i0] = (role == CLIENT) ? __tmp_in_tmp439 : 0;
	
}

auto tmp440 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp440 at (1713,1-1713,50) */
uint64_t __tmp_in_tmp440;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)512; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp440;
					
				}
				tmp440[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp440 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp441 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp441 at (1716,1-1716,38) */
uint64_t __tmp_in_tmp441;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp441;
		
	}
	tmp441[i0] = (role == CLIENT) ? __tmp_in_tmp441 : 0;
	
}

auto tmp442 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp442 at (1719,1-1719,38) */
uint64_t __tmp_in_tmp442;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp442;
		
	}
	tmp442[i0] = (role == CLIENT) ? __tmp_in_tmp442 : 0;
	
}

auto tmp443 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp443 at (1722,1-1722,38) */
uint64_t __tmp_in_tmp443;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp443;
		
	}
	tmp443[i0] = (role == CLIENT) ? __tmp_in_tmp443 : 0;
	
}

auto tmp444 = make_vector<uint64_t>( (int32_t)512 );
/* Variable to read the clear value corresponding to the input variable tmp444 at (1725,1-1725,38) */
uint64_t __tmp_in_tmp444;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)512; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp444;
		
	}
	tmp444[i0] = (role == CLIENT) ? __tmp_in_tmp444 : 0;
	
}

auto tmp445 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp445 at (1728,1-1728,49) */
uint64_t __tmp_in_tmp445;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp445;
					
				}
				tmp445[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp445 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp446 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp446 at (1731,1-1731,38) */
uint64_t __tmp_in_tmp446;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp446;
		
	}
	tmp446[i0] = (role == CLIENT) ? __tmp_in_tmp446 : 0;
	
}

auto tmp447 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp447 at (1734,1-1734,38) */
uint64_t __tmp_in_tmp447;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp447;
		
	}
	tmp447[i0] = (role == CLIENT) ? __tmp_in_tmp447 : 0;
	
}

auto tmp448 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp448 at (1737,1-1737,38) */
uint64_t __tmp_in_tmp448;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp448;
		
	}
	tmp448[i0] = (role == CLIENT) ? __tmp_in_tmp448 : 0;
	
}

auto tmp449 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp449 at (1740,1-1740,38) */
uint64_t __tmp_in_tmp449;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp449;
		
	}
	tmp449[i0] = (role == CLIENT) ? __tmp_in_tmp449 : 0;
	
}

auto tmp450 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp450 at (1743,1-1743,48) */
uint64_t __tmp_in_tmp450;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp450;
					
				}
				tmp450[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp450 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp451 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp451 at (1746,1-1746,38) */
uint64_t __tmp_in_tmp451;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp451;
		
	}
	tmp451[i0] = (role == CLIENT) ? __tmp_in_tmp451 : 0;
	
}

auto tmp452 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp452 at (1749,1-1749,38) */
uint64_t __tmp_in_tmp452;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp452;
		
	}
	tmp452[i0] = (role == CLIENT) ? __tmp_in_tmp452 : 0;
	
}

auto tmp453 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp453 at (1752,1-1752,38) */
uint64_t __tmp_in_tmp453;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp453;
		
	}
	tmp453[i0] = (role == CLIENT) ? __tmp_in_tmp453 : 0;
	
}

auto tmp454 = make_vector<uint64_t>( (int32_t)544 );
/* Variable to read the clear value corresponding to the input variable tmp454 at (1755,1-1755,38) */
uint64_t __tmp_in_tmp454;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)544; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp454;
		
	}
	tmp454[i0] = (role == CLIENT) ? __tmp_in_tmp454 : 0;
	
}

auto tmp455 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)544,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp455 at (1758,1-1758,49) */
uint64_t __tmp_in_tmp455;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)544; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp455;
					
				}
				tmp455[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp455 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp456 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp456 at (1761,1-1761,38) */
uint64_t __tmp_in_tmp456;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp456;
		
	}
	tmp456[i0] = (role == CLIENT) ? __tmp_in_tmp456 : 0;
	
}

auto tmp457 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp457 at (1764,1-1764,38) */
uint64_t __tmp_in_tmp457;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp457;
		
	}
	tmp457[i0] = (role == CLIENT) ? __tmp_in_tmp457 : 0;
	
}

auto tmp458 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp458 at (1767,1-1767,38) */
uint64_t __tmp_in_tmp458;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp458;
		
	}
	tmp458[i0] = (role == CLIENT) ? __tmp_in_tmp458 : 0;
	
}

auto tmp459 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp459 at (1770,1-1770,38) */
uint64_t __tmp_in_tmp459;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp459;
		
	}
	tmp459[i0] = (role == CLIENT) ? __tmp_in_tmp459 : 0;
	
}

auto tmp460 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp460 at (1773,1-1773,48) */
uint64_t __tmp_in_tmp460;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp460;
					
				}
				tmp460[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp460 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp461 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp461 at (1776,1-1776,38) */
uint64_t __tmp_in_tmp461;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp461;
		
	}
	tmp461[i0] = (role == CLIENT) ? __tmp_in_tmp461 : 0;
	
}

auto tmp462 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp462 at (1779,1-1779,38) */
uint64_t __tmp_in_tmp462;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp462;
		
	}
	tmp462[i0] = (role == CLIENT) ? __tmp_in_tmp462 : 0;
	
}

auto tmp463 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp463 at (1782,1-1782,38) */
uint64_t __tmp_in_tmp463;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp463;
		
	}
	tmp463[i0] = (role == CLIENT) ? __tmp_in_tmp463 : 0;
	
}

auto tmp464 = make_vector<uint64_t>( (int32_t)576 );
/* Variable to read the clear value corresponding to the input variable tmp464 at (1785,1-1785,38) */
uint64_t __tmp_in_tmp464;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)576; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp464;
		
	}
	tmp464[i0] = (role == CLIENT) ? __tmp_in_tmp464 : 0;
	
}

auto tmp465 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)576,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp465 at (1788,1-1788,49) */
uint64_t __tmp_in_tmp465;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)576; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp465;
					
				}
				tmp465[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp465 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp466 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp466 at (1791,1-1791,38) */
uint64_t __tmp_in_tmp466;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp466;
		
	}
	tmp466[i0] = (role == CLIENT) ? __tmp_in_tmp466 : 0;
	
}

auto tmp467 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp467 at (1794,1-1794,38) */
uint64_t __tmp_in_tmp467;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp467;
		
	}
	tmp467[i0] = (role == CLIENT) ? __tmp_in_tmp467 : 0;
	
}

auto tmp468 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp468 at (1797,1-1797,38) */
uint64_t __tmp_in_tmp468;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp468;
		
	}
	tmp468[i0] = (role == CLIENT) ? __tmp_in_tmp468 : 0;
	
}

auto tmp469 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp469 at (1800,1-1800,38) */
uint64_t __tmp_in_tmp469;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp469;
		
	}
	tmp469[i0] = (role == CLIENT) ? __tmp_in_tmp469 : 0;
	
}

auto tmp470 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp470 at (1803,1-1803,48) */
uint64_t __tmp_in_tmp470;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp470;
					
				}
				tmp470[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp470 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp471 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp471 at (1806,1-1806,38) */
uint64_t __tmp_in_tmp471;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp471;
		
	}
	tmp471[i0] = (role == CLIENT) ? __tmp_in_tmp471 : 0;
	
}

auto tmp472 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp472 at (1809,1-1809,38) */
uint64_t __tmp_in_tmp472;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp472;
		
	}
	tmp472[i0] = (role == CLIENT) ? __tmp_in_tmp472 : 0;
	
}

auto tmp473 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp473 at (1812,1-1812,38) */
uint64_t __tmp_in_tmp473;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp473;
		
	}
	tmp473[i0] = (role == CLIENT) ? __tmp_in_tmp473 : 0;
	
}

auto tmp474 = make_vector<uint64_t>( (int32_t)608 );
/* Variable to read the clear value corresponding to the input variable tmp474 at (1815,1-1815,38) */
uint64_t __tmp_in_tmp474;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)608; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp474;
		
	}
	tmp474[i0] = (role == CLIENT) ? __tmp_in_tmp474 : 0;
	
}

auto tmp475 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)608,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp475 at (1818,1-1818,49) */
uint64_t __tmp_in_tmp475;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)608; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp475;
					
				}
				tmp475[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp475 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp476 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp476 at (1821,1-1821,38) */
uint64_t __tmp_in_tmp476;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp476;
		
	}
	tmp476[i0] = (role == CLIENT) ? __tmp_in_tmp476 : 0;
	
}

auto tmp477 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp477 at (1824,1-1824,38) */
uint64_t __tmp_in_tmp477;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp477;
		
	}
	tmp477[i0] = (role == CLIENT) ? __tmp_in_tmp477 : 0;
	
}

auto tmp478 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp478 at (1827,1-1827,38) */
uint64_t __tmp_in_tmp478;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp478;
		
	}
	tmp478[i0] = (role == CLIENT) ? __tmp_in_tmp478 : 0;
	
}

auto tmp479 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp479 at (1830,1-1830,38) */
uint64_t __tmp_in_tmp479;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp479;
		
	}
	tmp479[i0] = (role == CLIENT) ? __tmp_in_tmp479 : 0;
	
}

auto tmp480 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp480 at (1833,1-1833,48) */
uint64_t __tmp_in_tmp480;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp480;
					
				}
				tmp480[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp480 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp481 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp481 at (1836,1-1836,38) */
uint64_t __tmp_in_tmp481;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp481;
		
	}
	tmp481[i0] = (role == CLIENT) ? __tmp_in_tmp481 : 0;
	
}

auto tmp482 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp482 at (1839,1-1839,38) */
uint64_t __tmp_in_tmp482;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp482;
		
	}
	tmp482[i0] = (role == CLIENT) ? __tmp_in_tmp482 : 0;
	
}

auto tmp483 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp483 at (1842,1-1842,38) */
uint64_t __tmp_in_tmp483;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp483;
		
	}
	tmp483[i0] = (role == CLIENT) ? __tmp_in_tmp483 : 0;
	
}

auto tmp484 = make_vector<uint64_t>( (int32_t)640 );
/* Variable to read the clear value corresponding to the input variable tmp484 at (1845,1-1845,38) */
uint64_t __tmp_in_tmp484;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)640; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp484;
		
	}
	tmp484[i0] = (role == CLIENT) ? __tmp_in_tmp484 : 0;
	
}

auto tmp485 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)640,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp485 at (1848,1-1848,49) */
uint64_t __tmp_in_tmp485;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)640; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp485;
					
				}
				tmp485[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp485 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp486 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp486 at (1851,1-1851,38) */
uint64_t __tmp_in_tmp486;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp486;
		
	}
	tmp486[i0] = (role == CLIENT) ? __tmp_in_tmp486 : 0;
	
}

auto tmp487 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp487 at (1854,1-1854,38) */
uint64_t __tmp_in_tmp487;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp487;
		
	}
	tmp487[i0] = (role == CLIENT) ? __tmp_in_tmp487 : 0;
	
}

auto tmp488 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp488 at (1857,1-1857,38) */
uint64_t __tmp_in_tmp488;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp488;
		
	}
	tmp488[i0] = (role == CLIENT) ? __tmp_in_tmp488 : 0;
	
}

auto tmp489 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp489 at (1860,1-1860,38) */
uint64_t __tmp_in_tmp489;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp489;
		
	}
	tmp489[i0] = (role == CLIENT) ? __tmp_in_tmp489 : 0;
	
}

auto tmp490 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp490 at (1863,1-1863,48) */
uint64_t __tmp_in_tmp490;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp490;
					
				}
				tmp490[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp490 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp491 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp491 at (1866,1-1866,38) */
uint64_t __tmp_in_tmp491;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp491;
		
	}
	tmp491[i0] = (role == CLIENT) ? __tmp_in_tmp491 : 0;
	
}

auto tmp492 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp492 at (1869,1-1869,38) */
uint64_t __tmp_in_tmp492;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp492;
		
	}
	tmp492[i0] = (role == CLIENT) ? __tmp_in_tmp492 : 0;
	
}

auto tmp493 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp493 at (1872,1-1872,38) */
uint64_t __tmp_in_tmp493;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp493;
		
	}
	tmp493[i0] = (role == CLIENT) ? __tmp_in_tmp493 : 0;
	
}

auto tmp494 = make_vector<uint64_t>( (int32_t)672 );
/* Variable to read the clear value corresponding to the input variable tmp494 at (1875,1-1875,38) */
uint64_t __tmp_in_tmp494;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)672; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp494;
		
	}
	tmp494[i0] = (role == CLIENT) ? __tmp_in_tmp494 : 0;
	
}

auto tmp495 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)672,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp495 at (1878,1-1878,49) */
uint64_t __tmp_in_tmp495;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)672; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp495;
					
				}
				tmp495[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp495 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp496 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp496 at (1881,1-1881,38) */
uint64_t __tmp_in_tmp496;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp496;
		
	}
	tmp496[i0] = (role == CLIENT) ? __tmp_in_tmp496 : 0;
	
}

auto tmp497 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp497 at (1884,1-1884,38) */
uint64_t __tmp_in_tmp497;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp497;
		
	}
	tmp497[i0] = (role == CLIENT) ? __tmp_in_tmp497 : 0;
	
}

auto tmp498 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp498 at (1887,1-1887,38) */
uint64_t __tmp_in_tmp498;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp498;
		
	}
	tmp498[i0] = (role == CLIENT) ? __tmp_in_tmp498 : 0;
	
}

auto tmp499 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp499 at (1890,1-1890,38) */
uint64_t __tmp_in_tmp499;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp499;
		
	}
	tmp499[i0] = (role == CLIENT) ? __tmp_in_tmp499 : 0;
	
}

auto tmp500 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp500 at (1893,1-1893,48) */
uint64_t __tmp_in_tmp500;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp500;
					
				}
				tmp500[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp500 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp501 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp501 at (1896,1-1896,38) */
uint64_t __tmp_in_tmp501;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp501;
		
	}
	tmp501[i0] = (role == CLIENT) ? __tmp_in_tmp501 : 0;
	
}

auto tmp502 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp502 at (1899,1-1899,38) */
uint64_t __tmp_in_tmp502;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp502;
		
	}
	tmp502[i0] = (role == CLIENT) ? __tmp_in_tmp502 : 0;
	
}

auto tmp503 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp503 at (1902,1-1902,38) */
uint64_t __tmp_in_tmp503;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp503;
		
	}
	tmp503[i0] = (role == CLIENT) ? __tmp_in_tmp503 : 0;
	
}

auto tmp504 = make_vector<uint64_t>( (int32_t)704 );
/* Variable to read the clear value corresponding to the input variable tmp504 at (1905,1-1905,38) */
uint64_t __tmp_in_tmp504;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)704; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp504;
		
	}
	tmp504[i0] = (role == CLIENT) ? __tmp_in_tmp504 : 0;
	
}

auto tmp505 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)704,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp505 at (1908,1-1908,49) */
uint64_t __tmp_in_tmp505;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)704; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp505;
					
				}
				tmp505[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp505 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp506 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp506 at (1911,1-1911,38) */
uint64_t __tmp_in_tmp506;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp506;
		
	}
	tmp506[i0] = (role == CLIENT) ? __tmp_in_tmp506 : 0;
	
}

auto tmp507 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp507 at (1914,1-1914,38) */
uint64_t __tmp_in_tmp507;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp507;
		
	}
	tmp507[i0] = (role == CLIENT) ? __tmp_in_tmp507 : 0;
	
}

auto tmp508 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp508 at (1917,1-1917,38) */
uint64_t __tmp_in_tmp508;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp508;
		
	}
	tmp508[i0] = (role == CLIENT) ? __tmp_in_tmp508 : 0;
	
}

auto tmp509 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp509 at (1920,1-1920,38) */
uint64_t __tmp_in_tmp509;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp509;
		
	}
	tmp509[i0] = (role == CLIENT) ? __tmp_in_tmp509 : 0;
	
}

auto tmp510 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp510 at (1923,1-1923,48) */
uint64_t __tmp_in_tmp510;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp510;
					
				}
				tmp510[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp510 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp511 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp511 at (1926,1-1926,38) */
uint64_t __tmp_in_tmp511;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp511;
		
	}
	tmp511[i0] = (role == CLIENT) ? __tmp_in_tmp511 : 0;
	
}

auto tmp512 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp512 at (1929,1-1929,38) */
uint64_t __tmp_in_tmp512;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp512;
		
	}
	tmp512[i0] = (role == CLIENT) ? __tmp_in_tmp512 : 0;
	
}

auto tmp513 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp513 at (1932,1-1932,38) */
uint64_t __tmp_in_tmp513;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp513;
		
	}
	tmp513[i0] = (role == CLIENT) ? __tmp_in_tmp513 : 0;
	
}

auto tmp514 = make_vector<uint64_t>( (int32_t)736 );
/* Variable to read the clear value corresponding to the input variable tmp514 at (1935,1-1935,38) */
uint64_t __tmp_in_tmp514;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)736; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp514;
		
	}
	tmp514[i0] = (role == CLIENT) ? __tmp_in_tmp514 : 0;
	
}

auto tmp515 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)736,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp515 at (1938,1-1938,49) */
uint64_t __tmp_in_tmp515;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)736; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp515;
					
				}
				tmp515[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp515 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp516 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp516 at (1941,1-1941,38) */
uint64_t __tmp_in_tmp516;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp516;
		
	}
	tmp516[i0] = (role == CLIENT) ? __tmp_in_tmp516 : 0;
	
}

auto tmp517 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp517 at (1944,1-1944,38) */
uint64_t __tmp_in_tmp517;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp517;
		
	}
	tmp517[i0] = (role == CLIENT) ? __tmp_in_tmp517 : 0;
	
}

auto tmp518 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp518 at (1947,1-1947,38) */
uint64_t __tmp_in_tmp518;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp518;
		
	}
	tmp518[i0] = (role == CLIENT) ? __tmp_in_tmp518 : 0;
	
}

auto tmp519 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp519 at (1950,1-1950,38) */
uint64_t __tmp_in_tmp519;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp519;
		
	}
	tmp519[i0] = (role == CLIENT) ? __tmp_in_tmp519 : 0;
	
}

auto tmp520 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp520 at (1953,1-1953,48) */
uint64_t __tmp_in_tmp520;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp520;
					
				}
				tmp520[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp520 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp521 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp521 at (1956,1-1956,38) */
uint64_t __tmp_in_tmp521;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp521;
		
	}
	tmp521[i0] = (role == CLIENT) ? __tmp_in_tmp521 : 0;
	
}

auto tmp522 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp522 at (1959,1-1959,38) */
uint64_t __tmp_in_tmp522;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp522;
		
	}
	tmp522[i0] = (role == CLIENT) ? __tmp_in_tmp522 : 0;
	
}

auto tmp523 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp523 at (1962,1-1962,38) */
uint64_t __tmp_in_tmp523;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp523;
		
	}
	tmp523[i0] = (role == CLIENT) ? __tmp_in_tmp523 : 0;
	
}

auto tmp524 = make_vector<uint64_t>( (int32_t)768 );
/* Variable to read the clear value corresponding to the input variable tmp524 at (1965,1-1965,38) */
uint64_t __tmp_in_tmp524;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)768; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp524;
		
	}
	tmp524[i0] = (role == CLIENT) ? __tmp_in_tmp524 : 0;
	
}

auto tmp525 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)768,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp525 at (1968,1-1968,49) */
uint64_t __tmp_in_tmp525;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)768; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp525;
					
				}
				tmp525[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp525 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp526 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp526 at (1971,1-1971,38) */
uint64_t __tmp_in_tmp526;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp526;
		
	}
	tmp526[i0] = (role == CLIENT) ? __tmp_in_tmp526 : 0;
	
}

auto tmp527 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp527 at (1974,1-1974,38) */
uint64_t __tmp_in_tmp527;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp527;
		
	}
	tmp527[i0] = (role == CLIENT) ? __tmp_in_tmp527 : 0;
	
}

auto tmp528 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp528 at (1977,1-1977,38) */
uint64_t __tmp_in_tmp528;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp528;
		
	}
	tmp528[i0] = (role == CLIENT) ? __tmp_in_tmp528 : 0;
	
}

auto tmp529 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp529 at (1980,1-1980,38) */
uint64_t __tmp_in_tmp529;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp529;
		
	}
	tmp529[i0] = (role == CLIENT) ? __tmp_in_tmp529 : 0;
	
}

auto tmp530 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp530 at (1983,1-1983,48) */
uint64_t __tmp_in_tmp530;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp530;
					
				}
				tmp530[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp530 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp531 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp531 at (1986,1-1986,38) */
uint64_t __tmp_in_tmp531;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp531;
		
	}
	tmp531[i0] = (role == CLIENT) ? __tmp_in_tmp531 : 0;
	
}

auto tmp532 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp532 at (1989,1-1989,38) */
uint64_t __tmp_in_tmp532;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp532;
		
	}
	tmp532[i0] = (role == CLIENT) ? __tmp_in_tmp532 : 0;
	
}

auto tmp533 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp533 at (1992,1-1992,38) */
uint64_t __tmp_in_tmp533;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp533;
		
	}
	tmp533[i0] = (role == CLIENT) ? __tmp_in_tmp533 : 0;
	
}

auto tmp534 = make_vector<uint64_t>( (int32_t)800 );
/* Variable to read the clear value corresponding to the input variable tmp534 at (1995,1-1995,38) */
uint64_t __tmp_in_tmp534;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp534;
		
	}
	tmp534[i0] = (role == CLIENT) ? __tmp_in_tmp534 : 0;
	
}

auto tmp535 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)800,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp535 at (1998,1-1998,49) */
uint64_t __tmp_in_tmp535;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)800; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp535;
					
				}
				tmp535[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp535 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp536 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp536 at (2001,1-2001,38) */
uint64_t __tmp_in_tmp536;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp536;
		
	}
	tmp536[i0] = (role == CLIENT) ? __tmp_in_tmp536 : 0;
	
}

auto tmp537 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp537 at (2004,1-2004,38) */
uint64_t __tmp_in_tmp537;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp537;
		
	}
	tmp537[i0] = (role == CLIENT) ? __tmp_in_tmp537 : 0;
	
}

auto tmp538 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp538 at (2007,1-2007,38) */
uint64_t __tmp_in_tmp538;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp538;
		
	}
	tmp538[i0] = (role == CLIENT) ? __tmp_in_tmp538 : 0;
	
}

auto tmp539 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp539 at (2010,1-2010,38) */
uint64_t __tmp_in_tmp539;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp539;
		
	}
	tmp539[i0] = (role == CLIENT) ? __tmp_in_tmp539 : 0;
	
}

auto tmp540 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp540 at (2013,1-2013,48) */
uint64_t __tmp_in_tmp540;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp540;
					
				}
				tmp540[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp540 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp541 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp541 at (2016,1-2016,38) */
uint64_t __tmp_in_tmp541;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp541;
		
	}
	tmp541[i0] = (role == CLIENT) ? __tmp_in_tmp541 : 0;
	
}

auto tmp542 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp542 at (2019,1-2019,38) */
uint64_t __tmp_in_tmp542;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp542;
		
	}
	tmp542[i0] = (role == CLIENT) ? __tmp_in_tmp542 : 0;
	
}

auto tmp543 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp543 at (2022,1-2022,38) */
uint64_t __tmp_in_tmp543;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp543;
		
	}
	tmp543[i0] = (role == CLIENT) ? __tmp_in_tmp543 : 0;
	
}

auto tmp544 = make_vector<uint64_t>( (int32_t)832 );
/* Variable to read the clear value corresponding to the input variable tmp544 at (2025,1-2025,38) */
uint64_t __tmp_in_tmp544;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)832; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp544;
		
	}
	tmp544[i0] = (role == CLIENT) ? __tmp_in_tmp544 : 0;
	
}

auto tmp545 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)832,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp545 at (2028,1-2028,49) */
uint64_t __tmp_in_tmp545;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)832; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp545;
					
				}
				tmp545[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp545 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp546 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp546 at (2031,1-2031,38) */
uint64_t __tmp_in_tmp546;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp546;
		
	}
	tmp546[i0] = (role == CLIENT) ? __tmp_in_tmp546 : 0;
	
}

auto tmp547 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp547 at (2034,1-2034,38) */
uint64_t __tmp_in_tmp547;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp547;
		
	}
	tmp547[i0] = (role == CLIENT) ? __tmp_in_tmp547 : 0;
	
}

auto tmp548 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp548 at (2037,1-2037,38) */
uint64_t __tmp_in_tmp548;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp548;
		
	}
	tmp548[i0] = (role == CLIENT) ? __tmp_in_tmp548 : 0;
	
}

auto tmp549 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp549 at (2040,1-2040,38) */
uint64_t __tmp_in_tmp549;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp549;
		
	}
	tmp549[i0] = (role == CLIENT) ? __tmp_in_tmp549 : 0;
	
}

auto tmp550 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp550 at (2043,1-2043,48) */
uint64_t __tmp_in_tmp550;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp550;
					
				}
				tmp550[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp550 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp551 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp551 at (2046,1-2046,38) */
uint64_t __tmp_in_tmp551;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp551;
		
	}
	tmp551[i0] = (role == CLIENT) ? __tmp_in_tmp551 : 0;
	
}

auto tmp552 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp552 at (2049,1-2049,38) */
uint64_t __tmp_in_tmp552;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp552;
		
	}
	tmp552[i0] = (role == CLIENT) ? __tmp_in_tmp552 : 0;
	
}

auto tmp553 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp553 at (2052,1-2052,38) */
uint64_t __tmp_in_tmp553;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp553;
		
	}
	tmp553[i0] = (role == CLIENT) ? __tmp_in_tmp553 : 0;
	
}

auto tmp554 = make_vector<uint64_t>( (int32_t)864 );
/* Variable to read the clear value corresponding to the input variable tmp554 at (2055,1-2055,38) */
uint64_t __tmp_in_tmp554;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)864; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp554;
		
	}
	tmp554[i0] = (role == CLIENT) ? __tmp_in_tmp554 : 0;
	
}

auto tmp555 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)864,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp555 at (2058,1-2058,49) */
uint64_t __tmp_in_tmp555;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)864; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp555;
					
				}
				tmp555[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp555 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp556 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp556 at (2061,1-2061,38) */
uint64_t __tmp_in_tmp556;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp556;
		
	}
	tmp556[i0] = (role == CLIENT) ? __tmp_in_tmp556 : 0;
	
}

auto tmp557 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp557 at (2064,1-2064,38) */
uint64_t __tmp_in_tmp557;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp557;
		
	}
	tmp557[i0] = (role == CLIENT) ? __tmp_in_tmp557 : 0;
	
}

auto tmp558 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp558 at (2067,1-2067,38) */
uint64_t __tmp_in_tmp558;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp558;
		
	}
	tmp558[i0] = (role == CLIENT) ? __tmp_in_tmp558 : 0;
	
}

auto tmp559 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp559 at (2070,1-2070,38) */
uint64_t __tmp_in_tmp559;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp559;
		
	}
	tmp559[i0] = (role == CLIENT) ? __tmp_in_tmp559 : 0;
	
}

auto tmp560 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp560 at (2073,1-2073,48) */
uint64_t __tmp_in_tmp560;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp560;
					
				}
				tmp560[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp560 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp561 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp561 at (2076,1-2076,38) */
uint64_t __tmp_in_tmp561;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp561;
		
	}
	tmp561[i0] = (role == CLIENT) ? __tmp_in_tmp561 : 0;
	
}

auto tmp562 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp562 at (2079,1-2079,38) */
uint64_t __tmp_in_tmp562;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp562;
		
	}
	tmp562[i0] = (role == CLIENT) ? __tmp_in_tmp562 : 0;
	
}

auto tmp563 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp563 at (2082,1-2082,38) */
uint64_t __tmp_in_tmp563;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp563;
		
	}
	tmp563[i0] = (role == CLIENT) ? __tmp_in_tmp563 : 0;
	
}

auto tmp564 = make_vector<uint64_t>( (int32_t)896 );
/* Variable to read the clear value corresponding to the input variable tmp564 at (2085,1-2085,38) */
uint64_t __tmp_in_tmp564;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)896; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp564;
		
	}
	tmp564[i0] = (role == CLIENT) ? __tmp_in_tmp564 : 0;
	
}

auto tmp565 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)896,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp565 at (2088,1-2088,49) */
uint64_t __tmp_in_tmp565;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)896; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp565;
					
				}
				tmp565[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp565 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp566 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp566 at (2091,1-2091,38) */
uint64_t __tmp_in_tmp566;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp566;
		
	}
	tmp566[i0] = (role == CLIENT) ? __tmp_in_tmp566 : 0;
	
}

auto tmp567 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp567 at (2094,1-2094,38) */
uint64_t __tmp_in_tmp567;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp567;
		
	}
	tmp567[i0] = (role == CLIENT) ? __tmp_in_tmp567 : 0;
	
}

auto tmp568 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp568 at (2097,1-2097,38) */
uint64_t __tmp_in_tmp568;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp568;
		
	}
	tmp568[i0] = (role == CLIENT) ? __tmp_in_tmp568 : 0;
	
}

auto tmp569 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp569 at (2100,1-2100,38) */
uint64_t __tmp_in_tmp569;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp569;
		
	}
	tmp569[i0] = (role == CLIENT) ? __tmp_in_tmp569 : 0;
	
}

auto tmp570 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp570 at (2103,1-2103,48) */
uint64_t __tmp_in_tmp570;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp570;
					
				}
				tmp570[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp570 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp571 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp571 at (2106,1-2106,38) */
uint64_t __tmp_in_tmp571;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp571;
		
	}
	tmp571[i0] = (role == CLIENT) ? __tmp_in_tmp571 : 0;
	
}

auto tmp572 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp572 at (2109,1-2109,38) */
uint64_t __tmp_in_tmp572;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp572;
		
	}
	tmp572[i0] = (role == CLIENT) ? __tmp_in_tmp572 : 0;
	
}

auto tmp573 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp573 at (2112,1-2112,38) */
uint64_t __tmp_in_tmp573;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp573;
		
	}
	tmp573[i0] = (role == CLIENT) ? __tmp_in_tmp573 : 0;
	
}

auto tmp574 = make_vector<uint64_t>( (int32_t)928 );
/* Variable to read the clear value corresponding to the input variable tmp574 at (2115,1-2115,38) */
uint64_t __tmp_in_tmp574;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)928; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp574;
		
	}
	tmp574[i0] = (role == CLIENT) ? __tmp_in_tmp574 : 0;
	
}

auto tmp575 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)928,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp575 at (2118,1-2118,49) */
uint64_t __tmp_in_tmp575;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)928; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp575;
					
				}
				tmp575[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp575 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp576 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp576 at (2121,1-2121,38) */
uint64_t __tmp_in_tmp576;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp576;
		
	}
	tmp576[i0] = (role == CLIENT) ? __tmp_in_tmp576 : 0;
	
}

auto tmp577 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp577 at (2124,1-2124,38) */
uint64_t __tmp_in_tmp577;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp577;
		
	}
	tmp577[i0] = (role == CLIENT) ? __tmp_in_tmp577 : 0;
	
}

auto tmp578 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp578 at (2127,1-2127,38) */
uint64_t __tmp_in_tmp578;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp578;
		
	}
	tmp578[i0] = (role == CLIENT) ? __tmp_in_tmp578 : 0;
	
}

auto tmp579 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp579 at (2130,1-2130,38) */
uint64_t __tmp_in_tmp579;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp579;
		
	}
	tmp579[i0] = (role == CLIENT) ? __tmp_in_tmp579 : 0;
	
}

auto tmp580 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp580 at (2133,1-2133,48) */
uint64_t __tmp_in_tmp580;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp580;
					
				}
				tmp580[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp580 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp581 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp581 at (2136,1-2136,38) */
uint64_t __tmp_in_tmp581;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp581;
		
	}
	tmp581[i0] = (role == CLIENT) ? __tmp_in_tmp581 : 0;
	
}

auto tmp582 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp582 at (2139,1-2139,38) */
uint64_t __tmp_in_tmp582;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp582;
		
	}
	tmp582[i0] = (role == CLIENT) ? __tmp_in_tmp582 : 0;
	
}

auto tmp583 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp583 at (2142,1-2142,38) */
uint64_t __tmp_in_tmp583;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp583;
		
	}
	tmp583[i0] = (role == CLIENT) ? __tmp_in_tmp583 : 0;
	
}

auto tmp584 = make_vector<uint64_t>( (int32_t)960 );
/* Variable to read the clear value corresponding to the input variable tmp584 at (2145,1-2145,38) */
uint64_t __tmp_in_tmp584;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)960; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp584;
		
	}
	tmp584[i0] = (role == CLIENT) ? __tmp_in_tmp584 : 0;
	
}

auto tmp585 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)960,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp585 at (2148,1-2148,49) */
uint64_t __tmp_in_tmp585;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)960; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp585;
					
				}
				tmp585[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp585 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp586 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp586 at (2151,1-2151,38) */
uint64_t __tmp_in_tmp586;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp586;
		
	}
	tmp586[i0] = (role == CLIENT) ? __tmp_in_tmp586 : 0;
	
}

auto tmp587 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp587 at (2154,1-2154,38) */
uint64_t __tmp_in_tmp587;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp587;
		
	}
	tmp587[i0] = (role == CLIENT) ? __tmp_in_tmp587 : 0;
	
}

auto tmp588 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp588 at (2157,1-2157,38) */
uint64_t __tmp_in_tmp588;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp588;
		
	}
	tmp588[i0] = (role == CLIENT) ? __tmp_in_tmp588 : 0;
	
}

auto tmp589 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp589 at (2160,1-2160,38) */
uint64_t __tmp_in_tmp589;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp589;
		
	}
	tmp589[i0] = (role == CLIENT) ? __tmp_in_tmp589 : 0;
	
}

auto tmp590 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp590 at (2163,1-2163,48) */
uint64_t __tmp_in_tmp590;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp590;
					
				}
				tmp590[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp590 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp591 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp591 at (2166,1-2166,38) */
uint64_t __tmp_in_tmp591;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp591;
		
	}
	tmp591[i0] = (role == CLIENT) ? __tmp_in_tmp591 : 0;
	
}

auto tmp592 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp592 at (2169,1-2169,38) */
uint64_t __tmp_in_tmp592;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp592;
		
	}
	tmp592[i0] = (role == CLIENT) ? __tmp_in_tmp592 : 0;
	
}

auto tmp593 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp593 at (2172,1-2172,38) */
uint64_t __tmp_in_tmp593;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp593;
		
	}
	tmp593[i0] = (role == CLIENT) ? __tmp_in_tmp593 : 0;
	
}

auto tmp594 = make_vector<uint64_t>( (int32_t)992 );
/* Variable to read the clear value corresponding to the input variable tmp594 at (2175,1-2175,38) */
uint64_t __tmp_in_tmp594;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)992; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp594;
		
	}
	tmp594[i0] = (role == CLIENT) ? __tmp_in_tmp594 : 0;
	
}

auto tmp595 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)992,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp595 at (2178,1-2178,49) */
uint64_t __tmp_in_tmp595;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)992; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp595;
					
				}
				tmp595[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp595 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp596 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp596 at (2181,1-2181,38) */
uint64_t __tmp_in_tmp596;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp596;
		
	}
	tmp596[i0] = (role == CLIENT) ? __tmp_in_tmp596 : 0;
	
}

auto tmp597 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp597 at (2184,1-2184,38) */
uint64_t __tmp_in_tmp597;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp597;
		
	}
	tmp597[i0] = (role == CLIENT) ? __tmp_in_tmp597 : 0;
	
}

auto tmp598 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp598 at (2187,1-2187,38) */
uint64_t __tmp_in_tmp598;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp598;
		
	}
	tmp598[i0] = (role == CLIENT) ? __tmp_in_tmp598 : 0;
	
}

auto tmp599 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp599 at (2190,1-2190,38) */
uint64_t __tmp_in_tmp599;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp599;
		
	}
	tmp599[i0] = (role == CLIENT) ? __tmp_in_tmp599 : 0;
	
}

auto tmp600 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp600 at (2193,1-2193,48) */
uint64_t __tmp_in_tmp600;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp600;
					
				}
				tmp600[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp600 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp601 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp601 at (2196,1-2196,39) */
uint64_t __tmp_in_tmp601;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp601;
		
	}
	tmp601[i0] = (role == CLIENT) ? __tmp_in_tmp601 : 0;
	
}

auto tmp602 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp602 at (2199,1-2199,39) */
uint64_t __tmp_in_tmp602;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp602;
		
	}
	tmp602[i0] = (role == CLIENT) ? __tmp_in_tmp602 : 0;
	
}

auto tmp603 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp603 at (2202,1-2202,39) */
uint64_t __tmp_in_tmp603;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp603;
		
	}
	tmp603[i0] = (role == CLIENT) ? __tmp_in_tmp603 : 0;
	
}

auto tmp604 = make_vector<uint64_t>( (int32_t)1024 );
/* Variable to read the clear value corresponding to the input variable tmp604 at (2205,1-2205,39) */
uint64_t __tmp_in_tmp604;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1024; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp604;
		
	}
	tmp604[i0] = (role == CLIENT) ? __tmp_in_tmp604 : 0;
	
}

auto tmp605 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)1000 );
/* Variable to read the clear value corresponding to the input variable tmp605 at (2208,1-2208,51) */
uint64_t __tmp_in_tmp605;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1024; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1000; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp605;
					
				}
				tmp605[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp605 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp606 = make_vector<uint64_t>( (int32_t)1000 );
/* Variable to read the clear value corresponding to the input variable tmp606 at (2211,1-2211,39) */
uint64_t __tmp_in_tmp606;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1000; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp606;
		
	}
	tmp606[i0] = (role == CLIENT) ? __tmp_in_tmp606 : 0;
	
}

leave_time();
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("Starting main protocol");
start_m();
touch_time();

auto tmp607 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3,  (int32_t)7,  (int32_t)7,  (int32_t)64,  (int32_t)2,  (int32_t)3,  (int32_t)2,  (int32_t)3,  (int32_t)2,  (int32_t)2, tmp0, tmp1,  (int32_t)12, tmp607 );
ClearMemSecret4( (int32_t)7,  (int32_t)7,  (int32_t)3,  (int32_t)64, tmp1 );
ClearMemSecret4( (int32_t)1,  (int32_t)224,  (int32_t)224,  (int32_t)3, tmp0 );

auto tmp610 = make_vector<uint64_t>( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp607, tmp2, tmp3,  (int32_t)12, tmp610 );
ClearMemSecret1( (int32_t)64, tmp3 );
ClearMemSecret4( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp607 );
ClearMemSecret1( (int32_t)64, tmp2 );

auto tmp614 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)1,  (int32_t)0,  (int32_t)1,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp610, tmp614 );
ClearMemSecret4( (int32_t)1,  (int32_t)112,  (int32_t)112,  (int32_t)64, tmp610 );

auto tmp616 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp614, tmp616 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp614 );

auto tmp618 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp616, tmp6, tmp7,  (int32_t)12, tmp618 );
ClearMemSecret1( (int32_t)64, tmp6 );
ClearMemSecret1( (int32_t)64, tmp7 );

auto tmp621 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp618, tmp621 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp618 );

auto tmp623 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp621, tmp10,  (int32_t)12, tmp623 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp621 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)128, tmp10 );

auto tmp626 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp623, tmp11, tmp12,  (int32_t)12, tmp626 );
ClearMemSecret1( (int32_t)128, tmp12 );
ClearMemSecret1( (int32_t)128, tmp11 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp623 );

auto tmp630 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp626, tmp630 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp626 );

auto tmp632 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp630, tmp15,  (int32_t)12, tmp632 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp15 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp630 );

auto tmp635 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp616,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp632,  (int32_t)3, tmp635 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp632 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp616 );

auto tmp639 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp635, tmp16, tmp17,  (int32_t)12, tmp639 );
ClearMemSecret1( (int32_t)96, tmp16 );
ClearMemSecret1( (int32_t)96, tmp17 );

auto tmp642 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp639, tmp642 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp639 );

auto tmp644 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp642, tmp20,  (int32_t)12, tmp644 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)96,  (int32_t)128, tmp20 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp642 );

auto tmp647 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp644, tmp21, tmp22,  (int32_t)12, tmp647 );
ClearMemSecret1( (int32_t)128, tmp22 );
ClearMemSecret1( (int32_t)128, tmp21 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp644 );

auto tmp651 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp647, tmp651 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp647 );

auto tmp653 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp651, tmp25,  (int32_t)12, tmp653 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp25 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp651 );

auto tmp656 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp635,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp653,  (int32_t)3, tmp656 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp653 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)96, tmp635 );
ClearMemPublic( (int32_t)3 );

auto tmp660 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp656, tmp26, tmp27,  (int32_t)12, tmp660 );
ClearMemSecret1( (int32_t)128, tmp26 );
ClearMemSecret1( (int32_t)128, tmp27 );

auto tmp663 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp660, tmp663 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp660 );

auto tmp665 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp663, tmp30,  (int32_t)12, tmp665 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)128, tmp30 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp663 );

auto tmp668 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp665, tmp31, tmp32,  (int32_t)12, tmp668 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp665 );
ClearMemSecret1( (int32_t)128, tmp31 );
ClearMemSecret1( (int32_t)128, tmp32 );

auto tmp672 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp668, tmp672 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp668 );

auto tmp674 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp672, tmp35,  (int32_t)12, tmp674 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp35 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp672 );

auto tmp677 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp656,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp674,  (int32_t)3, tmp677 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp656 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp674 );
ClearMemPublic( (int32_t)3 );

auto tmp681 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp677, tmp36, tmp37,  (int32_t)12, tmp681 );
ClearMemSecret1( (int32_t)160, tmp37 );
ClearMemSecret1( (int32_t)160, tmp36 );

auto tmp684 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp681, tmp684 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp681 );

auto tmp686 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp684, tmp40,  (int32_t)12, tmp686 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)160,  (int32_t)128, tmp40 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp684 );

auto tmp689 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp686, tmp41, tmp42,  (int32_t)12, tmp689 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp686 );
ClearMemSecret1( (int32_t)128, tmp42 );
ClearMemSecret1( (int32_t)128, tmp41 );

auto tmp693 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp689, tmp693 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp689 );

auto tmp695 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp693, tmp45,  (int32_t)12, tmp695 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp45 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp693 );

auto tmp698 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp677,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp695,  (int32_t)3, tmp698 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)160, tmp677 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp695 );

auto tmp702 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp698, tmp46, tmp47,  (int32_t)12, tmp702 );
ClearMemSecret1( (int32_t)192, tmp47 );
ClearMemSecret1( (int32_t)192, tmp46 );

auto tmp705 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp702, tmp705 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp702 );

auto tmp707 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp705, tmp50,  (int32_t)12, tmp707 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp705 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)128, tmp50 );

auto tmp710 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp707, tmp51, tmp52,  (int32_t)12, tmp710 );
ClearMemSecret1( (int32_t)128, tmp52 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp707 );
ClearMemSecret1( (int32_t)128, tmp51 );

auto tmp714 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp710, tmp714 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp710 );

auto tmp716 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp714, tmp55,  (int32_t)12, tmp716 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp55 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp714 );

auto tmp719 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp698,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp716,  (int32_t)3, tmp719 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp716 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)192, tmp698 );

auto tmp723 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp719, tmp56, tmp57,  (int32_t)12, tmp723 );
ClearMemSecret1( (int32_t)224, tmp56 );
ClearMemSecret1( (int32_t)224, tmp57 );

auto tmp726 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp723, tmp726 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp723 );

auto tmp728 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp726, tmp60,  (int32_t)12, tmp728 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)224,  (int32_t)128, tmp60 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp726 );

auto tmp731 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp728, tmp61, tmp62,  (int32_t)12, tmp731 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp728 );
ClearMemSecret1( (int32_t)128, tmp61 );
ClearMemSecret1( (int32_t)128, tmp62 );

auto tmp735 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp731, tmp735 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp731 );

auto tmp737 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp735, tmp65,  (int32_t)12, tmp737 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp735 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp65 );

auto tmp740 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp719,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp737,  (int32_t)3, tmp740 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)224, tmp719 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)32, tmp737 );

auto tmp744 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp740, tmp66, tmp67,  (int32_t)12, tmp744 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp740 );
ClearMemSecret1( (int32_t)256, tmp66 );
ClearMemSecret1( (int32_t)256, tmp67 );

auto tmp748 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp744, tmp748 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp744 );

auto tmp750 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp748, tmp70,  (int32_t)12, tmp750 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128, tmp70 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)256, tmp748 );

auto tmp753 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
AvgPool( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp750, tmp753 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp750 );

auto tmp755 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp753, tmp71, tmp72,  (int32_t)12, tmp755 );
ClearMemSecret1( (int32_t)128, tmp72 );
ClearMemSecret1( (int32_t)128, tmp71 );

auto tmp758 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp755, tmp758 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp755 );

auto tmp760 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp758, tmp75,  (int32_t)12, tmp760 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)128, tmp75 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp758 );

auto tmp763 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp760, tmp76, tmp77,  (int32_t)12, tmp763 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp760 );
ClearMemSecret1( (int32_t)128, tmp76 );
ClearMemSecret1( (int32_t)128, tmp77 );

auto tmp767 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp763, tmp767 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp763 );

auto tmp769 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp767, tmp80,  (int32_t)12, tmp769 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp767 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp80 );

auto tmp772 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp753,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp769,  (int32_t)3, tmp772 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp753 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp769 );

auto tmp776 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp772, tmp81, tmp82,  (int32_t)12, tmp776 );
ClearMemSecret1( (int32_t)160, tmp82 );
ClearMemSecret1( (int32_t)160, tmp81 );

auto tmp779 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp776, tmp779 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp776 );

auto tmp781 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp779, tmp85,  (int32_t)12, tmp781 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp779 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)160,  (int32_t)128, tmp85 );

auto tmp784 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp781, tmp86, tmp87,  (int32_t)12, tmp784 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp781 );
ClearMemSecret1( (int32_t)128, tmp87 );
ClearMemSecret1( (int32_t)128, tmp86 );

auto tmp788 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp784, tmp788 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp784 );

auto tmp790 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp788, tmp90,  (int32_t)12, tmp790 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp90 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp788 );

auto tmp793 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp772,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp790,  (int32_t)3, tmp793 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp790 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)160, tmp772 );
ClearMemPublic( (int32_t)3 );

auto tmp797 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp793, tmp91, tmp92,  (int32_t)12, tmp797 );
ClearMemSecret1( (int32_t)192, tmp91 );
ClearMemSecret1( (int32_t)192, tmp92 );

auto tmp800 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp797, tmp800 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp797 );

auto tmp802 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp800, tmp95,  (int32_t)12, tmp802 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp800 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)128, tmp95 );

auto tmp805 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp802, tmp96, tmp97,  (int32_t)12, tmp805 );
ClearMemSecret1( (int32_t)128, tmp96 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp802 );
ClearMemSecret1( (int32_t)128, tmp97 );

auto tmp809 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp805, tmp809 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp805 );

auto tmp811 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp809, tmp100,  (int32_t)12, tmp811 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp100 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp809 );

auto tmp814 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp793,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp811,  (int32_t)3, tmp814 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)192, tmp793 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp811 );
ClearMemPublic( (int32_t)3 );

auto tmp818 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp814, tmp101, tmp102,  (int32_t)12, tmp818 );
ClearMemSecret1( (int32_t)224, tmp101 );
ClearMemSecret1( (int32_t)224, tmp102 );

auto tmp821 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp818, tmp821 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp818 );

auto tmp823 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp821, tmp105,  (int32_t)12, tmp823 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp821 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)224,  (int32_t)128, tmp105 );

auto tmp826 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp823, tmp106, tmp107,  (int32_t)12, tmp826 );
ClearMemSecret1( (int32_t)128, tmp106 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp823 );
ClearMemSecret1( (int32_t)128, tmp107 );

auto tmp830 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp826, tmp830 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp826 );

auto tmp832 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp830, tmp110,  (int32_t)12, tmp832 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp110 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp830 );

auto tmp835 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp814,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp832,  (int32_t)3, tmp835 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp832 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)224, tmp814 );

auto tmp839 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp835, tmp111, tmp112,  (int32_t)12, tmp839 );
ClearMemSecret1( (int32_t)256, tmp112 );
ClearMemSecret1( (int32_t)256, tmp111 );

auto tmp842 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp839, tmp842 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp839 );

auto tmp844 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp842, tmp115,  (int32_t)12, tmp844 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128, tmp115 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp842 );

auto tmp847 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp844, tmp116, tmp117,  (int32_t)12, tmp847 );
ClearMemSecret1( (int32_t)128, tmp117 );
ClearMemSecret1( (int32_t)128, tmp116 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp844 );

auto tmp851 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp847, tmp851 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp847 );

auto tmp853 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp851, tmp120,  (int32_t)12, tmp853 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp120 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp851 );

auto tmp856 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp835,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp853,  (int32_t)3, tmp856 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp835 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp853 );

auto tmp860 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp856, tmp121, tmp122,  (int32_t)12, tmp860 );
ClearMemSecret1( (int32_t)288, tmp121 );
ClearMemSecret1( (int32_t)288, tmp122 );

auto tmp863 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp860, tmp863 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp860 );

auto tmp865 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp863, tmp125,  (int32_t)12, tmp865 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)288,  (int32_t)128, tmp125 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp863 );

auto tmp868 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp865, tmp126, tmp127,  (int32_t)12, tmp868 );
ClearMemSecret1( (int32_t)128, tmp127 );
ClearMemSecret1( (int32_t)128, tmp126 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp865 );

auto tmp872 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp868, tmp872 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp868 );

auto tmp874 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp872, tmp130,  (int32_t)12, tmp874 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp130 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp872 );

auto tmp877 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp856,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp874,  (int32_t)3, tmp877 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp874 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)288, tmp856 );

auto tmp881 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp877, tmp131, tmp132,  (int32_t)12, tmp881 );
ClearMemSecret1( (int32_t)320, tmp131 );
ClearMemSecret1( (int32_t)320, tmp132 );

auto tmp884 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp881, tmp884 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp881 );

auto tmp886 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp884, tmp135,  (int32_t)12, tmp886 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)320,  (int32_t)128, tmp135 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp884 );

auto tmp889 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp886, tmp136, tmp137,  (int32_t)12, tmp889 );
ClearMemSecret1( (int32_t)128, tmp136 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp886 );
ClearMemSecret1( (int32_t)128, tmp137 );

auto tmp893 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp889, tmp893 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp889 );

auto tmp895 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp893, tmp140,  (int32_t)12, tmp895 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp140 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp893 );

auto tmp898 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp877,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp895,  (int32_t)3, tmp898 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)320, tmp877 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp895 );

auto tmp902 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp898, tmp141, tmp142,  (int32_t)12, tmp902 );
ClearMemSecret1( (int32_t)352, tmp141 );
ClearMemSecret1( (int32_t)352, tmp142 );

auto tmp905 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp902, tmp905 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp902 );

auto tmp907 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp905, tmp145,  (int32_t)12, tmp907 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)352,  (int32_t)128, tmp145 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp905 );

auto tmp910 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp907, tmp146, tmp147,  (int32_t)12, tmp910 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp907 );
ClearMemSecret1( (int32_t)128, tmp147 );
ClearMemSecret1( (int32_t)128, tmp146 );

auto tmp914 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp910, tmp914 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp910 );

auto tmp916 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp914, tmp150,  (int32_t)12, tmp916 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp914 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp150 );

auto tmp919 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp898,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp916,  (int32_t)3, tmp919 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp916 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)352, tmp898 );

auto tmp923 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp919, tmp151, tmp152,  (int32_t)12, tmp923 );
ClearMemSecret1( (int32_t)384, tmp151 );
ClearMemSecret1( (int32_t)384, tmp152 );

auto tmp926 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp923, tmp926 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp923 );

auto tmp928 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp926, tmp155,  (int32_t)12, tmp928 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)128, tmp155 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp926 );

auto tmp931 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp928, tmp156, tmp157,  (int32_t)12, tmp931 );
ClearMemSecret1( (int32_t)128, tmp157 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp928 );
ClearMemSecret1( (int32_t)128, tmp156 );

auto tmp935 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp931, tmp935 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp931 );

auto tmp937 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp935, tmp160,  (int32_t)12, tmp937 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp935 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp160 );

auto tmp940 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp919,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp937,  (int32_t)3, tmp940 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp937 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)384, tmp919 );
ClearMemPublic( (int32_t)3 );

auto tmp944 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp940, tmp161, tmp162,  (int32_t)12, tmp944 );
ClearMemSecret1( (int32_t)416, tmp162 );
ClearMemSecret1( (int32_t)416, tmp161 );

auto tmp947 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp944, tmp947 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp944 );

auto tmp949 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp947, tmp165,  (int32_t)12, tmp949 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp947 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)416,  (int32_t)128, tmp165 );

auto tmp952 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp949, tmp166, tmp167,  (int32_t)12, tmp952 );
ClearMemSecret1( (int32_t)128, tmp167 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp949 );
ClearMemSecret1( (int32_t)128, tmp166 );

auto tmp956 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp952, tmp956 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp952 );

auto tmp958 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp956, tmp170,  (int32_t)12, tmp958 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp956 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp170 );

auto tmp961 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp940,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp958,  (int32_t)3, tmp961 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)416, tmp940 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp958 );
ClearMemPublic( (int32_t)3 );

auto tmp965 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp961, tmp171, tmp172,  (int32_t)12, tmp965 );
ClearMemSecret1( (int32_t)448, tmp172 );
ClearMemSecret1( (int32_t)448, tmp171 );

auto tmp968 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp965, tmp968 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp965 );

auto tmp970 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp968, tmp175,  (int32_t)12, tmp970 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)448,  (int32_t)128, tmp175 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp968 );

auto tmp973 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp970, tmp176, tmp177,  (int32_t)12, tmp973 );
ClearMemSecret1( (int32_t)128, tmp177 );
ClearMemSecret1( (int32_t)128, tmp176 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp970 );

auto tmp977 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp973, tmp977 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp973 );

auto tmp979 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp977, tmp180,  (int32_t)12, tmp979 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp180 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp977 );

auto tmp982 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp961,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp979,  (int32_t)3, tmp982 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp979 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)448, tmp961 );
ClearMemPublic( (int32_t)3 );

auto tmp986 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp982, tmp181, tmp182,  (int32_t)12, tmp986 );
ClearMemSecret1( (int32_t)480, tmp182 );
ClearMemSecret1( (int32_t)480, tmp181 );

auto tmp989 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp986, tmp989 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp986 );

auto tmp991 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp989, tmp185,  (int32_t)12, tmp991 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp989 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)480,  (int32_t)128, tmp185 );

auto tmp994 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp991, tmp186, tmp187,  (int32_t)12, tmp994 );
ClearMemSecret1( (int32_t)128, tmp187 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp991 );
ClearMemSecret1( (int32_t)128, tmp186 );

auto tmp998 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp994, tmp998 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp994 );

auto tmp1000 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp998, tmp190,  (int32_t)12, tmp1000 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp190 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)128, tmp998 );

auto tmp1003 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512 );
Concat2T444( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp982,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp1000,  (int32_t)3, tmp1003 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)480, tmp982 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)32, tmp1000 );

auto tmp1007 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1003, tmp191, tmp192,  (int32_t)12, tmp1007 );
ClearMemSecret1( (int32_t)512, tmp192 );
ClearMemSecret1( (int32_t)512, tmp191 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1003 );

auto tmp1011 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512 );
Relu4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1007, tmp1011 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1007 );

auto tmp1013 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256 );
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1011, tmp195,  (int32_t)12, tmp1013 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)512, tmp1011 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)256, tmp195 );

auto tmp1016 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256 );
AvgPool( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1013, tmp1016 );
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)256, tmp1013 );

auto tmp1018 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1016, tmp196, tmp197,  (int32_t)12, tmp1018 );
ClearMemSecret1( (int32_t)256, tmp197 );
ClearMemSecret1( (int32_t)256, tmp196 );

auto tmp1021 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1018, tmp1021 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1018 );

auto tmp1023 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1021, tmp200,  (int32_t)12, tmp1023 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)128, tmp200 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1021 );

auto tmp1026 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1023, tmp201, tmp202,  (int32_t)12, tmp1026 );
ClearMemSecret1( (int32_t)128, tmp201 );
ClearMemSecret1( (int32_t)128, tmp202 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1023 );

auto tmp1030 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1026, tmp1030 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1026 );

auto tmp1032 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1030, tmp205,  (int32_t)12, tmp1032 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp205 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1030 );

auto tmp1035 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1016,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1032,  (int32_t)3, tmp1035 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1032 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)256, tmp1016 );

auto tmp1039 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp1035, tmp206, tmp207,  (int32_t)12, tmp1039 );
ClearMemSecret1( (int32_t)288, tmp206 );
ClearMemSecret1( (int32_t)288, tmp207 );

auto tmp1042 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp1039, tmp1042 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp1039 );

auto tmp1044 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1042, tmp210,  (int32_t)12, tmp1044 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp1042 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)288,  (int32_t)128, tmp210 );

auto tmp1047 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1044, tmp211, tmp212,  (int32_t)12, tmp1047 );
ClearMemSecret1( (int32_t)128, tmp211 );
ClearMemSecret1( (int32_t)128, tmp212 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1044 );

auto tmp1051 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1047, tmp1051 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1047 );

auto tmp1053 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1051, tmp215,  (int32_t)12, tmp1053 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp215 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1051 );

auto tmp1056 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp1035,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1053,  (int32_t)3, tmp1056 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1053 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)288, tmp1035 );

auto tmp1060 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp1056, tmp216, tmp217,  (int32_t)12, tmp1060 );
ClearMemSecret1( (int32_t)320, tmp216 );
ClearMemSecret1( (int32_t)320, tmp217 );

auto tmp1063 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp1060, tmp1063 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp1060 );

auto tmp1065 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1063, tmp220,  (int32_t)12, tmp1065 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp1063 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)320,  (int32_t)128, tmp220 );

auto tmp1068 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1065, tmp221, tmp222,  (int32_t)12, tmp1068 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1065 );
ClearMemSecret1( (int32_t)128, tmp221 );
ClearMemSecret1( (int32_t)128, tmp222 );

auto tmp1072 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1068, tmp1072 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1068 );

auto tmp1074 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1072, tmp225,  (int32_t)12, tmp1074 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp225 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1072 );

auto tmp1077 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp1056,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1074,  (int32_t)3, tmp1077 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1074 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)320, tmp1056 );

auto tmp1081 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp1077, tmp226, tmp227,  (int32_t)12, tmp1081 );
ClearMemSecret1( (int32_t)352, tmp227 );
ClearMemSecret1( (int32_t)352, tmp226 );

auto tmp1084 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp1081, tmp1084 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp1081 );

auto tmp1086 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1084, tmp230,  (int32_t)12, tmp1086 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)352,  (int32_t)128, tmp230 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp1084 );

auto tmp1089 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1086, tmp231, tmp232,  (int32_t)12, tmp1089 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1086 );
ClearMemSecret1( (int32_t)128, tmp231 );
ClearMemSecret1( (int32_t)128, tmp232 );

auto tmp1093 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1089, tmp1093 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1089 );

auto tmp1095 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1093, tmp235,  (int32_t)12, tmp1095 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp235 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1093 );

auto tmp1098 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp1077,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1095,  (int32_t)3, tmp1098 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)352, tmp1077 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1095 );
ClearMemPublic( (int32_t)3 );

auto tmp1102 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp1098, tmp236, tmp237,  (int32_t)12, tmp1102 );
ClearMemSecret1( (int32_t)384, tmp237 );
ClearMemSecret1( (int32_t)384, tmp236 );

auto tmp1105 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp1102, tmp1105 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp1102 );

auto tmp1107 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1105, tmp240,  (int32_t)12, tmp1107 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp1105 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)128, tmp240 );

auto tmp1110 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1107, tmp241, tmp242,  (int32_t)12, tmp1110 );
ClearMemSecret1( (int32_t)128, tmp241 );
ClearMemSecret1( (int32_t)128, tmp242 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1107 );

auto tmp1114 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1110, tmp1114 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1110 );

auto tmp1116 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1114, tmp245,  (int32_t)12, tmp1116 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1114 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp245 );

auto tmp1119 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp1098,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1116,  (int32_t)3, tmp1119 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1116 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)384, tmp1098 );

auto tmp1123 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp1119, tmp246, tmp247,  (int32_t)12, tmp1123 );
ClearMemSecret1( (int32_t)416, tmp247 );
ClearMemSecret1( (int32_t)416, tmp246 );

auto tmp1126 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp1123, tmp1126 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp1123 );

auto tmp1128 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1126, tmp250,  (int32_t)12, tmp1128 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp1126 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)416,  (int32_t)128, tmp250 );

auto tmp1131 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1128, tmp251, tmp252,  (int32_t)12, tmp1131 );
ClearMemSecret1( (int32_t)128, tmp252 );
ClearMemSecret1( (int32_t)128, tmp251 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1128 );

auto tmp1135 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1131, tmp1135 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1131 );

auto tmp1137 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1135, tmp255,  (int32_t)12, tmp1137 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp255 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1135 );

auto tmp1140 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp1119,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1137,  (int32_t)3, tmp1140 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)416, tmp1119 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1137 );
ClearMemPublic( (int32_t)3 );

auto tmp1144 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp1140, tmp256, tmp257,  (int32_t)12, tmp1144 );
ClearMemSecret1( (int32_t)448, tmp256 );
ClearMemSecret1( (int32_t)448, tmp257 );

auto tmp1147 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp1144, tmp1147 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp1144 );

auto tmp1149 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1147, tmp260,  (int32_t)12, tmp1149 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)448,  (int32_t)128, tmp260 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp1147 );

auto tmp1152 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1149, tmp261, tmp262,  (int32_t)12, tmp1152 );
ClearMemSecret1( (int32_t)128, tmp262 );
ClearMemSecret1( (int32_t)128, tmp261 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1149 );

auto tmp1156 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1152, tmp1156 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1152 );

auto tmp1158 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1156, tmp265,  (int32_t)12, tmp1158 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp265 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1156 );

auto tmp1161 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp1140,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1158,  (int32_t)3, tmp1161 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1158 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)448, tmp1140 );

auto tmp1165 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp1161, tmp266, tmp267,  (int32_t)12, tmp1165 );
ClearMemSecret1( (int32_t)480, tmp267 );
ClearMemSecret1( (int32_t)480, tmp266 );

auto tmp1168 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp1165, tmp1168 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp1165 );

auto tmp1170 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1168, tmp270,  (int32_t)12, tmp1170 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp1168 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)480,  (int32_t)128, tmp270 );

auto tmp1173 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1170, tmp271, tmp272,  (int32_t)12, tmp1173 );
ClearMemSecret1( (int32_t)128, tmp271 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1170 );
ClearMemSecret1( (int32_t)128, tmp272 );

auto tmp1177 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1173, tmp1177 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1173 );

auto tmp1179 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1177, tmp275,  (int32_t)12, tmp1179 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1177 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp275 );

auto tmp1182 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp1161,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1179,  (int32_t)3, tmp1182 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1179 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)480, tmp1161 );

auto tmp1186 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1182, tmp276, tmp277,  (int32_t)12, tmp1186 );
ClearMemSecret1( (int32_t)512, tmp277 );
ClearMemSecret1( (int32_t)512, tmp276 );

auto tmp1189 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1186, tmp1189 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1186 );

auto tmp1191 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1189, tmp280,  (int32_t)12, tmp1191 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1189 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp280 );

auto tmp1194 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1191, tmp281, tmp282,  (int32_t)12, tmp1194 );
ClearMemSecret1( (int32_t)128, tmp281 );
ClearMemSecret1( (int32_t)128, tmp282 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1191 );

auto tmp1198 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1194, tmp1198 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1194 );

auto tmp1200 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1198, tmp285,  (int32_t)12, tmp1200 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp285 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1198 );

auto tmp1203 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1182,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1200,  (int32_t)3, tmp1203 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1200 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1182 );

auto tmp1207 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp1203, tmp286, tmp287,  (int32_t)12, tmp1207 );
ClearMemSecret1( (int32_t)544, tmp287 );
ClearMemSecret1( (int32_t)544, tmp286 );

auto tmp1210 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp1207, tmp1210 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp1207 );

auto tmp1212 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1210, tmp290,  (int32_t)12, tmp1212 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)544,  (int32_t)128, tmp290 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp1210 );

auto tmp1215 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1212, tmp291, tmp292,  (int32_t)12, tmp1215 );
ClearMemSecret1( (int32_t)128, tmp292 );
ClearMemSecret1( (int32_t)128, tmp291 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1212 );

auto tmp1219 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1215, tmp1219 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1215 );

auto tmp1221 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1219, tmp295,  (int32_t)12, tmp1221 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp295 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1219 );

auto tmp1224 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp1203,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1221,  (int32_t)3, tmp1224 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1221 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)544, tmp1203 );

auto tmp1228 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp1224, tmp296, tmp297,  (int32_t)12, tmp1228 );
ClearMemSecret1( (int32_t)576, tmp297 );
ClearMemSecret1( (int32_t)576, tmp296 );

auto tmp1231 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp1228, tmp1231 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp1228 );

auto tmp1233 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1231, tmp300,  (int32_t)12, tmp1233 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp1231 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)576,  (int32_t)128, tmp300 );

auto tmp1236 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1233, tmp301, tmp302,  (int32_t)12, tmp1236 );
ClearMemSecret1( (int32_t)128, tmp301 );
ClearMemSecret1( (int32_t)128, tmp302 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1233 );

auto tmp1240 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1236, tmp1240 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1236 );

auto tmp1242 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1240, tmp305,  (int32_t)12, tmp1242 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1240 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp305 );

auto tmp1245 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp1224,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1242,  (int32_t)3, tmp1245 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1242 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)576, tmp1224 );

auto tmp1249 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp1245, tmp306, tmp307,  (int32_t)12, tmp1249 );
ClearMemSecret1( (int32_t)608, tmp306 );
ClearMemSecret1( (int32_t)608, tmp307 );

auto tmp1252 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp1249, tmp1252 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp1249 );

auto tmp1254 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1252, tmp310,  (int32_t)12, tmp1254 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)608,  (int32_t)128, tmp310 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp1252 );

auto tmp1257 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1254, tmp311, tmp312,  (int32_t)12, tmp1257 );
ClearMemSecret1( (int32_t)128, tmp311 );
ClearMemSecret1( (int32_t)128, tmp312 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1254 );

auto tmp1261 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1257, tmp1261 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1257 );

auto tmp1263 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1261, tmp315,  (int32_t)12, tmp1263 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp315 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1261 );

auto tmp1266 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp1245,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1263,  (int32_t)3, tmp1266 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)608, tmp1245 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1263 );

auto tmp1270 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp1266, tmp316, tmp317,  (int32_t)12, tmp1270 );
ClearMemSecret1( (int32_t)640, tmp317 );
ClearMemSecret1( (int32_t)640, tmp316 );

auto tmp1273 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp1270, tmp1273 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp1270 );

auto tmp1275 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1273, tmp320,  (int32_t)12, tmp1275 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)640,  (int32_t)128, tmp320 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp1273 );

auto tmp1278 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1275, tmp321, tmp322,  (int32_t)12, tmp1278 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1275 );
ClearMemSecret1( (int32_t)128, tmp321 );
ClearMemSecret1( (int32_t)128, tmp322 );

auto tmp1282 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1278, tmp1282 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1278 );

auto tmp1284 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1282, tmp325,  (int32_t)12, tmp1284 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp325 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1282 );

auto tmp1287 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp1266,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1284,  (int32_t)3, tmp1287 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)640, tmp1266 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1284 );

auto tmp1291 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp1287, tmp326, tmp327,  (int32_t)12, tmp1291 );
ClearMemSecret1( (int32_t)672, tmp327 );
ClearMemSecret1( (int32_t)672, tmp326 );

auto tmp1294 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp1291, tmp1294 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp1291 );

auto tmp1296 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1294, tmp330,  (int32_t)12, tmp1296 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp1294 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)672,  (int32_t)128, tmp330 );

auto tmp1299 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1296, tmp331, tmp332,  (int32_t)12, tmp1299 );
ClearMemSecret1( (int32_t)128, tmp332 );
ClearMemSecret1( (int32_t)128, tmp331 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1296 );

auto tmp1303 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1299, tmp1303 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1299 );

auto tmp1305 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1303, tmp335,  (int32_t)12, tmp1305 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1303 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp335 );

auto tmp1308 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp1287,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1305,  (int32_t)3, tmp1308 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)672, tmp1287 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1305 );

auto tmp1312 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp1308, tmp336, tmp337,  (int32_t)12, tmp1312 );
ClearMemSecret1( (int32_t)704, tmp337 );
ClearMemSecret1( (int32_t)704, tmp336 );

auto tmp1315 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp1312, tmp1315 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp1312 );

auto tmp1317 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1315, tmp340,  (int32_t)12, tmp1317 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp1315 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)704,  (int32_t)128, tmp340 );

auto tmp1320 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1317, tmp341, tmp342,  (int32_t)12, tmp1320 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1317 );
ClearMemSecret1( (int32_t)128, tmp341 );
ClearMemSecret1( (int32_t)128, tmp342 );

auto tmp1324 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1320, tmp1324 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1320 );

auto tmp1326 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1324, tmp345,  (int32_t)12, tmp1326 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp345 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1324 );

auto tmp1329 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp1308,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1326,  (int32_t)3, tmp1329 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)704, tmp1308 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1326 );
ClearMemPublic( (int32_t)3 );

auto tmp1333 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp1329, tmp346, tmp347,  (int32_t)12, tmp1333 );
ClearMemSecret1( (int32_t)736, tmp347 );
ClearMemSecret1( (int32_t)736, tmp346 );

auto tmp1336 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp1333, tmp1336 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp1333 );

auto tmp1338 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1336, tmp350,  (int32_t)12, tmp1338 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp1336 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)736,  (int32_t)128, tmp350 );

auto tmp1341 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1338, tmp351, tmp352,  (int32_t)12, tmp1341 );
ClearMemSecret1( (int32_t)128, tmp351 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1338 );
ClearMemSecret1( (int32_t)128, tmp352 );

auto tmp1345 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1341, tmp1345 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1341 );

auto tmp1347 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1345, tmp355,  (int32_t)12, tmp1347 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1345 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp355 );

auto tmp1350 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp1329,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1347,  (int32_t)3, tmp1350 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)736, tmp1329 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1347 );
ClearMemPublic( (int32_t)3 );

auto tmp1354 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp1350, tmp356, tmp357,  (int32_t)12, tmp1354 );
ClearMemSecret1( (int32_t)768, tmp356 );
ClearMemSecret1( (int32_t)768, tmp357 );

auto tmp1357 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp1354, tmp1357 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp1354 );

auto tmp1359 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1357, tmp360,  (int32_t)12, tmp1359 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)768,  (int32_t)128, tmp360 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp1357 );

auto tmp1362 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1359, tmp361, tmp362,  (int32_t)12, tmp1362 );
ClearMemSecret1( (int32_t)128, tmp361 );
ClearMemSecret1( (int32_t)128, tmp362 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1359 );

auto tmp1366 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1362, tmp1366 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1362 );

auto tmp1368 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1366, tmp365,  (int32_t)12, tmp1368 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp365 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1366 );

auto tmp1371 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp1350,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1368,  (int32_t)3, tmp1371 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)768, tmp1350 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1368 );

auto tmp1375 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp1371, tmp366, tmp367,  (int32_t)12, tmp1375 );
ClearMemSecret1( (int32_t)800, tmp366 );
ClearMemSecret1( (int32_t)800, tmp367 );

auto tmp1378 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp1375, tmp1378 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp1375 );

auto tmp1380 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1378, tmp370,  (int32_t)12, tmp1380 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)800,  (int32_t)128, tmp370 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp1378 );

auto tmp1383 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1380, tmp371, tmp372,  (int32_t)12, tmp1383 );
ClearMemSecret1( (int32_t)128, tmp371 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1380 );
ClearMemSecret1( (int32_t)128, tmp372 );

auto tmp1387 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1383, tmp1387 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1383 );

auto tmp1389 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1387, tmp375,  (int32_t)12, tmp1389 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp375 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1387 );

auto tmp1392 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp1371,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1389,  (int32_t)3, tmp1392 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)800, tmp1371 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1389 );
ClearMemPublic( (int32_t)3 );

auto tmp1396 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp1392, tmp376, tmp377,  (int32_t)12, tmp1396 );
ClearMemSecret1( (int32_t)832, tmp377 );
ClearMemSecret1( (int32_t)832, tmp376 );

auto tmp1399 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp1396, tmp1399 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp1396 );

auto tmp1401 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1399, tmp380,  (int32_t)12, tmp1401 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp1399 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)832,  (int32_t)128, tmp380 );

auto tmp1404 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1401, tmp381, tmp382,  (int32_t)12, tmp1404 );
ClearMemSecret1( (int32_t)128, tmp382 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1401 );
ClearMemSecret1( (int32_t)128, tmp381 );

auto tmp1408 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1404, tmp1408 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1404 );

auto tmp1410 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1408, tmp385,  (int32_t)12, tmp1410 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1408 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp385 );

auto tmp1413 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp1392,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1410,  (int32_t)3, tmp1413 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)832, tmp1392 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1410 );

auto tmp1417 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp1413, tmp386, tmp387,  (int32_t)12, tmp1417 );
ClearMemSecret1( (int32_t)864, tmp386 );
ClearMemSecret1( (int32_t)864, tmp387 );

auto tmp1420 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp1417, tmp1420 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp1417 );

auto tmp1422 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1420, tmp390,  (int32_t)12, tmp1422 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp1420 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)864,  (int32_t)128, tmp390 );

auto tmp1425 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1422, tmp391, tmp392,  (int32_t)12, tmp1425 );
ClearMemSecret1( (int32_t)128, tmp391 );
ClearMemSecret1( (int32_t)128, tmp392 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1422 );

auto tmp1429 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1425, tmp1429 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1425 );

auto tmp1431 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1429, tmp395,  (int32_t)12, tmp1431 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp395 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1429 );

auto tmp1434 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp1413,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1431,  (int32_t)3, tmp1434 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1431 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)864, tmp1413 );
ClearMemPublic( (int32_t)3 );

auto tmp1438 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp1434, tmp396, tmp397,  (int32_t)12, tmp1438 );
ClearMemSecret1( (int32_t)896, tmp397 );
ClearMemSecret1( (int32_t)896, tmp396 );

auto tmp1441 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp1438, tmp1441 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp1438 );

auto tmp1443 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1441, tmp400,  (int32_t)12, tmp1443 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp1441 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)896,  (int32_t)128, tmp400 );

auto tmp1446 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1443, tmp401, tmp402,  (int32_t)12, tmp1446 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1443 );
ClearMemSecret1( (int32_t)128, tmp401 );
ClearMemSecret1( (int32_t)128, tmp402 );

auto tmp1450 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1446, tmp1450 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1446 );

auto tmp1452 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1450, tmp405,  (int32_t)12, tmp1452 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1450 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp405 );

auto tmp1455 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp1434,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1452,  (int32_t)3, tmp1455 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)896, tmp1434 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1452 );

auto tmp1459 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp1455, tmp406, tmp407,  (int32_t)12, tmp1459 );
ClearMemSecret1( (int32_t)928, tmp406 );
ClearMemSecret1( (int32_t)928, tmp407 );

auto tmp1462 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp1459, tmp1462 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp1459 );

auto tmp1464 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1462, tmp410,  (int32_t)12, tmp1464 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp1462 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)928,  (int32_t)128, tmp410 );

auto tmp1467 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1464, tmp411, tmp412,  (int32_t)12, tmp1467 );
ClearMemSecret1( (int32_t)128, tmp412 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1464 );
ClearMemSecret1( (int32_t)128, tmp411 );

auto tmp1471 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1467, tmp1471 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1467 );

auto tmp1473 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1471, tmp415,  (int32_t)12, tmp1473 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp415 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1471 );

auto tmp1476 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp1455,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1473,  (int32_t)3, tmp1476 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)928, tmp1455 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1473 );

auto tmp1480 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp1476, tmp416, tmp417,  (int32_t)12, tmp1480 );
ClearMemSecret1( (int32_t)960, tmp417 );
ClearMemSecret1( (int32_t)960, tmp416 );

auto tmp1483 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp1480, tmp1483 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp1480 );

auto tmp1485 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1483, tmp420,  (int32_t)12, tmp1485 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp1483 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)960,  (int32_t)128, tmp420 );

auto tmp1488 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1485, tmp421, tmp422,  (int32_t)12, tmp1488 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1485 );
ClearMemSecret1( (int32_t)128, tmp422 );
ClearMemSecret1( (int32_t)128, tmp421 );

auto tmp1492 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1488, tmp1492 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1488 );

auto tmp1494 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1492, tmp425,  (int32_t)12, tmp1494 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1492 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp425 );

auto tmp1497 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp1476,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1494,  (int32_t)3, tmp1497 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)960, tmp1476 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1494 );

auto tmp1501 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp1497, tmp426, tmp427,  (int32_t)12, tmp1501 );
ClearMemSecret1( (int32_t)992, tmp427 );
ClearMemSecret1( (int32_t)992, tmp426 );

auto tmp1504 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp1501, tmp1504 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp1501 );

auto tmp1506 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1504, tmp430,  (int32_t)12, tmp1506 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)992,  (int32_t)128, tmp430 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp1504 );

auto tmp1509 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1506, tmp431, tmp432,  (int32_t)12, tmp1509 );
ClearMemSecret1( (int32_t)128, tmp432 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1506 );
ClearMemSecret1( (int32_t)128, tmp431 );

auto tmp1513 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1509, tmp1513 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1509 );

auto tmp1515 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1513, tmp435,  (int32_t)12, tmp1515 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp435 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)128, tmp1513 );

auto tmp1518 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024 );
Concat2T444( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp1497,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1515,  (int32_t)3, tmp1518 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)32, tmp1515 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)992, tmp1497 );

auto tmp1522 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1518, tmp436, tmp437,  (int32_t)12, tmp1522 );
ClearMemSecret1( (int32_t)1024, tmp436 );
ClearMemSecret1( (int32_t)1024, tmp437 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1518 );

auto tmp1526 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024 );
Relu4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1522, tmp1526 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1522 );

auto tmp1528 = make_vector<uint64_t>( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512 );
Conv2DCSF( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1526, tmp440,  (int32_t)12, tmp1528 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)1024, tmp1526 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)512, tmp440 );

auto tmp1531 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512 );
AvgPool( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1528, tmp1531 );
ClearMemSecret4( (int32_t)1,  (int32_t)14,  (int32_t)14,  (int32_t)512, tmp1528 );

auto tmp1533 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1531, tmp441, tmp442,  (int32_t)12, tmp1533 );
ClearMemSecret1( (int32_t)512, tmp442 );
ClearMemSecret1( (int32_t)512, tmp441 );

auto tmp1536 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1533, tmp1536 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1533 );

auto tmp1538 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1536, tmp445,  (int32_t)12, tmp1538 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)128, tmp445 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1536 );

auto tmp1541 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1538, tmp446, tmp447,  (int32_t)12, tmp1541 );
ClearMemSecret1( (int32_t)128, tmp446 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1538 );
ClearMemSecret1( (int32_t)128, tmp447 );

auto tmp1545 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1541, tmp1545 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1541 );

auto tmp1547 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1545, tmp450,  (int32_t)12, tmp1547 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp450 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1545 );

auto tmp1550 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1531,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1547,  (int32_t)3, tmp1550 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1547 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)512, tmp1531 );
ClearMemPublic( (int32_t)3 );

auto tmp1554 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp1550, tmp451, tmp452,  (int32_t)12, tmp1554 );
ClearMemSecret1( (int32_t)544, tmp452 );
ClearMemSecret1( (int32_t)544, tmp451 );

auto tmp1557 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp1554, tmp1557 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp1554 );

auto tmp1559 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1557, tmp455,  (int32_t)12, tmp1559 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)544,  (int32_t)128, tmp455 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp1557 );

auto tmp1562 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1559, tmp456, tmp457,  (int32_t)12, tmp1562 );
ClearMemSecret1( (int32_t)128, tmp456 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1559 );
ClearMemSecret1( (int32_t)128, tmp457 );

auto tmp1566 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1562, tmp1566 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1562 );

auto tmp1568 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1566, tmp460,  (int32_t)12, tmp1568 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp460 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1566 );

auto tmp1571 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp1550,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1568,  (int32_t)3, tmp1571 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)544, tmp1550 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1568 );

auto tmp1575 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp1571, tmp461, tmp462,  (int32_t)12, tmp1575 );
ClearMemSecret1( (int32_t)576, tmp461 );
ClearMemSecret1( (int32_t)576, tmp462 );

auto tmp1578 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp1575, tmp1578 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp1575 );

auto tmp1580 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1578, tmp465,  (int32_t)12, tmp1580 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)576,  (int32_t)128, tmp465 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp1578 );

auto tmp1583 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1580, tmp466, tmp467,  (int32_t)12, tmp1583 );
ClearMemSecret1( (int32_t)128, tmp466 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1580 );
ClearMemSecret1( (int32_t)128, tmp467 );

auto tmp1587 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1583, tmp1587 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1583 );

auto tmp1589 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1587, tmp470,  (int32_t)12, tmp1589 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1587 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp470 );

auto tmp1592 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp1571,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1589,  (int32_t)3, tmp1592 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1589 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)576, tmp1571 );

auto tmp1596 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp1592, tmp471, tmp472,  (int32_t)12, tmp1596 );
ClearMemSecret1( (int32_t)608, tmp472 );
ClearMemSecret1( (int32_t)608, tmp471 );

auto tmp1599 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp1596, tmp1599 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp1596 );

auto tmp1601 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1599, tmp475,  (int32_t)12, tmp1601 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)608,  (int32_t)128, tmp475 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp1599 );

auto tmp1604 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1601, tmp476, tmp477,  (int32_t)12, tmp1604 );
ClearMemSecret1( (int32_t)128, tmp476 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1601 );
ClearMemSecret1( (int32_t)128, tmp477 );

auto tmp1608 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1604, tmp1608 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1604 );

auto tmp1610 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1608, tmp480,  (int32_t)12, tmp1610 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp480 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1608 );

auto tmp1613 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp1592,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1610,  (int32_t)3, tmp1613 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1610 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)608, tmp1592 );
ClearMemPublic( (int32_t)3 );

auto tmp1617 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp1613, tmp481, tmp482,  (int32_t)12, tmp1617 );
ClearMemSecret1( (int32_t)640, tmp481 );
ClearMemSecret1( (int32_t)640, tmp482 );

auto tmp1620 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp1617, tmp1620 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp1617 );

auto tmp1622 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1620, tmp485,  (int32_t)12, tmp1622 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp1620 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)640,  (int32_t)128, tmp485 );

auto tmp1625 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1622, tmp486, tmp487,  (int32_t)12, tmp1625 );
ClearMemSecret1( (int32_t)128, tmp486 );
ClearMemSecret1( (int32_t)128, tmp487 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1622 );

auto tmp1629 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1625, tmp1629 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1625 );

auto tmp1631 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1629, tmp490,  (int32_t)12, tmp1631 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1629 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp490 );

auto tmp1634 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp1613,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1631,  (int32_t)3, tmp1634 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1631 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)640, tmp1613 );
ClearMemPublic( (int32_t)3 );

auto tmp1638 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp1634, tmp491, tmp492,  (int32_t)12, tmp1638 );
ClearMemSecret1( (int32_t)672, tmp491 );
ClearMemSecret1( (int32_t)672, tmp492 );

auto tmp1641 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp1638, tmp1641 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp1638 );

auto tmp1643 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1641, tmp495,  (int32_t)12, tmp1643 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)672,  (int32_t)128, tmp495 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp1641 );

auto tmp1646 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1643, tmp496, tmp497,  (int32_t)12, tmp1646 );
ClearMemSecret1( (int32_t)128, tmp497 );
ClearMemSecret1( (int32_t)128, tmp496 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1643 );

auto tmp1650 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1646, tmp1650 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1646 );

auto tmp1652 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1650, tmp500,  (int32_t)12, tmp1652 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1650 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp500 );

auto tmp1655 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp1634,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1652,  (int32_t)3, tmp1655 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1652 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)672, tmp1634 );

auto tmp1659 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp1655, tmp501, tmp502,  (int32_t)12, tmp1659 );
ClearMemSecret1( (int32_t)704, tmp501 );
ClearMemSecret1( (int32_t)704, tmp502 );

auto tmp1662 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp1659, tmp1662 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp1659 );

auto tmp1664 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1662, tmp505,  (int32_t)12, tmp1664 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp1662 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)704,  (int32_t)128, tmp505 );

auto tmp1667 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1664, tmp506, tmp507,  (int32_t)12, tmp1667 );
ClearMemSecret1( (int32_t)128, tmp506 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1664 );
ClearMemSecret1( (int32_t)128, tmp507 );

auto tmp1671 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1667, tmp1671 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1667 );

auto tmp1673 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1671, tmp510,  (int32_t)12, tmp1673 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp510 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1671 );

auto tmp1676 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp1655,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1673,  (int32_t)3, tmp1676 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)704, tmp1655 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1673 );

auto tmp1680 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp1676, tmp511, tmp512,  (int32_t)12, tmp1680 );
ClearMemSecret1( (int32_t)736, tmp512 );
ClearMemSecret1( (int32_t)736, tmp511 );

auto tmp1683 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp1680, tmp1683 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp1680 );

auto tmp1685 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1683, tmp515,  (int32_t)12, tmp1685 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)736,  (int32_t)128, tmp515 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp1683 );

auto tmp1688 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1685, tmp516, tmp517,  (int32_t)12, tmp1688 );
ClearMemSecret1( (int32_t)128, tmp517 );
ClearMemSecret1( (int32_t)128, tmp516 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1685 );

auto tmp1692 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1688, tmp1692 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1688 );

auto tmp1694 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1692, tmp520,  (int32_t)12, tmp1694 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1692 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp520 );

auto tmp1697 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp1676,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1694,  (int32_t)3, tmp1697 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)736, tmp1676 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1694 );

auto tmp1701 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp1697, tmp521, tmp522,  (int32_t)12, tmp1701 );
ClearMemSecret1( (int32_t)768, tmp522 );
ClearMemSecret1( (int32_t)768, tmp521 );

auto tmp1704 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp1701, tmp1704 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp1701 );

auto tmp1706 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1704, tmp525,  (int32_t)12, tmp1706 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp1704 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)768,  (int32_t)128, tmp525 );

auto tmp1709 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1706, tmp526, tmp527,  (int32_t)12, tmp1709 );
ClearMemSecret1( (int32_t)128, tmp526 );
ClearMemSecret1( (int32_t)128, tmp527 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1706 );

auto tmp1713 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1709, tmp1713 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1709 );

auto tmp1715 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1713, tmp530,  (int32_t)12, tmp1715 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1713 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp530 );

auto tmp1718 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp1697,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1715,  (int32_t)3, tmp1718 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1715 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)768, tmp1697 );

auto tmp1722 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp1718, tmp531, tmp532,  (int32_t)12, tmp1722 );
ClearMemSecret1( (int32_t)800, tmp532 );
ClearMemSecret1( (int32_t)800, tmp531 );

auto tmp1725 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp1722, tmp1725 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp1722 );

auto tmp1727 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1725, tmp535,  (int32_t)12, tmp1727 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp1725 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)800,  (int32_t)128, tmp535 );

auto tmp1730 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1727, tmp536, tmp537,  (int32_t)12, tmp1730 );
ClearMemSecret1( (int32_t)128, tmp536 );
ClearMemSecret1( (int32_t)128, tmp537 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1727 );

auto tmp1734 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1730, tmp1734 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1730 );

auto tmp1736 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1734, tmp540,  (int32_t)12, tmp1736 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1734 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp540 );

auto tmp1739 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp1718,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1736,  (int32_t)3, tmp1739 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)800, tmp1718 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1736 );

auto tmp1743 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp1739, tmp541, tmp542,  (int32_t)12, tmp1743 );
ClearMemSecret1( (int32_t)832, tmp542 );
ClearMemSecret1( (int32_t)832, tmp541 );

auto tmp1746 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp1743, tmp1746 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp1743 );

auto tmp1748 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1746, tmp545,  (int32_t)12, tmp1748 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp1746 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)832,  (int32_t)128, tmp545 );

auto tmp1751 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1748, tmp546, tmp547,  (int32_t)12, tmp1751 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1748 );
ClearMemSecret1( (int32_t)128, tmp547 );
ClearMemSecret1( (int32_t)128, tmp546 );

auto tmp1755 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1751, tmp1755 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1751 );

auto tmp1757 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1755, tmp550,  (int32_t)12, tmp1757 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp550 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1755 );

auto tmp1760 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp1739,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1757,  (int32_t)3, tmp1760 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)832, tmp1739 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1757 );

auto tmp1764 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp1760, tmp551, tmp552,  (int32_t)12, tmp1764 );
ClearMemSecret1( (int32_t)864, tmp551 );
ClearMemSecret1( (int32_t)864, tmp552 );

auto tmp1767 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp1764, tmp1767 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp1764 );

auto tmp1769 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1767, tmp555,  (int32_t)12, tmp1769 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)864,  (int32_t)128, tmp555 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp1767 );

auto tmp1772 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1769, tmp556, tmp557,  (int32_t)12, tmp1772 );
ClearMemSecret1( (int32_t)128, tmp556 );
ClearMemSecret1( (int32_t)128, tmp557 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1769 );

auto tmp1776 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1772, tmp1776 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1772 );

auto tmp1778 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1776, tmp560,  (int32_t)12, tmp1778 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1776 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp560 );

auto tmp1781 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp1760,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1778,  (int32_t)3, tmp1781 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1778 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)864, tmp1760 );

auto tmp1785 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1781, tmp561, tmp562,  (int32_t)12, tmp1785 );
ClearMemSecret1( (int32_t)896, tmp562 );
ClearMemSecret1( (int32_t)896, tmp561 );

auto tmp1788 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1785, tmp1788 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1785 );

auto tmp1790 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1788, tmp565,  (int32_t)12, tmp1790 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1788 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)896,  (int32_t)128, tmp565 );

auto tmp1793 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1790, tmp566, tmp567,  (int32_t)12, tmp1793 );
ClearMemSecret1( (int32_t)128, tmp567 );
ClearMemSecret1( (int32_t)128, tmp566 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1790 );

auto tmp1797 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1793, tmp1797 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1793 );

auto tmp1799 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1797, tmp570,  (int32_t)12, tmp1799 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp570 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1797 );

auto tmp1802 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1781,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1799,  (int32_t)3, tmp1802 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1799 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)896, tmp1781 );
ClearMemPublic( (int32_t)3 );

auto tmp1806 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1802, tmp571, tmp572,  (int32_t)12, tmp1806 );
ClearMemSecret1( (int32_t)928, tmp571 );
ClearMemSecret1( (int32_t)928, tmp572 );

auto tmp1809 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1806, tmp1809 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1806 );

auto tmp1811 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1809, tmp575,  (int32_t)12, tmp1811 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1809 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)928,  (int32_t)128, tmp575 );

auto tmp1814 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1811, tmp576, tmp577,  (int32_t)12, tmp1814 );
ClearMemSecret1( (int32_t)128, tmp576 );
ClearMemSecret1( (int32_t)128, tmp577 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1811 );

auto tmp1818 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1814, tmp1818 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1814 );

auto tmp1820 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1818, tmp580,  (int32_t)12, tmp1820 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp580 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1818 );

auto tmp1823 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1802,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1820,  (int32_t)3, tmp1823 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1820 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)928, tmp1802 );

auto tmp1827 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1823, tmp581, tmp582,  (int32_t)12, tmp1827 );
ClearMemSecret1( (int32_t)960, tmp581 );
ClearMemSecret1( (int32_t)960, tmp582 );

auto tmp1830 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1827, tmp1830 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1827 );

auto tmp1832 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1830, tmp585,  (int32_t)12, tmp1832 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1830 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)960,  (int32_t)128, tmp585 );

auto tmp1835 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1832, tmp586, tmp587,  (int32_t)12, tmp1835 );
ClearMemSecret1( (int32_t)128, tmp587 );
ClearMemSecret1( (int32_t)128, tmp586 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1832 );

auto tmp1839 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1835, tmp1839 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1835 );

auto tmp1841 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1839, tmp590,  (int32_t)12, tmp1841 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp590 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1839 );

auto tmp1844 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1823,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1841,  (int32_t)3, tmp1844 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)960, tmp1823 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1841 );

auto tmp1848 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1844, tmp591, tmp592,  (int32_t)12, tmp1848 );
ClearMemSecret1( (int32_t)992, tmp592 );
ClearMemSecret1( (int32_t)992, tmp591 );

auto tmp1851 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1848, tmp1851 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1848 );

auto tmp1853 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1851, tmp595,  (int32_t)12, tmp1853 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)992,  (int32_t)128, tmp595 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1851 );

auto tmp1856 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1853, tmp596, tmp597,  (int32_t)12, tmp1856 );
ClearMemSecret1( (int32_t)128, tmp596 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1853 );
ClearMemSecret1( (int32_t)128, tmp597 );

auto tmp1860 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1856, tmp1860 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1856 );

auto tmp1862 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp1860, tmp600,  (int32_t)12, tmp1862 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)32, tmp600 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)128, tmp1860 );

auto tmp1865 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024 );
Concat2T444( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1844,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1862,  (int32_t)3, tmp1865 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)992, tmp1844 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)32, tmp1862 );

auto tmp1869 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024 );
FusedBatchNorm4411( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1865, tmp601, tmp602,  (int32_t)12, tmp1869 );
ClearMemSecret1( (int32_t)1024, tmp602 );
ClearMemSecret1( (int32_t)1024, tmp601 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1865 );

auto tmp1873 = make_vector<uint64_t>( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024 );
Relu4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1869, tmp1873 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1869 );

auto tmp1875 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1024 );
AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)7,  (int32_t)7,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1873, tmp1875 );
ClearMemSecret4( (int32_t)1,  (int32_t)7,  (int32_t)7,  (int32_t)1024, tmp1873 );

auto tmp1877 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000 );
Conv2DCSF( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)1,  (int32_t)1,  (int32_t)1000,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp1875, tmp605,  (int32_t)12, tmp1877 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1024,  (int32_t)1000, tmp605 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1024, tmp1875 );

auto tmp1880 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000 );
MatAddBroadCast4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000, tmp1877, tmp606, tmp1880 );
ClearMemSecret1( (int32_t)1000, tmp606 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000, tmp1877 );

auto tmp1883 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1 );
ArgMax3( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000, tmp1880,  (int32_t)3, tmp1883 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000, tmp1880 );
ClearMemPublic( (int32_t)3 );

for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1; i2++){
			print_integer(funcReconstruct2PCCons(tmp1883[i0][i1][i2], 1));
			
		}
		
	}
	
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
