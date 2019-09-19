#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "res_net_mem_opti.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef SQ_NET_IMAGE_NET

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
auto tmp0 = make_vector<uint64_t>( (int32_t)1,  (int32_t)227,  (int32_t)227,  (int32_t)3 );
/* Variable to read the clear value corresponding to the input variable tmp0 at (393,1-393,47) */
uint64_t __tmp_in_tmp0;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)227; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)227; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)3; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp0;
					
				}
				tmp0[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp0 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp1 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)3,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp1 at (396,1-396,44) */
uint64_t __tmp_in_tmp1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
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

auto tmp3 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)16 );
/* Variable to read the clear value corresponding to the input variable tmp3 at (402,1-402,45) */
uint64_t __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)16; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp3;
					
				}
				tmp3[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp3 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp4 = make_vector<uint64_t>( (int32_t)16 );
/* Variable to read the clear value corresponding to the input variable tmp4 at (405,1-405,35) */
uint64_t __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)16; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp4;
		
	}
	tmp4[i0] = (role == CLIENT) ? __tmp_in_tmp4 : 0;
	
}

auto tmp5 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp5 at (408,1-408,45) */
uint64_t __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp5;
					
				}
				tmp5[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp5 : 0;
				
			}
			
		}
		
	}
	
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

auto tmp7 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)16,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp7 at (414,1-414,45) */
uint64_t __tmp_in_tmp7;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp7;
					
				}
				tmp7[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp7 : 0;
				
			}
			
		}
		
	}
	
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

auto tmp9 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)16 );
/* Variable to read the clear value corresponding to the input variable tmp9 at (420,1-420,46) */
uint64_t __tmp_in_tmp9;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)128; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)16; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp9;
					
				}
				tmp9[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp9 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp10 = make_vector<uint64_t>( (int32_t)16 );
/* Variable to read the clear value corresponding to the input variable tmp10 at (423,1-423,36) */
uint64_t __tmp_in_tmp10;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)16; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp10;
		
	}
	tmp10[i0] = (role == CLIENT) ? __tmp_in_tmp10 : 0;
	
}

auto tmp11 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp11 at (426,1-426,46) */
uint64_t __tmp_in_tmp11;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp11;
					
				}
				tmp11[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp11 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp12 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp12 at (429,1-429,36) */
uint64_t __tmp_in_tmp12;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp12;
		
	}
	tmp12[i0] = (role == CLIENT) ? __tmp_in_tmp12 : 0;
	
}

auto tmp13 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)16,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp13 at (432,1-432,46) */
uint64_t __tmp_in_tmp13;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)16; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp13;
					
				}
				tmp13[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp13 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp14 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp14 at (435,1-435,36) */
uint64_t __tmp_in_tmp14;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp14;
		
	}
	tmp14[i0] = (role == CLIENT) ? __tmp_in_tmp14 : 0;
	
}

auto tmp15 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp15 at (438,1-438,47) */
uint64_t __tmp_in_tmp15;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
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

auto tmp16 = make_vector<uint64_t>( (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp16 at (441,1-441,36) */
uint64_t __tmp_in_tmp16;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)32; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp16;
		
	}
	tmp16[i0] = (role == CLIENT) ? __tmp_in_tmp16 : 0;
	
}

auto tmp17 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp17 at (444,1-444,47) */
uint64_t __tmp_in_tmp17;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp17;
					
				}
				tmp17[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp17 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp18 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp18 at (447,1-447,37) */
uint64_t __tmp_in_tmp18;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp18;
		
	}
	tmp18[i0] = (role == CLIENT) ? __tmp_in_tmp18 : 0;
	
}

auto tmp19 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp19 at (450,1-450,47) */
uint64_t __tmp_in_tmp19;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp19;
					
				}
				tmp19[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp19 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp20 = make_vector<uint64_t>( (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp20 at (453,1-453,37) */
uint64_t __tmp_in_tmp20;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)128; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp20;
		
	}
	tmp20[i0] = (role == CLIENT) ? __tmp_in_tmp20 : 0;
	
}

auto tmp21 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp21 at (456,1-456,47) */
uint64_t __tmp_in_tmp21;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)32; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp21;
					
				}
				tmp21[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp21 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp22 = make_vector<uint64_t>( (int32_t)32 );
/* Variable to read the clear value corresponding to the input variable tmp22 at (459,1-459,36) */
uint64_t __tmp_in_tmp22;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)32; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp22;
		
	}
	tmp22[i0] = (role == CLIENT) ? __tmp_in_tmp22 : 0;
	
}

auto tmp23 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp23 at (462,1-462,47) */
uint64_t __tmp_in_tmp23;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp23;
					
				}
				tmp23[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp23 : 0;
				
			}
			
		}
		
	}
	
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

auto tmp25 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)128 );
/* Variable to read the clear value corresponding to the input variable tmp25 at (468,1-468,47) */
uint64_t __tmp_in_tmp25;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)32; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)128; i3++){
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

auto tmp27 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)48 );
/* Variable to read the clear value corresponding to the input variable tmp27 at (474,1-474,47) */
uint64_t __tmp_in_tmp27;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)256; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)48; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp27;
					
				}
				tmp27[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp27 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp28 = make_vector<uint64_t>( (int32_t)48 );
/* Variable to read the clear value corresponding to the input variable tmp28 at (477,1-477,36) */
uint64_t __tmp_in_tmp28;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)48; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp28;
		
	}
	tmp28[i0] = (role == CLIENT) ? __tmp_in_tmp28 : 0;
	
}

auto tmp29 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp29 at (480,1-480,47) */
uint64_t __tmp_in_tmp29;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp29;
					
				}
				tmp29[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp29 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp30 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp30 at (483,1-483,37) */
uint64_t __tmp_in_tmp30;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp30;
		
	}
	tmp30[i0] = (role == CLIENT) ? __tmp_in_tmp30 : 0;
	
}

auto tmp31 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)48,  (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp31 at (486,1-486,47) */
uint64_t __tmp_in_tmp31;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp31;
					
				}
				tmp31[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp31 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp32 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp32 at (489,1-489,37) */
uint64_t __tmp_in_tmp32;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp32;
		
	}
	tmp32[i0] = (role == CLIENT) ? __tmp_in_tmp32 : 0;
	
}

auto tmp33 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)48 );
/* Variable to read the clear value corresponding to the input variable tmp33 at (492,1-492,47) */
uint64_t __tmp_in_tmp33;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)48; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp33;
					
				}
				tmp33[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp33 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp34 = make_vector<uint64_t>( (int32_t)48 );
/* Variable to read the clear value corresponding to the input variable tmp34 at (495,1-495,36) */
uint64_t __tmp_in_tmp34;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)48; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp34;
		
	}
	tmp34[i0] = (role == CLIENT) ? __tmp_in_tmp34 : 0;
	
}

auto tmp35 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp35 at (498,1-498,47) */
uint64_t __tmp_in_tmp35;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp35;
					
				}
				tmp35[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp35 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp36 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp36 at (501,1-501,37) */
uint64_t __tmp_in_tmp36;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp36;
		
	}
	tmp36[i0] = (role == CLIENT) ? __tmp_in_tmp36 : 0;
	
}

auto tmp37 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)48,  (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp37 at (504,1-504,47) */
uint64_t __tmp_in_tmp37;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)48; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)192; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp37;
					
				}
				tmp37[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp37 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp38 = make_vector<uint64_t>( (int32_t)192 );
/* Variable to read the clear value corresponding to the input variable tmp38 at (507,1-507,37) */
uint64_t __tmp_in_tmp38;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)192; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp38;
		
	}
	tmp38[i0] = (role == CLIENT) ? __tmp_in_tmp38 : 0;
	
}

auto tmp39 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp39 at (510,1-510,47) */
uint64_t __tmp_in_tmp39;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)384; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp39;
					
				}
				tmp39[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp39 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp40 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp40 at (513,1-513,36) */
uint64_t __tmp_in_tmp40;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp40;
		
	}
	tmp40[i0] = (role == CLIENT) ? __tmp_in_tmp40 : 0;
	
}

auto tmp41 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp41 at (516,1-516,47) */
uint64_t __tmp_in_tmp41;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp41;
					
				}
				tmp41[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp41 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp42 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp42 at (519,1-519,37) */
uint64_t __tmp_in_tmp42;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp42;
		
	}
	tmp42[i0] = (role == CLIENT) ? __tmp_in_tmp42 : 0;
	
}

auto tmp43 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp43 at (522,1-522,47) */
uint64_t __tmp_in_tmp43;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp43;
					
				}
				tmp43[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp43 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp44 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp44 at (525,1-525,37) */
uint64_t __tmp_in_tmp44;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp44;
		
	}
	tmp44[i0] = (role == CLIENT) ? __tmp_in_tmp44 : 0;
	
}

auto tmp45 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp45 at (528,1-528,47) */
uint64_t __tmp_in_tmp45;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)64; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp45;
					
				}
				tmp45[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp45 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp46 = make_vector<uint64_t>( (int32_t)64 );
/* Variable to read the clear value corresponding to the input variable tmp46 at (531,1-531,36) */
uint64_t __tmp_in_tmp46;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)64; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp46;
		
	}
	tmp46[i0] = (role == CLIENT) ? __tmp_in_tmp46 : 0;
	
}

auto tmp47 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp47 at (534,1-534,47) */
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

auto tmp48 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp48 at (537,1-537,37) */
uint64_t __tmp_in_tmp48;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp48;
		
	}
	tmp48[i0] = (role == CLIENT) ? __tmp_in_tmp48 : 0;
	
}

auto tmp49 = make_vector<uint64_t>( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp49 at (540,1-540,47) */
uint64_t __tmp_in_tmp49;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)3; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)3; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)64; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)256; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp49;
					
				}
				tmp49[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp49 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp50 = make_vector<uint64_t>( (int32_t)256 );
/* Variable to read the clear value corresponding to the input variable tmp50 at (543,1-543,37) */
uint64_t __tmp_in_tmp50;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)256; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp50;
		
	}
	tmp50[i0] = (role == CLIENT) ? __tmp_in_tmp50 : 0;
	
}

auto tmp51 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1000 );
/* Variable to read the clear value corresponding to the input variable tmp51 at (546,1-546,49) */
uint64_t __tmp_in_tmp51;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)512; i2++){
			for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)1000; i3++){
				if ((role == CLIENT)) {
					cin >> __tmp_in_tmp51;
					
				}
				tmp51[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp51 : 0;
				
			}
			
		}
		
	}
	
}

auto tmp52 = make_vector<uint64_t>( (int32_t)1000 );
/* Variable to read the clear value corresponding to the input variable tmp52 at (549,1-549,38) */
uint64_t __tmp_in_tmp52;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1000; i0++){
	if ((role == CLIENT)) {
		cin >> __tmp_in_tmp52;
		
	}
	tmp52[i0] = (role == CLIENT) ? __tmp_in_tmp52 : 0;
	
}

leave_time();
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("Starting main protocol");
start_m();
touch_time();

auto tmp53 = make_vector<uint64_t>( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)227,  (int32_t)227,  (int32_t)3,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2, tmp0, tmp1,  (int32_t)12, tmp53 );
ClearMemSecret4( (int32_t)1,  (int32_t)227,  (int32_t)227,  (int32_t)3, tmp0 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)3,  (int32_t)64, tmp1 );

auto tmp56 = make_vector<uint64_t>( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64 );
MatAddBroadCast4( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64, tmp53, tmp2, tmp56 );
ClearMemSecret4( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64, tmp53 );
ClearMemSecret1( (int32_t)64, tmp2 );

auto tmp59 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
MaxPool( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64, tmp56, tmp59 );
ClearMemSecret4( (int32_t)1,  (int32_t)113,  (int32_t)113,  (int32_t)64, tmp56 );

auto tmp61 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp59, tmp61 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp59 );

auto tmp63 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp61, tmp3,  (int32_t)12, tmp63 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp61 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)16, tmp3 );

auto tmp66 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16 );
MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp63, tmp4, tmp66 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp63 );
ClearMemSecret1( (int32_t)16, tmp4 );

auto tmp69 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp66, tmp69 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp66 );

auto tmp71 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp69, tmp5,  (int32_t)12, tmp71 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)64, tmp5 );

auto tmp73 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp71, tmp6, tmp73 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp71 );
ClearMemSecret1( (int32_t)64, tmp6 );

auto tmp76 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp73, tmp76 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp73 );

auto tmp78 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp69, tmp7,  (int32_t)12, tmp78 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)16,  (int32_t)64, tmp7 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp69 );

auto tmp81 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp78, tmp8, tmp81 );
ClearMemSecret1( (int32_t)64, tmp8 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp78 );

auto tmp84 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp81, tmp84 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp81 );

auto tmp86 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp76,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp84,  (int32_t)3, tmp86 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp76 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp84 );

auto tmp90 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp86, tmp9,  (int32_t)12, tmp90 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)16, tmp9 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp86 );

auto tmp93 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16 );
MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp90, tmp10, tmp93 );
ClearMemSecret1( (int32_t)16, tmp10 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp90 );

auto tmp96 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp93, tmp96 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp93 );

auto tmp98 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp96, tmp11,  (int32_t)12, tmp98 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)16,  (int32_t)64, tmp11 );

auto tmp100 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp98, tmp12, tmp100 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp98 );
ClearMemSecret1( (int32_t)64, tmp12 );

auto tmp103 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp100, tmp103 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp100 );

auto tmp105 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16,  (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp96, tmp13,  (int32_t)12, tmp105 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)16, tmp96 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)16,  (int32_t)64, tmp13 );

auto tmp108 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
MatAddBroadCast4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp105, tmp14, tmp108 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp105 );
ClearMemSecret1( (int32_t)64, tmp14 );

auto tmp111 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp108, tmp111 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp108 );

auto tmp113 = make_vector<uint64_t>( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128 );
Concat2T444( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp103,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp111,  (int32_t)3, tmp113 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp111 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)64, tmp103 );
ClearMemPublic( (int32_t)3 );

auto tmp117 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
MaxPool( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp113, tmp117 );
ClearMemSecret4( (int32_t)1,  (int32_t)56,  (int32_t)56,  (int32_t)128, tmp113 );

auto tmp119 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp117, tmp15,  (int32_t)12, tmp119 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)32, tmp15 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp117 );

auto tmp122 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32 );
MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp119, tmp16, tmp122 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp119 );
ClearMemSecret1( (int32_t)32, tmp16 );

auto tmp125 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32 );
Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp122, tmp125 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp122 );

auto tmp127 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp125, tmp17,  (int32_t)12, tmp127 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)128, tmp17 );

auto tmp129 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp127, tmp18, tmp129 );
ClearMemSecret1( (int32_t)128, tmp18 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp127 );

auto tmp132 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp129, tmp132 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp129 );

auto tmp134 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp125, tmp19,  (int32_t)12, tmp134 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)128, tmp19 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp125 );

auto tmp137 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp134, tmp20, tmp137 );
ClearMemSecret1( (int32_t)128, tmp20 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp134 );

auto tmp140 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp137, tmp140 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp137 );

auto tmp142 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256 );
Concat2T444( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp132,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp140,  (int32_t)3, tmp142 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp140 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp132 );

auto tmp146 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32 );
Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp142, tmp21,  (int32_t)12, tmp146 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)32, tmp21 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256, tmp142 );

auto tmp149 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32 );
MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp146, tmp22, tmp149 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp146 );
ClearMemSecret1( (int32_t)32, tmp22 );

auto tmp152 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32 );
Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp149, tmp152 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp149 );

auto tmp154 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)1,  (int32_t)1,  (int32_t)128,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp152, tmp23,  (int32_t)12, tmp154 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)32,  (int32_t)128, tmp23 );

auto tmp156 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp154, tmp24, tmp156 );
ClearMemSecret1( (int32_t)128, tmp24 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp154 );

auto tmp159 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp156, tmp159 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp156 );

auto tmp161 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Conv2DCSF( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32,  (int32_t)3,  (int32_t)3,  (int32_t)128,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp152, tmp25,  (int32_t)12, tmp161 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)32, tmp152 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)32,  (int32_t)128, tmp25 );

auto tmp164 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
MatAddBroadCast4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp161, tmp26, tmp164 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp161 );
ClearMemSecret1( (int32_t)128, tmp26 );

auto tmp167 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128 );
Relu4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp164, tmp167 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp164 );

auto tmp169 = make_vector<uint64_t>( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256 );
Concat2T444( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp159,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp167,  (int32_t)3, tmp169 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp159 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)128, tmp167 );
ClearMemPublic( (int32_t)3 );

auto tmp173 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
MaxPool( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256,  (int32_t)3,  (int32_t)3,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256, tmp169, tmp173 );
ClearMemSecret4( (int32_t)1,  (int32_t)27,  (int32_t)27,  (int32_t)256, tmp169 );

auto tmp175 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp173, tmp27,  (int32_t)12, tmp175 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp173 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)48, tmp27 );

auto tmp178 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp175, tmp28, tmp178 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp175 );
ClearMemSecret1( (int32_t)48, tmp28 );

auto tmp181 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp178, tmp181 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp178 );

auto tmp183 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp181, tmp29,  (int32_t)12, tmp183 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)192, tmp29 );

auto tmp185 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp183, tmp30, tmp185 );
ClearMemSecret1( (int32_t)192, tmp30 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp183 );

auto tmp188 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp185, tmp188 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp185 );

auto tmp190 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)3,  (int32_t)3,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp181, tmp31,  (int32_t)12, tmp190 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp181 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)48,  (int32_t)192, tmp31 );

auto tmp193 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp190, tmp32, tmp193 );
ClearMemSecret1( (int32_t)192, tmp32 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp190 );

auto tmp196 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp193, tmp196 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp193 );

auto tmp198 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384 );
Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp188,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp196,  (int32_t)3, tmp198 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp188 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp196 );
ClearMemPublic( (int32_t)3 );

auto tmp202 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp198, tmp33,  (int32_t)12, tmp202 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)48, tmp33 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384, tmp198 );

auto tmp205 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp202, tmp34, tmp205 );
ClearMemSecret1( (int32_t)48, tmp34 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp202 );

auto tmp208 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp205, tmp208 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp205 );

auto tmp210 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)1,  (int32_t)1,  (int32_t)192,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp208, tmp35,  (int32_t)12, tmp210 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)48,  (int32_t)192, tmp35 );

auto tmp212 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp210, tmp36, tmp212 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp210 );
ClearMemSecret1( (int32_t)192, tmp36 );

auto tmp215 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp212, tmp215 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp212 );

auto tmp217 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48,  (int32_t)3,  (int32_t)3,  (int32_t)192,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp208, tmp37,  (int32_t)12, tmp217 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)48,  (int32_t)192, tmp37 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)48, tmp208 );

auto tmp220 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp217, tmp38, tmp220 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp217 );
ClearMemSecret1( (int32_t)192, tmp38 );

auto tmp223 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp220, tmp223 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp220 );

auto tmp225 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384 );
Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp215,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp223,  (int32_t)3, tmp225 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp215 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)192, tmp223 );
ClearMemPublic( (int32_t)3 );

auto tmp229 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp225, tmp39,  (int32_t)12, tmp229 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)384, tmp225 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)384,  (int32_t)64, tmp39 );

auto tmp232 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp229, tmp40, tmp232 );
ClearMemSecret1( (int32_t)64, tmp40 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp229 );

auto tmp235 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp232, tmp235 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp232 );

auto tmp237 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp235, tmp41,  (int32_t)12, tmp237 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp41 );

auto tmp239 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp237, tmp42, tmp239 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp237 );
ClearMemSecret1( (int32_t)256, tmp42 );

auto tmp242 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp239, tmp242 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp239 );

auto tmp244 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp235, tmp43,  (int32_t)12, tmp244 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp235 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)256, tmp43 );

auto tmp247 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp244, tmp44, tmp247 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp244 );
ClearMemSecret1( (int32_t)256, tmp44 );

auto tmp250 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp247, tmp250 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp247 );

auto tmp252 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512 );
Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp242,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp250,  (int32_t)3, tmp252 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp242 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp250 );

auto tmp256 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp252, tmp45,  (int32_t)12, tmp256 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512, tmp252 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)64, tmp45 );

auto tmp259 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp256, tmp46, tmp259 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp256 );
ClearMemSecret1( (int32_t)64, tmp46 );

auto tmp262 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp259, tmp262 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp259 );

auto tmp264 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)1,  (int32_t)1,  (int32_t)256,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp262, tmp47,  (int32_t)12, tmp264 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)64,  (int32_t)256, tmp47 );

auto tmp266 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp264, tmp48, tmp266 );
ClearMemSecret1( (int32_t)256, tmp48 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp264 );

auto tmp269 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp266, tmp269 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp266 );

auto tmp271 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64,  (int32_t)3,  (int32_t)3,  (int32_t)256,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1, tmp262, tmp49,  (int32_t)12, tmp271 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)64, tmp262 );
ClearMemSecret4( (int32_t)3,  (int32_t)3,  (int32_t)64,  (int32_t)256, tmp49 );

auto tmp274 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp271, tmp50, tmp274 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp271 );
ClearMemSecret1( (int32_t)256, tmp50 );

auto tmp277 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp274, tmp277 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp274 );

auto tmp279 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512 );
Concat2T444( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp269,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp277,  (int32_t)3, tmp279 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp277 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)256, tmp269 );

auto tmp283 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000 );
Conv2DCSF( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512,  (int32_t)1,  (int32_t)1,  (int32_t)1000,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp279, tmp51,  (int32_t)12, tmp283 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)512, tmp279 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)512,  (int32_t)1000, tmp51 );

auto tmp286 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000 );
MatAddBroadCast4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp283, tmp52, tmp286 );
ClearMemSecret1( (int32_t)1000, tmp52 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp283 );

auto tmp289 = make_vector<uint64_t>( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000 );
Relu4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp286, tmp289 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp286 );

auto tmp291 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000 );
AvgPool( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000,  (int32_t)13,  (int32_t)13,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp289, tmp291 );
ClearMemSecret4( (int32_t)1,  (int32_t)13,  (int32_t)13,  (int32_t)1000, tmp289 );

auto tmp293 = make_vector<uint64_t>( (int32_t)1,  (int32_t)1,  (int32_t)1 );
ArgMax3( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000, tmp291,  (int32_t)3, tmp293 );
ClearMemPublic( (int32_t)3 );
ClearMemSecret4( (int32_t)1,  (int32_t)1,  (int32_t)1,  (int32_t)1000, tmp291 );

for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
	for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)1; i1++){
		for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1; i2++){
			print_integer(funcReconstruct2PCCons(tmp293[i0][i1][i2], 1));
			
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
