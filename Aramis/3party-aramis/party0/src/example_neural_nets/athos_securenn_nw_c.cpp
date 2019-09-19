#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include "res_net_mem_opti.h"
//#include<fstream>
#include "EzPCFunctionalities.h"
// SGX instream
#include "../utils_sgx_port/utils_input_sgx.h"

#ifdef ATHOS_NW_C

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

auto tmp0 = make_vector<uint64_t>( (int32_t)1,  (int32_t)784);
/* Variable to read the clear value corresponding to the input variable tmp0 at (393,1-393,39) */
uint64_t __tmp_in_tmp0;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)784; i1++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp0;
}
tmp0[i0][i1] = (role == CLIENT) ? __tmp_in_tmp0 : 0;
}
}

auto tmp1 = make_vector<uint64_t>( (int32_t)5,  (int32_t)5,  (int32_t)1,  (int32_t)20);
/* Variable to read the clear value corresponding to the input variable tmp1 at (396,1-396,44) */
uint64_t __tmp_in_tmp1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)5; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)5; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)1; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)20; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp1;
}
tmp1[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp1 : 0;
}
}
}
}

auto tmp2 = make_vector<uint64_t>( (int32_t)5,  (int32_t)5,  (int32_t)20,  (int32_t)50);
/* Variable to read the clear value corresponding to the input variable tmp2 at (399,1-399,45) */
uint64_t __tmp_in_tmp2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)5; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)5; i1++){
for (uint32_t i2 =  (uint32_t)0; i2 <  (int32_t)20; i2++){
for (uint32_t i3 =  (uint32_t)0; i3 <  (int32_t)50; i3++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp2;
}
tmp2[i0][i1][i2][i3] = (role == CLIENT) ? __tmp_in_tmp2 : 0;
}
}
}
}

auto tmp3 = make_vector<uint64_t>( (int32_t)800,  (int32_t)500);
/* Variable to read the clear value corresponding to the input variable tmp3 at (402,1-402,41) */
uint64_t __tmp_in_tmp3;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)800; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)500; i1++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp3;
}
tmp3[i0][i1] = (role == CLIENT) ? __tmp_in_tmp3 : 0;
}
}

auto tmp4 = make_vector<uint64_t>( (int32_t)500);
/* Variable to read the clear value corresponding to the input variable tmp4 at (405,1-405,36) */
uint64_t __tmp_in_tmp4;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)500; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp4;
}
tmp4[i0] = (role == CLIENT) ? __tmp_in_tmp4 : 0;
}

auto tmp5 = make_vector<uint64_t>( (int32_t)500,  (int32_t)10);
/* Variable to read the clear value corresponding to the input variable tmp5 at (408,1-408,40) */
uint64_t __tmp_in_tmp5;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)500; i0++){
for (uint32_t i1 =  (uint32_t)0; i1 <  (int32_t)10; i1++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp5;
}
tmp5[i0][i1] = (role == CLIENT) ? __tmp_in_tmp5 : 0;
}
}

auto tmp6 = make_vector<uint64_t>( (int32_t)10);
/* Variable to read the clear value corresponding to the input variable tmp6 at (411,1-411,35) */
uint64_t __tmp_in_tmp6;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)10; i0++){
if ((role == CLIENT)) {
cin >> __tmp_in_tmp6;
}
tmp6[i0] = (role == CLIENT) ? __tmp_in_tmp6 : 0;
}
//Main Point
leave_time();
//cout<<"Starting 2nd syncronize .. "<<endl;
synchronize(2000000); 
//cout<<"Syncronized .. now starting actual execution at "<<getCurrentTime()<<endl;
print_string("Starting main protocol");
start_m();
touch_time();
auto tmp7 = make_vector<int32_t>( (int32_t)4);
tmp7[ (int64_t)0] =  (int32_t)-1;
tmp7[ (int64_t)1] =  (int32_t)28;
tmp7[ (int64_t)2] =  (int32_t)28;
tmp7[ (int64_t)3] =  (int32_t)1;

auto tmp8 = make_vector<uint64_t>( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)1);

int64_t i5;

int64_t i4;

int64_t i3;

int64_t i2;

int64_t i1;

int64_t i0;
i0 =  (int64_t)0;
i1 =  (int64_t)0;
for (uint32_t i2 =  (int32_t)0; i2 <  (int32_t)1; i2++){
for (uint32_t i3 =  (int32_t)0; i3 <  (int32_t)28; i3++){
for (uint32_t i4 =  (int32_t)0; i4 <  (int32_t)28; i4++){
for (uint32_t i5 =  (int32_t)0; i5 <  (int32_t)1; i5++){
tmp8[i2][i3][i4][i5] = tmp0[i0][i1];
i1 = (i1 +  (int64_t)1);
if ((i1 ==  (int64_t)784)) {
i1 =  (int64_t)0;
i0 = (i0 +  (int64_t)1);
}
}
}
}
}
ClearMemSecret2( (int32_t)1,  (int32_t)784, tmp0);

auto tmp10 = make_vector<uint64_t>( (int32_t)1,  (int32_t)24,  (int32_t)24,  (int32_t)20);
Conv2DCSF( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)1,  (int32_t)5,  (int32_t)5,  (int32_t)20,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp8, tmp1,  (int32_t)12, tmp10);
ClearMemSecret4( (int32_t)5,  (int32_t)5,  (int32_t)1,  (int32_t)20, tmp1);
ClearMemSecret4( (int32_t)1,  (int32_t)28,  (int32_t)28,  (int32_t)1, tmp8);

auto tmp13 = make_vector<uint64_t>( (int32_t)1,  (int32_t)12,  (int32_t)12,  (int32_t)20);
MaxPool( (int32_t)1,  (int32_t)12,  (int32_t)12,  (int32_t)20,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)24,  (int32_t)24,  (int32_t)20, tmp10, tmp13);
ClearMemSecret4( (int32_t)1,  (int32_t)24,  (int32_t)24,  (int32_t)20, tmp10);

auto tmp15 = make_vector<uint64_t>( (int32_t)1,  (int32_t)12,  (int32_t)12,  (int32_t)20);
Relu4( (int32_t)1,  (int32_t)12,  (int32_t)12,  (int32_t)20, tmp13, tmp15);
ClearMemSecret4( (int32_t)1,  (int32_t)12,  (int32_t)12,  (int32_t)20, tmp13);

auto tmp17 = make_vector<uint64_t>( (int32_t)1,  (int32_t)8,  (int32_t)8,  (int32_t)50);
Conv2DCSF( (int32_t)1,  (int32_t)12,  (int32_t)12,  (int32_t)20,  (int32_t)5,  (int32_t)5,  (int32_t)50,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)1,  (int32_t)1, tmp15, tmp2,  (int32_t)12, tmp17);
ClearMemSecret4( (int32_t)1,  (int32_t)12,  (int32_t)12,  (int32_t)20, tmp15);
ClearMemSecret4( (int32_t)5,  (int32_t)5,  (int32_t)20,  (int32_t)50, tmp2);

auto tmp20 = make_vector<uint64_t>( (int32_t)1,  (int32_t)4,  (int32_t)4,  (int32_t)50);
MaxPool( (int32_t)1,  (int32_t)4,  (int32_t)4,  (int32_t)50,  (int32_t)2,  (int32_t)2,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)0,  (int32_t)2,  (int32_t)2,  (int32_t)1,  (int32_t)8,  (int32_t)8,  (int32_t)50, tmp17, tmp20);
ClearMemSecret4( (int32_t)1,  (int32_t)8,  (int32_t)8,  (int32_t)50, tmp17);

auto tmp22 = make_vector<uint64_t>( (int32_t)1,  (int32_t)4,  (int32_t)4,  (int32_t)50);
Relu4( (int32_t)1,  (int32_t)4,  (int32_t)4,  (int32_t)50, tmp20, tmp22);
ClearMemSecret4( (int32_t)1,  (int32_t)4,  (int32_t)4,  (int32_t)50, tmp20);

auto tmp24 = make_vector<int32_t>( (int32_t)2);
tmp24[ (int64_t)0] =  (int32_t)-1;
tmp24[ (int64_t)1] =  (int32_t)0;

auto tmp25 = make_vector<uint64_t>( (int32_t)1,  (int32_t)800);

int64_t i11;

int64_t i10;

int64_t i9;

int64_t i8;

int64_t i7;

int64_t i6;
i6 =  (int64_t)0;
i7 =  (int64_t)0;
i8 =  (int64_t)0;
i9 =  (int64_t)0;
for (uint32_t i10 =  (int32_t)0; i10 <  (int32_t)1; i10++){
for (uint32_t i11 =  (int32_t)0; i11 <  (int32_t)800; i11++){
tmp25[i10][i11] = tmp22[i6][i7][i8][i9];
i9 = (i9 +  (int64_t)1);
if ((i9 ==  (int64_t)50)) {
i9 =  (int64_t)0;
i8 = (i8 +  (int64_t)1);
if ((i8 ==  (int64_t)4)) {
i8 =  (int64_t)0;
i7 = (i7 +  (int64_t)1);
if ((i7 ==  (int64_t)4)) {
i7 =  (int64_t)0;
i6 = (i6 +  (int64_t)1);
}
}
}
}
}
ClearMemSecret4( (int32_t)1,  (int32_t)4,  (int32_t)4,  (int32_t)50, tmp22);

auto tmp27 = make_vector<uint64_t>( (int32_t)1,  (int32_t)500);
MatMulCSF2D( (int32_t)1,  (int32_t)800,  (int32_t)500, tmp25, tmp3, tmp27,  (int64_t)12);
ClearMemSecret2( (int32_t)1,  (int32_t)800, tmp25);
ClearMemSecret2( (int32_t)800,  (int32_t)500, tmp3);

auto tmp30 = make_vector<uint64_t>( (int32_t)1,  (int32_t)500);
MatAddBroadCast2( (int32_t)1,  (int32_t)500, tmp27, tmp4, tmp30);
ClearMemSecret2( (int32_t)1,  (int32_t)500, tmp27);
ClearMemSecret1( (int32_t)500, tmp4);

auto tmp33 = make_vector<uint64_t>( (int32_t)1,  (int32_t)500);
Relu2( (int32_t)1,  (int32_t)500, tmp30, tmp33);
ClearMemSecret2( (int32_t)1,  (int32_t)500, tmp30);

auto tmp35 = make_vector<uint64_t>( (int32_t)1,  (int32_t)10);
MatMulCSF2D( (int32_t)1,  (int32_t)500,  (int32_t)10, tmp33, tmp5, tmp35,  (int64_t)12);
ClearMemSecret2( (int32_t)1,  (int32_t)500, tmp33);
ClearMemSecret2( (int32_t)500,  (int32_t)10, tmp5);

auto tmp38 = make_vector<uint64_t>( (int32_t)1,  (int32_t)10);
MatAddBroadCast2( (int32_t)1,  (int32_t)10, tmp35, tmp6, tmp38);
ClearMemSecret1( (int32_t)10, tmp6);
ClearMemSecret2( (int32_t)1,  (int32_t)10, tmp35);

auto tmp41 = make_vector<uint64_t>( (int32_t)1);
ArgMax1( (int32_t)1,  (int32_t)1,  (int32_t)10, tmp38,  (int32_t)1, tmp41);
ClearMemSecret2( (int32_t)1,  (int32_t)10, tmp38);
ClearMemPublic( (int32_t)1);

for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)1; i0++){
print_integer(funcReconstruct2PCCons(tmp41[i0], 1));
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
