'''
Authors: Mayank Rathee.
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
'''

porthos_file = open("../../../Porthos/src/example_neural_nets/mainResNet50.cpp", 'r');

porthos_code = porthos_file.readlines()

aramis_code_prelude = "#include<vector>\n#include<math.h>\n#include<cstdlib>\n#include<iostream>\n#include \"main.h\"\n#include \"EzPCFunctionalities.h\"\n#include \"../utils_sgx_port/utils_input_sgx.h\"\n#ifdef INC_NN\nsgx_instream cin = sgx_instream();\nusing namespace std;\n"

aramis_code_prelude += """AESObject* aes_common;\n
AESObject* aes_indep;\n
AESObject* m_g_aes_indep_p0 = new AESObject(\"KeyA\");\n
AESObject* m_g_aes_common_p0 = new AESObject(\"KeyAB\");\n
AESObject* m_g_aes_indep_p1 = new AESObject(\"KeyB\");\n
AESObject* m_g_aes_common_p1 = new AESObject(\"KeyAB\");\n
AESObject* m_g_aes_indep_p2 = new AESObject(\"KeyC\");\n
AESObject* m_g_aes_common_p2 = new AESObject(\"KeyCD\");\n
AESObject* aes_a_1 = new AESObject(\"KeyD\");\n
AESObject* aes_a_2 = new AESObject(\"KeyD\");\n
AESObject* aes_b_1 = new AESObject(\"KeyD\");\n
AESObject* aes_b_2 = new AESObject(\"KeyD\");\n
AESObject* aes_c_1 = new AESObject(\"KeyD\");\n
AESObject* aes_share_conv_bit_shares_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_share_conv_bit_shares_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_share_conv_shares_mod_odd_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_share_conv_shares_mod_odd_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_comp_msb_shares_lsb_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_comp_msb_shares_lsb_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* aes_conv_opti_a_1 = new AESObject(\"KeyD\");\n
AESObject* aes_conv_opti_a_2 = new AESObject(\"KeyD\");\n
AESObject* aes_conv_opti_b_1 = new AESObject(\"KeyD\");\n
AESObject* aes_conv_opti_b_2 = new AESObject(\"KeyD\");\n
AESObject* aes_conv_opti_c_1 = new AESObject(\"KeyD\");\n
AESObject* threaded_aes_indep[NO_CORES];\n
AESObject* threaded_aes_common[NO_CORES];\n
AESObject* threaded_aes_a_1[NO_CORES];\n
AESObject* threaded_aes_a_2[NO_CORES];\n
AESObject* threaded_aes_b_1[NO_CORES];\n
AESObject* threaded_aes_b_2[NO_CORES];\n
AESObject* threaded_aes_c_1[NO_CORES];\n
AESObject* threaded_aes_share_conv_bit_shares_p0_p2[NO_CORES];\n
AESObject* threaded_aes_share_conv_bit_shares_p1_p2[NO_CORES];\n
AESObject* threaded_aes_share_conv_shares_mod_odd_p0_p2[NO_CORES];\n
AESObject* threaded_aes_share_conv_shares_mod_odd_p1_p2[NO_CORES];\n
AESObject* threaded_aes_comp_msb_shares_lsb_p0_p2[NO_CORES];\n
AESObject* threaded_aes_comp_msb_shares_lsb_p1_p2[NO_CORES];\n
AESObject* threaded_aes_comp_msb_shares_bit_vec_p0_p2[NO_CORES];\n
AESObject* threaded_aes_comp_msb_shares_bit_vec_p1_p2[NO_CORES];\n
AESObject* threaded_aes_conv_opti_a_1[NO_CORES];\n
AESObject* threaded_aes_conv_opti_a_2[NO_CORES];\n
AESObject* threaded_aes_conv_opti_b_1[NO_CORES];\n
AESObject* threaded_aes_conv_opti_b_2[NO_CORES];\n
AESObject* threaded_aes_conv_opti_c_1[NO_CORES];\n
AESObject* a_m_g_aes_indep_p0 = new AESObject(\"KeyA\");\n
AESObject* a_m_g_aes_common_p0 = new AESObject(\"KeyAB\");\n
AESObject* a_m_g_aes_indep_p1 = new AESObject(\"KeyB\");\n
AESObject* a_m_g_aes_common_p1 = new AESObject(\"KeyAB\");\n
AESObject* a_m_g_aes_indep_p2 = new AESObject(\"KeyC\");\n
AESObject* a_m_g_aes_common_p2 = new AESObject(\"KeyCD\");\n
AESObject* a_aes_a_1 = new AESObject(\"KeyD\");\n
AESObject* a_aes_a_2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_b_1 = new AESObject(\"KeyD\");\n
AESObject* a_aes_b_2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_c_1 = new AESObject(\"KeyD\");\n
AESObject* a_aes_share_conv_bit_shares_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_share_conv_bit_shares_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_comp_msb_shares_lsb_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_comp_msb_shares_lsb_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_conv_opti_a_1 = new AESObject(\"KeyD\");\n
AESObject* a_aes_conv_opti_a_2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_conv_opti_b_1 = new AESObject(\"KeyD\");\n
AESObject* a_aes_conv_opti_b_2 = new AESObject(\"KeyD\");\n
AESObject* a_aes_conv_opti_c_1 = new AESObject(\"KeyD\");\n
AESObject* b_m_g_aes_indep_p0 = new AESObject(\"KeyA\");\n
AESObject* b_m_g_aes_common_p0 = new AESObject(\"KeyAB\");\n
AESObject* b_m_g_aes_indep_p1 = new AESObject(\"KeyB\");\n
AESObject* b_m_g_aes_common_p1 = new AESObject(\"KeyAB\");\n
AESObject* b_m_g_aes_indep_p2 = new AESObject(\"KeyC\");\n
AESObject* b_m_g_aes_common_p2 = new AESObject(\"KeyCD\");\n
AESObject* b_aes_a_1 = new AESObject(\"KeyD\");\n
AESObject* b_aes_a_2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_b_1 = new AESObject(\"KeyD\");\n
AESObject* b_aes_b_2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_c_1 = new AESObject(\"KeyD\");\n
AESObject* b_aes_share_conv_bit_shares_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_share_conv_bit_shares_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_comp_msb_shares_lsb_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_comp_msb_shares_lsb_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_conv_opti_a_1 = new AESObject(\"KeyD\");\n
AESObject* b_aes_conv_opti_a_2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_conv_opti_b_1 = new AESObject(\"KeyD\");\n
AESObject* b_aes_conv_opti_b_2 = new AESObject(\"KeyD\");\n
AESObject* b_aes_conv_opti_c_1 = new AESObject(\"KeyD\");\n
AESObject* c_m_g_aes_indep_p0 = new AESObject(\"KeyA\");\n
AESObject* c_m_g_aes_common_p0 = new AESObject(\"KeyAB\");\n
AESObject* c_m_g_aes_indep_p1 = new AESObject(\"KeyB\");\n
AESObject* c_m_g_aes_common_p1 = new AESObject(\"KeyAB\");\n
AESObject* c_m_g_aes_indep_p2 = new AESObject(\"KeyC\");\n
AESObject* c_m_g_aes_common_p2 = new AESObject(\"KeyCD\");\n
AESObject* c_aes_a_1 = new AESObject(\"KeyD\");\n
AESObject* c_aes_a_2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_b_1 = new AESObject(\"KeyD\");\n
AESObject* c_aes_b_2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_c_1 = new AESObject(\"KeyD\");\n
AESObject* c_aes_share_conv_bit_shares_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_share_conv_bit_shares_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_comp_msb_shares_lsb_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_comp_msb_shares_lsb_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_conv_opti_a_1 = new AESObject(\"KeyD\");\n
AESObject* c_aes_conv_opti_a_2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_conv_opti_b_1 = new AESObject(\"KeyD\");\n
AESObject* c_aes_conv_opti_b_2 = new AESObject(\"KeyD\");\n
AESObject* c_aes_conv_opti_c_1 = new AESObject(\"KeyD\");\n
AESObject* d_m_g_aes_indep_p0 = new AESObject(\"KeyA\");\n
AESObject* d_m_g_aes_common_p0 = new AESObject(\"KeyAB\");\n
AESObject* d_m_g_aes_indep_p1 = new AESObject(\"KeyB\");\n
AESObject* d_m_g_aes_common_p1 = new AESObject(\"KeyAB\");\n
AESObject* d_m_g_aes_indep_p2 = new AESObject(\"KeyC\");\n
AESObject* d_m_g_aes_common_p2 = new AESObject(\"KeyCD\");\n
AESObject* d_aes_a_1 = new AESObject(\"KeyD\");\n
AESObject* d_aes_a_2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_b_1 = new AESObject(\"KeyD\");\n
AESObject* d_aes_b_2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_c_1 = new AESObject(\"KeyD\");\n
AESObject* d_aes_share_conv_bit_shares_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_share_conv_bit_shares_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_share_conv_shares_mod_odd_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_share_conv_shares_mod_odd_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_comp_msb_shares_lsb_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_comp_msb_shares_lsb_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_comp_msb_shares_bit_vec_p0_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_comp_msb_shares_bit_vec_p1_p2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_conv_opti_a_1 = new AESObject(\"KeyD\");\n
AESObject* d_aes_conv_opti_a_2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_conv_opti_b_1 = new AESObject(\"KeyD\");\n
AESObject* d_aes_conv_opti_b_2 = new AESObject(\"KeyD\");\n
AESObject* d_aes_conv_opti_c_1 = new AESObject(\"KeyD\");\n
ParallelAESObject* aes_parallel = new ParallelAESObject("");\n
int run_sequence = 0;\n"""

aramis_code_prelude += """extern int partyNum;
vector<uint64_t*> toFreeMemoryLaterArr;
int NUM_OF_PARTIES;"""

aramis_code_prelude += """uint32_t public_lrshift(uint32_t x, uint32_t y){
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
}"""

aramis_code_prelude += "\n#include \"ezpc.h\"\n"

aramis_code_prelude += """void MatAddBroadCast2(int32_t s1, int32_t s2, auto& A, auto& B, auto& outArr){
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
	
}"""

aramis_code_main = ""

aramis_code_main += """void main_aramis(int pnum)
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
"""

for i in range(479, len(porthos_code)):
    from_last = len(porthos_code)-i
    if(from_last >=32 and from_last <=35):
        continue
    if(from_last == 37):
        aramis_code_main += "print_integer((int64_t)" + porthos_code[i][9:-9] + ";\n"
        continue
    if(from_last == 5):
        aramis_code_main += porthos_code[i][-3] + ";\n"
        continue
    aramis_code_main += porthos_code[i]

aramis_file = open("compiled_aramis_file.cpp", 'w')

aramis_file.write(aramis_code_prelude)
aramis_file.write(aramis_code_main)

print("Compiled Aramis code will be written in src/main.cpp.\n")
# print("Copy this to src/main.cpp and overwrite old src/main.cpp.\nDo the same for all the 3 party directories i.e. party0/ party1/ and party2/")
