#include "gpu_data_types.h"

__global__ void beaver_add_group_elements(GPUGroupElement *A, GPUGroupElement *B, GPUGroupElement *C, GPUGroupElement *D, int N, int party);
__global__ void check_output(GPUGroupElement *O1, GPUGroupElement *O2, GPUGroupElement *R, int N);
__global__ void add_pool(GPUGroupElement *A, GPUGroupElement* B, int N, int C, int H, int W, int num_elems);
__global__ void add_group_elements(GPUGroupElement *A, GPUGroupElement *B, GPUGroupElement *C, int N);
__global__ void extract_group_elements(GPUGroupElement *A, double *A_1, double *A_2, double *A_3, double *A_4, int N);
__global__ void embed_group_elements(GPUGroupElement *A, double *A_1, double *A_2, double *A_3, double *A_4, int N);
__global__ void add_pool_backprop(GPUGroupElement *A, GPUGroupElement* B, int N, int C, int H, int W, int num_elems);
__global__ void xorBits(uint32_t *A, uint32_t* B, int N);
__global__ void addInPlace(GPUGroupElement* d_A, GPUGroupElement* d_B, int bw, int N);
__global__ void addModN(int numBits, uint32_t* A, uint32_t* B, int numInts);
__global__ void subtractInPlace(GPUGroupElement *A, GPUGroupElement *B, int N);
__global__ void beaver_add_group_elements(GPUGroupElement *A, GPUGroupElement *B, GPUGroupElement *C, int N, int party);
