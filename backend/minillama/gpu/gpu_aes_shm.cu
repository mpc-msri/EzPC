// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include <cstdlib>

#include "gpu_data_types.h"

// CUDA runtime
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include "AES_final.h"

__device__ const u32 RCON32C[RCON_SIZE] = {
	0x01000000, 0x02000000, 0x04000000, 0x08000000,
	0x10000000, 0x20000000, 0x40000000, 0x80000000,
	0x1B000000, 0x36000000, 0x6C000000, 0xD8000000,
	0xAB000000, 0x4D000000, 0x9A000000
};

// __host__ __device__ void printAESBlock3(uint8_t *b)
// {
//     for (int i = 0; i < 16; i+=4)
//         printf("%02x%02x%02x%02x ", b[i], b[i+1], b[i+2], b[i+3]);
//     printf("\n");
// }

__host__ __device__ void printU32(uint8_t *b)
{
    printf("%02x%02x%02x%02x\n", b[0], b[1], b[2], b[3]);
}

// Key expansion from given key set, populate rk[44]
__device__ void keyExpansion(u32* key, u32* rk, u32* t4_0S, u32* t4_1S, u32* t4_2S, u32* t4_3S) 
{
		u32 rk0, rk1, rk2, rk3;

		rk0 = key[0];
		rk1 = key[1];
		rk2 = key[2];
		rk3 = key[3];

		rk[0] = rk0;
		rk[1] = rk1;
		rk[2] = rk2;
		rk[3] = rk3;

		for (u8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++) {
			u32 temp = rk3;
			// need to replace this by byte perm
			rk0 = rk0 ^ t4_3S[(temp >> 16) & 0xff] ^ t4_2S[(temp >> 8) & 0xff] ^ t4_1S[(temp) & 0xff] ^ t4_0S[(temp >> 24)] ^ RCON32C[roundCount];
			rk1 = rk1 ^ rk0;
			rk2 = rk2 ^ rk1;
			rk3 = rk2 ^ rk3;

			rk[roundCount * 4 + 4] = rk0;
			rk[roundCount * 4 + 5] = rk1;
			rk[roundCount * 4 + 6] = rk2;
			rk[roundCount * 4 + 7] = rk3;
		}
}

__device__ void aesEncrypt(u32* pt, u32* rk, u32 (*t0S)[SHARED_MEM_BANK_SIZE], u8 (*Sbox)[32][4], int warpThreadIndex) {
	u32 s0, s1, s2, s3;
	
	s0 = pt[0];		
	s1 = pt[1];		
	s2 = pt[2];		
	s3 = pt[3];
	s0 = s0 ^ rk[0];		
	s1 = s1 ^ rk[1];		
	s2 = s2 ^ rk[2];		
	s3 = s3 ^ rk[3];

	u32 t0, t1, t2, t3;
	for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) 
	{
			// Table based round function
		u32 rkStart = roundCount * 4 + 4;
		t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk[rkStart];
		t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk[rkStart + 1];
		t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk[rkStart + 2];
		t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rk[rkStart + 3];
		s0 = t0;			
		s1 = t1;			
		s2 = t2;			
		s3 = t3;
	}
		// Last round uses s-box directly and XORs to produce output.
	s0 = arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rk[40];
	s1 = arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rk[41];
	s2 = arithmeticRightShiftBytePerm((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rk[42];
	s3 = arithmeticRightShiftBytePerm((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], SHIFT_3_RIGHT) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rk[43];

	pt[0] = s0;
	pt[1] = s1;
	pt[2] = s2;
	pt[3] = s3;
}

__device__ void loadSbox(AESGlobalContext* g, AESSharedContext* s) {
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	__shared__ u32 t4_0S[TABLE_SIZE];
	__shared__ u32 t4_1S[TABLE_SIZE];
	__shared__ u32 t4_2S[TABLE_SIZE];
	__shared__ u32 t4_3S[TABLE_SIZE];

	if (threadIdx.x < TABLE_SIZE) {
		t4_0S[threadIdx.x] = g->t4_0G[threadIdx.x];
		t4_1S[threadIdx.x] = g->t4_1G[threadIdx.x];
		t4_2S[threadIdx.x] = g->t4_2G[threadIdx.x];
		t4_3S[threadIdx.x] = g->t4_3G[threadIdx.x];
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {	
			t0S[threadIdx.x][bankIndex] = g->t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = g->SAES[threadIdx.x];
		}
	}
	__syncthreads();
    s->t0S = t0S;
	s->Sbox = Sbox;
	s->t4_0S = t4_0S;
	s->t4_1S = t4_1S;
	s->t4_2S = t4_2S;
	s->t4_3S = t4_3S;
}

__device__ void reverseBytes(u32* x) {
	x[0] = __byte_perm(x[0], 0, 291);
	x[1] = __byte_perm(x[1], 0, 291);
	x[2] = __byte_perm(x[2], 0, 291);
	x[3] = __byte_perm(x[3], 0, 291);
}

__device__ void applyAESPRG(AESSharedContext* s, u32* key, uint8_t pt, u32* ct1, u32* ct2) {
	reverseBytes(key);
	u32 rk[44];
	keyExpansion(key, rk, s->t4_0S, s->t4_1S, s->t4_2S, s->t4_3S);	
	memset(ct1, 0, 4 * sizeof(u32));
	memset(ct2, 0, 4 * sizeof(u32));
	((uint8_t*) ct1)[3] = pt;
	((uint8_t*) ct2)[3] = pt + 2;
	int warpThreadIndex = threadIdx.x & 31;

	// int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	// if(thread_id == 0) printf("AESx2\n");

	aesEncrypt(ct1, rk, s->t0S, s->Sbox, warpThreadIndex);		
	aesEncrypt(ct2, rk, s->t0S, s->Sbox, warpThreadIndex);
	reverseBytes(ct1);
	reverseBytes(ct2);
}

__global__ void aesWrapper(uint8_t pt, u32* keyG, AESGlobalContext g, u32* ct1, u32* ct2) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	// int warpThreadIndex = threadIdx.x & 31;
	AESSharedContext s;
	loadSbox(&g, &s);

	ct1 = &ct1[threadIndex * 4];
	ct2 = &ct2[threadIndex * 4];

	u32 key[4];
	memcpy(key, &keyG[threadIndex * 4], 4 * sizeof(u32));	
	u32 zero[4];
	u32 two[4];
	for (u32 i = 0; i < 64; i++) {
		applyAESPRG(&s, key, pt, zero, two);
		memcpy(key, zero, 4 * sizeof(u32));
	}
	memcpy(ct1, zero, 4 * sizeof(u32));
	memcpy(ct2, two, 4 * sizeof(u32));
}

extern "C" void initAESContext(AESGlobalContext* g) {
	gpuErrorCheck(cudaMalloc(&g->t0G, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&g->t4G, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&g->t4_0G, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&g->t4_1G, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&g->t4_2G, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&g->t4_3G, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&g->SAES, 256 * sizeof(u8))); // Cihangir

	gpuErrorCheck(cudaMemcpy(g->t0G, T0, TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(g->t4G, T4, TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(g->t4_0G, T4_0, TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(g->t4_1G, T4_1, TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(g->t4_2G, T4_2, TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(g->t4_3G, T4_3, TABLE_SIZE * sizeof(u32), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(g->SAES, SAES, 256 * sizeof(u8), cudaMemcpyHostToDevice)); // Cihangir
}

extern "C" void freeAESGlobalContext(AESGlobalContext* g) {
	cudaFree(g->t0G);
	cudaFree(g->t4G);
	cudaFree(g->t4_0G);
	cudaFree(g->t4_1G);
	cudaFree(g->t4_2G);
	cudaFree(g->t4_3G);
	cudaFree(g->SAES);
}

extern "C" int runAESCallChain(int num_aes, uint32_t* h_rk, uint8_t pt, uint32_t* h_ct1, uint32_t* h_ct2)
{
	u32 *rk, *ct1, *ct2;
	// gpuErrorCheck(cudaMalloc(&pt, num_aes * 4 * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&rk, num_aes * 4 * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&ct1, num_aes * 4 * sizeof(u32)));
	gpuErrorCheck(cudaMalloc(&ct2, num_aes * 4 * sizeof(u32)));
	// gpuErrorCheck(cudaMemcpy(pt, h_pt, num_aes * 4 * sizeof(u32), cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(rk, h_rk, num_aes * 4 * sizeof(u32), cudaMemcpyHostToDevice));

	AESGlobalContext g;
	initAESContext(&g);

	// clock_t beginTime = clock();

	aesWrapper << <(num_aes-1) / THREADS + 1, THREADS >> >(pt, rk, g, ct1, ct2);

	gpuErrorCheck(cudaDeviceSynchronize());
	// printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	// printf("-------------------------------\n");
	printLastCUDAError();

	gpuErrorCheck(cudaMemcpy(h_ct1, ct1, num_aes * 4 * sizeof(u32), cudaMemcpyDeviceToHost));
	gpuErrorCheck(cudaMemcpy(h_ct2, ct2, num_aes * 4 * sizeof(u32), cudaMemcpyDeviceToHost));

	cudaFree(rk);
	cudaFree(ct1);
	cudaFree(ct2);
	freeAESGlobalContext(&g);
	return 0;
}
