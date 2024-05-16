// Author: Neha Jawalkar, https://github.com/cihangirtezcan/CUDA_AES.git
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// #pragma once

// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include <cstdlib>

#include "utils/gpu_data_types.h"

// CUDA runtime
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include "AES_final.h"

__device__ const u32 RCON32C[RCON_SIZE] = {
	0x01000000, 0x02000000, 0x04000000, 0x08000000,
	0x10000000, 0x20000000, 0x40000000, 0x80000000,
	0x1B000000, 0x36000000, 0x6C000000, 0xD8000000,
	0xAB000000, 0x4D000000, 0x9A000000};

__host__ __device__ void printU32(uint8_t *b)
{
	printf("%02x%02x%02x%02x\n", b[0], b[1], b[2], b[3]);
}

// Key expansion from given key set, populate rk[44]
__device__ void keyExpansion(u32 *key, u32 *rk, u32 *t4_0S, u32 *t4_1S, u32 *t4_2S, u32 *t4_3S)
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

	for (u8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++)
	{
		u32 temp = rk3;
		// need to replace this by byte perm
		rk0 = rk0 ^ t4_3S[(temp >> 16) & 0xff] ^ t4_2S[(temp >> 8) & 0xff] ^ t4_1S[(temp)&0xff] ^ t4_0S[(temp >> 24)] ^ RCON32C[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk2 ^ rk3;

		rk[roundCount * 4 + 4] = rk0;
		rk[roundCount * 4 + 5] = rk1;
		rk[roundCount * 4 + 6] = rk2;
		rk[roundCount * 4 + 7] = rk3;
	}
}

__device__ void aesEncrypt(u32 *pt, u32 *rk, u32 (*t0S)[SHARED_MEM_BANK_SIZE], u8 (*Sbox)[32][4], int warpThreadIndex)
{
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
// size of shared memory needed:
// 256 * 32 * 4
// 64 * 32 * 4
// 4 * 256 * 4
// 15 + 13 + 12
// 15 + 25
// 40
// 40 - 10 = 4
// 16KB
__device__ void loadSbox(AESGlobalContext *g, AESSharedContext *s)
{
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	__shared__ u32 t4_0S[TABLE_SIZE];
	__shared__ u32 t4_1S[TABLE_SIZE];
	__shared__ u32 t4_2S[TABLE_SIZE];
	__shared__ u32 t4_3S[TABLE_SIZE];
	// tb size might be small but it will be non-zero
	for (int i = 0; i < max(TABLE_SIZE / blockDim.x, u32(1)); i++)
	{
		// stride
		int idx = threadIdx.x + i * blockDim.x;
		if (/*threadIdx.x*/ idx < TABLE_SIZE)
		{
			t4_0S[idx] = g->t4_0G[idx];
			t4_1S[idx] = g->t4_1G[idx];
			t4_2S[idx] = g->t4_2G[idx];
			t4_3S[idx] = g->t4_3G[idx];
			for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++)
			{
				t0S[idx][bankIndex] = g->t0G[idx];
				Sbox[idx / 4][bankIndex][idx % 4] = g->SAES[idx];
			}
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

__device__ void reverseBytes(u32 *x)
{
	x[0] = __byte_perm(x[0], 0, 291);
	x[1] = __byte_perm(x[1], 0, 291);
	x[2] = __byte_perm(x[2], 0, 291);
	x[3] = __byte_perm(x[3], 0, 291);
}

__device__ void applyAESPRG(AESSharedContext *s, u32 *key, uint8_t pt, u32 *ct1)
{
	reverseBytes(key);
	u32 rk[44];
	keyExpansion(key, rk, s->t4_0S, s->t4_1S, s->t4_2S, s->t4_3S);
	memset(ct1, 0, 4 * sizeof(u32));
	// memset(ct2, 0, 4 * sizeof(u32));
	((uint8_t *)ct1)[3] = pt;
	// ((uint8_t*) ct2)[3] = pt + 2;
	int warpThreadIndex = threadIdx.x & 31;

	// int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	// if(thread_id == 0) printf("AESx2\n");

	aesEncrypt(ct1, rk, s->t0S, s->Sbox, warpThreadIndex);
	// aesEncrypt(ct2, rk, s->t0S, s->Sbox, warpThreadIndex);
	reverseBytes(ct1);
	// reverseBytes(ct2);
}

__device__ void applyAESPRGTwoTimes(AESSharedContext *s, u32 *key, uint8_t pt, u32 *ct1, u32 *ct2)
{
	reverseBytes(key);
	u32 rk[44];
	keyExpansion(key, rk, s->t4_0S, s->t4_1S, s->t4_2S, s->t4_3S);
	memset(ct1, 0, 4 * sizeof(u32));
	memset(ct2, 0, 4 * sizeof(u32));
	((uint8_t *)ct1)[3] = pt;
	((uint8_t *)ct2)[3] = pt + 2;
	int warpThreadIndex = threadIdx.x & 31;

	// int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	// if(thread_id == 0) printf("AESx2\n");

	aesEncrypt(ct1, rk, s->t0S, s->Sbox, warpThreadIndex);
	aesEncrypt(ct2, rk, s->t0S, s->Sbox, warpThreadIndex);
	reverseBytes(ct1);
	reverseBytes(ct2);
}

__device__ void applyAESPRGFourTimes(AESSharedContext *s, u32 *key, u32 *ct1, u32 *ct2, u32 *ct3, u32 *ct4)
{
	reverseBytes(key);
	u32 rk[44];
	keyExpansion(key, rk, s->t4_0S, s->t4_1S, s->t4_2S, s->t4_3S);
	memset(ct1, 0, 4 * sizeof(u32));
	memset(ct2, 0, 4 * sizeof(u32));
	memset(ct3, 0, 4 * sizeof(u32));
	memset(ct4, 0, 4 * sizeof(u32));
	// ((uint8_t*) ct1)[3] = 0;
	((uint8_t *)ct2)[3] = 1;
	((uint8_t *)ct3)[3] = 2;
	((uint8_t *)ct4)[3] = 3;
	int warpThreadIndex = threadIdx.x & 31;

	aesEncrypt(ct1, rk, s->t0S, s->Sbox, warpThreadIndex);
	aesEncrypt(ct2, rk, s->t0S, s->Sbox, warpThreadIndex);
	aesEncrypt(ct3, rk, s->t0S, s->Sbox, warpThreadIndex);
	aesEncrypt(ct4, rk, s->t0S, s->Sbox, warpThreadIndex);

	reverseBytes(ct1);
	reverseBytes(ct2);
	reverseBytes(ct3);
	reverseBytes(ct4);
}

__global__ void aesWrapper(uint8_t pt, u32 *keyG, AESGlobalContext g, u32 *ct1, u32 *ct2)
{
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	// int warpThreadIndex = threadIdx.x & 31;
	AESSharedContext s;
	loadSbox(&g, &s);

	ct1 = &ct1[threadIndex * 4];
	ct2 = &ct2[threadIndex * 4];

	u32 key[4];
	memcpy(key, &keyG[threadIndex * 4], 4 * sizeof(u32));
	u32 zero[4];
	u32 two[4];
	for (u32 i = 0; i < 64; i++)
	{
		applyAESPRGTwoTimes(&s, key, pt, zero, two);
		memcpy(key, zero, 4 * sizeof(u32));
	}
	memcpy(ct1, zero, 4 * sizeof(u32));
	memcpy(ct2, two, 4 * sizeof(u32));
}

void initAESContext(AESGlobalContext *g)
{
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

void freeAESGlobalContext(AESGlobalContext *g)
{
	cudaFree(g->t0G);
	cudaFree(g->t4G);
	cudaFree(g->t4_0G);
	cudaFree(g->t4_1G);
	cudaFree(g->t4_2G);
	cudaFree(g->t4_3G);
	cudaFree(g->SAES);
}
