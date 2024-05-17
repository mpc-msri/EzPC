// Author: Neha Jawalkar
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

#include "utils/gpu_data_types.h"
#include "utils/helper_cuda.h"
#include "utils/gpu_mem.h"

__device__ const u32 RCON32C[15] = {
	0x01000000, 0x02000000, 0x04000000, 0x08000000,
	0x10000000, 0x20000000, 0x40000000, 0x80000000,
	0x1B000000, 0x36000000, 0x6C000000, 0xD8000000,
	0xAB000000, 0x4D000000, 0x9A000000};

inline __device__ u32 readSBoxByte(u32 byteIn, u8 (*Sbox)[32][4])
{
	int wTid = threadIdx.x & 31;
	auto i = (byteIn & 0xff) / 4;
	return (u32)Sbox[i][wTid][byteIn & 3];
}

inline __device__ u32 readSBoxByteAndShift(u32 byteIn, u8 (*Sbox)[32][4], int leftShift)
{
	return __byte_perm(readSBoxByte(byteIn, Sbox), 0, leftShift);
}

__device__ void aesKeySchedule(u32 *key, u32 *roundKey, u8 (*Sbox)[32][4])
{
	u32 roundKey0, roundKey1, roundKey2, roundKey3;

	roundKey0 = key[0];
	roundKey1 = key[1];
	roundKey2 = key[2];
	roundKey3 = key[3];

	roundKey[0] = roundKey0;
	roundKey[1] = roundKey1;
	roundKey[2] = roundKey2;
	roundKey[3] = roundKey3;

	for (u8 r = 0; r < AES_128_ROUNDS; r++)
	{
		auto byte0 = readSBoxByteAndShift(__byte_perm(roundKey3, 0, 0x4442), Sbox, 0x0444);
		auto byte1 = readSBoxByteAndShift(__byte_perm(roundKey3, 0, 0x4441), Sbox, 0x4044);
		auto byte2 = readSBoxByteAndShift(roundKey3, Sbox, 0x4404);
		auto byte3 = readSBoxByteAndShift(__byte_perm(roundKey3, 0, 0x4443), Sbox, 0x4440);
		roundKey0 = roundKey0 ^ byte0 ^ byte1 ^ byte2 ^ byte3 ^ RCON32C[r];
		roundKey1 = roundKey1 ^ roundKey0;
		roundKey2 = roundKey2 ^ roundKey1;
		roundKey3 = roundKey3 ^ roundKey2;

		roundKey[4 * r + 4] = roundKey0;
		roundKey[4 * r + 5] = roundKey1;
		roundKey[4 * r + 6] = roundKey2;
		roundKey[4 * r + 7] = roundKey3;
	}
}

inline __device__ u32 cyclicRot(u32 s, u32 rot)
{
	return __byte_perm(s, s, rot);
}

inline __device__ u32 computeOne(u32 s0, u32 s1, u32 s2, u32 s3, u32 *roundKey, int rkIdx, u32 (*t0_s)[NUM_SHARED_MEM_BANKS])
{
	int wTid = threadIdx.x & 31;
	return t0_s[__byte_perm(s0, 0, 0x4443)][wTid] ^
		   cyclicRot(t0_s[__byte_perm(s1, 0, 0x4442)][wTid], CYCLIC_ROT_RIGHT_1) ^
		   cyclicRot(t0_s[__byte_perm(s2, 0, 0x4441)][wTid], CYCLIC_ROT_RIGHT_2) ^
		   cyclicRot(t0_s[s3 & 0xff][wTid], CYCLIC_ROT_RIGHT_3) ^
		   roundKey[rkIdx];
}

inline __device__ u32 readSBoxByteAndCyclicShift(u32 byteIn, u8 (*Sbox)[32][4], int shift)
{
	return cyclicRot(readSBoxByte(byteIn, Sbox), shift);
}

inline __device__ u32 computeLast(u32 t0, u32 t1, u32 t2, u32 t3, u8 (*Sbox)[32][4], u32 roundKey)
{
	return readSBoxByteAndCyclicShift(t0 >> 24, Sbox, CYCLIC_ROT_RIGHT_1) ^ readSBoxByteAndCyclicShift(t1 >> 16, Sbox, CYCLIC_ROT_RIGHT_2) ^ readSBoxByteAndCyclicShift(t2 >> 8, Sbox, CYCLIC_ROT_RIGHT_3) ^ readSBoxByte(t3, Sbox) ^ roundKey;
}

__device__ void aesEncrypt(u32 *pt, u32 *roundKey, u32 (*t0_s)[NUM_SHARED_MEM_BANKS], u8 (*Sbox)[32][4])
{
	u32 s0, s1, s2, s3;
	s0 = pt[0];
	s1 = pt[1];
	s2 = pt[2];
	s3 = pt[3];
	s0 = s0 ^ roundKey[0];
	s1 = s1 ^ roundKey[1];
	s2 = s2 ^ roundKey[2];
	s3 = s3 ^ roundKey[3];

	u32 t0, t1, t2, t3;
	u32 rkIdx = 4;
	for (u8 r = 0; r < AES_128_ROUNDS_MIN_1; r++)
	{
		// Table based round function
		t0 = computeOne(s0, s1, s2, s3, roundKey, rkIdx, t0_s);
		t1 = computeOne(s1, s2, s3, s0, roundKey, rkIdx + 1, t0_s);
		t2 = computeOne(s2, s3, s0, s1, roundKey, rkIdx + 2, t0_s);
		t3 = computeOne(s3, s0, s1, s2, roundKey, rkIdx + 3, t0_s);
		s0 = t0;
		s1 = t1;
		s2 = t2;
		s3 = t3;
		rkIdx += 4;
	}
	s0 = computeLast(t0, t1, t2, t3, Sbox, roundKey[40]);
	s1 = computeLast(t1, t2, t3, t0, Sbox, roundKey[41]);
	s2 = computeLast(t2, t3, t0, t1, Sbox, roundKey[42]);
	s3 = computeLast(t3, t0, t1, t2, Sbox, roundKey[43]);

	pt[0] = s0;
	pt[1] = s1;
	pt[2] = s2;
	pt[3] = s3;
}

__device__ void loadSbox(AESGlobalContext *g, AESSharedContext *s)
{
	__shared__ u32 t0_s[AES_128_TABLE_SIZE][NUM_SHARED_MEM_BANKS];
	__shared__ u8 Sbox[64][32][4];
	// tb size might be small but it will be non-zero
	for (int i = 0; i < max(AES_128_TABLE_SIZE / blockDim.x, u32(1)); i++)
	{
		// stride
		int idx = threadIdx.x + i * blockDim.x;
		if (idx < AES_128_TABLE_SIZE)
		{
			for (u8 bank = 0; bank < NUM_SHARED_MEM_BANKS; bank++)
			{
				t0_s[idx][bank] = g->t0_g[idx];
				Sbox[idx / 4][bank][idx % 4] = g->Sbox_g[idx];
			}
		}
	}
	__syncthreads();
	s->t0_s = t0_s;
	s->Sbox = Sbox;
}

__device__ void reverseBytes(u32 *x)
{
	x[0] = __byte_perm(x[0], 0, 0x123);
	x[1] = __byte_perm(x[1], 0, 0x123);
	x[2] = __byte_perm(x[2], 0, 0x123);
	x[3] = __byte_perm(x[3], 0, 0x123);
}

__device__ void applyAESPRG(AESSharedContext *s, u32 *key, uint8_t pt, u32 *ct1)
{
	reverseBytes(key);
	u32 roundKey[44];
	aesKeySchedule(key, roundKey, s->Sbox);
	memset(ct1, 0, 4 * sizeof(u32));
	((uint8_t *)ct1)[3] = pt;
	aesEncrypt(ct1, roundKey, s->t0_s, s->Sbox);
	reverseBytes(ct1);
}

__device__ void applyAESPRGTwoTimes(AESSharedContext *s, u32 *key, uint8_t pt, u32 *ct1, u32 *ct2)
{
	reverseBytes(key);
	u32 roundKey[44];
	aesKeySchedule(key, roundKey, s->Sbox);
	memset(ct1, 0, 4 * sizeof(u32));
	memset(ct2, 0, 4 * sizeof(u32));
	((uint8_t *)ct1)[3] = pt;
	((uint8_t *)ct2)[3] = pt + 2;
	aesEncrypt(ct1, roundKey, s->t0_s, s->Sbox);
	aesEncrypt(ct2, roundKey, s->t0_s, s->Sbox);
	reverseBytes(ct1);
	reverseBytes(ct2);
}

__device__ void applyAESPRGFourTimes(AESSharedContext *s, u32 *key, u32 *ct1, u32 *ct2, u32 *ct3, u32 *ct4)
{
	reverseBytes(key);
	u32 roundKey[44];
	aesKeySchedule(key, roundKey, s->Sbox);
	memset(ct1, 0, 4 * sizeof(u32));
	memset(ct2, 0, 4 * sizeof(u32));
	memset(ct3, 0, 4 * sizeof(u32));
	memset(ct4, 0, 4 * sizeof(u32));
	((uint8_t *)ct2)[3] = 1;
	((uint8_t *)ct3)[3] = 2;
	((uint8_t *)ct4)[3] = 3;
	aesEncrypt(ct1, roundKey, s->t0_s, s->Sbox);
	aesEncrypt(ct2, roundKey, s->t0_s, s->Sbox);
	aesEncrypt(ct3, roundKey, s->t0_s, s->Sbox);
	aesEncrypt(ct4, roundKey, s->t0_s, s->Sbox);
	reverseBytes(ct1);
	reverseBytes(ct2);
	reverseBytes(ct3);
	reverseBytes(ct4);
}

void initAESContext(AESGlobalContext *g)
{
	g->t0_g = (u32 *)moveToGPU((u8 *)T0, AES_128_TABLE_SIZE * sizeof(u32), NULL);
	g->Sbox_g = (u8 *)moveToGPU((u8 *)Sbox_g, 256 * sizeof(u8), NULL);
}

void freeAESGlobalContext(AESGlobalContext *g)
{
	gpuFree(g->t0_g);
	gpuFree(g->Sbox_g);
}
