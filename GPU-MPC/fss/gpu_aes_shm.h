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

#pragma once

#include "gpu_aes_table.h"

#define NUM_SHARED_MEM_BANKS 32

#define AES_128_ROUNDS 10
#define AES_128_ROUNDS_MIN_1 9

#define CYCLIC_ROT_RIGHT_1 0x4321
#define CYCLIC_ROT_RIGHT_2 0x5432
#define CYCLIC_ROT_RIGHT_3 0x6543

struct AESGlobalContext
{
	u32 *t0_g;
	u8 *Sbox_g;
	u32 *t4_0G, *t4_1G, *t4_2G, *t4_3G;
};

struct AESSharedContext
{
	u32 (*t0_s)[NUM_SHARED_MEM_BANKS];
	u8 (*Sbox)[32][4];
	u32 *t4_0S;
	u32 *t4_1S;
	u32 *t4_2S;
	u32 *t4_3S;
};

#include "gpu_aes_shm.cu"