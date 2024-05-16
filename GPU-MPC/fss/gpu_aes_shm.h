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

#pragma once

struct AESGlobalContext {
    uint32_t *t0G, *t4G, *t4_0G, *t4_1G, *t4_2G, *t4_3G;
    uint8_t* SAES;
};


struct AESSharedContext {
    uint32_t (*t0S)[SHARED_MEM_BANK_SIZE];
	uint8_t (*Sbox)[32][4];
	uint32_t *t4_0S;
	uint32_t *t4_1S;
	uint32_t *t4_2S;
	uint32_t *t4_3S;
};

#include "gpu_aes_shm.cu"