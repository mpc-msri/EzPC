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