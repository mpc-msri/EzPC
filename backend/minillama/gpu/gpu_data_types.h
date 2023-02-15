// #ifndef GPU_DATA_TYPES_H
// #define GPU_DATA_TYPES_H

#pragma once

#include <utility>
#include <stdint.h>
#include <cstddef> 

#define SHARED_MEM_BANK_SIZE			32
#define TABLE_SIZE						256

#define TORCH_CSPRNG_HOST_DEVICE __host__ __device__
#define TORCH_CSPRNG_CONSTANT __constant__


typedef uint64_t GPUGroupElement;
typedef unsigned __int128 AESBlock;
typedef std::pair<GPUGroupElement *, GPUGroupElement *> GPUGroupElementPair;

#define PACKING_SIZE 32
#define PACK_TYPE uint32_t

struct GPUMatmulKey
{
    int Bin, Bout, M, K, N;
    bool rowMajA, rowMajB, rowMajC;
    size_t size_A, size_B, size_C;
    size_t mem_size_A, mem_size_B, mem_size_C;    
    GPUGroupElement *A, *B, *C;
};

struct GPUConv2DKey
{
    int Bin, Bout, N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight,
        zPadWLeft, zPadWRight,
        strideH, strideW, OH, OW;
    size_t size_I, size_F, size_O;
    size_t mem_size_I, mem_size_F, mem_size_O;    
    GPUGroupElement *I, *F, *O;
};

struct GPUDCFKey
{
    int Bin, Bout, num_dcfs, out_vec_len;
    AESBlock *scw;
    GPUGroupElement *vcw;
    unsigned long int size_scw, size_vcw, mem_size_scw, mem_size_vcw;
};

struct GPUReLUTruncateKey
{
    int Bin, Bout, shift, num_rts;
    GPUGroupElement *zTruncate; // 16 * 8 = 128 bits
    GPUGroupElement *a, *b, *c, *d1, *d2, *a2;
    GPUDCFKey dcfKeyN;
    GPUDCFKey dcfKeyS;
};

// struct GPULocalTruncateReluKey
// {
//     int Bin, Bout, num_relus;
//     GPUGroupElement *rin, *routDReluZn;
//     GPUGroupElement *rout, *oneBitDcfKey1, *oneBitDcfKey2;
//     GPUGroupElement *routDReluZ2;
//     GPUDCFKey rinDcfKey;
// };

struct GPURTContext
{
    GPUGroupElement *d_lrs0;//, *d_a;
    // GPUGroupElement *h_lrs0;
    uint32_t *d_drelu0;//, *h_drelu0;
};


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

// enum TruncateType {
//     None,
//     LocalLRS,
//     LocalARS,
//     StochasticTruncate
// };

// static const GPUGroupElement lr_fp = 1;
// static const GPUGroupElement lr_scale = 6;
// static const GPUGroupElement mom_fp = 29;
// static const GPUGroupElement mom_scale = 5;
// static const GPUGroupElement scale = 24;


// struct GPUDReluKey {
//     GPUDCFKey dcfKey;
//     uint32_t* dReluMask;
// };



// #endif

// void printAESBlock(uint8_t *b)
// {
//     for (int i = 0; i < 16; i++)
//         printf("%02X", b[i]);
//     printf("\n");
// }