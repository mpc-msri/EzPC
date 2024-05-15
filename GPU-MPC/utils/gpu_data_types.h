// #ifndef GPU_DATA_TYPES_H
// #define GPU_DATA_TYPES_H

#pragma once

#include <utility>
#include <stdint.h>
#include <cstddef>

#include <sytorch/tensor.h>

#include "gpu_stats.h"

typedef unsigned __int128 AESBlock;

#define SERVER0 0
#define SERVER1 1
#define AES_BLOCK_LEN_IN_BITS 128
#define FULL_MASK 0xffffffff
#define LOG_AES_BLOCK_LEN 7

#define PACKING_SIZE 32
#define PACK_TYPE uint32_t

#define SHARED_MEM_BANK_SIZE 32

using orcaTemplateClass = u64;

namespace dcf
{
    namespace orca
    {
        namespace global
        {
            static const int bw = 64;
            static const int scale = 24;
        }
    }
}