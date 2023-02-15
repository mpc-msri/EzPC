#include "gpu_data_types.h"
#include "gpu_and.h"
#include "gpu_file_utils.h"
#include <cassert>
#include <omp.h>

GPUAndKey readGPUAndKey(uint8_t** key_as_bytes) {
    GPUAndKey k;
    k.N = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);
    int num_ints = (k.N - 1) / PACKING_SIZE + 1;
    size_t size_in_bytes = num_ints * sizeof(uint32_t);
    k.b0 = (uint32_t*) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b1 = (uint32_t*) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b2 = (uint32_t*) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    return k;
}

void genGPUAndKey(std::ostream& f1, std::ostream& f2, GroupElement *b0, GroupElement *b1, GroupElement *b2, int N) {
    GPUGroupElement *newB2 = new GPUGroupElement[N];
    for(int i = 0; i < N; i++) {
        assert(b0[i] == 0 || b0[i] == 1);
        assert(b1[i] == 0 || b1[i] == 1);
        assert(b2[i] == 0 || b2[i] == 1);
        newB2[i] = (b0[i]*b1[i] + b2[i]) & 1ULL;
    }
    writeSecretSharesToFile(f1, f2, 1, N, b0);
    writeSecretSharesToFile(f1, f2, 1, N, b1);
    writeSecretSharesToFile(f1, f2, 1, N, newB2);
    delete[] newB2;
}

void genAndKey(GroupElement b0, GroupElement b1, GroupElement b2, AndKey* key1, AndKey* key2) {
    assert(b0 == 0 || b0 == 1);
    assert(b1 == 0 || b1 == 1);
    assert(b2 == 0 || b2 == 1);
    
    auto sharesB0 = splitShare(b0, 1);
    key1->b0 = sharesB0.first;
    key2->b0 = sharesB0.second;

    auto sharesB1 = splitShare(b1, 1);
    key1->b1 = sharesB1.first;
    key2->b1 = sharesB1.second;

    auto sharesB2 = splitShare((b0*b1 + b2) & 1ULL, 1);
    key1->b2 = sharesB2.first;
    key2->b2 = sharesB2.second;

    assert(key1->b0 == 0 || key1->b0 == 1);
    assert(key1->b1 == 0 || key1->b1 == 1);
    assert(key1->b2 == 0 || key1->b2 == 1);

    assert(key2->b0 == 0 || key2->b0 == 1);
    assert(key2->b1 == 0 || key2->b1 == 1);
    assert(key2->b2 == 0 || key2->b2 == 1);

}

void writeAndKeyToFile(std::ostream& f, AndKey* k, int N) {
    f.write((char*) &N, sizeof(int));
    int num_ints = (N - 1) / PACKING_SIZE + 1;
    printf("%d\n", num_ints);
    uint32_t *b0 = new uint32_t[num_ints];
    uint32_t *b1 = new uint32_t[num_ints];
    uint32_t *b2 = new uint32_t[num_ints];
    // printf("boo\n");
    // #pragma omp parallel for
    for (int i = 0; i < N; i += PACKING_SIZE)
    {
        PACK_TYPE packed_b0 = 0;
        PACK_TYPE packed_b1 = 0;
        PACK_TYPE packed_b2 = 0;
        for(int j = 0; j < PACKING_SIZE; j++) {
            int idx = i + j;
            uint64_t v0 = (idx < N) ? k[idx].b0 : 0ULL;
            assert(v0 == 0 || v0 == 1);
            v0 <<= j;
            packed_b0 |= static_cast<PACK_TYPE>(v0);

            uint64_t v1 = (idx < N) ? k[idx].b1 : 0ULL;
            assert(v1 == 0 || v1 == 1);
            v1 <<= j;
            packed_b1 |= static_cast<PACK_TYPE>(v1);

            uint64_t v2 = (idx < N) ? k[idx].b2 : 0ULL;
            assert(v2 == 0 || v2 == 1);
            v2 <<= j;
            packed_b2 |= static_cast<PACK_TYPE>(v2);

        }
        int packed_idx = i / PACKING_SIZE;
        b0[packed_idx] = packed_b0;
        b1[packed_idx] = packed_b1;
        b2[packed_idx] = packed_b2;
        // printf("%d: %u %u %u\n", packed_idx, packed_b0, packed_b1, packed_b2);
    }
    f.write((char*) b0, num_ints * sizeof(uint32_t));
    f.write((char*) b1, num_ints * sizeof(uint32_t));
    f.write((char*) b2, num_ints * sizeof(uint32_t));   

    delete[] b0;
    delete[] b1;
    delete[] b2;
}