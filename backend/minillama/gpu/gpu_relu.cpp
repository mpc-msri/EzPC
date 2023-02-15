#include "gpu_relu.h"
#include "gpu_dcf.h"
#include <../freekey.h>
#include "gpu_file_utils.h"
#include <cassert>
#include <omp.h>

GPUDReluKey readGPUDReluKey(uint8_t** key_as_bytes) {
    GPUDReluKey k;
    k.dcfKey = readGPUDCFKey(key_as_bytes);
    k.dReluMask = (uint32_t *) *key_as_bytes;
    // number of 32-bit integers * sizeof(int)
    *key_as_bytes += ((k.dcfKey.Bout * k.dcfKey.num_dcfs - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    return k;
}

GPU2RoundReLUKey readTwoRoundReluKey(uint8_t** key_as_bytes)
{
    GPU2RoundReLUKey k;
    k.bin = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);

    k.bout = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);

    k.numRelus = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);

    size_t size_in_bytes = k.numRelus * sizeof(GPUGroupElement);

    k.dreluKey = readGPUDReluKey(key_as_bytes);
    k.selectKey = readGPUSelectKey(key_as_bytes, k.numRelus);
    return k;
} 

void writeDReluKeyToFile(std::ostream& f, DReluKey* k, int numDRelus) {
    DCFKeyPack *dcfKeys = new DCFKeyPack[numDRelus];
    for(int i = 0; i < numDRelus; i++) {
        dcfKeys[i] = k[i].rinDcfKey;
    }
    writeDCFKeyWithOneBitOutputToFile(f, dcfKeys, numDRelus);
    delete[] dcfKeys;

    //drelu mask
    for(int i = 0; i < numDRelus; i += PACKING_SIZE) {
        PACK_TYPE packed_bits = 0;
        for(int j = 0; j < PACKING_SIZE; j++) {
            uint64_t value = 0;
            int idx = i + j;
            if(idx < numDRelus) {
                value = k[idx].routDReluZ2;
            }
            assert(value == 0 || value == 1);
            value <<= j;
            packed_bits |= static_cast<PACK_TYPE>(value);
        }
        f.write((char*) &packed_bits, sizeof(PACK_TYPE));
    }
}


void writeTwoRoundReluKeyToFile(std::ostream& f, int bin, int bout, int numRelus, TwoRoundReluKey* k) {
    f.write((char *)&bin, sizeof(int));
    f.write((char *)&bout, sizeof(int));
    f.write((char *)&numRelus, sizeof(int));

    DReluKey *dreluKeys = new DReluKey[numRelus];
    // #pragma omp parallel for
    for(int i = 0; i < numRelus; i++) {
        dreluKeys[i] = k[i].dreluKey;
    }
    writeDReluKeyToFile(f, dreluKeys, numRelus);
    delete[] dreluKeys;

    SelectKey* selectKey = new SelectKey[numRelus];
    // #pragma omp parallel for
    for(int i = 0; i < numRelus; i++) {
        selectKey[i] = k[i].selectKey;
    }
    writeSelectKeyToFile(f, selectKey, numRelus);
    delete[] selectKey;
}

void genDReluKey(int bin, int bout, GroupElement rin, GroupElement routDReluZ2, DReluKey* k1, DReluKey* k2) {
    rin &= ((1ULL << bin) - 1);
    // rout &= ((1ULL << bout) - 1);

    auto dcfKeyPair = cpuKeyGenDCF(bin, 1, rin, GroupElement(1));
    k1->rinDcfKey = dcfKeyPair.first;
    k2->rinDcfKey = dcfKeyPair.second;

    auto sharesRoutDReluZ2 = splitShare(routDReluZ2, 1);
    k1->routDReluZ2 = sharesRoutDReluZ2.first;
    k2->routDReluZ2 = sharesRoutDReluZ2.second;
}

void genTwoRoundReluKey(int bin, int bout, GroupElement rin, GroupElement routDReluZ2, GroupElement rout, TwoRoundReluKey* k1, TwoRoundReluKey* k2) {
    // rin &= ((1ULL << bin) - 1);
    rout &= ((1ULL << bout) - 1);
    genDReluKey(bin, bout, rin, routDReluZ2, &(k1->dreluKey), &(k2->dreluKey));
    genSelectKey(/*bin,*/ bout, routDReluZ2, rin, rout, &(k1->selectKey), &(k2->selectKey));
    // do we need to change the bitlength of idx before passing 
    // it as input to keygen?
    // auto dcfKeyPair = keyGenDCF(bin, 1, rin, GroupElement(1));
    // k1->rinDcfKey = dcfKeyPair.first;
    // k2->rinDcfKey = dcfKeyPair.second;
}


void freeTwoRoundReluKeys(TwoRoundReluKey* key1, TwoRoundReluKey* key2, int N) {
    for(int i = 0; i < N; i++) {
        auto keyPair = std::make_pair(key1[i].dreluKey.rinDcfKey, key2[i].dreluKey.rinDcfKey);
        freeDCFKeyPackPair(keyPair);
    }
    // printf("done freeing keys\n");
    // delete[] key1;
    // delete[] key2;
}

void freeDCFKeys(DCFKeyPack* dcfKey1, DCFKeyPack* dcfKey2, int N) {
    for(int i = 0; i < N; i++) {
        auto keyPair = std::make_pair(dcfKey1[i], dcfKey2[i]);
        freeDCFKeyPackPair(keyPair);
    }
}

void genGPUDReluKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int numDRelus, GPUGroupElement* inMask, GPUGroupElement* outMask) {
    DCFKeyPack *dcfKey1 = new DCFKeyPack[numDRelus];
    DCFKeyPack *dcfKey2 = new DCFKeyPack[numDRelus];

    printf("generating %d dcf keys...\n", numDRelus);

    #pragma omp parallel for
    for(int i = 0; i < numDRelus; i++) {
        auto keyPair = cpuKeyGenDCF(bin, bout, inMask[i], GroupElement(1));
        dcfKey1[i] = keyPair.first;
        dcfKey2[i] = keyPair.second;
    }
    printf("done generating dcf keys\n");
    #pragma omp parallel 
    {
        #pragma omp sections 
        {
            #pragma omp section 
            {
                writeDCFKeyWithOneBitOutputToFile(f1, dcfKey1, numDRelus);
            }
            #pragma omp section 
            {
                writeDCFKeyWithOneBitOutputToFile(f2, dcfKey2, numDRelus);
            }
        }
    }
    // writeDCFKeyWithOneBitOutputToFile(f1, dcfKey1, numDRelus);
    // writeDCFKeyWithOneBitOutputToFile(f2, dcfKey2, numDRelus);

    freeDCFKeys(dcfKey1, dcfKey2, numDRelus);

    delete[] dcfKey1;
    delete[] dcfKey2;

    writeSecretSharesToFile(f1, f2, bout, numDRelus, outMask);
}