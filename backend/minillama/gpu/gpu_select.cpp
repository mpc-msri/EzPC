#include "gpu_select.h"
#include "gpu_data_types.h"
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include <cassert>

void genSelectKey(/*int bin,*/ int bout, GroupElement maskB, GroupElement maskX, GroupElement maskOut, SelectKey* k1, SelectKey* k2) {
    auto sharesMaskB = splitShare(maskB, bout);
    k1->maskB = sharesMaskB.first;
    k2->maskB = sharesMaskB.second;

    auto sharesMaskX = splitShare(maskX, bout);
    k1->maskX = sharesMaskX.first;
    k2->maskX = sharesMaskX.second;

    auto sharesMaskOut = splitShare((maskB * maskX + maskOut) & ((1ULL << bout) - 1), bout);
    k1->maskOut = sharesMaskOut.first;
    k2->maskOut = sharesMaskOut.second;

    GroupElement oneBitDcfKey1(0), oneBitDcfKey2(0);
    if(maskB == 1) {
        oneBitDcfKey1 = 2;
        oneBitDcfKey2 = (-2 * maskX) & ((1ULL << bout) - 1);
    }

    auto sharesOneBitDcfKey1 = splitShare(oneBitDcfKey1, bout);
    k1->oneBitDcfKey1 = sharesOneBitDcfKey1.first;
    k2->oneBitDcfKey1 = sharesOneBitDcfKey1.second;

    auto sharesOneBitDcfKey2 = splitShare(oneBitDcfKey2, bout);
    k1->oneBitDcfKey2 = sharesOneBitDcfKey2.first;
    k2->oneBitDcfKey2 = sharesOneBitDcfKey2.second;
}

void genGPUSelectKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int N, GroupElement* maskB, GroupElement* maskX, GroupElement* maskOut) {
    assert(bin == bout);
    GroupElement *newMaskOut = new GPUGroupElement[N];
    GroupElement *oneBitDcfKey1 = new GPUGroupElement[N];
    GroupElement *oneBitDcfKey2 = new GPUGroupElement[N];
    for(int i = 0; i < N; i++) {
        assert(maskB[i] == 0 || maskB[i] == 1);
        assert(bin == 64 || maskX[i] < (1ULL << bin));
        assert(bout == 64 || maskOut[i] < (1ULL << bout));
        // printf("%lu\n", maskOut[i]);
        maskOut[i] = randomGE(bout);
        newMaskOut[i] = maskB[i]*maskX[i] + maskOut[i];
        // printf("%lu\n", maskOut[i]);
        mod(newMaskOut[i], bout);
        //  & ((1ULL << bout) - 1);
        
        oneBitDcfKey1[i] = 0;
        oneBitDcfKey2[i] = 0;
        if(maskB[i] == 1) {
            oneBitDcfKey1[i] = 2;
            oneBitDcfKey2[i] = -2 * maskX[i]; 
            mod(oneBitDcfKey1[i], bout);
            mod(oneBitDcfKey2[i], bout);
            // & ((1ULL << bout) - 1);
        }
        // printf("%lu %lu %lu %lu %lu\n", maskB[i], maskX[i], maskOut[i], oneBitDcfKey1[i], oneBitDcfKey2[i]);
    }
    writeSecretSharesToFile(f1, f2, bout, N, maskB);
    writeSecretSharesToFile(f1, f2, bout, N, maskX);
    writeSecretSharesToFile(f1, f2, bout, N, newMaskOut);
    writeSecretSharesToFile(f1, f2, bout, N, oneBitDcfKey1);
    writeSecretSharesToFile(f1, f2, bout, N, oneBitDcfKey2);

    delete[] newMaskOut;
    delete[] oneBitDcfKey1;
    delete[] oneBitDcfKey2;
}


void writeSelectKeyToFile(std::ostream& f, SelectKey* k, int numSelects) {
    for(int i = 0; i < numSelects; i++) {
        f.write((char *)&k[i].maskB, sizeof(uint64_t));
    }

    for(int i = 0; i < numSelects; i++) {
        f.write((char *)&k[i].maskX, sizeof(uint64_t));
    }

    for(int i = 0; i < numSelects; i++) {
        f.write((char *)&k[i].maskOut, sizeof(uint64_t));
    }

    for(int i = 0; i < numSelects; i++) {
        f.write((char *)&k[i].oneBitDcfKey1, sizeof(uint64_t));
    }

    for(int i = 0; i < numSelects; i++) {
        f.write((char *)&k[i].oneBitDcfKey2, sizeof(uint64_t));
    }
}



GPUSelectKey readGPUSelectKey(uint8_t** key_as_bytes, int N) {
    GPUSelectKey k;
    k.N = N;

    size_t size_in_bytes = N * sizeof(GPUGroupElement);

    k.a = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.b = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.c = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d1 = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    k.d2 = (GPUGroupElement *) *key_as_bytes;
    *key_as_bytes += size_in_bytes;

    return k;
}
