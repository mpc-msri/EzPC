#pragma once

#include <../group_element.h>

struct SelectKey {
    GroupElement maskB, maskX, maskOut, oneBitDcfKey1, oneBitDcfKey2;
};

struct GPUSelectKey {
    int N;
    GroupElement *a, *b, *c, *d1, *d2;
};

void genSelectKey(/*int bin,*/ int bout, GroupElement maskB, GroupElement maskX, GroupElement maskOut, SelectKey* k1, SelectKey* k2);
void genGPUSelectKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int N, GroupElement* maskB, GroupElement* maskX, GroupElement* maskOut);
void writeSelectKeyToFile(std::ostream& f, SelectKey* k, int numSelects);
GPUSelectKey readGPUSelectKey(uint8_t** key_as_bytes, int N);
