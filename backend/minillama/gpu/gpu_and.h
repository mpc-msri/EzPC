#pragma once

#include <../group_element.h>

struct AndKey {
    GroupElement b0, b1, b2;
};

struct GPUAndKey {
    int N;
    uint32_t *b0, *b1, *b2;
};

void genAndKey(GroupElement b0, GroupElement b1, GroupElement b2, AndKey* key1, AndKey* key2);
void genGPUAndKey(std::ostream& f1, std::ostream& f2, GroupElement *b0, GroupElement *b1, GroupElement *b2, int N);
void writeAndKeyToFile(std::ostream& f, AndKey* k, int N);
GPUAndKey readGPUAndKey(uint8_t** key_as_bytes);
