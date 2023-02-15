#pragma once

#include "gpu_data_types.h"
#include "gpu_select.h"
// #include "group_element.h"
#include <../dcf.h>

struct DReluKey {
    DCFKeyPack rinDcfKey;
    GroupElement routDReluZ2;
};

struct TwoRoundReluKey {
    DReluKey dreluKey;
    SelectKey selectKey;
};

struct GPUDReluKey {
    GPUDCFKey dcfKey;
    uint32_t* dReluMask;
};

struct GPU2RoundReLUKey {
    int bin, bout, numRelus;
    GPUDReluKey dreluKey;
    GPUSelectKey selectKey;
};

void writeTwoRoundReluKeyToFile(std::ostream& f, int bin, int bout, int numRelus, TwoRoundReluKey* k);
void genTwoRoundReluKey(int bin, int bout, GroupElement rin, GroupElement routDReluZ2, GroupElement rout, TwoRoundReluKey* k1, TwoRoundReluKey* k2);
void genGPUDReluKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int numDRelus, GPUGroupElement* inMask, GPUGroupElement* outMask);
void freeTwoRoundReluKeys(TwoRoundReluKey* key1, TwoRoundReluKey* key2, int N);
GPUDReluKey readGPUDReluKey(uint8_t** key_as_bytes);
GPU2RoundReLUKey readTwoRoundReluKey(uint8_t** key_as_bytes);