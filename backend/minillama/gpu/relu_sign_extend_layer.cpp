#include <cassert>

#include "gpu_data_types.h"
#include "relu_sign_extend_layer.h"
#include <../dcf.h>
#include <../freekey.h>
#include "gpu_dcf.h"
#include "gpu_mem.h"
#include "gpu_fss_utils.h"
#include "gpu_file_utils.h"
#include "gpu_relu.h"
#include <omp.h>
// #include
// #include "dcf.h"



extern "C" std::pair<GPUGroupElement*, GPUGroupElement*> evalDRelu(GPUDCFKey k, 
int party, GPUGroupElement *d_in, GPUGroupElement* h_dReLUMask, GPUGroupElement* h_xLTRinMask, 
AESGlobalContext* gaes, Stats* stats, bool returnXLTRin);
// extern "C" void gpuAddSharesModN(int numBits, uint32_t* d_A, uint32_t* d_B, int N);
extern "C" void gpuReluSignExtendMux(int party, int bin, int N, 
GPUGroupElement* d_I, GPUGroupElement* h_oneHot, GPUGroupElement* h_outMask, GPUGroupElement* d_drelu, 
GPUGroupElement* d_xLTRin, Stats* s);

extern "C" GPUGroupElement* gpuSelectForMaxpool(GPUSelectKey k,
                                                uint32_t *d_drelu,
                                                GPUGroupElement *d_diff, /*GPUGroupElement* d_curMax,*/
                                                int party, Stats* stats);

ReluSignExtendLayer::ReluSignExtendLayer(int bin, int bout, int numRelus) {
    this->bin = bin;
    this->bout = bout;
    this->numRelus = numRelus;
    dReluMask = new GPUGroupElement[numRelus];
}

struct ReluSignExtendKey {
    DCFKeyPack dcfKey;
    GPUGroupElement dcfMask;
    GPUGroupElement dReluMask;
    GPUGroupElement oneHot[4];
    GPUGroupElement outMask[2];
};


void writeReluSignExtendKeyToFile(std::ostream& f, ReluSignExtendKey* k, int numRelus) {
    // printf("writing key %d\n", numRelus);
    DCFKeyPack *dcfKey = new DCFKeyPack[numRelus];
    for(int i = 0; i < numRelus; i++) {
        dcfKey[i] = k[i].dcfKey;
    }
    writeDCFKeyWithOneBitOutputToFile(f, dcfKey, numRelus);
    delete[] dcfKey;
    // printf("done writing dcf key\n");
    GPUGroupElement* dReluMask = new GPUGroupElement[numRelus];
    for(int i = 0; i < numRelus; i++) {
        dReluMask[i] = k[i].dReluMask;
    }
    writePackedBitsToFile(f, dReluMask, 2, numRelus);
    delete[] dReluMask;
    // printf("done writing drelu mask\n");
    GPUGroupElement* dcfMask = new GPUGroupElement[numRelus];
    for(int i = 0; i < numRelus; i++) {
        dcfMask[i] = k[i].dcfMask;
    }
    writePackedBitsToFile(f, dcfMask, 2, numRelus);
    delete[] dcfMask;
    // printf("done writing dcf mask\n");
    for(int i = 0; i < numRelus; i++) {
        for(int j = 0; j < 4; j++) {
            f.write((char*) &k[i].oneHot[j], sizeof(GPUGroupElement));
        }
    }

    for(int i = 0; i < numRelus; i++) {
        for(int j = 0; j < 2; j++) {
            f.write((char*) &k[i].outMask[j], sizeof(GPUGroupElement));
        }
    }
}
// have we feed memory in this place?
void ReluSignExtendLayer::genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* inputMask, /*GPUGroupElement* dReluMask, GPUGroupElement* dcfMask,*/ GPUGroupElement* outputMask) {
    ReluSignExtendKey *k1 = new ReluSignExtendKey[numRelus];
    ReluSignExtendKey *k2 = new ReluSignExtendKey[numRelus];
    printf("%d\n", numRelus);
    
    #pragma omp parallel for
    for(int i = 0; i < numRelus; i++) {
        // generate a 2-bit dcf key
        // remember to & later
        auto dcfKeyPair = cpuKeyGenDCF(bin, 2, inputMask[i], GroupElement(1));
        k1[i].dcfKey = dcfKeyPair.first;
        k2[i].dcfKey = dcfKeyPair.second;

        dReluMask[i] = randomGE(2);
        auto sharesdReluMask = splitShare(dReluMask[i], 2);
        k1[i].dReluMask = sharesdReluMask.first;
        k2[i].dReluMask = sharesdReluMask.second;
        // printf("drelu mask %d: %lu --> %lu %lu\n", i, dReluMask[i], k1[i].dReluMask, k2[i].dReluMask);

        auto dcfMask = randomGE(2);
        auto sharesDcfMask = splitShare(dcfMask/*[i]*/, 2);
        k1[i].dcfMask = sharesDcfMask.first;
        k2[i].dcfMask = sharesDcfMask.second;
        // printf("dcf mask %d: %lu --> %lu %lu\n", i, dcfMask[i], k1[i].dcfMask, k2[i].dcfMask);


        // GPUGroupElement oneHotVec[4];
        // memset(oneHotVec, 0, 4 * sizeof(GPUGroupElement));
        auto onePos = (-(2 * dReluMask[i] + dcfMask/*[i]*/)) & 3ULL;
        assert(onePos >= 0 && onePos < 4);
        // oneHotVec[onePos] = 1;

        for(int j = 0; j < 4; j++) {
            GPUGroupElement toSplit = (j == onePos ? 1ULL : 0ULL);
            auto sharesOneHot = splitShare(toSplit, bout);
            k1[i].oneHot[j] = sharesOneHot.first;
            k2[i].oneHot[j] = sharesOneHot.second;
            // if(j == 2 || j == 3) printf("onehot shares %d %d %lu --> %lu %lu\n", i, j, toSplit, k1[i].oneHot[j], k2[i].oneHot[j]);
        } 

        int outputMask0Idx = dReluMask[i] % 2;
        int outputMask1Idx = 1 - outputMask0Idx;

        outputMask[i] = randomGE(bout);
        auto sharesMask0 = splitShare(outputMask[i], bout);
        k1[i].outMask[outputMask0Idx] = sharesMask0.first;
        k2[i].outMask[outputMask0Idx] = sharesMask0.second;
        // printf("output mask shares %d %d %lu --> %lu %lu\n", i, outputMask0Idx, outputMask[i], k1[i].outMask[outputMask0Idx], k2[i].outMask[outputMask0Idx]);

        auto sharesMask1 = splitShare(outputMask[i] - inputMask[i], bout);
        k1[i].outMask[outputMask1Idx] = sharesMask1.first;
        k2[i].outMask[outputMask1Idx] = sharesMask1.second;
        // printf("%d %d %lu --> %lu %lu\n", i, outputMask1Idx, outputMask[i] - inputMask[i], k1[i].outMask[outputMask1Idx], k2[i].outMask[outputMask1Idx]);


    }

    #pragma omp parallel 
    {
        #pragma omp sections 
        {
            #pragma omp section 
            {
                writeReluSignExtendKeyToFile(f1, k1, numRelus);
            }
            #pragma omp section 
            {
                writeReluSignExtendKeyToFile(f2, k2, numRelus);
            }
        }
    }
    // writeReluSignExtendKeyToFile(f1, k1, numRelus);
    // writeReluSignExtendKeyToFile(f2, k2, numRelus);

    // #pragma omp parallel for
    for(int i = 0; i < numRelus; i++) {
        auto keyPair = std::make_pair(k1[i].dcfKey, k2[i].dcfKey);
        freeDCFKeyPackPair(keyPair);
    }

    delete[] k1;
    delete[] k2;


    // for(int i = 0; i < numRelus; i++) dReluMask[i] &= 1;
    // genGPUSelectKey(f1, f2, bout, bout, numRelus, dReluMask, incomingGradMask, outgoingGradMask);
    // printf("done here\n");
}

void ReluSignExtendLayer::genBackwardKey(std::ostream& f1, std::ostream& f2, /*GPUGroupElement* inputMask, GPUGroupElement* dReluMask, GPUGroupElement* dcfMask, GPUGroupElement* outputMask,*/ GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMask) {
    // GPUGroupElement* newDReluMask = new GPUGroupElement[numRelus];
    for(int i = 0; i < numRelus; i++) dReluMask[i] = dReluMask[i] & 1;
    genGPUSelectKey(f1, f2, bout, bout, numRelus, /*newDReluMask*/dReluMask, incomingGradMask, outgoingGradMask);
    // delete[] newDReluMask;
}


void ReluSignExtendLayer::readForwardKey(uint8_t** key_as_bytes) {
    reluSignExtendKey.dReluKey = readGPUDReluKey(key_as_bytes);
    reluSignExtendKey.dcfMask = (uint32_t*) *key_as_bytes;
    *key_as_bytes += ((2 * numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE);
    // printf("dcfMaskSize: %d\n", ((2 * numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE));
    reluSignExtendKey.oneHot = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += 4 * numRelus * sizeof(GPUGroupElement);
    reluSignExtendKey.outMask = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += 2 * numRelus * sizeof(GPUGroupElement);
}

void ReluSignExtendLayer::readBackwardKey(uint8_t** key_as_bytes) {
    backpropSelectKey = readGPUSelectKey(key_as_bytes, numRelus);
}

// void ReluSignExtendLayer::init(uint8_t** key_as_bytes, AESGlobalContext* g) {
//     gaes = g;

// } 
// no memory leak
GPUGroupElement* ReluSignExtendLayer::forward(Peer *peer, int party, GPUGroupElement* d_I, AESGlobalContext* gaes) {
    auto dreluOutput = evalDRelu(reluSignExtendKey.dReluKey.dcfKey, party, d_I, (GPUGroupElement*) reluSignExtendKey.dReluKey.dReluMask, (GPUGroupElement*) reluSignExtendKey.dcfMask, gaes, &s, true);
    auto d_xLTRin = dreluOutput.first;
    auto d_drelu = dreluOutput.second;
    gpuReconstructInPlace(d_drelu, 2, numRelus, peer, party, &s);
    gpuReconstructInPlace(d_xLTRin, 2, numRelus, peer, party, &s);
    gpuReluSignExtendMux(party, bin, numRelus, d_I, reluSignExtendKey.oneHot, reluSignExtendKey.outMask, d_drelu, d_xLTRin, &s);
    drelu = (uint32_t*) moveToCPU((uint8_t*) d_xLTRin, ((numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE), &s);
    gpuReconstructInPlace(d_I, bout, numRelus, peer, party, &s);
    gpuFree(d_xLTRin);
    gpuFree(d_drelu);
    return d_I;
}

// no memory leak
GPUGroupElement* ReluSignExtendLayer::backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* g) {
    auto d_drelu = (uint32_t*) moveToGPU((uint8_t*) drelu, ((numRelus - 1) / PACKING_SIZE + 1) * sizeof(PACK_TYPE), &s);
    auto d_selectOutput = gpuSelectForMaxpool(backpropSelectKey, d_drelu, d_incomingGrad, party, &s);
    gpuReconstructInPlace(d_selectOutput, bout, numRelus, peer, party, &s);
    gpuFree(d_drelu);
    gpuFree(d_incomingGrad);
    return d_selectOutput;
}
