#include "gpu_truncate.h"
#include <../dcf.h>    
#include "gpu_data_types.h"
#include "gpu_fss_utils.h"
#include "gpu_file_utils.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include <cassert>

extern "C" void gpuSelectForTruncate(int party, int N, GPUGroupElement* d_I, GPUGroupElement* d_maskedDcfBit, GPUGroupElement* h_outMask, GPUGroupElement* h_p, Stats* s);
extern "C" void gpuLocalLRS(int N, int shift, GPUGroupElement* d_I);
extern "C" void gpuLocalARS(int bin, int shift, int N, GPUGroupElement* d_I);
extern "C" GPUGroupElement *gpuMaskedDcf(GPUMaskedDCFKey k, int party, GPUGroupElement *d_in, AESGlobalContext* g, Stats* s);
extern "C" void gpuAddConstant(int bw, GPUGroupElement* d_I, GPUGroupElement c, int N);
extern "C" void gpuLocalARSWrapper(int bin, int shift, int N, GPUGroupElement* h_I, GPUGroupElement* h_O);
extern "C" void gpuLocalLRSWrapper(int bin, int shift, int N, GPUGroupElement* h_I, GPUGroupElement* h_O);


// GroupElement extend(GroupElement x, int bin, int bout) {
//     assert(bout > bin);
//     assert(bout == 64);
//     GroupElement msb = (x & (1ULL << (bin - 1))) >> (bin - 1);
//     GroupElement signMask = (-msb) << bin;
//     auto y = (x | signMask);// & ((1ULL << bout) - 1);
//     return y;
// }

void genGPUStochasticTruncateKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int shift, int N, GPUGroupElement* inMask, GPUGroupElement* outMask) {
    GPUGroupElement* truncatedInMask = new GPUGroupElement[N];
    for(int i = 0; i < N; i++) 
        truncatedInMask[i] = inMask[i] >> shift;
    genGPUSignExtendKey(f1, f2, bin - shift, bout, N, truncatedInMask, outMask);
    delete[] truncatedInMask;
}
 

void genGPUSignExtendKey(std::ostream& f1, std::ostream& f2, int bin, int bout, int N, GPUGroupElement* inMask, GPUGroupElement* outMask) {
    f1.write((char*) &bin, sizeof(int));
    f2.write((char*) &bin, sizeof(int));

    f1.write((char*) &bout, sizeof(int));
    f2.write((char*) &bout, sizeof(int));

    f1.write((char*) &N, sizeof(int));
    f2.write((char*) &N, sizeof(int));

    auto dcfMask = new GPUGroupElement[N];
    initRandomInPlace(dcfMask, N, 1);
    genGPUMaskedDCFKey(f1, f2, bin, 1, N, inMask, dcfMask);

    GroupElement *t = new GPUGroupElement[N];
    GroupElement *p = new GroupElement[2*N];
    for(int i = 0; i < N; i++) {
        outMask[i] = randomGE(bout);
        t[i] = outMask[i] - /*extend(inMask[i], bin, bout)*/inMask[i] - (1ULL << (bin - 1));
        mod(t[i], bout);
        // printf("%d: %lu %d %lu\n", i, t[i], bout, dcfMask[i]);
        assert(dcfMask[i] == 0 || dcfMask[i] == 1);
        int idx0 = dcfMask[i];
        int idx1 = 1 - idx0;
        p[2*i + idx0] = 0;
        p[2*i + idx1] = (1ULL << bin);
    }
    writeSecretSharesToFile(f1, f2, bout, N, t);
    writeSecretSharesToFile(f1, f2, bout, 2*N, p);
    
    delete[] dcfMask;
    delete[] t;
    delete[] p;
}

GPUSignExtendKey readGPUSignExtendKey(uint8_t** key_as_bytes) {
    GPUSignExtendKey k;
    k.bin = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);

    k.bout = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);

    k.N = *((int*) *key_as_bytes);
    *key_as_bytes += sizeof(int);

    k.dcfKey = readGPUMaskedDCFKey(key_as_bytes);
    // change this ugly name
    size_t sizeInBytes = k.dcfKey.dcfKey.num_dcfs * sizeof(GPUGroupElement);
    k.t = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += sizeInBytes;
    k.p = (GPUGroupElement*) *key_as_bytes;
    *key_as_bytes += 2*sizeInBytes;
    return k;
}

// no memory leaks
void gpuSignExtend(GPUSignExtendKey k, int party, Peer* peer, GPUGroupElement* d_I, AESGlobalContext* g, Stats* s) {
    gpuAddConstant(k.bin, d_I, 1ULL << (k.bin - 1), k.N);
    auto d_maskedDcfBit = gpuMaskedDcf(k.dcfKey, party, d_I, g, s);
    gpuReconstructInPlace(d_maskedDcfBit, 1, k.N, peer, party, s);
    gpuSelectForTruncate(party, k.N, d_I, d_maskedDcfBit, k.t, k.p, s);
    gpuReconstructInPlace(d_I, k.bout, k.N, peer, party, s);
    gpuFree(d_maskedDcfBit);
}

void gpuStochasticTruncate(GPUSignExtendKey k, /*int bin, int bout,*/ int shift, int party, Peer* peer, GPUGroupElement* d_I, AESGlobalContext* g, Stats* s) {
    gpuLocalLRS(k.N, shift, d_I);
    gpuSignExtend(k, party, peer, d_I, g, s);
}

GPUGroupElement cpuArs(GPUGroupElement x, int bin, int shift) {
    GPUGroupElement msb = (x & (1ULL << (bin - 1))) >> (bin - 1);
    GPUGroupElement signMask = (((1ULL << shift) - msb) << (64 - shift));
    x = (x >> shift) | signMask;
    // printf("%lu %lu %lu\n", msb, signMask, x);
    return x;
}

void gpuTruncate(int bin, int bout, TruncateType t, GPUSignExtendKey signExtendKey, int shift, Peer* peer, int party, int N, GPUGroupElement* d_I, AESGlobalContext* gaes, Stats* s) {
    switch(t) {
        case TruncateType::LocalLRS: 
            gpuLocalLRS(N, shift, d_I);
            break;
        case TruncateType::LocalARS:
            gpuLocalARS(bin, shift, N, d_I);
            break;
        case TruncateType::StochasticTruncate:
            gpuStochasticTruncate(signExtendKey, shift, party, peer , d_I, gaes, s);
            break;
        default:
            assert(t == TruncateType::None);
    }
}

void genGPUTruncateKey(std::ostream& f1, std::ostream& f2, TruncateType t, int bin, int bout, int shift, int N, GPUGroupElement *inMask, GPUGroupElement *outMask) {
    switch(t) {
        case TruncateType::StochasticTruncate:
            genGPUStochasticTruncateKey(f1, f2, bin, bout, shift, N, inMask, outMask);
            break;
        case TruncateType::LocalARS:
            gpuLocalARSWrapper(bin, shift, N, inMask, outMask);
            break;
        case TruncateType::LocalLRS:
            gpuLocalLRSWrapper(bin, shift, N, inMask, outMask);
            break;
        default:
            memcpy(outMask, inMask, N * sizeof(GPUGroupElement));
            assert(t == TruncateType::None);
    }
}

void readGPUTruncateKey(TruncateType t, GPUSignExtendKey *truncateKey, uint8_t** key_as_bytes) {
    switch(t) {
        case TruncateType::StochasticTruncate:
            *truncateKey = readGPUSignExtendKey(key_as_bytes);
            break;
        default:
            assert(t == TruncateType::None || t == TruncateType::LocalARS || t == TruncateType::LocalLRS);
    }
}

void checkStochasticTruncate(int bin, int bout, int shift, int N, GPUGroupElement* h_masked_A, GPUGroupElement* h_mask_A, GPUGroupElement* h_A_ct) {
    for(int i = 0; i < N; i++) {
        auto truncated_A = cpuArs(h_A_ct[i], bin, shift);
        mod(truncated_A, bout);
        auto output = h_masked_A[i] - h_mask_A[i];
        mod(output, bout);
        auto diff = output - truncated_A;
        mod(diff, bout);
        assert(diff <= 1);
        if(i < 10) printf("%lu %lu %lu %lu\n", output, truncated_A, diff, h_mask_A[i]);
    }
}