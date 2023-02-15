#include "gpu_data_types.h"
#include "maxpool_layer.h"
#include <../input_prng.h>
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include "cpu_fss.h"
#include <cassert>
// #include "gpu_fss.h"

extern "C" void initAESContext(AESGlobalContext* g);

    void maxPool2D(MaxpoolParams p, GPUGroupElement* in, GPUGroupElement* out, GPUGroupElement* incomingGrad, GPUGroupElement* outgoingGrad) {
        memset(outgoingGrad, 0, p.N * p.imgH * p.imgW * p.C * sizeof(GPUGroupElement));
        for(int i = 0; i < p.N; i++) {
            for(int j = 0; j < p.H; j++) {
                for(int k = 0; k < p.W; k++) {
                    for(int l = 0; l < p.C; l++) {
                        osuCrypto::u64 max = 0;
                        osuCrypto::u64 maxIdxI = 0;
                        osuCrypto::u64 maxIdxJ = 0;
                        for(int m = 0; m < p.FH; m++) {
                            for(int n = 0; n < p.FW; n++) {
                                osuCrypto::u64 val = Arr4DIdx(in, p.N, p.imgH, p.imgW, p.C, i, j*p.strideH+m, k*p.strideW+n, l);
                                if(m == 0 && n == 0) max = val;
                                if(((val - max) & ((1ULL << p.bin) - 1)) < (1ULL << (p.bin - 1))) {
                                    max = val;
                                    maxIdxI = m;
                                    maxIdxJ = n;
                                }
                            }
                        }
                        // printf("max value at: %d %d\n", maxIdxI, maxIdxJ);
                        Arr4DIdx(out, p.N, p.H, p.W, p.C, i, j, k, l) = max;
                        auto inGrad = Arr4DIdx(incomingGrad, p.N, p.H, p.W, p.C, i, j, k, l);
                        auto gradSum = Arr4DIdx(outgoingGrad, p.N, p.imgH, p.imgW, p.C, i, j*p.strideH+maxIdxI, k*p.strideW+maxIdxJ, l);
                        gradSum = (gradSum + inGrad) & ((1ULL << p.bin) - 1);
                        Arr4DIdx(outgoingGrad, p.N, p.imgH, p.imgW, p.C, i, j*p.strideH+maxIdxI, k*p.strideW+maxIdxJ, l) = gradSum;
                    }
                }
            }
        }
    }


void maxPool2DCollectGradients(MaxpoolParams p, GPUGroupElement* outgoingGradMaskExpanded, GPUGroupElement* outgoingGradMask) {
    memset(outgoingGradMask, 0, p.N * p.imgH * p.imgW * p.C * sizeof(GPUGroupElement));
    for(int n = 0; n < p.N; n++) {
        for(int h = 0; h < p.H; h++) {
            for(int w = 0; w < p.W; w++) {
                for(int c = 0; c < p.C; c++) {
                    int leftTopCornerH = h * p.strideH - p.zPadHLeft;
                    int leftTopCornerW = w * p.strideW - p.zPadWLeft;  
                    for(int fh = 0; fh < p.FH; fh++) {
                        for(int fw = 0; fw < p.FW; fw++) {
                            int curPosH = leftTopCornerH + fh;
                            int curPosW = leftTopCornerW + fw;
                            auto grad = Arr5DIdx(outgoingGradMaskExpanded, p.N, p.H, p.W, p.C, p.FH*p.FW, n, h, w, c, fh*p.FW+fw);
                            auto gradSum = Arr4DIdx(outgoingGradMask, p.N, p.imgH, p.imgW, p.C, n, curPosH, curPosW, c);
                            gradSum = (gradSum + grad) & ((1ULL << p.bin) - 1);
                            Arr4DIdx(outgoingGradMask, p.N, p.imgH, p.imgW, p.C, n, curPosH, curPosW, c) = gradSum;
                        }
                    }
                }
            }
        }
    }
}


int main(int argc, char *argv[]) {
    prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    initCPURandomness();
    AESGlobalContext g;
    initAESContext(&g);
    int N = 128;
    int bin = 40;
    int bout = 40;
    int imgH = 30;
    int imgW = 30;
    int C = 3;//64;
    int FH = 3;
    int FW = 3;
    int strideH = 2;
    int strideW = 2;
    int zPadHLeft = 0;
    int zPadHRight = 0;
    int zPadWLeft = 0;
    int zPadWRight = 0; 
    int party = atoi(argv[1]);

    auto maxpool_layer = MaxPool2DLayer(bin, bout, N, imgH, imgW, 
                    C, FH, FW, strideH, 
                    strideW, zPadHLeft, zPadHRight, 
                    zPadWLeft, zPadWRight);
    int inputSize = N * imgH * imgW * C;
    int outputSize = N * maxpool_layer.p.H * maxpool_layer.p.W * C;

    GPUGroupElement *h_inputMask, *h_outputMask, *h_incomingGradMask, *h_outgoingGradMask;
    GPUGroupElement *h_I, *h_incomingGrad;

    if(party == 0) {
        std::ofstream f1("maxpool_key1.dat"), f2("maxpool_key2.dat"); 
        h_inputMask = initRandom(inputSize, bin);
        h_outputMask = new GPUGroupElement[outputSize];
        h_incomingGradMask = initRandom(outputSize, bout);
        h_outgoingGradMask = new GPUGroupElement[inputSize];
        maxpool_layer.genForwardKey(f1, f2, h_inputMask, h_outputMask/*, h_oneHotOutputMask*/);
        maxpool_layer.genBackwardKey(f1, f2, /*h_oneHotOutputMask,*/ h_incomingGradMask, h_outgoingGradMask);
        f1.close();
        f2.close();
    }
    Peer* peer = connectToPeer(party, argv[2]);
    size_t file_size;
    uint8_t* key_as_bytes = readFile("maxpool_key" + std::to_string(party+1) + ".dat", &file_size);
    printf("boo\n");
    maxpool_layer.readForwardKey(&key_as_bytes);
    printf("boo\n");
    maxpool_layer.readBackwardKey(&key_as_bytes);
    printf("boo\n");
    auto d_masked_I = getMaskedInputOnGpu(inputSize, bin, party, peer, h_inputMask, &h_I);
    auto d_O = maxpool_layer.forward(peer, party, d_masked_I, &g);
    auto d_maskedIncomingGrad = getMaskedInputOnGpu(outputSize, bout, party, peer, h_incomingGradMask, &h_incomingGrad);
    auto d_maskedOutgoingGrad = maxpool_layer.backward(peer, party, d_maskedIncomingGrad, &g);
    if(party == 0) {
        auto h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_O, outputSize * sizeof(GPUGroupElement), NULL);
        GPUGroupElement *ct_o = new GPUGroupElement[outputSize];
        GPUGroupElement *outgoingGradCt = new GPUGroupElement[inputSize];
        maxPool2D(maxpool_layer.p, h_I, ct_o, h_incomingGrad, outgoingGradCt);
        
        for(int i = 0; i < outputSize; i++) {
            auto unmasked_output = h_O[i] - h_outputMask[i];
            mod(unmasked_output, bout);
            assert(unmasked_output == ct_o[i]);
            if(i < 10) 
                printf("%lu %lu\n", unmasked_output, ct_o[i]);
        }

        GPUGroupElement* h_maskedOutgoingGrad = (GPUGroupElement*) moveToCPU((uint8_t*) d_maskedOutgoingGrad, inputSize * sizeof(GPUGroupElement), NULL); 
        for(int i = 0; i < inputSize; i++) {
            auto outGrad = h_maskedOutgoingGrad[i] - h_outgoingGradMask[i];
            mod(outGrad, bout);
            if(i < 10) 
                printf("%lu %lu\n", outGrad, outgoingGradCt[i]);
            assert(outGrad == outgoingGradCt[i]);
        }
    }
}