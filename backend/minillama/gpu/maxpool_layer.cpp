#include <cassert>

#include "gpu_data_types.h"
#include "maxpool_layer.h"
#include <../dcf.h>
#include "gpu_dcf.h"
#include "gpu_mem.h"
#include "gpu_fss_utils.h"
#include "gpu_file_utils.h"
#include "gpu_relu.h"
#include <omp.h>
// #include "dcf.h"

extern "C" void gpuAddSharesInPlace(GPUGroupElement* d_A, GPUGroupElement* d_B, int bw, int N);


extern "C" std::pair<uint32_t*, GPUGroupElement*> gpuDReluForMaxPool(MaxpoolParams p, GPUDReluKey k, int party, int fh, int fw,
                                  GPUGroupElement* d_curMax, GPUGroupElement *d_in,
                                  AESGlobalContext* gaes, Stats* stats);

extern "C" GPUGroupElement* gpuSelectForMaxpool(GPUSelectKey k,
                                                 uint32_t *d_drelu, 
                                                 GPUGroupElement *d_diff, /*GPUGroupElement* d_curMax,*/
                                                  int party, Stats* stats);

extern "C" void gpuAndForMaxpool(MaxpoolParams p, int pos, GPUAndKey k,
                                            uint32_t *d_drelu,/* uint32_t *d_drelu2,*/ 
                                            uint32_t *d_oneHot,
                                            int party, Stats* stats);

extern "C" GPUGroupElement* gpuSelectForMaxpoolBackprop(MaxpoolParams p, GPUSelectKey k,
                                                uint32_t *d_oneHot, 
                                                GPUGroupElement *d_incomingGrad, 
                                                int party, Stats* stats);

extern "C" GPUGroupElement* gpuCollectGradients(MaxpoolParams p, GPUGroupElement* d_outgoingGradExpanded, Stats* s);
// void gpuReconstructBits(uint32_t* d_A0, int N, Peer* peer, int party, Stats* s);



MaxPool2DLayer::MaxPool2DLayer(int bin, int bout, int N, int imgH, int imgW, int C, int FH, int FW, int strideH, 
int strideW, int zPadHLeft, int zPadHRight, int zPadWLeft, int zPadWRight) {
    assert(bin == bout);
    p.bin = bin;
    p.bout = bout;
    p.N = N;
    p.imgH = imgH; 
    p.imgW = imgW; 
    p.C = C;
    p.FH = FH; 
    p.FW = FW; 
    assert(imgH >= FH);
    assert(imgW >= FW);
    p.strideH = strideH; 
    p.strideW = strideW; 
    //no padding for now
    assert(zPadHLeft == 0 && zPadHRight == 0 && zPadWLeft == 0 && zPadWRight == 0);
    p.zPadHLeft = zPadHLeft; 
    p.zPadHRight = zPadHRight; 
    p.zPadWLeft = zPadWLeft; 
    p.zPadWRight = zPadWRight;
    p.H = ((imgH - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    p.W = ((imgW - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    oneHotOutputMask = new GPUGroupElement[p.FH * p.FW * p.N * p.H * p.W * p.C];
}

void MaxPool2DLayer::genForwardKey(std::ostream& f1, std::ostream& f2, GPUGroupElement* h_inputMask, GPUGroupElement* h_outputMask/*, GPUGroupElement* oneHotOutputMask, GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMaskExpanded*/) {
    GroupElement *maxUntilNow_mask = h_outputMask;
    memset(oneHotOutputMask, 0, p.FH * p.FW * p.N * p.H * p.W * p.C * sizeof(GPUGroupElement));

    TwoRoundReluKey* reluKey1 = new TwoRoundReluKey[p.N * p.H * p.W * p.C];
    TwoRoundReluKey* reluKey2 = new TwoRoundReluKey[p.N * p.H * p.W * p.C];

    AndKey *andKey1 = new AndKey[p.FH * p.FW * p.N * p.H * p.W * p.C];
    AndKey *andKey2 = new AndKey[p.FH * p.FW * p.N * p.H * p.W * p.C];
    // printf("num and keys: %d\n", p.FH * p.FW * p.N * p.H * p.W * p.C);

    // might be over counting but who cares
    // printf("%d %d %d %d %d %d %d %d\n", p.FH, p.FW, p.N, p.imgH, p.imgW, p.C, p.H, p.W);
    int numRelus = p.N * p.H * p.W * p.C;
    for (int fh = 0; fh < p.FH; fh++) {
        for(int fw = 0; fw < p.FW; fw++) {
            // int i = 0;
            auto start = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for collapse(4)
            for (int n = 0; n < p.N; n++) {
                for(int ctH = 0; ctH < p.H; ctH++) {
                    for(int ctW = 0; ctW < p.W; ctW++) {
                        for (int c = 0; c < p.C; c++) {
                            // printf("%d %d %d %d %d %d\n", fh, fw, n, c, ctH, ctW);
                            int leftTopCornerH = ctH * p.strideH - p.zPadHLeft;
                            int leftTopCornerW = ctW * p.strideW - p.zPadWLeft;

                            if (fh == 0 && fw == 0) {
                                if (leftTopCornerH < 0 || leftTopCornerW < 0 || leftTopCornerH >= p.imgH || leftTopCornerW >= p.imgW) {
                                    Arr4DIdx(maxUntilNow_mask, p.N, p.H, p.W, p.C, n, ctH, ctW, c) = GroupElement(0);
                                }
                                else {
                                    Arr4DIdx(maxUntilNow_mask, p.N, p.H, p.W, p.C, n, ctH, ctW, c) = Arr4DIdx(h_inputMask, p.N, p.imgH, p.imgW, p.C, n, leftTopCornerH, leftTopCornerW, c);
                                }
                            }
                            else {
                                int curPosH = leftTopCornerH + fh;
                                int curPosW = leftTopCornerW + fw;

                                GroupElement maxi_mask = Arr4DIdx(maxUntilNow_mask, p.N, p.H, p.W, p.C, n, ctH, ctW, c);
                                GroupElement temp_mask;
                                if ((((curPosH < 0) || (curPosH >= p.imgH)) || ((curPosW < 0) || (curPosW >= p.imgW)))) {
                                    temp_mask = GroupElement(0);
                                }
                                else {
                                    temp_mask = Arr4DIdx(h_inputMask, p.N, p.imgH, p.imgW, p.C, n, curPosH, curPosW, c);
                                }
                                GroupElement rout = randomGE(p.bin);
                                GroupElement routBit = randomGE(1);
                                int unrolledIdx = n * p.H * p.W * p.C + ctH * p.W * p.C + ctW * p.C + c;
                                // printf("writing to %d\n", unrolledIdx);
                                assert(unrolledIdx < p.N * p.H * p.W * p.C);
                                genTwoRoundReluKey(p.bin, p.bin, temp_mask - maxi_mask, routBit, rout - maxi_mask, &reluKey1[unrolledIdx], &reluKey2[unrolledIdx]);
                                auto oneHotLen = fh*p.FW + fw + 1;
                                for(int j = 0; j < oneHotLen; j++) {
                                    GroupElement curOutputMask;
                                    curOutputMask = randomGE(1);
                                    int idx = n * p.H * p.W * p.C * p.FH * p.FW + ctH * p.W * p.C * p.FH * p.FW + ctW * p.C * p.FH * p.FW + c * p.FH * p.FW + j;
                                    int andKeyIdx = n * p.H * p.W * p.C * oneHotLen + ctH * p.W * p.C * oneHotLen + ctW * p.C * oneHotLen + c * oneHotLen + j;
                                    if(j == oneHotLen - 1 && oneHotOutputMask[idx] != 0) assert(false && "input mask for last diff must be zero!");
                                    // printf("%d\n", andKeyIdx);
                                    genAndKey(routBit, oneHotOutputMask[idx], (curOutputMask - oneHotOutputMask[idx]) & GroupElement(1), &andKey1[andKeyIdx], &andKey2[andKeyIdx]);
                                    assert(curOutputMask == 0 || curOutputMask == 1);
                                    oneHotOutputMask[idx] = curOutputMask;
                                }
                                // i++;
                                Arr4DIdx(maxUntilNow_mask, p.N, p.H, p.W, p.C, n, ctH, ctW, c) = rout;
                            }
                        }
                    }
                }
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto elapsed = stop - start;
            printf("to generate keys: %lu milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
            if(fh == 0 && fw == 0) continue;
            // printf("%d\n", p.N * p.H * p.W * p.C);
            #pragma omp parallel 
            {
                #pragma omp sections 
                {
                    #pragma omp section 
                    {
                        writeTwoRoundReluKeyToFile(f1, p.bin, p.bout, numRelus, reluKey1);
                    }
                    #pragma omp section 
                    {
                        writeTwoRoundReluKeyToFile(f2, p.bin, p.bout, numRelus, reluKey2);
                    }
                }
            }
            // writeTwoRoundReluKeyToFile(f1, p.bin, p.bout, numRelus, reluKey1);
            // writeTwoRoundReluKeyToFile(f2, p.bin, p.bout, numRelus, reluKey2);
            auto stop2 = std::chrono::high_resolution_clock::now();
            elapsed = stop2 - stop;
            printf("to write relu keys: %lu milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
            int numAnds = p.N * p.H * p.W * p.C * (fh * p.FW + fw + 1);
            writeAndKeyToFile(f1, andKey1, numAnds);
            writeAndKeyToFile(f2, andKey2, numAnds);
            freeTwoRoundReluKeys(reluKey1, reluKey2, numRelus);
            auto stop3 = std::chrono::high_resolution_clock::now();
            elapsed = stop3 - stop2;
            printf("to write and keys: %lu milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
        }
    }
    delete[] reluKey1;
    delete[] reluKey2;
    delete[] andKey1;
    delete[] andKey2;
}

void MaxPool2DLayer::genBackwardKey(std::ostream& f1, std::ostream& f2, /*GPUGroupElement* oneHotOutputMask,*/ GPUGroupElement* incomingGradMask, GPUGroupElement* outgoingGradMask) {
    int numGradElems = p.N * p.H * p.W * p.C;
    int numSelects = p.FH * p.FW * p.N * p.H * p.W * p.C;
    int oneHotLen = p.FH * p.FW;
    SelectKey *selectKey1 = new SelectKey[numSelects];
    SelectKey *selectKey2 = new SelectKey[numSelects];
    memset(outgoingGradMask, 0, p.N * p.imgH * p.imgW * p.C * sizeof(GPUGroupElement));

    for (int n = 0; n < p.N; n++) {
        for(int ctH = 0; ctH < p.H; ctH++) {
            for(int ctW = 0; ctW < p.W; ctW++) {
                for (int c = 0; c < p.C; c++) {
                    auto incomingGrad = incomingGradMask[n * p.H * p.W * p.C + ctH * p.W * p.C + ctW * p.C + c];
                    int leftTopCornerH = ctH * p.strideH - p.zPadHLeft;
                    int leftTopCornerW = ctW * p.strideW - p.zPadWLeft;        
                    for (int fh = 0; fh < p.FH; fh++) {
                        for(int fw = 0; fw < p.FW; fw++) {
                            int idx = n * p.H * p.W * p.C * p.FH * p.FW + ctH * p.W * p.C * p.FH * p.FW + ctW * p.C * p.FH * p.FW + c * p.FH * p.FW + fh * p.FW + fw;
                            auto oneHot = oneHotOutputMask[idx];
                            auto outputMask = /*outgoingGradMaskExpanded[idx];*/randomGE(p.bout);
                            genSelectKey(/*p.bin,*/ p.bout, oneHot, incomingGrad, outputMask, &selectKey1[idx], &selectKey2[idx]);
                            int curPosH = leftTopCornerH + fh;
                            int curPosW = leftTopCornerW + fw;
                            // printf("%d %d %d %d <-- %d %d %d %d %d %d\n", n, curPosH, curPosW, c, n, ctH, ctW, c, fh, fw);
                            Arr4DIdx(outgoingGradMask, p.N, p.imgH, p.imgW, p.C, n, curPosH, curPosW, c) += outputMask;
                        }
                    }
                }
            }
        }
    }
    writeSelectKeyToFile(f1, selectKey1, numSelects);
    writeSelectKeyToFile(f2, selectKey2, numSelects);

    delete[] selectKey1;
    delete[] selectKey2;
}

void MaxPool2DLayer::readForwardKey(uint8_t** key_as_bytes) {
    // maxpoolKey.p = p;
    // gaes = g;
    maxpoolKey.reluKey = new GPU2RoundReLUKey[p.FH * p.FW];
    maxpoolKey.andKey = new GPUAndKey[p.FH * p.FW];
    for(int i = 0; i < p.FH; i++) {
        for(int j = 0; j < p.FW; j++) {
            if(i == 0 && j == 0) continue;
            maxpoolKey.reluKey[i * p.FW + j] = readTwoRoundReluKey(key_as_bytes);
            maxpoolKey.andKey[i * p.FW + j] = readGPUAndKey(key_as_bytes);
        }
    }
    // int numSelects = p.N * p.H * p.W * p.C * p.FH * p.FW;
    // backpropSelectKey = readGPUSelectKey(key_as_bytes, numSelects);

    // maxpoolKey.relumaxpoolKey.dcfKeyN = readTwoRoundReluKey(key_as_bytes);
}

void MaxPool2DLayer::readBackwardKey(uint8_t** key_as_bytes) {
    int numSelects = p.N * p.H * p.W * p.C * p.FH * p.FW;
    backpropSelectKey = readGPUSelectKey(key_as_bytes, numSelects);
}

// need to see if memory is properly freed everywhere
// relu(diff) + curMax
GPUGroupElement* gpuMaxpool(MaxpoolParams p, Peer* peer, int party, GPU2RoundReLUKey k, GPUAndKey andKey, int i, int j, GPUGroupElement* d_I, GPUGroupElement* d_curMax, uint32_t* d_oneHot, AESGlobalContext* gaes, Stats* s) {
    auto res = gpuDReluForMaxPool(p, k.dreluKey, party, i, j, d_curMax, d_I, gaes, s);
    auto d_drelu = res.first;
    auto d_diff = res.second;
    gpuReconstructInPlace((GPUGroupElement*) d_drelu, 1, k.numRelus, peer, party, s);
    auto d_max = gpuSelectForMaxpool(k.selectKey, d_drelu, d_diff, party, s);
    if(party == 0) gpuAddSharesInPlace(d_max, d_curMax, p.bout, k.selectKey.N);
    gpuFree(d_diff);
    gpuReconstructInPlace(d_max, p.bin, k.selectKey.N, peer, party, s);
    gpuAndForMaxpool(p, i * p.FH + j + 1, andKey, d_drelu, d_oneHot, party, s);
    int numBits = k.selectKey.N * p.FH * p.FW;
    gpuReconstructInPlace((GPUGroupElement*) d_oneHot, 1, numBits, peer, party, s);
    gpuFree(d_drelu);
    return d_max;
}

// no memory leak
GPUGroupElement* MaxPool2DLayer::forward(Peer* peer, int party, GPUGroupElement* d_I, AESGlobalContext* gaes) {
    int out_size = p.N * p.H * p.W * p.C;
    GPUGroupElement* d_curMax = (GPUGroupElement*) gpuMalloc(out_size * sizeof(GPUGroupElement));
    int num_ints = ((out_size * p.FH * p.FW - 1) / PACKING_SIZE + 1); 
    uint32_t* d_oneHot = (uint32_t*) gpuMalloc(num_ints * sizeof(uint32_t));
    for(int i = 0; i < p.FH; i++) {
        for(int j = 0; j < p.FW; j++) {
            if(i == 0 && j == 0) continue;
            auto d_curMaxNew = gpuMaxpool(p, peer, party, maxpoolKey.reluKey[i * p.FW + j], maxpoolKey.andKey[i * p.FW + j], i, j, d_I, d_curMax, d_oneHot, gaes, &s);
            if(d_curMax) gpuFree(d_curMax);
            d_curMax = d_curMaxNew;
        }
    }
    gpuFree(d_I);
    oneHot = (uint32_t*) moveToCPU((uint8_t*) d_oneHot, num_ints * sizeof(uint32_t), &s);
    gpuFree(d_oneHot);
    return d_curMax;
}

// no memory leak
GPUGroupElement* MaxPool2DLayer::backward(Peer* peer, int party, GPUGroupElement* d_incomingGrad, AESGlobalContext* g) {
    int incomingGradSize = p.N * p.H * p.W * p.C;
    int outgoingGradSize = p.N * p.imgH * p.imgW * p.C;
    int oneHotSize = incomingGradSize * p.FH * p.FW;
    int numInts = (oneHotSize - 1) / PACKING_SIZE + 1;
    auto d_oneHot = (uint32_t*) moveToGPU((uint8_t*) oneHot, numInts * sizeof(uint32_t), &s);
    auto d_outgoingGradExpanded = gpuSelectForMaxpoolBackprop(p, backpropSelectKey, d_oneHot, d_incomingGrad, 
                                                party, &s);
    gpuFree(d_incomingGrad);
    auto d_outgoingGrad = gpuCollectGradients(p, d_outgoingGradExpanded, &s);
    gpuReconstructInPlace(d_outgoingGrad, p.bout, outgoingGradSize, peer, party, &s);
    gpuFree(d_oneHot);
    gpuFree(d_outgoingGradExpanded);
    return d_outgoingGrad;
}