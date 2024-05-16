// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <cassert>
#include <cstdint>

#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/gpu_conv2d.h"
#include "nn/orca/conv2d_layer.h"


using T = u64;

using namespace dcf;
using namespace dcf::orca;

int main(int argc, char *argv[])
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    int bin = global::bw;  //64;
    int bout = global::bw; //64;
    int N = 128;
    int H = 32;
    int W = 32;
    int CI = 3;
    int FH = 5;         //11;//5;
    int FW = 5;         //11;//5;
    int CO = 64;        //96;//64;
    int zPadHLeft = 1;  //9;//1;
    int zPadHRight = 1; //9;//1;
    int zPadWLeft = 1;  //9;//1;
    int zPadWRight = 1; //9;//1;
    int strideH = 1;    //4;//1;
    int strideW = 1;    //4;//1;
    bool useMomentum = true;
    int epoch = 0;

    int party = atoi(argv[1]);
    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);
    auto conv2d_layer = Conv2DLayer<T>(bin, bout, N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, true, dcf::TruncateType::StochasticTruncate, dcf::TruncateType::StochasticTruncate, true, false);
    conv2d_layer.setTrain(useMomentum);

    T *h_I, *h_F, *h_b, *h_grad, *h_Vf, *h_Vb;

    auto d_mask_I = randomGEOnGpu<T>(conv2d_layer.p.size_I, bin);
    // checkCudaErrors(cudaMemset(d_mask_I, 0, conv2d_layer.p.size_I * sizeof(T)));
    auto d_masked_I = getMaskedInputOnGpu<T>(conv2d_layer.p.size_I, bin, d_mask_I, &h_I);
    auto d_mask_F = randomGEOnGpu<T>(conv2d_layer.p.size_F, bin);
    // checkCudaErrors(cudaMemset(d_mask_F, 0, conv2d_layer.p.size_F * sizeof(T)));
    auto h_masked_F = getMaskedInputOnCpu<T>(conv2d_layer.p.size_F, bin, d_mask_F, &h_F);
    auto d_mask_b = randomGEOnGpu<T>(CO, bin);
    // checkCudaErrors(cudaMemset(d_mask_b, 0, conv2d_layer.p.CO * sizeof(T)));
    auto h_masked_b = getMaskedInputOnCpu<T>(CO, bin, d_mask_b, &h_b);

    auto d_mask_grad = randomGEOnGpu<T>(conv2d_layer.p.size_O, bin);
    auto d_masked_grad = getMaskedInputOnGpu<T>(conv2d_layer.p.size_O, bin, d_mask_grad, &h_grad);
    auto d_mask_Vf = randomGEOnGpu<T>(conv2d_layer.p.size_F, bin);
    auto h_masked_Vf = getMaskedInputOnCpu<T>(conv2d_layer.p.size_F, bin, d_mask_Vf, &h_Vf);
    auto d_mask_Vb = randomGEOnGpu<T>(CO, bin);
    auto h_masked_Vb = getMaskedInputOnCpu<T>(CO, bin, d_mask_Vb, &h_Vb);

    moveIntoCPUMem((u8 *)conv2d_layer.mask_F, (u8 *)d_mask_F, conv2d_layer.p.size_F * sizeof(T), NULL);
    moveIntoCPUMem((u8 *)conv2d_layer.mask_Vf, (u8 *)d_mask_Vf, conv2d_layer.p.size_F * sizeof(T), NULL);
    moveIntoCPUMem((u8 *)conv2d_layer.mask_b, (u8 *)d_mask_b, CO * sizeof(T), NULL);
    moveIntoCPUMem((u8 *)conv2d_layer.mask_Vb, (u8 *)d_mask_Vb, CO * sizeof(T), NULL);

    auto startPtr = cpuMalloc(10 * OneGB);
    auto curPtr = startPtr;

    auto d_mask_C = conv2d_layer.genForwardKey(&curPtr, party, d_mask_I, &g);
    auto h_mask_C = (T *)moveToCPU((u8 *)d_mask_C, conv2d_layer.convKey.mem_size_O, NULL);
    printf("mask C=%lu, %lu\n", h_mask_C[0], h_mask_C[1]);
    auto d_mask_dI = conv2d_layer.genBackwardKey(&curPtr, party, d_mask_grad, &g, epoch);
    auto h_mask_dI = (T *)moveToCPU((u8 *)d_mask_dI, conv2d_layer.convKey.mem_size_I, NULL);

    // it says dF but actually means updated F
    auto h_mask_new_Vf = (T *)cpuMalloc(conv2d_layer.p.size_F * sizeof(T));
    auto h_mask_new_Vb = (T *)cpuMalloc(CO * sizeof(T));
    auto h_mask_new_F = (T *)cpuMalloc(conv2d_layer.p.size_F * sizeof(T));
    auto h_mask_new_b = (T *)cpuMalloc(CO * sizeof(T));

    memcpy(h_mask_new_Vf, conv2d_layer.mask_Vf, conv2d_layer.p.size_F * sizeof(T));
    memcpy(h_mask_new_Vb, conv2d_layer.mask_Vb, CO * sizeof(T));
    memcpy(h_mask_new_F, conv2d_layer.mask_F, conv2d_layer.p.size_F * sizeof(T));
    memcpy(h_mask_new_b, conv2d_layer.mask_b, CO * sizeof(T));

    curPtr = startPtr;
    conv2d_layer.readForwardKey(&curPtr);
    conv2d_layer.readBackwardKey(&curPtr, epoch);

    memcpy(conv2d_layer.F, h_masked_F, conv2d_layer.convKey.mem_size_F);
    memcpy(conv2d_layer.b, h_masked_b, CO * sizeof(T));
    auto d_masked_C = conv2d_layer.forward(peer, party, d_masked_I, &g);
    memcpy(conv2d_layer.Vf, h_masked_Vf, conv2d_layer.convKey.mem_size_F);
    memcpy(conv2d_layer.Vb, h_masked_Vb, CO * sizeof(T));
    auto d_masked_dI = conv2d_layer.backward(peer, party, d_masked_grad, &g, epoch);
    auto h_masked_C = (T *)moveToCPU((u8 *)d_masked_C, conv2d_layer.convKey.mem_size_O, NULL);
    auto h_masked_dI = (T *)moveToCPU((u8 *)d_masked_dI, conv2d_layer.convKey.mem_size_I, NULL);

    auto h_C_ct = gpuConv2DWrapper<T>(conv2d_layer.convKey, h_I, h_F, h_b, 0, true);
    printf("Checking C\n");
    checkTrStWithTol(bin, bout /* - scale*/, global::scale, conv2d_layer.p.size_O, h_masked_C, h_mask_C, h_C_ct);
    printf("Checking dI\n");
    auto h_dI_ct = gpuConv2DWrapper<T>(conv2d_layer.convKeydI, h_grad, h_F, NULL, 1, false);
    checkTrStWithTol<T>(bin, bout, global::scale, conv2d_layer.p.size_I, h_masked_dI, h_mask_dI, h_dI_ct);

    auto h_dF_ct = gpuConv2DWrapper<T>(conv2d_layer.convKeydF, h_grad, h_I, NULL, 2, false);
    printf("Checking sgd for F, momentum=%d\n", useMomentum);
    checkOptimizer(bin, bout, conv2d_layer.p.size_F, h_F, h_Vf, h_dF_ct, conv2d_layer.F, conv2d_layer.Vf,
                   h_mask_new_F, h_mask_new_Vf, global::scale, 2 * global::scale, 2 * global::scale, useMomentum, epoch);
    auto h_db_ct = getBiasGradWrapper(conv2d_layer.p.size_O / CO, CO, bout, h_grad);
    printf("Checking sgd for b, momentum=%d\n", useMomentum);
    checkOptimizer(bin, bout, CO, h_b, h_Vb, h_db_ct, conv2d_layer.b, conv2d_layer.Vb, h_mask_new_b, h_mask_new_Vb, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, useMomentum, epoch);
    return 0;
}