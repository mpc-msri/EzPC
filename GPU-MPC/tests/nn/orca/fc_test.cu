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

#include "utils/gpu_mem.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_random.h"

#include "fss/gpu_matmul.h"
#include "nn/orca/fc_layer.h"

using T = u64;

using namespace dcf;
using namespace dcf::orca;

int main(int argc, char *argv[])
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    int bin = 64, bout = 64, M = 100, N = 10, K = 64;
    bool useMomentum = true;
    int epoch = 0;

    int party = atoi(argv[1]);
    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);

    auto fc_layer = FCLayer<T>(bin, bout, M, N, K, dcf::TruncateType::StochasticTruncate, dcf::TruncateType::StochasticTruncate, true, true, false);
    fc_layer.setTrain(useMomentum);
    T *h_X, *h_W, *h_Y, *h_Z, *h_grad, *h_Vw, *h_Vy;

    // check: have you reconstructed the masked output in the protocol?
    auto d_mask_X = randomGEOnGpu<T>(fc_layer.p.size_A, bin);
    auto d_masked_X = getMaskedInputOnGpu<T>(fc_layer.p.size_A, bin, d_mask_X, &h_X);
    auto d_mask_W = randomGEOnGpu<T>(fc_layer.p.size_B, bin);
    auto h_masked_W = getMaskedInputOnCpu<T>(fc_layer.p.size_B, bin, d_mask_W, &h_W);
    auto d_mask_Y = randomGEOnGpu<T>(N, bin);
    auto h_masked_Y = getMaskedInputOnCpu<T>(N, bin, d_mask_Y, &h_Y);

    auto d_mask_grad = randomGEOnGpu<T>(fc_layer.p.size_C, bin);
    auto d_masked_grad = getMaskedInputOnGpu<T>(fc_layer.p.size_C, bin, d_mask_grad, &h_grad);

    auto d_mask_Vw = randomGEOnGpu<T>(fc_layer.p.size_B, bin);
    auto h_masked_Vw = getMaskedInputOnCpu<T>(fc_layer.p.size_B, bin, d_mask_Vw, &h_Vw);
    auto d_mask_Vy = randomGEOnGpu<T>(N, bin);
    auto h_masked_Vy = getMaskedInputOnCpu<T>(N, bin, d_mask_Vy, &h_Vy);

    moveIntoCPUMem((u8 *)fc_layer.mask_W, (u8 *)d_mask_W, fc_layer.p.size_B * sizeof(T), NULL);
    moveIntoCPUMem((u8 *)fc_layer.mask_Vw, (u8 *)d_mask_Vw, fc_layer.p.size_B * sizeof(T), NULL);
    moveIntoCPUMem((u8 *)fc_layer.mask_Y, (u8 *)d_mask_Y, N * sizeof(T), NULL);
    moveIntoCPUMem((u8 *)fc_layer.mask_Vy, (u8 *)d_mask_Vy, N * sizeof(T), NULL);

    auto startPtr = cpuMalloc(5 * OneGB);
    auto curPtr = startPtr;

    auto d_mask_Z = fc_layer.genForwardKey(&curPtr, party, d_mask_X, &g);
    auto h_mask_Z = (T *)moveToCPU((u8 *)d_mask_Z, fc_layer.mmKey.mem_size_C, NULL);
    auto d_mask_dX = fc_layer.genBackwardKey(&curPtr, party, d_mask_grad, &g, epoch);
    auto h_mask_dX = (T *)moveToCPU((u8 *)d_mask_dX, fc_layer.mmKey.mem_size_A, NULL);

    auto h_mask_new_Vw = (T *)cpuMalloc(fc_layer.p.size_B * sizeof(T));
    auto h_mask_new_Vy = (T *)cpuMalloc(N * sizeof(T));
    auto h_mask_new_W = (T *)cpuMalloc(fc_layer.p.size_B * sizeof(T));
    auto h_mask_new_Y = (T *)cpuMalloc(N * sizeof(T));

    memcpy(h_mask_new_Vw, fc_layer.mask_Vw, fc_layer.p.size_B * sizeof(T));
    memcpy(h_mask_new_W, fc_layer.mask_W, fc_layer.p.size_B * sizeof(T));
    // uncomment for bias
    memcpy(h_mask_new_Vy, fc_layer.mask_Vy, N * sizeof(T));
    memcpy(h_mask_new_Y, fc_layer.mask_Y, N * sizeof(T));

    curPtr = startPtr;
    fc_layer.readForwardKey(&curPtr);
    fc_layer.readBackwardKey(&curPtr, epoch);

    memcpy(fc_layer.W, h_masked_W, fc_layer.mmKey.mem_size_B);
    memcpy(fc_layer.Y, h_masked_Y, N * sizeof(T));
    auto d_masked_Z = fc_layer.forward(peer, party, d_masked_X, &g);

    memcpy(fc_layer.Vw, h_masked_Vw, fc_layer.mmKey.mem_size_B);
    // uncommment for bias
    memcpy(fc_layer.Vy, h_masked_Vy, N * sizeof(T));

    auto d_masked_dX = fc_layer.backward(peer, party, d_masked_grad, &g, epoch);

    auto h_masked_Z = (T *)moveToCPU((u8 *)d_masked_Z, fc_layer.mmKey.mem_size_C, NULL);
    auto h_masked_dX = (T *)moveToCPU((u8 *)d_masked_dX, fc_layer.mmKey.mem_size_A, NULL);
    auto h_Z_ct = gpuMatmulWrapper<T>(fc_layer.p, h_X, h_W, h_Y, true);

    printf("Checking Z\n");
    checkTrStWithTol<T>(bin, bout, global::scale, fc_layer.p.size_C, h_masked_Z, h_mask_Z, h_Z_ct);

    auto h_dX_ct = gpuMatmulWrapper<T>(fc_layer.pdX, h_grad, h_W, NULL, false);
    printf("Checking dX\n");

    checkTrStWithTol<T>(bin, bout, global::scale, fc_layer.p.size_A, h_masked_dX, h_mask_dX, h_dX_ct);
    auto h_dW_ct = gpuMatmulWrapper<T>(fc_layer.pdW, h_X, h_grad, NULL, false);

    printf("Checking sgd for W, momentum=%d\n", useMomentum);
    checkOptimizer<T>(bin, bout, fc_layer.p.size_B, h_W, h_Vw, h_dW_ct, fc_layer.W, fc_layer.Vw,
                      h_mask_new_W, h_mask_new_Vw, global::scale, 2 * global::scale, 2 * global::scale, useMomentum, epoch);

    auto h_dY_ct = getBiasGradWrapper<T>(M, N, bout, h_grad);
    printf("Checking sgd for Y, momentum=%d\n", useMomentum);
    checkOptimizer<T>(bin, bout, N, h_Y, h_Vy, h_dY_ct, fc_layer.Y, fc_layer.Vy, h_mask_new_Y, h_mask_new_Vy, 2 * global::scale, 2 * global::scale - lr_scale[epoch], global::scale, useMomentum, epoch);
    return 0;
}
