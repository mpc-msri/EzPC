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

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/dcf/gpu_dcf.h"
#include "fss/dcf/gpu_truncate.h"

#include "nn/orca/relu_layer.h"

using T = u64;

using namespace dcf;
using namespace dcf::orca;

int main(int argc, char *argv[]) {
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();

    int bin = 40;//64;
    int bout = 64;//64;
    int shift = 24;//24;
    int N = atoi(argv[3]);
    bool useMomentum = true;
    int epoch = 0;

    int party = atoi(argv[1]);
    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);

    auto relu_layer = ReluLayer<T>(bin - shift, bout, N);
    relu_layer.setTrain(useMomentum);
    T *h_I, *h_incomingGrad;

    u8* startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);

    auto d_inputMask = randomGEOnGpu<T>(N, bin);
    auto d_masked_I = getMaskedInputOnGpu(N, bin, d_inputMask, &h_I);
    if(shift > 0) gpuLocalTr<T, T, ars>(party, bin, shift, N, d_inputMask, true);
    auto d_outputMask = relu_layer.genForwardKey(&curPtr, party, d_inputMask, &g);
    auto h_outputMask = (T*) moveToCPU((u8*) d_outputMask, N * sizeof(T), NULL);
    
    
    auto d_incomingGradMask = randomGEOnGpu<T>(N, bout);
    auto d_maskedIncomingGrad = getMaskedInputOnGpu(N, bout, d_incomingGradMask, &h_incomingGrad);
    if(shift > 0) gpuLocalTr<T, T, ars>(party, bin, shift, N, d_incomingGradMask, true);
    auto d_outgoingGradMask = relu_layer.genBackwardKey(&curPtr, party, d_incomingGradMask, &g, epoch);
    auto h_outgoingGradMask = (T*) moveToCPU((u8*) d_outgoingGradMask, N * sizeof(T), NULL);

    curPtr = startPtr;

    relu_layer.readForwardKey(&curPtr);
    relu_layer.readBackwardKey(&curPtr, epoch);

    if(shift > 0) gpuLocalTr<T, T, ars>(party, bin, shift, N, d_masked_I, true);
    auto d_masked_O = relu_layer.forward(peer, party, d_masked_I, &g);
    if(shift > 0) gpuLocalTr<T, T, ars>(party, bin, shift, N, d_maskedIncomingGrad, true);
    auto d_maskedOutgoingGrad = relu_layer.backward(peer, party, d_maskedIncomingGrad, &g, epoch);
    auto h_masked_O = (T*) moveToCPU((u8*) d_masked_O, N * sizeof(T), NULL);
    auto h_masked_grad = (T*) moveToCPU((u8*) d_maskedOutgoingGrad, N * sizeof(T), NULL);
    int forwardCount = 0, backwardCount = 0;
    for(int i = 0; i < N; i++) {
        auto unmasked_o = h_masked_O[i] - h_outputMask[i];
        cpuMod(unmasked_o, bout);
        auto shifted_I = h_I[i] >> shift;
        int dReLU = (h_I[i] < (T(1) << (bin - 1)));
        auto relu = dReLU * shifted_I;
        if(i < 10) printf("%lu %lu %lu\n", h_I[i], u64(relu), u64(unmasked_o));
        // assert(dReLU * h_I[i] == unmasked_o);
        int64_t tol = 1;
        if(abs(static_cast<int64_t>(relu - unmasked_o)) > tol) forwardCount++;
        auto shifted_grad = h_incomingGrad[i] >> shift;
        auto unmasked_grad = h_masked_grad[i] - h_outgoingGradMask[i];
        cpuMod(unmasked_grad, bout);
        auto relu_back = dReLU * shifted_grad;
        if(i < 10) printf("%lu %lu\n", relu_back, unmasked_grad);
        // assert(dReLU * h_incomingGrad[i] == unmasked_grad);
        if(abs(static_cast<int64_t>(relu_back - unmasked_grad)) > tol) backwardCount++;
    }
    printf("Num errors forward=%d\n", forwardCount);
    printf("Num errors backward=%d\n", backwardCount);
    return 0;
}
