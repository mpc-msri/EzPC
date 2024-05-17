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


#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "nn/orca/relu_extend_layer.h"

#include <cassert>

using T = u64;

using namespace dcf;
using namespace dcf::orca;

int main(int argc, char *argv[]) {
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();

    int bin = 40;
    int bout = 64;
    int N = atoi(argv[3]);//1638400;
    bool useMomentum = true;
    int epoch = 0;

    int party = atoi(argv[1]);
    auto peer = new GpuPeer(false);
    peer->connect(party, argv[2]);
    
    auto relu_extend_layer = ReluExtendLayer<T>(bin, bout, N);
    relu_extend_layer.setTrain(useMomentum);
    T *h_I, *h_incomingGrad;

    u8* startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 4 * OneGB);

    auto d_inputMask = randomGEOnGpu<T>(N, bin);
    // checkCudaErrors(cudaMemset(d_inputMask, 0, N * sizeof(T)));
    auto d_masked_I = getMaskedInputOnGpu(N, bin, d_inputMask, &h_I);
    auto d_outputMask = relu_extend_layer.genForwardKey(&curPtr, party, d_inputMask, &g);
    auto h_outputMask = (T*) moveToCPU((u8*) d_outputMask, N * sizeof(T), NULL);
    
    
    auto d_incomingGradMask = randomGEOnGpu<T>(N, bout);
    auto d_maskedIncomingGrad = getMaskedInputOnGpu(N, bout, d_incomingGradMask, &h_incomingGrad);
    auto h_incomingGradMask = (T*) moveToCPU((u8*) d_incomingGradMask, N * sizeof(T), NULL);
    auto d_outgoingGradMask = relu_extend_layer.genBackwardKey(&curPtr, party, d_incomingGradMask, &g, epoch);
    auto h_outgoingGradMask = (T*) moveToCPU((u8*) d_outgoingGradMask, N * sizeof(T), NULL);

    curPtr = startPtr;
    relu_extend_layer.readForwardKey(&curPtr);
    relu_extend_layer.readBackwardKey(&curPtr, epoch);
    
    auto d_masked_O = relu_extend_layer.forward(peer, party, d_masked_I, &g);
    auto d_maskedOutgoingGrad = relu_extend_layer.backward(peer, party, d_maskedIncomingGrad, &g, epoch);
    auto h_masked_O = (T*) moveToCPU((u8*) d_masked_O, N * sizeof(T), NULL);
    for(int i = 0; i < N; i++) {
        auto unmasked_output = h_masked_O[i] - h_outputMask[i];
        auto relu = (h_I[i] < (T(1) << (bin - 1)) ? h_I[i] : 0);
        if(i < 10 || unmasked_output != relu) printf("%d: %lu, %lu %lu\n", i, h_I[i], u64(unmasked_output), u64(relu));
        assert(unmasked_output == relu);
    }
    // printf("\n");
    auto h_maskedOutgoingGrad = (T*) moveToCPU((u8*) d_maskedOutgoingGrad, N * sizeof(T), NULL);
    for(int i = 0; i < N; i++) {
        auto outgoingGradCt = (h_I[i] < (T(1) << (bin - 1)) ? h_incomingGrad[i] : 0);
        auto outgoingGrad = h_maskedOutgoingGrad[i] - h_outgoingGradMask[i];
        cpuMod(outgoingGrad, bout);
        if(i < 10) printf("%lu %lu\n", u64(outgoingGrad), u64(outgoingGradCt));
        assert(outgoingGrad == outgoingGradCt);
    }
    return 0;
}