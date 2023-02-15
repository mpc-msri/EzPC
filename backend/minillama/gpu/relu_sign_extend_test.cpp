#include "gpu_data_types.h"
#include "relu_sign_extend_layer.h"
#include <../input_prng.h>
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include <cassert>
#include "cpu_fss.h"

extern "C" void initAESContext(AESGlobalContext* g);

// int LlamaConfig::bitlength = 64;
// int LlamaConfig::party = DEALER;
// int party = DEALER;
// bool LlamaConfig::stochasticT = true;
// bool LlamaConfig::stochasticRT = true;
// int LlamaConfig::num_threads;
// Peer* LlamaConfig::peer;
// Peer* LlamaConfig::client;
// Peer* LlamaConfig::server;
// Dealer* LlamaConfig::dealer;
// u64 accumulatedInputTimeOnline;
// u64 accumulatedInputTimeOffline;

// GPUGroupElement mod(GPUGroupElement x, int bw) {
//     return x & ((1ULL << bw) - 1);
// }

int main(int argc, char *argv[]) {
    prng.SetSeed(osuCrypto::toBlock(0, time(NULL)));
    initCPURandomness();
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 40;
    int bout = 64;
    int N = 1638400;
    // int f = 24;
    int party = atoi(argv[1]);
    printf("boo\n");
    auto relu_sign_extend_layer = ReluSignExtendLayer(bin, bout, N);
    printf("boo\n");
    GPUGroupElement *h_inputMask, *h_outputMask, *h_incomingGradMask, *h_outgoingGradMask;
    GPUGroupElement *h_I, *h_O, *h_incomingGrad, *h_outgoingGrad;
// check: have you reconstructed the masked output in the protocol?
    if(party == 0) {
        std::ofstream f1("relu_sign_extend_key1.dat"), f2("relu_sign_extend_key2.dat"); 
        h_inputMask = /*initWithConst(N, bin, 0);*/initRandom(N, bin);
        h_outputMask = new GPUGroupElement[N]; /*initWithConst(N, bout, 0);initRandom(N, bout);*/
        h_incomingGradMask = /*initWithConst(N, bout, 6);*/initRandom(N, bout);
        h_outgoingGradMask = new GPUGroupElement[N];/*initWithConst(N, bout, 0);initRandom(N, bout);*/
        relu_sign_extend_layer.genForwardKey(f1, f2, h_inputMask, /*h_dReluMask, h_dcfMask,*/ h_outputMask/*, h_incomingGradMask, h_outgoingGradMask*/);
        relu_sign_extend_layer.genBackwardKey(f1, f2, /*h_dReluMask,*/ h_incomingGradMask, h_outgoingGradMask);
        f1.close();
        f2.close();
                // printf("here\n");
    }
    Peer* peer = connectToPeer(party, argv[2]);
    size_t file_size;
    uint8_t* key_as_bytes = readFile("relu_sign_extend_key" + std::to_string(party+1) + ".dat", &file_size);
    relu_sign_extend_layer.readForwardKey(&key_as_bytes);
    relu_sign_extend_layer.readBackwardKey(&key_as_bytes);
    auto d_masked_I = getMaskedInputOnGpu(N, bin, party, peer, h_inputMask, &h_I);
    relu_sign_extend_layer.forward(peer, party, d_masked_I, &g);
    auto d_maskedIncomingGrad = getMaskedInputOnGpu(N, bout, party, peer, h_incomingGradMask, &h_incomingGrad);
    auto d_maskedOutgoingGrad = relu_sign_extend_layer.backward(peer, party, d_maskedIncomingGrad, &g);
    if(party == 0) {
        auto h_masked_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_masked_I, N * sizeof(GPUGroupElement), NULL);
        for(int i = 0; i < N; i++) {
            auto unmasked_output = h_masked_O[i] - h_outputMask[i];
            auto relu = (h_I[i] < (1ULL << (bin - 1)) ? h_I[i] : 0);
            assert(unmasked_output == relu);
        }
        auto h_maskedOutgoingGrad = (GPUGroupElement*) moveToCPU((uint8_t*) d_maskedOutgoingGrad, N * sizeof(GPUGroupElement), NULL);
        for(int i = 0; i < N; i++) {
            auto outgoingGradCt = (h_I[i] < (1ULL << (bin - 1)) ? h_incomingGrad[i] : 0);
            auto outgoingGrad = h_maskedOutgoingGrad[i] - h_outgoingGradMask[i];
            mod(outgoingGrad, bout);
            assert(outgoingGrad == outgoingGradCt);
            if(i < 10) printf("%lu %lu\n", outgoingGrad, outgoingGradCt);
            // printf("%d: %lu %lu %lu %lu %lu\n", i, unmasked_input[i], incomingGrad, outgoingGrad, outgoingGradCt, h_outgoingGradMask[i]);
        }
    }

}