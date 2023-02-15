#include "gpu_data_types.h"
#include "gpu_truncate.h"
#include "input_prng.h"
#include "gpu_file_utils.h"
#include "gpu_fss_utils.h"
#include "gpu_comms.h"
#include "gpu_mem.h"
#include <cassert>

extern "C" void initAESContext(AESGlobalContext* g);

int LlamaConfig::bitlength = 64;
int LlamaConfig::party = DEALER;
int party = DEALER;
bool LlamaConfig::stochasticT = true;
bool LlamaConfig::stochasticRT = true;
int LlamaConfig::num_threads;
Peer* LlamaConfig::peer;
Peer* LlamaConfig::client;
Peer* LlamaConfig::server;
Dealer* LlamaConfig::dealer;
u64 accumulatedInputTimeOnline;
u64 accumulatedInputTimeOffline;

GPUGroupElement ars(GPUGroupElement x, int bin, int shift) {
    GPUGroupElement msb = (x & (1ULL << (bin - 1))) >> (bin - 1);
    GPUGroupElement signMask = (((1ULL << shift) - msb) << (64 - shift));
    x = (x >> shift) | signMask;
    // printf("%lu %lu %lu\n", msb, signMask, x);
    return x;
}

int main(int argc, char *argv[]) {
    prng.SetSeed(toBlock(0, time(NULL)));
    AESGlobalContext g;
    initAESContext(&g);
    int bin = 64;
    int bout = 64;
    int shift = 24;
    int N = atoi(argv[3]);
    // int f = 24;
    int party = atoi(argv[1]);

    GPUGroupElement *h_inputMask, *h_outputMask;
// check: have you reconstructed the masked output in the protocol?
    if(party == 0) {
        std::ofstream f1("stochastic_truncate_key1.dat"), f2("stochastic_truncate_key2.dat"); 
        h_inputMask = /*initWithConst(N, bin, 0);*/initRandom(N, bin);
        h_outputMask = (GPUGroupElement*) cpuMalloc(N * sizeof(GPUGroupElement));
        //initWithConst(N, bout, 0);//initRandom(N, bout);
        genGPUStochasticTruncateKey(f1, f2, bin, bout, shift, N, h_inputMask, h_outputMask);
    }
    Peer* peer = connectToPeer(party, argv[2]);
    size_t file_size;
    uint8_t* key_as_bytes = readFile("stochastic_truncate_key" + std::to_string(party+1) + ".dat", &file_size);
    auto d_I = getMaskedInput(N, bin, party, peer, h_inputMask);
    auto h_I = (GPUGroupElement*) moveToCPU((uint8_t*) d_I, N * sizeof(GPUGroupElement), NULL);
    auto k = readGPUSignExtendKey(&key_as_bytes);
    gpuStochasticTruncate(k, /*bin, bout,*/ shift, party, peer, d_I, &g, NULL);
    if(party == 0) {
        auto h_O = (GPUGroupElement*) moveToCPU((uint8_t*) d_I, N * sizeof(GPUGroupElement), NULL);
        // GPUGroupElement *unmasked_input = new GPUGroupElement[N];
        for(int i = 0; i < N; i++) {
            auto unmasked_input = (h_I[i] - h_inputMask[i]) & ((1ULL << bin) - 1);
            auto unmasked_output = h_O[i] - h_outputMask[i];
            auto truncatedInput = ars(unmasked_input, bin, shift);
            // if(i < 10) printf("%lu %lu %lu\n", unmasked_output, truncatedInput, h_outputMask[i]);
            assert(unmasked_output - truncatedInput <= 1);
            // auto truncatedMask = h_inputMask[i] >> shift;
            // assert(unmasked_output == truncatedInput);
            // auto dcfOutput = (((h_I[i] >> shift) + (1ULL << 39)) & ((1ULL << 40) - 1)) < truncatedMask;
            // printf("%d: %lu %lu %lu %lu %lu %d %lu %lu %lu\n", i, unmasked_output, truncatedInput, unmasked_input, h_inputMask[i], h_inputMask[i] >> 24, dcfOutput, h_I[i], (((h_I[i] >> shift) + (1ULL << 39)) & ((1ULL << 40) - 1)), ars(h_inputMask[i], bin, shift));
        }

    }

}