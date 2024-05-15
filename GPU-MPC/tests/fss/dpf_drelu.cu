#include <cassert>
#include <sytorch/tensor.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_relu.h"

using T = u64;

int main(int argc, char *argv[])
{
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bw = 64;
    int N = atoi(argv[3]); 
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);

    uint8_t *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 10 * OneGB);

    initGPURandomness();
    auto d_mask_X = randomGEOnGpu<T>(N, bw);
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    T *h_X;
    auto d_masked_X = getMaskedInputOnGpu(N, bw, d_mask_X, &h_X);

    auto d_dreluMask = gpuKeyGenDRelu(&curPtr, party, bw, N, d_mask_X, &g);
    auto h_mask_O = (T *)moveToCPU((u8 *)d_dreluMask, N * sizeof(T), NULL);

    curPtr = startPtr;
    auto k = readGPUDReluKey(&curPtr);
    u32 *d_O;
    for (int i = 0; i < 1; i++)
    {
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<u32 *> h_mask({k.mask});
        Stats s;
        d_O = gpuDcf<T, 1, dReluPrologue<0>, dReluEpilogue<0, false>>(k.dpfKey, party, d_masked_X, &g, &s, &h_mask);
        peer->reconstructInPlace(d_O, 1, N, (Stats*) &s);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
        printf("Comm time=%lu micros, Transfer time=%lu micros, Comm=%lu B\n", s.comm_time, s.transfer_time, peer->bytesSent() + peer->bytesReceived());
    }
    printf("Mem size out=%lu\n", k.dpfKey.memSzOut);
    auto h_O = (u32 *)moveToCPU((uint8_t *)d_O, k.dpfKey.memSzOut, (Stats *)NULL);
    gpuFree(d_O);
    destroyGPURandomness();

    printf("bw=%d, N=%d\n", bw, N);
    for (int i = 0; i < N; i++)
    {
        auto drelu = h_X[i] < (1ULL << (bw - 1));
        auto o = ((h_O[i / 32] >> (i % 32)) & 1) ^ uint32_t(h_mask_O[i]);
        if (i < 10 || o != drelu)
            printf("%d: %ld, %ld, %ld, %ld, %u\n", i, h_X[i], drelu, o, h_mask_X[i], uint32_t(h_mask_O[i]));
        assert(o == drelu);

    }
    return 0;
}