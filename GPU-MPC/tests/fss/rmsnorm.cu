#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/gpu_layernorm.h"

#include <cassert>
#include <numeric>
#include <sytorch/backend/cleartext.h>
#include <sytorch/backend/llama_transformer.h>

using T = u64;

int main(int argc, char *argv[])
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    AvgPoolParams p;
    p.bw = 50;
    p.bin = 38;
    p.scale = 12;
    p.scaleDiv = 0;
    p.bwBackprop = 0;
    p.N = 1;
    p.imgH = atoi(argv[2]); //128 * 12;
    p.imgW = atoi(argv[3]); //128;
    p.C = 1;
    p.FH = 1;
    p.FW = p.imgW;
    p.strideH = 1;
    p.strideW = p.FW;
    p.zPadHLeft = 0;
    p.zPadHRight = 0;
    p.zPadWLeft = 0;
    p.zPadWRight = 0;
    initPoolParams(p);

    int party = atoi(argv[1]);
    // init llama here
    sytorch_init();
    auto llama = new LlamaTransformer<u64>();
    srand(time(NULL));

    const u64 scale = 12;

    LlamaConfig::bitlength = p.bw;
    LlamaConfig::party = 1;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    llama->init("0.0.0.0");
    int inSz = getInSz(p);

    T *h_I, *h_A, *h_B;
    auto d_mask_I = randomGEOnGpu<T>(inSz, p.bin);
    // checkCudaErrors(cudaMemset(d_mask_I, 0, inSz * sizeof(T)));
    auto d_mask_A = randomGEOnGpu<T>(p.imgW, p.bin);
    // checkCudaErrors(cudaMemset(d_mask_A, 0, p.imgW * sizeof(T)));
    auto d_mask_B = randomGEOnGpu<T>(p.imgW, p.bin);
    // checkCudaErrors(cudaMemset(d_mask_B, 0, p.imgW * sizeof(T)));
    // checkCudaErrors(cudaMemset(d_mask_I, 0, inSz * sizeof(T)));
    // auto h_mask_I = (T *)moveToCPU((u8 *)d_mask_I, inSz * sizeof(T), NULL);
    auto d_masked_I = getMaskedInputOnGpu(inSz, p.bw, d_mask_I, &h_I, true, 15);
    auto d_masked_A = getMaskedInputOnGpu(p.imgW, p.bw, d_mask_A, &h_A, true, 15);
    auto d_masked_B = getMaskedInputOnGpu(p.imgW, p.bw, d_mask_B, &h_B, true, 15);

    printf("A=%ld, B=%ld, I=%ld, %ld, %ld, %ld\n", h_A[0], h_B[0], h_I[0], h_I[1], h_I[2], h_I[3]);
    u8 *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 8 * OneGB);
    llama::start();
    auto d_mask_O = gpuKeygenLayerNorm(&curPtr, party, p, d_mask_A, d_mask_B, d_mask_I, &g, false);
    llama::end();
    auto h_mask_O = (T *)moveToCPU((u8 *)d_mask_O, inSz * sizeof(T), NULL);
    llama->finalize();

    auto k = readGPULayerNormKey<T>(p, &startPtr, false);
    Stats s;
    T *d_O;
    auto peer = new GpuPeer(true);
    for (int i = 0; i < 1; i++)
    {
        LlamaConfig::party = party + 2;
        llama->init(argv[4]);
        peer->peer = LlamaConfig::peer;
        s.compute_time = 0;
        s.comm_time = 0;
        s.transfer_time = 0;
        llama::start();
        d_O = gpuLayerNorm(peer, party, p, k, d_masked_A, d_masked_B, d_masked_I, NULL, &g, (Stats *)&s, false);
        printf("Layernorm time=%lu micros\n", s.compute_time);
        printf("Comm time=%lu micros\n", s.comm_time);
        printf("Transfer time=%lu micros\n", s.transfer_time);
        llama::end();
        llama->finalize();
    }
    unmaskValues(p.bw, inSz, d_O, d_mask_O, NULL);
    auto h_O = (T *)moveToCPU((u8 *)d_O, inSz * sizeof(T), NULL);
    
    auto ct = new ClearText<i64>();
    ct->bw = p.bw;

    Tensor<i64> t((i64 *)h_I, {(u64)p.imgH, (u64)p.imgW});
    ct->rmsnorm(Tensor1D<i64>((i64 *)h_A, (u64)p.imgW), Tensor1D<i64>((i64 *)h_B, (u64)p.imgW), t, t, p.scale); //(t, t, p.scale, 0);
    ct->Backend<i64>::truncate(t, (u64)p.scale);
    for (int i = 0; i < inSz; i++)
    {
        if(i < 10) printf("Index %d=%ld, %ld\n", i, t.data[i], h_O[i]);
        if (T(t.data[i]) != h_O[i])
        {
            printf("Index %d=%ld, %ld\n", i, t.data[i], h_O[i]);
            assert(0);
            // break;
        }
    }
    return 0;
}