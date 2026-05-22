#include <llama/dcf.h>
#include <sytorch/random.h>
#include <sytorch/utils.h>
#include "fss/gpu_aes_shm.h"
#include "fss/gpu_fss_helper.h"

using namespace osuCrypto;

__global__ void aesWrapper(AESBlock *ss0, int N, AESGlobalContext gaes)
{
    AESSharedContext saes;
    loadSbox(&gaes, &saes);
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < N)
    {
        printf("AES seed=\n");
        printAESBlock(&ss0[threadId]);
        auto ss0_l = ss0[threadId];
        auto ss1_l = ss0_l;
        auto ss2_l = ss0_l;

        AESBlock ct[4];

        applyAESPRG(&saes, (u32*) &ss0_l, 0, (u32*) ct);
        printf("Encrypt one=\n");
        printAESBlock(ct);

        applyAESPRGTwoTimes(&saes, (u32*) &ss1_l, 0, (u32*) ct, (u32*) &ct[1]);
        printf("Encrypt two=\n");
        printAESBlock(ct);
        printAESBlock(&ct[1]);

        applyAESPRGFourTimes(&saes, (u32*) &ss2_l, (u32*) ct, (u32*) &ct[1], (u32*) &ct[2], (u32*) &ct[3]);
        printf("Encrypt four=\n");
        printAESBlock(ct);
        printAESBlock(&ct[1]);
        printAESBlock(&ct[2]);
        printAESBlock(&ct[3]);
    }
}

int main()
{
    sytorch_init();
    initGPUMemPool();
    osuCrypto::block aesSeed(prngWeights.get<block>());
    static const block ZeroBlock = toBlock(0, 0);
    static const block OneBlock = toBlock(0, 1);
    static const block TwoBlock = toBlock(0, 2);
    static const block ThreeBlock = toBlock(0, 3);
    osuCrypto::AES ak(aesSeed);
    osuCrypto::block ct[4];
    osuCrypto::block pt[4] = {ZeroBlock, TwoBlock, OneBlock, ThreeBlock};

    printf("AES seed=\n");
    printAESBlock((AESBlock*) &aesSeed);
    
    printf("Encrypt one=\n");
    ak.ecbEncBlock(pt[0], ct[0]);
    printAESBlock((AESBlock*) &ct[0]);
    
    printf("Encrypt two=\n");
    ak.ecbEncTwoBlocks(pt, ct);
    printAESBlock((AESBlock*) &ct[0]);
    printAESBlock((AESBlock*) &ct[1]);

    printf("Encrypt four=\n");
    ak.ecbEncFourBlocks(pt, ct);
    printAESBlock((AESBlock*) &ct[0]);
    printAESBlock((AESBlock*) &ct[1]);
    printAESBlock((AESBlock*) &ct[2]);
    printAESBlock((AESBlock*) &ct[3]);

    int N = 1;
    auto d_aesSeed = (AESBlock*) moveToGPU((u8*) &aesSeed, N * sizeof(AESBlock), (Stats*) NULL);
    AESGlobalContext g;
    initAESContext(&g);
    aesWrapper<<<(N - 1) / 256 + 1, 256>>>(d_aesSeed, N, g);
    checkCudaErrors(cudaDeviceSynchronize());
}
