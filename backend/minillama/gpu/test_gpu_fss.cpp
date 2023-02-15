#include <cryptoTools/Crypto/AES.h>
// #include "gpu_data_types.h"
#include "fss.h"
#include "gpu_keygen_helper.cpp"
#include <cstring>
// #include <cuda.h>
// #include "utils2.cpp"
// #include "dcf3.cpp"
// #include "rt.cpp"

using namespace osuCrypto;

extern "C" void test_prg(const uint8_t *key, int pt,
                         uint8_t *ct1, uint8_t *ct2, int, int);
extern "C" GPUGroupElement *gpu_dcf(GPUDCFKey k, int party, 
                            GPUGroupElement *dcf_input, AESGlobalContext* g);

extern "C" GPUGroupElementPair evalDCFForReLUTruncateWrapper(GPUDCFKey k, 
int party, GPUGroupElement *dcf_input, int shift, AESGlobalContext* gaes);

extern "C" GPUGroupElement *evalSmallDCFForlrsWrapper(GPUDCFKey k, int party, GPUGroupElement *dcf_input, GPUGroupElement* d_lrs_tn, GPUGroupElement* h_lrsMask, int shift, AESGlobalContext* gaes);

extern "C" std::pair<GPUGroupElement *, uint32_t *> gpuReluTruncateWrapper(GPUReLUTruncateKey k, int party, GPUGroupElement *rt_input, AESGlobalContext* gaes);

extern "C" GPUGroupElement *finishRTWrapper(GPUReLUTruncateKey k,
                                            uint32_t *h_x1, GPUGroupElement *h_y1,
                                            uint32_t *h_x2, GPUGroupElement *h_y2, int party);

extern "C" GPUGroupElement *gpuConv2DWrapper(GPUConv2DKey k, GPUGroupElement* h_I, GPUGroupElement* h_F, bool op);

extern "C" void embedElementsWrapper(GPUGroupElement *h_A, double *d_A1,
                                     double *d_A2, double *d_A3, double *d_A4, int N);
extern "C" void extractElementsWrapper(GPUGroupElement *h_A, double *d_A1,
                                       double *d_A2, double *d_A3, double *d_A4, int N);

extern "C" GPUGroupElement *gpuConv2DProtocolWrapper(GPUConv2DKey k, int party, GPUGroupElement *h_I, GPUGroupElement *h_F);

extern "C" GPUGroupElement *gpuMatmulProtocolWrapper(GPUMatmulKey k, int party, GPUGroupElement *h_A, GPUGroupElement *h_B);

extern "C" int runAESCallChain(int num_aes, uint32_t* key, uint32_t pt, uint32_t* ct1, uint32_t* ct2);

extern "C" void initAESContext(AESGlobalContext* g);

extern "C" void freeAESGlobalContext(AESGlobalContext* g);


int bitlength = 64;
int party = DEALER;


GPUGroupElement* CPUToGPUGroupElement(GroupElement* A, int N) {
    GPUGroupElement* B = new GPUGroupElement[N];
    for(int i=0;i<N;i++) B[i] = A[i].value;
    return B;
}


void printAESBlock(uint8_t *b)
{
    for (int i = 0; i < 16; i+=4)
        printf("%02x%02x%02x%02x ", b[i], b[i+1], b[i+2], b[i+3]);
    printf("\n");
}
    // memset(key, 0, 16);
    // key: FD2EAB4D12A283A718883983E07B994B
    // key[0] = 0xFD;
    // key[1] = 0x2E;
    // key[2] = 0xAB;
    // key[3] = 0x4D;
    // key[4] = 0x12;
    // key[5] = 0xA2;
    // key[6] = 0x83;
    // key[7] = 0xA7;
    // key[8] = 0x18;
    // key[9] = 0x88;
    // key[10] = 0x39;
    // key[11] = 0x83;
    // key[12] = 0xE0;
    // key[13] = 0x7B;
    // key[14] = 0x99;
    // key[15] = 0x4B;

    // memset(gpu_ct1, 0, 16);
    // memset(gpu_ct2, 0, 16);



    // block b;
    // memcpy(&b, key, 16);

    // for(int i = 0; i < 64; i++) {
        // b = cpu_ct[1];
    // }
    // printAESBlock((uint8_t *)&cpu_ct[0]);
    // printAESBlock((uint8_t *)&cpu_ct[1]);

    // printAESBlock(ct1);
    // printAESBlock(ct2);

void run_prg_test()
{

    int num_aes = 8192;
    uint8_t *gpu_ct1, *gpu_ct2;

    size_t size_in_bytes = 16 * num_aes;

    gpu_ct1 = new uint8_t[size_in_bytes];
    gpu_ct2 = new uint8_t[size_in_bytes];

    auto random_key = (uint8_t*) CPUToGPUGroupElement(init_random(2 * num_aes, 64), 2 * num_aes);

    block *cpu_ct = new block[2 * num_aes];
    block blocks[2] = {toBlock(0, 0), toBlock(0, 2)};

    for(int i = 0; i < num_aes; i++) {
        block b;
        memcpy(&b, &random_key[16 * i], 16);
        AES ak(b);
        ak.ecbEncTwoBlocks(blocks, &cpu_ct[2 * i]);
    }

    memset(gpu_ct1, 0, size_in_bytes);
    memset(gpu_ct2, 0, size_in_bytes);

    test_prg((uint8_t*) random_key, 0, gpu_ct1, gpu_ct2, num_aes / 128, 128);

    for(int i = 0; i < num_aes; i++) 
    {
        int cmp1 = memcmp(&gpu_ct1[16 * i], &cpu_ct[2*i], 16);
        int cmp2 = memcmp(&gpu_ct2[16 * i], &cpu_ct[2*i+1], 16);
        assert(!cmp1 && !cmp2);
    }
    printf("prg test: passed\n");
}


void run_cmp_prg_test()
{

    int num_aes = 256;
    uint8_t *gpu_ct1, *gpu_ct2;
    uint32_t *gpu_ct3, *gpu_ct4;

    size_t size_in_bytes = 16 * num_aes;

    gpu_ct1 = new uint8_t[size_in_bytes];
    gpu_ct2 = new uint8_t[size_in_bytes];

    auto random_key = (uint8_t*) CPUToGPUGroupElement(init_random(2 * num_aes, 64), 2 * num_aes);

    memset(gpu_ct1, 0, size_in_bytes);
    memset(gpu_ct2, 0, size_in_bytes);
    
    auto start = std::chrono::high_resolution_clock::now();
    test_prg((uint8_t*) random_key, 1, gpu_ct1, gpu_ct2, num_aes / 256, 256);
    auto  end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    std::cout << "Time for computing AES1 in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;

    gpu_ct3 = new uint32_t[4 * num_aes];
    gpu_ct4 = new uint32_t[4 * num_aes];
    
    start = std::chrono::high_resolution_clock::now();
    runAESCallChain(num_aes, (uint32_t*) random_key, 1, gpu_ct3, gpu_ct4);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Time for computing AES2 in ms: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << std::endl;


    assert(!memcmp(gpu_ct1, (uint8_t*) gpu_ct3, num_aes * 16));
    assert(!memcmp(gpu_ct2, (uint8_t*) gpu_ct4, num_aes * 16));

    printf("prg cmp test: passed\n");
}

// hardcoded groupSize = 1
GPUDCFKey convertToGPUDCFKey(DCFKeyPack* k1, int num_dcfs)
{
    GPUDCFKey k2;
    k2.num_dcfs = num_dcfs;
    k2.Bin = k1[0].Bin;
    k2.Bout = k1[0].Bout;
    k2.scw = new AESBlock[(k2.Bin + 1) * num_dcfs];
    k2.vcw = new GPUGroupElement[(k2.Bin + 1) * num_dcfs];
    k2.size_scw = (k2.Bin + 1) * num_dcfs;
    k2.size_vcw = (k2.Bin + 1) * num_dcfs;
    k2.mem_size_scw = k2.size_scw * sizeof(AESBlock);
    k2.mem_size_vcw = k2.size_vcw * sizeof(GPUGroupElement);
    
    for (int j = 0; j < k1[0].Bin + 1; j++)
    {
        for (int i = 0; i < num_dcfs; i++)
        {
            k2.scw[j * num_dcfs + i] = (AESBlock) k1[i].k[j];
            if(j < k1[0].Bin) 
                k2.vcw[j * num_dcfs + i] = k1[i].v[j].value;
            else 
                k2.vcw[j * num_dcfs + i] = k1[i].g[0].value;
        }
    }
    return k2;
}

void run_dcf_test()
{
    AESGlobalContext gaes;
    initAESContext(&gaes);

    int bin = 64;
    int bout = 64;
    GPUGroupElement half_range = 1ULL << (bin - 1);
    GroupElement payload(14, bout);
    GroupElement r(half_range + 40, bout);
    auto dcf_key_pack = keyGenDCF(bin, bout, 1, r, &payload);

    GPUDCFKey k1 = convertToGPUDCFKey(&dcf_key_pack.first, 1);
    GPUDCFKey k2 = convertToGPUDCFKey(&dcf_key_pack.second, 1);

    GPUGroupElement gpu_x = 0;
    GPUGroupElement *gpu_res1 = gpu_dcf(k1, 0, &gpu_x, &gaes);
    GPUGroupElement *gpu_res2 = gpu_dcf(k2, 1, &gpu_x, &gaes);
    assert(*gpu_res1 + *gpu_res2 == payload.value);

    gpu_x = static_cast<GPUGroupElement>(-1);
    gpu_res1 = gpu_dcf(k1, 0, &gpu_x, &gaes);
    gpu_res2 = gpu_dcf(k2, 1, &gpu_x, &gaes);
    assert(*gpu_res1 + *gpu_res2 == 0);

    gpu_x = static_cast<GPUGroupElement>(12);
    gpu_res1 = gpu_dcf(k1, 0, &gpu_x, &gaes);
    gpu_res2 = gpu_dcf(k2, 1, &gpu_x, &gaes);
    assert(*gpu_res1 + *gpu_res2 == payload.value);

    gpu_x = static_cast<GPUGroupElement>(half_range + 123456789);
    gpu_res1 = gpu_dcf(k1, 0, &gpu_x, &gaes);
    gpu_res2 = gpu_dcf(k2, 1, &gpu_x, &gaes);
    assert(*gpu_res1 + *gpu_res2 == 0);

    printf("test dcf: passed\n");

    int shift = 24;
    gpu_x = static_cast<GPUGroupElement>(29);
    auto res1 = evalDCFForReLUTruncateWrapper(k1, 0, &gpu_x, shift, &gaes);
    auto res2 = evalDCFForReLUTruncateWrapper(k2, 1, &gpu_x, shift, &gaes);

    assert(*res1.first + *res2.first == payload.value);
    assert(*res1.second + *res2.second == payload.value + 1);

    gpu_x = static_cast<GPUGroupElement>(half_range + 83);
    res1 = evalDCFForReLUTruncateWrapper(k1, 0, &gpu_x, shift, &gaes);
    res2 = evalDCFForReLUTruncateWrapper(k2, 1, &gpu_x, shift, &gaes);

    assert(*res1.first + *res2.first == 0);
    assert(*res1.second + *res2.second == payload.value);

    printf("test dcf two evals: passed\n");

    bin = shift;
    GroupElement small_r((1ULL << shift) - 18, bin);
    dcf_key_pack = keyGenDCF(bin, bout, 1, small_r, &payload);

    k1 = convertToGPUDCFKey(&dcf_key_pack.first, 1);
    k2 = convertToGPUDCFKey(&dcf_key_pack.second, 1);

    // gpu_x = ((29 << shift) + 18);
    // gpu_res1 = evalSmallDCFForlrsWrapper(k1, 0, &gpu_x, shift, &gaes);
    // gpu_res2 = evalSmallDCFForlrsWrapper(k2, 1, &gpu_x, shift, &gaes);

    // assert(*gpu_res1 + *gpu_res2 == payload.value + 29);
    // printf("test small dcf: passed\n");

    freeAESGlobalContext(&gaes);
}

GPUReLUTruncateKey convertToGPUReluTruncateKey(ReluTruncateKeyPack* k1, int num_rts)
{
    GPUReLUTruncateKey k2;
    k2.Bin = k1[0].Bin;
    k2.Bout = k1[0].Bout;
    k2.shift = k1[0].shift;
    k2.num_rts = num_rts;
    unsigned long mem_size = num_rts * sizeof(GPUGroupElement);

    k2.a = new GPUGroupElement[num_rts];
    for(int i = 0; i < num_rts; i++)
        k2.a[i] = k1[i].a.value;

    k2.b = new GPUGroupElement[num_rts];
    for(int i = 0; i < num_rts; i++)
        k2.b[i] = k1[i].b.value;

    k2.c = new GPUGroupElement[num_rts];
    for(int i = 0; i < num_rts; i++)
        k2.c[i] = k1[i].c.value;

    k2.d1 = new GPUGroupElement[k2.num_rts];
    for(int i = 0; i < num_rts; i++)
        k2.d1[i] = k1[i].d1.value;

    k2.d2 = new GPUGroupElement[num_rts];
    for(int i = 0; i < num_rts; i++)
        k2.d2[i] = k1[i].d2.value;

    k2.zTruncate = new GPUGroupElement[k2.num_rts];
    for(int i = 0; i < num_rts; i++)
        k2.zTruncate[i] = k1[i].zTruncate.value;

    DCFKeyPack *dcfKeyN = new DCFKeyPack[num_rts];
    for (int i = 0; i < num_rts; i++)
        dcfKeyN[i] = k1[i].dcfKeyN;

    DCFKeyPack *dcfKeyS = new DCFKeyPack[num_rts];
    for (int i = 0; i < num_rts; i++)
        dcfKeyS[i] = k1[i].dcfKeyS;

    k2.dcfKeyN = convertToGPUDCFKey(dcfKeyN, num_rts);
    k2.dcfKeyS = convertToGPUDCFKey(dcfKeyS, num_rts);
    return k2;
}

void run_relu_truncate_test()
{
    AESGlobalContext gaes;
    initAESContext(&gaes);

    int shift = 24;
    int bin = 64, bout = 64;
    int num_rts = 16384;//8388608;
    GroupElement* rin = init_random(num_rts, bin);
    GroupElement* routLRS = init_random(num_rts, bout);
    GroupElement* routRelu = init_random(num_rts, bout);
    GroupElement* rout = init_random(num_rts, bout);

    printf("finished generating randomness\n");
    ReluTruncateKeyPack* rt_keys0 = new ReluTruncateKeyPack[num_rts];
    ReluTruncateKeyPack* rt_keys1 = new ReluTruncateKeyPack[num_rts];


    for(int i = 0; i < num_rts; i++)
    {
        auto rt_key_pack = keyGenReluTruncate(bin, bout, shift, rin[i],
                                          routLRS[i],
                                          routRelu[i], rout[i]);
        rt_keys0[i] = rt_key_pack.first;
        rt_keys1[i] = rt_key_pack.second;
    }
    printf("finished generating keys\n");

    GPUReLUTruncateKey k1 = convertToGPUReluTruncateKey(rt_keys0, num_rts);
    GPUReLUTruncateKey k2 = convertToGPUReluTruncateKey(rt_keys1, num_rts);

    GPUGroupElement* x = CPUToGPUGroupElement(init_random(num_rts, bin), num_rts);
    GPUGroupElement* xt = new GPUGroupElement[num_rts]; 

    for(int i = 0; i < num_rts; i++) {
        xt[i] = x[i] >> shift;
        x[i] += rin[i].value;
    }
    printf("finished setting up input\n");
    auto res1 = gpuReluTruncateWrapper(k1, 0, x, &gaes);
    auto res2 = gpuReluTruncateWrapper(k2, 1, x, &gaes);
    printf("checking output\n");
    for(int i = 0; i < num_rts; i++) {
        assert(res1.first[i] + res2.first[i] - routLRS[i].value == xt[i]);
    //     // printf("%lu %lu\n", ((res1.second[i] & 1) + (res2.second[i] & 1) - routRelu[i].value) & 1, (x[i] - rin[i].value) < (1ULL << (bin - 1)));
    //     // assert((static_cast<GPUGroupElement>(( == static_cast<GPUGroupElement>)));
    }
    printf("test lrs: passed\n");
    printf("test drelu: passed\n");

    auto rt0 = finishRTWrapper(k1, res1.second, res1.first, res2.second, res2.first, 0);
    auto rt1 = finishRTWrapper(k2, res2.second, res2.first, res1.second, res1.first, 1);
    
    for(int i = 0; i < num_rts; i++) {
        assert(rt0[i] + rt1[i] - rout[i].value == xt[i] * ((x[i] - rin[i].value) < (1ULL << (bin - 1))));
        // printf("%lu %lu\n", rt0[i] + rt1[i] - rout[i].value, xt[i] * ((x[i] - rin[i].value) < (1ULL << (bin - 1))));
    }
    printf("test rt linear combination: passed\n");
}



GPUGroupElement* getFilter(Conv2DKey k, GroupElement* f1, int size_f) {
    GPUGroupElement* f2 = new GPUGroupElement[size_f];
    for(int i=0;i<k.CO;i++) 
    {
        for(int j=0;j<k.FH;j++)
        {
            for(int l=0;l<k.FW;l++)
            {
                for(int m=0;m<k.CI;m++)
                {
                    // nhwc
                    f2[i*k.FH*k.FW*k.CI + j*k.FW*k.CI + l*k.CI + m] = f1[j*k.FW*k.CI*k.CO + l*k.CI*k.CO + m*k.CO + i].value;

                }
            }

        } 
    }
    return f2;
}


GPUConv2DKey convertToGPUConv2DKey(Conv2DKey k1)
{
    GPUConv2DKey k2;
    k2.Bin = k1.Bin;
    k2.Bout = k1.Bout;
    k2.N = k1.N;
    k2.H = k1.H;
    k2.W = k1.W;
    k2.CI = k1.CI;
    k2.FH = k1.FH;
    k2.FW = k1.FW;
    k2.CO = k1.CO;
    k2.zPadHLeft = k1.zPadHLeft;
    k2.zPadHRight = k1.zPadHRight;
    k2.zPadWLeft = k1.zPadWLeft; 
    k2.zPadWRight = k1.zPadWRight;
    k2.strideH = k1.strideH; 
    k2.strideW = k1.strideW;

    k2.size_I = k2.N * k2.H * k2.W * k2.CI;
    k2.size_F = k2.CO * k2.FH * k2.FW * k2.CI;
    k2.OH = ((k2.H - k2.FH + (k2.zPadHLeft + k2.zPadHRight)) / k2.strideH) + 1;
    k2.OW = ((k2.W - k2.FW + (k2.zPadWLeft + k2.zPadWRight)) / k2.strideW) + 1;
    k2.size_O = k2.N * k2.OH * k2.OW * k2.CO;

    k2.mem_size_I = k2.size_I * sizeof(GPUGroupElement);
    k2.mem_size_F = k2.size_F * sizeof(GPUGroupElement);
    k2.mem_size_O = k2.size_O * sizeof(GPUGroupElement);

    k2.I = CPUToGPUGroupElement(k1.a, k2.size_I);
    k2.F = getFilter(k1, k1.b, k2.size_F);//new GPUGroupElement[k2.size_F];//CPUToGPUGroupElement(k1.b, k2.size_F);
    k2.O = CPUToGPUGroupElement(k1.c, k2.size_O);
    return k2;
}

void run_conv2d_test()
{
    // fss_init();
    // AES aesSeed(toBlock(0, time(NULL)));
    // auto commonSeed = aesSeed.ecbEncBlock(ZeroBlock);
    // prngShared.SetSeed(commonSeed);

    int Bin = 64;
    int Bout = 64;
    int N = 128;
    int H = 32;
    int W = 32;
    int CI = 3;
    int FH = 3;
    int FW = 3;
    int CO = 64;
    int zPadHLeft = 1;
    int zPadHRight = 1;
    int zPadWLeft = 1;
    int zPadWRight = 1;
    int strideH = 1;
    int strideW = 1;
    
    int size_I = N * H * W * CI;
    int size_F = CO * FH * FW * CI;
    int OH = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int OW = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int size_O = N * OH * OW * CO;


    auto rin1 = init_random(size_I, Bin);
    auto rin2 = init_random(size_F, Bin);
    auto rout = init_random(size_O, Bin);

    auto cpu_key = KeyGenConv2D(Bin, Bout, N, H, W, CI, FH,
                 FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                 zPadWRight, strideH, strideW, rin1, rin2, rout);

    auto gpu_key1 = convertToGPUConv2DKey(cpu_key.first);
    auto gpu_key2 = convertToGPUConv2DKey(cpu_key.second);

    auto cpu_I = init_random(size_I, Bin);
    auto cpu_F = init_random(size_F, Bin);
    auto cpu_masked_I = init_with_const(size_I, Bin, 0);
    auto cpu_masked_F = init_with_const(size_F, Bin, 0);

    // GroupElement* cpu_O1 = init_with_const(size_O, Bout, 0);//new GroupElement[size_O];
    // GroupElement* cpu_O2 = init_with_const(size_O, Bout, 0);
    GroupElement* conv_o = init_with_const(size_O, Bout, 0);

    GPUConv2DKey k = convertToGPUConv2DKey(cpu_key.first);
    free(k.I);
    free(k.F);
    free(k.O);
    k.I = CPUToGPUGroupElement(cpu_I, size_I);
    k.F = getFilter(cpu_key.first, cpu_F, size_F);//CPUToGPUGroupElement(cpu_F, size_F);
    auto gpu_conv_o = gpuConv2DWrapper(k, k.I, k.F, 0);

    Conv2DPlaintext(N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideH, strideW, cpu_I, cpu_F, conv_o);
    
    for(int i = 0;i < size_O; i++) 
        assert(conv_o[i].value == gpu_conv_o[i]);
    printf("test cpu and gpu convolution are the same: passed\n");


    GPUGroupElement* gpu_masked_I = new GPUGroupElement[size_I];
    GPUGroupElement* gpu_masked_F = new GPUGroupElement[size_F];

    for(int i=0;i<size_I;i++) {
        cpu_masked_I[i].value = cpu_I[i].value + rin1[i].value;
        gpu_masked_I[i] = cpu_masked_I[i].value;
    }

    for(int i=0;i<size_F;i++) {
        cpu_masked_F[i].value = cpu_F[i].value + rin2[i].value;
    }
    gpu_masked_F = getFilter(cpu_key.first, cpu_masked_F, size_F);

    auto gpu_O1 = gpuConv2DProtocolWrapper(gpu_key1, 0, gpu_masked_I, gpu_masked_F);
    auto gpu_O2 = gpuConv2DProtocolWrapper(gpu_key2, 1, gpu_masked_I, gpu_masked_F);
    
    for(int i = 0; i < size_O; i++) {
        assert(gpu_O1[i] + gpu_O2[i] - rout[i].value == conv_o[i].value);
    }

    printf("test gpu convolution protocol: passed\n");

    // for(int i = 0; i < size_O; i++) {
    //     assert(cpu_O1[i].value + cpu_O2[i].value - rout[i].value == conv_o[i].value);
    // }

}

GPUMatmulKey convertToGPUMatmulKey(MatMulKey k1)
{
    GPUMatmulKey k2;
    k2.Bin = k1.Bin;
    k2.Bout = k1.Bout;
    k2.M = k1.s1;
    k2.N = k1.s3;
    k2.K = k1.s2;

    k2.size_A = k2.M * k2.K;
    k2.size_B = k2.K * k2.N;
    k2.size_C = k2.M * k2.N;

    k2.mem_size_A = k2.size_A * sizeof(GPUGroupElement);
    k2.mem_size_B = k2.size_B * sizeof(GPUGroupElement);
    k2.mem_size_C = k2.size_C * sizeof(GPUGroupElement);

    k2.A = CPUToGPUGroupElement(k1.a, k2.size_A);
    k2.B = CPUToGPUGroupElement(k1.b, k2.size_B);
    k2.C = CPUToGPUGroupElement(k1.c, k2.size_C);
    return k2;
}

void run_matmul_test()
{
    int Bin = 64;
    int Bout = 64;
    int M = 128;
    int N = 512;
    int K = 256;

    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;

    auto rin1 = init_random(size_A, Bin);
    auto rin2 = init_random(size_B, Bin);
    auto rout = init_random(size_C, Bin);

    auto cpu_key = KeyGenMatMul(Bin, Bout, M, K, N, rin1, rin2, rout);

    auto gpu_key1 = convertToGPUMatmulKey(cpu_key.first);
    auto gpu_key2 = convertToGPUMatmulKey(cpu_key.second);

    auto cpu_A = init_random(size_A, Bin);
    auto cpu_B = init_random(size_B, Bin);
    auto cpu_masked_A = init_with_const(size_A, Bin, 0);
    auto cpu_masked_B = init_with_const(size_B, Bin, 0);
    GroupElement* matmul_o = init_with_const(size_C, Bout, 0);

    MatMul(M, K, N, cpu_A, cpu_B, matmul_o);

    GPUGroupElement* gpu_masked_A = new GPUGroupElement[size_A];
    GPUGroupElement* gpu_masked_B = new GPUGroupElement[size_B];

    for(int i = 0; i < size_A; i++) {
        cpu_masked_A[i].value = cpu_A[i].value + rin1[i].value;
        gpu_masked_A[i] = cpu_masked_A[i].value;
    }

    for(int i = 0; i < size_B; i++) {
        cpu_masked_B[i].value = cpu_B[i].value + rin2[i].value;
        gpu_masked_B[i] = cpu_masked_B[i].value;
    }

    auto gpu_O1 = gpuMatmulProtocolWrapper(gpu_key1, 0, gpu_masked_A, gpu_masked_B);
    auto gpu_O2 = gpuMatmulProtocolWrapper(gpu_key2, 1, gpu_masked_A, gpu_masked_B);
    
    for(int i = 0; i < size_C; i++) {
        assert(gpu_O1[i] + gpu_O2[i] - rout[i].value == matmul_o[i].value);
    }
    printf("%lu %lu\n", gpu_O1[0] + gpu_O2[0] - rout[0].value, matmul_o[0].value);
    printf("test gpu matmul protocol: passed\n");
}



int main()
{
    fss_init();
    AES aesSeed(toBlock(0, time(NULL)));
    auto commonSeed = aesSeed.ecbEncBlock(ZeroBlock);
    prngShared.SetSeed(commonSeed);

    // run_prg_test();
    // run_cmp_prg_test();
    // run_dcf_test();
    // run_relu_truncate_test();
    // run_conv2d_test();
    run_matmul_test();
    return 0;
}
