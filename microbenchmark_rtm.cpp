#include <sytorch/backend/llama_improved.h>

void microbenchmark_rtm_llamaimp(int dim, int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    llama->init(ip, false);

    u64 bs = 8;
    u64 ks = 3;
    // u64 ks = 11;

    Tensor4D<u64> in(bs, dim, dim, 64);
    u64 inSize = bs * dim * dim * 64;
    in.fill(0);

    Tensor4D<u64> out(bs, dim - ks + 1, dim - ks + 1, 64);
    u64 outSize = out.d1 * out.d2 * out.d3 * out.d4;
    Tensor<u64> maxBit((ks * ks - 1) * outSize);
    
    Tensor4D<u64> outRelu(bs, dim - ks + 1, dim - ks + 1, 64);

    if (party != DEALER)
        in.fill(1ULL << scale);

    llama::start();

    OrcaSTR(inSize, in.data, in.data, scale);
    LlamaConfig::bitlength -= scale; 
    MaxPoolDouble(out.d1, out.d2, out.d3, out.d4, ks, ks, 0, 0, 0, 0, 1, 1, in.d1, in.d2, in.d3, in.d4, in.data, in.data, out.data, out.data, maxBit.data);
    LlamaConfig::bitlength += scale;

    for (int i = 0; i < outSize; ++i)
        mod(out.data[i], LlamaConfig::bitlength - scale);
    ReluExtend(outSize, LlamaConfig::bitlength - scale, LlamaConfig::bitlength, out.data, outRelu.data, maxBit.data);
    
    llama::end();
    // llama->output(outRelu);
    // for (int i = 0; i < 10; ++i)
    //     std::cout << outRelu.data[i] << std::endl;
    llama->finalize();
}

int main(int argc, char** argv) {
    always_assert(argc > 2);
    int dim = atoi(argv[1]);
    int party = atoi(argv[2]);
    microbenchmark_rtm_llamaimp(dim, party);
}
