#include <sytorch/backend/llama_improved.h>

void microbenchmark_rt_llamaimp(int num, int party) {
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    const u64 scale = 24;
    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::num_threads = 4;
    std::string ip = "127.0.0.1";
    llama->init(ip, false);

    Tensor4D<u64> images(1, num, 1, 1);
    Tensor4D<u64> drelu(1, num, 1, 1);
    Tensor4D<u64> out(1, num, 1, 1);
    images.fill(0);
    
    if (party != DEALER)
    for (int i = 0; i < num; ++i)
    {
        images.data[i] = (1 - 2 * (i % 2)) * (1ULL << scale);
    }

    llama::start();
    OrcaSTR(num, images.data, images.data, scale);
    for (int i = 0; i < num; ++i)
        mod(images.data[i], LlamaConfig::bitlength - scale);
    ReluExtend(num, LlamaConfig::bitlength - scale, LlamaConfig::bitlength, images.data, out.data, drelu.data);
    llama::end();
    // llama->output(out);
    // for (int i = 0; i < 10; ++i)
    //     std::cout << out.data[i] << std::endl;
    llama->finalize();
}

int main(int argc, char** argv) {
    always_assert(argc > 2);
    int num = atoi(argv[1]);
    int party = atoi(argv[2]);
    microbenchmark_rt_llamaimp(num, party);
}
