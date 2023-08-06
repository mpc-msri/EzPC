#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_improved.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <sytorch/utils.h>

int main(int __argc, char**__argv) {
    int party = atoi(__argv[1]);
    srand(time(NULL));
    const u64 scale = 8;
    LlamaConfig::party = party;
    LlamaConfig::bitlength = 64;
    // LlamaExtended<u64>::init("172.31.45.158");
    using LlamaVersion = LlamaImproved<u64>;
    LlamaVersion *llama = new LlamaVersion();
    llama->init("127.0.0.1", false);
    int size = 100;
    
    u64 *x = new u64[size];
    Tensor4D<u64> y(size, 1, 1, 1);
    
    u64 z = 280;
    for(int i = 0; i < size; ++i) {
        x[i] = z;
    }
    
    if (LlamaConfig::party == 1) {
        for(int i = 0; i < size; ++i) {
            x[i] = 0;
        }
    }

    llama::start();
    OrcaSTR(size, x, y.data, scale);
    llama::end();
    llama->output(y);
    llama->finalize();
    u64 zr = z >> scale;

    u64 c0 = 0;
    u64 c1 = 0;
    if (LlamaConfig::party != 1) {
        for(int i = 0; i < size; ++i) {
            std::cout << (y.data[i] % (1LL << (LlamaConfig::bitlength - scale))) << std::endl;
            if ((y.data[i] % 256) == zr)
            {
                c0 += 1;
            }
            else
            {
                c1 += 1;
            }
        }
        std::cout << "left : " << double(c0 * 100) / size << "%" << std::endl;
        std::cout << "right: " << double(c1 * 100) / size << "%" << std::endl;
    }
    return 0;
}