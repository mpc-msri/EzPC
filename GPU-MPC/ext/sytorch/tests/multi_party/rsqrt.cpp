#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

int main(int __argc, char**__argv){

    sytorch_init();

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));

    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::num_threads = 4;
    
    if(__argc > 2){
        ip = __argv[2];
    }
    llama->init(ip, true);

    u64 num_samples = 500;

    Tensor<u64> input({num_samples});
    Tensor<u64> input_ct({num_samples});

    u64 scale = 12;

    if(party == CLIENT)
    {
        for (int i = 0; i < num_samples; ++i) {
            input.data[i] = (i + 1) * (1LL << (2*scale));
            input_ct.data[i] = input.data[i];
        }

    }
    Tensor<u64> output({num_samples});
    llama->initializeInferencePartyB(input);

    llama::start();
    Rsqrt(num_samples, input.data, output.data, GroupElement(1), scale);
    llama::end();

    llama->outputA(output);
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            // always_assert(output.data[i] == u64(std::exp(-double(input_ct.data[i]) / double(1LL<<scale)) * (1LL << scale)));
            GroupElement expected = (1LL<<scale) / std::sqrt(i+1);
            GroupElement diff = std::abs((i64)(expected - output.data[i]));
            // std::cout << diff << std::endl;
            always_assert(diff <= 1);
        }
    }
    llama->finalize();

    return 0;
}