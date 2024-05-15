#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

i64 u2i(u64 x, int bitlength) {
    if (bitlength == 64)
        return (i64) x;
    if (x >= (1LL << (bitlength - 1))) {
        return x - (1LL << bitlength);
    }
    return x;
}

int main(int __argc, char**__argv){

    sytorch_init();

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));

    LlamaConfig::bitlength = 50;
    LlamaConfig::party = party;
    LlamaConfig::num_threads = 4;
    
    if(__argc > 2){
        ip = __argv[2];
    }
    llama->init(ip, true);

    u64 num_samples = 1000;

    Tensor<u64> input({num_samples});
    Tensor<i64> input_ct({num_samples});

    u64 scale = 12;

    if(party == CLIENT)
    {
        for (int i = 0; i < num_samples; ++i) {
            input.data[i] = random_ge(LlamaConfig::bitlength - 1);
            input_ct.data[i] = input.data[i];
        }

    }
    Tensor<u64> output({num_samples});
    Tensor<i64> output_ct({num_samples});
    llama->initializeInferencePartyB(input);

    llama::start();
    llama->gelu(input, output, scale);
    // Gelu(input.size(), input.data, output.data, scale);
    llama::end();

    ClearText<i64> *ct = new ClearText<i64>();
    ct->gelu(input_ct, output_ct, scale);

    llama->outputA(output);
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            mod(output.data[i], LlamaConfig::bitlength);
            i64 diff = std::abs(u2i(output.data[i], LlamaConfig::bitlength) - output_ct.data[i]);
            // if (diff > 5)
            //     std::cout << diff << std::endl;
            always_assert(diff == 0);
        }
    }
    llama->finalize();

    return 0;
}