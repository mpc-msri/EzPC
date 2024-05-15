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

    u64 scale = 8;

    if(party == CLIENT)
    {
        for (int i = 0; i < num_samples; ++i) {
            input_ct.data[i] = i - (num_samples / 2);
            input.data[i] = input_ct.data[i];
            mod(input.data[i], LlamaConfig::bitlength);
        }

    }
    Tensor<u64> output({num_samples});
    llama->initializeInferencePartyB(input);

    llama::start();
    SlothARS(num_samples, input.data, output.data, scale);
    llama::end();

    llama->outputA(output);
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            mod(output.data[i], LlamaConfig::bitlength);
            u64 expected = input_ct.data[i] >> scale;
            mod(expected, LlamaConfig::bitlength);
            always_assert(output.data[i] == expected);
            // std::cout << output.data[i] % (1LL << (LlamaConfig::bitlength - scale)) << " " << (input_ct.data[i] >> scale) << std::endl;
        }
    }
    llama->finalize();

    return 0;
}