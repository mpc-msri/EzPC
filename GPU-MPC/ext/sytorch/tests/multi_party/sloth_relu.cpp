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

    u64 num_samples = 100;

    Tensor<u64> input({num_samples});
    Tensor<u64> input_ct({num_samples});

    if(party == CLIENT)
    {
        for (int i = 0; i < num_samples; ++i) {
            input.data[i] = random_ge(LlamaConfig::bitlength);
            input_ct.data[i] = input.data[i];
        }

    }
    Tensor<u64> output({num_samples});
    llama->initializeInferencePartyB(input);

    llama::start();
    SlothRelu(num_samples, input.data, output.data);
    llama::end();

    llama->outputA(output);
    
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            mod(output.data[i], LlamaConfig::bitlength);
            if (input_ct.data[i] < (1ULL << (LlamaConfig::bitlength - 1)))
            {
                always_assert(output.data[i] == input_ct.data[i]);
            }
            else
            {
                always_assert(output.data[i] == 0);
            }
        }
    }
    llama->finalize();

    return 0;
}