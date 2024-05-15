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

    LlamaConfig::bitlength = 40;
    LlamaConfig::party = party;
    LlamaConfig::num_threads = 4;
    
    if(__argc > 2){
        ip = __argv[2];
    }
    llama->init(ip, true);

    u64 s1 = 1;
    u64 s2 = 8;

    Tensor<u64> input({s1, s2});
    Tensor<u64> input_ct({s1, s2});

    if(party == CLIENT)
    {
        for (int i = 0; i < s2; ++i) {
            input.data[i] = i - 5;
            input_ct.data[i] = input.data[i];
        }

    }
    Tensor<u64> output({s1});
    llama->initializeInferencePartyB(input);

    llama::start();
    SlothMaxpool(s1, s2, input.data, output.data);
    llama::end();

    llama->outputA(output);
    
    if (party == CLIENT) {
        for (int i = 0; i < s1; ++i) {
            mod(output.data[i], LlamaConfig::bitlength);
            std::cout << output.data[i] << std::endl;
        }
    }
    llama->finalize();

    return 0;
}