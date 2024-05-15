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

    u64 num_samples = 1000;

    Tensor<u64> input({num_samples});
    Tensor<u64> input_ct({num_samples});

    u64 lutbw = 16;
    std::vector<u64> lut(1LL<<lutbw);
    u64 scale = 6;
    for(int i = 0; i < (1LL<<lutbw); ++i)
    {
        // lut[i] = (rand() % 50); // this doesnt work, why?
        lut[i] = u64(std::exp(-i / double(1LL<<scale)) * (1LL<<scale));
    }

    if(party == CLIENT)
    {
        for (int i = 0; i < num_samples; ++i) {
            input.data[i] = (rand() % (1LL << 8));
            input_ct.data[i] = input.data[i];
        }

    }
    Tensor<u64> output({num_samples});
    llama->initializeInferencePartyB(input);

    llama::start();
    LUT_dpf(num_samples, lutbw, 64, lut, input.data, output.data);
    llama::end();

    llama->outputA(output);
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            // std::cout << output.data[i] << " " << lut[input_ct.data[i]] << std::endl;
            // std::cout << output.data[i] << " " << u64(std::exp(-double(input_ct.data[i]) / double(1LL<<scale)) * (1LL << scale)) << std::endl;
            always_assert(output.data[i] == lut[input_ct.data[i]]);
        }
    }
    llama->finalize();

    return 0;
}