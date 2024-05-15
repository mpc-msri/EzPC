#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

i64 u2i(u64 x, int bitlength) {
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

    LlamaConfig::bitlength = 40;
    LlamaConfig::party = party;
    LlamaConfig::num_threads = 4;
    
    if(__argc > 2){
        ip = __argv[2];
    }
    llama->init(ip, true);

    u64 num_samples = 1000;
    int scale = 12;

    Tensor<u64> input({num_samples});
    Tensor<i64> input_ct({num_samples});

    if(party == CLIENT)
    {
        for (int i = 0; i < num_samples; ++i) {
            input.data[i] = random_ge(LlamaConfig::bitlength - 1) - (1LL << (LlamaConfig::bitlength - 2));
            mod(input.data[i], LlamaConfig::bitlength);
            input_ct.data[i] = u2i(input.data[i], LlamaConfig::bitlength);
        }

    }
    Tensor<u64> output({num_samples});
    llama->initializeInferencePartyB(input);

    llama::start();
    EdabitsPrTrunc(num_samples, input.data, output.data, scale);
    llama::end();

    llama->outputA(output);
    
    if (party == CLIENT) {
        for (int i = 0; i < num_samples; ++i) {
            mod(output.data[i], LlamaConfig::bitlength);
            i64 expected = (input_ct.data[i] >> scale);
            i64 got = u2i(output.data[i], LlamaConfig::bitlength);
            always_assert(got == expected || got == expected + 1);
        }
    }
    llama->finalize();

    return 0;
}