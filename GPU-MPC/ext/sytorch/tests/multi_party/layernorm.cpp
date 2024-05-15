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

    u64 n_seq = 11;

    Tensor<u64> input({1, n_seq});
    Tensor<i64> input_ct(input.shape);

    u64 scale = 12;

    if(party == CLIENT)
    {
        for (int i = 0; i < input.size(); ++i) {
            input.data[i] = i * (1LL << scale);
            // input.data[i] = i;
            input_ct.data[i] = input.data[i];
        }

    }
    Tensor<u64> output(input.shape);
    Tensor<i64> output_ct(input.shape);
    llama->initializeInferencePartyB(input);

    Tensor1D<u64> A(n_seq);
    Tensor1D<u64> B(n_seq);
    A.fill(1LL << scale);
    auto A_nd = A.as_nd();
    llama->initializeInferencePartyB(A_nd);
    B.fill(0);

    llama::start();

    llama->layernorm(A, B, input, output, scale);
    llama::end();

    Tensor1D<i64> A_ct(n_seq);
    Tensor1D<i64> B_ct(n_seq);
    A_ct.fill(1LL << scale);
    B_ct.fill(0);
    ClearText<i64> *ct = new ClearText<i64>();
    ct->layernorm(A_ct, B_ct, input_ct, output_ct, scale);

    llama->outputA(output);
    if (party == CLIENT) {
        for (int i = 0; i < input.size(); ++i) {
            i64 diff = std::abs((i64)output.data[i] - output_ct.data[i]);
            always_assert(diff == 0);
            // std::cout << double((i64)output.data[i]) / double(1LL << (2*scale)) << " " << double((i64)output_ct.data[i]) / double(1LL << (2*scale)) << std::endl;
        }
    }
    llama->finalize();

    return 0;
}