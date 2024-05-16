// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

    u64 n_seq = 10;

    Tensor<u64> input({n_seq, n_seq});
    Tensor<i64> input_ct(input.shape);

    u64 scale = 12;

    if(party == CLIENT)
    {
        for (int i = 0; i < input.size(); ++i) {
            // input.data[i] = i * (1LL << scale);
            input.data[i] = rand();
            if ((rand() % 2) == 0)
                input.data[i] = -input.data[i];
            input_ct.data[i] = input.data[i];
        }

    }
    Tensor<u64> output(input.shape);
    Tensor<i64> output_ct(input.shape);
    llama->initializeInferencePartyB(input);

    llama::start();
    for (int i = 0; i < 144; ++i)
        llama->softmax(input, output, scale, 0);
    llama::end();

    ClearText<i64> *ct = new ClearText<i64>();
    ct->softmax(input_ct, output_ct, scale, 1);

    llama->outputA(output);
    if (party == CLIENT) {
        for (int i = 0; i < input.size(); ++i) {
            i64 diff = std::abs((i64)output.data[i] - output_ct.data[i]);
            always_assert(diff == 0);
        }
    }
    llama->finalize();

    return 0;
}