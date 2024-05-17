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

    u64 s1 = 2;
    u64 s2 = 2;
    u64 s3 = 2;
    Tensor<u64> x({s1, s2});
    Tensor<u64> A({s2, s3});
    Tensor<u64> y({s1, s3});

    u64 scale = 12;

    if(party == CLIENT)
    {
        for (int i = 0; i < x.size(); ++i) {
            x.data[i] = (i+1) * (1LL << scale);
            A.data[i] = (i+1) * (1LL << scale);
        }
    }

    llama->initializeInferencePartyB(x);
    llama->initializeInferencePartyB(A);

    llama::start();
    // MatMul2D(s1, s2, s3, x.data, x.data, A.data, A.data, y.data, y.data, false);
    SlothGemm(s1, s2, s3, x.data, A.data, y.data, scale);
    llama::end();

    llama->outputA(y);
    if (party == CLIENT) {
        print(y, scale, LlamaConfig::bitlength);
    }
    llama->finalize();

    return 0;
}