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

#include <sytorch/backend/llama_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>
#include <filesystem>

template <typename T>
class FFN : public SytorchModule<T>
{
    using SytorchModule<T>::silu;
    using SytorchModule<T>::mul;

    u64 in;
    u64 intermediate_size;

public:
    FC<T> *up1;
    FC<T> *up2;
    FC<T> *down;

    FFN(u64 in, u64 intermediate_size) : in(in), intermediate_size(intermediate_size)
    {
        up1 = new FC<T>(in, intermediate_size, false);
        up2 = new FC<T>(in, intermediate_size, false);
        down = new FC<T>(intermediate_size, in, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &a = up1->forward(input);
        auto &b = up2->forward(input);
        return down->forward(mul(silu(a), b));
    }
};

template <typename T>
class MultiHeadAttention : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::softmax_triangular;
    using SytorchModule<T>::concat;

    using SytorchModule<T>::mul;
    using SytorchModule<T>::add;
    using SytorchModule<T>::silu;
    using SytorchModule<T>::rotary_embedding;

public:
    FC<T> *q_attn;
    FC<T> *k_attn;
    FC<T> *v_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttention(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        q_attn = new FC<T>(n_embd, n_embd, false);
        k_attn = new FC<T>(n_embd, n_embd, false);
        v_attn = new FC<T>(n_embd, n_embd, false);
        c_proj = new FC<T>(n_embd, n_embd, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &q_heads = q_attn->forward(input);
        auto &k_heads = k_attn->forward(input);
        auto &v_heads = v_attn->forward(input);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);

            auto &q1 = rotary_embedding(q);
            auto &k1 = rotary_embedding(k);

            auto &kt = transpose(k1);
            auto &qk = matmul_triangular(q1, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax_triangular(qks);

            auto &qks_sm_v = matmul(qks_sm, v);
            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class TransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    MultiHeadAttention<T> *attn;
    FFN<T> *ffn;
    RMSNorm<T> *ln0;
    RMSNorm<T> *ln1;

    u64 n_heads, n_embd, intermediate_size;

public:
    TransformerBlock(u64 n_heads, u64 n_embd, u64 intermediate_size) : n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd);
        ffn = new FFN<T>(n_embd, intermediate_size);
        ln0 = new RMSNorm<T>(n_embd, false);
        ln1 = new RMSNorm<T>(n_embd, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &ln0_out = ln0->forward(input);
        auto &attn_out = attn->forward(ln0_out);
        auto &attn_out_add = add(attn_out, input);
        auto &ln1_out = ln1->forward(attn_out_add);
        auto &ffn_out = ffn->forward(ln1_out);
        auto &ffn_out_add = add(ffn_out, attn_out_add);
        return ffn_out_add;
    }
};

template <typename T>
class LLAMA_MODEL : public SytorchModule<T>
{
    std::vector<TransformerBlock<T> *> blocks;
    RMSNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd, intermediate_size;

public:
    LLAMA_MODEL(u64 n_layer, u64 n_heads, u64 n_embd, u64 intermediate_size) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd, intermediate_size));
        }
        ln_f = new RMSNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;
        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return *x;
        // return ln_f->forward(*x);
    }
};

template <typename T>
class LlamaNextWordLogits : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    LLAMA_MODEL<T> *llama_model;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_vocab, intermediate_size;

public:
    LlamaNextWordLogits(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_vocab, u64 intermediate_size) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_vocab(n_vocab), intermediate_size(intermediate_size)
    {
        llama_model = new LLAMA_MODEL<T>(n_layer, n_heads, n_embd, intermediate_size);
        fc = new FC<T>(n_embd, n_vocab, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = llama_model->forward(input);
        return fc_in;
        // auto &fc_out = fc->forward(fc_in);
        // return view(fc_out, -1);
    }
};

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    return n_elements / (4 * n_embd);
}

void ct_main(std::string inpName)
{
    sytorch_init();

    // const u64 n_vocab = 32000;
    // const u64 n_embd = 4096;
    // const u64 n_head = 32;
    // const u64 n_layer = 32;
    // const u64 intermediate_size = 11008;
    // const u64 scale = 12;

    const u64 n_vocab = 32000;
    const u64 n_ctx = 4096;
    const u64 n_embd = 5120;
    const u64 n_head = 40;  // 40;
    const u64 n_layer = 1; // 40;
    const u64 intermediate_size = 13824;
    const u64 scale = 12;

    LlamaNextWordLogits<i64> llama_model(n_layer, n_head, n_embd, n_vocab, intermediate_size);
    u64 n_seq = 128; // get_n_seq(fname, n_embd);
    Tensor<i64> input({n_seq, n_embd});
    llama_model.init(scale, input);

    auto ct = new ClearText<i64>();
    ct->bw = 48;
    llama_model.setBackend(ct);

    // llama_model.load("meta_llama2_7b.dat");
    llama_model.load("meta_llama2_13b.dat");

    // std::string fname = std::string("lambada-meta-llama2-7b/") + /*std::to_string(i)*/ +"999.dat";
    std::string fname = std::string("lambada-meta-llama2-13b/") + /*std::to_string(i)*/ inpName;
    input.load(fname, scale);
    auto &res = llama_model.forward(input);
    auto signedAct = Tensor<i64>((i64 *)res.data, res.shape);
    print(signedAct, scale, ct->bw);
    auto maxIdx = signedAct.as_2d().argmax(0);
    std::cout << "Output:" << std::endl;
    std::cout << maxIdx << std::endl;
    std::cout << res.data[maxIdx] << std::endl;
    printf("%ld\n", signedAct.data[res.size() - 1]);
}

void lt_main(std::string inpName, int party)
{
    sytorch_init();

    // const u64 n_vocab = 32000;
    // const u64 n_embd = 4096;
    // const u64 n_head = 32;
    // const u64 n_layer = 32;//32;
    // const u64 intermediate_size = 11008;
    // const u64 scale = 12;

    const u64 n_vocab = 32000;
    const u64 n_ctx = 4096;
    const u64 n_embd = 5120;
    const u64 n_head = 40;  // 40;
    const u64 n_layer = 40; // 40;
    const u64 intermediate_size = 13824;
    const u64 scale = 12;

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    LlamaConfig::bitlength = 48;
    LlamaConfig::party = party;
    llama->init("0.0.0.0", true);

    LlamaNextWordLogits<u64> llama_model(n_layer, n_head, n_embd, n_vocab, intermediate_size);
    u64 n_seq = 128; // get_n_seq(fname, n_embd);
    Tensor<u64> input({n_seq, n_embd});
    input.zero();
    llama_model.init(scale, input);
    llama_model.setBackend(llama);
    llama_model.optimize();
    llama_model.zero();

    if (party != DEALER)
    {
        // llama_model.load("meta_llama2_7b.dat");
        llama_model.load("meta_llama2_13b.dat");
        std::string fname = std::string("lambada-meta-llama2-13b/") + /*std::to_string(i)*/ inpName;
        input.load(fname, scale);
    }

    // std::string fname = std::string("lambada-meta-llama2-7b/") + /*std::to_string(i)*/ +"999.dat";
    llama->initializeInferencePartyA(llama_model.root);
    llama->initializeInferencePartyB(input);

    llama::start();
    auto &res = llama_model.forward(input);
    llama::end();

    auto &output = llama_model.activation;
    llama->outputA(output);
    llama->finalize();

    if (party == CLIENT)
    {
        auto signedAct = Tensor<i64>((i64 *)llama_model.activation.data, llama_model.activation.shape);
        print(signedAct, scale, LlamaConfig::bitlength);
        auto maxIdx = signedAct.as_2d().argmax(0);
        std::cout << "Output:" << std::endl;
        std::cout << maxIdx << std::endl;
        std::cout << output.data[maxIdx] << std::endl;
        printf("%ld\n", signedAct.data[output.size() - 1]);
    }
}

int main(int __argc, char **__argv)
{
    int party = atoi(__argv[1]);
    if (party == 0)
        ct_main("999.dat");
    else
        lt_main("999.dat", party);
    return 0;
}