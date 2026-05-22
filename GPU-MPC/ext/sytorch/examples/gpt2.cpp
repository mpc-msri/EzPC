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
#include <sytorch/backend/piranha_cleartext.h>
#include <sytorch/backend/secureml_cleartext.h>
#include <sytorch/backend/crypten_cleartext.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

bool hasPrinted = true;
bool hasInit = false;

template <typename T>
class FFN : public SytorchModule<T>
{
    using SytorchModule<T>::gelu;

    u64 in;
    u64 hidden;
public:
    FC<T> *up;
    FC<T> *down;

    FFN(u64 in, u64 hidden) : in(in), hidden(hidden) 
    {
        up = new FC<T>(in, hidden, true);
        down = new FC<T>(hidden, in, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        return down->forward(gelu(up->forward(input)));
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
    using SytorchModule<T>::scalardiv;
    using SytorchModule<T>::softmax_triangular;
    using SytorchModule<T>::concat;

public:
    FC<T> *c_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttention(u64 n_heads, u64 n_embd): n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        c_attn = new FC<T>(n_embd, 3*n_embd, true);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &x = c_attn->forward(input);
        auto &qkv_heads = split(x, 3);
        auto &q_heads = view(qkv_heads, 0);
        auto &k_heads = view(qkv_heads, 1);
        auto &v_heads = view(qkv_heads, 2);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T>*> qks_sm_vs;
        for(u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul_triangular(q, kt);
            auto &qks = scalardiv(qk, divisor);

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
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;
    
    u64 n_heads, n_embd;
public:

    TransformerBlock(u64 n_heads, u64 n_embd): n_heads(n_heads), n_embd(n_embd)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd);
        ffn = new FFN<T>(n_embd, 4*n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
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
class GPT2 : public SytorchModule<T>
{
    std::vector<TransformerBlock<T> *> blocks;
    // LayerNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd;

public:
    
    GPT2(u64 n_layer, u64 n_heads, u64 n_embd): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd));
        }
        // ln_f = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;

        for(u64 i = 0; i < n_layer - 1; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }

        auto &block = blocks[n_layer - 1];
        return block->forward(*x);
        
        // for(u64 i = 0; i < n_layer; ++i)
        // {
        //     auto &block = blocks[i];
        //     auto &x_out = block->forward(*x);
        //     x = &x_out;
        // }
        // return ln_f->forward(*x);
    }
};

/*
template <typename T>
class GPT2SequenceClassification : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    GPT2<T> *gpt2;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_labels;
public:
    
    GPT2SequenceClassification(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_labels): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_labels(n_labels)
    {
        gpt2 = new GPT2<T>(n_layer, n_heads, n_embd);
        fc = new FC<T>(n_embd, n_labels, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt2->forward(input);
        // printshape(fc_in.shape);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, -1);
    }
};
*/

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    return n_elements / (4 * n_embd);
}

/*
int ct_main(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;

    GPT2SequenceClassification<i64> gpt2(n_layer, n_head, n_embd, 2);
    gpt2.init(scale);
    hasInit = true;
    gpt2.load("gpt2lmr.dat");

    u64 imgidx = atoi(__argv[1]);
    std::string expected_lab = (imgidx >= 12500 ? "neg" : "pos");
    std::string fname = "/home/t-kanavgupta/gpt2lmr/dataset/" + expected_lab + "/" + std::to_string(imgidx) + ".dat";
    // std::string fname = __argv[1];
    // std::string fname = "./dataset/" + expected_lab + "/" + std::to_string(imgidx) + ".dat";
    u64 n_seq = get_n_seq(fname, n_embd);
    Tensor<i64> input({n_seq, n_embd});
    input.load(fname, scale);

    auto t1 = std::chrono::high_resolution_clock::now();
    gpt2.forward(input);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;
    print(gpt2.activation, scale);


    return 0;
}

int baseline_main(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;

    GPT2SequenceClassification<i64> gpt2(n_layer, n_head, n_embd, 2);
    gpt2.init(scale);
    gpt2.setBackend(new BaselineClearText<i64>());
    hasInit = true;
    gpt2.load("gpt2lmr.dat");

    u64 imgidx = atoi(__argv[1]);
    std::string expected_lab = (imgidx >= 12500 ? "neg" : "pos");
    std::string fname = "/home/t-kanavgupta/gpt2lmr/dataset/" + expected_lab + "/" + std::to_string(imgidx) + ".dat";
    // std::string fname = __argv[1];
    // std::string fname = "./dataset/" + expected_lab + "/" + std::to_string(imgidx) + ".dat";
    u64 n_seq = get_n_seq(fname, n_embd);
    Tensor<i64> input({n_seq, n_embd});
    input.load(fname, scale);

    auto t1 = std::chrono::high_resolution_clock::now();
    gpt2.forward(input);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;
    print(gpt2.activation, scale);


    return 0;
}

int float_main(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 0;

    GPT2SequenceClassification<float> gpt2(n_layer, n_head, n_embd, 2);
    gpt2.init(scale);
    gpt2.load("gpt2lmr.dat");

    std::string fname = __argv[1];
    u64 n_seq = get_n_seq(fname, n_embd);
    Tensor<float> input({n_seq, n_embd});
    input.load(fname, scale);

    auto t1 = std::chrono::high_resolution_clock::now();
    gpt2.forward(input);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;
    gpt2.activation.print();

    return 0;
}
*/

int lt_main(int __argc, char**__argv){
    
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;
    u64 bitlength = 50;
    GPT2<i64> net(n_layer, n_head, n_embd);
    Tensor<i64> input({128, n_embd});
    net.init(scale, input);
    auto ct = new ClearText<i64>();
    ct->bw = 50;
    net.setBackend(ct);
    net.load("gpt2-weights.dat");
    input.load("15469.dat", scale);
    printf("Starting\n");
    net.forward(input);
    auto &output = net.activation;
    print(output, scale, bitlength);
    return 0;
}

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

/*
int acc_main()
{

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;
    using T = i64;
    sytorch_init();

    int numThreads = omp_thread_count();
    std::cout << "using threads = " << numThreads << std::endl;
    std::vector<GPT2SequenceClassification<T> *> models;
    
    for(int i = 0; i < numThreads; ++i) {
        auto *model = new GPT2SequenceClassification<T>(n_layer, n_head, n_embd, 2);
        model->init(scale);
        model->load("gpt2lmr.dat");
        models.push_back(model);
    }

    hasInit = true;
    std::cout << "[*] Loaded all the models" << std::endl;

    std::vector<Tensor<T> *> images(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        images[i] = new Tensor<T>({1, n_embd});
    }

    std::ofstream outfiles[numThreads];
    std::string lab = "pos";
    for(int i = 0; i < numThreads; ++i) {
        outfiles[i] = std::ofstream("results_final_4june/" + lab + "/thread-" + std::to_string(i));
    }

    #pragma omp parallel for
    for (int i = 0; i < 12500; ++i) {
        int tid = omp_get_thread_num();
        std::string imgFile = "/home/t-kanavgupta/gpt2lmr/dataset/" + lab + "/" + std::to_string((lab == "neg" ? 12500 : 0) + i) + ".dat";
        u64 n_seq = get_n_seq(imgFile, n_embd);
        // std::cout << n_seq << std::endl;
        images[tid]->resize({n_seq, n_embd});
        images[tid]->load(imgFile, scale);
        models[tid]->forward(*images[tid]);
        auto &out = models[tid]->activation;
        // printshape(out.shape);
        outfiles[tid] << (i+1) << " " << (out.data[0] / double(1LL << scale)) << " " << (out.data[1] / double(1LL << scale)) << " " << (out.data[0] > out.data[1] ? "neg" : "pos") << std::endl;
    }

    lab = "neg";
    std::ofstream outfiles2[numThreads];
    for(int i = 0; i < numThreads; ++i) {
        outfiles2[i] = std::ofstream("results_final_4june/" + lab + "/thread-" + std::to_string(i));
    }

    #pragma omp parallel for
    for (int i = 0; i < 12500; ++i) {
        int tid = omp_get_thread_num();
        std::string imgFile = "/home/t-kanavgupta/gpt2lmr/dataset/" + lab + "/" + std::to_string((lab == "neg" ? 12500 : 0) + i) + ".dat";
        u64 n_seq = get_n_seq(imgFile, n_embd);
        // std::cout << n_seq << std::endl;
        images[tid]->resize({n_seq, n_embd});
        images[tid]->load(imgFile, scale);
        models[tid]->forward(*images[tid]);
        auto &out = models[tid]->activation;
        // printshape(out.shape);
        outfiles2[tid] << (i+1) << " " << (out.data[0] / double(1LL << scale)) << " " << (out.data[1] / double(1LL << scale)) << " " << (out.data[0] > out.data[1] ? "neg" : "pos") << std::endl;
    }

    return 0;
}

void softmax_test()
{
    u64 scale = 12;
    Tensor<i64> x {1, 10};
    for(u64 i = 0; i < 10; ++i)
    {
        x.data[i] = (i << scale);
    }
    Tensor<i64> out {1, 10};
    ClearText<i64> backend;
    backend.softmax(x, out, scale, 0);
    print(out, scale);
    backend.softmax(x, out, scale, 1);
    print(out, scale);
}

void ex_test()
{
    u64 scale = 12;
    ClearText<i64> backend;
    // Tensor<i64> x {1, 10};
    // for(u64 i = 0; i < 10; ++i)
    // {
    //     x.data[i] = (i << scale);
    // }
    // Tensor<i64> out {1, 10};
    // backend.softmax(x, out, scale, 0);
    // print(out, scale);
    // backend.softmax(x, out, scale, 2);
    // print(out, scale);
    Tensor<i64> x {1};
    Tensor<i64> out {1};
    float inp = 3.0;
    x.data[0] = inp * (1 << scale);
    u64 max_deg = 7;
    std::vector<double> coeffs(max_deg);
    coeffs[max_deg-1] = 1.0;
    for (u64 i = 1; i < max_deg; i++) {
        coeffs[max_deg-1-i] = coeffs[max_deg-i] * (-1.0 / i);
    }
    backend.polyeval(x, out, coeffs, scale);
    print(out, scale);
    std::cout << "exp(-3) = " << std::exp(-inp) << std::endl;
}

int edabits_main(int __argc, char**__argv){
    
    sytorch_init();

    int party = atoi(__argv[1]);
    std::string ip = "127.0.0.1";

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));

    const u64 scale = 12;

    LlamaConfig::bitlength = 64;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    LlamaConfig::num_threads = 8;
    
    if(__argc > 2){
        ip = __argv[2];
    }
    llama->init(ip, true);

    Tensor<u64> input({10});
    input.zero();
    if(party == CLIENT)
    {
        for (u64 i = 0; i < input.size(); ++i)
        {
            input.data[i] = (i << scale) + 1000;
            if (i % 2 == 0)
            {
                input.data[i] *= -1;
            }
        }
    }
    llama->initializeInferencePartyB(input);

    llama::start();
    EdabitsPrTrunc(input.size(), input.data, input.data, scale);
    llama::end();

    auto &output = input;
    llama->outputA(output);
    if (party == CLIENT) 
    {
        if (LlamaConfig::bitlength == 64) {
            for (u64 i = 0; i < output.size(); ++i) {
                std::cout << (i64)output.data[i] << " ";
            }
        }
        else
        {
            for (u64 i = 0; i < output.size(); ++i) {
                std::cout << output.data[i] % (1LL << (LlamaConfig::bitlength)) << " ";
            }
        }
        std::cout << std::endl;
    }
    llama->finalize();

    return 0;
}

int gpt2_sst2()
{
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;

    const u64 scale = 12;
    using T = i64;

    GPT2SequenceClassification<T> bert(n_layer, n_head, n_embd, 2);
    bert.init(scale);
    // bert.setBackend(new BaselineClearText<T>);
    // bert.setBackend(new PiranhaClearText<T>);
    bert.setBackend(new SecureMLClearText<T>);
    // bert.setBackend(new CryptenClearText<T>);
    hasInit = true;
    bert.load("/Users/kanav/Projects/transformers/gpt2-sst2/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 872; ++i) {
        std::string fname = std::string("/Users/kanav/Projects/transformers/gpt2-sst2/dataset/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        print(bert.activation, scale);

        // bert.activation.print();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cerr << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int gpt2_mrpc()
{
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;

    const u64 scale = 12;
    using T = i64;

    GPT2SequenceClassification<T> bert(n_layer, n_head, n_embd, 2);
    bert.init(scale);
    // bert.setBackend(new BaselineClearText<T>);
    // bert.setBackend(new PiranhaClearText<T>);
    bert.setBackend(new SecureMLClearText<T>);
    // bert.setBackend(new CryptenClearText<T>);
    hasInit = true;
    bert.load("/Users/kanav/Projects/transformers/gpt2-mrpc/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 408; ++i) {
        std::string fname = std::string("/Users/kanav/Projects/transformers/gpt2-mrpc/dataset/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        print(bert.activation, scale);

        // bert.activation.print();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cerr << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int gpt2_mnli()
{
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;

    const u64 scale = 12;
    using T = i64;

    GPT2SequenceClassification<T> bert(n_layer, n_head, n_embd, 3);
    bert.init(scale);
    // bert.setBackend(new PiranhaClearText<T>);
    // bert.setBackend(new SecureMLClearText<T>);
    bert.setBackend(new CryptenClearText<T>);
    hasInit = true;
    bert.load("/Users/kanav/Projects/transformers/gpt2-mnli/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 9815; ++i) {
        std::string fname = std::string("/Users/kanav/Projects/transformers/gpt2-mnli/dataset/matched/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        print(bert.activation, scale);

        // bert.activation.print();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cerr << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int gpt2_mnli_mm()
{
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;

    const u64 scale = 12;
    using T = i64;

    GPT2SequenceClassification<T> bert(n_layer, n_head, n_embd, 3);
    bert.init(scale);
    // bert.setBackend(new PiranhaClearText<T>);
    bert.setBackend(new SecureMLClearText<T>);
    // bert.setBackend(new CryptenClearText<T>);
    hasInit = true;
    bert.load("/Users/kanav/Projects/transformers/gpt2-mnli/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 9832; ++i) {
        std::string fname = std::string("/Users/kanav/Projects/transformers/gpt2-mnli/dataset/mismatched/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        print(bert.activation, scale);

        // bert.activation.print();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cerr << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int gpt2_qnli()
{
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;

    const u64 scale = 12;
    using T = i64;

    GPT2SequenceClassification<T> bert(n_layer, n_head, n_embd, 2);
    bert.init(scale);
    // bert.setBackend(new BaselineClearText<T>);
    // bert.setBackend(new PiranhaClearText<T>);
    // bert.setBackend(new SecureMLClearText<T>);
    // bert.setBackend(new CryptenClearText<T>);
    hasInit = true;
    bert.load("/Users/kanav/Projects/transformers/gpt2-qnli/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5463; ++i) {
        std::string fname = std::string("/Users/kanav/Projects/transformers/gpt2-qnli/dataset/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        print(bert.activation, scale);

        // bert.activation.print();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cerr << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}
*/
int main(int __argc, char**__argv)
{
    // softmax_test();
    // ex_test();
    // invsqrt_test();
    // ct_main(__argc, __argv);
    // float_main(__argc, __argv);
    lt_main(__argc, __argv);
    // acc_main();
    // edabits_main(__argc, __argv);
    // gpt2_sst2();
    // gpt2_mrpc();
    // gpt2_mnli();
    // gpt2_mnli_mm();
    // gpt2_qnli();
}