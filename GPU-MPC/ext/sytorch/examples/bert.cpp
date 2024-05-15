#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/backend/piranha_cleartext.h>
#include <sytorch/backend/secureml_cleartext.h>
#include <sytorch/backend/float.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

bool hasPrinted = true;
bool hasInit = false;

template <typename T>
void printfe(Tensor<T> &t)
{
    if (hasInit) {
        std::cout << t.data[0] << std::endl;
    }
}

template <typename T>
class FFN : public SytorchModule<T>
{
    public:
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
    public:
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::invsqrt;
    using SytorchModule<T>::softmax;
    using SytorchModule<T>::concat;
    using SytorchModule<T>::attention_mask;

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

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T>*> qks_sm_vs;
        for(u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul(q, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax(qks);

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
    public:
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
        auto &attn_out = attn->forward(input);
        auto &add0_out = add(attn_out, input);
        auto &ln0_out = ln0->forward(add0_out);

        auto &ffn_out = ffn->forward(ln0_out);
        auto &add1_out = add(ffn_out, ln0_out);
        auto &ln1_out = ln1->forward(add1_out);
        return ln1_out;
    }
};

template <typename T>
class BERT : public SytorchModule<T>
{
    public:
    using SytorchModule<T>::tanh;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::unsqueeze;
    std::vector<TransformerBlock<T> *> blocks;
    LayerNorm<T> *ln_f;
    FC<T> *pool;
    u64 n_layer, n_heads, n_embd;

public:
    
    BERT(u64 n_layer, u64 n_heads, u64 n_embd): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd));
        }
        ln_f = new LayerNorm<T>(n_embd);
        pool = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &y = ln_f->forward(input);
        Tensor<T> *x = &y;
        
        for(u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }

        auto &x0 = view(*x, 0);
        auto &x0_unsqueeze = unsqueeze(x0);
        auto &pool_out = pool->forward(x0_unsqueeze);
        auto &tanh_out = tanh(pool_out);
        // return view(tanh_out, 0);
        return tanh_out;
    }
};


template <typename T>
class BERTSequenceClassification : public SytorchModule<T>
{
    public:
    using SytorchModule<T>::view;
    BERT<T> *gpt2;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_labels;
public:
    
    BERTSequenceClassification(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_labels): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_labels(n_labels)
    {
        gpt2 = new BERT<T>(n_layer, n_heads, n_embd);
        fc = new FC<T>(n_embd, n_labels, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt2->forward(input);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, 0);
    }
};

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    return n_elements / (4 * n_embd);
}

int float_sst2_validation(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 0;

    BERTSequenceClassification<float> bert(n_layer, n_head, n_embd, 2);
    bert.init(0);
    hasInit = true;
    bert.load("bert_sst2_90.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 872; ++i) {
        std::string fname = std::string("../../../transformers/datasets/sst2/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        // std::cout << n_seq << std::endl;
        Tensor<float> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        bert.activation.print();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int fixed_sst2_validation(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;

    using T = i64;

    BERTSequenceClassification<T> bert(n_layer, n_head, n_embd, 2);
    bert.init(scale);
    // bert.setBackend(new PiranhaClearText<T>);
    bert.setBackend(new SecureMLClearText<T>);
    hasInit = true;
    bert.load("bert_sst2_90.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 872; ++i) {
        std::string fname = std::string("../../../transformers/datasets/sst2/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        // std::cout << n_seq << std::endl;
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        print(bert.activation, scale);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int float_mrpc_validation(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 0;

    BERTSequenceClassification<float> bert(n_layer, n_head, n_embd, 2);
    bert.init(0);
    hasInit = true;
    bert.load("bert_mrpc_90_t.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 408; ++i) {
        std::string fname = std::string("../../../transformers/datasets/mrpc/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        // std::cout << n_seq << std::endl;
        Tensor<float> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        bert.activation.print();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int fixed_mrpc_validation(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;

    using T = i64;

    BERTSequenceClassification<T> bert(n_layer, n_head, n_embd, 2);
    bert.init(scale);
    hasInit = true;
    bert.load("bert_mrpc_90_t.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 408; ++i) {
        std::string fname = std::string("../../../transformers/datasets/mrpc/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        // std::cout << n_seq << std::endl;
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        bert.forward(input);
        print(bert.activation, scale);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int ct_main(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;

    BERTSequenceClassification<i64> bert(n_layer, n_head, n_embd, 2);
    bert.init(scale);
    hasInit = true;
    bert.load("bertclass.dat");

    std::string fname = __argv[1];
    u64 n_seq = get_n_seq(fname, n_embd);
    Tensor<i64> input({n_seq, n_embd});
    input.load(fname, scale);

    auto t1 = std::chrono::high_resolution_clock::now();
    bert.forward(input);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;
    print(bert.activation, scale);

    return 0;
}

int lt_main(int __argc, char**__argv){
    
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;

    int party = atoi(__argv[1]);
    if (party == 0) {
        return ct_main(__argc, __argv + 1);
    }
    std::string ip = __argv[2];

    using LlamaVersion = LlamaTransformer<u64>;
    LlamaVersion *llama = new LlamaVersion();
    srand(time(NULL));
    
    const u64 scale = 12;

    LlamaConfig::bitlength = 50;
    LlamaConfig::party = party;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    LlamaConfig::num_threads = 4;

    llama->init(ip, true);

    BERTSequenceClassification<u64> net(n_layer, n_head, n_embd, 2);
    net.init(scale);
    hasInit = true;
    net.setBackend(llama);
    // net.optimize();
    if(party == SERVER){
        net.load("bertclass.dat");
    }
    else if(party == DEALER){
        net.zero();
    }
    llama->initializeInferencePartyA(net.root);

    // std::string fname = __argv[2];
    // u64 n_seq = get_n_seq(fname, n_embd);
    u64 n_seq = 128;
    Tensor<u64> input({n_seq, n_embd});
    if(party == CLIENT){
        // input.load(fname, scale);
        input.fill(1LL << (scale-2));
    }
    llama->initializeInferencePartyB(input);

    llama::start();
    net.forward(input);
    llama::end();

    auto &output = net.activation;
    llama->outputA(output);
    if (party == CLIENT) {
        print(output, scale, LlamaConfig::bitlength);
    }
    llama->finalize();

    return 0;
}

int float_mrpc_single(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 0;

    BERTSequenceClassification<float> bert(n_layer, n_head, n_embd, 2);
    bert.init(0);
    hasInit = true;
    bert.load("bert_mrpc_90_t.dat");
    // std::cout << bert.gpt2->ln_f->A.data[0] << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    u64 i = atoi(__argv[1]);
    std::string fname = std::string("../../../transformers/datasets/mrpc/") + std::to_string(i) + ".dat";
    u64 n_seq = get_n_seq(fname, n_embd);
    std::cout << n_seq << std::endl;
    Tensor<float> input({n_seq, n_embd});
    input.load(fname, scale);
    bert.forward(input);
    bert.activation.print();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int float_sst2_single(int __argc, char**__argv) {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 0;

    BERTSequenceClassification<double> bert(n_layer, n_head, n_embd, 2);
    bert.init(0);
    hasInit = true;
    bert.load("bert_sst2_90.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    u64 i = atoi(__argv[1]);
    std::string fname = std::string("../../../transformers/datasets/sst2/") + std::to_string(i) + ".dat";
    u64 n_seq = get_n_seq(fname, n_embd);
    std::cout << n_seq << std::endl;
    Tensor<double> input({n_seq, n_embd});
    input.load(fname, scale);
    // std::cout << input.data[0] << " " << input.data[1] << std::endl;
    bert.forward(input);
    bert.activation.print();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int main(int __argc, char**__argv)
{
    // float_sst2_validation(__argc, __argv);
    fixed_sst2_validation(__argc, __argv);
    // lt_main(__argc, __argv);
    // float_mrpc_validation(__argc, __argv);
    // fixed_mrpc_validation(__argc, __argv);
    // float_sst2_single(__argc, __argv);
    // float_mrpc_single(__argc, __argv);
}
