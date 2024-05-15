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
class TransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    _MHADummy<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;
    
    u64 n_heads, n_embd;
public:

    TransformerBlock(u64 n_heads, u64 n_embd): n_heads(n_heads), n_embd(n_embd)
    {
        attn = new _MHADummy<T>();
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
    LayerNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd;

public:
    
    GPT2(u64 n_layer, u64 n_heads, u64 n_embd): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd));
        }
        ln_f = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;
        
        for(u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

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

int main(int __argc, char**__argv)
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
}
