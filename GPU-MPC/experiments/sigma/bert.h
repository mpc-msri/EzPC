#pragma once

#include <sytorch/module.h>

template <typename T>
class GPUBertTransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    _MHADummy<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;

    u64 n_heads, n_embd;
    std::string attnMask, qkvFormat;

public:
    GPUBertTransformerBlock(u64 n_heads, u64 n_embd, std::string attnMask, std::string qkvFormat) //: GPUGPT2TransformerBlock<T>(n_heads, n_embd, attnMask, qkvFormat)
        : n_heads(n_heads), n_embd(n_embd), attnMask(attnMask), qkvFormat(qkvFormat)
    {
        assert(n_embd % n_heads == 0);
        auto dim_W = n_embd / n_heads;
        attn = new _MHADummy<T>(n_heads, n_embd, (int)dim_W, attnMask, qkvFormat, true);
        ffn = new FFN<T>(n_embd, 4 * n_embd);
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
class GPUBERT : public SytorchModule<T>
{
public:
    using SytorchModule<T>::tanh;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::unsqueeze;
    std::vector<GPUBertTransformerBlock<T> *> blocks;
    LayerNorm<T> *ln_f;
    FC<T> *pool;
    u64 n_layer, n_heads, n_embd;
    std::string attnMask, qkvFormat;

public:
    GPUBERT(u64 n_layer, u64 n_heads, u64 n_embd, std::string attnMask, std::string qkvFormat) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), attnMask(attnMask), qkvFormat(qkvFormat)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new GPUBertTransformerBlock<T>(n_heads, n_embd, attnMask, qkvFormat));
        }
        ln_f = new LayerNorm<T>(n_embd);
        pool = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        // auto &y = ln_f->forward(input);
        // Tensor<T> *x = &y;

        Tensor<T> *x = &input;
        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return *x;
    }
};