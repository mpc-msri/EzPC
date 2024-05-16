// Author: Neha Jawalkar
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

#pragma once

#include <sytorch/module.h>

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
class GPUGPT2TransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

public:
    _MHADummy<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;

    u64 n_heads, n_embd;
    // std::string attnMask, qkvFormat;
    // bool doNormQKt;

    // public:
    GPUGPT2TransformerBlock(u64 n_heads, u64 n_embd, std::string attnMask, std::string qkvFormat, bool doNormQKt) : n_heads(n_heads), n_embd(n_embd)
    // , attnMask(attnMask), qkvFormat(qkvFormat)
    // , doNormQKt(doNormQKt)
    {
        assert(n_embd % n_heads == 0);
        auto dim_W = n_embd / n_heads;
        attn = new _MHADummy<T>(n_heads, n_embd, (int)dim_W, attnMask, qkvFormat, doNormQKt);
        ffn = new FFN<T>(n_embd, 4 * n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
    }

    virtual Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &ln0_out = ln0->forward(input);
        // return ln0_out;
        auto &attn_out = attn->forward(ln0_out);
        // return attn_out;
        auto &attn_out_add = add(attn_out, input);
        auto &ln1_out = ln1->forward(attn_out_add);
        auto &ffn_out = ffn->forward(ln1_out);
        auto &ffn_out_add = add(ffn_out, attn_out_add);
        return ffn_out_add;
    }
};

template <typename T>
class GPUGPT2 : public SytorchModule<T>
{
    std::vector<GPUGPT2TransformerBlock<T> *> blocks;
    // LayerNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd;
    std::string attnMask, qkvFormat;
    bool doNormQKt;

public:
    GPUGPT2(u64 n_layer, u64 n_heads, u64 n_embd, std::string attnMask, std::string qkvFormat, bool doNormQKt=true) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), attnMask(attnMask), qkvFormat(qkvFormat), doNormQKt(doNormQKt)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new GPUGPT2TransformerBlock<T>(n_heads, n_embd, attnMask, qkvFormat, doNormQKt));
        }
        // ln_f = new LayerNorm<T>(n_embd);
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