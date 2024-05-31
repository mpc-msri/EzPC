// Author: Neha Jawalkar,Tanmay Rajore
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
class LlamaFFN : public SytorchModule<T>
{
    using SytorchModule<T>::silu;
    using SytorchModule<T>::mul;

    u64 in;
    u64 intermediate_size;

public:
    FC<T> *up1;
    FC<T> *up2;
    FC<T> *down;

    LlamaFFN(u64 in, u64 intermediate_size) : in(in), intermediate_size(intermediate_size)
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
class GPULlamaTransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    _MHADummy<T> *attn;
    LlamaFFN<T> *ffn;
    RMSNorm<T> *ln0;
    RMSNorm<T> *ln1;
    u64 n_heads, n_embd, intermediate_size;
    bool rotatory;

public:

    GPULlamaTransformerBlock(u64 n_heads, u64 n_embd, u64 intermediate_size,bool rotatory=true): n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size),rotatory(rotatory)
    {
        auto dim_W = n_embd / n_heads;
        attn = new _MHADummy<T>(n_heads, n_embd, (int)dim_W, "self", "qkvsep", true, rotatory);
        ffn = new LlamaFFN<T>(n_embd, intermediate_size);
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
class GPULlama : public SytorchModule<T>
{
    std::vector<GPULlamaTransformerBlock<T> *> blocks;
    // RMSNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd, intermediate_size;
    bool rotatory;

public:
    
    GPULlama(u64 n_layer, u64 n_heads, u64 n_embd, u64 intermediate_size,bool rotatory=true): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), intermediate_size(intermediate_size),rotatory(rotatory)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new GPULlamaTransformerBlock<T>(n_heads, n_embd, intermediate_size,rotatory));
        }
        // ln_f = new RMSNorm<T>(n_embd);
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
        return *x;
    }
};