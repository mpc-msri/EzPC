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

#ifndef CORRECTNESS
#define CORRECTNESS 0
#endif

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
#if CORRECTNESS
    LayerNorm<T> *ln_f;
    FC<T> *pool;
#endif
    u64 n_layer, n_heads, n_embd;
    std::string attnMask, qkvFormat;

public:
    GPUBERT(u64 n_layer, u64 n_heads, u64 n_embd, std::string attnMask, std::string qkvFormat) : n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), attnMask(attnMask), qkvFormat(qkvFormat)
    {
        for (u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new GPUBertTransformerBlock<T>(n_heads, n_embd, attnMask, qkvFormat));
        }
#if CORRECTNESS
        ln_f = new LayerNorm<T>(n_embd);
        pool = new FC<T>(n_embd, n_embd, true);
#endif
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;
#if CORRECTNESS
        auto &ln_out = ln_f->forward(input);
        x = &ln_out;
#endif
        for (u64 i = 0; i < n_layer; ++i)
        {
            auto &x_out = blocks[i]->forward(*x);
            x = &x_out;
        }
        return *x;
    }
};