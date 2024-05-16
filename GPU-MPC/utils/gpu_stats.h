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

class Stats
{
public:
    uint64_t transfer_time = 0;
    uint64_t compute_time = 0;
    uint64_t comm_time = 0;

    uint64_t conv_time = 0;
    uint64_t conv_compute_time = 0;
    uint64_t conv_comm_time = 0;

    uint64_t matmul_time = 0;
    uint64_t matmul_compute_time = 0;
    uint64_t matmul_comm_time = 0;

    uint64_t relu_time = 0;
    uint64_t reluext_time = 0;
    uint64_t reluext_comm_time = 0;

    uint64_t maxpool_time = 0;
    uint64_t maxpool_comm_time = 0;
    uint64_t avgpool_time = 0;

    uint64_t truncate_time = 0;
    uint64_t truncate_comm_time = 0;
    uint64_t signext_time = 0;

    uint64_t gelu_time = 0;
    uint64_t layernorm_time = 0;
    uint64_t softmax_time = 0;
    uint64_t mha_time = 0;

    uint64_t linear_comm_bytes = 0;
    uint64_t gelu_comm_bytes = 0;
    uint64_t softmax_comm_bytes = 0;
    uint64_t layernorm_comm_bytes = 0;

    void reset()
    {
        transfer_time = 0;
        compute_time = 0;
        comm_time = 0;

        conv_time = 0;
        conv_compute_time = 0;
        conv_comm_time = 0;

        matmul_time = 0;
        matmul_compute_time = 0;
        matmul_comm_time = 0;

        relu_time = 0;
        reluext_time = 0;
        reluext_comm_time = 0;

        maxpool_time = 0;
        maxpool_comm_time = 0;

        avgpool_time = 0;
        
        truncate_time = 0;
        truncate_comm_time = 0;

        signext_time = 0;

        gelu_time = 0;
        layernorm_time = 0;
        softmax_time = 0;
        layernorm_comm_bytes = 0;
        linear_comm_bytes = 0;
        softmax_comm_bytes = 0;
        gelu_comm_bytes = 0;
        mha_time = 0;
    }
};