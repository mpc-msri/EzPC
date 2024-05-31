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

#include <sytorch/module.h>
#include "gpt2.h"
#include "bert.h"
#include "llama2.h"
#include "backend/sigma.h"

inline std::string toGB(u64 bytes) {
    return std::to_string(bytes) + " B (" + std::to_string((float) bytes / (1024.0f * 1024.0f * 1024.0f)) + " GB)";
}

int main(int __argc, char **__argv)
{
    sytorch_init();

    u64 n_embd = 0;
    u64 n_head = 0;
    u64 n_layer = 0;
    std::string attnMask = "none";
    std::string qkvFormat = "qkvconcat";
    int bw = 0;
    u64 scale = 12;
    u64 n_seq = atoi(__argv[2]);
    int role = atoi(__argv[3]);
    int party = atoi(__argv[4]);

    std::string model(__argv[1]);
    printf("Model=%s\n", model.data());
    std::string keyDir(__argv[5]);
    auto keyFile = keyDir + model + "_inference_key";
    u64 keyBufSz = 0;
    SytorchModule<u64> *net;

    if (model == "gpt2")
    {
        n_layer = 12;
        n_head = 12;
        n_embd = 768;
        attnMask = "self";
        bw = 50;
        keyBufSz = 20 * OneGB;
        net = new GPUGPT2<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
    }
    else if (model == "bert-tiny")
    {
        n_layer = 2;
        n_head = 2;
        n_embd = 128;
        bw = 37;
        keyBufSz = OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
    }
    else if (model == "bert-base")
    {
        n_layer = 12;
        n_head = 12;
        n_embd = 768;
        bw = 50;
        keyBufSz = 70 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
    }
    else if (model == "bert-large")
    {
        n_layer = 24;
        n_head = 16;
        n_embd = 1024;
        bw = 50;
        keyBufSz = 50 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
    }
    else if (model == "gpt-neo")
    {
        n_layer = 24;
        n_head = 16;
        n_embd = 2048;
        attnMask = "self";
        qkvFormat = "kvqsep";
        bw = 51;
        keyBufSz = 80 * OneGB;
        net = new GPUGPT2<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat, false);
    }
    else if (model == "gpt-neo-large")
    {
        n_layer = 32;
        n_head = 20;
        n_embd = 2560;
        attnMask = "self";
        qkvFormat = "concat";
        bw = 51; // 52;
        keyBufSz = 200 * OneGB;
        net = new GPUGPT2<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat, false);
    }
    else if (model == "llama7b")
    {
        n_layer = 32;
        n_head = 32;
        n_embd = 4096;
        attnMask = "self";
        qkvFormat = "qkvsep";
        bw = 48;
        u64 intermediate_size = 11008;
        keyBufSz = 500 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size);
    }
    else if (model == "llama13b")
    {
        n_layer = 40;
        n_head = 40;
        n_embd = 5120;
        attnMask = "self";
        qkvFormat = "qkvsep";
        bw = 48;
        u64 intermediate_size = 13824;
        keyBufSz = 450 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size);
    }
    else if (model == "airavata")
    {
        n_layer = 32;
        n_head = 32;
        n_embd = 4096;
        attnMask = "self";
        qkvFormat = "qkvsep";
        bw = 48;
        u64 intermediate_size = 11008;
        keyBufSz = 500 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size,false);
    }
    else
    {
        printf("Invalid model\n");
        return 1;
    }

    Tensor<u64> input({n_seq, n_embd});
    net->init(scale, input);
    srand(time(NULL));

    if (role == 0)
    {
        auto sigma = new SIGMAKeygen<u64>(party, bw, scale, keyFile, keyBufSz);
        net->setBackend(sigma);
        net->optimize();
        input.d_data = (u64 *)moveToGPU((u8 *)input.data, input.size() * sizeof(u64), (Stats *)NULL);
        auto &activation = net->forward(input);
        sigma->output(activation);
        sigma->close();
    }
    else
    {
        std::string ip(__argv[6]);
        auto sigma = new SIGMA<u64>(party, ip, keyFile, bw, scale, n_seq, n_embd, atoi(__argv[7]));
        net->setBackend(sigma);
        net->optimize();
        sigma->peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        input.d_data = (u64 *)moveToGPU((u8 *)input.data, input.size() * sizeof(u64), (Stats *)NULL);
        auto &activation = net->forward(input);
        sigma->output(activation);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        sigma->close();

        std::stringstream ss;

        ss << "Time in ms" << std::endl;
        ss << "Total time=" + std::to_string(elapsed.count());
        ss << std::endl;
        ss << "Comm time=" + std::to_string(sigma->s.comm_time);
        ss << std::endl;
        ss << "Transfer time=" + std::to_string(sigma->s.transfer_time);
        ss << std::endl;
        ss << "MHA time=" + std::to_string(sigma->s.mha_time);
        ss << std::endl;
        ss << "Matmul time=" + std::to_string(sigma->s.matmul_time);
        ss << std::endl;
        ss << "Truncate time=" + std::to_string(sigma->s.truncate_time);
        ss << std::endl;
        ss << "Gelu time=" + std::to_string(sigma->s.gelu_time);
        ss << std::endl;
        ss << "Softmax time=" + std::to_string(sigma->s.softmax_time);
        ss << std::endl;
        ss << "Layernorm time=" + std::to_string(sigma->s.layernorm_time);
        ss << std::endl;
        ss << std::endl;
        ss << "Total Comm=" + toGB(sigma->peer->bytesSent() + sigma->peer->bytesReceived());
        ss << std::endl;
        ss << "Gelu Comm=" + toGB(sigma->s.gelu_comm_bytes);
        ss << std::endl;
        ss << "Softmax Comm=" + toGB(sigma->s.softmax_comm_bytes);
        ss << std::endl;
        ss << "Layernorm Comm=" + toGB(sigma->s.layernorm_comm_bytes);
        ss << std::endl;

        auto inferenceDir = "output/P" + std::to_string(party) + "/";
        std::ofstream statsFile(inferenceDir + model + ".txt");
        statsFile << ss.rdbuf();
        statsFile.close();
    }
    return 0;
}