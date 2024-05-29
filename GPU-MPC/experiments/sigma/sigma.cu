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

#include <sytorch/module.h>
#include <sytorch/utils.h>
#include "gpt2.h"
#include "bert.h"
#include "llama2.h"
#include "backend/sigma.h"

inline std::string toGB(u64 bytes)
{
    return std::to_string(bytes) + " B (" + std::to_string((float)bytes / (1024.0f * 1024.0f * 1024.0f)) + " GB)";
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
    int party = atoi(__argv[3]);

    std::string model(__argv[1]);
    printf("Model=%s\n", model.data());
    u64 keyBufSz = 0;
    SytorchModule<u64> *net;
    Tensor<u64> input({n_seq, n_embd});

    if (model == "gpt2")
    {
        n_layer = 12;
        n_head = 12;
        n_embd = 768;
        attnMask = "self";
        bw = 50;
        u64 mul = (u64)std::pow(2.3, std::log2(n_seq / 64));
        keyBufSz = 10 * mul * OneGB;
        net = new GPUGPT2<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
    }
    else if (model == "bert-tiny")
    {
        n_layer = 2;
        n_head = 2;
        n_embd = 128;
        bw = 37;
        keyBufSz = OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
    }
    else if (model == "bert-base")
    {
        n_layer = 12;
        n_head = 12;
        n_embd = 768;
        bw = 50;
        keyBufSz = 20 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
    }
    else if (model == "bert-large")
    {
        n_layer = 24;
        n_head = 16;
        n_embd = 1024;
        bw = 50;
        keyBufSz = 50 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
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
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
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
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
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
        keyBufSz = 300 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
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
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        net->zero();
    }
    srand(time(NULL));
    std::string outDir = "output/P" + std::to_string(party) + "/models/";
    makeDir(outDir);
    auto inferenceDir = outDir + model + "-" + std::to_string(n_seq) + "/";
    makeDir(inferenceDir);

    auto sigmaKeygen = new SIGMAKeygen<u64>(party, bw, scale, "", keyBufSz);
    net->setBackend(sigmaKeygen);
    net->optimize();
    auto start = std::chrono::high_resolution_clock::now();
    input.d_data = (u64 *)moveToGPU((u8 *)input.data, input.size() * sizeof(u64), (Stats *)NULL);
    auto &activation = net->forward(input);
    sigmaKeygen->output(activation);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    sigmaKeygen->close();
    std::stringstream ss;
    ss << "Total time=" + std::to_string(elapsed.count()) + " us";
    ss << std::endl;
    ss << "Key size=" + toGB(sigmaKeygen->keySize);
    ss << std::endl;
    std::ofstream statsFile(inferenceDir + "dealer.txt");
    statsFile << ss.rdbuf();
    statsFile.close();

    std::string ip(__argv[4]);
    auto sigma = new SIGMA<u64>(party, ip, "", bw, scale, n_seq, n_embd, atoi(__argv[5]), false);
    sigma->keyBuf = sigmaKeygen->startPtr;
    sigma->startPtr = sigma->keyBuf;
    sigma->keySize = sigmaKeygen->keySize;
    net->setBackend(sigma);
    sigma->peer->sync();
    start = std::chrono::high_resolution_clock::now();
    input.d_data = (u64 *)moveToGPU((u8 *)input.data, input.size() * sizeof(u64), (Stats *)NULL);
    activation = net->forward(input);
    sigma->output(activation);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    sigma->close();
    auto signedAct = Tensor<i64>((i64 *)activation.data, activation.shape).as_2d();
    // print(signedAct.as_nd(), scale, (u64) bw);
    auto maxIdx = signedAct.argmax(0);
    printf("%d, %ld\n", maxIdx, activation.data[maxIdx]);

    ss.clear();

    ss << "Total time=" + std::to_string(elapsed.count()) + " us";
    ss << std::endl;
    ss << "Comm time=" + std::to_string(sigma->s.comm_time) + " us";
    ss << std::endl;
    ss << "Transfer time=" + std::to_string(sigma->s.transfer_time) + " us";
    ss << std::endl;
    ss << "MHA time=" + std::to_string(sigma->s.mha_time) + " us";
    ss << std::endl;
    ss << "Matmul time=" + std::to_string(sigma->s.matmul_time) + " us";
    ss << std::endl;
    ss << "Truncate time=" + std::to_string(sigma->s.truncate_time) + " us";
    ss << std::endl;
    ss << "Gelu time=" + std::to_string(sigma->s.gelu_time) + " us";
    ss << std::endl;
    ss << "Softmax time=" + std::to_string(sigma->s.softmax_time) + " us";
    ss << std::endl;
    ss << "Layernorm time=" + std::to_string(sigma->s.layernorm_time) + " us";
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

    statsFile.open(inferenceDir + "evaluator.txt");
    statsFile << ss.rdbuf();
    statsFile.close();
    return 0;
}