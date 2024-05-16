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

#include <cassert>
#include <cstdint>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_comms.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"

#include "fss/gpu_mha.h"
#include <sytorch/module.h>

using T = u64;

template <typename T>
class MultiHeadAttention : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::softmax_triangular;
    using SytorchModule<T>::concat;

public:
    FC<T> *c_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttention(u64 n_heads, u64 n_embd) : n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        c_attn = new FC<T>(n_embd, 3 * n_embd, true);
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

        std::vector<Tensor<T> *> qks_sm_vs;
        for (u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul_triangular(q, kt);
            auto &qks = scalarmul(qk, divisor);

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
T *sigmaMHA(int party, string ip, int bw, int scale, int n_embed, int n_seq, int n_heads, int dim_W, T *h_WQKV, T *h_YQKV, T *h_WProj, T *h_YProj, T *h_I)
{
    AESGlobalContext g;
    initAESContext(&g);
    initGPURandomness();
    initCPURandomness();

    auto mhaTab = initMHATables<T>(n_seq, scale);
    MHAParams pMHA = {n_seq, n_embed, n_heads, dim_W, true, true, false};
    MHAMulParams pMHAMul = initMHAMulParams(pMHA, bw, scale);

    auto d_mask_X = randomGEOnGpu<T>(pMHAMul.pQKV.size_A, bw);
    auto d_masked_X = (T *)moveToGPU((u8 *)h_I, pMHAMul.pQKV.size_A * sizeof(T), NULL);
    gpuLinearComb(bw, pMHAMul.pQKV.size_A, d_masked_X, T(1), d_masked_X, T(1), d_mask_X);

    auto h_mask_WQKV = randomGEOnCpu<T>(pMHAMul.pQKV.size_B, bw);
    auto h_mask_WProj = randomGEOnCpu<T>(pMHAMul.pProj.size_B, bw);
    int biasSzQKV = pMHAMul.pQKV.N * pMHAMul.pQKV.batchSz;
    printf("Bias size=%d\n", biasSzQKV);
    auto h_mask_YQKV = randomGEOnCpu<T>(biasSzQKV, bw);
    auto h_mask_YProj = randomGEOnCpu<T>(pMHAMul.pProj.N, bw);

    printf("Trying to allocate keyBuf\n");
    auto startPtr = cpuMalloc(10 * OneGB);
    auto curPtr = startPtr;

    auto d_mask_Z = gpuKeygenMHA(&curPtr, party, bw, scale, pMHA, pMHAMul, h_mask_WQKV, h_mask_YQKV, h_mask_WProj, h_mask_YProj, d_mask_X, &g);
    printf("Key size=%lu\n", curPtr - startPtr);
    auto h_mask_Z = (T *)moveToCPU((u8 *)d_mask_Z, pMHAMul.pProj.size_C * sizeof(T), NULL);
    printf("Mask Z: %ld\n", u64(h_mask_Z[0]));

    auto h_masked_WQKV = h_mask_WQKV;
    auto h_masked_YQKV = h_mask_YQKV;
    auto h_masked_WProj = h_mask_WProj;
    auto h_masked_YProj = h_mask_YProj;

    for (int i = 0; i < pMHAMul.pQKV.size_B; i++)
    {
        h_masked_WQKV[i] += h_WQKV[i];
        cpuMod(h_masked_WQKV[i], bw);
    }
    for (int i = 0; i < biasSzQKV; i++)
    {
        h_masked_YQKV[i] += h_YQKV[i];
        cpuMod(h_masked_YQKV[i], bw);
        // printf("Y[%d]=%ld, Masked Y=%ld\n", i, h_YQKV[i], mha_layer.YQKV[i]);
    }
    for (int i = 0; i < pMHAMul.pProj.size_B; i++)
    {
        h_masked_WProj[i] += h_WProj[i];
        cpuMod(h_masked_WProj[i], bw);
    }
    for (int i = 0; i < pMHAMul.pProj.N; i++)
    {
        h_masked_YProj[i] += h_YProj[i];
        cpuMod(h_masked_YProj[i], bw);
    }

    auto k = readGPUMHAKey<T>(pMHA, pMHAMul, &startPtr);
    Stats s;
    T *d_masked_Z;

    auto peer = new GpuPeer(true);
    peer->connect(party, ip);

    for (int i = 0; i < 1; i++)
    {
        s.reset();
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        d_masked_Z = gpuMHA(peer, party, bw, scale, pMHA, pMHAMul, k, h_masked_WQKV, h_masked_YQKV, h_masked_WProj, h_masked_YProj, d_masked_X, mhaTab, &g, &s);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Time taken=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
        printf("Softmax time=%lu micros\n", s.compute_time);
        printf("Comm time=%lu micros\n", s.comm_time);
        printf("Transfer time=%lu micros\n", s.transfer_time);
        printf("Bytes sent=%lu B\n", peer->bytesSent());
        printf("Bytes received=%lu B\n", peer->bytesReceived());
    }

    unmaskValues(bw, pMHAMul.pProj.size_C, d_masked_Z, h_mask_Z, NULL);
    auto h_Z = (T *)moveToCPU((u8 *)d_masked_Z, k.mmKeyProj.mem_size_C, NULL);
    return h_Z;
}

int main(int __argc, char **__argv)
{
    sytorch_init();
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_seq = 128;
    int bw = 50;
    int party = atoi(__argv[1]);
    std::string ip = __argv[2];
    const u64 scale = 12;

    MultiHeadAttention<i64> ctNet(n_head, n_embd);
    Tensor<i64> input({n_seq, n_embd});
    ctNet.SytorchModule<i64>::init(scale, input);
    input.as_2d().randomize(1LL << scale);
    (ctNet.c_attn)->weight.randomize(10000.0f);
    (ctNet.c_attn)->bias.randomize(10000.0f);
    (ctNet.c_proj)->weight.randomize(10000.0f);
    (ctNet.c_proj)->bias.randomize(10000.0f);
    ctNet.forward(input);
    // printf("CPU W size=%d\n", (ctNet.c_attn)->weight.size());
    // printshape(ctNet.activation.shape);
    // print(ctNet.activation, scale);//, 52);
    // printf("\n");

    assert(n_embd % n_head == 0);
    auto h_O = sigmaMHA(party, ip, bw, (int)scale, n_embd, n_seq, n_head, n_embd / n_head, (u64 *)(ctNet.c_attn)->weight.data, (u64 *)(ctNet.c_attn)->bias.data, (u64 *)(ctNet.c_proj)->weight.data, (u64 *)(ctNet.c_proj)->bias.data, (u64 *)input.data);
    for (int i = 0; i < ctNet.activation.size(); i++)
    {
        if (i < 10)
        {
            printf("%d=%ld, %ld\n", i, ctNet.activation.data[i], h_O[i]);
        }
        if ((u64)ctNet.activation.data[i] != (u64)h_O[i])
        {
            printf("%d=%ld, %ld\n", i, ctNet.activation.data[i], h_O[i]);
            assert(0);
        }
    }
    return 0;
}