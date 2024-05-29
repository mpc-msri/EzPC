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

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include <cassert>

#include <sytorch/tensor.h>
#include <sytorch/backend/cleartext.h>

#include "fss/gpu_mha.h"

using T = u64;

int main(int argc, char *argv[])
{
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);

    int bw = 48;
    int scale = 12;
    auto ct = new ClearText<i64>();
    ct->bw = bw;

    int party = atoi(argv[1]);
    int n_seq = 128;
    int n_heads = 32;
    int n_embed = 4096;
    int dim_W = 128;
    MHAParams pMHA = {n_seq, n_embed, n_heads, dim_W, true, true, true};
    int N = pMHA.n_heads * pMHA.n_seq * pMHA.dim_W;

    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);

    uint8_t *startPtr, *curPtr;
    getKeyBuf(&startPtr, &curPtr, 10 * OneGB);

    initGPURandomness();
    auto d_mask_X = randomGEOnGpu<T>(N, bw);
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    T *h_X;
    auto d_masked_X = getMaskedInputOnGpu(N, bw, d_mask_X, &h_X, true, bw - scale);
    auto d_mask_O = gpuKeygenRotEmb(&curPtr, party, bw, scale, pMHA, d_mask_X, &g);
    auto h_mask_O = (T *)moveToCPU((u8 *)d_mask_O, N * sizeof(T), NULL);
    auto k = readGPUTruncateKey<T>(TruncateType::TrWithSlack, &startPtr);
    T *d_O;
    Stats s;
    for (int i = 0; i < 1; i++)
    {
        s.comm_time = 0;
        s.transfer_time = 0;
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        d_O = gpuRotEmb(peer, party, bw, scale, pMHA, k, d_masked_X, &g, &s);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Comm time=%lu micros\n", s.comm_time);
        printf("Transfer time=%lu micros\n", s.transfer_time);
        printf("Rotary embedding time=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    }
    unmaskValues(bw, N, d_O, h_mask_O, NULL);
    auto h_O = (T *)moveToCPU((uint8_t *)d_O, N * sizeof(T), (Stats *)NULL);
    printf("%ld, %ld\n", h_O[0], h_O[1]);
    gpuFree(d_O);
    destroyGPURandomness();
    for (int i = 0; i < n_heads; i++)
    {
        auto h_X_temp = (i64 *)(h_X + i * n_seq * dim_W);
        auto h_O_temp = (i64 *)(h_O + i * n_seq * dim_W);
        Tensor<i64> x((i64 *)h_X_temp, {(u64)n_seq, (u64)dim_W});
        Tensor<i64> y({(u64)n_seq, (u64)dim_W});
        ct->rotary_embedding(x, y, (u64)scale);
        for (int j = 0; j < n_seq * dim_W; j++)
        {
            if (i * n_seq * dim_W + j < 10)
            {
                printf("%d=%ld, %ld\n", i * n_seq * dim_W + j, y.data[j], h_O_temp[j]);
            }
            auto diff = std::abs((i64)((i64)y.data[j] - (i64)h_O_temp[j]));
            if (diff > 0)
            {
                printf("%d=%ld, %ld, %ld\n", i * n_seq * dim_W + j, y.data[j], h_O_temp[j], diff);
                assert(0);
            }
        }
    }
    return 0;
}