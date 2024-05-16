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

#include <sytorch/backend/llama_base.h>
#include <sytorch/softmax.h>

int main(int argc, char *argv[])
{
    int s1 = 128;
    Tensor4D<i64> in(s1, 10, 1, 1);
    for (int i = 0; i < s1 * 10; i++)
    {
        in.data[i] = (rand() % 10) * (1LL << 22);
    }
    // in.fill(1LL << 24);
    Tensor4D<i64> out(s1, 10, 1, 1);
    u64 scale = 24;
    pirhana_softmax_ct(in, out, scale);

    int party = atoi(argv[1]);
    if (party == DEALER)
    {
        LlamaConfig::party = DEALER;
        auto llama = new LlamaBase<u64>();
        llama->init("0.0.0.0");
        Tensor4D<u64> inMask(s1, 10, 1, 1);
        inMask.fill(0);
        Tensor4D<u64> outMask(s1, 10, 1, 1);
        pirhana_softmax(inMask, outMask, scale);
        auto outMask_nd = outMask.as_nd();
        llama->output(outMask_nd);
        llama->finalize();
    }
    else
    {
        LlamaConfig::party = party;
        auto llama = new LlamaBase<u64>();
        llama->init("0.0.0.0");
        Tensor4D<u64> inp(s1, 10, 1, 1);
        memcpy(inp.data, in.data, s1 * 10 * sizeof(u64));
        Tensor4D<u64> out2(s1, 10, 1, 1);
        pirhana_softmax(inp, out2, scale);
        reconstruct(out2.d1 * out2.d2, out2.data, 64);
        auto output_nd = out2.as_nd();
        llama->output(output_nd);
        llama->finalize();
        for (int i = 0; i < s1; i++)
        {
            // printf("Max %d=%ld\n", i, out2.data[i]);
            if (i < 10 || (out2.data[i] - out.data[i] > 5))
                printf("%d=%ld, %ld\n", i, out2.data[i], out.data[i]);
            assert(out2.data[i] - out.data[i] <= 5);
        }
    }
}