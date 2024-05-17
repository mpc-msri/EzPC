// Authors: Kanav Gupta, Neha Jawalkar
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

#include <sytorch/backend/float.h>
#include <sytorch/backend/cleartext.h>

int main(int argc, char** argv)
{
    ClearText<i64> b1;
    FloatClearText<float> b2;
    int scale = 12;
    u64 size = 1ULL<<(scale+3);

    Tensor<i64>   t1({size});
    Tensor<i64>   v1({size});
    Tensor<float> t2({size});
    Tensor<float> v2({size});

    i64 sign = 1;
    if (argc > 1)
        sign = atoi(argv[1]);
    
    always_assert(sign == 1 || sign == -1);

    for (int i = 0; i < size; ++i)
    {
        float f = sign * i / float(1LL<<scale);
        t1.data[i] = sign * i;
        t2.data[i] = f;
    }

    b1.gelu(t1, v1, scale);
    b2.gelu(t2, v2, 0);

    float maxdiff = 0;
    int maxi;
    i64 maxdiff_i = 0;
    for (int i = 0; i < size; ++i)
    {
        float diff = std::abs((v1.data[i] / double(1LL << scale)) - v2.data[i]);
        if (diff > maxdiff)
        {
            maxdiff = diff;
            maxi = i;
            maxdiff_i = std::abs(v1.data[i] - (1LL<<12) * v2.data[i]);
        }
    }
    std::cout << "Max diff: " << maxdiff << " at x = " << (sign * maxi / double(1LL<<12)) << std::endl;
    std::cout << "ULP: " << maxdiff_i << std::endl;
}
