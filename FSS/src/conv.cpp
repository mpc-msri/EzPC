/*
Authors: Deepak Kumaraswamy, Kanav Gupta
Copyright:
Copyright (c) 2022 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "conv.h"
#include "array.h"
#include "utils.h"
#include "comms.h"
#include <assert.h>

std::pair<MatMulKey, MatMulKey> KeyGenMatMul(int Bin, int Bout, int s1, int s2, int s3, GroupElement *rin1, GroupElement *rin2, GroupElement *rout){
    MatMulKey k0;
    MatMulKey k1;

    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;
    k0.s1 = s1; k0.s2 = s2; k0.s3 = s3;
    k1.s1 = s1; k1.s2 = s2; k1.s3 = s3;

    k0.a = make_array<GroupElement>(s1, s2);
    k0.b = make_array<GroupElement>(s2, s3);
    k0.c = make_array<GroupElement>(s1, s3);

    k1.a = make_array<GroupElement>(s1, s2);
    k1.b = make_array<GroupElement>(s2, s3);
    k1.c = make_array<GroupElement>(s1, s3);
    
    GroupElement *c = make_array<GroupElement>(s1, s3);
    MatMul(s1, s2, s3, rin1, rin2, c);
    MatAdd(s1, s3, c, rout, c);

    for (int i = 0; i < s1; i++)
    {
        for (int j = 0; j < s2; j++)
        {
            auto rin1_split = splitShareCommonPRNG(Arr2DIdxRowM(rin1, s1, s2, i, j));
            Arr2DIdxRowM(k0.a, s1, s2, i, j) = rin1_split.first;
            Arr2DIdxRowM(k1.a, s1, s2, i, j) = rin1_split.second;
        }
    }
    
    for(int i = 0; i < s2; i++)
    {
        for(int j = 0; j < s3; j++)
        {
            auto rin2_split = splitShareCommonPRNG(Arr2DIdxRowM(rin2, s2, s3, i, j));
            Arr2DIdxRowM(k0.b, s2, s3, i, j) = rin2_split.first;
            Arr2DIdxRowM(k1.b, s2, s3, i, j) = rin2_split.second;
        }
    }

    for(int i = 0; i < s1; i++)
    {
        for(int j = 0; j < s3; j++)
        {
            auto rout_split = splitShareCommonPRNG(Arr2DIdxRowM(c, s1, s3, i, j));
            Arr2DIdxRowM(k0.c, s1, s3, i, j) = rout_split.first;
            Arr2DIdxRowM(k1.c, s1, s3, i, j) = rout_split.second;
        }
    }

    delete[] c;

    return std::make_pair(k0, k1);
}

std::pair<Conv2DKey, Conv2DKey> KeyGenConv2D(
    int Bin, int Bout,
    int N, int H, int W, int CI, int FH, int FW, int CO,
    int zPadHLeft, int zPadHRight, 
    int zPadWLeft, int zPadWRight,
    int strideH, int strideW,
    GroupElement *rin1,  GroupElement * rin2, GroupElement * rout)
{
    Conv2DKey k0;
    Conv2DKey k1;

    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;
    k0.a = make_array<GroupElement>(N, H, W, CI);
    k1.a = make_array<GroupElement>(N, H, W, CI);
    k0.b = make_array<GroupElement>(FH, FW, CI, CO);
    k1.b = make_array<GroupElement>(FH, FW, CI, CO);
    k0.c = make_array<GroupElement>(d0, d1, d2, d3);
    k1.c = make_array<GroupElement>(d0, d1, d2, d3);
    k0.N = N; k0.H = H; k0.W = W; k0.CI = CI; k0.FH = FH; k0.FW = FW; k0.CO = CO; 
    k1.N = N; k1.H = H; k1.W = W; k1.CI = CI; k1.FH = FH; k1.FW = FW; k1.CO = CO;
    k0.strideH = strideH; k0.strideW = strideW; k0.zPadHLeft = zPadHLeft; k0.zPadHRight = zPadHRight; k0.zPadWLeft = zPadWLeft; k0.zPadWRight = zPadWRight;
    k1.strideH = strideH; k1.strideW = strideW; k1.zPadHLeft = zPadHLeft; k1.zPadHRight = zPadHRight; k1.zPadWLeft = zPadWLeft; k1.zPadWRight = zPadWRight;

    // Need temp array - matmul cant be done inplace and hence conv2d is not inplace
    GroupElement* c = make_array<GroupElement>(d0 * d1 * d2 * d3);
    
    Conv2DPlaintext(N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideH, strideW, rin1, rin2, c);

    MatAdd4(d0, d1, d2, d3, c, rout, c);

    for(int n = 0; n < N; ++n) {
        for(int h = 0; h < H; ++h) {
            for(int w = 0; w < W; ++w) {
                for(int ci = 0; ci < CI; ++ci) {
                    auto rin1_split = splitShareCommonPRNG(Arr4DIdxRowM(rin1, N, H, W, CI, n, h, w, ci));
                    Arr4DIdxRowM(k0.a, N, H, W, CI, n, h, w, ci) = rin1_split.first;
                    Arr4DIdxRowM(k1.a, N, H, W, CI, n, h, w, ci) = rin1_split.second;
                }
            }
        }
    }

    for(int fh = 0; fh < FH; ++fh) {
        for(int fw = 0; fw < FW; ++fw) {
            for(int ci = 0; ci < CI; ++ci) {
                for(int co = 0; co < CO; ++co) {
                    auto rin2_split = splitShareCommonPRNG(Arr4DIdxRowM(rin2, FH, FW, CI, CO, fh, fw, ci, co));
                    Arr4DIdxRowM(k0.b, FH, FW, CI, CO, fh, fw, ci, co) = rin2_split.first;
                    Arr4DIdxRowM(k1.b, FH, FW, CI, CO, fh, fw, ci, co) = rin2_split.second;
                }
            }
        }
    }

    for(int i = 0; i < d0; ++i) {
        for(int j = 0; j < d1; ++j) {
            for(int k = 0; k < d2; ++k) {
                for(int l = 0; l < d3; ++l) {
                    auto c_split = splitShareCommonPRNG(Arr4DIdxRowM(c, d0, d1, d2, d3, i, j, k, l));
                    Arr4DIdxRowM(k0.c, d0, d1, d2, d3, i, j, k, l) = c_split.first;
                    Arr4DIdxRowM(k1.c, d0, d1, d2, d3, i, j, k, l) = c_split.second;
                }
            }
        }
    }

    delete[] c;

    return std::make_pair(k0, k1);
}


void EvalConv2D(int party, const Conv2DKey &key,
    int N, int H, int W, int CI, int FH, int FW, int CO,
    int zPadHLeft, int zPadHRight, 
    int zPadWLeft, int zPadWRight,
    int strideH, int strideW, GroupElement* input, GroupElement* filter, GroupElement* output)
{
    int d0 = N;
    int d1 = ((H - FH + (zPadHLeft + zPadHRight)) / strideH) + 1;
    int d2 = ((W - FW + (zPadWLeft + zPadWRight)) / strideW) + 1;
    int d3 = CO;

    GroupElement* temp = make_array<GroupElement>(d0 * d1 * d2 * d3);
    MatCopy4(d0, d1, d2, d3, key.c, output);

    if (party == SERVER)
    {
        Conv2DPlaintext(N, H, W, CI, FH, FW, CO,
            zPadHLeft, zPadHRight, 
            zPadWLeft, zPadWRight,
            strideH, strideW, input, filter, temp);
        MatAdd4(d0, d1, d2, d3, temp, output, output);
    }

    Conv2DPlaintext(N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideH, strideW, input, key.b, temp);
    MatSub4(d0, d1, d2, d3, output, temp, output);

    Conv2DPlaintext(N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideH, strideW, key.a, filter, temp);
    MatSub4(d0, d1, d2, d3, output, temp, output);
    MatFinalize4(d0, d1, d2, d3, output);

    delete[] temp;
}



