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

#pragma once

#include <cryptoTools/Common/Defines.h>
#include <llama/group_element.h>

struct DCFKeyPack{
    int Bin, Bout, groupSize;
    osuCrypto::block *k;   // size Bin+1
    GroupElement *g;    // bitsize Bout, size groupSize
    GroupElement *v;   // bitsize Bout, size Bin x groupSize
    DCFKeyPack(int Bin, int Bout, int groupSize,
                osuCrypto::block *k,
                GroupElement *g,
                GroupElement *v) : Bin(Bin), Bout(Bout), groupSize(groupSize), k(k), g(g), v(v){}
    DCFKeyPack() {
        Bin = Bout = groupSize = 0;
        k = nullptr;
        g = nullptr;
        v = nullptr;
    }
};

struct DualDCFKeyPack{  
    int Bin, Bout, groupSize;
    DCFKeyPack dcfKey;
    GroupElement *sb;   // size: groupSize
    DualDCFKeyPack() {}
};

struct AddKey{
    int Bin, Bout;
    GroupElement rb;
};

struct MultKey{
    int Bin, Bout;
    GroupElement a, b, c;
};

struct MatMulKey{
    int Bin, Bout;
    int s1, s2, s3;
    GroupElement *a, *b, *c;    
};

struct MultKeyNew {
    GroupElement a, b, c;
    DCFKeyPack k1, k2, k3, k4;
};

struct Conv2DKey{
    int Bin, Bout;
    int N, H, W, CI, FH, FW, CO,
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideH, strideW;
    GroupElement *a, *b, *c;    
};

struct Conv3DKey{
    int Bin, Bout;
    int N, D, H, W, CI, FD, FH, FW, CO,
        zPadDLeft, zPadDRight, 
        zPadHLeft, zPadHRight, 
        zPadWLeft, zPadWRight,
        strideD, strideH, strideW;
    GroupElement *a, *b, *c;    
};

struct TripleKeyPack {
    int bw;
    int64_t na, nb, nc;
    GroupElement *a, *b, *c;
};

struct ScmpKeyPack
{
    int Bin, Bout;
    DualDCFKeyPack dualDcfKey;
    GroupElement rb;
};

struct PublicICKeyPack
{
    int Bin, Bout;
    DCFKeyPack dcfKey;
    GroupElement zb;
};

struct PublicDivKeyPack
{
    int Bin, Bout;
    DualDCFKeyPack dualDcfKey;
    ScmpKeyPack scmpKey;
    GroupElement zb;
};

struct SignedPublicDivKeyPack
{
    int Bin, Bout;
    GroupElement d;     // divisor
    DCFKeyPack dcfKey;
    PublicICKeyPack publicICkey;
    ScmpKeyPack scmpKey;
    GroupElement A_share, corr_share, B_share, rdiv_share;
    GroupElement rout_temp_share, rout_share;
};

struct ReluKeyPack
{
    int Bin, Bout;
    osuCrypto::block *k;
    GroupElement *g, *v;
    GroupElement e_b0, e_b1;		 // size: degree+1 (same as beta)
    GroupElement beta_b0, beta_b1;	 // size: degree+1 (shares of beta, which is set of poly coeffs) (beta: highest to lowest power left to right)
    GroupElement r_b;
    GroupElement drelu;
};

struct MaxpoolKeyPack
{
    int Bin, Bout;
    ReluKeyPack reluKey;
    GroupElement rb;
};

struct ARSKeyPack
{
    // arithmetic right shift
    int Bin, Bout, shift;
    DCFKeyPack dcfKey;
    DualDCFKeyPack dualDcfKey;      // groupSize = 2 for payload
    GroupElement rb;
    ARSKeyPack() {}
};

struct ReluTruncateKeyPack {
    int Bin, Bout, shift;
    DCFKeyPack dcfKeyN;
    DCFKeyPack dcfKeyS;
    GroupElement zTruncate;
    GroupElement a, b, c, d1, d2;
};

struct Relu2RoundKeyPack {
    int effectiveBin, Bin;
    DCFKeyPack dcfKey;
    GroupElement a, b, c, d1, d2;
};

/*
struct SplineOneKeyPack
{
    int Bin, Bout;
    int degree; // degree of poly in payload beta
    DCFKeyPack dcfKey;
    std::vector<GroupElement> e_b;		 // size: degree+1 (same as beta)
    std::vector<GroupElement> beta_b;	 // size: degree+1 (shares of beta, which is set of poly coeffs) (beta: highest to lowest power left to right)
    GroupElement r_b;
};
*/
struct SplineKeyPack
{
    int Bin, Bout;
    int numPoly, degree;
    DCFKeyPack dcfKey;
    std::vector<GroupElement> p;        // spline breakpoints, size: numPoly + 1; p[0] = 0 and p[numPoly] = N-1
    std::vector<std::vector<GroupElement>> e_b; // 2d array dim: numPoly x (degree+1) (size is same as beta)
    std::vector<GroupElement> beta_b;           // 1d array size: numPoly * (degree+1) (shares of beta, which is set of poly coeffs) (beta: highest to lowest power left to right)
    GroupElement r_b;
};

struct PrivateScaleKeyPack
{
    GroupElement rin;
    GroupElement rout;
};

struct SquareKey {
    GroupElement b;
    GroupElement c;
};

struct TaylorSqKey {
    GroupElement a;
    GroupElement b;
};

struct MICKeyPack {
    DCFKeyPack dcfKey;
    GroupElement *z;
};

struct MSNZBKeyPack {
    MICKeyPack micKey;
    GroupElement r;
};

struct BulkyLRSKeyPack
{
    DCFKeyPack dcfKeyN;
    DCFKeyPack *dcfKeyS;
    GroupElement *z;
    GroupElement out;
};

struct TaylorKeyPack {
    MSNZBKeyPack msnzbKey;
    TaylorSqKey squareKey;
    BulkyLRSKeyPack lrsKeys[2];
    PrivateScaleKeyPack privateScaleKey;
};

struct SelectKeyPack {
    int Bin;
    GroupElement a, b, c, d1, d2;
};

struct MaxpoolDoubleKeyPack
{
    int Bin, Bout;
    Relu2RoundKeyPack reluKey;
    GroupElement rb;
};

struct BitwiseAndKeyPack
{
    GroupElement t[4];
};

struct FixToFloatKeyPack
{
    MICKeyPack micKey;
    GroupElement rs, rpow, ry, rm;
    SelectKeyPack selectKey;
};

struct FloatToFixKeyPack
{
    GroupElement rm, re, rw, /*rt,*/ rh;
    DCFKeyPack dcfKey;
    SelectKeyPack selectKey;
    ARSKeyPack arsKey;
    GroupElement p[1024];
    GroupElement q[1024];
};

struct ReluExtendKeyPack
{
    DCFKeyPack dcfKey;
    GroupElement rd, rw;
    GroupElement p[4];
    GroupElement q[2];
};

struct SignExtend2KeyPack
{
    DCFKeyPack dcfKey;
    GroupElement rw;
    GroupElement p[2];
};

struct EdabitsPrTruncKeyPack
{
    GroupElement a, b;
};

class DPFKeyPack
{
public:
    int bin, bout;
    osuCrypto::block *s;
    union {
        struct {
            uint64_t tLcw;
            uint64_t tRcw;
        };
        uint64_t tcw[2];
    };
    GroupElement payload;

    DPFKeyPack(int bin, int bout) : bin(bin), bout(bout)
    {
        s = new osuCrypto::block[bin+1];
        tLcw = 0;
        tRcw = 0;
    }

    DPFKeyPack()
    {
        s = nullptr;
        tLcw = 0;
        tRcw = 0;
    }
};

class DPFETKeyPack
{
public:
    int bin;
    osuCrypto::block *s;
    union {
        struct {
            uint64_t tLcw;
            uint64_t tRcw;
        };
        uint64_t tcw[2];
    };
    osuCrypto::block leaf;

    DPFETKeyPack(int bin) : bin(bin)
    {
        s = new osuCrypto::block[bin+1-7];
        tLcw = 0;
        tRcw = 0;
    }

    DPFETKeyPack()
    {
        s = nullptr;
        tLcw = 0;
        tRcw = 0;
    }
};

struct PubCmpKeyPack {
    int bin;
    DCFKeyPack dcfKey;
    GroupElement rout;
};

struct ClipKeyPack {
    int bin;
    PubCmpKeyPack cmpKey;
    GroupElement a, b, c, d1, d2;
};

struct LUTKeyPack {
    int bin, bout;
    DPFKeyPack dpfKey;
    GroupElement rout;
};

struct F2BF16KeyPack {
    int bin;
    DCFKeyPack dcfKey, dcfTruncate;
    GroupElement rout_k, rout_m, rin, prod, rout, rProd;
};

struct TruncateReduceKeyPack {
    int bin, shift;
    DCFKeyPack dcfKey;
    GroupElement rout;
};

struct LUTSSKeyPack {
    int bin, bout;
    GroupElement b0, b1, b2, b3;
    GroupElement routRes, routCorr;
    GroupElement rout;
};

struct LUTDPFETKeyPack {
    int bin, bout;
    DPFETKeyPack dpfKey;
    GroupElement routRes, routCorr;
};

struct SlothDreluKeyPack {
    int bin;
    DPFETKeyPack dpfKey;
    GroupElement r;
};

struct WrapSSKeyPack {
    int bin;
    uint64_t b0, b1;
};

struct WrapDPFKeyPack {
    int bin;
    DPFETKeyPack dpfKey;
    GroupElement r;
};

struct SlothLRSKeyPack {
    int bin, shift;
    GroupElement msb;
    GroupElement rout;
    GroupElement select;
};

struct SlothTRKeyPack {
    int bin, shift;
    GroupElement rout;
    GroupElement select;
};

struct SlothSignExtendKeyPack {
    int bin, bout;
    GroupElement rout;
    GroupElement select;
};
