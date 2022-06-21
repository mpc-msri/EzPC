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

#include "group_element.h"
#include "keypack.h"
#include "dcf.h"
#include "utils.h"
#include <assert.h>
#include <utility>

std::pair<ScmpKeyPack, ScmpKeyPack> keyGenSCMP(int Bin, int Bout, GroupElement rin1, GroupElement rin2,
                                GroupElement rout)
{
    // return 1 if x-rin1 >= y-rin2 else 0

    ScmpKeyPack key0, key1;

    key0.Bin = Bin; key1.Bin = Bin;
    key0.Bout = Bout; key1.Bout = Bout;

    GroupElement y = -(rin1 - rin2);
    uint8_t y_msb = y[0];
    GroupElement y_idx = y - ((uint64_t)y_msb << (Bin-1));
    GroupElement payload1(1 ^ y_msb, Bout), payload2(y_msb, Bout);
    auto keys = keyGenDualDCF(Bin-1, Bout, y_idx, payload1, payload2);
    key0.dualDcfKey = keys.first; 
    key1.dualDcfKey = keys.second;
    auto rout_split = splitShare(rout);
    key0.rb = rout_split.first; key1.rb = rout_split.second;
    return std::make_pair(key0, key1);
}

GroupElement evalSCMP(int party, ScmpKeyPack key,
                    GroupElement x, GroupElement y)
{
    // return 1 if x-rin1 >= y-rin2 else 0

    GroupElement z = x - y;
    uint8_t z_msb = z[0];
    GroupElement z_n_1 = z - ((uint64_t)z_msb << (key.Bin-1));
    GroupElement z_idx = ((uint64_t)1 << (key.Bin-1)) - z_n_1 - 1; 
    GroupElement mb(0, key.Bout);
    evalDualDCF(party, &mb, z_idx, key.dualDcfKey);
    return party - (party * z_msb + mb - 2*z_msb*mb) + key.rb;
}


std::pair<PublicICKeyPack, PublicICKeyPack> keyGenPublicIC(int Bin, int Bout, GroupElement p, GroupElement q,
                                    GroupElement rin, GroupElement rout)
{
    GroupElement gamma, q1, alpha_p, alpha_q, alpha_q1, z0, z1, z;
    PublicICKeyPack k0, k1;

    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;

    gamma = rin - 1;
    GroupElement payload(1, Bout);
    auto dcfKeys = keyGenDCF(Bin, Bout, 1, gamma, &payload);
    q1 = q + 1;
    alpha_p = p + rin;
    alpha_q = q + rin;
    alpha_q1 = q + rin + 1;
    z = rout + (alpha_p > alpha_q) - (alpha_p > p) + (alpha_q1 > q1) + (alpha_q == GroupElement(-1, Bin));
    auto z_split = splitShare(z);
    k0.zb = z_split.first; k1.zb = z_split.second;
    k0.dcfKey = dcfKeys.first;
    k1.dcfKey = dcfKeys.second;
    return std::make_pair(k0, k1);
}

GroupElement evalPublicIC(int party, PublicICKeyPack key, 
                    GroupElement x,GroupElement p, GroupElement q)
{
    GroupElement q1, xp, xq1, yb;
    q1 = q + 1;
    xp = x - p - 1;
    xq1 = x - q1 - 1;
    // todo: handle multiblock grouplen
    GroupElement sp(0, key.Bout);
    evalDCF(key.Bin, key.Bout, 1, &sp, party, xp, key.dcfKey.k, key.dcfKey.g, key.dcfKey.v);
    GroupElement sq1(0, key.Bout);
    evalDCF(key.Bin, key.Bout, 1, &sq1 ,party, xq1, key.dcfKey.k, key.dcfKey.g, key.dcfKey.v);
    yb = (party) * ((x > p) - (x > q1)) - sp + sq1 + key.zb;
    return yb;
}

// std::pair<PublicDivKeyPack, PublicDivKeyPack> keyGenPublicDiv(int Bin, int Bout, GroupElement rin, GroupElement rout, 
//                                         GroupElement rout1, GroupElement d)
// {
//     PublicDivKeyPack k0, k1;
    
//     k0.Bin = Bin; k1.Bin = Bin;
//     k0.Bout = Bout; k1.Bout = Bout;

//     GroupElement r1, r0, n1, n0;
//     // r1 = rin / d; r0 = rin % d;
//     r1 = signedDivide(rin, d); r0 = signedMod(rin, d);

//     if (Bin == 64) {
//         // When Bin = 64, we can use the following trick to avoid the overflow
//         // (1L << (Bin)) = (1L << (Bin-1)) * 2
//         // let k = (1L << (Bin-1))
//         // we want 2k / d and 2k % d
//         // 2k % d = (2(k % d)) % d
//         // 2k / d = 2*(k/d) + (2* (k % d))/d
//         uint64_t d1 = ((uint64_t)1 << 63) / d.value;
//         uint64_t d0 = ((uint64_t)1 << 63) % d.value;
//         n0 = GroupElement((2 * d0) % d.value, Bin);
//         n1 = GroupElement((2 * d1) + ((2 * d0) / d.value), Bin);
//     }
//     else {
//         n1 = GroupElement(((uint64_t)1 << Bin) / d.value, Bin);
//         n0 = GroupElement(((uint64_t)1 << Bin) % d.value, Bin);
//     }
//     block seed;
    
//     auto dualDcfKeys = keyGenDualDCF(Bin, Bout, rin, 1 + rout1, rout1);
//     auto scmpKeys = keyGenSCMP(Bin, Bout, r0 + n0*rout1, GroupElement(0, Bin), GroupElement(0, Bout));
    
//     k0.dualDcfKey = dualDcfKeys.first; k1.dualDcfKey = dualDcfKeys.second;
//     k0.scmpKey = scmpKeys.first; k1.scmpKey = scmpKeys.second;
    
//     auto zb_split = splitShare(-r1 - n1*rout1 + rout);
//     k0.zb = zb_split.first; k1.zb = zb_split.second;
//     return std::make_pair(k0, k1);
// }


// GroupElement evalPublicDiv_First(int party, const PublicDivKeyPack &key, GroupElement x, GroupElement d)
// {
//     GroupElement result_evalPublicDiv_First(0, key.Bout);
//     evalDualDCF(party, &result_evalPublicDiv_First, x, key.dualDcfKey);
//     return result_evalPublicDiv_First;
// }

// GroupElement evalPublicDiv_Second(int party, const PublicDivKeyPack &key, GroupElement x, GroupElement d,
//                                 GroupElement result_evalPublicDiv_First)
// {
//     GroupElement x1, x0, n1, n0, y_partial, yb_first, yb_second;
//     // x1 = x / d; x0 = x % d;
//      x1 = signedDivide(x, d); x0 = signedMod(x, d);

//     if (key.Bin == 64) {
//         // When Bin = 64, we can use the following trick to avoid the overflow
//         // ((uint64_t)1 << (Bin)) = ((uint64_t)1 << (Bin-1)) * 2
//         // let k = ((uint64_t)1 << (Bin-1))
//         // we want 2k / d and 2k % d
//         // 2k % d = (2(k % d)) % d
//         // 2k / d = 2*(k/d) + (2* (k % d))/d
//         uint64_t d1 = ((uint64_t)1 << 63) / d.value;
//         uint64_t d0 = ((uint64_t)1 << 63) % d.value;
//         n0 = GroupElement((2 * d0) % d.value, 64);
//         n1 = GroupElement((2 * d1) + ((2 * d0) / d.value), 64);
//     }
//     else {
//         n1 = GroupElement(((uint64_t)1 << key.Bin) / d.value, key.Bin);
//         n0 = GroupElement(((uint64_t)1 << key.Bin) % d.value, key.Bin);
//     }
//     y_partial = x0 + result_evalPublicDiv_First*n0;
    
//      // we want (y_partial < 0) and (y_partial < d)
//      // scmp gate takes input (x, y) and gives output (x >= y)
//      // so, to get (x < y) = 1 - (x >= y) call 1 - scmpgate(x, y)

//      yb_first = GroupElement(party * 1, key.Bout) - evalSCMP(party, key.scmpKey, y_partial, GroupElement(0, key.Bin));
//      yb_second = GroupElement(party * 1, key.Bout) - evalSCMP(party, key.scmpKey, y_partial, d);

//     //  std::cout << "y_partial " << y_partial << std::endl;
//     //  std::cout << "yb_first " << yb_first << std::endl;
//     //  std::cout << "yb_second " << yb_second << std::endl;

//     GroupElement res = (party)*(x1 + 1 + result_evalPublicDiv_First*n1) + key.zb - yb_first - yb_second;
//     return res;
// }

std::pair<ARSKeyPack, ARSKeyPack> keyGenARS(int Bin, int Bout, uint64_t shift, GroupElement rin, GroupElement rout)
{
    ARSKeyPack k0, k1;
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;
    k0.shift = shift; k1.shift = shift;

    GroupElement y = -rin;
    uint8_t y_msb = y[0];
    uint64_t alpha_n = y.value & (((uint64_t)1 << (Bin - 1)) - 1);

    // store last shift bits of y in alpha_s 
    uint64_t ones = ((uint64_t)1 << shift) - 1;
    GroupElement alpha_s(y.value & ones, shift);

    // std::cout << "keygen alpha_n (dualdcf alpha) " << alpha_n << " alpha_s (dcf alpha)" << alpha_s << std::endl;

    auto dcfKeys = keyGenDCF(shift, Bout, alpha_s, GroupElement(1, Bout));
    k0.dcfKey = dcfKeys.first; k1.dcfKey = dcfKeys.second;

    if (Bout > Bin - shift) {
        GroupElement *payload1 = new GroupElement[2], *payload2 = new GroupElement[2];

        payload1[0] = GroupElement(1, Bout);
        payload1[1] = GroupElement(1 ^ y_msb, Bout);
        payload2[0] = GroupElement(0, Bout);
        payload2[1] = GroupElement(y_msb, Bout);
        auto dualDcfKeys = keyGenDualDCF(Bin - 1, Bout, 2, GroupElement(alpha_n, Bin - 1), payload1, payload2);
        k0.dualDcfKey = dualDcfKeys.first; k1.dualDcfKey = dualDcfKeys.second;
        auto rb_split = splitShare(rout + GroupElement(alpha_n >> shift, Bout));
        k0.rb = rb_split.first; k1.rb = rb_split.second;
    }
    else {
        auto rb_split = splitShare(rout + GroupElement(y.value >> shift, Bout));
        k0.rb = rb_split.first; k1.rb = rb_split.second;
    }


    return std::make_pair(k0, k1);
}

GroupElement evalARS(int party, GroupElement x, uint64_t shift, const ARSKeyPack &k)
{
    // last shift bits of x
    uint64_t ones = ((uint64_t)1 << shift) - 1;
    // todo: ideally x_s should have bitsize = "shift". 
    // defining a groupelem by bitshize shift, will this cause problems?
    // maybe in getDataFromBlock?
    GroupElement x_s(x.value & ones, shift);

    // last n-1 bits of x
    uint8_t x_msb = x[0];
    // todo: bitsize of x_n should have been k.Bin - 1
    uint64_t x_n = x.value & (((uint64_t)1 << (k.Bin - 1)) - 1);
    // std::cout << "x_n " << x_n.value << std::endl;

    GroupElement dcfIdx = ((uint64_t)1 << shift) - x_s - 1;
    dcfIdx = changeBitsize(dcfIdx, shift);
    // GroupElement t_s = evalDCF(party, dcfIdx, k.dcfKey);
    GroupElement t_s;
    evalDCF(party, &t_s, dcfIdx, k.dcfKey);

    GroupElement res;
    if (k.Bout > k.Bin - k.shift) {

    GroupElement dualDcfIdx(((uint64_t)1 << (k.Bin - 1)) - x_n - 1, k.Bin - 1);
    GroupElement ddcfOut[2];
    ddcfOut[0].bitsize = k.Bout;
    ddcfOut[1].bitsize = k.Bout;
    evalDualDCF(party, ddcfOut, dualDcfIdx, k.dualDcfKey);

    GroupElement t_n = ddcfOut[0], m_n = ddcfOut[1];
    GroupElement mb = GroupElement(party * x_msb, k.Bout) + m_n - 2 * x_msb * m_n;
    res = party * GroupElement(x_n >> shift, k.Bout) + k.rb + t_s - ((uint64_t)1 << (k.Bin - shift - 1)) * (t_n + mb);
    }
    else {
        res = party * GroupElement(x.value >> shift, k.Bout) + k.rb + t_s;
    }

    
    // std::cout << "x_n " << x_n << std::endl;
    // std::cout << "x_s " << x_s << std::endl;
    // std::cout << "dcfIdx " << dcfIdx << std::endl;
    // std::cout << "dualDcfIdx " << dualDcfIdx << std::endl;
    // std::cout << "dcf out " << t_s << std::endl;
    // std::cout << "dualDcf out1 " << t_n << std::endl;
    // std::cout << "dualDcf out2 " << m_n << std::endl;
    // std::cout << "mb " << mb << std::endl;

    return res; 
}

std::pair<SignedPublicDivKeyPack, SignedPublicDivKeyPack> keyGenSignedPublicDiv(int Bin, int Bout, GroupElement rin, GroupElement rout_temp, 
                                        GroupElement rout, GroupElement d)
{
    // bitsize of rout_temp is Bin
    SignedPublicDivKeyPack k0, k1;
    
    k0.Bin = Bin; k1.Bin = Bin;
    k0.Bout = Bout; k1.Bout = Bout;
    k0.d = d; k1.d = d;

    GroupElement a11, a01, n1, n0, n_prime, an1;
    // r1 = rin / d; r0 = rin % d;
    GroupElement a1 = -rin;
    // a11 = signedDivide(a1, d); a01 = signedMod(a1, d);
    a11 = a1 / d; a01 = a1 % d;
    an1 = a1; 

    if (Bin == 64) {
        // todo: do this repeated computation outside keygen somewhere
        // When Bin = 64, we can use the following trick to avoid the overflow
        // ((uint64_t)1 << (Bin)) = ((uint64_t)1 << (Bin-1)) * 2
        // let k = ((uint64_t)1 << (Bin-1))
        // we want 2k / d and 2k % d
        // 2k % d = (2(k % d)) % d
        // 2k / d = 2*(k/d) + (2* (k % d))/d
        uint64_t d1 = ((uint64_t)1 << 63) / d.value;
        uint64_t d0 = ((uint64_t)1 << 63) % d.value;
        n0 = GroupElement((2 * d0) % d.value, Bin);
        n1 = GroupElement((2 * d1) + ((2 * d0) / d.value), Bin);
    }
    else {
        n1 = GroupElement(((uint64_t)1 << Bin) / d.value, Bin);
        n0 = GroupElement(((uint64_t)1 << Bin) % d.value, Bin);
    }

    n_prime = GroupElement((uint64_t)1 << (Bin - 1), Bin);   // n' = ceil(n/2)

    auto dcfKeys = keyGenDCF(Bin, Bin, a1, GroupElement(1, Bout));
    k0.dcfKey = dcfKeys.first; k1.dcfKey = dcfKeys.second;

    GroupElement t0 = -a1;
    auto publicICkeys = keyGenPublicIC(Bin, Bin, n_prime, GroupElement(-1, Bin), t0, GroupElement(0, Bout));
    k0.publicICkey = publicICkeys.first; k1.publicICkey = publicICkeys.second;

    auto A_share = splitShare(a01);
    k0.A_share = A_share.first; k1.A_share = A_share.second;

    auto corr_share = splitShare(GroupElement(a1.value >= n_prime.value, Bout));
    k0.corr_share = corr_share.first; k1.corr_share = corr_share.second;

    GroupElement t1 = a01 - GroupElement(a1 >= n_prime, Bin) * n0;
    auto B_share = splitShare(signedDivide(t1, d));
    k0.B_share = B_share.first; k1.B_share = B_share.second;

    auto rdiv_share = splitShare(changeBitsize(signedDivide(an1, d), Bout));
    k0.rdiv_share = rdiv_share.first; k1.rdiv_share = rdiv_share.second;

    auto rout_temp_share = splitShare(rout_temp);
    k0.rout_temp_share = rout_temp_share.first; k1.rout_temp_share = rout_temp_share.second;

    auto rout_share = splitShare(rout);
    k0.rout_share = rout_share.first; k1.rout_share = rout_share.second;

    auto scmpKeys = keyGenSCMP(Bin, Bout, rout_temp, GroupElement(0, Bin), GroupElement(0, Bout));
    k0.scmpKey = scmpKeys.first; k1.scmpKey = scmpKeys.second;

    return std::make_pair(k0, k1);
}

GroupElement evalSignedPublicDiv_First(int party, const SignedPublicDivKeyPack &key, GroupElement x, 
                    GroupElement &w_share, GroupElement &publicICresult_share)
{
    // goal is to output shares of A + rout_temp (bitsize is Bin)
    // additionally, compute and store values of w_share, publicICresult_share
    // to be later used in evalSignedPublicDiv_Second

    GroupElement a0 = x;
    // GroupElement a00 = signedMod(a0, key.d);
    GroupElement a00 = a0 % key.d;
    GroupElement n1, n0, d1, d0;

    if (key.Bin == 64) {
        // When Bin = 64, we can use the following trick to avoid the overflow
        // ((uint64_t)1 << (Bin)) = ((uint64_t)1 << (Bin-1)) * 2
        // let k = ((uint64_t)1 << (Bin-1))
        // we want 2k / d and 2k % d
        // 2k % d = (2(k % d)) % d
        // 2k / d = 2*(k/d) + (2* (k % d))/d
        uint64_t d1 = ((uint64_t)1 << 63) / key.d.value;
        uint64_t d0 = ((uint64_t)1 << 63) % key.d.value;
        n0 = GroupElement((2 * d0) % key.d.value, key.Bin);
        n1 = GroupElement((2 * d1) + ((2 * d0) / key.d.value), key.Bin);
    }
    else {
        n1 = GroupElement(((uint64_t)1 << key.Bin) / key.d.value, key.Bin);
        n0 = GroupElement(((uint64_t)1 << key.Bin) % key.d.value, key.Bin);
    }

    GroupElement n_prime = GroupElement((uint64_t)1 << (key.Bin - 1), key.Bin);   // n' = ceil(n/2)

    // w_share = evalDCF(party, -a0 - GroupElement(1, key.Bin), key.dcfKey);
    GroupElement t0 = -a0 - GroupElement(1, key.Bin);
    evalDCF(party, &w_share, t0, key.dcfKey);
    publicICresult_share = evalPublicIC(party, key.publicICkey, a0, n_prime, GroupElement(-1, key.Bin));

    GroupElement A_share = GroupElement(party, key.Bin) * a00 + key.A_share - w_share * n0 - publicICresult_share * n0 + key.rout_temp_share;
    return A_share;
}

GroupElement evalSignedPublicDiv_Second(int party, const SignedPublicDivKeyPack &key, GroupElement x,
                                GroupElement result_evalSignedPublicDiv_First, GroupElement w_share, GroupElement publicICresult_share)
{
    // goal is to output signedDivide(x – rin) + rout

    GroupElement a0, an0, n_prime, a00, n0, n1, d;
    d = key.d; a0 = x; an0 = a0; 
    // a00 = signedMod(a0, d);
    a00 = a0 % d;

    if (key.Bin == 64) {
        // When Bin = 64, we can use the following trick to avoid the overflow
        // ((uint64_t)1 << (Bin)) = ((uint64_t)1 << (Bin-1)) * 2
        // let k = ((uint64_t)1 << (Bin-1))
        // we want 2k / d and 2k % d
        // 2k % d = (2(k % d)) % d
        // 2k / d = 2*(k/d) + (2* (k % d))/d
        uint64_t d1 = ((uint64_t)1 << 63) / d.value;
        uint64_t d0 = ((uint64_t)1 << 63) % d.value;
        n0 = GroupElement((2 * d0) % d.value, key.Bin);
        n1 = GroupElement((2 * d1) + ((2 * d0) / d.value), key.Bin);
        mod(n0); mod(n1);
    }
    else {
        n1 = GroupElement(((uint64_t)1 << key.Bin) / d.value, key.Bin);
        n0 = GroupElement(((uint64_t)1 << key.Bin) % d.value, key.Bin);
        mod(n0); mod(n1);
    }

    n_prime = ((uint64_t)1 << (key.Bin - 1));   // n' = ceil(n/2)

    GroupElement corr_share = GroupElement(party * (a0 >= n_prime), key.Bout) + key.corr_share - w_share - publicICresult_share;
    GroupElement t0 = a00 - GroupElement(a0 >= n_prime, key.Bin) * n0;
    mod(t0);
    GroupElement B_share = changeBitsize(party * signedDivide(t0, key.d), key.Bout) + key.B_share;

    GroupElement C_share1 = GroupElement(party, key.Bout) - evalSCMP(party, key.scmpKey, result_evalSignedPublicDiv_First, d);
    GroupElement C_share2 = GroupElement(party, key.Bout) - evalSCMP(party, key.scmpKey, result_evalSignedPublicDiv_First, GroupElement(0, key.Bin));
    GroupElement C_share3 = GroupElement(party, key.Bout) - evalSCMP(party, key.scmpKey, result_evalSignedPublicDiv_First, -d);

    GroupElement C_share = C_share1 + C_share2 + C_share3;
    GroupElement rdiv_share = changeBitsize(party * signedDivide(an0, d), key.Bout) + key.rdiv_share
                    + corr_share * n1 + GroupElement(party, key.Bout) - C_share - B_share + key.rout_share;

    // std::cout << "corr_share " << corr_share << std::endl;
    // std::cout << "B_share " << B_share << std::endl;
    // std::cout << "C_share " << C_share << std::endl;
    // std::cout << "C_share1 " << C_share1 << std::endl;
    // std::cout << "C_share2 " << C_share2 << std::endl;
    // std::cout << "C_share2 " << C_share3 << std::endl;

    return rdiv_share;
}
