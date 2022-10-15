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
    GroupElement x_s(x.value & ones, shift);

    // last n-1 bits of x
    uint8_t x_msb = x[0];
    // todo: bitsize of x_n should have been k.Bin - 1
    uint64_t x_n = x.value & (((uint64_t)1 << (k.Bin - 1)) - 1);
    // std::cout << "x_n " << x_n.value << std::endl;

    GroupElement dcfIdx = ((uint64_t)1 << shift) - x_s - 1;
    dcfIdx.bitsize = shift;
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

    return res; 
}
