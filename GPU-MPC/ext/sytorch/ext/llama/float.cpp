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

#include "float.h"
#include "pubdiv.h"


void fill_pq(GroupElement *p, GroupElement *q, int n)
{
    for(int i = 2*n; i >= 1; --i)
    {
        int idx = i - 1;
        if (i == 1) {
            p[idx] = 0;
        }
        else if (i <= n) {
            p[idx] = (1ULL<<(i - 2));
        }
        else if (i == n + 1) {
            p[idx] = (1ULL<<(n - 1));
        }
        else {
            p[idx] = -(1ULL<<(2*n - i + 1))+ 1;
            mod(p[idx], n);
        }

        if (i == 2*n) {
            q[idx] = -1;
            mod(q[idx], n);
        }
        else {
            q[idx] = p[idx+1] - 1;
        }
    }
}

pair<FixToFloatKeyPack> keyGenFixToFloat(int bin, int scale, GroupElement rin, GroupElement *p, GroupElement *q)
{
    pair<FixToFloatKeyPack> keys;
    auto micKeys = keyGenMIC(bin, bin, 2*bin, p, q, rin, nullptr);
    keys.first.micKey = micKeys.first;
    keys.second.micKey = micKeys.second;

    GroupElement rs = random_ge(1);
    auto rs_split = splitShare(rs, 1);
    keys.first.rs = rs_split.first;
    keys.second.rs = rs_split.second;

    GroupElement rpow = random_ge(bin);
    auto rpow_split = splitShare(rpow, bin);
    keys.first.rpow = rpow_split.first;
    keys.second.rpow = rpow_split.second;

    GroupElement rselect = random_ge(bin);
    auto selectKeys = keyGenSelect(bin, rs, rin, rselect);
    keys.first.selectKey = selectKeys.first;
    keys.second.selectKey = selectKeys.second;

    GroupElement ry = 2 * rselect - rin;
    auto ry_split = splitShare(ry, bin);
    keys.first.ry = ry_split.first;
    keys.second.ry = ry_split.second;

    GroupElement rm = ry * rpow;
    auto rm_split = splitShare(rm, bin);
    keys.first.rm = rm_split.first;
    keys.second.rm = rm_split.second;

    return keys;
}

inline uint64_t fp32_bias(uint64_t x) {
    return (x + 127) % 256;
}

inline uint64_t fp32_unbias(uint64_t x) {
    return (x - 127) % 256;
}

void evalFixToFloat_1(int party, int bin, int scale, GroupElement x, const FixToFloatKeyPack &key, GroupElement *p, GroupElement *q, GroupElement &m, GroupElement &e, GroupElement &z, GroupElement &s, GroupElement &pow, GroupElement &sm)
{
    mod(x, bin);
    GroupElement t[2*bin];
    evalMIC(party, bin, bin, 2*bin, p, q, x, key.micKey, t);
    z = t[0];
    mod(z, 1);
    s = 0;
    for(int i = bin; i < 2*bin; ++i)
    {
        s += t[i];
    }
    mod(s, 1);
    sm = s + key.rs;
    e = fp32_bias(-126) * t[0] + fp32_bias(bin - scale - 1) * t[bin];
    pow = key.rpow + t[bin];
    for (int i = 2; i <= bin; ++i) {
        e += (t[i-1] + t[2*bin + 1 - i]) * fp32_bias(i - scale - 2);
        pow += (t[i-1] + t[2*bin + 1 - i]) * (1ULL<<(bin - i + 1));
    }
}

GroupElement adjust(GroupElement m, GroupElement e) 
{
    mod(m, 24);
    mod(e, 10);
    if ((e >= 512) && (e <= (1024-24))) {
        return 0;
    }
    else if (e >= 512) {
        uint64_t s = 1024 - e;
        return m >> s;
    }
    else if (e < 64) {
        return m << (e);
    }
    else {
        return 0;
    }
}


pair<FloatToFixKeyPack> keyGenFloatToFix(int bin, int scale, GroupElement rout)
{
    pair<FloatToFixKeyPack> keys;
    GroupElement rm = random_ge(24);
    auto rm_split = splitShare(rm, 24);
    //shares of r(m)
    keys.first.rm = rm_split.first;
    keys.second.rm = rm_split.second;

    GroupElement re = random_ge(10);
    auto re_split = splitShare(re, 10);
    //shares of r(e) 
    keys.first.re = re_split.first;
    keys.second.re = re_split.second;

    auto dcfKeys = keyGenDCF(24, 1, rm, 1);
    //shares of dcfKey using rm
    keys.first.dcfKey = dcfKeys.first;
    keys.second.dcfKey = dcfKeys.second;

    GroupElement rw = random_ge(1); //
    auto rw_split = splitShare(rw, 1);
    //shares of r(w)
    keys.first.rw = rw_split.first;
    keys.second.rw = rw_split.second;


    GroupElement rh = random_ge(bin); 
    auto rh_split = splitShare(rh, bin);
    //shares of r(h)
    keys.first.rh = rh_split.first;
    keys.second.rh = rh_split.second;

    GroupElement rt = random_ge(bin); 
    auto rt_split = splitShare(rt, bin);
    
    //shares of r(t) //no need to split rt as it is not used in any other function in online mode
    //keys.first.rt = rt_split.first;
    //keys.second.rt = rt_split.second;

    auto selectKeys = keyGenSelect(bin, rw, rh, 0);
    //shares of select keys based on random r(h) and r(w)
    keys.first.selectKey = selectKeys.first;
    keys.second.selectKey = selectKeys.second;

    auto arskeys = keyGenARS(bin, bin, 23, rt, rout); // check the shift and bin value
    //shares of ARS keys based on random r(t) and rout
    keys.first.arsKey = arskeys.first;
    keys.second.arsKey = arskeys.second;

    GroupElement p;
    GroupElement q;
    for(int i = 0; i < 1024; ++i) {
        if (i == ((1024 - re) % 1024)) {
            p = 1ULL;
        }
        else {
            p = 0ULL;
        }
        auto p_split = splitShare(p, bin);
        keys.first.p[i] = p_split.first;
        keys.second.p[i] = p_split.second;
        q = rt - (rm * pow_helper(scale,(i-re)%1024));
        auto q_split = splitShare(q, bin);
        keys.first.q[i] = q_split.first;
        keys.second.q[i] = q_split.second;
    }
    //counter++; for debugging
    return keys;
}
