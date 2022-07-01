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
#include "comms.h"
#include "dcf.h"
#include "mult.h"
#include <assert.h>
#include <utility>

std::pair<MultKey, MultKey> MultGen(GroupElement rin1, GroupElement rin2, GroupElement rout)
{
    assert(rin1.bitsize == rin2.bitsize);
    assert(rout.bitsize == rin2.bitsize);
    
    MultKey k1, k2;
    // k1.Bin = Bin; k2.Bin = Bin;
    // k1.Bout = Bout; k2.Bout = Bout;

    GroupElement c  = rin1 * rin2 + rout;
    auto a_split = splitShare(rin1);
    auto b_split = splitShare(rin2);
    auto c_split = splitShare(c);
    
    k1.a = (a_split.first);
    k1.b = (b_split.first);
    k1.c = (c_split.first);
    
    k2.a = (a_split.second);
    k2.b = (b_split.second);
    k2.c = (c_split.second);
    
    return std::make_pair(k1, k2);
}

GroupElement MultEval(int party, const MultKey &k, const GroupElement &l, const GroupElement &r)
{
    return party * (l * r) - l * k.b - r * k.a + k.c;
}

GroupElement mult_helper(uint8_t party, GroupElement x, GroupElement y, GroupElement x_mask, GroupElement y_mask)
{
    if ((party == DEALER)) {
        GroupElement z_mask = random_ge(64);
        std::pair<MultKey, MultKey> keys = MultGen(x_mask, y_mask, z_mask);
        server->send_mult_key(keys.first);
        client->send_mult_key(keys.second);
        return z_mask;
    }
    else {
        MultKey key = dealer->recv_mult_key();
        GroupElement e = MultEval(party - SERVER, key, x, y);
        peer->send_input(e);
        return e + peer->recv_input();
    }
}

std::pair<MultKeyNew, MultKeyNew> new_mult_unsigned_gen(int bw1, int bw2, uint64_t rin1, uint64_t rin2, uint64_t rout) {
    GroupElement r1(rin1, bw1);
    GroupElement r2(rin2, bw2);
    GroupElement r3(rin1 * rin2 + rout, bw1 + bw2);
    GroupElement one1(1, bw1);
    GroupElement one2(1, bw2);
    auto k1 = keyGenDCF(bw2, bw1, 1, r2, &one1);
    auto k2 = keyGenDCF(bw1, bw2, 1, r1, &one2);
    auto k3 = keyGenDCF(bw2, bw1, 1, r2, &r1);
    auto k4 = keyGenDCF(bw1, bw2, 1, r1, &r2);
    r1.bitsize = bw1 + bw2;
    r2.bitsize = bw1 + bw2;
    auto a = splitShare(r1);
    auto b = splitShare(r2);
    auto c = splitShare(r3);

    mod(a.first);
    mod(b.first);
    mod(c.first);
    mod(a.second);
    mod(b.second);
    mod(c.second);

    MultKeyNew key1 {a.first, b.first, c.first, k1.first, k2.first, k3.first, k4.first}; 
    MultKeyNew key2 {a.second, b.second, c.second, k1.second, k2.second, k3.second, k4.second};
    return std::make_pair(key1, key2);
}

uint64_t new_mult_unsigned_eval(int party, int bw1, int bw2, const MultKeyNew &k, const uint64_t x, const uint64_t y) {
    GroupElement t1(0, bw1);
    GroupElement t2(0, bw2);
    GroupElement t3(0, bw1);
    GroupElement t4(0, bw2);
    GroupElement xg(x, bw1);
    GroupElement yg(y, bw2);

    evalDCF(party, &t1, yg, k.k1);
    evalDCF(party, &t2, xg, k.k2);
    evalDCF(party, &t3, yg, k.k3);
    evalDCF(party, &t4, xg, k.k4);
    mod(t1);
    mod(t2);
    mod(t3);
    mod(t4);

    uint64_t M = 1L << bw1;
    uint64_t N = 1L << bw2;

    uint64_t res = party * x * y - x * k.b.value - y * k.a.value + k.c.value + t1.value * x * N + t2.value * y * M - t3.value * N - t4.value * M;
    return res;
}

std::pair<MultKeyNew, MultKeyNew> new_mult_signed_gen(int bw1, int bw2, uint64_t rin1, uint64_t rin2, uint64_t rout)
{
    return new_mult_unsigned_gen(bw1, bw2, rin1, rin2, rout);
}

uint64_t new_mult_signed_eval(int party, int bw1, int bw2, const MultKeyNew &k, const uint64_t x, const uint64_t y) {
    GroupElement t1(0, bw1);
    GroupElement t2(0, bw2);
    GroupElement t3(0, bw1);
    GroupElement t4(0, bw2);
    uint64_t x2 = (x + (1<<(bw1-1))) % (1<<bw1);
    uint64_t y2 = (y + (1<<(bw2-1))) % (1<<bw2);
    GroupElement xg(x2, bw1);
    GroupElement yg(y2, bw2);

    evalDCF(party, &t1, yg, k.k1);
    evalDCF(party, &t2, xg, k.k2);
    evalDCF(party, &t3, yg, k.k3);
    evalDCF(party, &t4, xg, k.k4);
    mod(t1);
    mod(t2);
    mod(t3);
    mod(t4);

    uint64_t M = 1L << bw1;
    uint64_t N = 1L << bw2;

    uint64_t res = party * x2 * y2 - x2 * k.b.value - y2 * k.a.value + k.c.value + t1.value * x2 * N + t2.value * y2 * M - t3.value * N - t4.value * M - (party * x2 - k.a.value + M * t2.value) * (N>>1) - (party * y2 - k.b.value + N * t1.value) * (M>>1) + party * ((M*N) >> 2);
    return res;
}
