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

#include <llama/comms.h>
#include <llama/dcf.h>
#include "mult.h"
#include <assert.h>
#include <utility>
using namespace LlamaConfig;

std::pair<MultKey, MultKey> MultGen(GroupElement rin1, GroupElement rin2, GroupElement rout)
{
    
    MultKey k1, k2;
    // k1.Bin = Bin; k2.Bin = Bin;
    // k1.Bout = Bout; k2.Bout = Bout;

    GroupElement c  = rin1 * rin2 + rout;
    auto a_split = splitShare(rin1, 64);
    auto b_split = splitShare(rin2, 64);
    auto c_split = splitShare(c, 64);
    
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
    if (party == DEALER) {
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

std::pair<SquareKey, SquareKey> keyGenSquare(GroupElement rin, GroupElement rout)
{
    SquareKey k1, k2;

    GroupElement c  = rin * rin + rout;
    auto b_split = splitShare(2 * rin, 64);
    auto c_split = splitShare(c, 64);
    
    k1.b = (b_split.first);
    k1.c = (c_split.first);
    
    k2.b = (b_split.second);
    k2.c = (c_split.second);
    
    return std::make_pair(k1, k2);
}

GroupElement evalSquare(int party, GroupElement x, const SquareKey &k)
{
    return party * x * x - x * k.b + k.c;
}
