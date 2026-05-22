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

#include <llama/array.h>
#include "mic.h"

std::pair<MICKeyPack, MICKeyPack> keyGenMIC(int bin, int bout, int m, uint64_t *p, uint64_t *q, GroupElement rin, GroupElement *rout)
{
    std::pair<MICKeyPack, MICKeyPack> keys;
    keys.first.z = make_array<GroupElement>(m);
    keys.second.z = make_array<GroupElement>(m);

    GroupElement gamma = rin - 1;
    auto dcfKeys = keyGenDCF(bin, bout, gamma, 1);
    keys.first.dcfKey = dcfKeys.first;
    keys.second.dcfKey = dcfKeys.second;

    GroupElement neg1 = -1;
    mod(neg1, bin);
    for (int i = 1; i <= m; ++i)
    {
        GroupElement qi_prime(q[i-1] + 1);
        mod(qi_prime, bin);
        GroupElement alpha_ip(p[i-1] + rin);
        mod(alpha_ip, bin);
        GroupElement alpha_iq(q[i-1] + rin);
        mod(alpha_iq, bin);
        GroupElement alpha_iq_prime(q[i-1] + rin + 1);
        mod(alpha_iq_prime, bin);

        GroupElement z((rout == nullptr ? 0 : rout[i-1]) + (alpha_ip > alpha_iq) - (alpha_ip > p[i-1]) + (alpha_iq_prime > qi_prime) + (alpha_iq == neg1));
        mod(z, bout);

        auto zpair = splitShare(z, bout);
        keys.first.z[i-1] = zpair.first;
        keys.second.z[i-1] = zpair.second;
    }

    return keys;
}

// assumes intervals [pi, qi] are sorted
void evalMIC(int party, int bin, int bout, int m, uint64_t *p, uint64_t *q, GroupElement x, const MICKeyPack &key, GroupElement *y)
{
    GroupElement sp = 0, sq = 0;
    for(int i = 1; i <= m; ++i)
    {
        GroupElement qi_prime(q[i-1] + 1);
        mod(qi_prime, bin);
        GroupElement xi_p(x - 1 - p[i-1]);
        mod(xi_p, bin);
        GroupElement xi_qprime(x - 1 - qi_prime);
        mod(xi_qprime, bin);
        
        if ((i != 1) && (p[i-1] == q[i-2] + 1))
        {
            sp = sq;
        }
        else
        {
            evalDCF(party, &sp, xi_p, key.dcfKey);
        }
        evalDCF(party, &sq, xi_qprime, key.dcfKey);
        y[i-1] = sq - sp + key.z[i-1];
        if (party == 1)
        {
            y[i-1] = y[i-1] + (x > p[i-1]) - (x > qi_prime);
            mod(y[i-1], bout);
        }
    }
}
