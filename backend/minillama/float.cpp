#include "float.h"

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
    e = -126 * t[0] + (bin - scale - 1) * t[bin];
    pow = key.rpow + t[bin];
    for (int i = 2; i <= bin; ++i) {
        e += (t[i-1] + t[2*bin + 1 - i]) * (i - scale - 2);
        pow += (t[i-1] + t[2*bin + 1 - i]) * (1ULL<<(bin - i + 1));
    }
}

