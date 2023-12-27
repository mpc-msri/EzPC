#include "reluextend.h"
#include "dcf.h"

std::pair<ReluExtendKeyPack, ReluExtendKeyPack> keyGenReluExtend(int bin, int bout, GroupElement rin, GroupElement routY, GroupElement routDrelu)
{
    std::pair<ReluExtendKeyPack, ReluExtendKeyPack> keys;
    auto dcfKeys = keyGenDCFET2(bin, rin, 1);
    keys.first.dcfKey = dcfKeys.first;
    keys.second.dcfKey = dcfKeys.second;

    mod(routDrelu, 1);
    GroupElement rd = 2 * random_ge(1) + routDrelu;
    auto rd_split = splitShare(rd, 2);
    keys.first.rd = rd_split.first;
    keys.second.rd = rd_split.second;

    GroupElement rw = random_ge(2);
    auto rw_split = splitShare(rw, 2);
    keys.first.rw = rw_split.first;
    keys.second.rw = rw_split.second;

    GroupElement ri = 2 * rd + rw;
    mod(ri, 2);

    GroupElement p[4] = {0, 0, 0, 0};
    p[(4-ri)%4] = 1;
    for (int i = 0; i < 4; ++i) {
        auto p_split = splitShare(p[i], bout);
        keys.first.p[i] = p_split.first;
        keys.second.p[i] = p_split.second;
    }

    GroupElement q[2];
    if (rd % 2 == 0) {
        q[0] = routY;
        q[1] = routY - rin;
    } else {
        q[0] = routY - rin;
        q[1] = routY;
    }
    for (int i = 0; i < 2; ++i) {
        auto q_split = splitShare(q[i], bout);
        keys.first.q[i] = q_split.first;
        keys.second.q[i] = q_split.second;
    }

    return keys;
}
