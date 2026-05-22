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

#include <llama/dcf.h>
#include <omp.h>

using namespace osuCrypto;
// uint64_t aes_evals_count = 0;

#define SERVER0 0
#define SERVER1 1
#define GROUP_LOOP(s)                  \
    int lp = (evalGroupIdxStart + groupSize) % groupSize;        \
    int ctr = 0;                       \
    while(ctr < evalGroupIdxLen)       \
    {                                  \
        s                              \
        lp = (lp + 1) % groupSize;     \
        ctr++;                         \
    }

void clearAESevals()
{
    // aes_evals_count = 0;
}

inline int bytesize(const int bitsize) {
    return (bitsize % 8) == 0 ? bitsize / 8 : (bitsize / 8)  + 1;
}

void convert(const int bitsize, const int groupSize, const block &b, uint64_t *out)
{
    static const block notThreeBlock = toBlock(~0, ~3);
    const int bys = bytesize(bitsize);
    const int totalBys = bys * groupSize;
    if (bys * groupSize <= 16) {
        uint8_t *bptr = (uint8_t *)&b;
        for(int i = 0; i < groupSize; i++) {
            out[i] = *(uint64_t *)(bptr + i * bys);
        }
    }
    else {
        int numblocks = totalBys % 16 == 0 ? totalBys / 16 : (totalBys / 16) + 1;
        AES aes(b);
        block pt[numblocks];
        block ct[numblocks];
        for(int i = 0; i < numblocks; i++) {
            pt[i] = toBlock(0, i);
        }
        aes.ecbEncBlocks(pt, numblocks, ct);
        uint8_t *bptr = (uint8_t *)ct;
        for(int i = 0; i < groupSize; i++) {
            out[i] = *(uint64_t *)(bptr + i * bys);
        }
    }
}

block traverseOneDCF(int Bin, int Bout, int groupSize, int party,
                        const block &s,
                        const block &cw,
                        const u8 &keep,
                        GroupElement *v_share,
                        GroupElement *v,
                        uint64_t level,
                        bool geq,
                        int evalGroupIdxStart,
                        int evalGroupIdxLen)

{
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block TwoBlock = toBlock(0, 2);
    static const block ThreeBlock = toBlock(0, 3);
    static const block blocks[4] = {ZeroBlock, TwoBlock, OneBlock, ThreeBlock};

    block stcw;
    block ct[2]; // {tau, v_this_level}
    u8 t_previous = lsb(s);
    const auto scw = (cw & notThreeBlock);
    block ds[] = { ((cw >> 1) & OneBlock), (cw & OneBlock) };
    const auto mask = zeroAndAllOne[t_previous];
    auto ss = s & notThreeBlock;

    AES ak(ss);
    ak.ecbEncTwoBlocks(blocks + 2 * keep, ct);

    stcw = ((scw ^ ds[keep]) & mask) ^ ct[0];
    uint64_t sign = (party == SERVER1) ? -1 : 1;
    block temp = ZeroBlock;
    uint64_t v_this_level_converted[groupSize];
    convert(Bout, groupSize, ct[1], v_this_level_converted);
    GROUP_LOOP(
        v_share[lp] = v_share[lp] + sign * (v_this_level_converted[lp] + t_previous * (*(v + ((int)level) * groupSize + lp)));
    )
    return stcw;
}


block traversePathDCF(int Bin, int Bout, int groupSize, int party,
                        GroupElement idx,
                        block *k,
                        GroupElement *v_share,
                        GroupElement *v,
                        bool geq,
                        int evalGroupIdxStart,
                        int evalGroupIdxLen)
{
    block s = _mm_loadu_si128(k);
    GROUP_LOOP(v_share[lp] = 0;)

    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx >> (Bin - 1 - i)) & 1;
        s = traverseOneDCF(Bin, Bout, groupSize, party, s, _mm_loadu_si128(k + (i + 1)), keep, v_share, v, i, geq, evalGroupIdxStart, evalGroupIdxLen);
    }
    return s;
}


// Real Endpoints
std::pair<DCFKeyPack, DCFKeyPack> keyGenDCF(int Bin, int Bout, int groupSize,
                GroupElement idx, GroupElement* payload)
{
    // idx: bitsize Bin, payload: bitsize Bout & size groupSize
    bool greaterThan = false;

    static const block notOneBlock = toBlock(~0, ~1);
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block TwoBlock = toBlock(0, 2);
    static const block ThreeBlock = toBlock(0, 3);
    const static block pt[4] = {ZeroBlock, OneBlock, TwoBlock, ThreeBlock};

    int tid = omp_get_thread_num();
    auto s = LlamaConfig::prngs[tid].get<std::array<block, 2>>();
    block si[2][2];
    block vi[2][2];

    GroupElement *v_alpha = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; ++i)
    {
        v_alpha[i] = 0;
    }

    block *k0 = new block[Bin + 1];
    block *k1 = new block[Bin + 1];
    GroupElement *v0 = new GroupElement[Bin * groupSize];    // bitsize Bout, size Bin x groupSize
    GroupElement *g0 = new GroupElement[groupSize];     // bitsize: Bout

    s[0] = (s[0] & notOneBlock) ^ ((s[1] & OneBlock) ^ OneBlock);
    k0[0] = s[0];
    k1[0] = s[1];
    block ct[4];

    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx >> (Bin - 1 - i)) & 1;
        auto a = toBlock(keep);

        auto ss0 = s[0] & notThreeBlock;
        auto ss1 = s[1] & notThreeBlock;

        AES ak0(ss0);
        AES ak1(ss1);
        ak0.ecbEncFourBlocks(pt, ct);
        si[0][0] = ct[0];
        si[0][1] = ct[1];
        vi[0][0] = ct[2];
        vi[0][1] = ct[3];
        ak1.ecbEncFourBlocks(pt, ct);
        si[1][0] = ct[0];
        si[1][1] = ct[1];
        vi[1][0] = ct[2];
        vi[1][1] = ct[3];

        auto ti0 = lsb(s[0]);
        auto ti1 = lsb(s[1]);
        GroupElement sign = (ti1 == 1) ? -1 : +1;

        uint64_t vi_01_converted[groupSize];
        uint64_t vi_11_converted[groupSize];
        uint64_t vi_10_converted[groupSize];
        uint64_t vi_00_converted[groupSize];
        convert(Bout, groupSize, vi[0][keep], vi_00_converted);
        convert(Bout, groupSize, vi[1][keep], vi_10_converted);
        convert(Bout, groupSize, vi[0][keep ^ 1], vi_01_converted);
        convert(Bout, groupSize, vi[1][keep ^ 1], vi_11_converted);

        for (int lp = 0; lp < groupSize; ++lp)
        {
            v0[i * groupSize + lp] = sign * (-v_alpha[lp] - vi_01_converted[lp] + vi_11_converted[lp]);
            if (keep == 0 && greaterThan)
            {
                // Lose is R
                v0[i * groupSize + lp] = v0[i * groupSize + lp] + sign * payload[lp];
            }
            else if (keep == 1 && !greaterThan)
            {
                // Lose is L
                v0[i * groupSize + lp] = v0[i * groupSize + lp] + sign * payload[lp];
            }
            v_alpha[lp] = v_alpha[lp] - vi_10_converted[lp] + vi_00_converted[lp] + sign * v0[i * groupSize + lp];
        }

        std::array<block, 2> siXOR{si[0][0] ^ si[1][0], si[0][1] ^ si[1][1]};

        // get the left and right t_CW bits
        std::array<block, 2> t{
            (OneBlock & siXOR[0]) ^ a ^ OneBlock,
            (OneBlock & siXOR[1]) ^ a};

        // take scw to be the bits [127, 2] as scw = s0_loss ^ s1_loss
        auto scw = siXOR[keep ^ 1] & notThreeBlock;

        k0[i + 1] = k1[i + 1] = scw           // set bits [127, 2] as scw = s0_loss ^ s1_loss
                                ^ (t[0] << 1) // set bit 1 as tL
                                ^ t[1];       // set bit 0 as tR

        auto si0Keep = si[0][keep];
        auto si1Keep = si[1][keep];

        // extract the t^Keep_CW bit
        auto TKeep = t[keep];

        // set the next level of s,t
        s[0] = si0Keep ^ (zeroAndAllOne[ti0] & (scw ^ TKeep));
        s[1] = si1Keep ^ (zeroAndAllOne[ti1] & (scw ^ TKeep));
    }

    uint64_t s0_converted[groupSize];
    uint64_t s1_converted[groupSize];
    convert(Bout, groupSize, s[0] & notThreeBlock, s0_converted);
    convert(Bout, groupSize, s[1] & notThreeBlock, s1_converted);

    for (int lp = 0; lp < groupSize; ++lp)
    {
        g0[lp] = s1_converted[lp] - s0_converted[lp] - v_alpha[lp];
        if (lsb(s[1]) == 1)
        {
            g0[lp] = g0[lp] * -1;
        }
    }

    return std::make_pair(DCFKeyPack(Bin, Bout, groupSize, k0, g0, v0), DCFKeyPack(Bin, Bout, groupSize, k1, g0, v0));
}

std::pair<DCFKeyPack, DCFKeyPack> keyGenDCF(int Bin, int Bout,
                GroupElement idx, GroupElement payload)
{
    // idx: bitsize Bin, payload: bitsize Bout
    return keyGenDCF(Bin, Bout, 1, idx, &payload);
}

void evalDCF(int Bin, int Bout, int groupSize, 
                GroupElement *out, // groupSize
                int party, GroupElement idx, 
                block *k, // bin + 1
                GroupElement *g , // groupSize
                GroupElement *v, // bin * groupSize
                bool geq /*= false*/, int evalGroupIdxStart /*= 0*/,
                int evalGroupIdxLen /*= -1*/)
{
    if (evalGroupIdxLen == 0)
    {
        return;
    }
    if (evalGroupIdxLen == -1)
    {
        evalGroupIdxLen = groupSize;
    }

    auto s = traversePathDCF(Bin, Bout, groupSize, party, idx, k, out, v, geq, evalGroupIdxStart, evalGroupIdxLen);

    u8 t = lsb(s);
    block temp = ZeroBlock;

    uint64_t s_converted[groupSize];
    static const block notThreeBlock = toBlock(~0, ~3);
    convert(Bout, groupSize, s & notThreeBlock, s_converted);
    GROUP_LOOP(
        GroupElement final_term = s_converted[lp];
        if (t)
            final_term = final_term + g[lp];
        if (party == SERVER1)
        {
            final_term = -final_term;
        } out[lp] = out[lp] + final_term;)
}

void evalDCF(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key)
{
    evalDCF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k, key.g, key.v);
}

void evalDCFPartial(int party, GroupElement *res, GroupElement idx, const DCFKeyPack &key, int start, int len)
{
    evalDCF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.k, key.g, key.v, false, start, len);
}

// Dual DCF

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, int groupSize, GroupElement idx, GroupElement *payload1, GroupElement *payload2)
{
    DualDCFKeyPack key0, key1;

    key0.Bin = Bin; key1.Bin = Bin;
    key0.Bout = Bout; key1.Bout = Bout;
    key0.groupSize = groupSize; key1.groupSize = groupSize;

    GroupElement *payload = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; i++) {
        payload[i] = payload1[i] - payload2[i];
    }

    auto keys = keyGenDCF(Bin, Bout, groupSize, idx, payload);
    key0.dcfKey = keys.first, key1.dcfKey = keys.second;
    

    key0.sb = new GroupElement[groupSize];
    key1.sb = new GroupElement[groupSize];

    for (int i = 0; i < groupSize; i++) {
        auto payload2_split = splitShare(payload2[i], Bout);
        key0.sb[i] = payload2_split.first; key1.sb[i] = payload2_split.second;
    }

    return std::make_pair(key0, key1);
}

std::pair<DualDCFKeyPack, DualDCFKeyPack> keyGenDualDCF(int Bin, int Bout, GroupElement idx, GroupElement payload1, GroupElement payload2)
{
    return keyGenDualDCF(Bin, Bout, 1, idx, &payload1, &payload2);
}

void evalDualDCF(int party, GroupElement* res, GroupElement idx, const DualDCFKeyPack &key)
{
    evalDCF(key.Bin, key.Bout, key.groupSize, res, party, idx, key.dcfKey.k, key.dcfKey.g, key.dcfKey.v);
    for (int i = 0; i < key.groupSize; i++) {
        res[i] = res[i] + key.sb[i];
    }
}
