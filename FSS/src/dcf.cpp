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

#include "dcf.h"
#include "mini_aes.h"

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

GroupElement getDataFromBlock(int bitsize, block b, const int i)
{
    if (bitsize <= 8)
    {
        switch (i)
        {
        case 0:
            return GroupElement(_mm_extract_epi8(b, 0), bitsize);
        case 1:
            return GroupElement(_mm_extract_epi8(b, 1), bitsize);
        case 2:
            return GroupElement(_mm_extract_epi8(b, 2), bitsize);
        case 3:
            return GroupElement(_mm_extract_epi8(b, 3), bitsize);
        case 4:
            return GroupElement(_mm_extract_epi8(b, 4), bitsize);
        case 5:
            return GroupElement(_mm_extract_epi8(b, 5), bitsize);
        case 6:
            return GroupElement(_mm_extract_epi8(b, 6), bitsize);
        case 7:
            return GroupElement(_mm_extract_epi8(b, 7), bitsize);
        case 8:
            return GroupElement(_mm_extract_epi8(b, 8), bitsize);
        case 9:
            return GroupElement(_mm_extract_epi8(b, 9), bitsize);
        case 10:
            return GroupElement(_mm_extract_epi8(b, 10), bitsize);
        case 11:
            return GroupElement(_mm_extract_epi8(b, 11), bitsize);
        case 12:
            return GroupElement(_mm_extract_epi8(b, 12), bitsize);
        case 13:
            return GroupElement(_mm_extract_epi8(b, 13), bitsize);
        case 14:
            return GroupElement(_mm_extract_epi8(b, 14), bitsize);
        case 15:
            return GroupElement(_mm_extract_epi8(b, 15), bitsize);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else if (bitsize <= 16)
    {
        switch (i)
        {
        case 0:
            return GroupElement(_mm_extract_epi16(b, 0), bitsize);
        case 1:
            return GroupElement(_mm_extract_epi16(b, 1), bitsize);
        case 2:
            return GroupElement(_mm_extract_epi16(b, 2), bitsize);
        case 3:
            return GroupElement(_mm_extract_epi16(b, 3), bitsize);
        case 4:
            return GroupElement(_mm_extract_epi16(b, 4), bitsize);
        case 5:
            return GroupElement(_mm_extract_epi16(b, 5), bitsize);
        case 6:
            return GroupElement(_mm_extract_epi16(b, 6), bitsize);
        case 7:
            return GroupElement(_mm_extract_epi16(b, 7), bitsize);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else if (bitsize <= 32)
    {
        switch (i)
        {
        case 0:
            return GroupElement(_mm_extract_epi32(b, 0), bitsize);
        case 1:
            return GroupElement(_mm_extract_epi32(b, 1), bitsize);
        case 2:
            return GroupElement(_mm_extract_epi32(b, 2), bitsize);
        case 3:
            return GroupElement(_mm_extract_epi32(b, 3), bitsize);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else if (bitsize <= 64)
    {
        switch (i)
        {
        case 0:
            return GroupElement(_mm_extract_epi64(b, 0), bitsize);
        case 1:
            return GroupElement(_mm_extract_epi64(b, 1), bitsize);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else
    {
        throw std::invalid_argument("bitsize should be less than or equal to 64");
    }
}

uint64_t getDataFromBlock_old(int bitsize, block b, const int i)
{
    if (bitsize <= 8)
    {
        switch (i)
        {
        case 0:
            return _mm_extract_epi8(b, 0);
        case 1:
            return _mm_extract_epi8(b, 1);
        case 2:
            return _mm_extract_epi8(b, 2);
        case 3:
            return _mm_extract_epi8(b, 3);
        case 4:
            return _mm_extract_epi8(b, 4);
        case 5:
            return _mm_extract_epi8(b, 5);
        case 6:
            return _mm_extract_epi8(b, 6);
        case 7:
            return _mm_extract_epi8(b, 7);
        case 8:
            return _mm_extract_epi8(b, 8);
        case 9:
            return _mm_extract_epi8(b, 9);
        case 10:
            return _mm_extract_epi8(b, 10);
        case 11:
            return _mm_extract_epi8(b, 11);
        case 12:
            return _mm_extract_epi8(b, 12);
        case 13:
            return _mm_extract_epi8(b, 13);
        case 14:
            return _mm_extract_epi8(b, 14);
        case 15:
            return _mm_extract_epi8(b, 15);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else if (bitsize <= 16)
    {
        switch (i)
        {
        case 0:
            return _mm_extract_epi16(b, 0);
        case 1:
            return _mm_extract_epi16(b, 1);
        case 2:
            return _mm_extract_epi16(b, 2);
        case 3:
            return _mm_extract_epi16(b, 3);
        case 4:
            return _mm_extract_epi16(b, 4);
        case 5:
            return _mm_extract_epi16(b, 5);
        case 6:
            return _mm_extract_epi16(b, 6);
        case 7:
            return _mm_extract_epi16(b, 7);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else if (bitsize <= 32)
    {
        switch (i)
        {
        case 0:
            return _mm_extract_epi32(b, 0);
        case 1:
            return _mm_extract_epi32(b, 1);
        case 2:
            return _mm_extract_epi32(b, 2);
        case 3:
            return _mm_extract_epi32(b, 3);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else if (bitsize <= 64)
    {
        switch (i)
        {
        case 0:
            return _mm_extract_epi64(b, 0);
        case 1:
            return _mm_extract_epi64(b, 1);
        default:
            throw std::invalid_argument("bad selector");
        }
    }
    else
    {
        throw std::invalid_argument("bitsize should be less than or equal to 64");
    }
}

GroupElement convert_new(int bitsize, block b, block &temp, const int offset)
{
    int bytesize = bitsize / 8; 
    static const block notThreeBlock = toBlock(~0, ~3);
    const int outputPerBlock = sizeof(block) / bytesize;
    // static AES aes0(ZeroBlock);

    block s = (b & notThreeBlock) ^ toBlock(offset / outputPerBlock);

    if (offset == 0) {
        temp = s;
    }
    else if (offset % outputPerBlock == 0)
    {
        temp = aes_enc(s, 0);
        // aes_evals_count++;
        temp = temp ^ s;
    }
    else if (eq(temp, ZeroBlock)) {
        temp = aes_enc(s, 0);
        // aes_evals_count++;
        temp = temp ^ s;
    }

    return getDataFromBlock(bitsize, temp, offset % outputPerBlock);
}

uint64_t convert_new_uint(int bitsize, block b, block &temp, const int offset)
{
    int bytesize = bitsize / 8; 
    const block notThreeBlock = toBlock(~0, ~3);
    const int outputPerBlock = sizeof(block) / bytesize;

    block s = (b & notThreeBlock) ^ toBlock(offset / outputPerBlock);

    if (offset == 0) {
        temp = s;
    }
    else if (offset % outputPerBlock == 0)
    {
        temp = aes_enc(s, 0);
        // aes_evals_count++;
        temp = temp ^ s;
    }
    else if (eq(temp, ZeroBlock)) {
        temp = aes_enc(s, 0);
        // aes_evals_count++;
        temp = temp ^ s;
    }

    return getDataFromBlock_old(bitsize, temp, offset % outputPerBlock);
}

uint64_t convert_old(int bitsize, block b, block &temp, const int offset)
{
    int bytesize = bitsize / 8; 
    static const block notThreeBlock = toBlock(~0, ~3);
    const int outputPerBlock = sizeof(block) / bytesize;
    static AES aes0(ZeroBlock);

    block s = (b & notThreeBlock) ^ toBlock(offset / outputPerBlock);

    if (offset == 0) {
        temp = s;
    }
    else if (offset % outputPerBlock == 0)
    {
        temp = aes0.ecbEncBlock(s);
        // aes_evals_count++;
        temp = temp ^ s;
    }
    else if (eq(temp, ZeroBlock)) {
        temp = aes_enc(s, 0);
        // aes_evals_count++;
        temp = temp ^ s;
    }

    return getDataFromBlock_old(bitsize, temp, offset % outputPerBlock);
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
    const block notThreeBlock = toBlock(~0, ~3);
    const block TwoBlock = toBlock(0, 2);
    const block ThreeBlock = toBlock(0, 3);

    block tau, v_this_level, stcw;
    u8 t_previous = lsb(s);
    const auto scw = (cw & notThreeBlock);
    block ds[] = { ((cw >> 1) & OneBlock), (cw & OneBlock) };
    const auto mask = zeroAndAllOne[t_previous];
    auto ss = s & notThreeBlock;
    tau = aes_enc(ss, keep);
    tau = tau ^ ss;
    v_this_level = aes_enc(ss, keep + 2);
    stcw = ((scw ^ ds[keep]) & mask) ^ tau;
    // aes_evals_count += 2;
    v_this_level ^= ss;

    uint64_t sign = (party == SERVER1) ? -1 : 1;
    block temp = ZeroBlock;
    GROUP_LOOP(
        v_share[lp].value = v_share[lp].value + sign * (convert_new_uint(Bout, v_this_level, temp, lp) + t_previous * (v + ((int)level) * groupSize + lp)->value);
        // mod(v_share[lp]);
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
    block s = k[0];
    GROUP_LOOP(v_share[lp].value = 0;)

    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx.value >> (Bin - 1 - i)) & 1;
        s = traverseOneDCF(Bin, Bout, groupSize, party, s, k[i + 1], keep, v_share, v, i, geq, evalGroupIdxStart, evalGroupIdxLen);
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
    block temp0, temp1, temp2, temp3;

    auto s = prng.get<std::array<block, 2>>();
    block si[2][2];
    block vi[2][2];

    GroupElement *v_alpha = new GroupElement[groupSize];
    for (int i = 0; i < groupSize; ++i)
    {
        v_alpha[i] = GroupElement(0, Bout);
    }

    block *k0 = new block[Bin + 1];
    block *k1 = new block[Bin + 1];
    GroupElement *v0 = new GroupElement[Bin * groupSize];    // bitsize Bout, size Bin x groupSize
    GroupElement *g0 = new GroupElement[groupSize];     // bitsize: Bout

    s[0] = (s[0] & notOneBlock) ^ ((s[1] & OneBlock) ^ OneBlock);
    k0[0] = s[0];
    k1[0] = s[1];

    for (int i = 0; i < Bin; ++i)
    {
        const u8 keep = static_cast<uint8_t>(idx.value >> (Bin - 1 - i)) & 1;
        auto a = toBlock(keep);

        auto ss0 = s[0] & notThreeBlock;
        auto ss1 = s[1] & notThreeBlock;

        si[0][0] = aes_enc(ss0, 0);// aes0.ecbEncBlock(ss0, si[0][0]);
        // aes_evals_count++;
        si[0][1] = aes_enc(ss0, 1);// aes1.ecbEncBlock(ss0, si[0][1]);
        // aes_evals_count++;
        vi[0][0] = aes_enc(ss0, 2);// aes2.ecbEncBlock(ss0, vi[0][0]);
        // aes_evals_count++;
        vi[0][1] = aes_enc(ss0, 3);// aes3.ecbEncBlock(ss0, vi[0][1]);
        // aes_evals_count++;
        si[1][0] = aes_enc(ss1, 0);// aes0.ecbEncBlock(ss1, si[1][0]);
        // aes_evals_count++;
        si[1][1] = aes_enc(ss1, 1);// aes1.ecbEncBlock(ss1, si[1][1]);
        // aes_evals_count++;
        vi[1][0] = aes_enc(ss1, 2);// aes2.ecbEncBlock(ss1, vi[1][0]);
        // aes_evals_count++;
        vi[1][1] = aes_enc(ss1, 3);// aes3.ecbEncBlock(ss1, vi[1][1]);
        // aes_evals_count++;
        si[0][0] = si[0][0] ^ ss0;
        si[0][1] = si[0][1] ^ ss0;
        si[1][0] = si[1][0] ^ ss1;
        si[1][1] = si[1][1] ^ ss1;
        vi[0][0] = vi[0][0] ^ ss0;
        vi[0][1] = vi[0][1] ^ ss0;
        vi[1][0] = vi[1][0] ^ ss1;
        vi[1][1] = vi[1][1] ^ ss1;

        auto ti0 = lsb(s[0]);
        auto ti1 = lsb(s[1]);
        GroupElement sign((ti1 == 1) ? -1 : +1, Bout);

        for (int lp = 0; lp < groupSize; ++lp)
        {
            v0[i * groupSize + lp] = sign * (-v_alpha[lp] - convert_new(Bout, vi[0][keep ^ 1], temp0, lp) + convert_new(Bout, vi[1][keep ^ 1], temp1, lp));
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
            v_alpha[lp] = v_alpha[lp] - convert_new(Bout, vi[1][keep], temp3, lp) + convert_new(Bout, vi[0][keep], temp2, lp) + sign * v0[i * groupSize + lp];
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

    for (int lp = 0; lp < groupSize; ++lp)
    {
        g0[lp] = (convert_new(Bout, s[1], temp1, lp) - convert_new(Bout, s[0], temp0, lp) - v_alpha[lp]);
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

    GROUP_LOOP(
        GroupElement final_term = convert_new(Bout, s, temp, lp);
        if (t)
            final_term.value = final_term.value + g[lp].value;
        if (party == SERVER1)
        {
            final_term.value = -final_term.value;
        } out[lp].value = out[lp].value + final_term.value;)
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
        auto payload2_split = splitShare(payload2[i]);
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
