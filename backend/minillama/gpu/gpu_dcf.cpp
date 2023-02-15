#include <../dcf.h>
#include "gpu_fss_utils.h"

using namespace osuCrypto;

inline int bytesize(const int bitsize) {
    return (bitsize % 8) == 0 ? bitsize / 8 : (bitsize / 8)  + 1;
}

void cpuConvert(const int bitsize, const int groupSize, const block &b, uint64_t *out)
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

// Real Endpoints
std::pair<DCFKeyPack, DCFKeyPack> keyGenDCFHelper(int Bin, int Bout, int groupSize,
                GroupElement idx, GroupElement* payload)
{
    // idx: bitsize Bin, payload: bitsize Bout & size groupSize
    bool greaterThan = false;

    static const block notOneBlock = toBlock(~0, ~1);
    static const block notThreeBlock = toBlock(~0, ~3);
    static const block TwoBlock = toBlock(0, 2);
    static const block ThreeBlock = toBlock(0, 3);
    const static block pt[4] = {ZeroBlock, OneBlock, TwoBlock, ThreeBlock};

    auto s = getRandomAESBlockPair();
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
        cpuConvert(Bout, groupSize, vi[0][keep], vi_00_converted);
        cpuConvert(Bout, groupSize, vi[1][keep], vi_10_converted);
        cpuConvert(Bout, groupSize, vi[0][keep ^ 1], vi_01_converted);
        cpuConvert(Bout, groupSize, vi[1][keep ^ 1], vi_11_converted);

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
    cpuConvert(Bout, groupSize, s[0] & notThreeBlock, s0_converted);
    cpuConvert(Bout, groupSize, s[1] & notThreeBlock, s1_converted);

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

std::pair<DCFKeyPack, DCFKeyPack> cpuKeyGenDCF(int Bin, int Bout,
                GroupElement idx, GroupElement payload)
{
    // idx: bitsize Bin, payload: bitsize Bout
    return keyGenDCFHelper(Bin, Bout, 1, idx, &payload);
}
