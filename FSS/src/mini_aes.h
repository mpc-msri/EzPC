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

#pragma once
#include <cryptoTools/Common/Defines.h>
#include <wmmintrin.h>

typedef osuCrypto::block block;

extern block mRoundKey[4][11];

inline block keyGenHelper(block key, block keyRcon)
{
    keyRcon = _mm_shuffle_epi32(keyRcon, _MM_SHUFFLE(3, 3, 3, 3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, keyRcon);
}

inline void aes_init()
{
    for(int i = 0; i < 4; i++)
    {
        block userKey = osuCrypto::toBlock(i);
        mRoundKey[i][0] = userKey;
        mRoundKey[i][1] = keyGenHelper(mRoundKey[i][0], _mm_aeskeygenassist_si128(mRoundKey[i][0], 0x01));
        mRoundKey[i][2] = keyGenHelper(mRoundKey[i][1], _mm_aeskeygenassist_si128(mRoundKey[i][1], 0x02));
        mRoundKey[i][3] = keyGenHelper(mRoundKey[i][2], _mm_aeskeygenassist_si128(mRoundKey[i][2], 0x04));
        mRoundKey[i][4] = keyGenHelper(mRoundKey[i][3], _mm_aeskeygenassist_si128(mRoundKey[i][3], 0x08));
        mRoundKey[i][5] = keyGenHelper(mRoundKey[i][4], _mm_aeskeygenassist_si128(mRoundKey[i][4], 0x10));
        mRoundKey[i][6] = keyGenHelper(mRoundKey[i][5], _mm_aeskeygenassist_si128(mRoundKey[i][5], 0x20));
        mRoundKey[i][7] = keyGenHelper(mRoundKey[i][6], _mm_aeskeygenassist_si128(mRoundKey[i][6], 0x40));
        mRoundKey[i][8] = keyGenHelper(mRoundKey[i][7], _mm_aeskeygenassist_si128(mRoundKey[i][7], 0x80));
        mRoundKey[i][9] = keyGenHelper(mRoundKey[i][8], _mm_aeskeygenassist_si128(mRoundKey[i][8], 0x1B));
        mRoundKey[i][10] = keyGenHelper(mRoundKey[i][9], _mm_aeskeygenassist_si128(mRoundKey[i][9], 0x36));
    }
}

inline block aes_enc(const block & plaintext, const int key)
{
    block cyphertext = _mm_xor_si128(plaintext, mRoundKey[key][0]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][1]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][2]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][3]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][4]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][5]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][6]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][7]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][8]);
    cyphertext = _mm_aesenc_si128(cyphertext, mRoundKey[key][9]);
    cyphertext = _mm_aesenclast_si128(cyphertext, mRoundKey[key][10]);
    return cyphertext;
}