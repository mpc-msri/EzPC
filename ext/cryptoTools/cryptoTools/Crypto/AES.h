#pragma once
// This file and the associated implementation has been placed in the public domain, waiving all copyright. No restrictions are placed on its use.
#include <cryptoTools/Common/Defines.h>
#include <wmmintrin.h>

namespace osuCrypto {


    // An AES-NI implemenation of AES encryption. 
    class AES
    {
    public:

        // Default constructor leave the class in an invalid state
        // until setKey(...) is called.
        AES() {};
        AES(const AES&) = default;

        // Constructor to initialize the class with the given key
        AES(const block& userKey);

        // Set the key to be used for encryption.
        void setKey(const block& userKey);

        // Encrypts the plaintext block and stores the result in ciphertext
        void ecbEncBlock(const block& plaintext, block& ciphertext) const;

        // Encrypts the plaintext block and returns the result 
        block ecbEncBlock(const block& plaintext) const;

        // Encrypts blockLength starting at the plaintexts pointer and writes the result
        // to the ciphertext pointer
        void ecbEncBlocks(const block* plaintexts, u64 blockLength, block* ciphertext) const;

        void ecbEncBlocks(span<const block> plaintexts, span<block> ciphertext) const
        {
            if (plaintexts.size() != ciphertext.size())
                throw RTE_LOC;
            ecbEncBlocks(plaintexts.data(), plaintexts.size(), ciphertext.data());
        }


        // Encrypts 2 blocks pointer to by plaintexts and writes the result to ciphertext
        void ecbEncTwoBlocks(const block* plaintexts, block* ciphertext) const;

        // Encrypts 4 blocks pointer to by plaintexts and writes the result to ciphertext
        void ecbEncFourBlocks(const block* plaintexts, block* ciphertext) const;

        // Encrypts 16 blocks pointer to by plaintexts and writes the result to ciphertext
        void ecbEnc16Blocks(const block* plaintexts, block* ciphertext) const;

        // Encrypts the vector of blocks {baseIdx, baseIdx + 1, ..., baseIdx + length - 1} 
        // and writes the result to ciphertext.
        void ecbEncCounterMode(u64 baseIdx, u64 length, block* ciphertext) const;

        void ecbEncCounterMode(u64 baseIdx, span<block> ciphertext) const
        {
            ecbEncCounterMode(baseIdx, ciphertext.size(), ciphertext.data());
        }

        // Returns the current key.
        const block& getKey() const { return mRoundKey[0]; }

        // The expanded key.
        block mRoundKey[11];
    };


    // Specialization of the AES class to support encryption of N values under N different keys
    template<int N>
    class MultiKeyAES
    {
    public:
        std::array<AES, N> mAESs;

        // Default constructor leave the class in an invalid state
        // until setKey(...) is called.
        MultiKeyAES() = default;

        // Constructor to initialize the class with the given key
        MultiKeyAES(span<block> keys) { setKeys(keys); }

        // Set the N keys to be used for encryption.
        void setKeys(span<block> keys)
        {
            for (u64 i = 0; i < N; ++i)
            {
                mAESs[i].setKey(keys[i]);
            }
        }

        // Computes the encrpytion of N blocks pointed to by plaintext 
        // and stores the result at ciphertext.
        void ecbEncNBlocks(const block* plaintext, block* ciphertext) const
        {

            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_xor_si128(plaintext[i], mAESs[i].mRoundKey[0]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[1]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[2]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[3]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[4]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[5]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[6]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[7]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[8]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenc_si128(ciphertext[i], mAESs[i].mRoundKey[9]);
            for (int i = 0; i < N; ++i) ciphertext[i] = _mm_aesenclast_si128(ciphertext[i], mAESs[i].mRoundKey[10]);
        }

        // Utility to compare the keys.
        const MultiKeyAES<N>& operator=(const MultiKeyAES<N>& rhs)
        {
            for (u64 i = 0; i < N; ++i)
                for (u64 j = 0; j < 11; ++j)
                    mAESs[i].mRoundKey[j] = rhs.mAESs[i].mRoundKey[j];

            return rhs;
        }
    };

    // An AES instance with a fixed and public key.
    extern const AES mAesFixedKey;

    // A class to perform AES decryption.
    class AESDec
    {
    public:
        AESDec();
        AESDec(const block& userKey);
        void setKey(const block& userKey);
        void ecbDecBlock(const block& ciphertext, block& plaintext);
        block ecbDecBlock(const block& ciphertext);
        block mRoundKey[11];
    };

}
