#include <cryptoTools/Crypto/PRNG.h>
#include <cryptoTools/Common/Log.h>
#include <algorithm>
#include <cstring>

namespace osuCrypto {

    PRNG::PRNG(const block& seed, u64 bufferSize)
        :
        mBytesIdx(0),
        mBlockIdx(0)
    {
		SetSeed(seed, bufferSize);
    }

    PRNG::PRNG(PRNG && s) :
        mBuffer(std::move(s.mBuffer)),
        mAes(std::move(s.mAes)),
        mBytesIdx(s.mBytesIdx),
        mBlockIdx(s.mBlockIdx),
        mBufferByteCapacity(s.mBufferByteCapacity)
    {
        s.mBuffer.resize(0);
        s.mBytesIdx = 0;
        s.mBlockIdx = 0;
        s.mBufferByteCapacity = 0;
    }

    void PRNG::operator=(PRNG&&s) 
    {
        mBuffer = (std::move(s.mBuffer));
        mAes = (std::move(s.mAes));
        mBytesIdx = (s.mBytesIdx);
        mBlockIdx = (s.mBlockIdx);
        mBufferByteCapacity = (s.mBufferByteCapacity);

        s.mBuffer.resize(0);
        s.mBytesIdx = 0;
        s.mBlockIdx = 0;
        s.mBufferByteCapacity = 0;
    }


    void PRNG::SetSeed(const block& seed, u64 bufferSize)
    {
        mAes.setKey(seed);
        mBlockIdx = 0;

        if (mBuffer.size() == 0)
        {
            mBuffer.resize(bufferSize);
            mBufferByteCapacity = (sizeof(block) * bufferSize);
        }


        refillBuffer();
    }

    u8 PRNG::getBit() { return get<bool>(); }

    const block PRNG::getSeed() const
    {
		if(mBuffer.size())
	        return mAes.mRoundKey[0];

		throw std::runtime_error("PRNG has not been keyed " LOCATION);
    }

    void PRNG::refillBuffer()
    {
		if (mBuffer.size() == 0)
			throw std::runtime_error("PRNG has not been keyed " LOCATION);

		mAes.ecbEncCounterMode(mBlockIdx, mBuffer.size(), mBuffer.data());
		mBlockIdx += mBuffer.size();
        mBytesIdx = 0;
    }
}
