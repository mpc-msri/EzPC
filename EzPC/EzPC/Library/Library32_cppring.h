/*

Authors: Nishant Kumar.

Copyright:
Copyright (c) 2020 Microsoft Research
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

#if defined(TWO_POWER_RING)

	uint32_t modulo;
	uint32_t moduloMask;
	uint32_t moduloMidPt;

	void div_floor(int32_t a, int32_t b, int32_t& quot, int32_t& rem){
		assert(b>0);
		int32_t q = a/b;
		int32_t r = a%b;
		int32_t corr = ((r!=0) && (r<0));
		quot = q-corr;
		rem = (r+b)%b;
	}

	inline int32_t getSignedVal(uint32_t x){
		assert(x < modulo);
		int32_t sx = x;
		if (x >= moduloMidPt) sx = x - modulo;
		return sx;
	}

	inline uint32_t getRingElt(int32_t x){
		return ((uint32_t)x) & moduloMask;
	}

	inline uint32_t PublicAdd(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		return (x+y) & moduloMask;
	}

	inline uint32_t PublicSub(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		return (x-y) & moduloMask;
	}

	inline uint32_t PublicMult(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		return (x*y) & moduloMask; //This works because its a two-power ring
	}

	inline bool PublicGT(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx > sy);
	}

	inline bool PublicGTE(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx >= sy);
	}

	inline bool PublicLT(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx < sy);
	}

	inline bool PublicLTE(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx <= sy);
	}

	uint32_t PublicDiv(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		int32_t q,r;
		div_floor(sx,sy,q,r);
		return getRingElt(q);
	}

	uint32_t PublicMod(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		int32_t q,r;
		div_floor(sx,sy,q,r);
		return r;
	}

	inline uint32_t PublicRShiftA(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		int32_t sx = getSignedVal(x);
		int32_t ans = sx >> y;
		return getRingElt(ans);
	}

	inline uint32_t PublicRShiftL(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		return (x>>y);
	}

	inline uint32_t PublicLShift(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		return (x<<y) & moduloMask;	
	}

#elif defined(ODD_RING)

	//Assumption at some places in the following code is that 2*mod < (1<<64)
	//	which allows things like (x+y)%(1<<64).
	uint32_t modulo;
	uint32_t moduloMidPt;

	uint32_t moduloMult(uint32_t a, uint32_t b, uint32_t mod){ 
	    uint32_t res = 0;
	    a %= mod; 
	    while (b)
	    {
	        if (b & 1) res = (res + a) % mod; 
	        a = (2 * a) % mod; 
	        b >>= 1; 
	    }
	    return res;
	}

	void div_floor(int32_t a, int32_t b, int32_t& quot, int32_t& rem){
		assert(b>0);
		int32_t q = a/b;
		int32_t r = a%b;
		int32_t corr = ((r!=0) && (r<0));
		quot = q-corr;
		rem = (r+b)%b;
	}

	inline int32_t getSignedVal(uint32_t x){
		assert(x < modulo);
		int32_t sx = x;
		if (x > moduloMidPt) sx = x - modulo; //Here is the odd ring assumption
		return sx;
	}

	uint32_t getRingElt(int32_t x){
		if (x > 0) 
			return x%modulo;
		else{
			int32_t y = -x;
			int32_t temp = modulo - y;
			int32_t temp1 = temp % ((int32_t)modulo);
			uint32_t ans = (temp1 + modulo)%modulo;
			return ans;
		}
	}

	inline uint32_t PublicAdd(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		return (x+y)%modulo;
	}

	uint32_t PublicSub(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		uint32_t ans;
		if (x>=y) ans = (x-y)%modulo;
		else ans = ((x + modulo) - y)%modulo;
		return ans;
	}

	uint32_t PublicMult(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		uint64_t ix = x;
		uint64_t iy = y;
		uint64_t iz = x*y;

		return iz % modulo;
	}

	bool PublicGT(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx > sy);
	}

	bool PublicGTE(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx >= sy);
	}

	bool PublicLT(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx < sy);
	}

	bool PublicLTE(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx <= sy);
	}

	uint32_t PublicDiv(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		int32_t q,r;
		div_floor(sx,sy,q,r);
		return getRingElt(q);
	}

	uint32_t PublicMod(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		int32_t q,r;
		div_floor(sx,sy,q,r);
		return r;
	}

	uint32_t PublicRShiftA(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		int32_t sx = getSignedVal(x);
		int32_t ans = sx >> y;
		return getRingElt(ans);
	}

	uint32_t PublicRShiftL(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		return (x>>y);
	}

	uint32_t PublicLShift(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		return (x<<y)%modulo;	
	}

#else

	//Assumption at some places in the following code is that 2*mod < (1<<64)
	//	which allows things like (x+y)%(1<<64).
	uint32_t modulo;
	uint32_t moduloMult(uint32_t a, uint32_t b, uint32_t mod){ 
	    uint32_t res = 0;
	    a %= mod; 
	    while (b)
	    {
	        if (b & 1) res = (res + a) % mod; 
	        a = (2 * a) % mod; 
	        b >>= 1; 
	    }
	    return res;
	}

	void div_floor(int32_t a, int32_t b, int32_t& quot, int32_t& rem){
		assert(b>0);
		int32_t q = a/b;
		int32_t r = a%b;
		int32_t corr = ((r!=0) && (r<0));
		quot = q-corr;
		rem = (r+b)%b;
	}

	int32_t getSignedVal(uint32_t x){
		assert(x < modulo);
		bool xPos;
		if (modulo&1) xPos = (x <= (modulo/2));
		else xPos = (x < (modulo/2));
		int32_t sx = x;
		if (!xPos) sx = x - modulo;
		return sx;
	}

	uint32_t getRingElt(int32_t x){
		if (x > 0) 
			return x%modulo;
		else{
			int32_t y = -x;
			int32_t temp = modulo - y;
			int32_t temp1 = temp % ((int32_t)modulo);
			uint32_t ans = (temp1 + modulo)%modulo;
			return ans;
		}
	}

	uint32_t PublicAdd(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		return (x+y)%modulo;
	}

	uint32_t PublicSub(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		uint32_t ans;
		if (x>=y) ans = (x-y)%modulo;
		else ans = ((x + modulo) - y)%modulo;
		return ans;
	}

	uint32_t PublicMult(uint32_t x, uint32_t y){
		assert((x < modulo) && (y < modulo));
		uint64_t ix = x;
		uint64_t iy = y;
		uint64_t iz = x*y;

		return iz % modulo;
	}

	bool PublicGT(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx > sy);
	}

	bool PublicGTE(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx >= sy);
	}

	bool PublicLT(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx < sy);
	}

	bool PublicLTE(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		return (sx <= sy);
	}

	uint32_t PublicDiv(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		int32_t q,r;
		div_floor(sx,sy,q,r);
		return getRingElt(q);
	}

	uint32_t PublicMod(uint32_t x, uint32_t y){
		int32_t sx = getSignedVal(x);
		int32_t sy = getSignedVal(y);
		int32_t q,r;
		div_floor(sx,sy,q,r);
		return r;
	}

	uint32_t PublicRShiftA(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		int32_t sx = getSignedVal(x);
		int32_t ans = sx >> y;
		return getRingElt(ans);
	}

	uint32_t PublicRShiftL(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		return (x>>y);
	}

	uint32_t PublicLShift(uint32_t x, uint32_t y){
		assert((x<modulo) && (y<modulo));
		return (x<<y)%modulo;	
	}

#endif
