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

	uint64_t modulo;
	uint64_t moduloMask;
	uint64_t moduloMidPt;

	void div_floor(int64_t a, int64_t b, int64_t& quot, int64_t& rem){
		assert(b>0);
		int64_t q = a/b;
		int64_t r = a%b;
		int64_t corr = ((r!=0) && (r<0));
		quot = q-corr;
		rem = (r+b)%b;
	}

	inline int64_t getSignedVal(uint64_t x){
		assert(x < modulo);
		int64_t sx = x;
		if (x >= moduloMidPt) sx = x - modulo;
		return sx;
	}

	inline uint64_t getRingElt(int64_t x){
		return ((uint64_t)x) & moduloMask;
	}

	inline uint64_t PublicAdd(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
		return (x+y) & moduloMask;
	}

	inline uint64_t PublicSub(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
		return (x-y) & moduloMask;
	}

	inline uint64_t PublicMult(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
		return (x*y) & moduloMask; //This works because its a two-power ring
	}

	inline bool PublicGT(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx > sy);
	}

	inline bool PublicGTE(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx >= sy);
	}

	inline bool PublicLT(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx < sy);
	}

	inline bool PublicLTE(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx <= sy);
	}

	uint64_t PublicDiv(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		int64_t q,r;
		div_floor(sx,sy,q,r);
		return getRingElt(q);
	}

	uint64_t PublicMod(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		int64_t q,r;
		div_floor(sx,sy,q,r);
		return r;
	}

	inline uint64_t PublicRShiftA(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		int64_t sx = getSignedVal(x);
		int64_t ans = sx >> y;
		return getRingElt(ans);
	}

	inline uint64_t PublicRShiftL(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		return (x>>y);
	}

	inline uint64_t PublicLShift(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		return (x<<y) & moduloMask;	
	}

#elif defined(ODD_RING)

	//Assumption at some places in the following code is that 2*mod < (1<<64)
	//	which allows things like (x+y)%(1<<64).
	uint64_t modulo;
	uint64_t moduloMidPt;

	uint64_t moduloMult(uint64_t a, uint64_t b, uint64_t mod){ 
	    uint64_t res = 0;
	    a %= mod; 
	    while (b)
	    {
	        if (b & 1) res = (res + a) % mod; 
	        a = (2 * a) % mod; 
	        b >>= 1; 
	    }
	    return res;
	}

	void div_floor(int64_t a, int64_t b, int64_t& quot, int64_t& rem){
		assert(b>0);
		int64_t q = a/b;
		int64_t r = a%b;
		int64_t corr = ((r!=0) && (r<0));
		quot = q-corr;
		rem = (r+b)%b;
	}

	inline int64_t getSignedVal(uint64_t x){
		assert(x < modulo);
		int64_t sx = x;
		if (x > moduloMidPt) sx = x - modulo; //Here is the odd ring assumption
		return sx;
	}

	uint64_t getRingElt(int64_t x){
		if (x > 0) 
			return x%modulo;
		else{
			int64_t y = -x;
			int64_t temp = modulo - y;
			int64_t temp1 = temp % ((int64_t)modulo);
			uint64_t ans = (temp1 + modulo)%modulo;
			return ans;
		}
	}

	inline uint64_t PublicAdd(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
		return (x+y)%modulo;
	}

	uint64_t PublicSub(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
		uint64_t ans;
		if (x>=y) ans = (x-y)%modulo;
		else ans = ((x + modulo) - y)%modulo;
		return ans;
	}

	uint64_t PublicMult(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
	#ifdef __SIZEOF_INT128__
		__int128 ix = x;
		__int128 iy = y;
		__int128 iz = ix*iy;

		return iz % modulo;
	#else
		return moduloMult(x,y,modulo);
	#endif
	}

	bool PublicGT(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx > sy);
	}

	bool PublicGTE(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx >= sy);
	}

	bool PublicLT(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx < sy);
	}

	bool PublicLTE(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx <= sy);
	}

	uint64_t PublicDiv(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		int64_t q,r;
		div_floor(sx,sy,q,r);
		return getRingElt(q);
	}

	uint64_t PublicMod(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		int64_t q,r;
		div_floor(sx,sy,q,r);
		return r;
	}

	uint64_t PublicRShiftA(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		int64_t sx = getSignedVal(x);
		int64_t ans = sx >> y;
		return getRingElt(ans);
	}

	uint64_t PublicRShiftL(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		return (x>>y);
	}

	uint64_t PublicLShift(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		return (x<<y)%modulo;	
	}

#else

	//Assumption at some places in the following code is that 2*mod < (1<<64)
	//	which allows things like (x+y)%(1<<64).
	uint64_t modulo;
	uint64_t moduloMult(uint64_t a, uint64_t b, uint64_t mod){ 
	    uint64_t res = 0;
	    a %= mod; 
	    while (b)
	    {
	        if (b & 1) res = (res + a) % mod; 
	        a = (2 * a) % mod; 
	        b >>= 1; 
	    }
	    return res;
	}

	void div_floor(int64_t a, int64_t b, int64_t& quot, int64_t& rem){
		assert(b>0);
		int64_t q = a/b;
		int64_t r = a%b;
		int64_t corr = ((r!=0) && (r<0));
		quot = q-corr;
		rem = (r+b)%b;
	}

	int64_t getSignedVal(uint64_t x){
		assert(x < modulo);
		bool xPos;
		if (modulo&1) xPos = (x <= (modulo/2));
		else xPos = (x < (modulo/2));
		int64_t sx = x;
		if (!xPos) sx = x - modulo;
		return sx;
	}

	uint64_t getRingElt(int64_t x){
		if (x > 0) 
			return x%modulo;
		else{
			int64_t y = -x;
			int64_t temp = modulo - y;
			int64_t temp1 = temp % ((int64_t)modulo);
			uint64_t ans = (temp1 + modulo)%modulo;
			return ans;
		}
	}

	uint64_t PublicAdd(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
		return (x+y)%modulo;
	}

	uint64_t PublicSub(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
		uint64_t ans;
		if (x>=y) ans = (x-y)%modulo;
		else ans = ((x + modulo) - y)%modulo;
		return ans;
	}

	uint64_t PublicMult(uint64_t x, uint64_t y){
		assert((x < modulo) && (y < modulo));
	#ifdef __SIZEOF_INT128__
		__int128 ix = x;
		__int128 iy = y;
		__int128 iz = ix*iy;

		return iz % modulo;
	#else
		return moduloMult(x,y,modulo);
	#endif
	}

	bool PublicGT(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx > sy);
	}

	bool PublicGTE(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx >= sy);
	}

	bool PublicLT(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx < sy);
	}

	bool PublicLTE(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		return (sx <= sy);
	}

	uint64_t PublicDiv(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		int64_t q,r;
		div_floor(sx,sy,q,r);
		return getRingElt(q);
	}

	uint64_t PublicMod(uint64_t x, uint64_t y){
		int64_t sx = getSignedVal(x);
		int64_t sy = getSignedVal(y);
		int64_t q,r;
		div_floor(sx,sy,q,r);
		return r;
	}

	uint64_t PublicRShiftA(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		int64_t sx = getSignedVal(x);
		int64_t ans = sx >> y;
		return getRingElt(ans);
	}

	uint64_t PublicRShiftL(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		return (x>>y);
	}

	uint64_t PublicLShift(uint64_t x, uint64_t y){
		assert((x<modulo) && (y<modulo));
		return (x<<y)%modulo;	
	}

#endif
