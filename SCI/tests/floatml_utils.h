#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"

using namespace sci ;
using namespace std ;

#define MAX_THREADS 16

/********************* Public Variables *********************/

IOPack *iopackArr[MAX_THREADS] ;
OTPack *otpackArr[MAX_THREADS] ;

BoolOp *boolopArr[MAX_THREADS] ;
FixOp *fixopArr[MAX_THREADS] ;
FPOp *fpopArr[MAX_THREADS] ;
FPMath *fpmathArr[MAX_THREADS] ;

// Communication stuff
IOPack *__iopack = nullptr ;		
OTPack *__otpack = nullptr ;
string __address = "127.0.0.1" ;
int __port = 32000 ;
int __party ;
int __nt = MAX_THREADS ;
int __iters = 10 ;
int __chunk_exp = 15 ;

// Operations
BoolOp *__bool_op = nullptr ;		// bool
FixOp *__fix_op = nullptr ;			// int
FPOp *__fp_op = nullptr ;			// float
FPMath *__fp_math = nullptr ;		// float math operations

// Floating point descriptors
int __m_bits = 23 ;				// mantissa bits
int __e_bits = 8 ;				// exponent bits

// Output operations
BoolArray __bool_pub ;				// bool
FixArray __fix_pub ;				// int
FPArray __fp_pub ;					// float


// Handy Globals
int __old = 0;
int __sz1 = 0;
int __sz2 = 0;
int __sz3 = 0 ;
int BATCH = 0 ;
int __mom = 0 ;

// Initialization
void __init(int __argc, char **__argv) {
	cout.precision(15) ;
	ArgMapping __amap ;

	__amap.arg("r", __party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2") ;
	__amap.arg("nt", __nt, "Number of threads") ;
	__amap.arg("mbits", __m_bits, "mantissa bits") ;
	__amap.arg("ebits", __e_bits, "exponent bits") ;
	__amap.arg("add", __address, "address") ;
	__amap.arg("chunk", __chunk_exp, "chunk size") ;
	__amap.arg("sz1", __sz1, "sz1") ;
	__amap.arg("sz2", __sz2, "sz2") ;
	__amap.arg("sz3", __sz3, "sz3") ;
	__amap.arg("old", __old, "old") ;
	__amap.arg("batch", BATCH, "batch") ;
	__amap.arg("mom", __mom, "momentum") ;
	__amap.parse(__argc, __argv);

	// printf("Init - %d, %d\n", __m_bits, __e_bits) ;

    for (int i = 0; i < __nt ; i++) {
    	iopackArr[i] = new IOPack(__party, __port+i, __address) ;
    	if (i & 1)
    		otpackArr[i] = new OTPack(iopackArr[i], 3-__party) ;
    	else
    		otpackArr[i] = new OTPack(iopackArr[i], __party) ;
    }

    for (int i = 0 ; i < __nt ; i++) {
    	int pt ;
    	if (i & 1)
    		pt = 3 - __party ;
    	else
    		pt = __party ;

    	boolopArr[i] = new BoolOp(pt, iopackArr[i], otpackArr[i]) ;
    	fixopArr[i] = new FixOp(pt, iopackArr[i], otpackArr[i]) ;
    	fpopArr[i] = new FPOp(pt, iopackArr[i], otpackArr[i]) ;
    	fpmathArr[i] = new FPMath(pt, iopackArr[i], otpackArr[i]) ;
    }

    __iopack = iopackArr[0] ;
    __otpack = otpackArr[0] ;

    __bool_op = boolopArr[0] ;
    __fix_op = fixopArr[0] ;
    __fp_op = fpopArr[0] ;    
    __fp_math = fpmathArr[0] ;
}

/********************* Primitive Vectors *********************/

template<typename T>
vector<T> make_vector(size_t size) {
	return std::vector<T>(size) ;
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
	auto inner = make_vector<T>(sizes...) ;
	return vector<decltype(inner)>(first, inner) ;
}

/********************* Boolean Multidimensional Arrays *********************/

BoolArray __public_bool_to_boolean(uint8_t b, int party=ALICE) {
	uint8_t *_dummy = new uint8_t[1] ;
	_dummy[0] = b ;
	BoolArray _ret = __bool_op->input(party, 1, _dummy) ;
	delete[] _dummy ;
	return _ret ;
}

vector<BoolArray> make_vector_bool(int party, size_t last) {
	vector<BoolArray> _ret ;
	for (size_t i = 0 ; i < last ; i++) {
		_ret.push_back(__public_bool_to_boolean(false, party)) ;
	}
	return _ret ;
}

template <typename... Args>
auto make_vector_bool(int party, size_t first, Args... sizes) {
	auto _inner = make_vector_bool(party, sizes...) ;
	vector<decltype(_inner)> _ret ;
	_ret.push_back(_inner) ;
	for (size_t i = 1 ; i < first ; i++) {
		_ret.push_back(make_vector_bool(party, sizes...)) ;
	}
	return _ret ;
}

/********************* Integer Multidimensional Arrays *********************/

FixArray __public_int_to_arithmetic(uint64_t i, bool sign, int len, int party=ALICE) {
	uint64_t *_dummy = new uint64_t[1] ;
	_dummy[0] = i ;
	FixArray _ret = __fix_op->input(party, 1, _dummy, sign, len, 0) ;
	delete[] _dummy ;
	return _ret ;
}

vector<FixArray> make_vector_int(int party, bool sign, int len, size_t last) {
	vector<FixArray> _ret ;
	for (size_t i = 0 ; i < last ; i++) {
		_ret.push_back(__public_int_to_arithmetic(0, sign, len, party)) ;
	}
	return _ret ;
}

template <typename... Args>
auto make_vector_int(int party, bool sign, int len, size_t first, Args... sizes) {
	auto _inner = make_vector_int(party, sign, len, sizes...) ;
	vector<decltype(_inner)> _ret ;
	_ret.push_back(_inner) ;
	for (size_t i = 1 ; i < first ; i++) {
		_ret.push_back(make_vector_int(party, sign, len, sizes...)) ;
	}
	return _ret ;
}

/********************* Floating Multidimensional Arrays *********************/

FPArray __public_float_to_arithmetic(float f, int party=ALICE) {
	float *_dummy = new float[1] ;
	_dummy[0] = f ;
	FPArray _ret = __fp_op->input<float>(party, 1, _dummy, __m_bits, __e_bits) ;
	delete[] _dummy ;
	return _ret ;
}

FPArray __rand_float(int party=ALICE) {
	float *_dummy = new float[1] ;
	_dummy[0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) ;
	FPArray _ret = __fp_op->input<float>(party, 1, _dummy, __m_bits, __e_bits) ;
	delete[] _dummy ;
	return _ret ;
}

vector<FPArray> make_vector_float(int party, size_t last) {
	vector<FPArray> _ret ;
	for (size_t i = 0 ; i < last ; i++) {
		_ret.push_back(__public_float_to_arithmetic(0.0, party)) ;
	}
	return _ret ;
}

template <typename... Args>
auto make_vector_float(int party, size_t first, Args... sizes) {
	auto _inner = make_vector_float(party, sizes...) ;
	vector<decltype(_inner)> _ret ;
	_ret.push_back(_inner) ;
	for (size_t i = 1 ; i < first ; i++) {
		_ret.push_back(make_vector_float(party, sizes...)) ;
	}
	return _ret ;
}

vector<FPArray> make_vector_float_rand(int party, size_t last) {
	vector<FPArray> _ret ;
	for (size_t i = 0 ; i < last ; i++) {
		_ret.push_back(__rand_float(party)) ;
	}
	return _ret ;
}

template <typename... Args>
auto make_vector_float_rand(int party, size_t first, Args... sizes) {
	auto _inner = make_vector_float_rand(party, sizes...) ;
	vector<decltype(_inner)> _ret ;
	_ret.push_back(_inner) ;
	for (size_t i = 1 ; i < first ; i++) {
		_ret.push_back(make_vector_float_rand(party, sizes...)) ;
	}
	return _ret ;
}
