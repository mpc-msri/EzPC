#include "FloatingPoint/floating-point.h"

using namespace sci;
using namespace std;

/********************* Public Variables *********************/

// Communication stuff
IOPack *__iopack = nullptr ;		
OTPack *__otpack = nullptr ;
string __address = "127.0.0.1" ;
int __port = 8000 ;
int __party ;

// Operations
BoolOp *__bool_op = nullptr ;		// bool
FixOp *__fix_op = nullptr ;			// int
FPOp *__fp_op = nullptr ;			// float

// Floating point descriptors
uint8_t __m_bits = 23 ;				// mantissa bits
uint8_t __e_bits = 8 ;				// exponent bits

// Output operations
BoolArray __bool_pub ;				// bool
FixArray __fix_pub ;				// int
FPArray __fp_pub ;					// float


// Initialization
void __init(int __argc, char **__argv) {
	cout.precision(15) ;
	ArgMapping __amap ;

	__amap.arg("r", __party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2") ;
	__amap.parse(__argc, __argv);

	__iopack = new IOPack(__party, __port, __address) ;
	__otpack = new OTPack(__iopack, __party) ;
	__fp_op = new FPOp(__party, __iopack, __otpack) ;
	__fix_op = new FixOp(__party, __iopack, __otpack) ;
	__bool_op = new BoolOp(__party, __iopack, __otpack) ;
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

FPArray __public_float_to_baba(float f, int party=ALICE) {
	float *_dummy = new float[1] ;
	_dummy[0] = f ;
	FPArray _ret = __fp_op->input(party, 1, _dummy) ;
	delete[] _dummy ;
	return _ret ;
}

vector<FPArray> make_vector_float(int party, size_t last) {
	vector<FPArray> _ret ;
	for (size_t i = 0 ; i < last ; i++) {
		_ret.push_back(__public_float_to_baba(0.0, party)) ;
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
