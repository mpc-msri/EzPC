// Place this file in "SCI/test/"
#include "FloatingPoint/floating-point.h"

using namespace sci;
using namespace std;

IOPack *__iopack = nullptr ;
OTPack *__otpack = nullptr ;
FPOp *__fp_op = nullptr ;
FixOp *__fix_op = nullptr ;
BoolOp *__bool_op = nullptr ;
int __party ;
uint8_t __m_bits = 23 ;
uint8_t __e_bits = 8 ;
string __address = "127.0.0.1" ;
int __port = 8000 ;
FPArray __fp_pub ;
FixArray __fix_pub ;
BoolArray __bool_pub ;

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

FPArray __public_float_to_arithmetic(float f, int party=ALICE) {
	float *_dummy = new float[1] ;
	_dummy[0] = f ;
	FPArray _ret = __fp_op->input(party, 1, _dummy) ;
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
