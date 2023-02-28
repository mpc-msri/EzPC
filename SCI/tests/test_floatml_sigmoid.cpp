#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "library_float.h"

using namespace std ;
using namespace sci ;

int main (int __argc, char **__argv) {
	int m_bits, e_bits ;
	__init(__argc, __argv) ;
	m_bits = __m_bits ;
	e_bits = __e_bits ;
	int sz = __sz1 ;

	if (sz == 0)
		sz = 1000000 ;

	float* inp1_tmp = new float[1] ;
	vector<FPArray> inp1 = make_vector_float_rand(ALICE, sz) ;
	vector<FPArray> out = make_vector_float(ALICE, sz) ;

	Sigmoid(sz, inp1, out) ;
	__end() ;
	return 0 ;
}