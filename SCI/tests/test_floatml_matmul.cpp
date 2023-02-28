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

	int m, n, p ;
	m = __sz1 ;
	n = __sz2 ;
	p = __sz3 ;

	if (m == 0 || n == 0 || p == 0) {
		m = 100 ;
		n = 100 ;
		p = 100 ;
	}

	vector<vector<FPArray>> matA = make_vector_float_rand(ALICE, m, n) ;
	vector<vector<FPArray>> matB = make_vector_float_rand(ALICE, n, p) ;
	vector<vector<FPArray>> matC = make_vector_float_rand(ALICE, m, p) ;

	MatMul(m, n, p, matA, matB, matC) ;
	__end() ;

	return 0;
}


