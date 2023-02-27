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

	int sz1, sz2 ;
	sz1 = __sz1 ;
	sz2 = __sz2 ;

	vector<vector<FPArray>> inArr = make_vector_float_rand(ALICE, sz1, sz2) ;
	vector<FPArray> outArr = make_vector_float(ALICE, sz1) ;
	vectorSum2(sz1, sz2, inArr, outArr) ;
	__end() ;

	return 0;
}