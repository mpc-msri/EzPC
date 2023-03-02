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

	int rows, sz ;
	rows = __sz1 ;
	sz = __sz2 ;

	if (rows == 0 || sz == 0) {
		rows = 1000 ;
		sz = 100 ;
	}


	float* inp1_tmp = new float[1] ;
	vector<vector<FPArray>> inp1 = make_vector_float_rand(ALICE, rows, sz) ;
	vector<vector<FPArray>> out = make_vector_float(ALICE, rows, sz) ;
	
	Softmax2(rows, sz, inp1, out) ;
	__end() ;
	return 0;
}