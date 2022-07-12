
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include "secfloat.h"

using namespace std ;
using namespace sci ;


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

FPArray summ = __public_float_to_baba(0.) ;

FPArray ctrf = __public_float_to_baba(0.) ;

int32_t N = 100 ;

auto arr = make_vector_float(ALICE, N) ;

for (uint32_t i = 0; i < N; i++){
arr[i] = ctrf ;

ctrf = __fp_op->add(ctrf, __public_float_to_baba(1.)) ;

}
for (uint32_t i = 0; i < N; i++){
summ = __fp_op->add(summ, arr[i]) ;

}
if ((__party == BOB)) {
cout << "Value of summ : " ;

}
__fp_pub = __fp_op->output(PUBLIC, summ) ;

if ((__party == BOB)) {
cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

}
return 0;
}

