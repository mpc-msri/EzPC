
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

auto yy = make_vector_float(ALICE, 10) ;

FPArray y = __public_float_to_baba(1.) ;

yy[0] = y ;

int32_t i = 1 ;

while ((i < 10)) {
y = __fp_op->div(y, __public_float_to_baba(2.)) ;

yy[i] = __fp_op->add(yy[(i - 1)], y) ;

i = (i + 1) ;

}

cout << "Value of yy : " ;

for (uint32_t i0 = 0; i0 < 10; i0++){
__fp_pub = __fp_op->output(PUBLIC, yy[i0]) ;

cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

}
return 0;
}

