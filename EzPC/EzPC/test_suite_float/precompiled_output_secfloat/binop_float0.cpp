
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

FPArray a = __public_float_to_baba(5.) ;

FPArray b = __public_float_to_baba(10.) ;

FPArray c(ALICE, 1) ;

BoolArray e(ALICE, 1) ;

float d ;

e = __fp_op->GT(a, b) ;

c = __fp_op->div(__public_float_to_baba(1.), a) ;

c = __fp_op->sub(a, b) ;

c = __fp_op->mul(a, b) ;

c = __fp_op->div(b, __public_float_to_baba(4.)) ;

e = __public_bool_to_boolean((1 && 0)) ;

e = __public_bool_to_boolean((0 || 0)) ;

e = __public_bool_to_boolean((0 ^ 1)) ;

e = __fp_op->GT(a, b) ;

e = __fp_op->GE(a, b) ;

e = __fp_op->LE(a, b) ;

e = __fp_op->LT(a, b) ;

return 0;
}

