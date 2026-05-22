
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

FixArray x1 = __public_int_to_arithmetic(1, false, 32) ;

uint32_t x2 = 2 ;

FixArray x3 = __public_int_to_arithmetic(1, true, 32) ;

int32_t x4 = 2 ;

FixArray x5 = __public_int_to_arithmetic(2, true, 32) ;

FixArray x6 = __public_int_to_arithmetic(2, true, 32) ;

BoolArray x7 = __public_bool_to_boolean(0) ;

FPArray pi = __public_float_to_baba(3.14) ;

FPArray e = __public_float_to_baba(2.71) ;

return 0;
}

