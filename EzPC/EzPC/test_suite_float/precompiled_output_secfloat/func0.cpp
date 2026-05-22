
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

FPArray hello(FPArray a){
FPArray one = __public_float_to_baba(1.) ;

return __fp_op->add(a, one) ;

}


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

FPArray b = __public_float_to_baba(10.) ;

FPArray c = hello(b) ;

if ((__party == ALICE)) {
cout << "Value of c : " ;

}
__fp_pub = __fp_op->output(PUBLIC, c) ;

if ((__party == ALICE)) {
cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

}
return 0;
}

