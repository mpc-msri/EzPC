
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

float pi = 3.14 ;

float euler = 2.718 ;

FixArray a(ALICE, 1, true, 32) ;

if ((pi > euler)) {
a = __public_int_to_arithmetic(20, true, 32) ;

} else {
a = __public_int_to_arithmetic(30, true, 32) ;

}
if ((__party == BOB)) {
cout << "Value of a : " ;

}
__fix_pub = __fix_op->output(PUBLIC, a) ;

if ((__party == BOB)) {
cout << (__fix_pub.get_native_type<int32_t>()[0]) << endl ;

}
return 0;
}

