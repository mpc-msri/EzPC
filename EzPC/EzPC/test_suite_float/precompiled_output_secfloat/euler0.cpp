
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

int32_t terms ;

cout << ("Input terms:") << endl ;

int32_t *__tmp_in_terms = new int32_t[1] ;

cin >> __tmp_in_terms[0];
terms = __tmp_in_terms[0] ;

delete[] __tmp_in_terms ;

FPArray euler = __public_float_to_baba(1.) ;

FPArray fact = __public_float_to_baba(1.) ;

FPArray ctr = __public_float_to_baba(1.) ;

float t = 2. ;

for (uint32_t i = 1; i < terms; i++){
euler = __fp_op->add(euler, __fp_op->div(__public_float_to_baba(t), fact)) ;

}
cout << "Value of euler : " ;

__fp_pub = __fp_op->output(PUBLIC, euler) ;

cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

return 0;
}

