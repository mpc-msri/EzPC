
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

FPArray f(FPArray x){
return __fp_op->div(__public_float_to_baba(1.), x) ;

}


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

FPArray l(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input l:") << endl ;

}
float *__tmp_in_l = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_l[0];
}
l = __fp_op->input(BOB, 1, __tmp_in_l) ;

delete[] __tmp_in_l ;

FPArray r(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input r:") << endl ;

}
float *__tmp_in_r = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_r[0];
}
r = __fp_op->input(BOB, 1, __tmp_in_r) ;

delete[] __tmp_in_r ;

int32_t terms = 20 ;

float termsf = 20. ;

FPArray h = __fp_op->div(__fp_op->sub(r, l), __public_float_to_baba(termsf)) ;

FPArray acc = __public_float_to_baba(0.) ;

FPArray x = l ;

for (uint32_t i = 0; i < terms; i++){
acc = __fp_op->add(acc, __fp_op->mul(f(x), h)) ;

x = __fp_op->add(x, h) ;

}
if ((__party == BOB)) {
cout << "Value of acc : " ;

}
__fp_pub = __fp_op->output(PUBLIC, acc) ;

if ((__party == BOB)) {
cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

}
return 0;
}

