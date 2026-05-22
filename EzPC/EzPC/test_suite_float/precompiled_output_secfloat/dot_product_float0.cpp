
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

const uint32_t dim = 5;


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

auto w = make_vector_float(ALICE, dim) ;

if ((__party == ALICE)) {
cout << ("Input w:") << endl ;

}
float *__tmp_in_w = new float[1] ;

for (uint32_t i0 = 0; i0 < dim; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_w[0];
}
w[i0] = __fp_op->input(ALICE, 1, __tmp_in_w) ;

}
delete[] __tmp_in_w ;

FPArray b(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input b:") << endl ;

}
float *__tmp_in_b = new float[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_b[0];
}
b = __fp_op->input(ALICE, 1, __tmp_in_b) ;

delete[] __tmp_in_b ;

auto x = make_vector_float(ALICE, dim) ;

if ((__party == BOB)) {
cout << ("Input x:") << endl ;

}
float *__tmp_in_x = new float[1] ;

for (uint32_t i0 = 0; i0 < dim; i0++){
if ((__party == BOB)) {
cin >> __tmp_in_x[0];
}
x[i0] = __fp_op->input(BOB, 1, __tmp_in_x) ;

}
delete[] __tmp_in_x ;

uint32_t lower = 0 ;

uint32_t upper = dim ;

FPArray acc = __public_float_to_baba(0.) ;

for (uint32_t i = (lower - lower); i < upper; i++){
acc = __fp_op->add(acc, __fp_op->mul(w[i], x[i])) ;

}
if ((__party == BOB)) {
cout << "Value of ((acc) >_baba (b)) ?_baba (<public ~> baba> (true)) : (<public ~> baba> (false)) : " ;

}
__bool_pub = __bool_op->output(PUBLIC, __bool_op->if_else(__fp_op->GT(acc, b), __public_bool_to_boolean(1), __public_bool_to_boolean(0))) ;

if ((__party == BOB)) {
cout << ((bool)__bool_pub.data[0]) << endl ;

}
return 0;
}

