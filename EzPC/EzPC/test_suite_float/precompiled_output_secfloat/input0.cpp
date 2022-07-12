
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

FPArray w(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input w:") << endl ;

}
float *__tmp_in_w = new float[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_w[0];
}
w = __fp_op->input(ALICE, 1, __tmp_in_w) ;

delete[] __tmp_in_w ;

float x ;

cout << ("Input x:") << endl ;

float *__tmp_in_x = new float[1] ;

cin >> __tmp_in_x[0];
x = __tmp_in_x[0] ;

delete[] __tmp_in_x ;

auto z = make_vector_float(ALICE, 10) ;

if ((__party == ALICE)) {
cout << ("Input z:") << endl ;

}
float *__tmp_in_z = new float[1] ;

for (uint32_t i0 = 0; i0 < 10; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_z[0];
}
z[i0] = __fp_op->input(ALICE, 1, __tmp_in_z) ;

}
delete[] __tmp_in_z ;

FPArray a(ALICE, 1) ;

if ((__party == BOB)) {
cout << ("Input a:") << endl ;

}
float *__tmp_in_a = new float[1] ;

if ((__party == BOB)) {
cin >> __tmp_in_a[0];
}
a = __fp_op->input(BOB, 1, __tmp_in_a) ;

delete[] __tmp_in_a ;

BoolArray b(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input b:") << endl ;

}
uint8_t *__tmp_in_b = new uint8_t[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_b[0];
}
b = __bool_op->input(ALICE, 1, __tmp_in_b) ;

delete[] __tmp_in_b ;

BoolArray c(ALICE, 1) ;

if ((__party == ALICE)) {
cout << ("Input c:") << endl ;

}
uint8_t *__tmp_in_c = new uint8_t[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_c[0];
}
c = __bool_op->input(ALICE, 1, __tmp_in_c) ;

delete[] __tmp_in_c ;

return 0;
}

