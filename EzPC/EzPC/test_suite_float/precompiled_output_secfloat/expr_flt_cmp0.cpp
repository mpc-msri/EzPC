
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

auto x = make_vector_float(ALICE, 3) ;

if ((__party == ALICE)) {
cout << ("Input x:") << endl ;

}
float *__tmp_in_x = new float[1] ;

for (uint32_t i0 = 0; i0 < 3; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_x[0];
}
x[i0] = __fp_op->input(ALICE, 1, __tmp_in_x) ;

}
delete[] __tmp_in_x ;

auto y = make_vector_float(ALICE, 3) ;

if ((__party == BOB)) {
cout << ("Input y:") << endl ;

}
float *__tmp_in_y = new float[1] ;

for (uint32_t i0 = 0; i0 < 3; i0++){
if ((__party == BOB)) {
cin >> __tmp_in_y[0];
}
y[i0] = __fp_op->input(BOB, 1, __tmp_in_y) ;

}
delete[] __tmp_in_y ;

auto ans = make_vector_bool(ALICE, 3) ;

for (uint32_t i = 0; i < 3; i++){
ans[i] = __fp_op->LT(x[i], y[i]) ;

}
cout << "Value of ans : " ;

for (uint32_t i0 = 0; i0 < 3; i0++){
__bool_pub = __bool_op->output(PUBLIC, ans[i0]) ;

cout << ((bool)__bool_pub.data[0]) << endl ;

}
return 0;
}

