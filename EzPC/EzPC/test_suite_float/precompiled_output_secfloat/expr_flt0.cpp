
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

auto x = make_vector<float>(2, 2) ;

cout << ("Input x:") << endl ;

float *__tmp_in_x = new float[1] ;

for (uint32_t i0 = 0; i0 < 2; i0++){
for (uint32_t i1 = 0; i1 < 2; i1++){
cin >> __tmp_in_x[0];
x[i0][i1] = __tmp_in_x[0] ;

}
}
delete[] __tmp_in_x ;

float y ;

cout << ("Input y:") << endl ;

float *__tmp_in_y = new float[1] ;

cin >> __tmp_in_y[0];
y = __tmp_in_y[0] ;

delete[] __tmp_in_y ;

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

FPArray z = __fp_op->sub(__fp_op->mul(__public_float_to_baba((x[1][0] + y)), w), __public_float_to_baba(100.)) ;

if ((__party == BOB)) {
cout << "Value of z : " ;

}
__fp_pub = __fp_op->output(PUBLIC, z) ;

if ((__party == BOB)) {
cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

}
return 0;
}

