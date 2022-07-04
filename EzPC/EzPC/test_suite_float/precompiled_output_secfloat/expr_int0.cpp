
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

auto x = make_vector<int32_t>(2, 2) ;

cout << ("Input x:") << endl ;

int32_t *__tmp_in_x = new int32_t[1] ;

for (uint32_t i0 = 0; i0 < 2; i0++){
for (uint32_t i1 = 0; i1 < 2; i1++){
cin >> __tmp_in_x[0];
x[i0][i1] = __tmp_in_x[0] ;

}
}
delete[] __tmp_in_x ;

int32_t y ;

cout << ("Input y:") << endl ;

int32_t *__tmp_in_y = new int32_t[1] ;

cin >> __tmp_in_y[0];
y = __tmp_in_y[0] ;

delete[] __tmp_in_y ;

FixArray w(ALICE, 1, true, 32) ;

if ((__party == ALICE)) {
cout << ("Input w:") << endl ;

}
uint64_t *__tmp_in_w = new uint64_t[1] ;

if ((__party == ALICE)) {
cin >> __tmp_in_w[0];
}
w = __fix_op->input(ALICE, 1, __tmp_in_w, true, 32) ;

delete[] __tmp_in_w ;

FixArray z = __fix_op->sub(__fix_op->mul(__public_int_to_arithmetic((x[1][0] + y), true, 32), w), __public_int_to_arithmetic(100, true, 32)) ;

if ((__party == BOB)) {
cout << "Value of z : " ;

}
__fix_pub = __fix_op->output(PUBLIC, z) ;

if ((__party == BOB)) {
cout << (__fix_pub.get_native_type<int32_t>()[0]) << endl ;

}
return 0;
}

