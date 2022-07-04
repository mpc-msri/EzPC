
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

auto x = make_vector_int(ALICE, false, 32, 2, 3) ;

if ((__party == ALICE)) {
cout << ("Input x:") << endl ;

}
uint64_t *__tmp_in_x = new uint64_t[1] ;

for (uint32_t i0 = 0; i0 < 2; i0++){
for (uint32_t i1 = 0; i1 < 3; i1++){
if ((__party == ALICE)) {
cin >> __tmp_in_x[0];
}
x[i0][i1] = __fix_op->input(ALICE, 1, __tmp_in_x, false, 64) ;

}
}
delete[] __tmp_in_x ;

auto y = make_vector_int(ALICE, false, 32, 2, 3) ;

if ((__party == BOB)) {
cout << ("Input y:") << endl ;

}
uint64_t *__tmp_in_y = new uint64_t[1] ;

for (uint32_t i0 = 0; i0 < 2; i0++){
for (uint32_t i1 = 0; i1 < 3; i1++){
if ((__party == BOB)) {
cin >> __tmp_in_y[0];
}
y[i0][i1] = __fix_op->input(BOB, 1, __tmp_in_y, false, 64) ;

}
}
delete[] __tmp_in_y ;

FixArray acc = __public_int_to_arithmetic(0, false, 64) ;

for (uint32_t i = 0; i < 2; i++){
for (uint32_t j = 0; j < 3; j++){
acc = __fix_op->add(acc, __fix_op->sub(y[i][j], x[i][j])) ;

}
}
cout << "Value of acc : " ;

__fix_pub = __fix_op->output(PUBLIC, acc) ;

cout << (__fix_pub.get_native_type<uint64_t>()[0]) << endl ;

return 0;
}

