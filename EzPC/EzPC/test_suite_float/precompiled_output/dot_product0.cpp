
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std ;

uint32_t public_lrshift(uint32_t x, uint32_t y){
return (x >> y);
}

int32_t public_lrshift(int32_t x, uint32_t y){
return ((int32_t)(((uint32_t)x) >> y));
}

uint64_t public_lrshift(uint64_t x, uint64_t y){
return (x >> y);
}

int64_t public_lrshift(int64_t x, uint64_t y){
return ((int64_t)(((uint64_t)x) >> y));
}

template<typename T>
vector<T> make_vector(size_t size) {
return std::vector<T>(size) ;
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
auto inner = make_vector<T>(sizes...) ;
return vector<decltype(inner)>(first, inner) ;
}



const uint32_t dim = 2;


int main (int __argc, char **__argv) {

auto w = make_vector<float>(dim) ;

cout << ("Input w:") << endl ;

float *__tmp_in_w = new float[1] ;

for (uint32_t i0 = 0; i0 < dim; i0++){
cin >> __tmp_in_w[0];
w[i0] = __tmp_in_w[0] ;

}
delete[] __tmp_in_w ;

float b ;

cout << ("Input b:") << endl ;

float *__tmp_in_b = new float[1] ;

cin >> __tmp_in_b[0];
b = __tmp_in_b[0] ;

delete[] __tmp_in_b ;

auto x = make_vector<float>(dim) ;

cout << ("Input x:") << endl ;

float *__tmp_in_x = new float[1] ;

for (uint32_t i0 = 0; i0 < dim; i0++){
cin >> __tmp_in_x[0];
x[i0] = __tmp_in_x[0] ;

}
delete[] __tmp_in_x ;

float acc = 0. ;

uint32_t lower = 0 ;

uint32_t upper = dim ;

for (uint32_t i = (lower - lower); i < upper; i++){
acc = (acc + (w[i] * x[i])) ;

}
cout << "Value of ((acc) >_public (b)) ?_public (1) : (0) : " ;

cout << ((acc > b) ? 1 : 0) << endl ;

return 0;
}

