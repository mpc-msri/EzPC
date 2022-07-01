
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




int main (int __argc, char **__argv) {

int32_t terms ;

cout << ("Input terms:") << endl ;

int32_t *__tmp_in_terms = new int32_t[1] ;

cin >> __tmp_in_terms[0];
terms = __tmp_in_terms[0] ;

delete[] __tmp_in_terms ;

float euler = 1. ;

float fact = 1. ;

float ctr = 1. ;

float t = 2. ;

for (uint32_t i = 1; i < terms; i++){
euler = (euler + (t / fact)) ;

}
cout << "Value of euler : " ;

cout << (euler) << endl ;

return 0;
}

