
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



float hello(float a){
float one = 1. ;

return (a + one) ;

}


int main (int __argc, char **__argv) {

float b = 10. ;

float c = hello(b) ;

cout << "Value of c : " ;

cout << (c) << endl ;

return 0;
}

