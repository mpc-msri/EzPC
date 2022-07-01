
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

float summ = 0. ;

float ctrf = 0. ;

int32_t N = 100 ;

auto arr = make_vector<float>(N) ;

for (uint32_t i = 0; i < N; i++){
arr[i] = ctrf ;

ctrf = (ctrf + 1.) ;

}
for (uint32_t i = 0; i < N; i++){
summ = (summ + arr[i]) ;

}
cout << "Value of summ : " ;

cout << (summ) << endl ;

return 0;
}

