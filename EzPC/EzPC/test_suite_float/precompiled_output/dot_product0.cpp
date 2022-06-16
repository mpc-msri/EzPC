/*
This is an autogenerated file, generated using the EzPC compiler.
*/
#include<vector>
#include<math.h>
#include<cstdlib>
#include<iostream>
#include<fstream>

using namespace std;

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
return std::vector<T>(size);
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
auto inner = make_vector<T>(sizes...);
return vector<decltype(inner)>(first, inner);
}

template<typename T>
ostream& operator<< (ostream &os, const vector<T> &v)
{
for(auto it = v.begin (); it != v.end (); ++it) {
os << *it << endl;
}
return os;
}


const uint32_t dim =  (uint32_t)2;


int main () {

auto w = make_vector<double>(dim);
cout << ("Input w:") << endl;
/* Variable to read the clear value corresponding to the input variable w at (39,1-39,32) */
double __tmp_in_w;
for (uint32_t i0 =  (uint32_t)0; i0 < dim; i0++){
cin >> __tmp_in_w;
w[i0] = __tmp_in_w;
}

double b;
cout << ("Input b:") << endl;
/* Variable to read the clear value corresponding to the input variable b at (40,1-40,27) */
double __tmp_in_b;
cin >> __tmp_in_b;
b = __tmp_in_b;

auto x = make_vector<double>(dim);
cout << ("Input x:") << endl;
/* Variable to read the clear value corresponding to the input variable x at (41,1-41,32) */
double __tmp_in_x;
for (uint32_t i0 =  (uint32_t)0; i0 < dim; i0++){
cin >> __tmp_in_x;
x[i0] = __tmp_in_x;
}

double acc =  (double)0.;

uint32_t lower =  (uint32_t)0;

uint32_t upper = dim;
for (uint32_t i = (lower - lower); i < upper; i++){
acc = (acc + (w[i] * x[i]));
}
cout << ("Value of acc >_public b ?_public 1 : 0:") << endl;
cout << ((acc > b) ?  (int32_t)1 :  (int32_t)0) << endl;
return 0;
}
