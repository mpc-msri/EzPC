#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "library_float.h"

int32_t BATCH = 128 ;

using namespace std ;
using namespace sci ;

float linear_time = 0.0 ;
float linear_comm = 0.0 ;
int linear_rounds = 0 ;

float nonlinear_time = 0.0 ;
float nonlinear_comm = 0.0 ;
int nonlinear_rounds = 0 ;

extern float intToFloat(int32_t m);
extern void Softmax2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr);
extern void Ln(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
extern void getOutDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& batchSoft, vector < vector < FPArray > >& lab, vector < vector < FPArray > >& der);
extern void MatMul(int32_t s1, int32_t s2, int32_t s3, vector < vector < FPArray > >& mat1, vector < vector < FPArray > >& mat2, vector < vector < FPArray > >& mat3);
extern void GemmAdd(int32_t s1, int32_t s2, vector < vector < FPArray > >& prod, vector < FPArray >& bias, vector < vector < FPArray > >& out);

extern void dotProduct2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < FPArray >& outArr);
extern void Relu(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr, vector < BoolArray >& hotArr);
extern void getBiasDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& der, vector < FPArray >& biasDer);
extern void IfElse(int32_t s1, vector < FPArray >& dat, vector < BoolArray >& hot, vector < FPArray >& out, bool flip);
extern void updateWeights(int32_t s, float lr, vector < FPArray >& bias, vector < FPArray >& der);
extern void getLoss(int32_t m, vector < FPArray >& lossTerms, vector < FPArray >& loss);
extern void computeMSELoss(int32_t m, int32_t s, vector < vector < FPArray > >& target, vector < vector < FPArray > >& fwdOut, vector < FPArray >& loss);
void Reassign2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
arr2[i1][i2] = arr1[i1][i2] ;

}
}
}

void Transpose(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i1][i2] = inArr[i2][i1] ;

}
}
}

void Ln2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
inFlat[((i1 * s2) + i2)] = inArr[i1][i2] ;

}
}
Ln(sz, inFlat, outFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i1][i2] = outFlat[((i1 * s2) + i2)] ;

}
}
}

void Relu2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr, vector < vector < BoolArray > >& hotArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inArrFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outArrFlat = make_vector_float(ALICE, sz) ;

vector < BoolArray > hotArrFlat = make_vector_bool(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
inArrFlat[((i1 * s2) + i2)] = inArr[i1][i2] ;

}
}
Relu(sz, inArrFlat, outArrFlat, hotArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i1][i2] = outArrFlat[((i1 * s2) + i2)] ;

hotArr[i1][i2] = hotArrFlat[((i1 * s2) + i2)] ;

}
}
}

void updateWeights2(int32_t s1, int32_t s2, float lr, vector < vector < FPArray > >& weight, vector < vector < FPArray > >& der){
int32_t sz = (s1 * s2) ;

vector < FPArray > weightFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > derFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
weightFlat[((i1 * s2) + i2)] = weight[i1][i2] ;

derFlat[((i1 * s2) + i2)] = der[i1][i2] ;

}
}
updateWeights(sz, lr, weightFlat, derFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
weight[i1][i2] = weightFlat[((i1 * s2) + i2)] ;

}
}
}

void IfElse2(int32_t s1, int32_t s2, vector < vector < FPArray > >& dat, vector < vector < BoolArray > >& hot, vector < vector < FPArray > >& out, bool flip){
int32_t sz = (s1 * s2) ;

vector < FPArray > datFlat = make_vector_float(ALICE, sz) ;

vector < BoolArray > hotFlat = make_vector_bool(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
datFlat[((i1 * s2) + i2)] = dat[i1][i2] ;

hotFlat[((i1 * s2) + i2)] = hot[i1][i2] ;

}
}
IfElse(sz, datFlat, hotFlat, outFlat, flip);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
out[i1][i2] = outFlat[((i1 * s2) + i2)] ;

}
}
}

void computeCELoss(int32_t m, int32_t s2, vector < vector < FPArray > >& labels, vector < vector < FPArray > >& batchSoft, vector < FPArray >& loss){
vector < vector < FPArray > > batchLn = make_vector_float(ALICE, m, s2) ;

vector < FPArray > lossTerms = make_vector_float(ALICE, m) ;

Ln2(m, s2, batchSoft, batchLn);
dotProduct2(m, s2, batchLn, labels, lossTerms);
getLoss(m, lossTerms, loss);
}

void forward(vector < vector < FPArray > >& layer1W, vector < FPArray >& layer1b, vector < vector < FPArray > >& layer2W, vector < FPArray >& layer2b, vector < vector < FPArray > >& layer3W, vector < FPArray >& layer3b, vector < vector < FPArray > >& layer1In, vector < vector < BoolArray > >& layer1ReluHot, vector < vector < FPArray > >& layer1Out, vector < vector < FPArray > >& layer2In, vector < vector < BoolArray > >& layer2ReluHot, vector < vector < FPArray > >& layer2Out, vector < vector < FPArray > >& layer3In, vector < vector < FPArray > >& fwdOut){
auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = __get_comm() ;

vector < vector < FPArray > > layer1WReshaped = make_vector_float(ALICE, 784, 128) ;
vector < vector < FPArray > > layer1Temp = make_vector_float(ALICE, BATCH, 128) ;

Transpose(784, 128, layer1W, layer1WReshaped);
MatMul(BATCH, 784, 128, layer1In, layer1WReshaped, layer1Temp);
GemmAdd(BATCH, 128, layer1Temp, layer1b, layer1Out);

long long t = time_from(start);
float comm_end = __get_comm()  ;

linear_time += t/1000.0 ;
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Relu2(BATCH, 128, layer1Out, layer2In, layer1ReluHot);

t = time_from(start);
comm_end = __get_comm() ;

printf("\tLayer 1 done\n") ;

nonlinear_time += t/1000.0 ;
nonlinear_comm += (comm_end - comm_start)/(1<<20) ;
nonlinear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < FPArray > > layer2WReshaped = make_vector_float(ALICE, 128, 128) ;
vector < vector < FPArray > > layer2Temp = make_vector_float(ALICE, BATCH, 128) ;

Transpose(128, 128, layer2W, layer2WReshaped);
MatMul(BATCH, 128, 128, layer2In, layer2WReshaped, layer2Temp);
GemmAdd(BATCH, 128, layer2Temp, layer2b, layer2Out);

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Relu2(BATCH, 128, layer2Out, layer3In, layer2ReluHot);

t = time_from(start);
comm_end = __get_comm() ;

printf("\tLayer 2 done\n") ;

nonlinear_time += t/1000.0 ;
nonlinear_comm += (comm_end - comm_start)/(1<<20) ;
nonlinear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < FPArray > > layer3WReshaped = make_vector_float(ALICE, 128, 10) ;
vector < vector < FPArray > > layer3Temp = make_vector_float(ALICE, BATCH, 10) ;

Transpose(128, 10, layer3W, layer3WReshaped);
MatMul(BATCH, 128, 10, layer3In, layer3WReshaped, layer3Temp);
GemmAdd(BATCH, 10, layer3Temp, layer3b, layer3Temp);

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Softmax2(BATCH, 10, layer3Temp, fwdOut);

t = time_from(start);
comm_end = __get_comm() ;

printf("\tLayer 3 done\n") ;

nonlinear_time += t/1000.0 ;
nonlinear_comm += (comm_end - comm_start)/(1<<20) ;
nonlinear_rounds += __iopack->get_rounds() - initial_rounds ;

}

void backward(vector < vector < FPArray > >& target, vector < vector < FPArray > >& fwdOut, vector < vector < FPArray > >& layer1W, vector < FPArray >& layer1b, vector < vector < FPArray > >& layer2W, vector < FPArray >& layer2b, vector < vector < FPArray > >& layer3W, vector < FPArray >& layer3b, vector < vector < FPArray > >& layer1In, vector < vector < BoolArray > >& layer1ReluHot, vector < vector < FPArray > >& layer1Out, vector < vector < FPArray > >& layer2In, vector < vector < BoolArray > >& layer2ReluHot, vector < vector < FPArray > >& layer2Out, vector < vector < FPArray > >& layer3In) {

auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = __get_comm() ;

vector < vector < FPArray > > layer3Der = make_vector_float(ALICE, BATCH, 10) ;
vector < vector < FPArray > > layer3InReshaped = make_vector_float(ALICE, 128, BATCH) ;
vector < vector < FPArray > > layer3WDerReshaped = make_vector_float(ALICE, 128, 10) ;
vector < FPArray > layer3bDer = make_vector_float(ALICE, 10) ;
vector < vector < FPArray > > layer2ActDer = make_vector_float(ALICE, BATCH, 128) ;

getOutDer(BATCH, 10, fwdOut, target, layer3Der);
Transpose(128, BATCH, layer3In, layer3InReshaped);
MatMul(128, BATCH, 10, layer3InReshaped, layer3Der, layer3WDerReshaped);
getBiasDer(BATCH, 10, layer3Der, layer3bDer);
MatMul(BATCH, 10, 128, layer3Der, layer3W, layer2ActDer);

long long t = time_from(start);
float comm_end = __get_comm() ;

printf("\tLayer 3 done\n") ;

linear_time += t/1000.0 ;
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < FPArray > > layer2Der = make_vector_float(ALICE, BATCH, 128) ;
vector < vector < FPArray > > layer2InReshaped = make_vector_float(ALICE, 128, BATCH) ;
vector < vector < FPArray > > layer2WDerReshaped = make_vector_float(ALICE, 128, 128) ;
vector < FPArray > layer2bDer = make_vector_float(ALICE, 128) ;
vector < vector < FPArray > > layer1ActDer = make_vector_float(ALICE, BATCH, 128) ;

IfElse2(BATCH, 128, layer2ActDer, layer2ReluHot, layer2Der, 0);

t = time_from(start);
comm_end = __get_comm() ;

nonlinear_time += t/1000.0 ;
nonlinear_comm += (comm_end - comm_start)/(1<<20) ;
nonlinear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Transpose(128, BATCH, layer2In, layer2InReshaped);
MatMul(128, BATCH, 128, layer2InReshaped, layer2Der, layer2WDerReshaped);
getBiasDer(BATCH, 128, layer2Der, layer2bDer);
MatMul(BATCH, 128, 128, layer2Der, layer2W, layer1ActDer);

t = time_from(start);
comm_end = __get_comm() ;

printf("\tLayer 2 done\n") ;

linear_time += t/1000.0 ;
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < FPArray > > layer1Der = make_vector_float(ALICE, BATCH, 128) ;
vector < vector < FPArray > > layer1InReshaped = make_vector_float(ALICE, 784, BATCH) ;
vector < vector < FPArray > > layer1WDerReshaped = make_vector_float(ALICE, 784, 128) ;
vector < FPArray > layer1bDer = make_vector_float(ALICE, 128) ;

IfElse2(BATCH, 128, layer1ActDer, layer1ReluHot, layer1Der, 0);

t = time_from(start);
comm_end = __get_comm() ;

nonlinear_time += t/1000.0 ;
nonlinear_comm += (comm_end - comm_start)/(1<<20) ;
nonlinear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Transpose(784, BATCH, layer1In, layer1InReshaped);
MatMul(784, BATCH, 128, layer1InReshaped, layer1Der, layer1WDerReshaped);
getBiasDer(BATCH, 128, layer1Der, layer1bDer);

t = time_from(start);
comm_end = __get_comm() ;

printf("\tLayer 2 done\n") ;

linear_time += t/1000.0 ;
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < FPArray > > layer1WDer = make_vector_float(ALICE, 128, 784) ;
vector < vector < FPArray > > layer2WDer = make_vector_float(ALICE, 128, 128) ;
vector < vector < FPArray > > layer3WDer = make_vector_float(ALICE, 10, 128) ;

Transpose(128, 784, layer1WDerReshaped, layer1WDer);
Transpose(128, 128, layer2WDerReshaped, layer2WDer);
Transpose(10, 128, layer3WDerReshaped, layer3WDer);
updateWeights2(128, 784, 0.01, layer1W, layer1WDer);
updateWeights(128, 0.01, layer1b, layer1bDer);
updateWeights2(128, 128, 0.01, layer2W, layer2WDer);
updateWeights(128, 0.01, layer2b, layer2bDer);
updateWeights2(10, 128, 0.01, layer3W, layer3WDer);
updateWeights(10, 0.01, layer3b, layer3bDer);

t = time_from(start);
comm_end = __get_comm() ;

printf("\tUpdate Weights\n") ;

linear_time += t/1000.0 ;
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

}


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

vector < vector < FPArray > > inp = make_vector_float(ALICE, BATCH, 784) ;

if ((__party == BOB)) {
cout << ("Input inp:") << endl ;

}
float *__tmp_in_inp = new float[1] ;

for (uint32_t i0 = 0; i0 < BATCH; i0++){
for (uint32_t i1 = 0; i1 < 784; i1++){
if ((__party == BOB)) {
cin >> __tmp_in_inp[0];
}
inp[i0][i1] = __fp_op->input(BOB, 1, __tmp_in_inp, __m_bits, __e_bits) ;

}
}
delete[] __tmp_in_inp ;

vector < vector < FPArray > > target = make_vector_float(ALICE, BATCH, 10) ;

if ((__party == BOB)) {
cout << ("Input target:") << endl ;

}
float *__tmp_in_target = new float[1] ;

for (uint32_t i0 = 0; i0 < BATCH; i0++){
for (uint32_t i1 = 0; i1 < 10; i1++){
if ((__party == BOB)) {
cin >> __tmp_in_target[0];
}
target[i0][i1] = __fp_op->input(BOB, 1, __tmp_in_target, __m_bits, __e_bits) ;

}
}
delete[] __tmp_in_target ;

vector < vector < FPArray > > layer1W = make_vector_float(ALICE, 128, 784) ;

if ((__party == ALICE)) {
cout << ("Input layer1W:") << endl ;

}
float *__tmp_in_layer1W = new float[1] ;

for (uint32_t i0 = 0; i0 < 128; i0++){
for (uint32_t i1 = 0; i1 < 784; i1++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer1W[0];
}
layer1W[i0][i1] = __fp_op->input(ALICE, 1, __tmp_in_layer1W, __m_bits, __e_bits) ;

}
}
delete[] __tmp_in_layer1W ;

vector < FPArray > layer1b = make_vector_float(ALICE, 128) ;

if ((__party == ALICE)) {
cout << ("Input layer1b:") << endl ;

}
float *__tmp_in_layer1b = new float[1] ;

for (uint32_t i0 = 0; i0 < 128; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer1b[0];
}
layer1b[i0] = __fp_op->input(ALICE, 1, __tmp_in_layer1b, __m_bits, __e_bits) ;

}
delete[] __tmp_in_layer1b ;

vector < vector < FPArray > > layer2W = make_vector_float(ALICE, 128, 128) ;

if ((__party == ALICE)) {
cout << ("Input layer2W:") << endl ;

}
float *__tmp_in_layer2W = new float[1] ;

for (uint32_t i0 = 0; i0 < 128; i0++){
for (uint32_t i1 = 0; i1 < 128; i1++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer2W[0];
}
layer2W[i0][i1] = __fp_op->input(ALICE, 1, __tmp_in_layer2W, __m_bits, __e_bits) ;

}
}
delete[] __tmp_in_layer2W ;

vector < FPArray > layer2b = make_vector_float(ALICE, 128) ;

if ((__party == ALICE)) {
cout << ("Input layer2b:") << endl ;

}
float *__tmp_in_layer2b = new float[1] ;

for (uint32_t i0 = 0; i0 < 128; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer2b[0];
}
layer2b[i0] = __fp_op->input(ALICE, 1, __tmp_in_layer2b, __m_bits, __e_bits) ;

}
delete[] __tmp_in_layer2b ;

vector < vector < FPArray > > layer3W = make_vector_float(ALICE, 10, 128) ;

if ((__party == ALICE)) {
cout << ("Input layer3W:") << endl ;

}
float *__tmp_in_layer3W = new float[1] ;

for (uint32_t i0 = 0; i0 < 10; i0++){
for (uint32_t i1 = 0; i1 < 128; i1++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer3W[0];
}
layer3W[i0][i1] = __fp_op->input(ALICE, 1, __tmp_in_layer3W, __m_bits, __e_bits) ;

}
}
delete[] __tmp_in_layer3W ;

vector < FPArray > layer3b = make_vector_float(ALICE, 10) ;

if ((__party == ALICE)) {
cout << ("Input layer3b:") << endl ;

}
float *__tmp_in_layer3b = new float[1] ;

for (uint32_t i0 = 0; i0 < 10; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer3b[0];
}
layer3b[i0] = __fp_op->input(ALICE, 1, __tmp_in_layer3b, __m_bits, __e_bits) ;

}
delete[] __tmp_in_layer3b ;

vector < vector < BoolArray > > layer1ReluHot = make_vector_bool(ALICE, BATCH, 128) ;

vector < vector < FPArray > > layer1Out = make_vector_float(ALICE, BATCH, 128) ;

vector < vector < FPArray > > layer2In = make_vector_float(ALICE, BATCH, 128) ;

vector < vector < BoolArray > > layer2ReluHot = make_vector_bool(ALICE, BATCH, 128) ;

vector < vector < FPArray > > layer2Out = make_vector_float(ALICE, BATCH, 128) ;

vector < vector < FPArray > > layer3In = make_vector_float(ALICE, BATCH, 128) ;

vector < vector < FPArray > > fwdOut = make_vector_float(ALICE, BATCH, 10) ;

vector < FPArray > loss = make_vector_float(ALICE, 1) ;

float fwd_time=0.0, fwd_comm=0.0 ;
int fwd_rounds=0 ;

float back_time=0.0, back_comm=0.0 ;
int back_rounds=0 ;

printf("Forward -\n") ;
forward(layer1W, layer1b, layer2W, layer2b, layer3W, layer3b, inp, layer1ReluHot, layer1Out, layer2In, layer2ReluHot, layer2Out, layer3In, fwdOut);

// computeCELoss(BATCH, 10, target, fwdOut, loss);
// cout << "Value of loss[0] : " ;
// __fp_pub = __fp_op->output(PUBLIC, loss[0]) ;
// cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

printf("\nBackward - \n") ;
backward(target, fwdOut, layer1W, layer1b, layer2W, layer2b, layer3W, layer3b, inp, layer1ReluHot, layer1Out, layer2In, layer2ReluHot, layer2Out, layer3In);

printf("\n\tLinear time = %f\n", linear_time) ;
printf("\tLinear comms = %f\n", linear_comm) ;
printf("\tLinear rounds = %d\n", linear_rounds) ;

printf("\n\tNonlinear time = %f\n", nonlinear_time) ;
printf("\tNonlinear comms = %f\n", nonlinear_comm) ;
printf("\tNonlinear rounds = %d\n", nonlinear_rounds) ;

return 0;
}

