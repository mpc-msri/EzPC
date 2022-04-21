
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "library_float.h"

using namespace std ;
using namespace sci ;

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

void forward(vector < vector < FPArray > >& layer1W, vector < FPArray >& layer1b, vector < vector < FPArray > >& layer2W, vector < FPArray >& layer2b, vector < vector < FPArray > >& layer3W, vector < FPArray >& layer3b, vector < vector < FPArray > >& layer4W, vector < FPArray >& layer4b, vector < vector < FPArray > >& layer1In, vector < vector < BoolArray > >& layer1ReluHot, vector < vector < FPArray > >& layer1Out, vector < vector < FPArray > >& layer2In, vector < vector < BoolArray > >& layer2ReluHot, vector < vector < FPArray > >& layer2Out, vector < vector < FPArray > >& layer3In, vector < vector < BoolArray > >& layer3ReluHot, vector < vector < FPArray > >& layer3Out, vector < vector < FPArray > >& layer4In, vector < vector < FPArray > >& fwdOut){
vector < vector < FPArray > > layer1WReshaped = make_vector_float(ALICE, 874, 300) ;

vector < vector < FPArray > > layer1Temp = make_vector_float(ALICE, BATCH, 300) ;

Transpose(874, 300, layer1W, layer1WReshaped);
MatMul(BATCH, 874, 300, layer1In, layer1WReshaped, layer1Temp);
GemmAdd(BATCH, 300, layer1Temp, layer1b, layer1Out);
Relu2(BATCH, 300, layer1Out, layer2In, layer1ReluHot);

printf("\tLayer 1 done\n") ;

vector < vector < FPArray > > layer2WReshaped = make_vector_float(ALICE, 300, 200) ;

vector < vector < FPArray > > layer2Temp = make_vector_float(ALICE, BATCH, 200) ;

Transpose(300, 200, layer2W, layer2WReshaped);
MatMul(BATCH, 300, 200, layer2In, layer2WReshaped, layer2Temp);
GemmAdd(BATCH, 200, layer2Temp, layer2b, layer2Out);
Relu2(BATCH, 200, layer2Out, layer3In, layer2ReluHot);

printf("\tLayer 2 done\n") ;

vector < vector < FPArray > > layer3WReshaped = make_vector_float(ALICE, 200, 100) ;

vector < vector < FPArray > > layer3Temp = make_vector_float(ALICE, BATCH, 100) ;

Transpose(200, 100, layer3W, layer3WReshaped);
MatMul(BATCH, 200, 100, layer3In, layer3WReshaped, layer3Temp);
GemmAdd(BATCH, 100, layer3Temp, layer3b, layer3Out);
Relu2(BATCH, 100, layer3Out, layer4In, layer3ReluHot);

printf("\tLayer 3 done\n") ;

vector < vector < FPArray > > layer4WReshaped = make_vector_float(ALICE, 100, 4) ;

vector < vector < FPArray > > layer4Temp = make_vector_float(ALICE, BATCH, 4) ;

Transpose(100, 4, layer4W, layer4WReshaped);
MatMul(BATCH, 100, 4, layer4In, layer4WReshaped, layer4Temp);
GemmAdd(BATCH, 4, layer4Temp, layer4b, layer4Temp);
Softmax2(BATCH, 4, layer4Temp, fwdOut);

printf("\tLayer 4 done\n") ;
}

void backward(vector < vector < FPArray > >& target, vector < vector < FPArray > >& fwdOut, vector < vector < FPArray > >& layer1W, vector < FPArray >& layer1b, vector < vector < FPArray > >& layer2W, vector < FPArray >& layer2b, vector < vector < FPArray > >& layer3W, vector < FPArray >& layer3b, vector < vector < FPArray > >& layer4W, vector < FPArray >& layer4b, vector < vector < FPArray > >& layer1In, vector < vector < BoolArray > >& layer1ReluHot, vector < vector < FPArray > >& layer1Out, vector < vector < FPArray > >& layer2In, vector < vector < BoolArray > >& layer2ReluHot, vector < vector < FPArray > >& layer2Out, vector < vector < FPArray > >& layer3In, vector < vector < BoolArray > >& layer3ReluHot, vector < vector < FPArray > >& layer3Out, vector < vector < FPArray > >& layer4In){
vector < vector < FPArray > > layer4Der = make_vector_float(ALICE, BATCH, 4) ;

vector < vector < FPArray > > layer4InReshaped = make_vector_float(ALICE, 100, BATCH) ;

vector < vector < FPArray > > layer4WDerReshaped = make_vector_float(ALICE, 100, 4) ;

vector < FPArray > layer4bDer = make_vector_float(ALICE, 4) ;

vector < vector < FPArray > > layer3ActDer = make_vector_float(ALICE, BATCH, 100) ;

getOutDer(BATCH, 4, fwdOut, target, layer4Der);
Transpose(100, BATCH, layer4In, layer4InReshaped);
MatMul(100, BATCH, 4, layer4InReshaped, layer4Der, layer4WDerReshaped);
getBiasDer(BATCH, 4, layer4Der, layer4bDer);
MatMul(BATCH, 4, 100, layer4Der, layer4W, layer3ActDer);

printf("\tLayer 4 done\n") ;

vector < vector < FPArray > > layer3Der = make_vector_float(ALICE, BATCH, 100) ;

vector < vector < FPArray > > layer3InReshaped = make_vector_float(ALICE, 200, BATCH) ;

vector < vector < FPArray > > layer3WDerReshaped = make_vector_float(ALICE, 200, 100) ;

vector < FPArray > layer3bDer = make_vector_float(ALICE, 100) ;

vector < vector < FPArray > > layer2ActDer = make_vector_float(ALICE, BATCH, 200) ;

IfElse2(BATCH, 100, layer3ActDer, layer3ReluHot, layer3Der, 0);
Transpose(200, BATCH, layer3In, layer3InReshaped);
MatMul(200, BATCH, 100, layer3InReshaped, layer3Der, layer3WDerReshaped);
getBiasDer(BATCH, 100, layer3Der, layer3bDer);
MatMul(BATCH, 100, 200, layer3Der, layer3W, layer2ActDer);

printf("\tLayer 3 done\n") ;

vector < vector < FPArray > > layer2Der = make_vector_float(ALICE, BATCH, 200) ;

vector < vector < FPArray > > layer2InReshaped = make_vector_float(ALICE, 300, BATCH) ;

vector < vector < FPArray > > layer2WDerReshaped = make_vector_float(ALICE, 300, 200) ;

vector < FPArray > layer2bDer = make_vector_float(ALICE, 200) ;

vector < vector < FPArray > > layer1ActDer = make_vector_float(ALICE, BATCH, 300) ;

IfElse2(BATCH, 200, layer2ActDer, layer2ReluHot, layer2Der, 0);
Transpose(300, BATCH, layer2In, layer2InReshaped);
MatMul(300, BATCH, 200, layer2InReshaped, layer2Der, layer2WDerReshaped);
getBiasDer(BATCH, 200, layer2Der, layer2bDer);
MatMul(BATCH, 200, 300, layer2Der, layer2W, layer1ActDer);

printf("\tLayer 2 done\n") ;

vector < vector < FPArray > > layer1Der = make_vector_float(ALICE, BATCH, 300) ;

vector < vector < FPArray > > layer1InReshaped = make_vector_float(ALICE, 874, BATCH) ;

vector < vector < FPArray > > layer1WDerReshaped = make_vector_float(ALICE, 874, 300) ;

vector < FPArray > layer1bDer = make_vector_float(ALICE, 300) ;

IfElse2(BATCH, 300, layer1ActDer, layer1ReluHot, layer1Der, 0);
Transpose(874, BATCH, layer1In, layer1InReshaped);
MatMul(874, BATCH, 300, layer1InReshaped, layer1Der, layer1WDerReshaped);
getBiasDer(BATCH, 300, layer1Der, layer1bDer);

printf("\tLayer 1 done\n") ;

vector < vector < FPArray > > layer1WDer = make_vector_float(ALICE, 300, 874) ;

vector < vector < FPArray > > layer2WDer = make_vector_float(ALICE, 200, 300) ;

vector < vector < FPArray > > layer3WDer = make_vector_float(ALICE, 100, 200) ;

vector < vector < FPArray > > layer4WDer = make_vector_float(ALICE, 4, 100) ;

Transpose(300, 874, layer1WDerReshaped, layer1WDer);
Transpose(200, 300, layer2WDerReshaped, layer2WDer);
Transpose(100, 200, layer3WDerReshaped, layer3WDer);
Transpose(4, 100, layer4WDerReshaped, layer4WDer);
updateWeights2(300, 874, 0.01, layer1W, layer1WDer);
updateWeights(300, 0.01, layer1b, layer1bDer);
updateWeights2(200, 300, 0.01, layer2W, layer2WDer);
updateWeights(200, 0.01, layer2b, layer2bDer);
updateWeights2(100, 200, 0.01, layer3W, layer3WDer);
updateWeights(100, 0.01, layer3b, layer3bDer);
updateWeights2(4, 100, 0.01, layer4W, layer4WDer);
updateWeights(4, 0.01, layer4b, layer4bDer);


printf("\tApplied weight update rule\n") ;
}


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

	vector < vector < FPArray > > inp = make_vector_float_rand(ALICE, BATCH, 874) ;

	vector < vector < FPArray > > target = make_vector_float_rand(ALICE, BATCH, 4) ;

	vector < vector < FPArray > > layer1W = make_vector_float_rand(ALICE, 300, 874) ;

	vector < FPArray > layer1b = make_vector_float_rand(ALICE, 300) ;

	vector < vector < FPArray > > layer2W = make_vector_float_rand(ALICE, 200, 300) ;

	vector < FPArray > layer2b = make_vector_float_rand(ALICE, 200) ;

	vector < vector < FPArray > > layer3W = make_vector_float_rand(ALICE, 100, 200) ;

	vector < FPArray > layer3b = make_vector_float_rand(ALICE, 100) ;

	vector < vector < FPArray > > layer4W = make_vector_float_rand(ALICE, 4, 100) ;

	vector < FPArray > layer4b = make_vector_float_rand(ALICE, 4) ;

	vector < vector < BoolArray > > layer1ReluHot = make_vector_bool(ALICE, BATCH, 300) ;

	vector < vector < FPArray > > layer1Out = make_vector_float_rand(ALICE, BATCH, 300) ;

	vector < vector < FPArray > > layer2In = make_vector_float_rand(ALICE, BATCH, 300) ;

	vector < vector < BoolArray > > layer2ReluHot = make_vector_bool(ALICE, BATCH, 200) ;

	vector < vector < FPArray > > layer2Out = make_vector_float_rand(ALICE, BATCH, 200) ;

	vector < vector < FPArray > > layer3In = make_vector_float_rand(ALICE, BATCH, 200) ;

	vector < vector < BoolArray > > layer3ReluHot = make_vector_bool(ALICE, BATCH, 100) ;

	vector < vector < FPArray > > layer3Out = make_vector_float_rand(ALICE, BATCH, 100) ;

	vector < vector < FPArray > > layer4In = make_vector_float_rand(ALICE, BATCH, 100) ;

	vector < vector < FPArray > > fwdOut = make_vector_float_rand(ALICE, BATCH, 4) ;

	vector < FPArray > loss = make_vector_float_rand(ALICE, 1) ;

	printf("Starting the forward pass\n") ;
	auto start = clock_start() ;
	uint64_t initial_rounds = __iopack->get_rounds();
	float comm_start = __get_comm() ;

	forward(layer1W, layer1b, layer2W, layer2b, layer3W, layer3b, layer4W, layer4b, inp, layer1ReluHot, layer1Out, layer2In, layer2ReluHot, layer2Out, layer3In, layer3ReluHot, layer3Out, layer4In, fwdOut);

	long long t = time_from(start);
	float comm_end = __get_comm() ;
	float fwd_time = t/1000.0 ;		
	float fwd_comm = (comm_end - comm_start)/(1<<20) ;
	int fwd_rounds = __iopack->get_rounds() - initial_rounds ;

	printf("Forward pass done\n") ;

	printf("\nComputing cross-entropy loss\n") ;
	computeCELoss(BATCH, 4, target, fwdOut, loss);
	cout << "Value of loss : " ;
	__fp_pub = __fp_op->output(PUBLIC, loss[0]) ;
	cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

	printf("\nStarting the backward pass\n") ;
	start = clock_start() ;
	initial_rounds = __iopack->get_rounds();
	comm_start = __get_comm() ;
	
	backward(target, fwdOut, layer1W, layer1b, layer2W, layer2b, layer3W, layer3b, layer4W, layer4b, inp, layer1ReluHot, layer1Out, layer2In, layer2ReluHot, layer2Out, layer3In, layer3ReluHot, layer3Out, layer4In);

	t = time_from(start);
	comm_end = __get_comm() ;
	
	printf("Backward pass done\n") ;

	float back_time = t/1000.0 ;		
	float back_comm = (comm_end - comm_start)/(1<<20) ;
	int back_rounds = __iopack->get_rounds() - initial_rounds ;

	float infer_time, infer_comm, train_time, train_comm ;
	int infer_rounds, train_rounds ;

	infer_time = fwd_time ;
	infer_comm = fwd_comm ;
	infer_rounds = fwd_rounds ;

	train_time = fwd_time + back_time ;
	train_comm = fwd_comm + back_comm ;
	train_rounds = fwd_rounds + back_rounds ;

	printf("\n\n---------------------------------\n") ;

	printf("\nInference time = %.3f\n", infer_time) ;
	printf("Inference comm = %.3f\n", infer_comm) ;
	printf("Inference rounds = %d\n", infer_rounds) ;

	printf("\nTraining time = %.3f\n", train_time) ;
	printf("Training comm = %.3f\n", train_comm) ;
	printf("Training rounds = %d\n", train_rounds) ;

return 0;
}

