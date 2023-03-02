
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "library_float.h"

using namespace std ;
using namespace sci ;

float linear_time=0.0, linear_comm=0.0 ;
int linear_rounds=0 ;

float non_time=0.0, non_comm=0.0 ;
int non_rounds=0 ;

extern float intToFloat(int32_t m);
extern void Softmax2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr);
extern void Ln(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
extern void Sigmoid(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
extern void Tanh(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
extern void Relu(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr, vector < BoolArray >& hotArr);
extern void ElemWiseSub(int32_t s1, vector < FPArray >& inArr1, vector < FPArray >& inArr2, vector < FPArray >& outArr);
extern void ElemWiseMul(int32_t s1, vector < FPArray >& inArr1, vector < FPArray >& inArr2, vector < FPArray >& outArr);
extern void ElemWiseDiv(int32_t s1, vector < FPArray >& inArr1, vector < FPArray >& inArr2, vector < FPArray >& outArr);
extern void IfElse(int32_t s1, vector < FPArray >& dat, vector < BoolArray >& hot, vector < FPArray >& out, bool flip);
extern void updateWeights(int32_t s, float lr, vector < FPArray >& wt, vector < FPArray >& der);
extern void updateWeightsMomentum(int32_t s, float lr, float beta, vector < FPArray >& wt, vector < FPArray >& der, vector < FPArray >& mom);
extern void MatMul(int32_t s1, int32_t s2, int32_t s3, vector < vector < FPArray > >& mat1, vector < vector < FPArray > >& mat2, vector < vector < FPArray > >& mat3);
extern void GemmAdd(int32_t s1, int32_t s2, vector < vector < FPArray > >& prod, vector < FPArray >& bias, vector < vector < FPArray > >& out);
extern void SubtractOne(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
extern void getOutDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& batchSoft, vector < vector < FPArray > >& lab, vector < vector < FPArray > >& der);
extern void getBiasDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& der, vector < FPArray >& biasDer);
extern void dotProduct2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < FPArray >& outArr);
extern void getLoss(int32_t m, vector < FPArray >& lossTerms, vector < FPArray >& loss);
extern void Conv2DGroupWrapper(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t G, vector < vector < vector < vector < FPArray > > > >& inputArr, vector < vector < vector < vector < FPArray > > > >& filterArr, vector < vector < vector < vector < FPArray > > > >& outArr);
extern void MaxPool(int32_t N, int32_t H, int32_t W, int32_t C, int32_t ksizeH, int32_t ksizeW, int32_t strideH, int32_t strideW, int32_t imgH, int32_t imgW, vector < vector < vector < vector < FPArray > > > >& inArr, vector < vector < vector < vector < BoolArray > > > >& Pool, vector < vector < vector < vector < FPArray > > > >& outArr);
extern void ConvAdd(int32_t N, int32_t H, int32_t W, int32_t C, vector < vector < vector < vector < FPArray > > > >& inArr1, vector < FPArray >& bias, vector < vector < vector < vector < FPArray > > > >& outArr);
extern void ConvDerWrapper(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t G, vector < vector < vector < vector < FPArray > > > >& inputArr, vector < vector < vector < vector < FPArray > > > >& filterArr, vector < vector < vector < vector < FPArray > > > >& outArr);
extern void ConvBiasDer(int32_t N, int32_t H, int32_t W, int32_t C, vector < vector < vector < vector < FPArray > > > >& layerDer, vector < FPArray >& layerbDer);
extern void GetPooledDer(int32_t N, int32_t inH, int32_t inW, int32_t inC, int32_t outC, int32_t outH, int32_t outW, int32_t filterH, int32_t filterW, vector < vector < vector < vector < FPArray > > > >& convW, vector < vector < vector < vector < FPArray > > > >& outDer, vector < vector < vector < vector < FPArray > > > >& inDer);
extern void PoolProp(int32_t b, int32_t outc, int32_t img2, int32_t imgp, int32_t img1, int32_t pk, int32_t ps, vector < vector < vector < vector < FPArray > > > >& PooledDer, vector < vector < vector < vector < BoolArray > > > >& Pool, vector < vector < vector < vector < FPArray > > > >& ActDer, bool flip);
void Reassign2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
arr2[i1][i2] = arr1[i1][i2] ;

}
}
}

void Reassign3(int32_t s1, int32_t s2, int32_t s3, vector < vector < vector < FPArray > > >& arr1, vector < vector < vector < FPArray > > >& arr2){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
arr2[i1][i2][i3] = arr1[i1][i2][i3] ;

}
}
}
}

void Reassign4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector < vector < vector < vector < FPArray > > > >& arr1, vector < vector < vector < vector < FPArray > > > >& arr2){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
arr2[i1][i2][i3][i4] = arr1[i1][i2][i3][i4] ;

}
}
}
}
}

void Unflatten(int32_t S1, int32_t S234, int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector < vector < FPArray > >& inArr, vector < vector < vector < vector < FPArray > > > >& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var1 = (i2 * s3) ;

int32_t __tac_var2 = (__tac_var1 * s4) ;

int32_t __tac_var3 = (i3 * s4) ;

int32_t __tac_var4 = (__tac_var2 + __tac_var3) ;

int32_t __tac_var5 = (__tac_var4 + i4) ;

outArr[i1][i2][i3][i4] = inArr[i1][__tac_var5] ;

}
}
}
}
}

void Flatten(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t S1, int32_t S234, vector < vector < vector < vector < FPArray > > > >& inArr, vector < vector < FPArray > >& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var6 = (i2 * s3) ;

int32_t __tac_var7 = (__tac_var6 * s4) ;

int32_t __tac_var8 = (i3 * s4) ;

int32_t __tac_var9 = (__tac_var7 + __tac_var8) ;

int32_t __tac_var10 = (__tac_var9 + i4) ;

outArr[i1][__tac_var10] = inArr[i1][i2][i3][i4] ;

}
}
}
}
}

void PoolExpand(int32_t N, int32_t H, int32_t W, int32_t C, int32_t k1, int32_t k2, int32_t imgH, int32_t imgW, vector < vector < vector < vector < FPArray > > > >& inArr, vector < vector < vector < vector < FPArray > > > >& outArr){
for (uint32_t i1 = 0; i1 < N; i1++){
int32_t __tac_var11 = (H * k1) ;

for (uint32_t i2 = 0; i2 < __tac_var11; i2++){
int32_t __tac_var12 = (W * k2) ;

for (uint32_t i3 = 0; i3 < __tac_var12; i3++){
for (uint32_t i4 = 0; i4 < C; i4++){
outArr[i1][i2][i3][i4] = inArr[i1][(i2 / k1)][(i3 / k2)][i4] ;

}
}
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

vector < FPArray > inArrFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outArrFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var13 = (i1 * s2) ;

int32_t __tac_var14 = (__tac_var13 + i2) ;

inArrFlat[__tac_var14] = inArr[i1][i2] ;

}
}
Ln(sz, inArrFlat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var15 = (i1 * s2) ;

int32_t __tac_var16 = (__tac_var15 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var16] ;

}
}
}

void Sigmoid2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inArrFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outArrFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var17 = (i1 * s2) ;

int32_t __tac_var18 = (__tac_var17 + i2) ;

inArrFlat[__tac_var18] = inArr[i1][i2] ;

}
}
Sigmoid(sz, inArrFlat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var19 = (i1 * s2) ;

int32_t __tac_var20 = (__tac_var19 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var20] ;

}
}
}

void Tanh2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inArrFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outArrFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var21 = (i1 * s2) ;

int32_t __tac_var22 = (__tac_var21 + i2) ;

inArrFlat[__tac_var22] = inArr[i1][i2] ;

}
}
Tanh(sz, inArrFlat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var23 = (i1 * s2) ;

int32_t __tac_var24 = (__tac_var23 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var24] ;

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
int32_t __tac_var25 = (i1 * s2) ;

int32_t __tac_var26 = (__tac_var25 + i2) ;

inArrFlat[__tac_var26] = inArr[i1][i2] ;

}
}
Relu(sz, inArrFlat, outArrFlat, hotArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var27 = (i1 * s2) ;

int32_t __tac_var28 = (__tac_var27 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var28] ;

int32_t __tac_var29 = __tac_var27 ;

int32_t __tac_var30 = __tac_var28 ;

hotArr[i1][i2] = hotArrFlat[__tac_var28] ;

}
}
}

void Relu4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector < vector < vector < vector < FPArray > > > >& inArr, vector < vector < vector < vector < FPArray > > > >& outArr, vector < vector < vector < vector < BoolArray > > > >& hotArr){
int32_t __tac_var31 = (s1 * s2) ;

int32_t __tac_var32 = (__tac_var31 * s3) ;

int32_t sz = (__tac_var32 * s4) ;

vector < FPArray > inArrFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outArrFlat = make_vector_float(ALICE, sz) ;

vector < BoolArray > hotArrFlat = make_vector_bool(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var33 = (i1 * s2) ;

int32_t __tac_var34 = (__tac_var33 * s3) ;

int32_t __tac_var35 = (__tac_var34 * s4) ;

int32_t __tac_var36 = (i2 * s3) ;

int32_t __tac_var37 = (__tac_var36 * s4) ;

int32_t __tac_var38 = (__tac_var35 + __tac_var37) ;

int32_t __tac_var39 = (i3 * s4) ;

int32_t __tac_var40 = (__tac_var38 + __tac_var39) ;

int32_t __tac_var41 = (__tac_var40 + i4) ;

inArrFlat[__tac_var41] = inArr[i1][i2][i3][i4] ;

}
}
}
}
Relu(sz, inArrFlat, outArrFlat, hotArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var42 = (i1 * s2) ;

int32_t __tac_var43 = (__tac_var42 * s3) ;

int32_t __tac_var44 = (__tac_var43 * s4) ;

int32_t __tac_var45 = (i2 * s3) ;

int32_t __tac_var46 = (__tac_var45 * s4) ;

int32_t __tac_var47 = (__tac_var44 + __tac_var46) ;

int32_t __tac_var48 = (i3 * s4) ;

int32_t __tac_var49 = (__tac_var47 + __tac_var48) ;

int32_t __tac_var50 = (__tac_var49 + i4) ;

outArr[i1][i2][i3][i4] = outArrFlat[__tac_var50] ;

int32_t __tac_var51 = __tac_var42 ;

int32_t __tac_var52 = __tac_var43 ;

int32_t __tac_var53 = __tac_var44 ;

int32_t __tac_var54 = __tac_var45 ;

int32_t __tac_var55 = __tac_var46 ;

int32_t __tac_var56 = __tac_var47 ;

int32_t __tac_var57 = __tac_var48 ;

int32_t __tac_var58 = __tac_var49 ;

int32_t __tac_var59 = __tac_var50 ;

hotArr[i1][i2][i3][i4] = hotArrFlat[__tac_var50] ;

}
}
}
}
}

void ElemWiseMul2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < vector < FPArray > >& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > arr1Flat = make_vector_float(ALICE, sz) ;

vector < FPArray > arr2Flat = make_vector_float(ALICE, sz) ;

vector < FPArray > outArrFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var60 = (i1 * s2) ;

int32_t __tac_var61 = (__tac_var60 + i2) ;

arr1Flat[__tac_var61] = arr1[i1][i2] ;

int32_t __tac_var62 = __tac_var60 ;

int32_t __tac_var63 = __tac_var61 ;

arr2Flat[__tac_var63] = arr2[i1][i2] ;

}
}
ElemWiseMul(sz, arr1Flat, arr2Flat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var64 = (i1 * s2) ;

int32_t __tac_var65 = (__tac_var64 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var65] ;

}
}
}

void ElemWiseMul3(int32_t s1, int32_t s2, int32_t s3, vector < vector < vector < FPArray > > >& arr1, vector < vector < vector < FPArray > > >& arr2, vector < vector < vector < FPArray > > >& outArr){
int32_t __tac_var66 = (s1 * s2) ;

int32_t sz = (__tac_var66 * s3) ;

vector < FPArray > arr1Flat = make_vector_float(ALICE, sz) ;

vector < FPArray > arr2Flat = make_vector_float(ALICE, sz) ;

vector < FPArray > outArrFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
int32_t __tac_var67 = (i1 * s2) ;

int32_t __tac_var68 = (__tac_var67 * s3) ;

int32_t __tac_var69 = (i2 * s3) ;

int32_t __tac_var70 = (__tac_var68 + __tac_var69) ;

int32_t __tac_var71 = (__tac_var70 + i3) ;

arr1Flat[__tac_var71] = arr1[i1][i2][i3] ;

int32_t __tac_var72 = __tac_var67 ;

int32_t __tac_var73 = __tac_var68 ;

int32_t __tac_var74 = __tac_var69 ;

int32_t __tac_var75 = __tac_var70 ;

int32_t __tac_var76 = __tac_var71 ;

arr2Flat[__tac_var76] = arr2[i1][i2][i3] ;

}
}
}
ElemWiseMul(sz, arr1Flat, arr2Flat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
int32_t __tac_var77 = (i1 * s2) ;

int32_t __tac_var78 = (__tac_var77 * s3) ;

int32_t __tac_var79 = (i2 * s3) ;

int32_t __tac_var80 = (__tac_var78 + __tac_var79) ;

int32_t __tac_var81 = (__tac_var80 + i3) ;

outArr[i1][i2][i3] = outArrFlat[__tac_var81] ;

}
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
int32_t __tac_var82 = (i1 * s2) ;

int32_t __tac_var83 = (__tac_var82 + i2) ;

datFlat[__tac_var83] = dat[i1][i2] ;

int32_t __tac_var84 = __tac_var82 ;

int32_t __tac_var85 = __tac_var83 ;

hotFlat[__tac_var85] = hot[i1][i2] ;

}
}
IfElse(sz, datFlat, hotFlat, outFlat, flip);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var86 = (i1 * s2) ;

int32_t __tac_var87 = (__tac_var86 + i2) ;

out[i1][i2] = outFlat[__tac_var87] ;

}
}
}

void IfElse4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector < vector < vector < vector < FPArray > > > >& dat, vector < vector < vector < vector < BoolArray > > > >& hot, vector < vector < vector < vector < FPArray > > > >& out, bool flip){
int32_t __tac_var88 = (s1 * s2) ;

int32_t __tac_var89 = (__tac_var88 * s3) ;

int32_t sz = (__tac_var89 * s4) ;

vector < FPArray > datFlat = make_vector_float(ALICE, sz) ;

vector < BoolArray > hotFlat = make_vector_bool(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var90 = (i1 * s2) ;

int32_t __tac_var91 = (__tac_var90 * s3) ;

int32_t __tac_var92 = (__tac_var91 * s4) ;

int32_t __tac_var93 = (i2 * s3) ;

int32_t __tac_var94 = (__tac_var93 * s4) ;

int32_t __tac_var95 = (__tac_var92 + __tac_var94) ;

int32_t __tac_var96 = (i3 * s4) ;

int32_t __tac_var97 = (__tac_var95 + __tac_var96) ;

int32_t __tac_var98 = (__tac_var97 + i4) ;

datFlat[__tac_var98] = dat[i1][i2][i3][i4] ;

int32_t __tac_var99 = __tac_var90 ;

int32_t __tac_var100 = __tac_var91 ;

int32_t __tac_var101 = __tac_var92 ;

int32_t __tac_var102 = __tac_var93 ;

int32_t __tac_var103 = __tac_var94 ;

int32_t __tac_var104 = __tac_var95 ;

int32_t __tac_var105 = __tac_var96 ;

int32_t __tac_var106 = __tac_var97 ;

int32_t __tac_var107 = __tac_var98 ;

hotFlat[__tac_var107] = hot[i1][i2][i3][i4] ;

}
}
}
}
IfElse(sz, datFlat, hotFlat, outFlat, flip);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var108 = (i1 * s2) ;

int32_t __tac_var109 = (__tac_var108 * s3) ;

int32_t __tac_var110 = (__tac_var109 * s4) ;

int32_t __tac_var111 = (i2 * s3) ;

int32_t __tac_var112 = (__tac_var111 * s4) ;

int32_t __tac_var113 = (__tac_var110 + __tac_var112) ;

int32_t __tac_var114 = (i3 * s4) ;

int32_t __tac_var115 = (__tac_var113 + __tac_var114) ;

int32_t __tac_var116 = (__tac_var115 + i4) ;

out[i1][i2][i3][i4] = outFlat[__tac_var116] ;

}
}
}
}
}

void updateWeights2(int32_t s1, int32_t s2, float lr, vector < vector < FPArray > >& wt, vector < vector < FPArray > >& der){
int32_t sz = (s1 * s2) ;

vector < FPArray > wtFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > derFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var117 = (i1 * s2) ;

int32_t __tac_var118 = (__tac_var117 + i2) ;

wtFlat[__tac_var118] = wt[i1][i2] ;

int32_t __tac_var119 = __tac_var117 ;

int32_t __tac_var120 = __tac_var118 ;

derFlat[__tac_var120] = der[i1][i2] ;

}
}
updateWeights(sz, lr, wtFlat, derFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var121 = (i1 * s2) ;

int32_t __tac_var122 = (__tac_var121 + i2) ;

wt[i1][i2] = wtFlat[__tac_var122] ;

}
}
}

void updateWeights4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, float lr, vector < vector < vector < vector < FPArray > > > >& wt, vector < vector < vector < vector < FPArray > > > >& der){
int32_t __tac_var123 = (s1 * s2) ;

int32_t __tac_var124 = (__tac_var123 * s3) ;

int32_t sz = (__tac_var124 * s4) ;

vector < FPArray > wtFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > derFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var125 = (i1 * s2) ;

int32_t __tac_var126 = (__tac_var125 * s3) ;

int32_t __tac_var127 = (__tac_var126 * s4) ;

int32_t __tac_var128 = (i2 * s3) ;

int32_t __tac_var129 = (__tac_var128 * s4) ;

int32_t __tac_var130 = (__tac_var127 + __tac_var129) ;

int32_t __tac_var131 = (i3 * s4) ;

int32_t __tac_var132 = (__tac_var130 + __tac_var131) ;

int32_t __tac_var133 = (__tac_var132 + i4) ;

wtFlat[__tac_var133] = wt[i1][i2][i3][i4] ;

int32_t __tac_var134 = __tac_var125 ;

int32_t __tac_var135 = __tac_var126 ;

int32_t __tac_var136 = __tac_var127 ;

int32_t __tac_var137 = __tac_var128 ;

int32_t __tac_var138 = __tac_var129 ;

int32_t __tac_var139 = __tac_var130 ;

int32_t __tac_var140 = __tac_var131 ;

int32_t __tac_var141 = __tac_var132 ;

int32_t __tac_var142 = __tac_var133 ;

derFlat[__tac_var142] = der[i1][i2][i3][i4] ;

}
}
}
}
updateWeights(sz, lr, wtFlat, derFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var143 = (i1 * s2) ;

int32_t __tac_var144 = (__tac_var143 * s3) ;

int32_t __tac_var145 = (__tac_var144 * s4) ;

int32_t __tac_var146 = (i2 * s3) ;

int32_t __tac_var147 = (__tac_var146 * s4) ;

int32_t __tac_var148 = (__tac_var145 + __tac_var147) ;

int32_t __tac_var149 = (i3 * s4) ;

int32_t __tac_var150 = (__tac_var148 + __tac_var149) ;

int32_t __tac_var151 = (__tac_var150 + i4) ;

wt[i1][i2][i3][i4] = wtFlat[__tac_var151] ;

}
}
}
}
}

void updateWeightsMomentum2(int32_t s1, int32_t s2, float lr, float beta, vector < vector < FPArray > >& wt, vector < vector < FPArray > >& der, vector < vector < FPArray > >& mom){
int32_t sz = (s1 * s2) ;

vector < FPArray > wtFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > derFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > momFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var152 = (i1 * s2) ;

int32_t __tac_var153 = (__tac_var152 + i2) ;

wtFlat[__tac_var153] = wt[i1][i2] ;

int32_t __tac_var154 = __tac_var152 ;

int32_t __tac_var155 = __tac_var153 ;

derFlat[__tac_var155] = der[i1][i2] ;

int32_t __tac_var156 = __tac_var152 ;

int32_t __tac_var157 = __tac_var153 ;

momFlat[__tac_var157] = mom[i1][i2] ;

}
}
updateWeightsMomentum(sz, lr, beta, wtFlat, derFlat, momFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var158 = (i1 * s2) ;

int32_t __tac_var159 = (__tac_var158 + i2) ;

wt[i1][i2] = wtFlat[__tac_var159] ;

int32_t __tac_var160 = __tac_var158 ;

int32_t __tac_var161 = __tac_var159 ;

mom[i1][i2] = momFlat[__tac_var159] ;

}
}
}

void updateWeightsMomentum4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, float lr, float beta, vector < vector < vector < vector < FPArray > > > >& wt, vector < vector < vector < vector < FPArray > > > >& der, vector < vector < vector < vector < FPArray > > > >& mom){
int32_t __tac_var162 = (s1 * s2) ;

int32_t __tac_var163 = (__tac_var162 * s3) ;

int32_t sz = (__tac_var163 * s4) ;

vector < FPArray > wtFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > derFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > momFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var164 = (i1 * s2) ;

int32_t __tac_var165 = (__tac_var164 * s3) ;

int32_t __tac_var166 = (__tac_var165 * s4) ;

int32_t __tac_var167 = (i2 * s3) ;

int32_t __tac_var168 = (__tac_var167 * s4) ;

int32_t __tac_var169 = (__tac_var166 + __tac_var168) ;

int32_t __tac_var170 = (i3 * s4) ;

int32_t __tac_var171 = (__tac_var169 + __tac_var170) ;

int32_t __tac_var172 = (__tac_var171 + i4) ;

wtFlat[__tac_var172] = wt[i1][i2][i3][i4] ;

int32_t __tac_var173 = __tac_var164 ;

int32_t __tac_var174 = __tac_var165 ;

int32_t __tac_var175 = __tac_var166 ;

int32_t __tac_var176 = __tac_var167 ;

int32_t __tac_var177 = __tac_var168 ;

int32_t __tac_var178 = __tac_var169 ;

int32_t __tac_var179 = __tac_var170 ;

int32_t __tac_var180 = __tac_var171 ;

int32_t __tac_var181 = __tac_var172 ;

derFlat[__tac_var181] = der[i1][i2][i3][i4] ;

int32_t __tac_var182 = __tac_var164 ;

int32_t __tac_var183 = __tac_var165 ;

int32_t __tac_var184 = __tac_var166 ;

int32_t __tac_var185 = __tac_var167 ;

int32_t __tac_var186 = __tac_var168 ;

int32_t __tac_var187 = __tac_var169 ;

int32_t __tac_var188 = __tac_var170 ;

int32_t __tac_var189 = __tac_var171 ;

int32_t __tac_var190 = __tac_var172 ;

momFlat[__tac_var190] = mom[i1][i2][i3][i4] ;

}
}
}
}
updateWeightsMomentum(sz, lr, beta, wtFlat, derFlat, momFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
int32_t __tac_var191 = (i1 * s2) ;

int32_t __tac_var192 = (__tac_var191 * s3) ;

int32_t __tac_var193 = (__tac_var192 * s4) ;

int32_t __tac_var194 = (i2 * s3) ;

int32_t __tac_var195 = (__tac_var194 * s4) ;

int32_t __tac_var196 = (__tac_var193 + __tac_var195) ;

int32_t __tac_var197 = (i3 * s4) ;

int32_t __tac_var198 = (__tac_var196 + __tac_var197) ;

int32_t __tac_var199 = (__tac_var198 + i4) ;

wt[i1][i2][i3][i4] = wtFlat[__tac_var199] ;

int32_t __tac_var200 = __tac_var191 ;

int32_t __tac_var201 = __tac_var192 ;

int32_t __tac_var202 = __tac_var193 ;

int32_t __tac_var203 = __tac_var194 ;

int32_t __tac_var204 = __tac_var195 ;

int32_t __tac_var205 = __tac_var196 ;

int32_t __tac_var206 = __tac_var197 ;

int32_t __tac_var207 = __tac_var198 ;

int32_t __tac_var208 = __tac_var199 ;

mom[i1][i2][i3][i4] = momFlat[__tac_var199] ;

}
}
}
}
}

void computeCELoss(int32_t m, int32_t s, vector < vector < FPArray > >& labels, vector < vector < FPArray > >& batchSoft, vector < FPArray >& loss){
vector < vector < FPArray > > batchLn = make_vector_float(ALICE, m, s) ;

vector < FPArray > lossTerms = make_vector_float(ALICE, m) ;

Ln2(m, s, batchSoft, batchLn);
dotProduct2(m, s, batchLn, labels, lossTerms);
getLoss(m, lossTerms, loss);
}

int32_t BATCH = 4;

void forward(vector < vector < vector < vector < FPArray > > > >& layer1W, vector < FPArray >& layer1b, vector < vector < vector < vector < FPArray > > > >& layer2W, vector < FPArray >& layer2b, vector < vector < vector < vector < FPArray > > > >& layer3W, vector < FPArray >& layer3b, vector < vector < FPArray > >& layer4W, vector < FPArray >& layer4b, vector < vector < vector < vector < FPArray > > > >& layer1In, vector < vector < vector < vector < BoolArray > > > >& layer1Hot, vector < vector < vector < vector < FPArray > > > >& layer1Out, vector < vector < vector < vector < BoolArray > > > >& layer1Pool, vector < vector < vector < vector < FPArray > > > >& layer2In, vector < vector < vector < vector < BoolArray > > > >& layer2Hot, vector < vector < vector < vector < FPArray > > > >& layer2Out, vector < vector < vector < vector < BoolArray > > > >& layer2Pool, vector < vector < vector < vector < FPArray > > > >& layer3In, vector < vector < vector < vector < BoolArray > > > >& layer3Hot, vector < vector < vector < vector < FPArray > > > >& layer3Out, vector < vector < vector < vector < BoolArray > > > >& layer3Pool, vector < vector < FPArray > >& layer4In, vector < vector < FPArray > >& fwdOut){

auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = __get_comm() ;

vector < vector < vector < vector < FPArray > > > > layer1Tmp = make_vector_float(ALICE, 4, 30, 30, 64) ;

Conv2DGroupWrapper(4, 32, 32, 3, 5, 5, 64, 1, 1, 1, 1, 1, 1, 1, layer1In, layer1W, layer1Tmp);
ConvAdd(4, 30, 30, 64, layer1Tmp, layer1b, layer1Tmp);

long long t = time_from(start);
float comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Relu4(4, 30, 30, 64, layer1Tmp, layer1Out, layer1Hot);
MaxPool(4, 30, 30, 64, 3, 3, 2, 2, 14, 14, layer1Out, layer1Pool, layer2In);

t = time_from(start);
comm_end = __get_comm() ;

non_time += t/1000.0 ;		
non_comm += (comm_end - comm_start)/(1<<20) ;
non_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < vector < vector < FPArray > > > > layer2Tmp = make_vector_float(ALICE, 4, 12, 12, 64) ;

Conv2DGroupWrapper(4, 14, 14, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, 1, layer2In, layer2W, layer2Tmp);
ConvAdd(4, 12, 12, 64, layer2Tmp, layer2b, layer2Tmp);

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Relu4(4, 12, 12, 64, layer2Tmp, layer2Out, layer2Hot);
MaxPool(4, 12, 12, 64, 3, 3, 2, 2, 5, 5, layer2Out, layer2Pool, layer3In);

t = time_from(start);
comm_end = __get_comm() ;

non_time += t/1000.0 ;		
non_comm += (comm_end - comm_start)/(1<<20) ;
non_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < vector < vector < FPArray > > > > layer3Tmp = make_vector_float(ALICE, 4, 3, 3, 64) ;

Conv2DGroupWrapper(4, 5, 5, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, 1, layer3In, layer3W, layer3Tmp);
ConvAdd(4, 3, 3, 64, layer3Tmp, layer3b, layer3Tmp);

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Relu4(4, 3, 3, 64, layer3Tmp, layer3Out, layer3Hot);
vector < vector < vector < vector < FPArray > > > > layerLastPool = make_vector_float(ALICE, 4, 1, 1, 64) ;

MaxPool(4, 3, 3, 64, 3, 3, 2, 2, 1, 1, layer3Out, layer3Pool, layerLastPool);
Flatten(4, 1, 1, 64, 4, 64, layerLastPool, layer4In);

t = time_from(start);
comm_end = __get_comm() ;

non_time += t/1000.0 ;		
non_comm += (comm_end - comm_start)/(1<<20) ;
non_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < FPArray > > layer4WReshaped = make_vector_float(ALICE, 64, 10) ;

vector < vector < FPArray > > layer4Temp = make_vector_float(ALICE, 4, 10) ;

Transpose(64, 10, layer4W, layer4WReshaped);
MatMul(4, 64, 10, layer4In, layer4WReshaped, layer4Temp);
GemmAdd(4, 10, layer4Temp, layer4b, layer4Temp);

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

Softmax2(4, 10, layer4Temp, fwdOut);

t = time_from(start);
comm_end = __get_comm() ;

non_time += t/1000.0 ;		
non_comm += (comm_end - comm_start)/(1<<20) ;
non_rounds += __iopack->get_rounds() - initial_rounds ;

}

void backward(vector < vector < vector < vector < FPArray > > > >& layer1W, vector < FPArray >& layer1b, vector < vector < vector < vector < FPArray > > > >& layer2W, vector < FPArray >& layer2b, vector < vector < vector < vector < FPArray > > > >& layer3W, vector < FPArray >& layer3b, vector < vector < FPArray > >& layer4W, vector < FPArray >& layer4b, vector < vector < vector < vector < FPArray > > > >& layer1In, vector < vector < vector < vector < BoolArray > > > >& layer1Hot, vector < vector < vector < vector < FPArray > > > >& layer1Out, vector < vector < vector < vector < BoolArray > > > >& layer1Pool, vector < vector < vector < vector < FPArray > > > >& layer2In, vector < vector < vector < vector < BoolArray > > > >& layer2Hot, vector < vector < vector < vector < FPArray > > > >& layer2Out, vector < vector < vector < vector < BoolArray > > > >& layer2Pool, vector < vector < vector < vector < FPArray > > > >& layer3In, vector < vector < vector < vector < BoolArray > > > >& layer3Hot, vector < vector < vector < vector < FPArray > > > >& layer3Out, vector < vector < vector < vector < BoolArray > > > >& layer3Pool, vector < vector < FPArray > >& layer4In, vector < vector < FPArray > >& fwdOut, vector < vector < vector < vector < FPArray > > > >& layer1WMom, vector < FPArray >& layer1bMom, vector < vector < vector < vector < FPArray > > > >& layer2WMom, vector < FPArray >& layer2bMom, vector < vector < vector < vector < FPArray > > > >& layer3WMom, vector < FPArray >& layer3bMom, vector < vector < FPArray > >& layer4WMom, vector < FPArray >& layer4bMom, vector < vector < FPArray > >& target){
vector < vector < FPArray > > layer4Der = make_vector_float(ALICE, 4, 10) ;

vector < vector < FPArray > > layer4InReshaped = make_vector_float(ALICE, 64, 4) ;

vector < vector < FPArray > > layer4WDerReshaped = make_vector_float(ALICE, 64, 10) ;

vector < FPArray > layer4bDer = make_vector_float(ALICE, 10) ;

vector < vector < FPArray > > flatActDer = make_vector_float(ALICE, 4, 64) ;

vector < vector < vector < vector < FPArray > > > > layer3PooledDer = make_vector_float(ALICE, 4, 1, 1, 64) ;

auto start = clock_start() ;
uint64_t initial_rounds = __iopack->get_rounds();
float comm_start = __get_comm() ;

getOutDer(4, 10, fwdOut, target, layer4Der);
Transpose(64, 4, layer4In, layer4InReshaped);
MatMul(64, 4, 10, layer4InReshaped, layer4Der, layer4WDerReshaped);
getBiasDer(4, 10, layer4Der, layer4bDer);
MatMul(4, 10, 64, layer4Der, layer4W, flatActDer);

long long t = time_from(start);
float comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < vector < vector < FPArray > > > > layer3ExpandedPooledDer = make_vector_float(ALICE, 4, 3, 3, 64) ;

vector < vector < vector < vector < FPArray > > > > layer3ActDer = make_vector_float(ALICE, 4, 3, 3, 64) ;

vector < vector < vector < vector < FPArray > > > > layer3Der = make_vector_float(ALICE, 4, 3, 3, 64) ;

vector < vector < vector < vector < FPArray > > > > layer3WDer = make_vector_float(ALICE, 5, 5, 64, 64) ;

vector < FPArray > layer3bDer = make_vector_float(ALICE, 64) ;

vector < vector < vector < vector < FPArray > > > > layer2PooledDer = make_vector_float(ALICE, 4, 5, 5, 64) ;

Unflatten(4, 64, 4, 1, 1, 64, flatActDer, layer3PooledDer);
PoolProp(4, 64, 1, 3, 3, 3, 2, layer3PooledDer, layer3Pool, layer3ActDer, 0);
IfElse4(4, 3, 3, 64, layer3ActDer, layer3Hot, layer3Der, 1);

t = time_from(start);
comm_end = __get_comm() ;

non_time += t/1000.0 ;		
non_comm += (comm_end - comm_start)/(1<<20) ;
non_rounds += __iopack->get_rounds() - initial_rounds ;

cout << "\tLayer 3 done\n" ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

ConvDerWrapper(4, 5, 5, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, 1, layer3In, layer3WDer, layer3Der);
ConvBiasDer(4, 3, 3, 64, layer3Der, layer3bDer);
GetPooledDer(4, 5, 5, 64, 64, 3, 3, 5, 5, layer3W, layer3Der, layer2PooledDer);

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < vector < vector < FPArray > > > > layer2ExpandedPooledDer = make_vector_float(ALICE, 4, 12, 12, 64) ;

vector < vector < vector < vector < FPArray > > > > layer2ActDer = make_vector_float(ALICE, 4, 12, 12, 64) ;

vector < vector < vector < vector < FPArray > > > > layer2Der = make_vector_float(ALICE, 4, 12, 12, 64) ;

vector < vector < vector < vector < FPArray > > > > layer2WDer = make_vector_float(ALICE, 5, 5, 64, 64) ;

vector < FPArray > layer2bDer = make_vector_float(ALICE, 64) ;

vector < vector < vector < vector < FPArray > > > > layer1PooledDer = make_vector_float(ALICE, 4, 14, 14, 64) ;

PoolProp(4, 64, 5, 15, 12, 3, 2, layer2PooledDer, layer2Pool, layer2ActDer, 0);
IfElse4(4, 12, 12, 64, layer2ActDer, layer2Hot, layer2Der, 1);

t = time_from(start);
comm_end = __get_comm() ;

non_time += t/1000.0 ;		
non_comm += (comm_end - comm_start)/(1<<20) ;
non_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

cout << "\tLayer 2 done\n" ;

ConvDerWrapper(4, 14, 14, 64, 5, 5, 64, 1, 1, 1, 1, 1, 1, 1, layer2In, layer2WDer, layer2Der);
ConvBiasDer(4, 12, 12, 64, layer2Der, layer2bDer);
GetPooledDer(4, 14, 14, 64, 64, 12, 12, 5, 5, layer2W, layer2Der, layer1PooledDer);

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

vector < vector < vector < vector < FPArray > > > > layer1ExpandedPooledDer = make_vector_float(ALICE, 4, 30, 30, 64) ;

vector < vector < vector < vector < FPArray > > > > layer1ActDer = make_vector_float(ALICE, 4, 30, 30, 64) ;

vector < vector < vector < vector < FPArray > > > > layer1Der = make_vector_float(ALICE, 4, 30, 30, 64) ;

vector < vector < vector < vector < FPArray > > > > layer1WDer = make_vector_float(ALICE, 5, 5, 3, 64) ;

vector < FPArray > layer1bDer = make_vector_float(ALICE, 64) ;

PoolProp(4, 64, 14, 42, 30, 3, 2, layer1PooledDer, layer1Pool, layer1ActDer, 0);
IfElse4(4, 30, 30, 64, layer1ActDer, layer1Hot, layer1Der, 1);

t = time_from(start);
comm_end = __get_comm() ;

non_time += t/1000.0 ;		
non_comm += (comm_end - comm_start)/(1<<20) ;
non_rounds += __iopack->get_rounds() - initial_rounds ;

cout << "\tLayer 1 done\n" ;

start = clock_start() ;
initial_rounds = __iopack->get_rounds();
comm_start = __get_comm() ;

ConvDerWrapper(4, 32, 32, 3, 5, 5, 64, 1, 1, 1, 1, 1, 1, 1, layer1In, layer1WDer, layer1Der);
ConvBiasDer(4, 30, 30, 64, layer1Der, layer1bDer);
vector < vector < FPArray > > layer4WDer = make_vector_float(ALICE, 10, 64) ;

Transpose(10, 64, layer4WDerReshaped, layer4WDer);
updateWeightsMomentum4(5, 5, 3, 64, 0.01, 0.9, layer1W, layer1WDer, layer1WMom);
updateWeightsMomentum(64, 0.01, 0.9, layer1b, layer1bDer, layer1bMom);
updateWeightsMomentum4(5, 5, 64, 64, 0.01, 0.9, layer2W, layer2WDer, layer2WMom);
updateWeightsMomentum(64, 0.01, 0.9, layer2b, layer2bDer, layer2bMom);
updateWeightsMomentum4(5, 5, 64, 64, 0.01, 0.9, layer3W, layer3WDer, layer3WMom);
updateWeightsMomentum(64, 0.01, 0.9, layer3b, layer3bDer, layer3bMom);
updateWeightsMomentum2(10, 64, 0.01, 0.9, layer4W, layer4WDer, layer4WMom);
updateWeightsMomentum(10, 0.01, 0.9, layer4b, layer4bDer, layer4bMom);

cout << "\tWeight update done\n" ;

t = time_from(start);
comm_end = __get_comm() ;

linear_time += t/1000.0 ;		
linear_comm += (comm_end - comm_start)/(1<<20) ;
linear_rounds += __iopack->get_rounds() - initial_rounds ;
}


int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

vector < vector < vector < vector < FPArray > > > > inp = make_vector_float(ALICE, 4, 32, 32, 3) ;

if ((__party == BOB)) {
cout << ("Input inp:") << endl ;

}
float *__tmp_in_inp = new float[1] ;

for (uint32_t i0 = 0; i0 < 4; i0++){
for (uint32_t i1 = 0; i1 < 32; i1++){
for (uint32_t i2 = 0; i2 < 32; i2++){
for (uint32_t i3 = 0; i3 < 3; i3++){
if ((__party == BOB)) {
cin >> __tmp_in_inp[0];
}
inp[i0][i1][i2][i3] = __fp_op->input(BOB, 1, __tmp_in_inp, __m_bits, __e_bits) ;

}
}
}
}
delete[] __tmp_in_inp ;

vector < vector < FPArray > > target = make_vector_float(ALICE, 4, 10) ;

if ((__party == BOB)) {
cout << ("Input target:") << endl ;

}
float *__tmp_in_target = new float[1] ;

for (uint32_t i0 = 0; i0 < 4; i0++){
for (uint32_t i1 = 0; i1 < 10; i1++){
if ((__party == BOB)) {
cin >> __tmp_in_target[0];
}
target[i0][i1] = __fp_op->input(BOB, 1, __tmp_in_target, __m_bits, __e_bits) ;

}
}
delete[] __tmp_in_target ;

vector < vector < vector < vector < FPArray > > > > layer1W = make_vector_float(ALICE, 5, 5, 3, 64) ;

if ((__party == ALICE)) {
cout << ("Input layer1W:") << endl ;

}
float *__tmp_in_layer1W = new float[1] ;

for (uint32_t i0 = 0; i0 < 5; i0++){
for (uint32_t i1 = 0; i1 < 5; i1++){
for (uint32_t i2 = 0; i2 < 3; i2++){
for (uint32_t i3 = 0; i3 < 64; i3++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer1W[0];
}
layer1W[i0][i1][i2][i3] = __fp_op->input(ALICE, 1, __tmp_in_layer1W, __m_bits, __e_bits) ;

}
}
}
}
delete[] __tmp_in_layer1W ;

vector < FPArray > layer1b = make_vector_float(ALICE, 64) ;

if ((__party == ALICE)) {
cout << ("Input layer1b:") << endl ;

}
float *__tmp_in_layer1b = new float[1] ;

for (uint32_t i0 = 0; i0 < 64; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer1b[0];
}
layer1b[i0] = __fp_op->input(ALICE, 1, __tmp_in_layer1b, __m_bits, __e_bits) ;

}
delete[] __tmp_in_layer1b ;

vector < vector < vector < vector < FPArray > > > > layer2W = make_vector_float(ALICE, 5, 5, 64, 64) ;

if ((__party == ALICE)) {
cout << ("Input layer2W:") << endl ;

}
float *__tmp_in_layer2W = new float[1] ;

for (uint32_t i0 = 0; i0 < 5; i0++){
for (uint32_t i1 = 0; i1 < 5; i1++){
for (uint32_t i2 = 0; i2 < 64; i2++){
for (uint32_t i3 = 0; i3 < 64; i3++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer2W[0];
}
layer2W[i0][i1][i2][i3] = __fp_op->input(ALICE, 1, __tmp_in_layer2W, __m_bits, __e_bits) ;

}
}
}
}
delete[] __tmp_in_layer2W ;

vector < FPArray > layer2b = make_vector_float(ALICE, 64) ;

if ((__party == ALICE)) {
cout << ("Input layer2b:") << endl ;

}
float *__tmp_in_layer2b = new float[1] ;

for (uint32_t i0 = 0; i0 < 64; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer2b[0];
}
layer2b[i0] = __fp_op->input(ALICE, 1, __tmp_in_layer2b, __m_bits, __e_bits) ;

}
delete[] __tmp_in_layer2b ;

vector < vector < vector < vector < FPArray > > > > layer3W = make_vector_float(ALICE, 5, 5, 64, 64) ;

if ((__party == ALICE)) {
cout << ("Input layer3W:") << endl ;

}
float *__tmp_in_layer3W = new float[1] ;

for (uint32_t i0 = 0; i0 < 5; i0++){
for (uint32_t i1 = 0; i1 < 5; i1++){
for (uint32_t i2 = 0; i2 < 64; i2++){
for (uint32_t i3 = 0; i3 < 64; i3++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer3W[0];
}
layer3W[i0][i1][i2][i3] = __fp_op->input(ALICE, 1, __tmp_in_layer3W, __m_bits, __e_bits) ;

}
}
}
}
delete[] __tmp_in_layer3W ;

vector < FPArray > layer3b = make_vector_float(ALICE, 64) ;

if ((__party == ALICE)) {
cout << ("Input layer3b:") << endl ;

}
float *__tmp_in_layer3b = new float[1] ;

for (uint32_t i0 = 0; i0 < 64; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer3b[0];
}
layer3b[i0] = __fp_op->input(ALICE, 1, __tmp_in_layer3b, __m_bits, __e_bits) ;

}
delete[] __tmp_in_layer3b ;

vector < vector < FPArray > > layer4W = make_vector_float(ALICE, 10, 64) ;

if ((__party == ALICE)) {
cout << ("Input layer4W:") << endl ;

}
float *__tmp_in_layer4W = new float[1] ;

for (uint32_t i0 = 0; i0 < 10; i0++){
for (uint32_t i1 = 0; i1 < 64; i1++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer4W[0];
}
layer4W[i0][i1] = __fp_op->input(ALICE, 1, __tmp_in_layer4W, __m_bits, __e_bits) ;

}
}
delete[] __tmp_in_layer4W ;

vector < FPArray > layer4b = make_vector_float(ALICE, 10) ;

if ((__party == ALICE)) {
cout << ("Input layer4b:") << endl ;

}
float *__tmp_in_layer4b = new float[1] ;

for (uint32_t i0 = 0; i0 < 10; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_layer4b[0];
}
layer4b[i0] = __fp_op->input(ALICE, 1, __tmp_in_layer4b, __m_bits, __e_bits) ;

}
delete[] __tmp_in_layer4b ;

vector < vector < vector < vector < FPArray > > > > layer1WMom = make_vector_float(ALICE, 5, 5, 3, 64) ;

vector < FPArray > layer1bMom = make_vector_float(ALICE, 64) ;

vector < vector < vector < vector < FPArray > > > > layer2WMom = make_vector_float(ALICE, 5, 5, 64, 64) ;

vector < FPArray > layer2bMom = make_vector_float(ALICE, 64) ;

vector < vector < vector < vector < FPArray > > > > layer3WMom = make_vector_float(ALICE, 5, 5, 64, 64) ;

vector < FPArray > layer3bMom = make_vector_float(ALICE, 64) ;

vector < vector < FPArray > > layer4WMom = make_vector_float(ALICE, 10, 64) ;

vector < FPArray > layer4bMom = make_vector_float(ALICE, 10) ;

int32_t iters = 1 ;

for (uint32_t i = 0; i < 1; i++){
vector < vector < vector < vector < FPArray > > > > layer1In = make_vector_float(ALICE, 4, 32, 32, 3) ;

vector < vector < vector < vector < BoolArray > > > > layer1Hot = make_vector_bool(ALICE, 4, 30, 30, 64) ;

vector < vector < vector < vector < FPArray > > > > layer1Out = make_vector_float(ALICE, 4, 30, 30, 64) ;

vector < vector < vector < vector < BoolArray > > > > layer1Pool = make_vector_bool(ALICE, 4, 42, 42, 64) ;

vector < vector < vector < vector < FPArray > > > > layer2In = make_vector_float(ALICE, 4, 14, 14, 64) ;

vector < vector < vector < vector < BoolArray > > > > layer2Hot = make_vector_bool(ALICE, 4, 12, 12, 64) ;

vector < vector < vector < vector < FPArray > > > > layer2Out = make_vector_float(ALICE, 4, 12, 12, 64) ;

vector < vector < vector < vector < BoolArray > > > > layer2Pool = make_vector_bool(ALICE, 4, 15, 15, 64) ;

vector < vector < vector < vector < FPArray > > > > layer3In = make_vector_float(ALICE, 4, 5, 5, 64) ;

vector < vector < vector < vector < BoolArray > > > > layer3Hot = make_vector_bool(ALICE, 4, 3, 3, 64) ;

vector < vector < vector < vector < FPArray > > > > layer3Out = make_vector_float(ALICE, 4, 3, 3, 64) ;

vector < vector < vector < vector < BoolArray > > > > layer3Pool = make_vector_bool(ALICE, 4, 3, 3, 64) ;

vector < vector < FPArray > > layer4In = make_vector_float(ALICE, 4, 64) ;

vector < vector < FPArray > > fwdOut = make_vector_float(ALICE, 4, 10) ;

vector < FPArray > loss = make_vector_float(ALICE, 1) ;

forward(layer1W, layer1b, layer2W, layer2b, layer3W, layer3b, layer4W, layer4b, layer1In, layer1Hot, layer1Out, layer1Pool, layer2In, layer2Hot, layer2Out, layer2Pool, layer3In, layer3Hot, layer3Out, layer3Pool, layer4In, fwdOut);
// computeCELoss(4, 10, target, fwdOut, loss);
FPArray __tac_var209 = loss[0] ;

cout << "Value of __tac_var209 : " ;

__fp_pub = __fp_op->output(PUBLIC, __tac_var209) ;

cout << (__fp_pub.get_native_type<float>()[0]) << endl ;

backward(layer1W, layer1b, layer2W, layer2b, layer3W, layer3b, layer4W, layer4b, layer1In, layer1Hot, layer1Out, layer1Pool, layer2In, layer2Hot, layer2Out, layer2Pool, layer3In, layer3Hot, layer3Out, layer3Pool, layer4In, fwdOut, layer1WMom, layer1bMom, layer2WMom, layer2bMom, layer3WMom, layer3bMom, layer4WMom, layer4bMom, target);
}

printf("linear time = %f\n", linear_time) ;
printf("linear comm = %f\n", linear_comm) ;
printf("linear rounds = %d\n", linear_rounds) ;

printf("\nnon-linear time = %f\n", non_time) ;
printf("non-linear comm = %f\n", non_comm) ;
printf("non-linear rounds = %d\n", non_rounds) ;

return 0;
}

