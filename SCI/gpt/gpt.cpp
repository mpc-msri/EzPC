
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "library_float.h"

using namespace std ;
using namespace sci ;

int32_t n_tok = 10 ;
int32_t embed_dim = 768 ;
int32_t vocab_size = 50256 ;

float embed_time = 0.0 ; float embed_comm = 0.0 ;
float embed_add_time = 0.0 ; float embed_add_comm = 0.0 ;
float embed_norm_time = 0.0 ; float embed_norm_comm = 0.0 ;
float embed_linear_time = 0.0 ; float embed_linear_comm = 0.0 ;

float mha_time = 0.0 ; float mha_comm = 0.0 ;
float mha_linear_time = 0.0 ; float mha_linear_comm = 0.0 ;
float mha_nonlin_time = 0.0 ; float mha_nonlin_comm = 0.0 ;

float gelu_time = 0.0 ; float gelu_comm = 0.0 ;
float gelu_linear_time = 0.0 ; float gelu_linear_comm = 0.0 ;
float gelu_nonlin_time = 0.0 ; float gelu_nonlin_comm = 0.0 ;

float norm_mean_time = 0.0 ; float norm_mean_comm = 0.0 ;
float norm_stddev_time = 0.0 ; float norm_stddev_comm = 0.0 ;
float norm_step_time = 0.0 ; float norm_step_comm = 0.0 ;

float fatmul_time = 0.0 ; float fatmul_comm = 0.0 ;

extern void ElemWiseAdd(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
extern void ElemWiseSub(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
extern void ElemWiseMul(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
extern void ElemWiseDiv(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr) ;
extern void Sqrt(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);

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

// 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
// muls = 6
// adds = 2
// tanh = 1
void GeLU(int32_t sz) {
    auto arr1 = make_vector_float_rand(ALICE, sz) ;
    auto arr2 = make_vector_float_rand(ALICE, sz) ;

    for (int i = 0 ; i < 6 ; i++)
        ElemWiseMul(sz, arr1, arr2, arr1) ;

    for (int i = 0 ; i < 2 ; i++)
        ElemWiseAdd(sz, arr1, arr2, arr1) ;

    Tanh(sz, arr1, arr2) ;
}

void Linear(int m, int n, int p) {
    auto mat1 = make_vector_float_rand(ALICE, m, n) ;
    auto mat2 = make_vector_float_rand(ALICE, n, p) ;
    auto mat3 = make_vector_float_rand(ALICE, m, p) ;

    auto add1 = make_vector_float_rand(ALICE, m*p) ;
    auto add2 = make_vector_float_rand(ALICE, m*p) ;

    MatMul(m, n, p, mat1, mat2, mat3) ;
    ElemWiseAdd(m*p, add1, add2, add1) ;
}

void Norm() {
    auto add1 = make_vector_float_rand(ALICE, n_tok*embed_dim) ;
    auto add2 = make_vector_float_rand(ALICE, n_tok*embed_dim) ;

    auto mat1 = make_vector_float_rand(ALICE, n_tok, embed_dim) ;
    auto matsum1 = make_vector_float_rand(ALICE, n_tok) ;
    auto matsumdiv = make_vector_float_rand(ALICE, n_tok) ;

    // Calculate sigma
    auto bias1 = make_vector_float_rand(ALICE, embed_dim, n_tok) ;
    auto bias2 = make_vector_float_rand(ALICE, n_tok) ;

    /* ------------------- */
    auto start = clock_start() ;
    uint64_t initial_rounds = __iopack->get_rounds();
    float comm_start = __get_comm() ;
    /* ------------------- */

    getBiasDer(embed_dim, n_tok, bias1, bias2) ;
    ElemWiseDiv(n_tok, matsum1, matsumdiv, matsum1) ;

    /* ------------------- */
    long long t = time_from(start);
    float comm_end = __get_comm() ;
    norm_mean_time = t/1000.0 ;	
    norm_mean_comm = (comm_end - comm_start)/(1<<20) ;
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */

    // Calculate stddev
    ElemWiseSub(n_tok*embed_dim, add1, add2, add1) ;
    ElemWiseMul(n_tok*embed_dim, add1, add2, add1) ;
    getBiasDer(embed_dim, n_tok, bias1, bias2) ;
    ElemWiseDiv(n_tok, matsum1, matsumdiv, matsum1) ;
    Sqrt(n_tok, matsum1, matsum1) ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    norm_stddev_time = t/1000.0 ;	
    norm_stddev_comm = (comm_end - comm_start)/(1<<20) ;
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */

    // Normalization step
    ElemWiseSub(n_tok*embed_dim, add1, add2, add1) ;
    ElemWiseDiv(n_tok*embed_dim, add1, add2, add1) ;
    ElemWiseMul(n_tok*embed_dim, add1, add2, add1) ;
    ElemWiseAdd(n_tok*embed_dim, add1, add2, add1) ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    norm_step_time = t/1000.0 ;	
    norm_step_comm = (comm_end - comm_start)/(1<<20) ;
    /* ------------------- */
}

void EmbedLayerNormalization() {
    /* ------------------- */
    auto start = clock_start() ;
    uint64_t initial_rounds = __iopack->get_rounds();
    float comm_start = __get_comm() ;
    /* ------------------- */

    // Step a
    auto add1 = make_vector_float_rand(ALICE, n_tok*embed_dim) ;
    auto add2 = make_vector_float_rand(ALICE, n_tok*embed_dim) ;
    ElemWiseAdd(n_tok*embed_dim, add1, add2, add1) ;

    /* ------------------- */
    long long t = time_from(start);
    float comm_end = __get_comm() ;
    embed_add_time += t/1000.0 ;	
    embed_add_comm += (comm_end - comm_start)/(1<<20) ;
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */

    // Step b
    Norm() ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    embed_norm_time += t/1000.0 ;	
    embed_norm_comm += (comm_end - comm_start)/(1<<20) ;
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */
    
    // Step c
    Linear(n_tok, embed_dim, 3*embed_dim) ;    
    // cout << "\tELN : Finished Linear\n" ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    embed_linear_time += t/1000.0 ;	
    embed_linear_comm += (comm_end - comm_start)/(1<<20) ;
    /* ------------------- */
}

void MultiHeadAttention() {
    auto mat1 = make_vector_float_rand(ALICE, n_tok, 64) ;
    auto mat2 = make_vector_float_rand(ALICE, 64, n_tok) ;
    auto mat3 = make_vector_float_rand(ALICE, n_tok, n_tok) ;
    auto div1 = make_vector_float_rand(ALICE, n_tok*n_tok) ;
    auto div2 = make_vector_float_rand(ALICE, n_tok*n_tok) ;
    auto softin = make_vector_float_rand(ALICE, n_tok, n_tok) ;

    for (int i = 0 ; i < 1 ; i++) {
        /* ------------------- */
        auto start = clock_start() ;
        uint64_t initial_rounds = __iopack->get_rounds();
        float comm_start = __get_comm() ;
        /* ------------------- */

        MatMul(n_tok, 64, n_tok, mat1, mat2, mat3) ;
        ElemWiseDiv(n_tok*n_tok, div1, div2, div1) ;
        ElemWiseAdd(n_tok*n_tok, div1, div2, div1) ;
        // Outside softmax
        MatMul(n_tok, n_tok, 64, mat3, mat1, mat1) ;

        /* ------------------- */
        long long t = time_from(start);
        float comm_end = __get_comm() ;
        mha_linear_time += t/1000.0 ;	
        mha_linear_comm += (comm_end - comm_start)/(1<<20) ;
        start = clock_start() ;
        initial_rounds = __iopack->get_rounds();
        comm_start = __get_comm() ;
        /* ------------------- */

        // Softmax
        Softmax2(n_tok, n_tok, softin, softin) ;

        /* ------------------- */
        t = time_from(start);
        comm_end = __get_comm() ;
        mha_nonlin_time += t/1000.0 ;	
        mha_nonlin_comm += (comm_end - comm_start)/(1<<20) ;
        /* ------------------- */
    }
}

void Remaining() {
    /* ------------------- */
    auto start = clock_start() ;
    uint64_t initial_rounds = __iopack->get_rounds();
    float comm_start = __get_comm() ;
    /* ------------------- */

    auto add1 = make_vector_float_rand(ALICE, n_tok*embed_dim) ;
    auto add2 = make_vector_float_rand(ALICE, n_tok*embed_dim) ;
    ElemWiseAdd(n_tok*embed_dim, add1, add2, add1) ;
    Linear(n_tok, embed_dim, embed_dim) ;
    Norm() ;
    Linear(n_tok, embed_dim, 3*embed_dim) ;
    Linear(n_tok, 3*embed_dim, embed_dim) ;
    ElemWiseAdd(n_tok*embed_dim, add1, add2, add1) ;

    /* ------------------- */
    long long t = time_from(start);
    float comm_end = __get_comm() ;
    gelu_linear_time += t/1000.0 ;	
    gelu_linear_comm += (comm_end - comm_start)/(1<<20) ;
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */

    GeLU(n_tok*3*embed_dim) ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    gelu_nonlin_time += t/1000.0 ;	
    gelu_nonlin_comm += (comm_end - comm_start)/(1<<20) ;
    /* ------------------- */
}


void gpt() {
    /* ------------------- */
    auto start = clock_start() ;
    uint64_t initial_rounds = __iopack->get_rounds();
    float comm_start = __get_comm() ;
    /* ------------------- */

    EmbedLayerNormalization() ;

    /* ------------------- */
    long long t = time_from(start);
    float comm_end = __get_comm() ;
    embed_time += t/1000.0 ;		
    embed_comm += (comm_end - comm_start)/(1<<20) ;
    /* ------------------- */

    cout << "Finished embed layer norm\n" ;

    /* ------------------- */
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */
    
    MultiHeadAttention() ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    mha_time += t/1000.0 ;		
    mha_comm += (comm_end - comm_start)/(1<<20) ;
    /* ------------------- */

    cout << "Finished multi head attention\n" ;

    /* ------------------- */
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */

    Remaining() ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    gelu_time += t/1000.0 ;		
    gelu_comm += (comm_end - comm_start)/(1<<20) ;
    /* ------------------- */

    cout << "Finished remaining computation\n" ;

    /* ------------------- */
    start = clock_start() ;
    initial_rounds = __iopack->get_rounds();
    comm_start = __get_comm() ;
    /* ------------------- */

    Norm() ;

    auto mat1 = make_vector_float_rand(ALICE, 1, embed_dim) ;
    auto mat2 = make_vector_float_rand(ALICE, embed_dim, vocab_size) ;
    auto mat3 = make_vector_float_rand(ALICE, 1, vocab_size) ;

    MatMul(1, embed_dim, vocab_size, mat1, mat2, mat3) ;

    /* ------------------- */
    t = time_from(start);
    comm_end = __get_comm() ;
    fatmul_time += t/1000.0 ;		
    fatmul_comm += (comm_end - comm_start)/(1<<20) ;
    /* ------------------- */

    cout << "Finished big matmul\n" ;
}

int main (int __argc, char **__argv) {
    __init(__argc, __argv) ;

    cout << "Starting the GPT!\n" ;
    gpt() ;

    printf ("-------------------------\n") ;

    printf("embed time = %f\n", embed_time) ;
    printf("embed comm = %f\n", embed_comm) ;

    printf("embed add time = %f\n", embed_add_time) ;
    printf("embed add comm = %f\n", embed_add_comm) ;

    printf("embed norm time = %f\n", embed_norm_time) ;
    printf("embed norm comm = %f\n", embed_norm_comm) ;

    printf("embed linear time = %f\n", embed_linear_time) ;
    printf("embed linear comm = %f\n", embed_linear_comm) ;

    printf ("\n-------------------------\n") ;

    printf("\nmha time = %f\n", mha_time) ;
    printf("mha comm = %f\n", mha_comm) ;

    printf("mha linear time = %f\n", mha_linear_time) ;
    printf("mha linear comm = %f\n", mha_linear_comm) ;

    printf("mha nonlin time = %f\n", mha_nonlin_time) ;
    printf("mha nonlin comm = %f\n", mha_nonlin_comm) ;

    printf ("\n-------------------------\n") ;

    printf("\nGeLU time = %f\n", gelu_time) ;
    printf("GeLU comm = %f\n", gelu_comm) ;

    printf("gelu linear time = %f\n", gelu_linear_time) ;
    printf("gelu linear comm = %f\n", gelu_linear_comm) ;

    printf("gelu nonlin time = %f\n", gelu_nonlin_time) ;
    printf("gelu nonlin comm = %f\n", gelu_nonlin_comm) ;

    printf ("\n-------------------------\n") ;

    printf("\nFatmul time = %f\n", fatmul_time) ;
    printf("Fatmul comm = %f\n", fatmul_comm) ;

    printf ("\n-------------------------\n") ;

    printf("norm mean time = %f\n", norm_mean_time) ;
    printf("norm mean comm = %f\n", norm_mean_comm) ;

    printf("norm stddev time = %f\n", norm_stddev_time) ;
    printf("norm stddev comm = %f\n", norm_stddev_comm) ;

    printf("norm step time = %f\n", norm_step_time) ;
    printf("norm step comm = %f\n", norm_step_comm) ;

    return 0;
}