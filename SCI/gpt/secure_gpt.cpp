#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "library_float.h"

using namespace std ;
using namespace sci ;

extern void ElemWiseAdd(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
extern void ElemWiseSub(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
extern void ElemWiseMul(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
extern void ElemWiseDiv(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr) ;
extern void Sqrt(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);

extern float intToFloat(int32_t m);
extern void Softmax2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr);
extern void Ln(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
extern void Gelu(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
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

void Sqrt2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
inFlat[((i1 * s2) + i2)] = inArr[i1][i2] ;

}
}
Sqrt(sz, inFlat, outFlat);
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

void ElemWiseAdd2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < vector < FPArray > >& outArr){
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
ElemWiseAdd(sz, arr1Flat, arr2Flat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var64 = (i1 * s2) ;

int32_t __tac_var65 = (__tac_var64 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var65] ;

}
}
}

void ElemWiseSub2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < vector < FPArray > >& outArr){
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
ElemWiseSub(sz, arr1Flat, arr2Flat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var64 = (i1 * s2) ;

int32_t __tac_var65 = (__tac_var64 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var65] ;

}
}
}

void ElemWiseDiv2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < vector < FPArray > >& outArr){
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
ElemWiseDiv(sz, arr1Flat, arr2Flat, outArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var64 = (i1 * s2) ;

int32_t __tac_var65 = (__tac_var64 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var65] ;

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

void CastToint32(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = inArr[i];
    }
}

void CastToint32(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = inArr[i1][i2];
        }
    }
}

void Gelu2(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr) {
    int32_t size = (s1 * s2);
    auto reshapedInArr = make_vector_float(ALICE, size);
    auto reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }

    Gelu(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void FastGelu(int32_t batch_size, int32_t s1, int32_t s2, vector<vector<vector<FPArray>>> &input, vector<FPArray> &bias, vector<vector<vector<FPArray>>> &output){
    for(uint32_t b = 0; b <batch_size; b++){
        vector<vector<FPArray>> BiasAdded = make_vector_float(ALICE, s1, s2);
        GemmAdd(s1, s2, input[b], bias, BiasAdded) ;
        Gelu2(s1, s2, BiasAdded, output[b]) ;
    }
}

void calculate_means(int32_t s1, int32_t s2, vector<vector<FPArray>> &input, vector<FPArray> &output) {
    auto input_transpose = make_vector_float(ALICE, s2, s1) ;
    Transpose(s1, s2, input, input_transpose) ;
    auto means = make_vector_float(ALICE, s1) ;
    getBiasDer(s2, s1, input_transpose, means) ;
    auto mean_divver = make_vector_float(ALICE, s1) ;
    for (int i = 0 ; i < s1 ; i++)
        mean_divver[i] = __fp_op->input<float>(ALICE, 1, (float)s2, __m_bits, __e_bits) ;

    ElemWiseDiv(s1, means, mean_divver, output) ;
}

void LayerNormalization(int32_t s1, int32_t s2, vector<vector<FPArray>> &input, vector<FPArray> &weight, vector<FPArray> &bias, vector<vector<FPArray>> &output, float epsilon = 0.00001) {
    // Mean calculation
    auto means = make_vector_float(ALICE, s1) ;
    calculate_means(s1, s2, input, means) ;
    auto means_expanded = make_vector_float(ALICE, s1, s2) ;
    for (int i1 = 0 ; i1 < s1 ; i1++) {
        for (int i2 = 0 ; i2 < s2 ; i2++) {
            means_expanded[i1][i2] = means[i1] ;
        }
    }

    // Standard deviation calculation
    auto input_zero = make_vector_float(ALICE, s1, s2) ;
    ElemWiseSub2(s1, s2, input, means_expanded, input_zero) ;
    auto input_zero_sq = make_vector_float(ALICE, s1, s2) ;
    ElemWiseMul2(s1, s2, input_zero, input_zero, input_zero_sq) ;
    auto variance = make_vector_float(ALICE, s1) ;
    calculate_means(s1, s2, input_zero_sq, variance) ;
    auto variance_expanded = make_vector_float(ALICE, s1, s2) ;
    for (int i1 = 0 ; i1 < s1 ; i1++) {
        for (int i2 = 0 ; i2 < s2 ; i2++) {
            variance_expanded[i1][i2] = variance[i1] ;
        }
    }

    // Normalization step
    auto epsilon_expanded = make_vector_float(ALICE, s1, s2) ;
    for (int i1 = 0 ; i1 < s1 ; i1++) {
        for (int i2 = 0 ; i2 < s2 ; i2++) {
            epsilon_expanded[i1][i2] = __fp_op->input<float>(ALICE, 1, (float)epsilon, __m_bits, __e_bits) ;
        }
    }
    auto variance_epsilon_expanded = make_vector_float(ALICE, s1, s2) ;
    ElemWiseAdd2(s1, s2, variance_expanded, epsilon_expanded, variance_epsilon_expanded) ;
    auto denominator = make_vector_float(ALICE, s1, s2) ;
    Sqrt2(s1, s2, variance_epsilon_expanded, denominator) ;
    ElemWiseDiv2(s1, s2, input_zero, denominator, output) ;

    // Renormalization step
    auto weights_expanded = make_vector_float(ALICE, s1, s2) ;
    for (int i1 = 0 ; i1 < s1 ; i1++) {
        for (int i2 = 0 ; i2 < s2 ; i2++) {
            weights_expanded[i1][i2] = weight[i2] ;
        }
    }
    auto bias_expanded = make_vector_float(ALICE, s1, s2) ;
    for (int i1 = 0 ; i1 < s1 ; i1++) {
        for (int i2 = 0 ; i2 < s2 ; i2++) {
            bias_expanded[i1][i2] = bias[i2] ;
        }
    }
    auto mul_expanded = make_vector_float(ALICE, s1, s2) ;
    ElemWiseMul2(s1, s2, output, weights_expanded, output) ;
    ElemWiseAdd2(s1, s2, output, bias_expanded, output) ;
}


void EmbedLayerNormalization(int32_t batch_size, int32_t seq_len, vector<vector<FPArray>> &input_ids, vector<vector<FPArray>> &transformer_wte_weight, vector<vector<FPArray>> &transformer_wpe_weight, vector<FPArray> &transformer_h_ln_1_weight, vector<FPArray> &transformer_h_ln_1_bias, vector<vector<FPArray>> &position_ids, vector<vector<vector<FPArray>>> &EmbedLayerNormalization_output, vector<vector<vector<FPArray>>> &EmbedLayerNormalization_dummy_mask_index, vector<vector<vector<FPArray>>> &EmbedLayerNormalization_embedding_sum, int32_t embedding_len = 768){
	for(int32_t b = 0; b < batch_size; b++) {

        auto add1 = make_vector_float(ALICE, seq_len, embedding_len) ;
        auto add2 = make_vector_float(ALICE, seq_len, embedding_len) ;
        auto added = make_vector_float(ALICE, seq_len, embedding_len) ;

        for(int32_t i1 = 0; i1 < seq_len; i1++) {
            for(int32_t i2 = 0; i2 < embedding_len; i2++) {
                add1[i1][i2] = transformer_wte_weight[input_ids[b][i1]][i2] ;
                add2[i1][i2] = transformer_wpe_weight[position_ids[b][i1]][i2] ;
            }
        }

        ElemWiseAdd2(seq_len, embedding_len, add1, add2, EmbedLayerNormalization_embedding_sum[b]) ;        
        LayerNormalization(seq_len, embedding_len, EmbedLayerNormalization_embedding_sum[b], transformer_h_ln_1_weight, transformer_h_ln_1_bias, EmbedLayerNormalization_output[b]) ;
    }
}

void SkipLayerNormalization(int32_t batch_size, int32_t seq_len, int32_t embedding_len, vector<vector<vector<FPArray>>> &PreviousLayerNormalizationOutput, vector<vector<vector<FPArray>>> &Attention_matmul_output, vector<FPArray> &transformer_h_ln_weight, vector<FPArray> &transformer_h_ln_bias, vector<FPArray> &transformer_h_attn_c_proj_bias, vector<vector<vector<FPArray>>> &Output1, vector<vector<vector<FPArray>>> &Output2) {
    for(int32_t b = 0; b <batch_size; b++) {
        vector<vector<FPArray>> BiasAdded = make_vector_float(ALICE, seq_len, embedding_len);
        GemmAdd(seq_len, embedding_len, Attention_matmul_output[b], transformer_h_attn_c_proj_bias, BiasAdded) ;

        ElemWiseAdd2(seq_len, embedding_len, PreviousLayerNormalizationOutput[b], BiasAdded, Output2[b]) ;
        LayerNormalization(seq_len, embedding_len, Output2[b], transformer_h_ln_weight, transformer_h_ln_bias, Output1[b]) ;
    }
}

void Attention_Operation(int32_t s1, int32_t s2, vector<vector<FPArray>> &q, vector<vector<FPArray>> &k, vector<vector<FPArray>> &v, vector<vector<FPArray>> &mask, vector<vector<FPArray>> &output){
    vector<vector<FPArray>> k_transpose = make_vector_float(ALICE, s2, s1) ;
    Transpose(s1, s2, k, k_transpose) ;
    
    vector<vector<FPArray>> temp = make_vector_float(ALICE, s1, s1) ;
    MatMul(s1, s2, s1, q, k_transpose, temp) ;
    auto divver = make_vector_float(ALICE, s1, s1) ;
    for (int i1 = 0 ; i1 < s1 ; i1++) {
        for (int i2 = 0 ; i2 < s1 ; i2++) {
            divver[i1][i2] = __fp_op->input<float>(ALICE, 1, (float)sqrt(s2), __m_bits, __e_bits) ;
        }
    }
    
    ElemWiseDiv2(s1, s1, temp, divver, temp) ;
    ElemWiseAdd2(s1, s1, temp, mask, temp) ;
    
    vector<vector<FPArray>> softmaxed = make_vector_float(ALICE, s1, s1) ;
    Softmax2(s1, s1, temp, softmaxed) ;
    
    MatMul(s1, s1, s2, softmaxed, v, output) ;
}

void Attention(int32_t batch_size, int32_t seq_len, int32_t embedding_len, vector<vector<vector<FPArray>>> &LayerNormalizationOutput, vector<vector<FPArray>> &transformer_h_attn_c_attn_weight, vector<FPArray> &transformer_h_attn_c_attn_bias, vector<vector<FPArray>> &attention_mask, vector<vector<vector<FPArray>>> &AttentionOutput, int32_t num_heads = 12) {
    for(uint32_t b = 0; b <batch_size; b++){
        vector<vector<FPArray>> qkv = make_vector_float(ALICE, seq_len, 3*embedding_len);
        MatMul(seq_len, embedding_len, 3*embedding_len, LayerNormalizationOutput[b], transformer_h_attn_c_attn_weight, qkv) ;
        GemmAdd(seq_len, 3*embedding_len, qkv, transformer_h_attn_c_attn_bias, qkv) ;
        vector<vector<FPArray>> causal_mask = make_vector_float(ALICE, seq_len, seq_len);

        for(uint32_t i = 0; i <seq_len; i++){
            for(uint32_t j = 0; j <seq_len; j++){
                if (i<j) {
                    causal_mask[i][j] = __fp_op->input<float>(ALICE, 1, (float)-10000000000, __m_bits, __e_bits) ;
                }
                else {
                    causal_mask[i][j] = __fp_op->input<float>(ALICE, 1, (float)0, __m_bits, __e_bits) ; ;
                }
            }
        }

        for(uint32_t n = 0; n <num_heads; n++){
            vector<vector<FPArray>> q = make_vector_float(ALICE, seq_len, embedding_len/num_heads);
            vector<vector<FPArray>> k = make_vector_float(ALICE, seq_len, embedding_len/num_heads);
            vector<vector<FPArray>> v = make_vector_float(ALICE, seq_len, embedding_len/num_heads);
            vector<vector<FPArray>> head_output = make_vector_float(ALICE, seq_len, embedding_len/num_heads);
            for(uint32_t i = 0; i <seq_len; i++){
                for(uint32_t j = 0; j <embedding_len/num_heads; j++){
                    q[i][j] = qkv[i][n*64 + j] ;
                    k[i][j] = qkv[i][embedding_len + n*64 + j] ;
                    v[i][j] = qkv[i][2*embedding_len + n*64 + j] ;           
                }
            }
            Attention_Operation(seq_len,embedding_len/num_heads,q,k,v,causal_mask,head_output) ;
            for(uint32_t i = 0; i <seq_len; i++){
                for(uint32_t j = 0; j <embedding_len/num_heads; j++){
                   AttentionOutput[b][i][n*64 + j] = head_output[i][j] ;
                }
            }
        }
        
    }
}

void MatMulBatch(int32_t batch_size, int32_t M1D1, int32_t M1D2, int32_t M2D1, int32_t M2D2, vector<vector<vector<FPArray>>> &BatchInp, vector<vector<FPArray>> &MultiplierMat, vector<vector<vector<FPArray>>> &BatchOut){
    assert(M1D2==M2D1) ;
    for(int32_t b = 0; b <batch_size; b++){
        MatMul(M1D1, M1D2, M2D2, BatchInp[b], MultiplierMat, BatchOut[b]) ;
    }
    
}

///////////////////////

auto input1(int d1, int party)
{
    auto tmp0 = make_vector_float(party, d1);

    float *__tmp_in_tmp0 = new float[1];

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        if ((__party == party))
        {
            cin >> __tmp_in_tmp0[0];
        }
        tmp0[i0] = __fp_op->input(party, 1, __tmp_in_tmp0);
    }
    delete[] __tmp_in_tmp0;

    return tmp0;
}

auto input2(int d1, int d2, int party)
{
    auto tmp0 = make_vector_float(party, d1, d2);

    float *__tmp_in_tmp0 = new float[1];

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {

            if ((__party == party))
            {
                cin >> __tmp_in_tmp0[0];
            }
            tmp0[i0][i1] = __fp_op->input(party, 1, __tmp_in_tmp0);
        }
    }
    delete[] __tmp_in_tmp0;
    return tmp0;
}

auto input3(int d1, int d2, int d3, int party)
{
    auto tmp0 = make_vector_float(party, d1, d2, d3);

    float *__tmp_in_tmp0 = new float[1];

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            for (uint32_t i2 = 0; i2 < d3; i2++)
            {

                if ((__party == party))
                {
                    cin >> __tmp_in_tmp0[0];
                }
                tmp0[i0][i1][i2] = __fp_op->input(party, 1, __tmp_in_tmp0);
            }
        }
    }
    delete[] __tmp_in_tmp0;
    return tmp0;
}

auto input4(int d1, int d2, int d3, int d4, int party)
{
    auto tmp0 = make_vector_float(party, d1, d2, d3, d4);

    float *__tmp_in_tmp0 = new float[1];

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            for (uint32_t i2 = 0; i2 < d3; i2++)
            {
                for (uint32_t i3 = 0; i3 < d4; i3++)
                {
                    if ((__party == party))
                    {
                        cin >> __tmp_in_tmp0[0];
                    }
                    tmp0[i0][i1][i2][i3] = __fp_op->input(party, 1, __tmp_in_tmp0);
                }
            }
        }
    }
    delete[] __tmp_in_tmp0;
    return tmp0;
}

void output1(auto name, int d1, int party)
{
    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        __fp_pub = __fp_op->output(PUBLIC, name[i0]);

        if ((__party == party))
        {
            cout << (__fp_pub.get_native_type<float>()[0]) << endl;
        }
    }
}

void output2(auto name, int d1, int d2, int party)
{
    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            __fp_pub = __fp_op->output(PUBLIC, name[i0][i1]);

            if ((__party == party))
            {
                cout << (__fp_pub.get_native_type<float>()[0]) << endl;
            }
        }
    }
}

void output3(auto name, int d1, int d2, int d3, int party)
{
    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            for (uint32_t i2 = 0; i2 < d3; i2++)
            {
                __fp_pub = __fp_op->output(PUBLIC, name[i0][i1][i2]);

                if ((__party == party))
                {
                    cout << (__fp_pub.get_native_type<float>()[0]) << endl;
                }
            }
        }
    }
}

void output4(auto name, int d1, int d2, int d3, int d4, int party)
{
    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            for (uint32_t i2 = 0; i2 < d3; i2++)
            {
                for (uint32_t i3 = 0; i3 < d4; i3++)
                {
                    __fp_pub = __fp_op->output(PUBLIC, name[i0][i1][i2][i3]);

                    if ((__party == party))
                    {
                        cout << (__fp_pub.get_native_type<float>()[0]) << endl;
                    }
                }
            }
        }
    }
}

int main(int __argc, char **__argv)
{
    int batch_size = atoi(__argv[1]) ;
    int seq_len = atoi(__argv[2]) ;
    int total_seq_len = atoi(__argv[3]) ;
    int past_seq_len = atoi(__argv[4]) ;

    int __party=0;

    // Declaration and Input for variable attention_mask of shape ['batch_size', 'total_seq_len'] as var3
    if (__party == ALICE) cout << "Input var3:" << endl;
    auto var3 = input2(batch_size, total_seq_len, ALICE);

    // Function Call to Cast with inputs ['attention_mask'] and gives output ['attention_mask_int32']
    auto var165 = make_vector_float(ALICE, batch_size, total_seq_len);
    cout << "Inside Cast" << endl;
    // Call Cast(shape, input, output)
    // CastToint32(batch_size, total_seq_len, var3, var165);
    Reassign2(batch_size, total_seq_len, var3, var165);

    // Declaration and Input for variable input_ids of shape ['batch_size', 'seq_len'] as var1
    if (__party == ALICE) cout << "Input var1:" << endl;
    auto var1 = input2(batch_size, seq_len, ALICE);

    // Function Call to Cast with inputs ['input_ids'] and gives output ['197_int32']
    auto var166 = make_vector_float(ALICE, batch_size, seq_len);
    cout << "Inside Cast" << endl;
    // Call Cast(shape, input, output)
    Reassign2(batch_size, seq_len, var1, var166);

    // Declaration and Input for variable position_ids of shape ['batch_size', 'seq_len'] as var2
    if (__party == ALICE) cout << "Input var2:" << endl;
    auto var2 = input2(batch_size, seq_len, ALICE);

    // Function Call to Cast with inputs ['position_ids'] and gives output ['205_int32']
    auto var167 = make_vector_float(ALICE, batch_size, seq_len);
    cout << "Inside Cast" << endl;
    // Call Cast(shape, input, output)
    Reassign2(batch_size, seq_len, var2, var167);

    // Declaration and Input for variable transformer.wte.weight of shape [50257, 768] as var16
    if (__party == BOB) cout << "Input var16:" << endl;
    auto var16 = input2(50257, 768, BOB);

    // Declaration and Input for variable transformer.wpe.weight of shape [1024, 768] as var17
    if (__party == BOB) cout << "Input var17:" << endl;
    auto var17 = input2(1024, 768, BOB);

    // Declaration and Input for variable transformer.h.0.ln_1.weight of shape [768] as var18
    if (__party == BOB) cout << "Input var18:" << endl;
    auto var18 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.0.ln_1.bias of shape [768] as var19
    if (__party == BOB) cout << "Input var19:" << endl;
    auto var19 = input1(768, BOB);

    // Function Call to EmbedLayerNormalization with inputs ['197_int32', '', 'transformer.wte.weight', 'transformer.wpe.weight', '', 'transformer.h.0.ln_1.weight', 'transformer.h.0.ln_1.bias', '', '205_int32'] and gives output ['EmbedLayerNormalization_0_output', 'EmbedLayerNormalization_0_dummy_mask_index', 'EmbedLayerNormalization_0_embedding_sum']
    auto var168 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var169 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var170 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside EmbedLayerNormalization" << endl;
    // Call EmbedLayerNormalization(shape, input_ids, transformer_wte_weight, transformer_wpe_weight, transformer_h_ln_1_weight, transformer_h_ln_1_bias, position_ids, EmbedLayerNormalization_output, EmbedLayerNormalization_dummy_mask_index, EmbedLayerNormalization_embedding_sum)
    // EmbedLayerNormalization(batch_size, seq_len, var166, var16, var17, var18, var19, var167, var168, var169, var170);
    
    output2(var168[0], seq_len, 768, ALICE); ////////////////////////////
    
    // Declaration and Input for variable transformer.h.0.attn.c_attn.weight of shape [768, 2304] as var20
    if (__party == BOB) cout << "Input var20:" << endl;
    auto var20 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.0.attn.c_attn.bias of shape [2304] as var21
    if (__party == BOB) cout << "Input var21:" << endl;
    auto var21 = input1(2304, BOB);

    // Function Call to Attention with inputs ['EmbedLayerNormalization_0_output', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_attn.bias', 'attention_mask_int32', 'past_0'] and gives output ['GptAttention_0_output', 'present_0']
    auto var171 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var168, var20, var21, var165, var171);
    
    output2(var171[0], seq_len, 768, ALICE); ///////////////////////////
    
    // Declaration and Input for variable transformer.h.0.attn.c_proj.weight of shape [768, 768] as var22
    if (__party == BOB) cout << "Input var22:" << endl;
    auto var22 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_0_output', 'transformer.h.0.attn.c_proj.weight'] and gives output ['GptAttention_0_matmul_output']
    auto var173 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var171, var22, var173);

    // Declaration and Input for variable transformer.h.0.ln_2.weight of shape [768] as var24
    if (__party == BOB) cout << "Input var24:" << endl;
    auto var24 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.0.ln_2.bias of shape [768] as var25
    if (__party == BOB) cout << "Input var25:" << endl;
    auto var25 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.0.attn.c_proj.bias of shape [768] as var23
    if (__party == BOB) cout << "Input var23:" << endl;
    auto var23 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['EmbedLayerNormalization_0_embedding_sum', 'GptAttention_0_matmul_output', 'transformer.h.0.ln_2.weight', 'transformer.h.0.ln_2.bias', 'transformer.h.0.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_0_input', '', '', '398']
    auto var174 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var176 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var170, var173, var24, var25, var23, var174, var176);

    // Declaration and Input for variable transformer.h.0.mlp.c_fc.weight of shape [768, 3072] as var26
    if (__party == BOB) cout << "Input var26:" << endl;
    auto var26 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_0_input', 'transformer.h.0.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_0_output']
    auto var177 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var174, var26, var177);

    // Declaration and Input for variable transformer.h.0.mlp.c_fc.bias of shape [3072] as var27
    if (__party == BOB) cout << "Input var27:" << endl;
    auto var27 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_0_output', 'transformer.h.0.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_1_input']
    auto var178 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var177, var27, var178);
    output2(var178[0], seq_len, 3072, ALICE); //////////////////////////

    // Declaration and Input for variable transformer.h.0.mlp.c_proj.weight of shape [3072, 768] as var28
    if (__party == BOB) cout << "Input var28:" << endl;
    auto var28 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_1_input', 'transformer.h.0.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_1_output']
    auto var179 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var178, var28, var179);

    // Declaration and Input for variable transformer.h.1.ln_1.weight of shape [768] as var30
    if (__party == BOB) cout << "Input var30:" << endl;
    auto var30 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.1.ln_1.bias of shape [768] as var31
    if (__party == BOB) cout << "Input var31:" << endl;
    auto var31 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.0.mlp.c_proj.bias of shape [768] as var29
    if (__party == BOB) cout << "Input var29:" << endl;
    auto var29 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['398', 'FullyConnect_MatMul_1_output', 'transformer.h.1.ln_1.weight', 'transformer.h.1.ln_1.bias', 'transformer.h.0.mlp.c_proj.bias'] and gives output ['482', '', '', '471']
    auto var180 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var181 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var176, var179, var30, var31, var29, var180, var181);
    cout << "Special Test 1" << endl;
    output2(var180[0], seq_len, 768, ALICE); ///////////////////////////
    cout << "Special Test 2" << endl;
    output2(var181[0], seq_len, 768, ALICE); ///////////////////////////
    
    // Declaration and Input for variable transformer.h.1.attn.c_attn.weight of shape [768, 2304] as var32
    if (__party == BOB) cout << "Input var32:" << endl;
    auto var32 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.1.attn.c_attn.bias of shape [2304] as var33
    if (__party == BOB) cout << "Input var33:" << endl;
    auto var33 = input1(2304, BOB);

    // Function Call to Attention with inputs ['482', 'transformer.h.1.attn.c_attn.weight', 'transformer.h.1.attn.c_attn.bias', 'attention_mask_int32', 'past_1'] and gives output ['GptAttention_1_output', 'present_1']
    auto var182 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention1" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var180, var32, var33, var165, var182);
    output2(var182[0], seq_len, 768, ALICE); ///////////////////////////
    // Declaration and Input for variable transformer.h.1.attn.c_proj.weight of shape [768, 768] as var34
    if (__party == BOB) cout << "Input var34:" << endl;
    auto var34 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_1_output', 'transformer.h.1.attn.c_proj.weight'] and gives output ['GptAttention_1_matmul_output']
    auto var184 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var182, var34, var184);

    // Declaration and Input for variable transformer.h.1.ln_2.weight of shape [768] as var36
    if (__party == BOB) cout << "Input var36:" << endl;
    auto var36 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.1.ln_2.bias of shape [768] as var37
    if (__party == BOB) cout << "Input var37:" << endl;
    auto var37 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.1.attn.c_proj.bias of shape [768] as var35
    if (__party == BOB) cout << "Input var35:" << endl;
    auto var35 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['471', 'GptAttention_1_matmul_output', 'transformer.h.1.ln_2.weight', 'transformer.h.1.ln_2.bias', 'transformer.h.1.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_2_input', '', '', '643']
    auto var185 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var186 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var181, var184, var36, var37, var35, var185, var186);

    // Declaration and Input for variable transformer.h.1.mlp.c_fc.weight of shape [768, 3072] as var38
    if (__party == BOB) cout << "Input var38:" << endl;
    auto var38 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_2_input', 'transformer.h.1.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_2_output']
    auto var187 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var185, var38, var187);

    // Declaration and Input for variable transformer.h.1.mlp.c_fc.bias of shape [3072] as var39
    if (__party == BOB) cout << "Input var39:" << endl;
    auto var39 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_2_output', 'transformer.h.1.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_3_input']
    auto var188 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu1" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var187, var39, var188);
    output2(var188[0], seq_len, 3072, ALICE); //////////////////////////

    // Declaration and Input for variable transformer.h.1.mlp.c_proj.weight of shape [3072, 768] as var40
    if (__party == BOB) cout << "Input var40:" << endl;
    auto var40 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_3_input', 'transformer.h.1.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_3_output']
    auto var189 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var188, var40, var189);

    // Declaration and Input for variable transformer.h.2.ln_1.weight of shape [768] as var42
    if (__party == BOB) cout << "Input var42:" << endl;
    auto var42 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.2.ln_1.bias of shape [768] as var43
    if (__party == BOB) cout << "Input var43:" << endl;
    auto var43 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.1.mlp.c_proj.bias of shape [768] as var41
    if (__party == BOB) cout << "Input var41:" << endl;
    auto var41 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['643', 'FullyConnect_MatMul_3_output', 'transformer.h.2.ln_1.weight', 'transformer.h.2.ln_1.bias', 'transformer.h.1.mlp.c_proj.bias'] and gives output ['727', '', '', '716']
    auto var190 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var191 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var186, var189, var42, var43, var41, var190, var191);

    // Declaration and Input for variable transformer.h.2.attn.c_attn.weight of shape [768, 2304] as var44
    if (__party == BOB) cout << "Input var44:" << endl;
    auto var44 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.2.attn.c_attn.bias of shape [2304] as var45
    if (__party == BOB) cout << "Input var45:" << endl;
    auto var45 = input1(2304, BOB);

    // Function Call to Attention with inputs ['727', 'transformer.h.2.attn.c_attn.weight', 'transformer.h.2.attn.c_attn.bias', 'attention_mask_int32', 'past_2'] and gives output ['GptAttention_2_output', 'present_2']
    auto var192 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var190, var44, var45, var165, var192);

    // Declaration and Input for variable transformer.h.2.attn.c_proj.weight of shape [768, 768] as var46
    if (__party == BOB) cout << "Input var46:" << endl;
    auto var46 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_2_output', 'transformer.h.2.attn.c_proj.weight'] and gives output ['GptAttention_2_matmul_output']
    auto var194 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var192, var46, var194);

    // Declaration and Input for variable transformer.h.2.ln_2.weight of shape [768] as var48
    if (__party == BOB) cout << "Input var48:" << endl;
    auto var48 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.2.ln_2.bias of shape [768] as var49
    if (__party == BOB) cout << "Input var49:" << endl;
    auto var49 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.2.attn.c_proj.bias of shape [768] as var47
    if (__party == BOB) cout << "Input var47:" << endl;
    auto var47 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['716', 'GptAttention_2_matmul_output', 'transformer.h.2.ln_2.weight', 'transformer.h.2.ln_2.bias', 'transformer.h.2.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_4_input', '', '', '888']
    auto var195 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var196 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var191, var194, var48, var49, var47, var195, var196);

    // Declaration and Input for variable transformer.h.2.mlp.c_fc.weight of shape [768, 3072] as var50
    if (__party == BOB) cout << "Input var50:" << endl;
    auto var50 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_4_input', 'transformer.h.2.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_4_output']
    auto var197 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var195, var50, var197);

    // Declaration and Input for variable transformer.h.2.mlp.c_fc.bias of shape [3072] as var51
    if (__party == BOB) cout << "Input var51:" << endl;
    auto var51 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_4_output', 'transformer.h.2.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_5_input']
    auto var198 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var197, var51, var198);

    // Declaration and Input for variable transformer.h.2.mlp.c_proj.weight of shape [3072, 768] as var52
    if (__party == BOB) cout << "Input var52:" << endl;
    auto var52 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_5_input', 'transformer.h.2.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_5_output']
    auto var199 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var198, var52, var199);

    // Declaration and Input for variable transformer.h.3.ln_1.weight of shape [768] as var54
    if (__party == BOB) cout << "Input var54:" << endl;
    auto var54 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.3.ln_1.bias of shape [768] as var55
    if (__party == BOB) cout << "Input var55:" << endl;
    auto var55 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.2.mlp.c_proj.bias of shape [768] as var53
    if (__party == BOB) cout << "Input var53:" << endl;
    auto var53 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['888', 'FullyConnect_MatMul_5_output', 'transformer.h.3.ln_1.weight', 'transformer.h.3.ln_1.bias', 'transformer.h.2.mlp.c_proj.bias'] and gives output ['972', '', '', '961']
    auto var200 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var201 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var196, var199, var54, var55, var53, var200, var201);

    // Declaration and Input for variable transformer.h.3.attn.c_attn.weight of shape [768, 2304] as var56
    if (__party == BOB) cout << "Input var56:" << endl;
    auto var56 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.3.attn.c_attn.bias of shape [2304] as var57
    if (__party == BOB) cout << "Input var57:" << endl;
    auto var57 = input1(2304, BOB);

    // Function Call to Attention with inputs ['972', 'transformer.h.3.attn.c_attn.weight', 'transformer.h.3.attn.c_attn.bias', 'attention_mask_int32', 'past_3'] and gives output ['GptAttention_3_output', 'present_3']
    auto var202 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var200, var56, var57, var165, var202);

    // Declaration and Input for variable transformer.h.3.attn.c_proj.weight of shape [768, 768] as var58
    if (__party == BOB) cout << "Input var58:" << endl;
    auto var58 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_3_output', 'transformer.h.3.attn.c_proj.weight'] and gives output ['GptAttention_3_matmul_output']
    auto var204 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var202, var58, var204);

    // Declaration and Input for variable transformer.h.3.ln_2.weight of shape [768] as var60
    if (__party == BOB) cout << "Input var60:" << endl;
    auto var60 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.3.ln_2.bias of shape [768] as var61
    if (__party == BOB) cout << "Input var61:" << endl;
    auto var61 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.3.attn.c_proj.bias of shape [768] as var59
    if (__party == BOB) cout << "Input var59:" << endl;
    auto var59 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['961', 'GptAttention_3_matmul_output', 'transformer.h.3.ln_2.weight', 'transformer.h.3.ln_2.bias', 'transformer.h.3.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_6_input', '', '', '1133']
    auto var205 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var206 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var201, var204, var60, var61, var59, var205, var206);

    // Declaration and Input for variable transformer.h.3.mlp.c_fc.weight of shape [768, 3072] as var62
    if (__party == BOB) cout << "Input var62:" << endl;
    auto var62 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_6_input', 'transformer.h.3.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_6_output']
    auto var207 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var205, var62, var207);

    // Declaration and Input for variable transformer.h.3.mlp.c_fc.bias of shape [3072] as var63
    if (__party == BOB) cout << "Input var63:" << endl;
    auto var63 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_6_output', 'transformer.h.3.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_7_input']
    auto var208 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var207, var63, var208);

    // Declaration and Input for variable transformer.h.3.mlp.c_proj.weight of shape [3072, 768] as var64
    if (__party == BOB) cout << "Input var64:" << endl;
    auto var64 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_7_input', 'transformer.h.3.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_7_output']
    auto var209 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var208, var64, var209);

    // Declaration and Input for variable transformer.h.4.ln_1.weight of shape [768] as var66
    if (__party == BOB) cout << "Input var66:" << endl;
    auto var66 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.4.ln_1.bias of shape [768] as var67
    if (__party == BOB) cout << "Input var67:" << endl;
    auto var67 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.3.mlp.c_proj.bias of shape [768] as var65
    if (__party == BOB) cout << "Input var65:" << endl;
    auto var65 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1133', 'FullyConnect_MatMul_7_output', 'transformer.h.4.ln_1.weight', 'transformer.h.4.ln_1.bias', 'transformer.h.3.mlp.c_proj.bias'] and gives output ['1217', '', '', '1206']
    auto var210 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var211 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var206, var209, var66, var67, var65, var210, var211);

    // Declaration and Input for variable transformer.h.4.attn.c_attn.weight of shape [768, 2304] as var68
    if (__party == BOB) cout << "Input var68:" << endl;
    auto var68 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.4.attn.c_attn.bias of shape [2304] as var69
    if (__party == BOB) cout << "Input var69:" << endl;
    auto var69 = input1(2304, BOB);

    // Function Call to Attention with inputs ['1217', 'transformer.h.4.attn.c_attn.weight', 'transformer.h.4.attn.c_attn.bias', 'attention_mask_int32', 'past_4'] and gives output ['GptAttention_4_output', 'present_4']
    auto var212 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var210, var68, var69, var165, var212);

    // Declaration and Input for variable transformer.h.4.attn.c_proj.weight of shape [768, 768] as var70
    if (__party == BOB) cout << "Input var70:" << endl;
    auto var70 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_4_output', 'transformer.h.4.attn.c_proj.weight'] and gives output ['GptAttention_4_matmul_output']
    auto var214 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var212, var70, var214);

    // Declaration and Input for variable transformer.h.4.ln_2.weight of shape [768] as var72
    if (__party == BOB) cout << "Input var72:" << endl;
    auto var72 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.4.ln_2.bias of shape [768] as var73
    if (__party == BOB) cout << "Input var73:" << endl;
    auto var73 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.4.attn.c_proj.bias of shape [768] as var71
    if (__party == BOB) cout << "Input var71:" << endl;
    auto var71 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1206', 'GptAttention_4_matmul_output', 'transformer.h.4.ln_2.weight', 'transformer.h.4.ln_2.bias', 'transformer.h.4.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_8_input', '', '', '1378']
    auto var215 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var216 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var211, var214, var72, var73, var71, var215, var216);

    // Declaration and Input for variable transformer.h.4.mlp.c_fc.weight of shape [768, 3072] as var74
    if (__party == BOB) cout << "Input var74:" << endl;
    auto var74 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_8_input', 'transformer.h.4.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_8_output']
    auto var217 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var215, var74, var217);

    // Declaration and Input for variable transformer.h.4.mlp.c_fc.bias of shape [3072] as var75
    if (__party == BOB) cout << "Input var75:" << endl;
    auto var75 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_8_output', 'transformer.h.4.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_9_input']
    auto var218 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var217, var75, var218);

    // Declaration and Input for variable transformer.h.4.mlp.c_proj.weight of shape [3072, 768] as var76
    if (__party == BOB) cout << "Input var76:" << endl;
    auto var76 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_9_input', 'transformer.h.4.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_9_output']
    auto var219 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var218, var76, var219);

    // Declaration and Input for variable transformer.h.5.ln_1.weight of shape [768] as var78
    if (__party == BOB) cout << "Input var78:" << endl;
    auto var78 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.5.ln_1.bias of shape [768] as var79
    if (__party == BOB) cout << "Input var79:" << endl;
    auto var79 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.4.mlp.c_proj.bias of shape [768] as var77
    if (__party == BOB) cout << "Input var77:" << endl;
    auto var77 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1378', 'FullyConnect_MatMul_9_output', 'transformer.h.5.ln_1.weight', 'transformer.h.5.ln_1.bias', 'transformer.h.4.mlp.c_proj.bias'] and gives output ['1462', '', '', '1451']
    auto var220 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var221 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var216, var219, var78, var79, var77, var220, var221);

    // Declaration and Input for variable transformer.h.5.attn.c_attn.weight of shape [768, 2304] as var80
    if (__party == BOB) cout << "Input var80:" << endl;
    auto var80 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.5.attn.c_attn.bias of shape [2304] as var81
    if (__party == BOB) cout << "Input var81:" << endl;
    auto var81 = input1(2304, BOB);

    // Function Call to Attention with inputs ['1462', 'transformer.h.5.attn.c_attn.weight', 'transformer.h.5.attn.c_attn.bias', 'attention_mask_int32', 'past_5'] and gives output ['GptAttention_5_output', 'present_5']
    auto var222 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var220, var80, var81, var165, var222);

    // Declaration and Input for variable transformer.h.5.attn.c_proj.weight of shape [768, 768] as var82
    if (__party == BOB) cout << "Input var82:" << endl;
    auto var82 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_5_output', 'transformer.h.5.attn.c_proj.weight'] and gives output ['GptAttention_5_matmul_output']
    auto var224 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var222, var82, var224);

    // Declaration and Input for variable transformer.h.5.ln_2.weight of shape [768] as var84
    if (__party == BOB) cout << "Input var84:" << endl;
    auto var84 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.5.ln_2.bias of shape [768] as var85
    if (__party == BOB) cout << "Input var85:" << endl;
    auto var85 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.5.attn.c_proj.bias of shape [768] as var83
    if (__party == BOB) cout << "Input var83:" << endl;
    auto var83 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1451', 'GptAttention_5_matmul_output', 'transformer.h.5.ln_2.weight', 'transformer.h.5.ln_2.bias', 'transformer.h.5.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_10_input', '', '', '1623']
    auto var225 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var226 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var221, var224, var84, var85, var83, var225, var226);

    // Declaration and Input for variable transformer.h.5.mlp.c_fc.weight of shape [768, 3072] as var86
    if (__party == BOB) cout << "Input var86:" << endl;
    auto var86 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_10_input', 'transformer.h.5.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_10_output']
    auto var227 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var225, var86, var227);

    // Declaration and Input for variable transformer.h.5.mlp.c_fc.bias of shape [3072] as var87
    if (__party == BOB) cout << "Input var87:" << endl;
    auto var87 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_10_output', 'transformer.h.5.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_11_input']
    auto var228 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var227, var87, var228);

    // Declaration and Input for variable transformer.h.5.mlp.c_proj.weight of shape [3072, 768] as var88
    if (__party == BOB) cout << "Input var88:" << endl;
    auto var88 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_11_input', 'transformer.h.5.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_11_output']
    auto var229 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var228, var88, var229);

    // Declaration and Input for variable transformer.h.6.ln_1.weight of shape [768] as var90
    if (__party == BOB) cout << "Input var90:" << endl;
    auto var90 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.6.ln_1.bias of shape [768] as var91
    if (__party == BOB) cout << "Input var91:" << endl;
    auto var91 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.5.mlp.c_proj.bias of shape [768] as var89
    if (__party == BOB) cout << "Input var89:" << endl;
    auto var89 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1623', 'FullyConnect_MatMul_11_output', 'transformer.h.6.ln_1.weight', 'transformer.h.6.ln_1.bias', 'transformer.h.5.mlp.c_proj.bias'] and gives output ['1707', '', '', '1696']
    auto var230 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var231 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var226, var229, var90, var91, var89, var230, var231);

    // Declaration and Input for variable transformer.h.6.attn.c_attn.weight of shape [768, 2304] as var92
    if (__party == BOB) cout << "Input var92:" << endl;
    auto var92 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.6.attn.c_attn.bias of shape [2304] as var93
    if (__party == BOB) cout << "Input var93:" << endl;
    auto var93 = input1(2304, BOB);

    // Function Call to Attention with inputs ['1707', 'transformer.h.6.attn.c_attn.weight', 'transformer.h.6.attn.c_attn.bias', 'attention_mask_int32', 'past_6'] and gives output ['GptAttention_6_output', 'present_6']
    auto var232 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var230, var92, var93, var165, var232);

    // Declaration and Input for variable transformer.h.6.attn.c_proj.weight of shape [768, 768] as var94
    if (__party == BOB) cout << "Input var94:" << endl;
    auto var94 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_6_output', 'transformer.h.6.attn.c_proj.weight'] and gives output ['GptAttention_6_matmul_output']
    auto var234 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var232, var94, var234);

    // Declaration and Input for variable transformer.h.6.ln_2.weight of shape [768] as var96
    if (__party == BOB) cout << "Input var96:" << endl;
    auto var96 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.6.ln_2.bias of shape [768] as var97
    if (__party == BOB) cout << "Input var97:" << endl;
    auto var97 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.6.attn.c_proj.bias of shape [768] as var95
    if (__party == BOB) cout << "Input var95:" << endl;
    auto var95 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1696', 'GptAttention_6_matmul_output', 'transformer.h.6.ln_2.weight', 'transformer.h.6.ln_2.bias', 'transformer.h.6.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_12_input', '', '', '1868']
    auto var235 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var236 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var231, var234, var96, var97, var95, var235, var236);

    // Declaration and Input for variable transformer.h.6.mlp.c_fc.weight of shape [768, 3072] as var98
    if (__party == BOB) cout << "Input var98:" << endl;
    auto var98 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_12_input', 'transformer.h.6.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_12_output']
    auto var237 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var235, var98, var237);

    // Declaration and Input for variable transformer.h.6.mlp.c_fc.bias of shape [3072] as var99
    if (__party == BOB) cout << "Input var99:" << endl;
    auto var99 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_12_output', 'transformer.h.6.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_13_input']
    auto var238 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var237, var99, var238);

    // Declaration and Input for variable transformer.h.6.mlp.c_proj.weight of shape [3072, 768] as var100
    if (__party == BOB) cout << "Input var100:" << endl;
    auto var100 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_13_input', 'transformer.h.6.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_13_output']
    auto var239 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var238, var100, var239);

    // Declaration and Input for variable transformer.h.7.ln_1.weight of shape [768] as var102
    if (__party == BOB) cout << "Input var102:" << endl;
    auto var102 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.7.ln_1.bias of shape [768] as var103
    if (__party == BOB) cout << "Input var103:" << endl;
    auto var103 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.6.mlp.c_proj.bias of shape [768] as var101
    if (__party == BOB) cout << "Input var101:" << endl;
    auto var101 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1868', 'FullyConnect_MatMul_13_output', 'transformer.h.7.ln_1.weight', 'transformer.h.7.ln_1.bias', 'transformer.h.6.mlp.c_proj.bias'] and gives output ['1952', '', '', '1941']
    auto var240 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var241 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var236, var239, var102, var103, var101, var240, var241);

    // Declaration and Input for variable transformer.h.7.attn.c_attn.weight of shape [768, 2304] as var104
    if (__party == BOB) cout << "Input var104:" << endl;
    auto var104 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.7.attn.c_attn.bias of shape [2304] as var105
    if (__party == BOB) cout << "Input var105:" << endl;
    auto var105 = input1(2304, BOB);

    // Function Call to Attention with inputs ['1952', 'transformer.h.7.attn.c_attn.weight', 'transformer.h.7.attn.c_attn.bias', 'attention_mask_int32', 'past_7'] and gives output ['GptAttention_7_output', 'present_7']
    auto var242 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var240, var104, var105, var165, var242);

    // Declaration and Input for variable transformer.h.7.attn.c_proj.weight of shape [768, 768] as var106
    if (__party == BOB) cout << "Input var106:" << endl;
    auto var106 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_7_output', 'transformer.h.7.attn.c_proj.weight'] and gives output ['GptAttention_7_matmul_output']
    auto var244 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var242, var106, var244);

    // Declaration and Input for variable transformer.h.7.ln_2.weight of shape [768] as var108
    if (__party == BOB) cout << "Input var108:" << endl;
    auto var108 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.7.ln_2.bias of shape [768] as var109
    if (__party == BOB) cout << "Input var109:" << endl;
    auto var109 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.7.attn.c_proj.bias of shape [768] as var107
    if (__party == BOB) cout << "Input var107:" << endl;
    auto var107 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['1941', 'GptAttention_7_matmul_output', 'transformer.h.7.ln_2.weight', 'transformer.h.7.ln_2.bias', 'transformer.h.7.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_14_input', '', '', '2113']
    auto var245 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var246 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var241, var244, var108, var109, var107, var245, var246);

    // Declaration and Input for variable transformer.h.7.mlp.c_fc.weight of shape [768, 3072] as var110
    if (__party == BOB) cout << "Input var110:" << endl;
    auto var110 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_14_input', 'transformer.h.7.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_14_output']
    auto var247 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var245, var110, var247);

    // Declaration and Input for variable transformer.h.7.mlp.c_fc.bias of shape [3072] as var111
    if (__party == BOB) cout << "Input var111:" << endl;
    auto var111 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_14_output', 'transformer.h.7.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_15_input']
    auto var248 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var247, var111, var248);

    // Declaration and Input for variable transformer.h.7.mlp.c_proj.weight of shape [3072, 768] as var112
    if (__party == BOB) cout << "Input var112:" << endl;
    auto var112 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_15_input', 'transformer.h.7.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_15_output']
    auto var249 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var248, var112, var249);

    // Declaration and Input for variable transformer.h.8.ln_1.weight of shape [768] as var114
    if (__party == BOB) cout << "Input var114:" << endl;
    auto var114 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.8.ln_1.bias of shape [768] as var115
    if (__party == BOB) cout << "Input var115:" << endl;
    auto var115 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.7.mlp.c_proj.bias of shape [768] as var113
    if (__party == BOB) cout << "Input var113:" << endl;
    auto var113 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2113', 'FullyConnect_MatMul_15_output', 'transformer.h.8.ln_1.weight', 'transformer.h.8.ln_1.bias', 'transformer.h.7.mlp.c_proj.bias'] and gives output ['2197', '', '', '2186']
    auto var250 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var251 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var246, var249, var114, var115, var113, var250, var251);

    // Declaration and Input for variable transformer.h.8.attn.c_attn.weight of shape [768, 2304] as var116
    if (__party == BOB) cout << "Input var116:" << endl;
    auto var116 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.8.attn.c_attn.bias of shape [2304] as var117
    if (__party == BOB) cout << "Input var117:" << endl;
    auto var117 = input1(2304, BOB);

    // Function Call to Attention with inputs ['2197', 'transformer.h.8.attn.c_attn.weight', 'transformer.h.8.attn.c_attn.bias', 'attention_mask_int32', 'past_8'] and gives output ['GptAttention_8_output', 'present_8']
    auto var252 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var250, var116, var117, var165, var252);

    // Declaration and Input for variable transformer.h.8.attn.c_proj.weight of shape [768, 768] as var118
    if (__party == BOB) cout << "Input var118:" << endl;
    auto var118 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_8_output', 'transformer.h.8.attn.c_proj.weight'] and gives output ['GptAttention_8_matmul_output']
    auto var254 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var252, var118, var254);

    // Declaration and Input for variable transformer.h.8.ln_2.weight of shape [768] as var120
    if (__party == BOB) cout << "Input var120:" << endl;
    auto var120 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.8.ln_2.bias of shape [768] as var121
    if (__party == BOB) cout << "Input var121:" << endl;
    auto var121 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.8.attn.c_proj.bias of shape [768] as var119
    if (__party == BOB) cout << "Input var119:" << endl;
    auto var119 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2186', 'GptAttention_8_matmul_output', 'transformer.h.8.ln_2.weight', 'transformer.h.8.ln_2.bias', 'transformer.h.8.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_16_input', '', '', '2358']
    auto var255 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var256 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var251, var254, var120, var121, var119, var255, var256);

    // Declaration and Input for variable transformer.h.8.mlp.c_fc.weight of shape [768, 3072] as var122
    if (__party == BOB) cout << "Input var122:" << endl;
    auto var122 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_16_input', 'transformer.h.8.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_16_output']
    auto var257 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var255, var122, var257);

    // Declaration and Input for variable transformer.h.8.mlp.c_fc.bias of shape [3072] as var123
    if (__party == BOB) cout << "Input var123:" << endl;
    auto var123 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_16_output', 'transformer.h.8.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_17_input']
    auto var258 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var257, var123, var258);

    // Declaration and Input for variable transformer.h.8.mlp.c_proj.weight of shape [3072, 768] as var124
    if (__party == BOB) cout << "Input var124:" << endl;
    auto var124 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_17_input', 'transformer.h.8.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_17_output']
    auto var259 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var258, var124, var259);

    // Declaration and Input for variable transformer.h.9.ln_1.weight of shape [768] as var126
    if (__party == BOB) cout << "Input var126:" << endl;
    auto var126 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.9.ln_1.bias of shape [768] as var127
    if (__party == BOB) cout << "Input var127:" << endl;
    auto var127 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.8.mlp.c_proj.bias of shape [768] as var125
    if (__party == BOB) cout << "Input var125:" << endl;
    auto var125 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2358', 'FullyConnect_MatMul_17_output', 'transformer.h.9.ln_1.weight', 'transformer.h.9.ln_1.bias', 'transformer.h.8.mlp.c_proj.bias'] and gives output ['2442', '', '', '2431']
    auto var260 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var261 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var256, var259, var126, var127, var125, var260, var261);

    // Declaration and Input for variable transformer.h.9.attn.c_attn.weight of shape [768, 2304] as var128
    if (__party == BOB) cout << "Input var128:" << endl;
    auto var128 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.9.attn.c_attn.bias of shape [2304] as var129
    if (__party == BOB) cout << "Input var129:" << endl;
    auto var129 = input1(2304, BOB);

    // Function Call to Attention with inputs ['2442', 'transformer.h.9.attn.c_attn.weight', 'transformer.h.9.attn.c_attn.bias', 'attention_mask_int32', 'past_9'] and gives output ['GptAttention_9_output', 'present_9']
    auto var262 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var260, var128, var129, var165, var262);

    // Declaration and Input for variable transformer.h.9.attn.c_proj.weight of shape [768, 768] as var130
    if (__party == BOB) cout << "Input var130:" << endl;
    auto var130 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_9_output', 'transformer.h.9.attn.c_proj.weight'] and gives output ['GptAttention_9_matmul_output']
    auto var264 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var262, var130, var264);

    // Declaration and Input for variable transformer.h.9.ln_2.weight of shape [768] as var132
    if (__party == BOB) cout << "Input var132:" << endl;
    auto var132 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.9.ln_2.bias of shape [768] as var133
    if (__party == BOB) cout << "Input var133:" << endl;
    auto var133 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.9.attn.c_proj.bias of shape [768] as var131
    if (__party == BOB) cout << "Input var131:" << endl;
    auto var131 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2431', 'GptAttention_9_matmul_output', 'transformer.h.9.ln_2.weight', 'transformer.h.9.ln_2.bias', 'transformer.h.9.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_18_input', '', '', '2603']
    auto var265 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var266 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var261, var264, var132, var133, var131, var265, var266);

    // Declaration and Input for variable transformer.h.9.mlp.c_fc.weight of shape [768, 3072] as var134
    if (__party == BOB) cout << "Input var134:" << endl;
    auto var134 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_18_input', 'transformer.h.9.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_18_output']
    auto var267 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var265, var134, var267);

    // Declaration and Input for variable transformer.h.9.mlp.c_fc.bias of shape [3072] as var135
    if (__party == BOB) cout << "Input var135:" << endl;
    auto var135 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_18_output', 'transformer.h.9.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_19_input']
    auto var268 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var267, var135, var268);

    // Declaration and Input for variable transformer.h.9.mlp.c_proj.weight of shape [3072, 768] as var136
    if (__party == BOB) cout << "Input var136:" << endl;
    auto var136 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_19_input', 'transformer.h.9.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_19_output']
    auto var269 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var268, var136, var269);

    // Declaration and Input for variable transformer.h.10.ln_1.weight of shape [768] as var138
    if (__party == BOB) cout << "Input var138:" << endl;
    auto var138 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.10.ln_1.bias of shape [768] as var139
    if (__party == BOB) cout << "Input var139:" << endl;
    auto var139 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.9.mlp.c_proj.bias of shape [768] as var137
    if (__party == BOB) cout << "Input var137:" << endl;
    auto var137 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2603', 'FullyConnect_MatMul_19_output', 'transformer.h.10.ln_1.weight', 'transformer.h.10.ln_1.bias', 'transformer.h.9.mlp.c_proj.bias'] and gives output ['2687', '', '', '2676']
    auto var270 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var271 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var266, var269, var138, var139, var137, var270, var271);

    // Declaration and Input for variable transformer.h.10.attn.c_attn.weight of shape [768, 2304] as var140
    if (__party == BOB) cout << "Input var140:" << endl;
    auto var140 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.10.attn.c_attn.bias of shape [2304] as var141
    if (__party == BOB) cout << "Input var141:" << endl;
    auto var141 = input1(2304, BOB);

    // Function Call to Attention with inputs ['2687', 'transformer.h.10.attn.c_attn.weight', 'transformer.h.10.attn.c_attn.bias', 'attention_mask_int32', 'past_10'] and gives output ['GptAttention_10_output', 'present_10']
    auto var272 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var270, var140, var141, var165, var272);

    // Declaration and Input for variable transformer.h.10.attn.c_proj.weight of shape [768, 768] as var142
    if (__party == BOB) cout << "Input var142:" << endl;
    auto var142 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_10_output', 'transformer.h.10.attn.c_proj.weight'] and gives output ['GptAttention_10_matmul_output']
    auto var274 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var272, var142, var274);

    // Declaration and Input for variable transformer.h.10.ln_2.weight of shape [768] as var144
    if (__party == BOB) cout << "Input var144:" << endl;
    auto var144 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.10.ln_2.bias of shape [768] as var145
    if (__party == BOB) cout << "Input var145:" << endl;
    auto var145 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.10.attn.c_proj.bias of shape [768] as var143
    if (__party == BOB) cout << "Input var143:" << endl;
    auto var143 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2676', 'GptAttention_10_matmul_output', 'transformer.h.10.ln_2.weight', 'transformer.h.10.ln_2.bias', 'transformer.h.10.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_20_input', '', '', '2848']
    auto var275 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var276 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var271, var274, var144, var145, var143, var275, var276);

    // Declaration and Input for variable transformer.h.10.mlp.c_fc.weight of shape [768, 3072] as var146
    if (__party == BOB) cout << "Input var146:" << endl;
    auto var146 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_20_input', 'transformer.h.10.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_20_output']
    auto var277 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var275, var146, var277);

    // Declaration and Input for variable transformer.h.10.mlp.c_fc.bias of shape [3072] as var147
    if (__party == BOB) cout << "Input var147:" << endl;
    auto var147 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_20_output', 'transformer.h.10.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_21_input']
    auto var278 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var277, var147, var278);

    // Declaration and Input for variable transformer.h.10.mlp.c_proj.weight of shape [3072, 768] as var148
    if (__party == BOB) cout << "Input var148:" << endl;
    auto var148 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_21_input', 'transformer.h.10.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_21_output']
    auto var279 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var278, var148, var279);

    // Declaration and Input for variable transformer.h.11.ln_1.weight of shape [768] as var150
    if (__party == BOB) cout << "Input var150:" << endl;
    auto var150 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.11.ln_1.bias of shape [768] as var151
    if (__party == BOB) cout << "Input var151:" << endl;
    auto var151 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.10.mlp.c_proj.bias of shape [768] as var149
    if (__party == BOB) cout << "Input var149:" << endl;
    auto var149 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2848', 'FullyConnect_MatMul_21_output', 'transformer.h.11.ln_1.weight', 'transformer.h.11.ln_1.bias', 'transformer.h.10.mlp.c_proj.bias'] and gives output ['2932', '', '', '2921']
    auto var280 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var281 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var276, var279, var150, var151, var149, var280, var281);

    // Declaration and Input for variable transformer.h.11.attn.c_attn.weight of shape [768, 2304] as var152
    if (__party == BOB) cout << "Input var152:" << endl;
    auto var152 = input2(768, 2304, BOB);

    // Declaration and Input for variable transformer.h.11.attn.c_attn.bias of shape [2304] as var153
    if (__party == BOB) cout << "Input var153:" << endl;
    auto var153 = input1(2304, BOB);

    // Function Call to Attention with inputs ['2932', 'transformer.h.11.attn.c_attn.weight', 'transformer.h.11.attn.c_attn.bias', 'attention_mask_int32', 'past_11'] and gives output ['GptAttention_11_output', 'present_11']
    auto var282 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside Attention" << endl;
    // Call Attention(shape, LayerNormalizationOutput, transformer_h_attn_c_attn_weight, transformer_h_attn_c_attn_bias, attention_mask, AttentionOutput)
    Attention(batch_size, seq_len, 768, var280, var152, var153, var165, var282);

    // Declaration and Input for variable transformer.h.11.attn.c_proj.weight of shape [768, 768] as var154
    if (__party == BOB) cout << "Input var154:" << endl;
    auto var154 = input2(768, 768, BOB);

    // Function Call to MatMul with inputs ['GptAttention_11_output', 'transformer.h.11.attn.c_proj.weight'] and gives output ['GptAttention_11_matmul_output']
    auto var284 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 768, var282, var154, var284);

    // Declaration and Input for variable transformer.h.11.ln_2.weight of shape [768] as var156
    if (__party == BOB) cout << "Input var156:" << endl;
    auto var156 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.11.ln_2.bias of shape [768] as var157
    if (__party == BOB) cout << "Input var157:" << endl;
    auto var157 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.11.attn.c_proj.bias of shape [768] as var155
    if (__party == BOB) cout << "Input var155:" << endl;
    auto var155 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['2921', 'GptAttention_11_matmul_output', 'transformer.h.11.ln_2.weight', 'transformer.h.11.ln_2.bias', 'transformer.h.11.attn.c_proj.bias'] and gives output ['FullyConnect_MatMul_22_input', '', '', '3093']
    auto var285 = make_vector_float(ALICE, batch_size, seq_len, 768);
    auto var286 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var281, var284, var156, var157, var155, var285, var286);

    // Declaration and Input for variable transformer.h.11.mlp.c_fc.weight of shape [768, 3072] as var158
    if (__party == BOB) cout << "Input var158:" << endl;
    auto var158 = input2(768, 3072, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_22_input', 'transformer.h.11.mlp.c_fc.weight'] and gives output ['FullyConnect_MatMul_22_output']
    auto var287 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 3072, var285, var158, var287);

    // Declaration and Input for variable transformer.h.11.mlp.c_fc.bias of shape [3072] as var159
    if (__party == BOB) cout << "Input var159:" << endl;
    auto var159 = input1(3072, BOB);

    // Function Call to FastGelu with inputs ['FullyConnect_MatMul_22_output', 'transformer.h.11.mlp.c_fc.bias'] and gives output ['FullyConnect_MatMul_23_input']
    auto var288 = make_vector_float(ALICE, batch_size, seq_len, 3072);
    cout << "Inside FastGelu11" << endl;
    // Call FastGelu(shape, input, bias, output)
    FastGelu(batch_size, seq_len, 3072, var287, var159, var288);
    output2(var288[0], seq_len, 3072, ALICE); ////////////////////////// 
    
    // Declaration and Input for variable transformer.h.11.mlp.c_proj.weight of shape [3072, 768] as var160
    if (__party == BOB) cout << "Input var160:" << endl;
    auto var160 = input2(3072, 768, BOB);

    // Function Call to MatMul with inputs ['FullyConnect_MatMul_23_input', 'transformer.h.11.mlp.c_proj.weight'] and gives output ['FullyConnect_MatMul_23_output']
    auto var289 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 3072, 3072, 768, var288, var160, var289);

    // Declaration and Input for variable transformer.ln_f.weight of shape [768] as var162
    if (__party == BOB) cout << "Input var162:" << endl;
    auto var162 = input1(768, BOB);

    // Declaration and Input for variable transformer.ln_f.bias of shape [768] as var163
    if (__party == BOB) cout << "Input var163:" << endl;
    auto var163 = input1(768, BOB);

    // Declaration and Input for variable transformer.h.11.mlp.c_proj.bias of shape [768] as var161
    if (__party == BOB) cout << "Input var161:" << endl;
    auto var161 = input1(768, BOB);

    // Function Call to SkipLayerNormalization with inputs ['3093', 'FullyConnect_MatMul_23_output', 'transformer.ln_f.weight', 'transformer.ln_f.bias', 'transformer.h.11.mlp.c_proj.bias'] and gives output ['3177']
    auto var290 = make_vector_float(ALICE, batch_size, seq_len, 768);
    cout << "Inside SkipLayerNormalization" << endl;
    // Call SkipLayerNormalization(shape, PreviousLayerNormalizationOutput, Attention_matmul_output, transformer_h_ln_weight, transformer_h_ln_bias, transformer_h_attn_c_proj_bias, Output1, Output2)
    SkipLayerNormalization(batch_size, seq_len, 768, var286, var289, var162, var163, var161, var290, var290);

    // Declaration and Input for variable 3452 of shape [768, 50257] as var164
    if (__party == BOB) cout << "Input var164:" << endl;
    auto var164 = input2(768, 50257, BOB);

    // Function Call to MatMul with inputs ['3177', '3452'] and gives output ['logits']
    auto var291 = make_vector_float(ALICE, batch_size, seq_len, 50257);
    cout << "Inside MatMul" << endl;
    // Call MatMulBatch(shape,inputs,output)
    MatMulBatch(batch_size, seq_len, 768, 768, 50257, var290, var164, var291);

    // Output of variable 'logits' of shape ['batch_size', 'seq_len', 50257] as var291 to ALICE
//     output3(var291, batch_size, seq_len, 50257, ALICE);
    cout << "Logits Output" ;
    output1(var291[0][9], 50257, ALICE);

    return 0;

}