#include "cleartext_library_float.h"
#include "cleartext_inout.cpp"
/////////////////////////////////////
#include "math.h"
#include <assert.h>
/////////////////////////////////////

extern float intToFloat(int32_t m);
extern void Softmax2(int32_t s1, int32_t s2, vector<vector<float>> &inArr, vector<vector<float>> &outArr);
extern void Ln(int32_t s1, vector<float> &inArr, vector<float> &outArr);
extern void getOutDer(int32_t s1, int32_t s2, vector<vector<float>> &batchSoft, vector<vector<float>> &lab, vector<vector<float>> &der);
extern void MatMul(int32_t s1, int32_t s2, int32_t s3, vector<vector<float>> &mat1, vector<vector<float>> &mat2, vector<vector<float>> &mat3);
extern void GemmAdd(int32_t s1, int32_t s2, vector<vector<float>> &prod, vector<float> &bias, vector<vector<float>> &out);
extern void dotProduct2(int32_t s1, int32_t s2, vector<vector<float>> &arr1, vector<vector<float>> &arr2, vector<float> &outArr);
extern void Relu(int32_t s1, vector<float> &inArr, vector<float> &outArr, vector<bool> &hotArr);
extern void getBiasDer(int32_t s1, int32_t s2, vector<vector<float>> &der, vector<float> &biasDer);
extern void IfElse(int32_t s1, vector<float> &dat, vector<bool> &hot, vector<float> &out, bool flip);
extern void updateWeights(int32_t s, float lr, vector<float> &bias, vector<float> &der);
extern void getLoss(int32_t m, vector<float> &lossTerms, vector<float> &loss);
extern void computeMSELoss(int32_t m, int32_t s, vector<vector<float>> &target, vector<vector<float>> &fwdOut, vector<float> &loss);

extern void Tanh(int32_t s1, vector<float> &inArr, vector<float> &outArr);

extern void Gelu(int32_t s1, vector<float>& inArr, vector<float>& outArr) ;

void ElemWiseAdd(int32_t s1, int32_t s2, auto &arr1, auto &arr2, auto &outArr) ;
void Transpose2T10(int32_t s1, int32_t s2, auto &inArr, auto &outArr) ;
////////////////////////
void PrintMatrix(int32_t s1, int32_t s2, vector<vector<float>> &m){
    for(int32_t i1=0; i1<s1; i1++){
        for(int32_t i2=0; i2<s2; i2++)
            cout << m[i1][i2] << "\n" ;
        cout << "\n" ;
    }
}

void CastToint32(int32_t s1, vector<int64_t> &inArr, vector<int32_t> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = (int32_t)inArr[i];
    }
}

void CastToint32(int32_t s1, int32_t s2, auto &inArr, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = (int32_t)inArr[i1][i2];
        }
    }
}

void Gelu2(int32_t s1, int32_t s2, auto &inArr, auto &outArr) {
    int32_t size = (s1 * s2);
    auto reshapedInArr = make_vector<float>(size);
    auto reshapedOutArr = make_vector<float>(size);

    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }

    Gelu(size, reshapedInArr, reshapedOutArr);

    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void FastGelu(int32_t batch_size, int32_t s1, int32_t s2, vector<vector<vector<float>>> &input, vector<float> &bias, vector<vector<vector<float>>> &output){
    for(int32_t b = 0; b <batch_size; b++){
        vector<vector<float>> BiasAdded = make_vector<float>(s1, s2);
        GemmAdd(s1, s2, input[b], bias, BiasAdded) ;
        Gelu2(s1, s2, BiasAdded, output[b]) ;
    }
}


void LayerNormalization(int32_t s1, int32_t s2, vector<vector<float>> &input, vector<float> &weight, vector<float> &bias, vector<vector<float>> &output, float epsilon = 0.00001){
    for(int32_t i1 = 0; i1 <s1; i1++) {
        float sum = 0 ;
        for(uint32_t i2 = 0; i2 <s2; i2++){
            sum += input[i1][i2] ;
        }
        float avg = sum/s2 ;
        
        float squaredDiffSum = 0 ;
        for(int32_t i2 = 0; i2 <s2; i2++){
            squaredDiffSum += pow(input[i1][i2] - avg, 2) ;
        }
        float variance = squaredDiffSum/s2 ;
        float std = sqrt(variance + epsilon) ;
        
        for(int32_t i2 = 0; i2 <s2; i2++){
            output[i1][i2] = ((input[i1][i2]-avg)/std) * weight[i2] + bias[i2] ;
        }
    }
}


// void EmbedLayerNormalization(int32_t batch_size, int32_t seq_len, vector<vector<float>> &input_ids, vector<vector<float>> &transformer_wte_weight, vector<vector<float>> &transformer_wpe_weight, vector<float> &transformer_h_ln_1_weight, vector<float> &transformer_h_ln_1_bias, vector<vector<float>> &position_ids, vector<vector<vector<float>>> &EmbedLayerNormalization_output, vector<vector<vector<float>>> &EmbedLayerNormalization_dummy_mask_index, vector<vector<vector<float>>> &EmbedLayerNormalization_embedding_sum, int32_t embedding_len = 768){
    
//     for(int32_t b = 0; b <batch_size; b++){
//         for(int32_t i1 = 0; i1 <seq_len; i1++) {
//             for(int32_t i2 = 0; i2 <embedding_len; i2++){
//                 float x = transformer_wte_weight[input_ids[b][i1]][i2] ;
//                 float y = transformer_wpe_weight[position_ids[b][i1]][i2] ;

//                 EmbedLayerNormalization_embedding_sum[b][i1][i2] = x + y ;
//             }
//         }
        
//         LayerNormalization(seq_len, embedding_len, EmbedLayerNormalization_embedding_sum[b], transformer_h_ln_1_weight, transformer_h_ln_1_bias, EmbedLayerNormalization_output[b]) ;
//     }
    
// }


void EmbedLayerNormalization(int32_t batch_size, int32_t seq_len, vector<vector<vector<float>>> &input, vector<float> &transformer_h_ln_1_weight, vector<float> &transformer_h_ln_1_bias, vector<vector<vector<float>>> &EmbedLayerNormalization_output, vector<vector<vector<float>>> &EmbedLayerNormalization_dummy_mask_index, vector<vector<vector<float>>> &EmbedLayerNormalization_embedding_sum, int32_t embedding_len = 768){
    
    for(int32_t b = 0; b <batch_size; b++){
        for(int32_t i1 = 0; i1 <seq_len; i1++) {
            for(int32_t i2 = 0; i2 <embedding_len; i2++){

                EmbedLayerNormalization_embedding_sum[b][i1][i2] = input[b][i1][i2] ;
            }
        }
        
        LayerNormalization(seq_len, embedding_len, EmbedLayerNormalization_embedding_sum[b], transformer_h_ln_1_weight, transformer_h_ln_1_bias, EmbedLayerNormalization_output[b]) ;
    }
    
}

void SkipLayerNormalization(int32_t batch_size, int32_t seq_len, int32_t embedding_len, vector<vector<vector<float>>> &PreviousLayerNormalizationOutput, vector<vector<vector<float>>> &Attention_matmul_output, vector<float> &transformer_h_ln_weight, vector<float> &transformer_h_ln_bias, vector<float> &transformer_h_attn_c_proj_bias, vector<vector<vector<float>>> &Output1, vector<vector<vector<float>>> &Output2){
    
    for(int32_t b = 0; b <batch_size; b++){
        vector<vector<float>> BiasAdded = make_vector<float>(seq_len, embedding_len);
        GemmAdd(seq_len, embedding_len, Attention_matmul_output[b], transformer_h_attn_c_proj_bias, BiasAdded) ;

        //vector<vector<float>> PrevAdded = make_vector<float>(seq_len, embedding_len);
        ElemWiseAdd(seq_len, embedding_len, PreviousLayerNormalizationOutput[b], BiasAdded, Output2[b]) ;

        //LayerNormalization(seq_len, embedding_len, PrevAdded, transformer_h_ln_weight, transformer_h_ln_bias, Output1[b]) ;
        LayerNormalization(seq_len, embedding_len, Output2[b], transformer_h_ln_weight, transformer_h_ln_bias, Output1[b]) ;
        

//         for(int32_t i1 = 0; i1 <seq_len; i1++) {
//             for(int32_t i2 = 0; i2 <embedding_len; i2++){
//                 Output2[b][i1][i2] = Output1[b][i1][i2] ;
//             }
//         }
    }
}

void Attention_Operation(int32_t s1, int32_t s2, vector<vector<float>> &q, vector<vector<float>> &k, vector<vector<float>> &v, vector<vector<float>> &mask, vector<vector<float>> &output){
    vector<vector<float>> k_transpose = make_vector<float>(s2, s1) ;
    Transpose2T10(s1, s2, k, k_transpose) ;
//     //
//     cout << "&&&&&&&&&&&&8\n" ;
//     PrintMatrix(s2, s1, k_transpose) ;
//     //
    
    vector<vector<float>> temp = make_vector<float>(s1, s1) ;
    MatMul(s1, s2, s1, q, k_transpose, temp) ;
    
//     //
//     cout << "&&&&&&&&&&&&9\n" ;
//     PrintMatrix(s1, s1, temp) ;
//     //
    
    for(int32_t i = 0; i <s1; i++){
            for(int32_t j = 0; j <s1; j++){
                temp[i][j] /= sqrt(s2) ;
            }
    }
    
//     //
//     cout << "&&&&&&&&&&&&10\n" ;
//     PrintMatrix(s1, s1, temp) ;
//     //
    ElemWiseAdd(s1, s1, temp, mask, temp) ;
    
//     //
//     cout << "&&&&&&&&&&&&11\n" ;
//     PrintMatrix(s1, s1, temp) ;
//     //
    
    vector<vector<float>> softmaxed = make_vector<float>(s1, s1) ;
    Softmax2(s1, s1, temp, softmaxed) ;
    
//     //
//     cout << "&&&&&&&&&&&&12\n" ;
//     PrintMatrix(s1, s1, softmaxed) ;
//     //
    
    MatMul(s1, s1, s2, softmaxed, v, output) ;
}

// void Attention(int32_t batch_size, int32_t seq_len, int32_t embedding_len, vector<vector<vector<float>>> &LayerNormalizationOutput, vector<vector<float>> &transformer_h_attn_c_attn_weight, vector<float> &transformer_h_attn_c_attn_bias, vector<vector<float>> &attention_mask, vector<vector<vector<vector<vector<float>>>>> &past, vector<vector<vector<float>>> &AttentionOutput, vector<vector<vector<vector<vector<float>>>>> &present, int32_t num_heads = 12){
void Attention(int32_t batch_size, int32_t seq_len, int32_t embedding_len, vector<vector<vector<float>>> &LayerNormalizationOutput, vector<vector<float>> &transformer_h_attn_c_attn_weight, vector<float> &transformer_h_attn_c_attn_bias, vector<vector<float>> &attention_mask, vector<vector<vector<float>>> &AttentionOutput, int32_t num_heads = 12){
    for(int32_t b = 0; b <batch_size; b++){
        vector<vector<float>> qkv = make_vector<float>(seq_len, 3*embedding_len);
        MatMul(seq_len, embedding_len, 3*embedding_len, LayerNormalizationOutput[b], transformer_h_attn_c_attn_weight, qkv) ;
//         //
//         cout << "&&&&&&&&&&&&1\n" ;
//         PrintMatrix(seq_len, 3*embedding_len, qkv) ;
//         //
        
        GemmAdd(seq_len, 3*embedding_len, qkv, transformer_h_attn_c_attn_bias, qkv) ;
        
        //
        cout << "&&&&&&&&&&&&2\n" ;
        PrintMatrix(seq_len, 3*embedding_len, qkv) ;
        //
        
        vector<vector<float>> causal_mask = make_vector<float>(seq_len, seq_len);
        for(int32_t i = 0; i <seq_len; i++){
            for(int32_t j = 0; j <seq_len; j++){
                if (i<j){
                    causal_mask[i][j] = -10000000000 ;
                }
                else{
                    causal_mask[i][j] = 0 ;
                }
            }
        }
        
//         //
//         cout << "&&&&&&&&&&&&3\n" ;
//         PrintMatrix(seq_len, seq_len, causal_mask) ;
//         //
        
        for(int32_t n = 0; n <num_heads; n++){
            vector<vector<float>> q = make_vector<float>(seq_len, embedding_len/num_heads);
            vector<vector<float>> k = make_vector<float>(seq_len, embedding_len/num_heads);
            vector<vector<float>> v = make_vector<float>(seq_len, embedding_len/num_heads);
            vector<vector<float>> head_output = make_vector<float>(seq_len, embedding_len/num_heads);
            for(int32_t i = 0; i <seq_len; i++){
                for(int32_t j = 0; j <embedding_len/num_heads; j++){
                    q[i][j] = qkv[i][n*64 + j] ;
                    k[i][j] = qkv[i][embedding_len + n*64 + j] ;
                    v[i][j] = qkv[i][2*embedding_len + n*64 + j] ;           
                }
            }
//             //
//             cout << "&&&&&&&&&&&&4\n" ;
//             PrintMatrix(seq_len, embedding_len/num_heads, q) ;
//             //
//             //
//             cout << "&&&&&&&&&&&&5\n" ;
//             PrintMatrix(seq_len, embedding_len/num_heads, k) ;
//             //
//             //
//             cout << "&&&&&&&&&&&&6\n" ;
//             PrintMatrix(seq_len, embedding_len/num_heads, v) ;
//             //
            
            Attention_Operation(seq_len,embedding_len/num_heads,q,k,v,causal_mask,head_output) ;
            
//             //
//             cout << "&&&&&&&&&&&&7\n" ;
//             PrintMatrix(seq_len, embedding_len/num_heads, head_output) ;
//             //
            
            for(int32_t i = 0; i <seq_len; i++){
                for(int32_t j = 0; j <embedding_len/num_heads; j++){
                   AttentionOutput[b][i][n*64 + j] = head_output[i][j] ;            
                }
            }
        }
        
    }
}

// M1D1 = Matrix 1 Dimension 1
void MatMulBatch(int32_t batch_size, int32_t M1D1, int32_t M1D2, int32_t M2D1, int32_t M2D2, vector<vector<vector<float>>> &BatchInp, vector<vector<float>> &MultiplierMat, vector<vector<vector<float>>> &BatchOut){
    
    assert(M1D2==M2D1) ;
    for(int32_t b = 0; b <batch_size; b++){
        MatMul(M1D1, M1D2, M2D2, BatchInp[b], MultiplierMat, BatchOut[b]) ;
    }
    
}
///////////////////////

void ElemWiseAdd(int32_t s1, vector<float> &arr1, vector<float> &arr2, vector<float> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = arr1[i] + arr2[i];
    }
}

void ElemWiseAdd(int32_t s1, int32_t s2, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = arr1[i1][i2] + arr2[i1][i2];
        }
    }
}

void ElemWiseAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i2][i3][i4] = arr1[i1][i2][i3][i4] + arr2[i1][i2][i3][i4];
                }
            }
        }
    }
}

void ElemWiseSub(int32_t s1, vector<float> &arr1, vector<float> &arr2, vector<float> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = arr1[i] - arr2[i];
    }
}

void ElemWiseSub(int32_t s1, int32_t s2, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = arr1[i1][i2] - arr2[i1][i2];
        }
    }
}

void ElemWiseSub(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i2][i3][i4] = arr1[i1][i2][i3][i4] - arr2[i1][i2][i3][i4];
                }
            }
        }
    }
}

void ElemWiseMul(int32_t s1, vector<float> &arr1, vector<float> &arr2, vector<float> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = arr1[i] * arr2[i];
    }
}

void ElemWiseMul(int32_t s1, int32_t s2, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = arr1[i1][i2] * arr2[i1][i2];
        }
    }
}

void ElemWiseMul(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i2][i3][i4] = arr1[i1][i2][i3][i4] * arr2[i1][i2][i3][i4];
                }
            }
        }
    }
}

void ElemWiseDiv(int32_t s1, vector<float> &arr1, vector<float> &arr2, vector<float> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = arr1[i] / arr2[i];
    }
}

void ElemWiseDiv(int32_t s1, int32_t s2, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = arr1[i1][i2] / arr2[i1][i2];
        }
    }
}

void ElemWiseDiv(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i2][i3][i4] = arr1[i1][i2][i3][i4] / arr2[i1][i2][i3][i4];
                }
            }
        }
    }
}

void ElemWiseSqrt(int32_t s1, vector<float> &inArr, vector<float> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = std::sqrt(inArr[i]);
    }
}

void ElemWiseSqrt(int32_t s1, int32_t s2, auto &inArr, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = std::sqrt(inArr[i1][i2]);
        }
    }
}

void ElemWiseSqrt(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i2][i3][i4] = std::sqrt(inArr[i1][i2][i3][i4]);
                }
            }
        }
    }
}

void ElemWisePow(int32_t s1, vector<float> &arr1, vector<float> &arr2, vector<float> &outArr) {
    for (int32_t i = 0; i < s1; i++) {
        outArr[i] = std::pow(arr1[i], arr2[i]);
    }
}

void ElemWisePow(int32_t s1, int32_t s2, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[i1][i2] = std::pow(arr1[i1][i2], arr2[i1][i2]);
        }
    }
}

void ElemWisePow(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &arr1, auto &arr2, auto &outArr) {
    for (int32_t i1 = 0; i1 < s1; i1++) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i2][i3][i4] = std::pow(arr1[i1][i2][i3][i4], arr2[i1][i2][i3][i4]);
                }
            }
        }
    }
}

void Shape(int32_t s1, auto &inArr, auto &outArr) {
    outArr[0] = s1;
}

void Shape(int32_t s1, int32_t s2, auto &inArr, auto &outArr) {
    outArr[0] = s1;
    outArr[1] = s2;
}

void Shape(int32_t s1, int32_t s2, int32_t s3, auto &inArr, auto &outArr) {
    outArr[0] = s1;
    outArr[1] = s2;
    outArr[2] = s3;
}

void Shape(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr,auto &outArr) {
    outArr[0] = s1;
    outArr[1] = s2;
    outArr[2] = s3;
    outArr[3] = s4;
}

void Squeeze(int32_t s1, int32_t s2, int32_t axis, auto &inArr, auto &outArr) {
    if (axis == 0) {
        for (int32_t i1 = 0; i1 < 1; i1++) {
            for (int32_t i2 = 0; i2 < s2; i2++) {
                outArr[i2] = inArr[i1][i2];
            }
        }
    } else {
        for (int32_t i1 = 0; i1 < s1; i1++) {
            for (int32_t i2 = 0; i2 < 1; i2++) {
                outArr[i1] = inArr[i1][i2];
            }
        }
    }
}

void Unsqueeze(int32_t s1, int32_t axis, auto &inArr, auto &outArr) {
    outArr[0] = inArr;
}

void Unsqueeze(int32_t s1, int32_t s2, int32_t axis, auto &inArr, auto &outArr) {
    if (axis == 0) {
        for (int32_t i1 = 0; i1 < 1; i1++) {
            for (int32_t i2 = 0; i2 < s2; i2++) {
                outArr[i1][i2] = inArr[i2];
            }
        }
    } else {
        for (int32_t i1 = 0; i1 < s1; i1++) {
            for (int32_t i2 = 0; i2 < 1; i2++) {
                outArr[i1][i2] = inArr[i1];
            }
        }
    }
}

void ReduceMean(int32_t s1, int32_t s2, int32_t axis, auto &inArr, auto &outArr) {
    if (axis == 0 || axis == -2) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            outArr[0][i2] = 0.0f;
            for (int32_t i1 = 0; i1 < s1; i1++) {
                outArr[0][i2] += inArr[i1][i2];
            }
            outArr[0][i2] /= s1;
        }
    } else {
        for (int32_t i1 = 0; i1 < s1; i1++) {
            outArr[i1][0] = 0.0f;
            for (int32_t i2 = 0; i2 < s2; i2++) {
                outArr[i1][0] += inArr[i1][i2];
            }
            outArr[i1][0] /= s2;
        }
    }
}

void ReduceMean(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t axis, auto &inArr, auto &outArr) {
    if (axis == 0 || axis == -4) {
        for (int32_t i2 = 0; i2 < s2; i2++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (int32_t i4 = 0; i4 < s4; i4++) {
                    outArr[0][i2][i3][i4] = 0.0f;
                    for (int32_t i1 = 0; i1 < s1; i1++) {
                        outArr[0][i2][i3][i4] += inArr[i1][i2][i3][i4];
                    }
                    outArr[0][i2][i3][i4] /= s1;
                }
            }
        }
    } else if (axis == 1 || axis == -3) {
        for (int32_t i1 = 0; i1 < s1; i1++) {
            for (int32_t i3 = 0; i3 < s3; i3++) {
                for (int32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][0][i3][i4] = 0.0f;
                    for (int32_t i2 = 0; i2 < s2; i2++) {
                        outArr[i1][0][i3][i4] += inArr[i1][i2][i3][i4];
                    }
                    outArr[i1][0][i3][i4] /= s2;
                }
            }
        }
    } else if (axis == 2 || axis == -2) {
        for (int32_t i1 = 0; i1 < s1; i1++) {
            for (int32_t i2 = 0; i2 < s2; i2++) {
                for (int32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i2][0][i4] = 0.0f;
                    for (int32_t i3 = 0; i3 < s3; i3++) {
                        outArr[i1][i2][0][i4] += inArr[i1][i2][i3][i4];
                    }
                    outArr[i1][i2][0][i4] /= s3;
                }
            }
        }
    } else {
        for (int32_t i1 = 0; i1 < s1; i1++) {
            for (int32_t i2 = 0; i2 < s2; i2++) {
                for (int32_t i3 = 0; i3 < s3; i3++) {
                    outArr[i1][i2][i3][0] = 0.0f;
                    for (int32_t i4 = 0; i4 < s4; i4++) {
                        outArr[i1][i2][i3][0] += inArr[i1][i2][i3][i4];
                    }
                    outArr[i1][i2][i3][0] /= s4;
                }
            }
        }
    }
}

void ConstantOfShapeI64(int32_t s1, int64_t value, auto &inArr, auto &outArr) {
    for (uint32_t i = 0; i < s1; i++) {
        for (uint32_t i1 = 0; i1 < inArr[i]; i1++) {
            outArr[i1] = value;
        }
    }
}

void NonZero1D(int32_t s1, auto &inArr, auto &outArr) {
    uint32_t i = 0;
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        if (inArr[i1] == 0) {
            continue;
        } else {
            outArr[0][i] = i1;
            i++;
        }
    }
}

void Split2T3(int32_t s1, int32_t s2, int32_t s3, int32_t sp1, int32_t sp2, int32_t sp3, auto &inArr, auto &outArr1, auto &outArr2, auto &outArr3) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            uint32_t i = 0;
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                if (i3 < sp1) {
                    outArr1[i1][i2][i] = inArr[i1][i2][i3];
                    i++;
                } else if (i3 < sp1 + sp2) {
                    if (i3 == sp1) {
                        i  = 0;
                    }
                    outArr2[i1][i2][i] = inArr[i1][i2][i3];
                    i++;
                } else {
                    if (i3 == sp1 + sp2) {
                        i = 0;
                    }
                    outArr3[i1][i2][i] = inArr[i1][i2][i3];
                    i++;
                }
            }
        }
    }
}

void Slice(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t starts, int32_t ends, int32_t axis, int32_t steps, auto &inArr, auto &outArr) {
    uint32_t i = 0;
    if (axis == 0 || axis == -4) {
        for (uint32_t i1 = starts; i1 < ends; i1 += steps) {
            for (uint32_t i2 = 0; i2 < s2; i2++) {
                for (uint32_t i3 = 0; i3 < s3; i3++) {
                    for (uint32_t i4 = 0; i4 < s4; i4++) {
                        outArr[i][i2][i3][i4] = inArr[i1][i2][i3][i4];
                    }
                }
            }
            i++;
        }
    } else if (axis == 1 || axis == -3) {
        for (uint32_t i1 = 0; i1 < s1; i1++) {
            for (uint32_t i2 = starts; i2 < ends; i2 += steps) {
                for (uint32_t i3 = 0; i3 < s3; i3++) {
                    for (uint32_t i4 = 0; i4 < s4; i4++) {
                        outArr[i1][i][i3][i4] = inArr[i1][i2][i3][i4];
                    }
                }
                i++;
            }
        }
    } else if (axis == 2 || axis == -2) {
        for (uint32_t i1 = 0; i1 < s1; i1++) {
            for (uint32_t i2 = 0; i2 < s2; i2++) {
                for (uint32_t i3 = starts; i3 < ends; i3 += steps) {
                    for (uint32_t i4 = 0; i4 < s4; i4++) {
                        outArr[i1][i2][i][i4] = inArr[i1][i2][i3][i4];
                    }
                    i++;
                }
            }
        }
    } else {
        for (uint32_t i1 = 0; i1 < s1; i1++) {
            for (uint32_t i2 = 0; i2 < s2; i2++) {
                for (uint32_t i3 = 0; i3 < s3; i3++) {
                    for (uint32_t i4 = starts; i4 < ends; i4 += steps) {
                        outArr[i1][i2][i3][i] = inArr[i1][i2][i3][i4];
                        i++;
                    }
                }
            }
        }
    }
}

void Gather0(int32_t s1, int32_t s2, int32_t p1, int32_t p2, auto &inArr, auto &indices, auto &outArr) {
    for (uint32_t i1 = 0; i1 < p1; i1++) {
        for (uint32_t i2 = 0; i2 < p2; i2++) {
            uint32_t row_index = indices[i1][i2];
            for (uint32_t j2 = 0; j2 < s2; j2++) {
                outArr[0][i1][j2] = inArr[row_index][j2];
            }
        }
    }
    
}

void ElemWiseActModelVectorMult(int32_t s1, auto &arr1, auto &arr2, auto &outArr) {
    for (uint32_t ii = 0; ii < s1; ii++) {
        outArr[ii] = (arr1[ii] * arr2[ii]);
    }
}

void MaxPool(int32_t N, int32_t C, int32_t H, int32_t W, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t C1, int32_t imgH, int32_t imgW, auto &inArr, auto &outArr) {
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t c = 0; c < C; c++) {
            int32_t leftTopCornerH = (0 - zPadHLeft);
            int32_t extremeRightBottomCornerH = ((imgH - 1) + zPadHRight);
            int32_t ctH = 0;

            while ((((leftTopCornerH + ksizeH) - 1) <= extremeRightBottomCornerH)) {
                int32_t leftTopCornerW = (0 - zPadWLeft);
                int32_t extremeRightBottomCornerW = ((imgW - 1) + zPadWRight);
                int32_t ctW = 0;

                while ((((leftTopCornerW + ksizeW) - 1) <= extremeRightBottomCornerW)) {
                    float maxi = 0.0;

                    if ((((leftTopCornerH < 0) || (leftTopCornerH >= imgH)) || ((leftTopCornerW < 0) || (leftTopCornerW >= imgW)))) {
                        maxi = 0.0;
                    } else {
                        maxi = inArr[n][c][leftTopCornerH][leftTopCornerW];
                    }
                    for (uint32_t fh = 0; fh < ksizeH; fh++) {
                        for (uint32_t fw = 0; fw < ksizeW; fw++) {
                            int32_t curPosH = (leftTopCornerH + fh);
                            int32_t curPosW = (leftTopCornerW + fw);
                            float temp = 0.0;

                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))) {
                                temp = 0.0;
                            } else {
                                temp = inArr[n][c][curPosH][curPosW];
                            }
                            maxi = ((maxi - temp) < 0.0) ? temp : maxi;
                        }
                    }

                    outArr[n][c][ctH][ctW] = maxi;
                    leftTopCornerW = (leftTopCornerW + strideW);
                    ctW = (ctW + 1);
                }

                leftTopCornerH = (leftTopCornerH + strideH);
                ctH = (ctH + 1);
            }
        }
    }
}

void AvgPool(int32_t N, int32_t C, int32_t H, int32_t W, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t C1, int32_t imgH, int32_t imgW, auto &inArr, auto &outArr) {
    int32_t rows = (((N * C) * H) * W);
    auto filterAvg = make_vector<float>(rows);
    int32_t rowIdx = 0;

    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t c = 0; c < C; c++) {
            int32_t leftTopCornerH = (0 - zPadHLeft);
            int32_t extremeRightBottomCornerH = ((imgH - 1) + zPadHRight);
            int32_t ctH = 0;

            while ((((leftTopCornerH + ksizeH) - 1) <= extremeRightBottomCornerH)) {
                int32_t leftTopCornerW = (0 - zPadWLeft);
                int32_t extremeRightBottomCornerW = ((imgW - 1) + zPadWRight);
                int32_t ctW = 0;

                while ((((leftTopCornerW + ksizeW) - 1) <= extremeRightBottomCornerW)) {
                    float curFilterSum = 0.0;

                    for (uint32_t fh = 0; fh < ksizeH; fh++) {
                        for (uint32_t fw = 0; fw < ksizeW; fw++) {
                            int32_t curPosH = (leftTopCornerH + fh);
                            int32_t curPosW = (leftTopCornerW + fw);
                            float temp = 0.0;

                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW)))) {
                                temp = 0.0;
                            } else {
                                temp = inArr[n][c][curPosH][curPosW];
                            }

                            curFilterSum = curFilterSum + temp;
                        }
                    }

                    int32_t ksizeH64 = ksizeH;
                    int32_t ksizeW64 = ksizeW;
                    int32_t filterSz64 = (ksizeH64 * ksizeW64);
                    float curFilterAvg = curFilterSum / (float)filterSz64;
                    filterAvg[rowIdx] = curFilterAvg;
                    rowIdx = (rowIdx + 1);
                    leftTopCornerW = (leftTopCornerW + strideW);
                    ctW = (ctW + 1);
                }

                leftTopCornerH = (leftTopCornerH + strideH);
                ctH = (ctH + 1);
            }
        }
    }

    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t c = 0; c < C; c++) {
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    outArr[n][c][h][w] = filterAvg[((((((n * C) * H) * W) + ((c * H) * W)) + (h * W)) + w)];
                }
            }
        }
    }
}

void Relu(int32_t s1, int32_t s2, auto &inArr, auto &outArr) {
    int32_t size = (s1 * s2);
    auto reshapedInArr = make_vector<float>(size);
    auto reshapedOutArr = make_vector<float>(size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }

    Relu(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Relu(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr) {
    int32_t size = (((s1 * s2) * s3) * s4);

    auto reshapedInArr = make_vector<float>(size);

    auto reshapedOutArr = make_vector<float>(size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    reshapedInArr[linIdx] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }

    Relu(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

// void Sigmoid(int32_t s1, int32_t s2, auto &inArr, auto &outArr)
// {
//     int32_t size = (s1 * s2);

//     auto reshapedInArr = make_vector<float>(size);

//     auto reshapedOutArr = make_vector<float>(size);

//     for (uint32_t i1 = 0; i1 < s1; i1++)
//     {
//         for (uint32_t i2 = 0; i2 < s2; i2++)
//         {
//             int32_t linIdx = ((i1 * s2) + i2);

//             reshapedInArr[linIdx] = inArr[i1][i2];
//         }
//     }
//     Sigmoid(size, reshapedInArr, reshapedOutArr);
//     for (uint32_t i1 = 0; i1 < s1; i1++)
//     {
//         for (uint32_t i2 = 0; i2 < s2; i2++)
//         {
//             int32_t linIdx = ((i1 * s2) + i2);

//             outArr[i1][i2] = reshapedOutArr[linIdx];
//         }
//     }
// }

// void Sigmoid(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr)
// {
//     int32_t size = (((s1 * s2) * s3) * s4);

//     auto reshapedInArr = make_vector<float>(size);

//     auto reshapedOutArr = make_vector<float>(size);

//     for (uint32_t i1 = 0; i1 < s1; i1++)
//     {
//         for (uint32_t i2 = 0; i2 < s2; i2++)
//         {
//             for (uint32_t i3 = 0; i3 < s3; i3++)
//             {
//                 for (uint32_t i4 = 0; i4 < s4; i4++)
//                 {
//                     int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

//                     reshapedInArr[linIdx] = inArr[i1][i2][i3][i4];
//                 }
//             }
//         }
//     }

//     Sigmoid(size, reshapedInArr, reshapedOutArr);

//     for (uint32_t i1 = 0; i1 < s1; i1++)
//     {
//         for (uint32_t i2 = 0; i2 < s2; i2++)
//         {
//             for (uint32_t i3 = 0; i3 < s3; i3++)
//             {
//                 for (uint32_t i4 = 0; i4 < s4; i4++)
//                 {
//                     int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

//                     outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
//                 }
//             }
//         }
//     }
// }

// // tanh(x) = 2 * sigmoid(2 * x) - 1
// void Tanh(int32_t s1, vector<float> &inArr, vector<float> &outArr)
// {

//     const float one = 1.0;
//     const float two = 2.0;

//     // 2 * x
//     auto twice_input = make_vector<float>(s1);
//     for (int i = 0; i < s1; i++)
//     {
//         twice_input[i] = (inArr[i] * two);
//     }

//     // sigmoid(2 * x)
//     auto sigmoid_twice_input = make_vector<float>(s1);
//     Sigmoid(s1, twice_input, sigmoid_twice_input);

//     // tanh(x) = 2 * sigmoid(2 * x) - 1
//     for (int i = 0; i < s1; i++)
//     {
//         outArr[i] = (two * sigmoid_twice_input[i]);
//         outArr[i] = (outArr[i] - one);
//     }
// }

void Tanh(int32_t s1, int32_t s2, auto &inArr, auto &outArr) {
    int32_t size = (s1 * s2);
    auto reshapedInArr = make_vector<float>(size);
    auto reshapedOutArr = make_vector<float>(size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }

    Tanh(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Tanh(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr) {
    int32_t size = (((s1 * s2) * s3) * s4);
    auto reshapedInArr = make_vector<float>(size);
    auto reshapedOutArr = make_vector<float>(size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);
                    reshapedInArr[linIdx] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }

    Tanh(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);
                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

void MatMul2D(int32_t i, int32_t j, int32_t k, auto &A, auto &B, auto &C, bool isA = true) {
    for (uint32_t i1 = 0; i1 < i; i1++) {
        for (uint32_t i2 = 0; i2 < k; i2++) {
            C[i1][i2] = 0.0;

            for (uint32_t i3 = 0; i3 < j; i3++) {
                C[i1][i2] = (C[i1][i2] + (A[i1][i3] * B[i3][i2]));
            }
        }
    }
}

void Gemm(int32_t m, int32_t n, int32_t o, int32_t p, float alpha, float beta, int32_t transA, int32_t transB, int32_t x, int32_t k, auto &A, auto &B, auto &C, auto &output) {
    if (transA) {
        auto tmpA = make_vector<float>(n, m);

        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < n; j++) {
                tmpA[j][i] = A[i][j];
            }
        }

        A = tmpA;
        swap(m, n);
    }

    if (transB) {
        auto tmpB = make_vector<float>(p, o);

        for (uint32_t i = 0; i < o; i++) {
            for (uint32_t j = 0; j < p; j++) {
                tmpB[j][i] = B[i][j];
            }
        }

        B = tmpB;
        swap(o, p);
    }

    auto tmp = make_vector<float>(x, k);
    MatMul(m, n, p, A, B, tmp);
    GemmAdd(x, k, tmp, C, output);

    // for (uint32_t i0 = 0; i0 < x; i0++)
    // {
    //     for (uint32_t i1 = 0; i1 < k; i1++)
    //     {
    //         output[i0][i1] = Add(tmp[i0][i1], C[i1]);
    //     }
    // }
}

void Flatten(int32_t x, int32_t y, auto &A, auto &output) {
    (&output) = *reinterpret_cast<int(*)[x][y]>(A);
}

void FusedBatchNorm4411(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &multArr, auto &biasArr, auto &outputArr) {
    int32_t inpSize = (((s1 * s2) * s3) * s4);
    auto inArrReshaped = make_vector<float>(inpSize);
    auto multArrReshaped = make_vector<float>(inpSize);
    auto multExprAns = make_vector<float>(inpSize);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s4; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s2; i4++) {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);
                    inArrReshaped[linIdx] = inArr[i1][i4][i2][i3];
                    multArrReshaped[linIdx] = multArr[i4];
                }
            }
        }
    }

    ElemWiseActModelVectorMult(inpSize, inArrReshaped, multArrReshaped, multExprAns);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s4; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s2; i4++) {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);
                    outputArr[i1][i4][i2][i3] = (multExprAns[linIdx] + biasArr[i4]);
                }
            }
        }
    }

    delete &inArrReshaped;
    delete &multArrReshaped;
    delete &multExprAns;
    delete &biasArr;
}

void BatchNormalization(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &multArr, auto &biasArr, auto &outputArr) {
    FusedBatchNorm4411(s1, s2, s3, s4, inArr, multArr, biasArr, outputArr);
}

void Conv2DReshapeMatMulOPGroup(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, int32_t g, int32_t G, vector<vector<float>> &inputArr, vector<vector<vector<vector<float>>>> &outputArr) {
    int32_t COG = (CO / G);
    int32_t startCO = (g * COG);

    for (uint32_t co = 0; co < COG; co++) {
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t h = 0; h < finalH; h++) {
                for (uint32_t w = 0; w < finalW; w++) {
                    outputArr[n][(co + startCO)][h][w] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)];
                }
            }
        }
    }
}

void Conv2DReshapeFilterGroup(int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t g, int32_t G, vector<vector<vector<vector<float>>>> &inputArr, vector<vector<float>> &outputArr) {
    int32_t CIG = (CI / G);
    int32_t COG = (CO / G);
    int32_t startCO = (g * COG);

    for (uint32_t co = 0; co < COG; co++) {
        for (uint32_t fh = 0; fh < FH; fh++) {
            for (uint32_t fw = 0; fw < FW; fw++) {
                for (uint32_t ci = 0; ci < CIG; ci++) {
                    int32_t linIdx = ((((fh * FW) * CIG) + (fw * CIG)) + ci);
                    outputArr[co][linIdx] = inputArr[(co + startCO)][ci][fh][fw];
                }
            }
        }
    }
}

void Conv2DReshapeInputGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t g, int32_t G, int32_t RRows, int32_t RCols, vector<vector<vector<vector<float>>>> &inputArr, vector<vector<float>> &outputArr) {
    int32_t linIdxFilterMult = 0;
    int32_t CIG = (CI / G);

    for (uint32_t n = 0; n < N; n++) {
        int32_t leftTopCornerH = (0 - zPadHLeft);
        int32_t extremeRightBottomCornerH = ((H - 1) + zPadHRight);

        while ((((leftTopCornerH + FH) - 1) <= extremeRightBottomCornerH)) {
            int32_t leftTopCornerW = (0 - zPadWLeft);
            int32_t extremeRightBottomCornerW = ((W - 1) + zPadWRight);

            while ((((leftTopCornerW + FW) - 1) <= extremeRightBottomCornerW)) {
                for (uint32_t fh = 0; fh < FH; fh++) {
                    for (uint32_t fw = 0; fw < FW; fw++) {
                        int32_t curPosH = (leftTopCornerH + fh);
                        int32_t curPosW = (leftTopCornerW + fw);
                        float val = 0.0;
                        int32_t startCI = (g * CIG);

                        for (uint32_t ci = 0; ci < CIG; ci++) {
                            if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))) {
                                val = 0.0;
                            } else {
                                val = inputArr[n][(ci + startCI)][curPosH][curPosW];
                            }

                            outputArr[((((fh * FW) * CIG) + (fw * CIG)) + ci)][linIdxFilterMult] = val;
                        }
                    }
                }

                linIdxFilterMult = (linIdxFilterMult + 1);
                leftTopCornerW = (leftTopCornerW + strideW);
            }

            leftTopCornerH = (leftTopCornerH + strideH);
        }
    }
}

void Conv2DGroupWrapper(
    int32_t N, int32_t CI, int32_t H, int32_t W, int32_t FH, int32_t FW, int32_t CO,
    int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
    int32_t strideH, int32_t strideW, int32_t G,
    vector<vector<vector<vector<float>>>> &inputArr,
    vector<vector<vector<vector<float>>>> &filterArr,
    vector<vector<vector<vector<float>>>> &outArr) {
    int32_t CIG = (CI / G);
    int32_t reshapedFilterRows = (CO / G);
    int32_t reshapedFilterCols = ((FH * FW) * CIG);
    int32_t reshapedIPRows = ((FH * FW) * CIG);
    int32_t outH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) + 1);
    int32_t outW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) + 1);
    int32_t reshapedIPCols = ((N * outH) * outW);

    for (uint32_t g = 0; g < G; g++) {
        vector<vector<float>> inputReshaped = make_vector<float>(reshapedIPRows, reshapedIPCols);
        vector<vector<float>> filterReshaped = make_vector<float>(reshapedFilterRows, reshapedFilterCols);
        vector<vector<float>> matmulOP = make_vector<float>(reshapedFilterRows, reshapedIPCols);

        Conv2DReshapeFilterGroup(FH, FW, CI, CO, g, G, filterArr, filterReshaped);
        Conv2DReshapeInputGroup(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, g, G, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
        MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP);
        Conv2DReshapeMatMulOPGroup(N, outH, outW, CO, g, G, matmulOP, outArr);
    }
}

void ConvAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4,
             vector<vector<vector<vector<float>>>> &inArr,
             vector<float> &biasArr,
             vector<vector<vector<vector<float>>>> &outArr) {
    int sz;
    sz = s1 * s2 * s3 * s4;
    vector<float> arr1 = make_vector_float(ALICE, sz);
    vector<float> arr2 = make_vector_float(ALICE, sz);
    vector<float> out = make_vector_float(ALICE, sz);

    for (int i1 = 0; i1 < s1; i1++) {
        for (int i2 = 0; i2 < s3; i2++) {
            for (int i3 = 0; i3 < s4; i3++) {
                for (int i4 = 0; i4 < s2; i4++) {
                    arr1[i1*s3*s4*s2 + i2*s4*s2 + i3*s2 + i4] = inArr[i1][i4][i2][i3] ;
					arr2[i1*s3*s4*s2 + i2*s4*s2 + i3*s2 + i4] = biasArr[i4] ;
                }
            }
        }
    }

    ElemWiseAdd(sz, arr1, arr2, out);

    for (int i1 = 0; i1 < s1; i1++) {
        for (int i2 = 0; i2 < s3; i2++) {
            for (int i3 = 0; i3 < s4; i3++) {
                for (int i4 = 0; i4 < s2; i4++) {
                    outArr[i1][i4][i2][i3] = out[i1*s3*s4*s2 + i2*s4*s2 + i3*s2 + i4];
                }
            }
        }
    }
}

void Concat1T44(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                }
            }
        }
    }
}

void Transpose2T10(int32_t s1, int32_t s2, auto &inArr, auto &outArr) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            outArr[i2][i1] = inArr[i1][i2];
        }
    }
}

void Transpose4T0213(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i3][i2][i4] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }
}

void Transpose4T0231(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    outArr[i1][i3][i4][i2] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }
}

void Concat2T111(int32_t s1, int32_t inp1s1, auto &inp1, int32_t inp2s1, auto &inp2, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        if (i1 < inp1s1) {
            outp[i1] = inp1[i1];
        } else {
            outp[i1] = inp2[(i1 - inp1s1)];
        }
    }
}

void Concat2T222(int32_t s1, int32_t s2, int32_t inp1s1, int32_t inp1s2, auto &inp1, int32_t inp2s1, int32_t inp2s2, auto &inp2, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            if (axis == 0) {
                if ((i1 < inp1s1)) {
                    outp[i1][i2] = inp1[i1][i2];
                } else {
                    outp[i1][i2] = inp2[(i1 - inp1s1)][i2];
                }
            } else {
                if ((i2 < inp1s2)) {
                    outp[i1][i2] = inp1[i1][i2];
                } else {
                    outp[i1][i2] = inp2[i1][(i2 - inp1s2)];
                }
            }
        }
    }
}

void Concat2T444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if (axis == 0) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat3T1111(int32_t s1, int32_t inp1s1, auto &inp1, int32_t inp2s1, auto &inp2, int32_t inp3s1, auto &inp3, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        if (i1 < inp1s1) {
            outp[i1] = inp1[i1];
        } else if (i1 < inp1s1 + inp2s1) {
            outp[i1] = inp2[(i1 - inp1s1)];
        } else {
            outp[i1] = inp3[(i1 - inp1s1 - inp2s1)];
        }
    }
}

void Concat3T4444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat4T11111(int32_t s1, int32_t inp1s1, auto &inp1, int32_t inp2s1, auto &inp2, int32_t inp3s1, auto &inp3, int32_t inp4s1, auto &inp4, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        if (i1 < inp1s1) {
            outp[i1] = inp1[i1];
        } else if (i1 < inp1s1 + inp2s1) {
            outp[i1] = inp2[i1 - inp1s1];
        } else if (i1 < inp1s1 + inp2s1 + inp3s1) {
            outp[i1] = inp3[i1 - inp1s1 - inp2s1];
        } else {
            outp[i1] = inp3[i1 - inp1s1 - inp2s1 - inp3s1];
        }
    }
}

void Concat4T44444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat5T444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat6T4444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat7T44444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat8T444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat9T4444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat10T44444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat11T444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat12T4444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat13T44444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat14T444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat15T4444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat16T44444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat17T444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat18T4444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat19T44444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1))) {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            } else {
                                                                                                outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2))) {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                } else {
                                                                                                    outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    } else {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    } else {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat20T444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1))) {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            } else {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1))) {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                } else {
                                                                                                    outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2))) {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                } else {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2))) {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    } else {
                                                                                                        outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    } else {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        } else {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    } else {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        } else {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat21T4444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1))) {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            } else {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1))) {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                } else {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1))) {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    } else {
                                                                                                        outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2))) {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                } else {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2))) {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2))) {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        } else {
                                                                                                            outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    } else {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        } else {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            } else {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    } else {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        } else {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            } else {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat22T44444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1))) {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            } else {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1))) {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                } else {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1))) {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1))) {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        } else {
                                                                                                            outp[i1][i2][i3][i4] = inp22[(((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1)][i2][i3][i4];
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2))) {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                } else {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2))) {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2))) {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        } else {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2))) {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            } else {
                                                                                                                outp[i1][i2][i3][i4] = inp22[i1][(((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2)][i3][i4];
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    } else {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        } else {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            } else {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                } else {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][i2][(((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3)][i4];
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    } else {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        } else {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            } else {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                } else {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][i2][i3][(((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4)];
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat23T444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, auto &inp23, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1))) {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            } else {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1))) {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                } else {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1))) {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1))) {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        } else {
                                                                                                            if ((i1 < (((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1))) {
                                                                                                                outp[i1][i2][i3][i4] = inp22[(((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1)][i2][i3][i4];
                                                                                                            } else {
                                                                                                                outp[i1][i2][i3][i4] = inp23[((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1)][i2][i3][i4];
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2))) {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                } else {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2))) {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2))) {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        } else {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2))) {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            } else {
                                                                                                                if ((i2 < (((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][(((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2)][i3][i4];
                                                                                                                } else {
                                                                                                                    outp[i1][i2][i3][i4] = inp23[i1][((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2)][i3][i4];
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    } else {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        } else {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            } else {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                } else {
                                                                                                                    if ((i3 < (((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][(((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3)][i4];
                                                                                                                    } else {
                                                                                                                        outp[i1][i2][i3][i4] = inp23[i1][i2][((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3)][i4];
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    } else {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        } else {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            } else {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                } else {
                                                                                                                    if ((i4 < (((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][i3][(((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4)];
                                                                                                                    } else {
                                                                                                                        outp[i1][i2][i3][i4] = inp23[i1][i2][i3][((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4)];
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat24T4444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, auto &inp23, int32_t inp24s1, int32_t inp24s2, int32_t inp24s3, int32_t inp24s4, auto &inp24, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1))) {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            } else {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1))) {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                } else {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1))) {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1))) {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        } else {
                                                                                                            if ((i1 < (((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1))) {
                                                                                                                outp[i1][i2][i3][i4] = inp22[(((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1)][i2][i3][i4];
                                                                                                            } else {
                                                                                                                if ((i1 < ((((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1) + inp23s1))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp23[((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1)][i2][i3][i4];
                                                                                                                } else {
                                                                                                                    outp[i1][i2][i3][i4] = inp24[(((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1) - inp23s1)][i2][i3][i4];
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2))) {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                } else {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2))) {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2))) {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        } else {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2))) {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            } else {
                                                                                                                if ((i2 < (((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][(((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2)][i3][i4];
                                                                                                                } else {
                                                                                                                    if ((i2 < ((((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2) + inp23s2))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp23[i1][((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2)][i3][i4];
                                                                                                                    } else {
                                                                                                                        outp[i1][i2][i3][i4] = inp24[i1][(((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2) - inp23s2)][i3][i4];
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    } else {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        } else {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            } else {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                } else {
                                                                                                                    if ((i3 < (((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][(((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3)][i4];
                                                                                                                    } else {
                                                                                                                        if ((i3 < ((((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3) + inp23s3))) {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3)][i4];
                                                                                                                        } else {
                                                                                                                            outp[i1][i2][i3][i4] = inp24[i1][i2][(((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3) - inp23s3)][i4];
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    } else {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        } else {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            } else {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                } else {
                                                                                                                    if ((i4 < (((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][i3][(((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4)];
                                                                                                                    } else {
                                                                                                                        if ((i4 < ((((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4) + inp23s4))) {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][i3][((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4)];
                                                                                                                        } else {
                                                                                                                            outp[i1][i2][i3][i4] = inp24[i1][i2][i3][(((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4) - inp23s4)];
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Concat25T44444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, auto &inp23, int32_t inp24s1, int32_t inp24s2, int32_t inp24s3, int32_t inp24s4, auto &inp24, int32_t inp25s1, int32_t inp25s2, int32_t inp25s3, int32_t inp25s4, auto &inp25, int32_t axis, auto &outp) {
    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            for (uint32_t i3 = 0; i3 < s3; i3++) {
                for (uint32_t i4 = 0; i4 < s4; i4++) {
                    if ((axis == 0)) {
                        if ((i1 < inp1s1)) {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        } else {
                            if ((i1 < (inp1s1 + inp2s1))) {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            } else {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1))) {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                } else {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1))) {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    } else {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1))) {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        } else {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1))) {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            } else {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1))) {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                } else {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1))) {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    } else {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1))) {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        } else {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1))) {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            } else {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1))) {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                } else {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1))) {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    } else {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1))) {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        } else {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1))) {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            } else {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1))) {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                } else {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1))) {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    } else {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1))) {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        } else {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1))) {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            } else {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1))) {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                } else {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1))) {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1))) {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        } else {
                                                                                                            if ((i1 < (((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1))) {
                                                                                                                outp[i1][i2][i3][i4] = inp22[(((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1)][i2][i3][i4];
                                                                                                            } else {
                                                                                                                if ((i1 < ((((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1) + inp23s1))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp23[((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1)][i2][i3][i4];
                                                                                                                } else {
                                                                                                                    if ((i1 < (((((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1) + inp23s1) + inp24s1))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp24[(((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1) - inp23s1)][i2][i3][i4];
                                                                                                                    } else {
                                                                                                                        outp[i1][i2][i3][i4] = inp25[((((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1) - inp23s1) - inp24s1)][i2][i3][i4];
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if ((axis == 1)) {
                            if ((i2 < inp1s2)) {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            } else {
                                if ((i2 < (inp1s2 + inp2s2))) {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                } else {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2))) {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    } else {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2))) {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        } else {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2))) {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            } else {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2))) {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                } else {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2))) {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    } else {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2))) {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        } else {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2))) {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            } else {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2))) {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                } else {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2))) {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    } else {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2))) {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        } else {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2))) {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            } else {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2))) {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                } else {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2))) {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    } else {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2))) {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        } else {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2))) {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            } else {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2))) {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                } else {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2))) {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    } else {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2))) {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        } else {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2))) {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            } else {
                                                                                                                if ((i2 < (((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][(((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2)][i3][i4];
                                                                                                                } else {
                                                                                                                    if ((i2 < ((((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2) + inp23s2))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp23[i1][((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2)][i3][i4];
                                                                                                                    } else {
                                                                                                                        if ((i2 < (((((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2) + inp23s2) + inp24s2))) {
                                                                                                                            outp[i1][i2][i3][i4] = inp24[i1][(((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2) - inp23s2)][i3][i4];
                                                                                                                        } else {
                                                                                                                            outp[i1][i2][i3][i4] = inp25[i1][((((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2) - inp23s2) - inp24s2)][i3][i4];
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if ((axis == 2)) {
                                if ((i3 < inp1s3)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i3 < (inp1s3 + inp2s3))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    } else {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        } else {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            } else {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                } else {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    } else {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        } else {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            } else {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                } else {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    } else {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        } else {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            } else {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                } else {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    } else {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        } else {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            } else {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                } else {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    } else {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        } else {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            } else {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                } else {
                                                                                                                    if ((i3 < (((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][(((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3)][i4];
                                                                                                                    } else {
                                                                                                                        if ((i3 < ((((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3) + inp23s3))) {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3)][i4];
                                                                                                                        } else {
                                                                                                                            if ((i3 < (((((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3) + inp23s3) + inp24s3))) {
                                                                                                                                outp[i1][i2][i3][i4] = inp24[i1][i2][(((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3) - inp23s3)][i4];
                                                                                                                            } else {
                                                                                                                                outp[i1][i2][i3][i4] = inp25[i1][i2][((((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3) - inp23s3) - inp24s3)][i4];
                                                                                                                            }
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if ((i4 < inp1s4)) {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                } else {
                                    if ((i4 < (inp1s4 + inp2s4))) {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    } else {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4))) {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        } else {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4))) {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            } else {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4))) {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                } else {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4))) {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    } else {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4))) {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        } else {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4))) {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            } else {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4))) {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                } else {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4))) {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    } else {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4))) {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        } else {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4))) {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            } else {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4))) {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                } else {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4))) {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    } else {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4))) {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        } else {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4))) {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            } else {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4))) {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                } else {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4))) {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    } else {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4))) {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        } else {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4))) {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            } else {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4))) {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                } else {
                                                                                                                    if ((i4 < (((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4))) {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][i3][(((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4)];
                                                                                                                    } else {
                                                                                                                        if ((i4 < ((((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4) + inp23s4))) {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][i3][((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4)];
                                                                                                                        } else {
                                                                                                                            if ((i4 < (((((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4) + inp23s4) + inp24s4))) {
                                                                                                                                outp[i1][i2][i3][i4] = inp24[i1][i2][i3][(((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4) - inp23s4)];
                                                                                                                            } else {
                                                                                                                                outp[i1][i2][i3][i4] = inp25[i1][i2][i3][((((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4) - inp23s4) - inp24s4)];
                                                                                                                            }
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
