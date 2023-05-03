#include "inout.cpp"
#include "concat.cpp"

extern float intToFloat(int32_t m);
extern void Softmax2(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr);
extern void Ln(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr);
extern void getOutDer(int32_t s1, int32_t s2, vector<vector<FPArray>> &batchSoft, vector<vector<FPArray>> &lab, vector<vector<FPArray>> &der);
extern void MatMul(int32_t s1, int32_t s2, int32_t s3, vector<vector<FPArray>> &mat1, vector<vector<FPArray>> &mat2, vector<vector<FPArray>> &mat3);
extern void GemmAdd(int32_t s1, int32_t s2, vector<vector<FPArray>> &prod, vector<FPArray> &bias, vector<vector<FPArray>> &out);
extern void dotProduct2(int32_t s1, int32_t s2, vector<vector<FPArray>> &arr1, vector<vector<FPArray>> &arr2, vector<FPArray> &outArr);
extern void Relu(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr, vector<BoolArray> &hotArr);
extern void LeakyRelu(int32_t s1, float alpha, vector<FPArray> &inArr, vector<FPArray> &outArr, vector<BoolArray> &hotArr);
extern void getBiasDer(int32_t s1, int32_t s2, vector<vector<FPArray>> &der, vector<FPArray> &biasDer);
extern void IfElse(int32_t s1, vector<FPArray> &dat, vector<BoolArray> &hot, vector<FPArray> &out, bool flip);
extern void updateWeights(int32_t s, float lr, vector<FPArray> &bias, vector<FPArray> &der);
extern void getLoss(int32_t m, vector<FPArray> &lossTerms, vector<FPArray> &loss);
extern void computeMSELoss(int32_t m, int32_t s, vector<vector<FPArray>> &target, vector<vector<FPArray>> &fwdOut, vector<FPArray> &loss);
extern void Sigmoid(int32_t s1, vector<FPArray> &inArr, vector<FPArray> &outArr);

void ElemWiseAdd(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
void ElemWiseSub(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
void ElemWiseMul(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr);
void ElemWiseDiv(int32_t s1, vector<FPArray> &arr1, vector<FPArray> &arr2, vector<FPArray> &outArr) ;

FPArray __public_float_to_arithmetic(float f, int party = ALICE)
{
    float *_dummy = new float[1];
    _dummy[0] = f;
    FPArray _ret = __fp_op->input<float>(party, 1, _dummy, __m_bits, __e_bits);
    delete[] _dummy;
    return _ret;
}

FPArray __public_float_to_baba(float f, int party = ALICE)
{
    float *_dummy = new float[1];
    _dummy[0] = f;
    FPArray _ret = __fp_op->input(party, 1, _dummy);
    delete[] _dummy;
    return _ret;
}

// Gemm

void Gemm(int32_t m, int32_t n, int32_t o, int32_t p, float alpha, float beta, int32_t transA, int32_t transB, int32_t x, int32_t k, vector<vector<FPArray>> &A, vector<vector<FPArray>> &B, vector<FPArray> &C, vector<vector<FPArray>> &output)
{
    if (transA)
    {
        vector<vector<FPArray>> tmpA = make_vector_float(ALICE, n, m);
        for (uint32_t i = 0; i < m; i++)
        {
            for (uint32_t j = 0; j < n; j++)
            {
                tmpA[j][i] = A[i][j];
            }
        }
        A = tmpA;
        swap(m, n);
    }
    if (transB)
    {
        vector<vector<FPArray>> tmpB = make_vector_float(ALICE, p, o);
        for (uint32_t i = 0; i < o; i++)
        {
            for (uint32_t j = 0; j < p; j++)
            {
                tmpB[j][i] = B[i][j];
            }
        }
        B = tmpB;
        swap(o, p);
    }

    vector<vector<FPArray>> tmp = make_vector_float(ALICE, x, k);

    // Performing the matrix multiplication followed by the bias addition
    MatMul(m, n, p, A, B, tmp);
    GemmAdd(x,k,tmp,C,output);
}

// Relu2 to Relu 4

void Relu(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr)
{
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }
    Relu_nomask(size, reshapedInArr, reshapedOutArr);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Relu(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr, vector<vector<vector<vector<FPArray>>>> &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    reshapedInArr[linIdx] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }

    Relu_nomask(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

// LeakyRelu2 and LeakyRelu4


void Leaky_Relu(int32_t s1, int32_t s2, float alpha, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr)
{
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }
    Leaky_Relu_nomask(size, alpha, reshapedInArr, reshapedOutArr);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Leaky_Relu(int32_t s1, int32_t s2, int32_t s3, int32_t s4, float alpha, vector<vector<vector<vector<FPArray>>>> &inArr, vector<vector<vector<vector<FPArray>>>> &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    reshapedInArr[linIdx] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }
    Leaky_Relu_nomask(size, alpha, reshapedInArr, reshapedOutArr);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

// Sigmoid2 and Sigmoid4

void Sigmoid(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr)
{
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }
    Sigmoid(size, reshapedInArr, reshapedOutArr);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Sigmoid(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr, vector<vector<vector<vector<FPArray>>>> &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    reshapedInArr[linIdx] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }

    Sigmoid(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

// Tanh2 and Tanh4

void Tanh(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr, vector<vector<FPArray>> &outArr)
{
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }
    Tanh(size, reshapedInArr, reshapedOutArr);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Tanh(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr, vector<vector<vector<vector<FPArray>>>> &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    vector<FPArray> reshapedInArr = make_vector_float(ALICE, size);

    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    reshapedInArr[linIdx] = inArr[i1][i2][i3][i4];
                }
            }
        }
    }

    Tanh(size, reshapedInArr, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

// Elementary elemwise operations

void ElemWiseAdd2(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr1, vector<vector<FPArray>> &inArr2, vector<vector<FPArray>> &outArr) {
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr1[linIdx] = inArr1[i1][i2];
            reshapedInArr2[linIdx] = inArr2[i1][i2];
        }
    }

    ElemWiseAdd(size, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void ElemWiseSub2(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr1, vector<vector<FPArray>> &inArr2, vector<vector<FPArray>> &outArr) {
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr1[linIdx] = inArr1[i1][i2];
            reshapedInArr2[linIdx] = inArr2[i1][i2];
        }
    }

    ElemWiseSub(size, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void ElemWiseMul2(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr1, vector<vector<FPArray>> &inArr2, vector<vector<FPArray>> &outArr) {
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr1[linIdx] = inArr1[i1][i2];
            reshapedInArr2[linIdx] = inArr2[i1][i2];
        }
    }

    ElemWiseMul(size, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void ElemWiseDiv2(int32_t s1, int32_t s2, vector<vector<FPArray>> &inArr1, vector<vector<FPArray>> &inArr2, vector<vector<FPArray>> &outArr) {
    int32_t size = (s1 * s2);

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, size);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            reshapedInArr1[linIdx] = inArr1[i1][i2];
            reshapedInArr2[linIdx] = inArr2[i1][i2];
        }
    }

    ElemWiseDiv(size, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++) {
        for (uint32_t i2 = 0; i2 < s2; i2++) {
            int32_t linIdx = ((i1 * s2) + i2);
            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void ElemWiseAdd4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr1, vector<vector<vector<vector<FPArray>>>> &inArr2, vector<vector<vector<vector<FPArray>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, sz);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    reshapedInArr1[linIdx] = inArr1[i1][i2][i3][i4];
                    reshapedInArr2[linIdx] = inArr2[i1][i2][i3][i4];
                }
            }
        }
    }

    ElemWiseAdd(sz, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

void ElemWiseSub4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr1, vector<vector<vector<vector<FPArray>>>> &inArr2, vector<vector<vector<vector<FPArray>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, sz);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    reshapedInArr1[linIdx] = inArr1[i1][i2][i3][i4];
                    reshapedInArr2[linIdx] = inArr2[i1][i2][i3][i4];
                }
            }
        }
    }

    ElemWiseSub(sz, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

void ElemWiseMul4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr1, vector<vector<vector<vector<FPArray>>>> &inArr2, vector<vector<vector<vector<FPArray>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, sz);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    reshapedInArr1[linIdx] = inArr1[i1][i2][i3][i4];
                    reshapedInArr2[linIdx] = inArr2[i1][i2][i3][i4];
                }
            }
        }
    }

    ElemWiseMul(sz, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

void ElemWiseDiv4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr1, vector<vector<vector<vector<FPArray>>>> &inArr2, vector<vector<vector<vector<FPArray>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<FPArray> reshapedInArr1 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedInArr2 = make_vector_float(ALICE, sz);
    vector<FPArray> reshapedOutArr = make_vector_float(ALICE, sz);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    reshapedInArr1[linIdx] = inArr1[i1][i2][i3][i4];
                    reshapedInArr2[linIdx] = inArr2[i1][i2][i3][i4];
                }
            }
        }
    }

    ElemWiseDiv(sz, reshapedInArr1, reshapedInArr2, reshapedOutArr);

    for (uint32_t i1 = 0; i1 < s1; i1++){
        for (uint32_t i2 = 0; i2 < s2; i2++){
            for (uint32_t i3 = 0; i3 < s3; i3++){
                for (uint32_t i4 = 0; i4 < s4; i4++){
                    int32_t linIdx = (((i1*s2+i2)*s3+i3)*s4+i4);
                    outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx];
                }
            }
        }
    }
}

void BatchNormalization(
	int32_t N, int32_t C, int32_t H, int32_t W,
	vector<vector<vector<vector<FPArray>>>> &inArr,
	vector<FPArray> &multArr,
	vector<FPArray> &biasArr,
	vector<vector<vector<vector<FPArray>>>> &outArr
) {
    vector<vector<vector<vector<FPArray>>>> mult_expanded = make_vector_float(ALICE, N, C, H, W);
    vector<vector<vector<vector<FPArray>>>> bias_expanded = make_vector_float(ALICE, N, C, H, W);

    for (int32_t i = 0; i < N; i++){
        for (int32_t j = 0; j < C; j++){
            for (int32_t k = 0; k < H; k++){
                for (int32_t l = 0; l < W; l++){
                    mult_expanded[i][j][k][l] = multArr[j];
                    bias_expanded[i][j][k][l] = biasArr[j];
                }
            }
        }
    }

    ElemWiseMul4(N, C, H, W, inArr, mult_expanded, outArr);
    ElemWiseAdd4(N, C, H, W, outArr, bias_expanded, outArr);
}
