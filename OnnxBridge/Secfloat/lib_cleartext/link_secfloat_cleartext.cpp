#include "cleartext_inout.cpp"
#include "cleartext_common.cpp"


void Relu(int32_t s1, int32_t s2, vector<vector<float>> &inArr, vector<vector<float>> &outArr)
{
    int32_t size = (s1 * s2);

    vector<float> reshapedInArr = make_vector<float>(size);

    vector<float> reshapedOutArr = make_vector<float>(size);

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

void Relu(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    vector<float> reshapedInArr = make_vector<float>(size);

    vector<float> reshapedOutArr = make_vector<float>(size);

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

void Tanh(int32_t s1, int32_t s2, vector<vector<float>> &inArr, vector<vector<float>> &outArr)
{
    int32_t size = (s1 * s2);

    vector<float> reshapedInArr = make_vector<float>(size);

    vector<float> reshapedOutArr = make_vector<float>(size);

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

void Tanh(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    vector<float> reshapedInArr = make_vector<float>(size);

    vector<float> reshapedOutArr = make_vector<float>(size);

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


void ElemWiseAdd2(int32_t s1, int32_t s2, vector<vector<float>> &inArr1, vector<vector<float>> &inArr2, vector<vector<float>> &outArr) {
    int32_t size = (s1 * s2);

    vector<float> reshapedInArr1 = make_vector<float>(size);
    vector<float> reshapedInArr2 = make_vector<float>(size);
    vector<float> reshapedOutArr = make_vector<float>(size);

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

void ElemWiseSub2(int32_t s1, int32_t s2, vector<vector<float>> &inArr1, vector<vector<float>> &inArr2, vector<vector<float>> &outArr) {
    int32_t size = (s1 * s2);

    vector<float> reshapedInArr1 = make_vector<float>(size);
    vector<float> reshapedInArr2 = make_vector<float>(size);
    vector<float> reshapedOutArr = make_vector<float>(size);

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

void ElemWiseMul2(int32_t s1, int32_t s2, vector<vector<float>> &inArr1, vector<vector<float>> &inArr2, vector<vector<float>> &outArr) {
    int32_t size = (s1 * s2);

    vector<float> reshapedInArr1 = make_vector<float>(size);
    vector<float> reshapedInArr2 = make_vector<float>(size);
    vector<float> reshapedOutArr = make_vector<float>(size);

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

void ElemWiseDiv2(int32_t s1, int32_t s2, vector<vector<float>> &inArr1, vector<vector<float>> &inArr2, vector<vector<float>> &outArr) {
    int32_t size = (s1 * s2);

    vector<float> reshapedInArr1 = make_vector<float>(size);
    vector<float> reshapedInArr2 = make_vector<float>(size);
    vector<float> reshapedOutArr = make_vector<float>(size);

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

void ElemWiseAdd4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<float>>>> &inArr1, vector<vector<vector<vector<float>>>> &inArr2, vector<vector<vector<vector<float>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<float> reshapedInArr1 = make_vector<float>(sz);
    vector<float> reshapedInArr2 = make_vector<float>(sz);
    vector<float> reshapedOutArr = make_vector<float>(sz);

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

void ElemWiseSub4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<float>>>> &inArr1, vector<vector<vector<vector<float>>>> &inArr2, vector<vector<vector<vector<float>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<float> reshapedInArr1 = make_vector<float>(sz);
    vector<float> reshapedInArr2 = make_vector<float>(sz);
    vector<float> reshapedOutArr = make_vector<float>(sz);

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

void ElemWiseMul4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<float>>>> &inArr1, vector<vector<vector<vector<float>>>> &inArr2, vector<vector<vector<vector<float>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<float> reshapedInArr1 = make_vector<float>(sz);
    vector<float> reshapedInArr2 = make_vector<float>(sz);
    vector<float> reshapedOutArr = make_vector<float>(sz);

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

void ElemWiseDiv4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<float>>>> &inArr1, vector<vector<vector<vector<float>>>> &inArr2, vector<vector<vector<vector<float>>>> &outArr){
    int32_t sz = s1*s2*s3*s4 ;

    vector<float> reshapedInArr1 = make_vector<float>(sz);
    vector<float> reshapedInArr2 = make_vector<float>(sz);
    vector<float> reshapedOutArr = make_vector<float>(sz);

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

void Gemm(int32_t m, int32_t n, int32_t o, int32_t p, float alpha, float beta, int32_t transA, int32_t transB, int32_t x, int32_t k, vector<vector<float>> &A, vector<vector<float>> &B, vector<float> &C, vector<vector<float>> &output)
{
    if (transA)
    {
        vector<vector<float>> tmpA = make_vector<float>(n, m);
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
        vector<vector<float>> tmpB = make_vector<float>(p, o);
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

    vector<vector<float>> tmp = make_vector<float>(x, k);
    MatMul(m, n, p, A, B, tmp);

    GemmAdd(x, k, tmp, C, output);
}

void BatchNormalization(int32_t N, int32_t C, int32_t H, int32_t W, vector<vector<vector<vector<float>>>> &inArr, vector<float> &multArr, vector<float> &biasArr, vector<vector<vector<vector<float>>>> &outArr)
{
    vector<vector<vector<vector<float>>>> mult_expanded = make_vector_float(ALICE, N, C, H, W);
    vector<vector<vector<vector<float>>>> bias_expanded = make_vector_float(ALICE, N, C, H, W);

    for (int32_t i = 0; i < N; i++)
    {
        for (int32_t j = 0; j < C; j++)
        {
            for (int32_t k = 0; k < H; k++)
            {
                for (int32_t l = 0; l < W; l++)
                {
                    mult_expanded[i][j][k][l] = multArr[j];
                    bias_expanded[i][j][k][l] = biasArr[j];
                }
            }
        }
    }

    ElemWiseMul4(N, C, H, W, inArr, mult_expanded, outArr);
    ElemWiseAdd4(N, C, H, W, outArr, bias_expanded, outArr);
}