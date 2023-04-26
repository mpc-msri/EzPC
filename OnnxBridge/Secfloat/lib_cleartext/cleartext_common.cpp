#include "cleartext_library_float.h"
#include "cleartext_inout.cpp"
#include "cleartext_concat.cpp"

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

void ElemWiseAdd(int32_t s1, vector<float> &arr1, vector<float> &arr2, vector<float> &outArr)
{
    for (int i = 0; i < s1; i++)
        outArr[i] = arr1[i] + arr2[i];
}

void ElemWiseActModelVectorMult(int32_t s1, vector<float> &arr1, vector<float> &arr2, vector<float> &outArr)
{
    for (uint32_t ii = 0; ii < s1; ii++)
    {
        outArr[ii] = (arr1[ii] * arr2[ii]);
    }
}

void MaxPool(int32_t N, int32_t C, int32_t H, int32_t W, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t C1, int32_t imgH, int32_t imgW, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    for (uint32_t n = 0; n < N; n++)
    {
        for (uint32_t c = 0; c < C; c++)
        {
            int32_t leftTopCornerH = (0 - zPadHLeft);

            int32_t extremeRightBottomCornerH = ((imgH - 1) + zPadHRight);

            int32_t ctH = 0;

            while ((((leftTopCornerH + ksizeH) - 1) <= extremeRightBottomCornerH))
            {
                int32_t leftTopCornerW = (0 - zPadWLeft);

                int32_t extremeRightBottomCornerW = ((imgW - 1) + zPadWRight);

                int32_t ctW = 0;

                while ((((leftTopCornerW + ksizeW) - 1) <= extremeRightBottomCornerW))
                {
                    float maxi = 0.0;

                    if ((((leftTopCornerH < 0) || (leftTopCornerH >= imgH)) || ((leftTopCornerW < 0) || (leftTopCornerW >= imgW))))
                    {
                        maxi = 0.0;
                    }
                    else
                    {
                        maxi = inArr[n][c][leftTopCornerH][leftTopCornerW];
                    }
                    for (uint32_t fh = 0; fh < ksizeH; fh++)
                    {
                        for (uint32_t fw = 0; fw < ksizeW; fw++)
                        {
                            int32_t curPosH = (leftTopCornerH + fh);

                            int32_t curPosW = (leftTopCornerW + fw);

                            float temp = 0.0;

                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                            {
                                temp = 0.0;
                            }
                            else
                            {
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

void AvgPool(int32_t N, int32_t C, int32_t H, int32_t W, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t C1, int32_t imgH, int32_t imgW, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    int32_t rows = (((N * C) * H) * W);

    vector<float> filterAvg = make_vector<float>(rows);

    int32_t rowIdx = 0;

    for (uint32_t n = 0; n < N; n++)
    {
        for (uint32_t c = 0; c < C; c++)
        {
            int32_t leftTopCornerH = (0 - zPadHLeft);

            int32_t extremeRightBottomCornerH = ((imgH - 1) + zPadHRight);

            int32_t ctH = 0;

            while ((((leftTopCornerH + ksizeH) - 1) <= extremeRightBottomCornerH))
            {
                int32_t leftTopCornerW = (0 - zPadWLeft);

                int32_t extremeRightBottomCornerW = ((imgW - 1) + zPadWRight);

                int32_t ctW = 0;

                while ((((leftTopCornerW + ksizeW) - 1) <= extremeRightBottomCornerW))
                {
                    float curFilterSum = 0.0;

                    for (uint32_t fh = 0; fh < ksizeH; fh++)
                    {
                        for (uint32_t fw = 0; fw < ksizeW; fw++)
                        {
                            int32_t curPosH = (leftTopCornerH + fh);

                            int32_t curPosW = (leftTopCornerW + fw);

                            float temp = 0.0;

                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                            {
                                temp = 0.0;
                            }
                            else
                            {
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
    for (uint32_t n = 0; n < N; n++)
    {
        for (uint32_t c = 0; c < C; c++)
        {
            for (uint32_t h = 0; h < H; h++)
            {
                for (uint32_t w = 0; w < W; w++)
                {
                    outArr[n][c][h][w] = filterAvg[((((((n * C) * H) * W) + ((c * H) * W)) + (h * W)) + w)];
                }
            }
        }
    }
}

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
    Relu(size, reshapedInArr, reshapedOutArr);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Relu(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<float>>> &inArr, vector<vector<vector<float>>> &outArr)
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

    Relu(size, reshapedInArr, reshapedOutArr);

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

void Tanh(int32_t s1, int32_t s2, vector<float> &inArr, vector<float> &outArr)
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

void MatMul2D(int32_t i, int32_t j, int32_t k, vector<vector<float>> &A, vector<vector<float>> &B, vector<vector<float>> &C)
{
    for (uint32_t i1 = 0; i1 < i; i1++)
    {
        for (uint32_t i2 = 0; i2 < k; i2++)
        {
            C[i1][i2] = 0.0;

            for (uint32_t i3 = 0; i3 < j; i3++)
            {
                C[i1][i2] = (C[i1][i2] + (A[i1][i3] * B[i3][i2]));
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

// void Flatten(int32_t x, int32_t y, auto &A, auto &output)
// {
//     (&output) = *reinterpret_cast<int(*)[x][y]>(A);
// }

void BatchNormalization(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<float>>>> &inArr, vector<float> &multArr, vector<float> &biasArr, vector<vector<vector<vector<float>>>> &outputArr)
{
    int32_t inpSize = (((s1 * s2) * s3) * s4);

    vector<float> inArrReshaped = make_vector<float>(inpSize);

    vector<float> multArrReshaped = make_vector<float>(inpSize);

    vector<float> multExprAns = make_vector<float>(inpSize);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s4; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s2; i4++)
                {
                    int32_t linIdx = ((((((i1 * s2) * s3) * s4) + ((i2 * s3) * s4)) + (i3 * s4)) + i4);

                    inArrReshaped[linIdx] = inArr[i1][i4][i2][i3];

                    multArrReshaped[linIdx] = multArr[i4];
                }
            }
        }
    }
    ElemWiseActModelVectorMult(inpSize, inArrReshaped, multArrReshaped, multExprAns);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s4; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s2; i4++)
                {
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

void Conv2DReshapeMatMulOPGroup(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, int32_t g, int32_t G, vector<vector<float>> &inputArr, vector<vector<vector<vector<float>>>> &outputArr)
{
    int32_t COG = (CO / G);
    int32_t startCO = (g * COG);
    for (uint32_t co = 0; co < COG; co++)
    {
        for (uint32_t n = 0; n < N; n++)
        {
            for (uint32_t h = 0; h < finalH; h++)
            {
                for (uint32_t w = 0; w < finalW; w++)
                {
                    outputArr[n][(co + startCO)][h][w] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)];
                }
            }
        }
    }
}

void Conv2DReshapeFilterGroup(int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t g, int32_t G, vector<vector<vector<vector<float>>>> &inputArr, vector<vector<float>> &outputArr)
{
    int32_t CIG = (CI / G);
    int32_t COG = (CO / G);
    int32_t startCO = (g * COG);

    for (uint32_t co = 0; co < COG; co++)
    {
        for (uint32_t fh = 0; fh < FH; fh++)
        {
            for (uint32_t fw = 0; fw < FW; fw++)
            {
                for (uint32_t ci = 0; ci < CIG; ci++)
                {
                    int32_t linIdx = ((((fh * FW) * CIG) + (fw * CIG)) + ci);
                    outputArr[co][linIdx] = inputArr[(co + startCO)][ci][fh][fw];
                }
            }
        }
    }
}

void Conv2DReshapeInputGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t g, int32_t G, int32_t RRows, int32_t RCols, vector<vector<vector<vector<float>>>> &inputArr, vector<vector<float>> &outputArr)
{
    int32_t linIdxFilterMult = 0;
    int32_t CIG = (CI / G);

    for (uint32_t n = 0; n < N; n++)
    {
        int32_t leftTopCornerH = (0 - zPadHLeft);
        int32_t extremeRightBottomCornerH = ((H - 1) + zPadHRight);

        while ((((leftTopCornerH + FH) - 1) <= extremeRightBottomCornerH))
        {
            int32_t leftTopCornerW = (0 - zPadWLeft);
            int32_t extremeRightBottomCornerW = ((W - 1) + zPadWRight);

            while ((((leftTopCornerW + FW) - 1) <= extremeRightBottomCornerW))
            {
                for (uint32_t fh = 0; fh < FH; fh++)
                {
                    for (uint32_t fw = 0; fw < FW; fw++)
                    {

                        int32_t curPosH = (leftTopCornerH + fh);
                        int32_t curPosW = (leftTopCornerW + fw);
                        float val = 0.0;
                        int32_t startCI = (g * CIG);

                        for (uint32_t ci = 0; ci < CIG; ci++)
                        {
                            if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W))))
                            {
                                val = 0.0;
                            }
                            else
                            {
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
    vector<vector<vector<vector<float>>>> &outArr)
{

    int32_t CIG = (CI / G);
    int32_t reshapedFilterRows = (CO / G);
    int32_t reshapedFilterCols = ((FH * FW) * CIG);
    int32_t reshapedIPRows = ((FH * FW) * CIG);
    int32_t outH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) + 1);
    int32_t outW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) + 1);
    int32_t reshapedIPCols = ((N * outH) * outW);

    for (uint32_t g = 0; g < G; g++)
    {
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
             vector<vector<vector<vector<float>>>> &outArr)
{
    int sz;

    sz = s1 * s2 * s3 * s4;

    vector<float> arr1 = make_vector_float(ALICE, sz);
    vector<float> arr2 = make_vector_float(ALICE, sz);
    vector<float> out = make_vector_float(ALICE, sz);

    for (int i1 = 0; i1 < s1; i1++)
    {
        for (int i2 = 0; i2 < s3; i2++)
        {
            for (int i3 = 0; i3 < s4; i3++)
            {
                for (int i4 = 0; i4 < s2; i4++)
                {
                    arr1[i1*s3*s4*s2 + i2*s4*s2 + i3*s2 + i4] = inArr[i1][i4][i2][i3] ;
					arr2[i1*s3*s4*s2 + i2*s4*s2 + i3*s2 + i4] = biasArr[i4] ;
                }
            }
        }
    }

    ElemWiseAdd(sz, arr1, arr2, out);

    for (int i1 = 0; i1 < s1; i1++)
    {
        for (int i2 = 0; i2 < s3; i2++)
        {
            for (int i3 = 0; i3 < s4; i3++)
            {
                for (int i4 = 0; i4 < s2; i4++)
                {
                    outArr[i1][i4][i2][i3] = out[i1*s3*s4*s2 + i2*s4*s2 + i3*s2 + i4];
                }
            }
        }
    }
}