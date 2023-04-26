#include <iostream>
#include "link_secfloat.cpp"

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

// void Flatten(int32_t x, int32_t y, vector<vector<FPArray>> &A, vector<FPArray> &output)
// {
//     (&output) = *reinterpret_cast<int(*)[x][y]>(A);
// }

void BatchNormalization(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector<vector<vector<vector<FPArray>>>> &inArr, vector<FPArray> &multArr, vector<FPArray> &biasArr, vector<vector<vector<vector<FPArray>>>> &outputArr)
{
    int32_t inpSize = (((s1 * s2) * s3) * s4);

    vector<FPArray> inArrReshaped = make_vector_float(ALICE, inpSize);

    vector<FPArray> multArrReshaped = make_vector_float(ALICE, inpSize);

    vector<FPArray> multExprAns = make_vector_float(ALICE, inpSize);

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

                    outputArr[i1][i4][i2][i3] = Add(multExprAns[linIdx], biasArr[i4]);
                }
            }
        }
    }
    delete &inArrReshaped;
    delete &multArrReshaped;
    delete &multExprAns;
    delete &biasArr;
}

void __onnxbridge_Conv2DReshapeMatMulOPGroup(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, int32_t g, int32_t G, vector<vector<FPArray>> &inputArr, vector<vector<vector<vector<FPArray>>>> &outputArr)
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

void __onnxbridge_Conv2DReshapeFilterGroup(int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t g, int32_t G, vector<vector<vector<vector<FPArray>>>> &inputArr, vector<vector<FPArray>> &outputArr)
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

void __onnxbridge_Conv2DReshapeInputGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t g, int32_t G, int32_t RRows, int32_t RCols, vector<vector<vector<vector<FPArray>>>> &inputArr, vector<vector<FPArray>> &outputArr)
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
                        FPArray val = __public_float_to_arithmetic(0., ALICE);
                        int32_t startCI = (g * CIG);

                        for (uint32_t ci = 0; ci < CIG; ci++)
                        {
                            if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W))))
                            {
                                val = __public_float_to_arithmetic(0., ALICE);
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

void __onnxbridge_Conv2DGroupWrapper(
    int32_t N, int32_t CI, int32_t H, int32_t W, int32_t FH, int32_t FW, int32_t CO,
    int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
    int32_t strideH, int32_t strideW, int32_t G,
    vector<vector<vector<vector<FPArray>>>> &inputArr,
    vector<vector<vector<vector<FPArray>>>> &filterArr,
    vector<vector<vector<vector<FPArray>>>> &outArr)
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
        vector<vector<FPArray>> inputReshaped = make_vector_float(ALICE, reshapedIPRows, reshapedIPCols);
        vector<vector<FPArray>> filterReshaped = make_vector_float(ALICE, reshapedFilterRows, reshapedFilterCols);
        vector<vector<FPArray>> matmulOP = make_vector_float(ALICE, reshapedFilterRows, reshapedIPCols);

        __onnxbridge_Conv2DReshapeFilterGroup(FH, FW, CI, CO, g, G, filterArr, filterReshaped);
        __onnxbridge_Conv2DReshapeInputGroup(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, g, G, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
        MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP);
        __onnxbridge_Conv2DReshapeMatMulOPGroup(N, outH, outW, CO, g, G, matmulOP, outArr);
    }
}

void __onnxbridge_ConvAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4,
             vector<vector<vector<vector<FPArray>>>> &inArr,
             vector<FPArray> &biasArr,
             vector<vector<vector<vector<FPArray>>>> &outArr)
{
    int m_bits, e_bits;
    int sz;

    m_bits = inArr[0][0][0][0].m_bits;
    e_bits = inArr[0][0][0][0].e_bits;
    sz = s1 * s2 * s3 * s4;

    vector<FPArray> arr1 = make_vector_float(ALICE, sz);
    vector<FPArray> arr2 = make_vector_float(ALICE, sz);
    vector<FPArray> out = make_vector_float(ALICE, sz);

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