#include <iostream>
#include "link_secfloat.cpp"

void MatMul2D(int32_t i, int32_t j, int32_t k, auto &A, auto &B, auto &C, bool isA = true)
{
    for (uint32_t i1 = 0; i1 < i; i1++)
    {
        for (uint32_t i2 = 0; i2 < k; i2++)
        {
            C[i1][i2] = __public_float_to_baba(0., ALICE);

            for (uint32_t i3 = 0; i3 < j; i3++)
            {
                C[i1][i2] = Add(C[i1][i2], Mul(A[i1][i3], B[i3][i2]));
            }
        }
    }
}

void Gemm(int32_t m, int32_t n, int32_t o, int32_t p, float alpha, float beta, int32_t transA, int32_t transB, int32_t x, int32_t k, auto &A, auto &B, auto &C, auto &output)
{
    if (transA)
    {
        auto tmpA = make_vector_float(ALICE, n, m);
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
        auto tmpB = make_vector_float(ALICE, p, o);
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

    auto tmp = make_vector_float(ALICE, x, k);
    MatMul(m, n, p, A, B, tmp);


    GemmAdd(x,k,tmp,C,output);

    // for (uint32_t i0 = 0; i0 < x; i0++)
    // {
    //     for (uint32_t i1 = 0; i1 < k; i1++)
    //     {
    //         output[i0][i1] = Add(tmp[i0][i1], C[i1]);
    //     }
    // }
}

void Flatten(int32_t x, int32_t y, auto &A, auto &output)
{
    (&output) = *reinterpret_cast<int(*)[x][y]>(A);
}

void FusedBatchNorm4411(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &multArr, auto &biasArr, auto &outputArr)
{
    int32_t inpSize = (((s1 * s2) * s3) * s4);

    auto inArrReshaped = make_vector_float(ALICE, inpSize);

    auto multArrReshaped = make_vector_float(ALICE, inpSize);

    auto multExprAns = make_vector_float(ALICE, inpSize);

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

void BatchNormalization(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &multArr, auto &biasArr, auto &outputArr)
{
    FusedBatchNorm4411(s1, s2, s3, s4, inArr, multArr, biasArr, outputArr);
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

void Concat1T44(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                }
            }
        }
    }
}

void Concat2T222(int32_t s1, int32_t s2, int32_t inp1s1, int32_t inp1s2, auto &inp1, int32_t inp2s1, int32_t inp2s2, auto &inp2, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            if ((axis == 0))
            {
                if ((i1 < inp1s1))
                {
                    outp[i1][i2] = inp1[i1][i2];
                }
                else
                {
                    outp[i1][i2] = inp2[(i1 - inp1s1)][i2];
                }
            }
            else
            {
                if ((i2 < inp1s2))
                {
                    outp[i1][i2] = inp1[i1][i2];
                }
                else
                {
                    outp[i1][i2] = inp2[i1][(i2 - inp1s2)];
                }
            }
        }
    }
}

void Concat2T444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
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

void Concat3T4444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                            }
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                }
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                    }
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
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

void Concat4T44444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                }
                            }
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                    }
                                }
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
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

void Concat5T444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
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

void Concat6T4444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
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

void Concat7T44444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
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

void Concat8T444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
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

void Concat9T4444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
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

void Concat10T44444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
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

void Concat11T444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
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

void Concat12T4444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
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

void Concat13T44444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
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

void Concat14T444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
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

void Concat15T4444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
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

void Concat16T44444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
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

void Concat17T444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
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

void Concat18T4444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
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

void Concat19T44444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    }
                                                                                                    else
                                                                                                    {
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

void Concat20T444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        }
                                                                                                        else
                                                                                                        {
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

void Concat21T4444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            }
                                                                                                            else
                                                                                                            {
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

void Concat22T44444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                }
                                                                                                                else
                                                                                                                {
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

void Concat23T444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, auto &inp23, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i1 < (((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp22[(((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1)][i2][i3][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i2 < (((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][(((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2)][i3][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i3 < (((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][(((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3)][i4];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i4 < (((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][i3][(((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4)];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
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

void Concat24T4444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, auto &inp23, int32_t inp24s1, int32_t inp24s2, int32_t inp24s3, int32_t inp24s4, auto &inp24, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i1 < (((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp22[(((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1)][i2][i3][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i1 < ((((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1) + inp23s1)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp23[((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1)][i2][i3][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i2 < (((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][(((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2)][i3][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i2 < ((((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2) + inp23s2)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp23[i1][((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2)][i3][i4];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i3 < (((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][(((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3)][i4];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
                                                                                                                        if ((i3 < ((((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3) + inp23s3)))
                                                                                                                        {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3)][i4];
                                                                                                                        }
                                                                                                                        else
                                                                                                                        {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i4 < (((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][i3][(((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4)];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
                                                                                                                        if ((i4 < ((((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4) + inp23s4)))
                                                                                                                        {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][i3][((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4)];
                                                                                                                        }
                                                                                                                        else
                                                                                                                        {
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

void Concat25T44444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, auto &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, auto &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, auto &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, auto &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, auto &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, auto &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, auto &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, auto &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, auto &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, auto &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, auto &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, auto &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, auto &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, auto &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, auto &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, auto &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, auto &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, auto &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, auto &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, auto &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, auto &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, auto &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, auto &inp23, int32_t inp24s1, int32_t inp24s2, int32_t inp24s3, int32_t inp24s4, auto &inp24, int32_t inp25s1, int32_t inp25s2, int32_t inp25s3, int32_t inp25s4, auto &inp25, int32_t axis, auto &outp)
{
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            for (uint32_t i3 = 0; i3 < s3; i3++)
            {
                for (uint32_t i4 = 0; i4 < s4; i4++)
                {
                    if ((axis == 0))
                    {
                        if ((i1 < inp1s1))
                        {
                            outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                        }
                        else
                        {
                            if ((i1 < (inp1s1 + inp2s1)))
                            {
                                outp[i1][i2][i3][i4] = inp2[(i1 - inp1s1)][i2][i3][i4];
                            }
                            else
                            {
                                if ((i1 < ((inp1s1 + inp2s1) + inp3s1)))
                                {
                                    outp[i1][i2][i3][i4] = inp3[((i1 - inp1s1) - inp2s1)][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i1 < (((inp1s1 + inp2s1) + inp3s1) + inp4s1)))
                                    {
                                        outp[i1][i2][i3][i4] = inp4[(((i1 - inp1s1) - inp2s1) - inp3s1)][i2][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i1 < ((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1)))
                                        {
                                            outp[i1][i2][i3][i4] = inp5[((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1)][i2][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i1 < (((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1)))
                                            {
                                                outp[i1][i2][i3][i4] = inp6[(((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1)][i2][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i1 < ((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp7[((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1)][i2][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i1 < (((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp8[(((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1)][i2][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i1 < ((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp9[((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1)][i2][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i1 < (((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp10[(((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1)][i2][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i1 < ((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp11[((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1)][i2][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i1 < (((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp12[(((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1)][i2][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i1 < ((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp13[((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1)][i2][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i1 < (((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp14[(((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1)][i2][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i1 < ((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp15[((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1)][i2][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i1 < (((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp16[(((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1)][i2][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i1 < ((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp17[((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1)][i2][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i1 < (((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp18[(((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1)][i2][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i1 < ((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp19[((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1)][i2][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i1 < (((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp20[(((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1)][i2][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i1 < ((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp21[((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1)][i2][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i1 < (((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp22[(((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1)][i2][i3][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i1 < ((((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1) + inp23s1)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp23[((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1)][i2][i3][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i1 < (((((((((((((((((((((((inp1s1 + inp2s1) + inp3s1) + inp4s1) + inp5s1) + inp6s1) + inp7s1) + inp8s1) + inp9s1) + inp10s1) + inp11s1) + inp12s1) + inp13s1) + inp14s1) + inp15s1) + inp16s1) + inp17s1) + inp18s1) + inp19s1) + inp20s1) + inp21s1) + inp22s1) + inp23s1) + inp24s1)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp24[(((((((((((((((((((((((i1 - inp1s1) - inp2s1) - inp3s1) - inp4s1) - inp5s1) - inp6s1) - inp7s1) - inp8s1) - inp9s1) - inp10s1) - inp11s1) - inp12s1) - inp13s1) - inp14s1) - inp15s1) - inp16s1) - inp17s1) - inp18s1) - inp19s1) - inp20s1) - inp21s1) - inp22s1) - inp23s1)][i2][i3][i4];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
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
                    }
                    else
                    {
                        if ((axis == 1))
                        {
                            if ((i2 < inp1s2))
                            {
                                outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                            }
                            else
                            {
                                if ((i2 < (inp1s2 + inp2s2)))
                                {
                                    outp[i1][i2][i3][i4] = inp2[i1][(i2 - inp1s2)][i3][i4];
                                }
                                else
                                {
                                    if ((i2 < ((inp1s2 + inp2s2) + inp3s2)))
                                    {
                                        outp[i1][i2][i3][i4] = inp3[i1][((i2 - inp1s2) - inp2s2)][i3][i4];
                                    }
                                    else
                                    {
                                        if ((i2 < (((inp1s2 + inp2s2) + inp3s2) + inp4s2)))
                                        {
                                            outp[i1][i2][i3][i4] = inp4[i1][(((i2 - inp1s2) - inp2s2) - inp3s2)][i3][i4];
                                        }
                                        else
                                        {
                                            if ((i2 < ((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2)))
                                            {
                                                outp[i1][i2][i3][i4] = inp5[i1][((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2)][i3][i4];
                                            }
                                            else
                                            {
                                                if ((i2 < (((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp6[i1][(((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2)][i3][i4];
                                                }
                                                else
                                                {
                                                    if ((i2 < ((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp7[i1][((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2)][i3][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i2 < (((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp8[i1][(((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2)][i3][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i2 < ((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp9[i1][((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2)][i3][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i2 < (((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp10[i1][(((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2)][i3][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i2 < ((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp11[i1][((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2)][i3][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i2 < (((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp12[i1][(((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2)][i3][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i2 < ((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp13[i1][((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2)][i3][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i2 < (((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp14[i1][(((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2)][i3][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i2 < ((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp15[i1][((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2)][i3][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i2 < (((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp16[i1][(((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2)][i3][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i2 < ((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp17[i1][((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2)][i3][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i2 < (((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp18[i1][(((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2)][i3][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i2 < ((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp19[i1][((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2)][i3][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i2 < (((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp20[i1][(((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2)][i3][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i2 < ((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp21[i1][((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2)][i3][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i2 < (((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp22[i1][(((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2)][i3][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i2 < ((((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2) + inp23s2)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp23[i1][((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2)][i3][i4];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
                                                                                                                        if ((i2 < (((((((((((((((((((((((inp1s2 + inp2s2) + inp3s2) + inp4s2) + inp5s2) + inp6s2) + inp7s2) + inp8s2) + inp9s2) + inp10s2) + inp11s2) + inp12s2) + inp13s2) + inp14s2) + inp15s2) + inp16s2) + inp17s2) + inp18s2) + inp19s2) + inp20s2) + inp21s2) + inp22s2) + inp23s2) + inp24s2)))
                                                                                                                        {
                                                                                                                            outp[i1][i2][i3][i4] = inp24[i1][(((((((((((((((((((((((i2 - inp1s2) - inp2s2) - inp3s2) - inp4s2) - inp5s2) - inp6s2) - inp7s2) - inp8s2) - inp9s2) - inp10s2) - inp11s2) - inp12s2) - inp13s2) - inp14s2) - inp15s2) - inp16s2) - inp17s2) - inp18s2) - inp19s2) - inp20s2) - inp21s2) - inp22s2) - inp23s2)][i3][i4];
                                                                                                                        }
                                                                                                                        else
                                                                                                                        {
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
                        }
                        else
                        {
                            if ((axis == 2))
                            {
                                if ((i3 < inp1s3))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i3 < (inp1s3 + inp2s3)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][(i3 - inp1s3)][i4];
                                    }
                                    else
                                    {
                                        if ((i3 < ((inp1s3 + inp2s3) + inp3s3)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][((i3 - inp1s3) - inp2s3)][i4];
                                        }
                                        else
                                        {
                                            if ((i3 < (((inp1s3 + inp2s3) + inp3s3) + inp4s3)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][(((i3 - inp1s3) - inp2s3) - inp3s3)][i4];
                                            }
                                            else
                                            {
                                                if ((i3 < ((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3)][i4];
                                                }
                                                else
                                                {
                                                    if ((i3 < (((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][(((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3)][i4];
                                                    }
                                                    else
                                                    {
                                                        if ((i3 < ((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3)][i4];
                                                        }
                                                        else
                                                        {
                                                            if ((i3 < (((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][(((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3)][i4];
                                                            }
                                                            else
                                                            {
                                                                if ((i3 < ((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3)][i4];
                                                                }
                                                                else
                                                                {
                                                                    if ((i3 < (((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][(((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3)][i4];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i3 < ((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3)][i4];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i3 < (((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][(((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3)][i4];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i3 < ((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3)][i4];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i3 < (((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][(((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3)][i4];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i3 < ((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3)][i4];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i3 < (((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][(((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3)][i4];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i3 < ((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3)][i4];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i3 < (((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][(((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3)][i4];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i3 < ((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3)][i4];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i3 < (((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][(((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3)][i4];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i3 < ((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3)][i4];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i3 < (((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][(((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3)][i4];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
                                                                                                                        if ((i3 < ((((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3) + inp23s3)))
                                                                                                                        {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3)][i4];
                                                                                                                        }
                                                                                                                        else
                                                                                                                        {
                                                                                                                            if ((i3 < (((((((((((((((((((((((inp1s3 + inp2s3) + inp3s3) + inp4s3) + inp5s3) + inp6s3) + inp7s3) + inp8s3) + inp9s3) + inp10s3) + inp11s3) + inp12s3) + inp13s3) + inp14s3) + inp15s3) + inp16s3) + inp17s3) + inp18s3) + inp19s3) + inp20s3) + inp21s3) + inp22s3) + inp23s3) + inp24s3)))
                                                                                                                            {
                                                                                                                                outp[i1][i2][i3][i4] = inp24[i1][i2][(((((((((((((((((((((((i3 - inp1s3) - inp2s3) - inp3s3) - inp4s3) - inp5s3) - inp6s3) - inp7s3) - inp8s3) - inp9s3) - inp10s3) - inp11s3) - inp12s3) - inp13s3) - inp14s3) - inp15s3) - inp16s3) - inp17s3) - inp18s3) - inp19s3) - inp20s3) - inp21s3) - inp22s3) - inp23s3)][i4];
                                                                                                                            }
                                                                                                                            else
                                                                                                                            {
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
                            }
                            else
                            {
                                if ((i4 < inp1s4))
                                {
                                    outp[i1][i2][i3][i4] = inp1[i1][i2][i3][i4];
                                }
                                else
                                {
                                    if ((i4 < (inp1s4 + inp2s4)))
                                    {
                                        outp[i1][i2][i3][i4] = inp2[i1][i2][i3][(i4 - inp1s4)];
                                    }
                                    else
                                    {
                                        if ((i4 < ((inp1s4 + inp2s4) + inp3s4)))
                                        {
                                            outp[i1][i2][i3][i4] = inp3[i1][i2][i3][((i4 - inp1s4) - inp2s4)];
                                        }
                                        else
                                        {
                                            if ((i4 < (((inp1s4 + inp2s4) + inp3s4) + inp4s4)))
                                            {
                                                outp[i1][i2][i3][i4] = inp4[i1][i2][i3][(((i4 - inp1s4) - inp2s4) - inp3s4)];
                                            }
                                            else
                                            {
                                                if ((i4 < ((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4)))
                                                {
                                                    outp[i1][i2][i3][i4] = inp5[i1][i2][i3][((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4)];
                                                }
                                                else
                                                {
                                                    if ((i4 < (((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4)))
                                                    {
                                                        outp[i1][i2][i3][i4] = inp6[i1][i2][i3][(((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4)];
                                                    }
                                                    else
                                                    {
                                                        if ((i4 < ((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4)))
                                                        {
                                                            outp[i1][i2][i3][i4] = inp7[i1][i2][i3][((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4)];
                                                        }
                                                        else
                                                        {
                                                            if ((i4 < (((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4)))
                                                            {
                                                                outp[i1][i2][i3][i4] = inp8[i1][i2][i3][(((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4)];
                                                            }
                                                            else
                                                            {
                                                                if ((i4 < ((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4)))
                                                                {
                                                                    outp[i1][i2][i3][i4] = inp9[i1][i2][i3][((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4)];
                                                                }
                                                                else
                                                                {
                                                                    if ((i4 < (((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4)))
                                                                    {
                                                                        outp[i1][i2][i3][i4] = inp10[i1][i2][i3][(((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4)];
                                                                    }
                                                                    else
                                                                    {
                                                                        if ((i4 < ((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4)))
                                                                        {
                                                                            outp[i1][i2][i3][i4] = inp11[i1][i2][i3][((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4)];
                                                                        }
                                                                        else
                                                                        {
                                                                            if ((i4 < (((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4)))
                                                                            {
                                                                                outp[i1][i2][i3][i4] = inp12[i1][i2][i3][(((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4)];
                                                                            }
                                                                            else
                                                                            {
                                                                                if ((i4 < ((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4)))
                                                                                {
                                                                                    outp[i1][i2][i3][i4] = inp13[i1][i2][i3][((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4)];
                                                                                }
                                                                                else
                                                                                {
                                                                                    if ((i4 < (((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4)))
                                                                                    {
                                                                                        outp[i1][i2][i3][i4] = inp14[i1][i2][i3][(((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4)];
                                                                                    }
                                                                                    else
                                                                                    {
                                                                                        if ((i4 < ((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4)))
                                                                                        {
                                                                                            outp[i1][i2][i3][i4] = inp15[i1][i2][i3][((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4)];
                                                                                        }
                                                                                        else
                                                                                        {
                                                                                            if ((i4 < (((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4)))
                                                                                            {
                                                                                                outp[i1][i2][i3][i4] = inp16[i1][i2][i3][(((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4)];
                                                                                            }
                                                                                            else
                                                                                            {
                                                                                                if ((i4 < ((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4)))
                                                                                                {
                                                                                                    outp[i1][i2][i3][i4] = inp17[i1][i2][i3][((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4)];
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                    if ((i4 < (((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4)))
                                                                                                    {
                                                                                                        outp[i1][i2][i3][i4] = inp18[i1][i2][i3][(((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4)];
                                                                                                    }
                                                                                                    else
                                                                                                    {
                                                                                                        if ((i4 < ((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4)))
                                                                                                        {
                                                                                                            outp[i1][i2][i3][i4] = inp19[i1][i2][i3][((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4)];
                                                                                                        }
                                                                                                        else
                                                                                                        {
                                                                                                            if ((i4 < (((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4)))
                                                                                                            {
                                                                                                                outp[i1][i2][i3][i4] = inp20[i1][i2][i3][(((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4)];
                                                                                                            }
                                                                                                            else
                                                                                                            {
                                                                                                                if ((i4 < ((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4)))
                                                                                                                {
                                                                                                                    outp[i1][i2][i3][i4] = inp21[i1][i2][i3][((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4)];
                                                                                                                }
                                                                                                                else
                                                                                                                {
                                                                                                                    if ((i4 < (((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4)))
                                                                                                                    {
                                                                                                                        outp[i1][i2][i3][i4] = inp22[i1][i2][i3][(((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4)];
                                                                                                                    }
                                                                                                                    else
                                                                                                                    {
                                                                                                                        if ((i4 < ((((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4) + inp23s4)))
                                                                                                                        {
                                                                                                                            outp[i1][i2][i3][i4] = inp23[i1][i2][i3][((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4)];
                                                                                                                        }
                                                                                                                        else
                                                                                                                        {
                                                                                                                            if ((i4 < (((((((((((((((((((((((inp1s4 + inp2s4) + inp3s4) + inp4s4) + inp5s4) + inp6s4) + inp7s4) + inp8s4) + inp9s4) + inp10s4) + inp11s4) + inp12s4) + inp13s4) + inp14s4) + inp15s4) + inp16s4) + inp17s4) + inp18s4) + inp19s4) + inp20s4) + inp21s4) + inp22s4) + inp23s4) + inp24s4)))
                                                                                                                            {
                                                                                                                                outp[i1][i2][i3][i4] = inp24[i1][i2][i3][(((((((((((((((((((((((i4 - inp1s4) - inp2s4) - inp3s4) - inp4s4) - inp5s4) - inp6s4) - inp7s4) - inp8s4) - inp9s4) - inp10s4) - inp11s4) - inp12s4) - inp13s4) - inp14s4) - inp15s4) - inp16s4) - inp17s4) - inp18s4) - inp19s4) - inp20s4) - inp21s4) - inp22s4) - inp23s4)];
                                                                                                                            }
                                                                                                                            else
                                                                                                                            {
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
