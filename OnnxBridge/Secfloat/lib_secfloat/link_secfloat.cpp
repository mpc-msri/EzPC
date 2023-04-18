#include "inout.cpp"

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

auto Add(const FPArray &x, const FPArray &y)
{
    return __fp_op->add(x, y);
}

auto Mul(const FPArray &x, const FPArray &y)
{
    return __fp_op->mul(x, y);
}

void ElemWiseSecretSharedVectorMult(int32_t s1, auto &arr1, auto &arr2, auto &outArr)
{
    for (uint32_t ii = 0; ii < s1; ii++)
    {
        outArr[ii] = Mul(arr1[ii], arr2[ii]);
    }
}

void ElemWiseActModelVectorMult(int32_t s1, auto &arr1, auto &arr2, auto &outArr)
{
    ElemWiseSecretSharedVectorMult(s1, arr1, arr2, outArr);
}

void MaxPool(int32_t N, int32_t C, int32_t H, int32_t W, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t C1, int32_t imgH, int32_t imgW, auto &inArr, auto &outArr)
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
                    FPArray maxi = __public_float_to_baba(0., ALICE);

                    if ((((leftTopCornerH < 0) || (leftTopCornerH >= imgH)) || ((leftTopCornerW < 0) || (leftTopCornerW >= imgW))))
                    {
                        maxi = __public_float_to_baba(0., ALICE);
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

                            FPArray temp = __public_float_to_baba(0., ALICE);

                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                            {
                                temp = __public_float_to_baba(0., ALICE);
                            }
                            else
                            {
                                temp = inArr[n][c][curPosH][curPosW];
                            }
                            maxi = __fp_op->if_else(__fp_op->LT(__fp_op->sub(maxi, temp), __public_float_to_baba(0., ALICE)), temp, maxi);
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

void AvgPool(int32_t N, int32_t C, int32_t H, int32_t W, int32_t ksizeH, int32_t ksizeW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t N1, int32_t C1, int32_t imgH, int32_t imgW, auto &inArr, auto &outArr)
{
    int32_t rows = (((N * C) * H) * W);

    auto filterAvg = make_vector_float(ALICE, rows);

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
                    FPArray curFilterSum = __public_float_to_baba(0., ALICE);

                    for (uint32_t fh = 0; fh < ksizeH; fh++)
                    {
                        for (uint32_t fw = 0; fw < ksizeW; fw++)
                        {
                            int32_t curPosH = (leftTopCornerH + fh);

                            int32_t curPosW = (leftTopCornerW + fw);

                            FPArray temp = __public_float_to_baba(0., ALICE);

                            if ((((curPosH < 0) || (curPosH >= imgH)) || ((curPosW < 0) || (curPosW >= imgW))))
                            {
                                temp = __public_float_to_baba(0., ALICE);
                            }
                            else
                            {
                                temp = inArr[n][c][curPosH][curPosW];
                            }
                            curFilterSum = __fp_op->add(curFilterSum, temp);
                        }
                    }
                    int32_t ksizeH64 = ksizeH;

                    int32_t ksizeW64 = ksizeW;

                    int32_t filterSz64 = (ksizeH64 * ksizeW64);

                    FPArray curFilterAvg = __fp_op->div(curFilterSum, __public_float_to_baba(intToFloat(filterSz64), ALICE));

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

void Relu(int32_t s1, int32_t s2, auto &inArr, auto &outArr)
{
    int32_t size = (s1 * s2);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

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

void Relu(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

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

void Leaky_Relu(int32_t s1, int32_t s2, float alpha, auto &inArr, auto &outArr)
{
    int32_t size = (s1 * s2);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            reshapedInArr[linIdx] = inArr[i1][i2];
        }
    }
    Leaky_Relu(size, alpha, reshapedInArr, reshapedOutArr);
    for (uint32_t i1 = 0; i1 < s1; i1++)
    {
        for (uint32_t i2 = 0; i2 < s2; i2++)
        {
            int32_t linIdx = ((i1 * s2) + i2);

            outArr[i1][i2] = reshapedOutArr[linIdx];
        }
    }
}

void Leaky_Relu(int32_t s1, int32_t s2, int32_t s3, int32_t s4, float alpha, auto &inArr, auto &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

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
    Leaky_Relu(size, alpha, reshapedInArr, reshapedOutArr);
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

void Sigmoid(int32_t s1, int32_t s2, auto &inArr, auto &outArr)
{
    int32_t size = (s1 * s2);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

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

void Sigmoid(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

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

void Tanh(int32_t s1, int32_t s2, auto &inArr, auto &outArr)
{
    int32_t size = (s1 * s2);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

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

void Tanh(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto &inArr, auto &outArr)
{
    int32_t size = (((s1 * s2) * s3) * s4);

    auto reshapedInArr = make_vector_float(ALICE, size);

    auto reshapedOutArr = make_vector_float(ALICE, size);

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
