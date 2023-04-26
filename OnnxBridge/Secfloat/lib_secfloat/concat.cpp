

void Concat1T44(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat2T222(int32_t s1, int32_t s2, int32_t inp1s1, int32_t inp1s2, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat2T444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat3T4444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat4T44444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat5T444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat6T4444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat7T44444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat8T444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat9T4444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat10T44444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat11T444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat12T4444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat13T44444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat14T444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat15T4444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat16T44444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat17T444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat18T4444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat19T44444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, vector<vector<vector<vector<FPArray>>>> &inp19, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat20T444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, vector<vector<vector<vector<FPArray>>>> &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, vector<vector<vector<vector<FPArray>>>> &inp20, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat21T4444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, vector<vector<vector<vector<FPArray>>>> &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, vector<vector<vector<vector<FPArray>>>> &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, vector<vector<vector<vector<FPArray>>>> &inp21, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat22T44444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, vector<vector<vector<vector<FPArray>>>> &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, vector<vector<vector<vector<FPArray>>>> &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, vector<vector<vector<vector<FPArray>>>> &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, vector<vector<vector<vector<FPArray>>>> &inp22, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat23T444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, vector<vector<vector<vector<FPArray>>>> &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, vector<vector<vector<vector<FPArray>>>> &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, vector<vector<vector<vector<FPArray>>>> &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, vector<vector<vector<vector<FPArray>>>> &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, vector<vector<vector<vector<FPArray>>>> &inp23, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat24T4444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, vector<vector<vector<vector<FPArray>>>> &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, vector<vector<vector<vector<FPArray>>>> &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, vector<vector<vector<vector<FPArray>>>> &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, vector<vector<vector<vector<FPArray>>>> &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, vector<vector<vector<vector<FPArray>>>> &inp23, int32_t inp24s1, int32_t inp24s2, int32_t inp24s3, int32_t inp24s4, vector<vector<vector<vector<FPArray>>>> &inp24, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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

void Concat25T44444444444444444444444444(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t inp1s1, int32_t inp1s2, int32_t inp1s3, int32_t inp1s4, vector<vector<vector<vector<FPArray>>>> &inp1, int32_t inp2s1, int32_t inp2s2, int32_t inp2s3, int32_t inp2s4, vector<vector<vector<vector<FPArray>>>> &inp2, int32_t inp3s1, int32_t inp3s2, int32_t inp3s3, int32_t inp3s4, vector<vector<vector<vector<FPArray>>>> &inp3, int32_t inp4s1, int32_t inp4s2, int32_t inp4s3, int32_t inp4s4, vector<vector<vector<vector<FPArray>>>> &inp4, int32_t inp5s1, int32_t inp5s2, int32_t inp5s3, int32_t inp5s4, vector<vector<vector<vector<FPArray>>>> &inp5, int32_t inp6s1, int32_t inp6s2, int32_t inp6s3, int32_t inp6s4, vector<vector<vector<vector<FPArray>>>> &inp6, int32_t inp7s1, int32_t inp7s2, int32_t inp7s3, int32_t inp7s4, vector<vector<vector<vector<FPArray>>>> &inp7, int32_t inp8s1, int32_t inp8s2, int32_t inp8s3, int32_t inp8s4, vector<vector<vector<vector<FPArray>>>> &inp8, int32_t inp9s1, int32_t inp9s2, int32_t inp9s3, int32_t inp9s4, vector<vector<vector<vector<FPArray>>>> &inp9, int32_t inp10s1, int32_t inp10s2, int32_t inp10s3, int32_t inp10s4, vector<vector<vector<vector<FPArray>>>> &inp10, int32_t inp11s1, int32_t inp11s2, int32_t inp11s3, int32_t inp11s4, vector<vector<vector<vector<FPArray>>>> &inp11, int32_t inp12s1, int32_t inp12s2, int32_t inp12s3, int32_t inp12s4, vector<vector<vector<vector<FPArray>>>> &inp12, int32_t inp13s1, int32_t inp13s2, int32_t inp13s3, int32_t inp13s4, vector<vector<vector<vector<FPArray>>>> &inp13, int32_t inp14s1, int32_t inp14s2, int32_t inp14s3, int32_t inp14s4, vector<vector<vector<vector<FPArray>>>> &inp14, int32_t inp15s1, int32_t inp15s2, int32_t inp15s3, int32_t inp15s4, vector<vector<vector<vector<FPArray>>>> &inp15, int32_t inp16s1, int32_t inp16s2, int32_t inp16s3, int32_t inp16s4, vector<vector<vector<vector<FPArray>>>> &inp16, int32_t inp17s1, int32_t inp17s2, int32_t inp17s3, int32_t inp17s4, vector<vector<vector<vector<FPArray>>>> &inp17, int32_t inp18s1, int32_t inp18s2, int32_t inp18s3, int32_t inp18s4, vector<vector<vector<vector<FPArray>>>> &inp18, int32_t inp19s1, int32_t inp19s2, int32_t inp19s3, int32_t inp19s4, vector<vector<vector<vector<FPArray>>>> &inp19, int32_t inp20s1, int32_t inp20s2, int32_t inp20s3, int32_t inp20s4, vector<vector<vector<vector<FPArray>>>> &inp20, int32_t inp21s1, int32_t inp21s2, int32_t inp21s3, int32_t inp21s4, vector<vector<vector<vector<FPArray>>>> &inp21, int32_t inp22s1, int32_t inp22s2, int32_t inp22s3, int32_t inp22s4, vector<vector<vector<vector<FPArray>>>> &inp22, int32_t inp23s1, int32_t inp23s2, int32_t inp23s3, int32_t inp23s4, vector<vector<vector<vector<FPArray>>>> &inp23, int32_t inp24s1, int32_t inp24s2, int32_t inp24s3, int32_t inp24s4, vector<vector<vector<vector<FPArray>>>> &inp24, int32_t inp25s1, int32_t inp25s2, int32_t inp25s3, int32_t inp25s4, vector<vector<vector<vector<FPArray>>>> &inp25, int32_t axis, vector<vector<vector<vector<FPArray>>>> &outp)
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
