#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "library_float.h"

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