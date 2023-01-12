#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include "cleartext_library_float.h"

const static int PUBLIC = 0;
const static int ALICE = 0;
const static int BOB = 0;

vector<float> make_vector_float(int party, size_t size) {
return make_vector<float>(size);
}

template <typename... Args>
auto make_vector_float(int party, size_t first, Args... sizes)
{
return make_vector<float>(first, sizes...);
}

auto input1(int d1, int party)
{
    auto tmp0 = make_vector<float>(d1);

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        cin >> tmp0[i0];
    }

    return tmp0;
}

auto input2(int d1, int d2, int party)
{
    auto tmp0 = make_vector<float>(d1, d2);

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {

            cin >> tmp0[i0][i1];
        }
    }
    return tmp0;
}

auto input3(int d1, int d2, int d3, int party)
{
    auto tmp0 = make_vector<float>(d1, d2, d3);

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            for (uint32_t i2 = 0; i2 < d3; i2++)
            {

                cin >> tmp0[i0][i1][i2];
            }
        }
    }
    return tmp0;
}

auto input4(int d1, int d2, int d3, int d4, int party)
{
    auto tmp0 = make_vector<float>(d1, d2, d3, d4);

    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            for (uint32_t i2 = 0; i2 < d3; i2++)
            {
                for (uint32_t i3 = 0; i3 < d4; i3++)
                {

                    cin >> tmp0[i0][i1][i2][i3];
                }
            }
        }
    }
    return tmp0;
}

void output1(auto name, int d1, int party)
{
    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        cout << name[i0] << endl;
    }
}

void output2(auto name, int d1, int d2, int party)
{
    for (uint32_t i0 = 0; i0 < d1; i0++)
    {
        for (uint32_t i1 = 0; i1 < d2; i1++)
        {
            cout << name[i0][i1] << endl;
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
                cout << name[i0][i1][i2] << endl;
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
                    cout << name[i0][i1][i2][i3] << endl;
                }
            }
        }
    }
}