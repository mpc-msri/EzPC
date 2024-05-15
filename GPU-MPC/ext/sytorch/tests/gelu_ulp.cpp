#include <sytorch/backend/float.h>
#include <sytorch/backend/cleartext.h>

int main(int argc, char** argv)
{
    ClearText<i64> b1;
    FloatClearText<float> b2;
    int scale = 12;
    u64 size = 1ULL<<(scale+3);

    Tensor<i64>   t1({size});
    Tensor<i64>   v1({size});
    Tensor<float> t2({size});
    Tensor<float> v2({size});

    i64 sign = 1;
    if (argc > 1)
        sign = atoi(argv[1]);
    
    always_assert(sign == 1 || sign == -1);

    for (int i = 0; i < size; ++i)
    {
        float f = sign * i / float(1LL<<scale);
        t1.data[i] = sign * i;
        t2.data[i] = f;
    }

    b1.gelu(t1, v1, scale);
    b2.gelu(t2, v2, 0);

    float maxdiff = 0;
    int maxi;
    i64 maxdiff_i = 0;
    for (int i = 0; i < size; ++i)
    {
        float diff = std::abs((v1.data[i] / double(1LL << scale)) - v2.data[i]);
        if (diff > maxdiff)
        {
            maxdiff = diff;
            maxi = i;
            maxdiff_i = std::abs(v1.data[i] - (1LL<<12) * v2.data[i]);
        }
    }
    std::cout << "Max diff: " << maxdiff << " at x = " << (sign * maxi / double(1LL<<12)) << std::endl;
    std::cout << "ULP: " << maxdiff_i << std::endl;
}
