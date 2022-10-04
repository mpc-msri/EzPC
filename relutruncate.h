#include "tensor.h"

template <typename T>
void relutruncate(Tensor4D<T> &in, Tensor4D<T> &out, u64 shift) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    out[i][j][k][l] = in[i][j][k][l] > 0 ? (in[i][j][k][l] >> shift) : 0;
                }
            }
        }
    }
}

template <typename T>
void truncate(Tensor4D<T> &in, Tensor4D<T> &out, u64 shift) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    out[i][j][k][l] = in[i][j][k][l] >> shift;
                }
            }
        }
    }
}
