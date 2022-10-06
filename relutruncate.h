#include "tensor.h"

template <typename T>
void relutruncate(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu, u64 shift) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d1 == drelu.d1);
    assert(in.d2 == drelu.d2);
    assert(in.d3 == drelu.d3);
    assert(in.d4 == drelu.d4);
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                    out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? (in(i, j, k, l) >> shift) : 0;
                }
            }
        }
    }
}

template <typename T>
void relu(const Tensor4D<T> &in, const Tensor4D<T> &out, const Tensor4D<T> &drelu) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d1 == drelu.d1);
    assert(in.d2 == drelu.d2);
    assert(in.d3 == drelu.d3);
    assert(in.d4 == drelu.d4);
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    drelu(i, j, k, l) = (T)(in(i, j, k, l) > 0);
                    assert(drelu(i, j, k, l) == 1 || drelu(i, j, k, l) == 0);
                    out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                }
            }
        }
    }
}

template <typename T>
void truncate(const Tensor4D<T> &in, const Tensor4D<T> &out, u64 shift) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    if constexpr (std::is_floating_point<T>::value) {
                        out(i, j, k, l) = in(i, j, k, l) / (1 << shift);
                    } else {
                        out(i, j, k, l) = in(i, j, k, l) >> shift;
                    }
                }
            }
        }
    }
}

template <typename T>
void truncate(const Tensor4D<T> &in, u64 shift) {
    // Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2 * in.d3 * in.d4);
    // eA = eA / ((T)(1LL << shift));
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    if constexpr (std::is_floating_point<T>::value) {
                        in(i, j, k, l) = in(i, j, k, l) / ((T)(1ULL << shift));
                    } else {
                        in(i, j, k, l) = in(i, j, k, l) >> shift;
                    }
                }
            }
        }
    }
}

template <typename T>
void truncate(const Tensor2D<T> &in, u64 shift) {
//    Eigen::Map<Eigen::ArrayX<T>> eA(in.data, in.d1 * in.d2);
//    eA = eA / ((T)(1LL << shift)); // this gives bad accuracy, why?
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            if constexpr (std::is_floating_point<T>::value) {
                in(i, j) = in(i, j) / ((T)(1ULL << shift));
            } else {
                in(i, j) = in(i, j) >> shift;
            }
        }
    }
}

template <typename T>
void select(const Tensor4D<T> &in, const Tensor4D<T> &drelu, const Tensor4D<T> &out) {
    assert(in.d1 == out.d1);
    assert(in.d2 == out.d2);
    assert(in.d3 == out.d3);
    assert(in.d4 == out.d4);
    assert(in.d1 == drelu.d1);
    assert(in.d2 == drelu.d2);
    assert(in.d3 == drelu.d3);
    assert(in.d4 == drelu.d4);
    for (u64 i = 0; i < in.d1; i++) {
        for (u64 j = 0; j < in.d2; j++) {
            for (u64 k = 0; k < in.d3; k++) {
                for (u64 l = 0; l < in.d4; l++) {
                    assert(drelu(i, j, k, l) == 0 || drelu(i, j, k, l) == 1);
                    out(i, j, k, l) = (drelu(i, j, k, l) == 1) ? in(i, j, k, l) : 0;
                }
            }
        }
    }
}