#pragma once
#include <cstdint>
#include <cassert>

typedef uint64_t u64;
typedef int64_t i64;

template <typename T>
class Tensor {
public:
    T *data;
    u64 size;

    Tensor(u64 s) : size(s), data(new T[s]) {}

    void randomize() {
        for(u64 i = 0; i < this->size; i++) {
            this->data[i] = (T)0;
        }
    }

    ~Tensor() {
        delete[] this->data;
    }

    T &operator()(u64 i) const {
        assert(i < this->size);
        return this->data[i];
    }
};

template <typename T>
class Tensor2D {
public:
    T *data;
    u64 d1, d2;

    Tensor2D(u64 d1, u64 d2) : d1(d1), d2(d2), data(new T[d1 * d2]) {}

    void randomize() {
        for(u64 i = 0; i < this->d1; i++) {
            for(u64 j = 0; j < this->d2; j++) {
                this->data[i * this->d2 + j] = (T)1;
            }
        }
    }

    ~Tensor2D() {
        delete[] this->data;
    }

    void updateWeight(const Tensor2D<T> &e, float lr) {
        assert(this->d1 == e.d1);
        assert(this->d2 == e.d2);
        for(u64 i = 0; i < this->d1; i++) {
            for(u64 j = 0; j < this->d2; j++) {
                this->data[i * this->d2 + j] -= (T)(lr * e(i, j));
            }
        }
    }

    T& operator()(u64 i, u64 j) const {
        assert(i < this->d1);
        assert(j < this->d2);
        return this->data[i * this->d2 + j];
    }

    void print() {
        for(u64 i = 0; i < this->d1; i++) {
            for(u64 j = 0; j < this->d2; j++) {
                std::cout << this->data[i * this->d2 + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    void zero() {
        for(u64 i = 0; i < this->d1; i++) {
            for(u64 j = 0; j < this->d2; j++) {
                this->data[i * this->d2 + j] = (T)0;
            }
        }
    }

};

template <typename T>
class Tensor4D {
public:
    T *data;
    u64 d1, d2, d3, d4;

    
    Tensor4D(u64 d1, u64 d2, u64 d3, u64 d4) : d1(d1), d2(d2), d3(d3), d4(d4) {
        data = new T[d1 * d2 * d3 * d4];
    }

    ~Tensor4D() {
        delete[] data;
    }

    void addBias(const Tensor<T> &bias) {
        assert(bias.size == d4);
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        data[i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l] += bias[l];
                    }
                }
            }
        }
    }

    void addBias2D(const Tensor<T> &bias) {
        assert(bias.size == d2);
        assert(d3 == 1);
        assert(d4 == 1);

        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                data[i * d2 + j] += bias(j);
            }
        }
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        delete[] data;
        this->d1 = d1;
        this->d2 = d2;
        this->d3 = d3;
        this->d4 = d4;
        data = new T[d1 * d2 * d3 * d4];
    }

    void copy(const Tensor4D<T> &other) {
        assert(d1 == other.d1);
        assert(d2 == other.d2);
        assert(d3 == other.d3);
        assert(d4 == other.d4);
        data = new T[d1 * d2 * d3 * d4];
        for (u64 i = 0; i < d1 * d2 * d3 * d4; i++) {
            data[i] = other.data[i];
        }
    }

    void randomize() {
        for (u64 i = 0; i < d1 * d2 * d3 * d4; i++) {
            data[i] = (T)1;
        }
    }
 
    T& operator()(u64 i, u64 j, u64 k, u64 l) const {
        assert(i < d1);
        assert(j < d2);
        assert(k < d3);
        assert(l < d4);
        return data[i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l];
    }

    void transpose2D()
    {
        assert(d3 == 1);
        assert(d4 == 1);
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                std::swap(data[j * d1 + i], data[i * d2 + j]);
            }
        }
        std::swap(d1, d2);
    }

    void updateWeight(const Tensor4D<T> &grad, float learningRate) {
        assert(d1 == grad.d1);
        assert(d2 == grad.d2);
        assert(d3 == grad.d3);
        assert(d4 == grad.d4);
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        data[i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l] -= learningRate * grad(i, j, k, l);
                    }
                }
            }
        }
    }

    void print() const {
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        std::cout << data[i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void zero() {
        for (u64 i = 0; i < d1 * d2 * d3 * d4; i++) {
            data[i] = (T)0;
        }
    }

};