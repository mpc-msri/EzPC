#pragma once
#include <cstdint>
#include <cassert>
#include <Eigen/Dense>
#include <iostream>
#include "random.h"

typedef uint64_t u64;
typedef int64_t i64;

template <typename T>
class Tensor4D;

template <typename T>
class Tensor {
public:
    T *data;
    u64 size;

    Tensor(u64 s) : size(s), data(new T[s]) {}

    void randomize(double range) {
        for(u64 i = 0; i < this->size; i++) {
            auto r = (double)prngWeights.get<int32_t>();
            this->data[i] = (T)((r / (1ULL << 32)) * range);
            // this->data[i] = (T)((i % 2) * range);
            // this->data[i] = ((T)range) / 2;
        }
    }

    ~Tensor() {
        delete[] this->data;
    }

    T &operator()(u64 i) const {
        assert(i < this->size);
        return this->data[i];
    }

    void print() const {
        for (u64 i = 0; i < this->size; i++) {
            std::cout << this->data[i] << " ";
        }
        std::cout << std::endl;
    }

    void fill(T val) {
        for (u64 i = 0; i < this->size; i++) {
            this->data[i] = val;
        }
    }
};

template <typename T>
class Tensor2D {
public:
    T *data;
    u64 d1, d2;

    Tensor2D(u64 d1, u64 d2) : d1(d1), d2(d2), data(new T[d1 * d2]) {}

    void randomize(double range) {
        for(u64 i = 0; i < this->d1; i++) {
            for(u64 j = 0; j < this->d2; j++) {
                // this->data[i * this->d2 + j] = (T)((j % 2) * range);
                auto r = (double)prngWeights.get<int32_t>();
                this->data[i * this->d2 + j] = (T)((r / (1ULL << 32)) * range);
                // this->data[i * this->d2 + j] = ((T)range) / 2;
            }
        }
    }

    void resize(u64 d1, u64 d2) {
        if (this->d1 == d1 && this->d2 == d2) {
            return;
        }
        delete[] data;
        this->d1 = d1;
        this->d2 = d2;
        data = new T[d1 * d2];
    }

    ~Tensor2D() {
        delete[] this->data;
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

    void fill(T val) {
        for(u64 i = 0; i < this->d1; i++) {
            for(u64 j = 0; j < this->d2; j++) {
                this->data[i * this->d2 + j] = val;
            }
        }
    }

    void printshape() const {
        std::cout << "(" << d1 << ", " << d2 << ")" << std::endl;
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
                        data[i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l] += bias(l);
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
        if (this->d1 == d1 && this->d2 == d2 && this->d3 == d3 && this->d4 == d4) {
            return;
        }
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
        memcpy(data, other.data, d1 * d2 * d3 * d4 * sizeof(T));
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

    void print() const {
        for (u64 i = 0; i < d1; i++) {
            for (u64 j = 0; j < d2; j++) {
                for (u64 k = 0; k < d3; k++) {
                    for (u64 l = 0; l < d4; l++) {
                        std::cout << data[i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l] << " ";
                    }
                    if (d4 > 1) {
                        std::cout << std::endl;
                    }
                }
                if (d3 > 1) {
                    std::cout << std::endl;
                }
            }
            if (d2 > 1) {
                std::cout << std::endl;
            }
        }
        if (d1 > 1) {
            std::cout << std::endl;
        }
    }

    void zero() {
        for (u64 i = 0; i < d1 * d2 * d3 * d4; i++) {
            data[i] = (T)0;
        }
    }

    u64 argmax(u64 i) {
        assert(d3 == 1);
        assert(d4 == 1);
        assert(i < d1);
        u64 maxIndex = 0;
        T maxValue = data[i * d2];
        for (u64 j = 1; j < d2; j++) {
            if (data[i * d2 + j] > maxValue) {
                maxValue = data[i * d2 + j];
                maxIndex = j;
            }
        }
        return maxIndex;
    }

    void fill(T x) {
        for (u64 i = 0; i < d1 * d2 * d3 * d4; i++) {
            data[i] = x;
        }
    }

    void printshape() const {
        std::cout << "(" << d1 << ", " << d2 << ", " << d3 << ", " << d4 << ")" << std::endl;
    }

};