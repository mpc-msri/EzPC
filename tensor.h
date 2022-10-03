#include <cstdint>
#include <cassert>

typedef uint64_t u64;
typedef int64_t i64;

template <typename T>
class ViewTensor {
public:
    T *data;
    u64 size;

    ViewTensor(T* ptr, u64 size) : size(size) {
        data = ptr;
    }

    T& operator[](u64 i) {
        assert(i < size);
        return data[i];
    }

    const T& operator[](u64 i) const {
        assert(i < size);
        return data[i];
    }

};

template <typename T>
class Tensor : public ViewTensor<T> {
public:
    Tensor(u64 s) : ViewTensor<T>(new T[s], s) {}

    void randomize() {
        for(u64 i = 0; i < this->size; i++) {
            this->data[i] = (T)0;
        }
    }

    ~Tensor() {
        delete[] this->data;
    }
};

template <typename T>
class ViewTensor2D {
public:
    T *data;
    u64 d1, d2;

    ViewTensor2D(T* ptr, u64 d1, u64 d2) : d1(d1), d2(d2) {
        data = ptr;
    }

    ViewTensor<T> operator[](u64 i) {
        assert(i < d1);
        return ViewTensor<T>(&data[i * d2], d2);
    }

};

template <typename T>
class Tensor2D : public ViewTensor2D<T> {
public:
    Tensor2D(u64 d1, u64 d2) : ViewTensor2D<T>(new T[d1 * d2], d1, d2) {}

    ~Tensor2D() {
        delete[] this->data;
    }

};


template <typename T>
class ViewTensor3D {
public:
    T *data;
    u64 d1, d2, d3;

    ViewTensor3D(T* ptr, u64 d1, u64 d2, u64 d3) : d1(d1), d2(d2), d3(d3) {
        data = ptr;
    }

    ViewTensor2D<T> operator[](u64 i) {
        assert(i < d1);
        return ViewTensor2D<T>(&data[i * d2 * d3], d2, d3);
    }

};

template <typename T>
class Tensor3D : public ViewTensor3D<T> {
public:
    Tensor3D(u64 d1, u64 d2, u64 d3) : ViewTensor3D<T>(new T[d1 * d2 * d3], d1, d2, d3) {}

    ~Tensor3D() {
        delete[] this->data;
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

    Tensor4D(T* ptr, u64 d1, u64 d2, u64 d3, u64 d4) : d1(d1), d2(d2), d3(d3), d4(d4) {
        data = ptr;
    }

    ~Tensor4D() {
        delete[] data;
    }

    void addBias(Tensor<T> bias) {
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
 
    ViewTensor3D<T> operator[](u64 i) {
        assert(i < d1);
        return ViewTensor3D<T>(&data[i * d2 * d3 * d4], d2, d3, d4);
    }

};