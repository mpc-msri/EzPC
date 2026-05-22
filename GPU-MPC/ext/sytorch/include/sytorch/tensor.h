// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <cassert>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <sytorch/random.h>
#include <sytorch/graph.h>
#include <llama/assert.h>
#include <cmath>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int64_t i64;
typedef int32_t i32;

template <typename T>
class TensorRef
{
public:
    T *data;
    T *d_data = nullptr;
    u64 size;
    TensorRef(T *data, u64 size) : data(data), size(size) {}
    void zero()
    {
        for (u64 i = 0; i < size; i++)
        {
            data[i] = 0;
        }
    }
};

template <typename T>
class Tensor5D;

template <typename T>
class Tensor4D;

template <typename T>
class Tensor2D;

template <typename T>
class Tensor1D;

template <typename T>
class Tensor
{
    bool isFreed = false;

public:
    Tensor(){};
    T *data;
    T *d_data = nullptr;
    std::vector<u64> shape;
    bool isOwner = true;

    bool graphGenMode = false;
    LayerGraphNode<T> *graphNode = nullptr;
    bool wasCopiedToCpu = false;

    bool wasFreed() { return isFreed; }

    T value_at(const std::vector<u64> &idx) const
    {
        always_assert(idx.size() == this->shape.size());
        u64 offset = 0;
        for (u64 i = 0; i < idx.size(); i++)
        {
            always_assert(idx[i] < this->shape[i]);
            u64 stride = 1;
            for (u64 j = i + 1; j < idx.size(); j++)
            {
                stride *= this->shape[j];
            }
            offset += idx[i] * stride;
        }
        return this->data[offset];
    }

    void allocate(const std::vector<u64> &s)
    {
        always_assert(isOwner);
        this->shape = s;
        if (this->size() > 0)
        {
            this->data = new T[this->size()];
            isFreed = false;
        }
        else
        {
            this->data = nullptr;
            isFreed = true;
        }
    }

    void free()
    {
        always_assert(isOwner);
        if (isFreed)
        {
            return;
        }
        if (this->size() == 0)
        {
            return;
        }
        delete[] data;
        // if (d_data)
        //     cudaFree(d_data);
        this->shape = {};
        isFreed = true;
    }

    void freeGpu()
    {
        always_assert(isOwner);
        // if (isFreed)
        // {
        //     return;
        // }
        // if (this->size() == 0)
        // {
        //     return;
        // }
        // delete[] data;
        if (d_data)
        {
            cudaFreeAsync(d_data, 0);
            d_data = nullptr;
        }
        // this->shape = {};
        // isFreed = true;
    }

    void resize(const std::vector<u64> &s)
    {
        always_assert(isOwner);
        if (s.size() == this->shape.size())
        {
            bool allSameDims = true;
            for (u64 i = 0; i < s.size(); i++)
            {
                if (s[i] != this->shape[i])
                {
                    allSameDims = false;
                    break;
                }
            }
            if (allSameDims)
            {
                return;
            }
        }
        free();
        allocate(s);
    }

    Tensor(const std::vector<u64> &s)
    {
        allocate(s);
    }

    Tensor(std::initializer_list<u64> s)
    {
        allocate(s);
    }

    Tensor(T *data, T *d_data, const std::vector<u64> &s)
    {
        this->data = data;
        this->d_data = d_data;
        this->shape = s;
        this->isOwner = false;
    }

    Tensor(T *data, const std::vector<u64> &s)
    {
        this->data = data;
        this->d_data = nullptr;
        this->shape = s;
        this->isOwner = false;
    }

    ~Tensor()
    {
        // printf("Freeing tensor=%d, %lx, %lu\n", isOwner, data, size());
        if (isOwner)
            free();
    }

    u64 size() const
    {
        if (this->shape.size() == 0)
        {
            return 0;
        }
        u64 s = 1;
        for (auto d : this->shape)
        {
            s *= d;
        }
        return s;
    }

    bool is_same_shape(const Tensor<T> &other) const
    {
        if (!(this->shape.size() == other.shape.size()))
        {
            return false;
        }
        for (u64 i = 0; i < this->shape.size(); i++)
        {
            if (!(this->shape[i] == other.shape[i]))
            {
                return false;
            }
        }
        return true;
    }

    void assert_same_shape(const Tensor<T> &other)
    {
        always_assert(this->shape.size() == other.shape.size());
        for (u64 i = 0; i < this->shape.size(); i++)
        {
            always_assert(this->shape[i] == other.shape[i]);
        }
    }

    void copy(const Tensor<T> &other, bool copyGraph = true, bool gpuToCpu = false)
    {
        assert_same_shape(other);
        if (gpuToCpu)
        {
            cudaMemcpy(data, other.d_data, other.size() * sizeof(T), cudaMemcpyDeviceToHost);
        }
        else
        {
// memcpy(data, other.data, size() * sizeof(T));
#pragma omp parallel for
            for (u64 i = 0; i < size(); ++i)
            {
                data[i] = other.data[i];
            }
            if (copyGraph)
                this->graphNode = other.graphNode;
        }
    }

    void fill(T x)
    {
        for (u64 i = 0; i < size(); i++)
        {
            data[i] = x;
        }
    }

    void zero()
    {
        fill(0);
    }

    void input(int scale)
    {
        for (u64 i = 0; i < size(); i++)
        {
            double d;
            std::cin >> d;
            data[i] = (i64)(d * (1LL << scale));
        }
    }

    void input_nchw(int scale)
    {
        always_assert(this->shape.size() >= 2); // atleast batch and channel axis

        u64 batch_size = shape[0];
        u64 num_channel = shape.back();
        u64 rest_size = size() / (batch_size * num_channel);

        for (u64 i = 0; i < size(); i++)
        {
            double d;
            std::cin >> d;
            u64 curr_batch = i / (num_channel * rest_size);
            u64 curr_channel = (i / rest_size) % num_channel;
            u64 curr_rest = i % rest_size;
            u64 new_idx = curr_batch * (num_channel * rest_size) + curr_rest * num_channel + curr_channel;
            data[new_idx] = (i64)(d * (1LL << scale));
        }
    }

    void print()
    {
        // std::cout << "Tensor(";
        // for (int i = 0; i < this->shape.size(); i++)
        // {
        //     std::cout << this->shape[i] << ", ";
        // }
        // std::cout << ")" << std::endl;
        for (u64 i = 0; i < size(); i++)
        {
            std::cout << (i64)data[i] << " ";
        }
        std::cout << std::endl;
    }

    void printshape()
    {
        std::cout << "(";
        for (int i = 0; i < this->shape.size(); i++)
        {
            std::cout << this->shape[i] << ", ";
        }
        std::cout << ")" << std::endl;
    }

    T multidir_broadcast_value(const std::vector<u64> &broadcast_shape, const std::vector<u64> &idx) const
    {
        always_assert(broadcast_shape.size() >= this->shape.size());
        always_assert(broadcast_shape.size() == idx.size());
        int num_broadcast_dims = broadcast_shape.size() - this->shape.size();
        std::vector<u64> new_idx;
        for (u64 i = 0; i < this->shape.size(); i++)
        {
            always_assert(this->shape[i] == 1 || this->shape[i] == broadcast_shape[i + num_broadcast_dims]);
            if (this->shape[i] == 1)
            {
                new_idx.push_back(0);
            }
            else
            {
                always_assert(idx[i + num_broadcast_dims] < this->shape[i]);
                new_idx.push_back(idx[i + num_broadcast_dims]);
            }
        }
        return this->value_at(new_idx);
    }

    void load(const std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> &arr, int scale)
    {
        int d1 = arr.size();
        int d2 = arr[0].size();
        int d3 = arr[0][0].size();
        int d4 = arr[0][0][0].size();
        int d5 = arr[0][0][0][0].size();
        always_assert(d1 == this->shape[0]);
        always_assert(d2 == this->shape[1]);
        always_assert(d3 == this->shape[2]);
        always_assert(d4 == this->shape[3]);
        always_assert(d5 == this->shape[4]);
        for (int i = 0; i < d1; i++)
        {
            for (int j = 0; j < d2; j++)
            {
                for (int k = 0; k < d3; k++)
                {
                    for (int l = 0; l < d4; l++)
                    {
                        for (int m = 0; m < d5; m++)
                        {
                            this->data[i * d2 * d3 * d4 * d5 + j * d3 * d4 * d5 + k * d4 * d5 + l * d5 + m] = (i64)(arr[i][j][k][l][m] * scale);
                        }
                    }
                }
            }
        }
    }

    void load(const std::string filename, u64 scale)
    {
        size_t size_in_bytes = std::filesystem::file_size(filename);
        std::cout << size() << std::endl;
        std::cout << size_in_bytes << std::endl;
        // always_assert(size_in_bytes >= size() * 4);
        size_t wSize = std::min(size(), size_in_bytes / 4);
        float *floatInput = new float[/*size()*/ wSize];
        std::ifstream file(filename, std::ios::binary);
        file.read((char *)floatInput, /*size_in_bytes*/ /*size()*/ wSize * 4);
        file.close();
        for (u64 i = 0; i < size(); ++i)
        {
            data[i] = T(floatInput[i % (size_in_bytes / 4)] * (1LL << scale));
        }
        delete[] floatInput;
    }

    Tensor5D<T> as_5d()
    {
        assert(this->shape.size() == 5);
        return Tensor5D<T>(this->data, this->shape[0], this->shape[1], this->shape[2], this->shape[3], this->shape[4]);
    }

    Tensor4D<T> as_4d() const
    {
        assert(this->shape.size() == 4);
        return Tensor4D<T>(this->data, this->d_data, this->shape[0], this->shape[1], this->shape[2], this->shape[3]);
    }

    Tensor2D<T> as_2d()
    {
        always_assert(this->shape.size() == 2);
        return Tensor2D<T>(this->data, this->d_data, this->shape[0], this->shape[1]);
    }

    Tensor<T> view(u64 i)
    {
        assert(i < shape[0]);
        u64 newsize = size() / shape[0];
        auto newshape = shape;
        newshape.erase(newshape.begin());
        return Tensor<T>(data + i * newsize, newshape);
    }
};

template <typename T>
class Tensor1D
{
public:
    T *data, *d_data = nullptr;
    u64 d1;
    bool isOwner = true;

    Tensor1D(u64 s) : d1(s), data(new T[s]) {}
    Tensor1D(T *data, u64 s) : d1(s), data(new T[s])
    {
        memcpy(this->data, data, s * sizeof(T));
    }

    void randomize(double range)
    {
        for (u64 i = 0; i < this->d1; i++)
        {
            auto r = (double)prngWeights.get<int32_t>();
            // if(i == 0) printf("Inside randomize=%lf, %ld range=%lf\n", ((r / (1LL << 31)) * range), u64(((r / (1LL << 31)) * range)), range);
            this->data[i] = (T)((r / (1LL << 31)) * range);
        }
    }

    void randomize()
    {
        for (u64 i = 0; i < this->d1; i++)
        {
            this->data[i] = (T)prngWeights.get<T>();
        }
    }

    ~Tensor1D()
    {
        if (isOwner)
        {
            delete[] this->data;
            // if (d_data)
            //     cudaFree(d_data);
        }
    }

    u64 size() const
    {
        return d1;
    }

    TensorRef<T> ref()
    {
        return TensorRef<T>(data, size());
    }

    T &operator()(u64 i) const
    {
        assert(i < this->d1);
        return this->data[i];
    }

    void fill(T val)
    {
        for (u64 i = 0; i < this->d1; i++)
        {
            this->data[i] = val;
        }
    }

    Tensor<T> as_nd()
    {
        return Tensor<T>(data, d_data, {d1});
    }
};

template <typename T>
class Tensor2D
{
public:
    u64 d1, d2;
    T *data, *d_data = nullptr;
    bool isOwner = true;

    Tensor2D(u64 d1, u64 d2) : d1(d1), d2(d2), data(new T[d1 * d2]) {}

    Tensor2D(T *data, u64 d1, u64 d2) : d1(d1), d2(d2), data(data), isOwner(false) {}

    Tensor2D(T *data, T *d_data, u64 d1, u64 d2) : d1(d1), d2(d2), data(data), d_data(d_data), isOwner(false) {}

    void randomize(double range)
    {
        for (u64 i = 0; i < this->d1; i++)
        {
            for (u64 j = 0; j < this->d2; j++)
            {
                auto r = (double)prngWeights.get<int32_t>();
                // if(i == 0 && j == 0) printf("Inside randomize=%lf, %ld range=%lf\n", ((r / (1LL << 31)) * range), u64(((r / (1LL << 31)) * range)), range);
                this->data[i * this->d2 + j] = (T)((r / (1LL << 31)) * range);
            }
        }
    }

    u64 size() const
    {
        return d1 * d2;
    }

    TensorRef<T> ref()
    {
        return TensorRef<T>(data, size());
    }

    void resize(u64 d1, u64 d2)
    {
        always_assert(this->isOwner);
        if (this->d1 == d1 && this->d2 == d2)
        {
            return;
        }
        delete[] data;
        this->d1 = d1;
        this->d2 = d2;
        data = new T[d1 * d2];
    }

    ~Tensor2D()
    {
        if (this->isOwner)
        {
            delete[] this->data;
            // if (d_data)
            //     cudaFree(d_data);
        }
    }

    T &operator()(u64 i, u64 j) const
    {
        assert(i < this->d1);
        assert(j < this->d2);
        return this->data[i * this->d2 + j];
    }

    void zero()
    {
        fill(0);
    }

    void fill(T val)
    {
        for (u64 i = 0; i < this->d1; i++)
        {
            for (u64 j = 0; j < this->d2; j++)
            {
                this->data[i * this->d2 + j] = val;
            }
        }
    }

    u64 argmax(u64 i)
    {
        assert(i < d1);
        u64 maxIndex = 0;
        T maxValue = data[i * d2];
        for (u64 j = 1; j < d2; j++)
        {
            if (data[i * d2 + j] > maxValue)
            {
                maxValue = data[i * d2 + j];
                maxIndex = j;
            }
        }
        return maxIndex;
    }

    Tensor<T> as_nd()
    {
        return Tensor<T>(data, d_data, {d1, d2});
    }
};

template <typename T>
class Tensor4D
{
public:
    u64 d1, d2, d3, d4;
    T *data, *d_data = nullptr;
    bool isOwner = true;

    Tensor4D(u64 d1, u64 d2, u64 d3, u64 d4) : d1(d1), d2(d2), d3(d3), d4(d4)
    {
        data = new T[d1 * d2 * d3 * d4];
    }

    Tensor4D(T *data, u64 d1, u64 d2, u64 d3, u64 d4) : data(data), d1(d1), d2(d2), d3(d3), d4(d4)
    {
        isOwner = false;
    }

    Tensor4D(T *data, T *d_data, u64 d1, u64 d2, u64 d3, u64 d4) : data(data), d_data(d_data), d1(d1), d2(d2), d3(d3), d4(d4)
    {
        isOwner = false;
    }

    ~Tensor4D()
    {
        if (isOwner)
        {
            delete[] data;
            // if (d_data)
            //     cudaFree(d_data);
        }
    }

    u64 size() const
    {
        return d1 * d2 * d3 * d4;
    }

    TensorRef<T> ref()
    {
        return TensorRef<T>(data, size());
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4)
    {
        always_assert(isOwner);
        if (this->d1 == d1 && this->d2 == d2 && this->d3 == d3 && this->d4 == d4)
        {
            return;
        }
        delete[] data;
        this->d1 = d1;
        this->d2 = d2;
        this->d3 = d3;
        this->d4 = d4;
        data = new T[d1 * d2 * d3 * d4];
    }

    void resize(const std::vector<u64> &shape)
    {
        always_assert(isOwner);
        always_assert(shape.size() == 4);
        resize(shape[0], shape[1], shape[2], shape[3]);
    }

    T &operator()(u64 i, u64 j, u64 k, u64 l) const
    {
        assert(i < d1);
        assert(j < d2);
        assert(k < d3);
        assert(l < d4);
        return data[i * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l];
    }

    u64 argmax(u64 i)
    {
        assert(d3 == 1);
        assert(d4 == 1);
        assert(i < d1);
        u64 maxIndex = 0;
        T maxValue = data[i * d2];
        for (u64 j = 1; j < d2; j++)
        {
            if (data[i * d2 + j] > maxValue)
            {
                maxValue = data[i * d2 + j];
                maxIndex = j;
            }
        }
        return maxIndex;
    }

    Tensor<T> as_nd()
    {
        return Tensor<T>(data, d_data, {d1, d2, d3, d4});
    }

    void fill(T val)
    {
        for (u64 i = 0; i < size(); i++)
        {
            this->data[i] = val;
        }
    }
};

template <typename T>
class Tensor5D
{
public:
    u64 d1, d2, d3, d4, d5;
    T *data, *d_data = nullptr;
    bool isOwner = true;

    Tensor5D(u64 d1, u64 d2, u64 d3, u64 d4, u64 d5) : d1(d1), d2(d2), d3(d3), d4(d4), d5(d5)
    {
        data = new T[d1 * d2 * d3 * d4 * d5];
    }

    Tensor5D(T *data, u64 d1, u64 d2, u64 d3, u64 d4, u64 d5) : data(data), d1(d1), d2(d2), d3(d3), d4(d4), d5(d5)
    {
        isOwner = false;
    }

    ~Tensor5D()
    {
        if (isOwner)
        {
            delete[] data;
            // if (d_data)
            //     cudaFree(d_data);
        }
    }

    u64 size() const
    {
        return d1 * d2 * d3 * d4 * d5;
    }

    TensorRef<T> ref()
    {
        return TensorRef<T>(data, size());
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4, u64 d5)
    {
        always_assert(isOwner);
        if (this->d1 == d1 && this->d2 == d2 && this->d3 == d3 && this->d4 == d4 && this->d5 == d5)
        {
            return;
        }
        delete[] data;
        this->d1 = d1;
        this->d2 = d2;
        this->d3 = d3;
        this->d4 = d4;
        this->d5 = d5;
        data = new T[d1 * d2 * d3 * d4 * d5];
    }

    void resize(const std::vector<u64> &shape)
    {
        always_assert(isOwner);
        always_assert(shape.size() == 5);
        resize(shape[0], shape[1], shape[2], shape[3], shape[4]);
    }

    T &operator()(u64 i, u64 j, u64 k, u64 l, u64 m) const
    {
        assert(i < d1);
        assert(j < d2);
        assert(k < d3);
        assert(l < d4);
        assert(m < d5);
        return data[i * d2 * d3 * d4 * d5 + j * d3 * d4 * d5 + k * d4 * d5 + l * d5 + m];
    }

    Tensor<T> as_nd()
    {
        return Tensor<T>(data, d_data, {d1, d2, d3, d4, d5});
    }
};
