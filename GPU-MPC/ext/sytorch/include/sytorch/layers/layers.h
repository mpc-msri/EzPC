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
#include <sytorch/utils.h>
#include <llama/assert.h>
#include <sytorch/backend/default.h>
#include <string>
#include <omp.h>

template <typename T>
bool shapesOkay(std::vector<Tensor<T> *> &a)
{
    if (a.size() == 0)
        return false;
    for (auto i : a)
    {
        if (i->size() == 0)
            return false;
        for (auto j : i->shape)
        {
            if (j <= 0)
            {
                return false;
            }
        }
    }
    // printf("Resizing the input\n");
    return true;
}

template <typename T>
class Layer
{
public:
    std::string name;
    Tensor<T> activation, inputDerivative;
    Backend<T> *backend = nullptr;

    // config options
    bool doTruncationForward = false;
    bool doPreSignExtension = false;
    bool doPostSignExtension = false;
    bool isFirst = false;
    u64 scale = 0;
    int mode = 0; // only used in ReLU in llama improved to decide between relu and reluext, might need something cleaner?
    int forwardTruncationMode = 0;
    bool useBias = true;
    bool isTrainingMode = false;
    std::string paramstring = "";

    LayerGraphNode<T> *node = nullptr;

    Layer(){};

    Layer(const std::string &name) : activation({0}), name(name)
    {
        backend = defaultBackend<T>();
    }

    virtual void _initScale(u64 scale) {};
    void initScale(u64 scale)
    {
        always_assert(std::is_integral<T>::value || scale == 0);
        this->scale = scale;
        _initScale(scale);
    };

    virtual void _resize(const std::vector<std::vector<u64>> &shapes) {};
    void resize(const std::vector<std::vector<u64>> &shapes)
    {
        inputDerivative.resize(shapes[0]);
        // printf("Shapes=%d\n", shapes[0].size());
        auto outdims = this->get_output_dims(shapes);
        activation.resize(outdims);
        _resize(shapes);
    }

    virtual void _forward(Tensor<T> &a) = 0;
    virtual void _forward(std::vector<Tensor<T> *> &a)
    {
        if (a.size() != 1)
            throw std::runtime_error("variable input cardinality not supported in this layer");
        _forward(*a[0]);
    };

    template <typename... Args>
    Tensor<T> &forward(Args &...args)
    {
        std::vector<Tensor<T> *> a = collect(args...);
        return forward(a);
    }

    Tensor<T> &forward(std::vector<Tensor<T> *> &a)
    {
        if (a[0]->graphGenMode)
        {
            for (auto &i : a)
            {
                always_assert(i->graphGenMode);
            }
            if (a[0]->graphNode->layer->name == "Input")
            {
                this->isFirst = true;
                // assert(this->name.compare("Conv2D") == 0);
                // printf("Setting isFirst=true for %s\n", this->name.data());
            }
            node = new LayerGraphNode<T>();
            node->layer = this;
            node->allNodesInExecutionOrderRef = a[0]->graphNode->allNodesInExecutionOrderRef;
            node->allNodesInExecutionOrderRef->push_back(node);
            for (auto &i : a)
            {
                auto parentNode = i->graphNode;
                always_assert(parentNode->allNodesInExecutionOrderRef == node->allNodesInExecutionOrderRef);
                node->parents.push_back(parentNode);
                parentNode->children.push_back(node);
            }
            node->currTensor = &activation;
            activation.graphNode = node;
            activation.graphGenMode = true;
            if (shapesOkay(a))
            {
                resize(getShapes(a));
            }
            return activation;
        }
        // check if we have the graph generated already
        always_assert(node != nullptr);
        for (auto &i : a)
        {
            always_assert(i->graphNode != nullptr);
        }
        activation.graphGenMode = false;
        if (activation.size() == 0)
            resize(getShapes(a));
        // node->currTensor = &activation;
        activation.graphNode = node;

        if (doPreSignExtension)
        {
            for (auto &i : a)
            {
                this->backend->signext(*i, scale);
            }
        }
        _forward(a);
        // printf("Layer=%s, doTruncationForward=%d\n", this->name.data(), this->doTruncationForward);
        if (doTruncationForward)
        {
            this->backend->truncateForward(activation, scale, forwardTruncationMode);
        }
        if (doPostSignExtension)
        {
            this->backend->signext(activation, scale);
        }
        for (auto &i : a)
        {
            i->graphNode->incrementAndGc();
        }
        return activation;
    }

    virtual TensorRef<T> getweights() { return TensorRef<T>(nullptr, 0); };
    virtual TensorRef<T> getbias() { return TensorRef<T>(nullptr, 0); };
    virtual std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes) = 0;

    virtual void setBackend(Backend<T> *b)
    {
        backend = b;
    }

    void train()
    {
        isTrainingMode = true;
    }

    void eval()
    {
        isTrainingMode = false;
    }
};

template <typename T>
class Conv2D : public Layer<T>
{
public:
    Tensor4D<T> inp;
    Tensor2D<T> filter;
    Tensor1D<T> bias;

    Tensor2D<T> filterGrad;
    Tensor2D<T> Vw;
    Tensor1D<T> biasGrad;
    Tensor1D<T> Vb;

    u64 ci, co;
    u64 fh, fw, padding, stride;

    Conv2D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("Conv2D"), ci(ci), co(co), fh(f), fw(f),
                                                                                           padding(padding), stride(stride), filter(co, f * f * ci), bias(co), inp({0, 0, 0, 0}), filterGrad(co, f * f * ci), Vw(co, f * f * ci), biasGrad(co), Vb(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv2D(u64 ci, u64 co, const std::array<u64, 2> f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("Conv2D"), ci(ci), co(co), fh(f[0]), fw(f[1]),
                                                                                                                padding(padding), stride(stride), filter(co, f[0] * f[1] * ci), bias(co), inp({0, 0, 0, 0})
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale)
    {
        double xavier = 1.0 / sqrt(ci * fh * fw);
        filter.randomize(xavier * (1ULL << scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL << (2 * scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        always_assert(shape[3] == ci);
        if (this->isTrainingMode)
            inp.resize(shape);
    }

    void _forward(Tensor<T> &a)
    {
        always_assert(a.shape.size() == 4);
        assert(a.shape[3] == ci);
        // if (this->isTrainingMode)
        //     inp.as_nd().copy(a, false, a.d_data != nullptr);
        auto act_4d = this->activation.as_4d();
        this->backend->conv2D(fh, fw, padding, stride, ci, co, a.as_4d(), filter, this->useBias, bias, act_4d, this->isFirst);
        this->activation.d_data = act_4d.d_data;
    }

    TensorRef<T> getweights() { return filter.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        always_assert(inShape[3] == ci);
        u64 newH = (((inShape[1] + 2 * padding - fh) / stride) + 1);
        u64 newW = (((inShape[2] + 2 * padding - fw) / stride) + 1);
        return {inShape[0], newH, newW, co};
    }
};

template <typename T>
class Conv3D : public Layer<T>
{
public:
    Tensor<T> inp;
    Tensor2D<T> filter;
    Tensor1D<T> bias;
    u64 ci, co;
    u64 fd, fh, fw;
    u64 pd, ph, pw;
    u64 sd, sh, sw;
    u64 dd, dh, dw;

    Conv3D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, u64 dialation = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f), fh(f), fw(f),
                                                                                                              pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), filter(co, f * f * f * ci), bias(co), inp({0, 0, 0, 0, 0})
    {
        always_assert(dialation == 1);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv3D(u64 ci, u64 co, const std::array<u64, 3> f, u64 padding = 0, u64 stride = 1, u64 dialation = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]),
                                                                                                                                   pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), dd(dialation), dh(dialation), dw(dialation), filter(co, f[0] * f[1] * f[2] * ci), bias(co), inp({0, 0, 0, 0, 0})
    {
        always_assert(dialation == 1);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv3D(u64 ci, u64 co, const std::array<u64, 3> f, const std::array<u64, 3> padding = {0, 0, 0}, u64 stride = 1, u64 dialation = 1, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]),
                                                                                                                                                                pd(padding[0]), ph(padding[1]), pw(padding[2]), sd(stride), sh(stride), sw(stride), dd(dialation), dh(dialation), dw(dialation), filter(co, f[0] * f[1] * f[2] * ci), bias(co), inp({0, 0, 0, 0, 0})
    {
        always_assert(dialation == 1);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    Conv3D(u64 ci, u64 co, const std::array<u64, 3> f, const std::array<u64, 6> padding = {0, 0, 0, 0, 0, 0}, const std::array<u64, 3> stride = {1, 1, 1}, const std::array<u64, 3> dialation = {1, 1, 1}, bool useBias = false) : Layer<T>("Conv3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]),
                                                                                                                                                                                                                                   pd(padding[0]), ph(padding[1]), pw(padding[2]), sd(stride[0]), sh(stride[1]), sw(stride[2]), dd(dialation[0]), dh(dialation[1]), dw(dialation[2]), filter(co, f[0] * f[1] * f[2] * ci), bias(co), inp({0, 0, 0, 0, 0})
    {
        always_assert(dialation[0] == 1);
        always_assert(dialation[1] == 1);
        always_assert(dialation[2] == 1);
        always_assert(padding[3] == padding[0]);
        always_assert(padding[4] == padding[1]);
        always_assert(padding[5] == padding[2]);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale)
    {
        double xavier = 1.0 / sqrt(ci * fd * fh * fw);
        filter.randomize(xavier * (1ULL << scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL << (2 * scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 5);
        always_assert(shape[4] == ci);
        if (this->isTrainingMode)
            inp.resize(shape);
    }

    void _forward(Tensor<T> &a)
    {
        always_assert(a.shape.size() == 5);
        assert(a.shape[4] == ci);
        auto act_5d = this->activation.as_5d();
        this->backend->conv3D(fd, fh, fw, pd, ph, pw, sd, sh, sw, dd, dh, dw, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->backend->addbias(this->activation, bias);
    }

    TensorRef<T> getweights() { return filter.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 5);
        always_assert(inShape[4] == ci);
        u64 newD = (((inShape[1] + 2 * pd - fd - (fd - 1) * (dd - 1)) / sd) + 1);
        u64 newH = (((inShape[2] + 2 * ph - fh - (fh - 1) * (dh - 1)) / sh) + 1);
        u64 newW = (((inShape[3] + 2 * pw - fw - (fw - 1) * (dw - 1)) / sw) + 1);
        return {inShape[0], newD, newH, newW, co};
    }
};

template <typename T>
class AvgPool2D : public Layer<T>
{
public:
    u64 ks, padding, stride;

    AvgPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("AvgPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride)
    {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
    }

    void _forward(Tensor<T> &a)
    {
        always_assert(a.shape.size() == 4);
        auto a_4d = a.as_4d();
        auto act_4d = this->activation.as_4d();
        this->backend->avgPool2D(ks, padding, stride, a_4d, act_4d, this->scale);
        this->activation.d_data = act_4d.d_data;
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        u64 newH = (((inShape[1] + 2 * padding - ks) / stride) + 1);
        u64 newW = (((inShape[2] + 2 * padding - ks) / stride) + 1);
        return {inShape[0], newH, newW, inShape[3]};
    }
};

template <typename T>
class SumPool2D : public Layer<T>
{
public:
    u64 ks, padding, stride;

    SumPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("SumPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->sumPool2D(ks, padding, stride, a.as_4d(), this->activation.as_4d());
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        u64 newH = (((inShape[1] + 2 * padding - ks) / stride) + 1);
        u64 newW = (((inShape[2] + 2 * padding - ks) / stride) + 1);
        return {inShape[0], newH, newW, inShape[3]};
    }
};

template <typename T>
class MaxPool2D : public Layer<T>
{
public:
    u64 ks, padding, stride;
    Tensor4D<u64> maxIndex;

    MaxPool2D(u64 ks, u64 padding = 0, u64 _stride = 0) : Layer<T>("MaxPool2D"), ks(ks), padding(padding), stride(_stride == 0 ? ks : _stride), maxIndex(0, 0, 0, 0) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        this->maxIndex.resize(this->activation.shape);
    }

    void _forward(Tensor<T> &a)
    {
        auto a_4d = a.as_4d();
        auto act_4d = this->activation.as_4d();
        this->backend->maxPool2D(ks, padding, stride, a_4d, act_4d, maxIndex, this->scale, this->mode);
        this->activation.d_data = act_4d.d_data;
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        u64 newH = (((inShape[1] + 2 * padding - ks) / stride) + 1);
        u64 newW = (((inShape[2] + 2 * padding - ks) / stride) + 1);
        // printf("%d, %d, %d, %d\n", newH, newW, inShape[1], inShape[2]);
        return {inShape[0], newH, newW, inShape[3]};
    }
};

template <typename T>
class Flatten : public Layer<T>
{
public:
    bool transpose = true;
    Flatten() : Layer<T>("Flatten") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() >= 2);
    }

    void _forward(Tensor<T> &a)
    {
        if (transpose && (a.shape.size() == 4 || a.shape.size() == 5))
        {
            if (a.shape.size() == 4)
            {
                auto a_4d = a.as_4d();
                auto act_2d = this->activation.as_2d();
                u64 d1 = a.shape[0];
                u64 d2 = a.shape[1];
                u64 d3 = a.shape[2];
                u64 d4 = a.shape[3];

#pragma omp parallel for collapse(4)
                for (u64 i = 0; i < d1; i++)
                {
                    for (u64 j = 0; j < d2; j++)
                    {
                        for (u64 k = 0; k < d3; k++)
                        {
                            for (u64 l = 0; l < d4; l++)
                            {
                                act_2d(i, l * d2 * d3 + j * d3 + k) = a_4d(i, j, k, l);
                            }
                        }
                    }
                }
            }
            else if (a.shape.size() == 5)
            {
                auto a_5d = a.as_5d();
                auto act_2d = this->activation.as_2d();
                u64 d1 = a.shape[0];
                u64 d2 = a.shape[1];
                u64 d3 = a.shape[2];
                u64 d4 = a.shape[3];
                u64 d5 = a.shape[4];

#pragma omp parallel for collapse(5)
                for (u64 i = 0; i < d1; i++)
                {
                    for (u64 j = 0; j < d2; j++)
                    {
                        for (u64 k = 0; k < d3; k++)
                        {
                            for (u64 l = 0; l < d4; l++)
                            {
                                for (u64 m = 0; m < d5; ++m)
                                {
                                    act_2d(i, m * d2 * d3 * d4 + j * d3 * d4 + k * d4 + l) = a_5d(i, j, k, l, m);
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            u64 sz = a.size();
#pragma omp parallel for
            for (u64 i = 0; i < sz; i++)
            {
                this->activation.data[i] = a.data[i];
            }
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        u64 prod = 1;
        for (int i = 1; i < inShape.size(); i++)
        {
            prod *= inShape[i];
        }
        return {inShape[0], prod};
    }
};

template <typename T>
class FC : public Layer<T>
{
public:
    Tensor<T> inp;
    Tensor2D<T> weight;
    Tensor1D<T> bias;
    u64 in, out;

    Tensor2D<T> weightGrad;
    Tensor2D<T> Vw;
    Tensor1D<T> Vb;

    FC(u64 in, u64 out, bool useBias = false) : Layer<T>("FC"), in(in), out(out), weight(in, out), bias(out), inp({0, 0, 0, 0}), weightGrad(in, out), Vw(in, out), Vb(out)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale)
    {
        double xavier = 1.0 / sqrt(in);
        weight.randomize(xavier * (1ULL << scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL << (2 * scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 2);
        always_assert(shape[1] == in);
        inp.resize(shape);
    }

    void _forward(Tensor<T> &a)
    {
        auto a_2d = a.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->matmul(a_2d, weight, act_2d, this->useBias, bias, this->isFirst);
        this->activation.d_data = act_2d.d_data;
    }

    TensorRef<T> getweights() { return weight.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        // always_assert(inShape.size() == 2);
        // assert(inShape[1] == in);
        u64 sz = 1;
        for (int i = 1; i < inShape.size(); i++)
            sz *= inShape[i];
        always_assert(sz == in);
        return {inShape[0], out};
    }
};

template <typename T>
class ReLU : public Layer<T>
{
public:
    Tensor<T> drelu;
    ReLU() : Layer<T>("ReLU"), drelu({0}) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        this->drelu.resize(shape);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->relu(a, this->activation, this->drelu, this->scale, this->mode);
        // printf("Relu=%ld, %ld\n", this->activation.data[0], this->activation.data[1]);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class BatchNormInference : public Layer<T>
{
public:
    Tensor1D<T> A; // scale = s
    Tensor1D<T> B; // scale = 2s

    BatchNormInference(u64 channels) : Layer<T>("BatchNormInference"), A(channels), B(channels)
    {
        this->A.fill(0);
        this->B.fill(0);
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        // always_assert(shape.size() == 4);
        always_assert(shape.back() == this->A.d1);
    }

    void _forward(Tensor<T> &a)
    {
        // always_assert(a.shape.size() == 4);
        assert(a.shape.back() == this->A.d1);
        if (this->isTrainingMode)
        {
            std::runtime_error("BatchNormInference should not be used in training mode");
        }
        else
        {
            this->backend->batchNormInference(this->A, this->B, a, this->activation, this->scale);
        }
    }

    TensorRef<T> getweights() { return A.ref(); }
    TensorRef<T> getbias() { return B.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        // always_assert(inShape.size() == 4);
        always_assert(inShape.back() == this->A.d1);
        return inShape;
    }
};

template <typename T>
class Identity : public Layer<T>
{
public:
    Identity() : Layer<T>("Identity") {}

    void _forward(Tensor<T> &a)
    {
        this->activation.copy(a, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class GlobalAvgPool2D : public Layer<T>
{
public:
    GlobalAvgPool2D() : Layer<T>("GlobalAvgPool2D")
    {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 4);
        always_assert(shape[1] == shape[2]);
    }

    void _forward(Tensor<T> &a)
    {
        auto a_4d = a.as_4d();
        auto act_4d = this->activation.as_4d();
        this->backend->avgPool2D(a_4d.d2, 0, 1, a_4d, act_4d, this->scale);
        this->activation.d_data = act_4d.d_data;
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 4);
        always_assert(inShape[1] == inShape[2]);
        return {inShape[0], 1, 1, inShape[3]};
    }
};

template <typename T>
class ConvTranspose3D : public Layer<T>
{
public:
    Tensor2D<T> filter;
    Tensor1D<T> bias;
    u64 ci, co;
    u64 fd, fh, fw;
    u64 pd, ph, pw;
    u64 sd, sh, sw;

    ConvTranspose3D(u64 ci, u64 co, u64 f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f), fh(f), fw(f),
                                                                                                    pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), filter(co, f * f * f * ci), bias(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    ConvTranspose3D(u64 ci, u64 co, const std::array<u64, 3> f, u64 padding = 0, u64 stride = 1, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]),
                                                                                                                         pd(padding), ph(padding), pw(padding), sd(stride), sh(stride), sw(stride), filter(co, f[0] * f[1] * f[2] * ci), bias(co)
    {
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    ConvTranspose3D(u64 ci, u64 co, const std::array<u64, 3> f, const std::array<u64, 6> padding = {0, 0, 0, 0, 0, 0}, const std::array<u64, 3> stride = {1, 1, 1}, const std::array<u64, 3> dialation = {1, 1, 1}, bool useBias = false) : Layer<T>("ConvTranspose3D"), ci(ci), co(co), fd(f[0]), fh(f[1]), fw(f[2]),
                                                                                                                                                                                                                                            pd(padding[0]), ph(padding[1]), pw(padding[2]), sd(stride[0]), sh(stride[1]), sw(stride[2]), filter(co, f[0] * f[1] * f[2] * ci), bias(co)
    {
        always_assert(dialation[0] == 1);
        always_assert(dialation[1] == 1);
        always_assert(dialation[2] == 1);
        always_assert(padding[3] == padding[0]);
        always_assert(padding[4] == padding[1]);
        always_assert(padding[5] == padding[2]);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _initScale(u64 scale)
    {
        double xavier = 1.0 / sqrt(ci * fd * fh * fw);
        filter.randomize(xavier * (1ULL << scale));
        if (this->useBias)
            bias.randomize(xavier * (1ULL << (2 * scale)));
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 5);
        always_assert(shape[4] == ci);
    }

    void _forward(Tensor<T> &a)
    {
        always_assert(a.shape.size() == 5);
        assert(a.shape[4] == ci);
        auto act_5d = this->activation.as_5d();
        this->backend->convTranspose3D(fd, fh, fw, pd, ph, pw, sd, sh, sw, ci, co, a.as_5d(), filter, act_5d);
        if (this->useBias)
            this->backend->addbias(this->activation, bias);
    }

    TensorRef<T> getweights() { return filter.ref(); }
    TensorRef<T> getbias() { return bias.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.size() == 5);
        always_assert(inShape[4] == ci);
        u64 newD = (((inShape[1] - 1) * sd + fd - 2 * pd));
        u64 newH = (((inShape[2] - 1) * sh + fh - 2 * ph));
        u64 newW = (((inShape[3] - 1) * sw + fw - 2 * pw));
        return {inShape[0], newD, newH, newW, co};
    }
};

template <typename T>
class PlaceHolderLayer : public Layer<T>
{
public:
    PlaceHolderLayer(const std::string &s) : Layer<T>(s)
    {
    }

    void initScale(u64 scale)
    {
        throw std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("PlaceHolderLayer only to be used for tree traversal");
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class Add : public Layer<T>
{
public:
    Add() : Layer<T>("Add") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        auto &shape0 = shapes[0];
        for (auto &shape : shapes)
        {
            always_assert(shape.size() == shape0.size());
            for (u64 i = 0; i < shape.size(); i++)
            {
                always_assert(shape[i] == shape0[i]);
            }
        }
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        this->backend->add(a, this->activation);
    }

    void _forward(Tensor<T> &a)
    {
        this->activation.copy(a, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        auto &shape0 = inShapes[0];
        for (auto &shape : inShapes)
        {
            always_assert(shape.size() == shape0.size());
            for (u64 i = 0; i < shape.size(); i++)
            {
                assert(shape[i] == shape0[i]);
            }
        }
        auto &inShape = inShapes[0];
        return inShape;
    }
};

// concat along the last axis (channel)
template <typename T>
class Concat : public Layer<T>
{
public:
    Concat() : Layer<T>("Concat") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        auto &shape0 = shapes[0];
        for (auto &shape : shapes)
        {
            for (u64 i = 0; i < shape.size() - 1; i++)
            {
                always_assert(shape[i] == shape0[i]);
            }
        }
    }

    void _forward(std::vector<Tensor<T> *> &arr)
    {
        u64 outchannels = 0;
        u64 sz = 0;
        for (auto &t : arr)
        {
            outchannels += t->shape.back();
            sz += t->size();
        }

#pragma omp parallel for
        for (int i = 0; i < sz; ++i)
        {
            u64 l = i % outchannels;
            u64 rest = i / outchannels;
            for (auto &a : arr)
            {
                if (l < a->shape.back())
                {
                    this->activation.data[i] = a->data[rest * a->shape.back() + l];
                    break;
                }
                l -= a->shape.back();
            }
        }
    }

    void _forward(Tensor<T> &a)
    {
        this->activation.copy(a, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        auto &shape0 = inShapes[0];
        for (auto &shape : inShapes)
        {
            for (u64 i = 0; i < shape.size() - 1; i++)
            {
                always_assert(shape[i] == shape0[i]);
            }
        }

        std::vector<u64> outShape = shape0;
        outShape.back() = 0;
        for (auto &shape : inShapes)
        {
            outShape.back() += shape.back();
        }
        return outShape;
    }
};

template <typename T>
class GeLU : public Layer<T>
{
public:
    GeLU() : Layer<T>("GeLU") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->gelu(a, this->activation, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class SiLU : public Layer<T>
{
public:
    SiLU() : Layer<T>("SiLU") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->silu(a, this->activation, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class SoftMax : public Layer<T>
{
public:
    SoftMax() : Layer<T>("SoftMax") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        always_assert(shapes[0].size() == 2);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->softmax(a, this->activation, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        always_assert(inShapes[0].size() == 2);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class SoftMaxTriangular : public Layer<T>
{
public:
    SoftMaxTriangular() : Layer<T>("SoftMaxTriangular") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        always_assert(shapes[0].size() == 2);
        always_assert(shapes[0][0] == shapes[0][1]);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->softmax_triangular(a, this->activation, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        always_assert(inShapes[0].size() == 2);
        auto &inShape = inShapes[0];
        always_assert(inShape[0] == inShape[1]);
        return inShape;
    }
};

template <typename T>
class LayerNorm : public Layer<T>
{
public:
    Tensor1D<T> A; // scale = s
    Tensor1D<T> B; // scale = 2s

    LayerNorm(u64 channels) : Layer<T>("LayerNorm"), A(channels), B(channels)
    {
        this->A.fill(0);
        this->B.fill(0);
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.back() == this->A.d1);
    }

    void _forward(Tensor<T> &a)
    {
        // always_assert(a.shape.size() == 4);
        assert(a.shape.back() == this->A.d1);
        // A.as_nd().print();
        this->backend->layernorm(this->A, this->B, a, this->activation, this->scale);
    }

    TensorRef<T> getweights() { return A.ref(); }
    TensorRef<T> getbias() { return B.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.back() == this->A.d1);
        return inShape;
    }
};

template <typename T>
class RMSNorm : public Layer<T>
{
public:
    Tensor1D<T> A; // scale = s
    Tensor1D<T> B; // scale = 2s

    RMSNorm(u64 channels, bool useBias = false) : Layer<T>("RMSNorm"), A(channels), B(channels)
    {
        this->A.fill(0);
        this->B.fill(0);
        this->doTruncationForward = true;
        this->useBias = useBias;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.back() == this->A.d1);
    }

    void _forward(Tensor<T> &a)
    {
        // always_assert(a.shape.size() == 4);
        assert(a.shape.back() == this->A.d1);
        this->backend->rmsnorm(this->A, this->B, a, this->activation, this->scale);
    }

    TensorRef<T> getweights() { return A.ref(); }
    TensorRef<T> getbias() { return B.ref(); }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        always_assert(inShape.back() == this->A.d1);
        return inShape;
    }
};

template <typename T>
class Split : public Layer<T>
{
public:
    u64 n_splits;

    Split(u64 n_splits) : Layer<T>("Split"), n_splits(n_splits) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.back() % n_splits == 0);
    }

    void _forward(Tensor<T> &a)
    {
        always_assert(a.shape.back() % n_splits == 0);
        u64 split_size = a.shape.back() / n_splits; // 3
        u64 rest_size = a.size() / a.shape.back();  // 2

#pragma omp parallel for
        for (u64 i = 0; i < a.size(); ++i)
        {
            u64 p = i / a.shape.back();
            u64 q = i % a.shape.back();
            u64 r = q / split_size;
            u64 s = q % split_size;
            this->activation.data[r * split_size * rest_size + p * split_size + s] = a.data[i];
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        always_assert(shape.back() % n_splits == 0);
        shape.back() /= n_splits;
        shape.insert(shape.begin(), n_splits);
        return shape;
    }
};

template <typename T>
class View : public Layer<T>
{
public:
    i64 idx;

    View(i64 idx) : Layer<T>("View"), idx(idx) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        // auto &shape = shapes[0];
        // always_assert(idx < shape[0]);
    }

    void _forward(Tensor<T> &a)
    {
        // always_assert(idx < a.shape[0]);
        // std::cout << idx << std::endl;
        // std::cout << (idx % a.shape[0]) << std::endl;
        u64 i = (idx + a.shape[0]) % a.shape[0];
        auto v = a.view(i);
        this->activation.copy(v, false);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        if (inShapes[0].size() < 1)
            printshape(inShapes[0]);
        shape.erase(shape.begin());
        return shape;
    }
};

template <typename T>
class Transpose : public Layer<T>
{
public:
    Transpose() : Layer<T>("Transpose") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto &shape = shapes[0];
        always_assert(shape.size() == 2);
    }

    void _forward(Tensor<T> &a)
    {
        always_assert(a.shape.size() == 2);
#pragma omp parallel for collapse(2)
        for (u64 i = 0; i < a.shape[0]; ++i)
        {
            for (u64 j = 0; j < a.shape[1]; ++j)
            {
                this->activation.data[j * a.shape[0] + i] = a.data[i * a.shape[1] + j];
            }
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        always_assert(shape.size() == 2);
        return {shape[1], shape[0]};
    }
};

template <typename T>
class _MatMul : public Layer<T>
{
public:
    _MatMul() : Layer<T>("_MatMul")
    {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape0[1] == shape1[0]);
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("single input not allowed in matmul");
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        always_assert(a.size() == 2);
        auto &a0 = *a[0];
        auto a0_2d = a0.as_2d();
        auto &a1 = *a[1];
        auto a1_2d = a1.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->matmul(a0_2d, a1_2d, act_2d);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape0[1] == shape1[0]);
        return {shape0[0], shape1[1]};
    }
};

template <typename T>
class _MatMulTriangular : public Layer<T>
{
public:
    _MatMulTriangular() : Layer<T>("_MatMulTriangular")
    {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape0[1] == shape1[0]);
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("single input not allowed in matmul");
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        always_assert(a.size() == 2);
        auto &a0 = *a[0];
        auto a0_2d = a0.as_2d();
        auto &a1 = *a[1];
        auto a1_2d = a1.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->matmul_triangular(a0_2d, a1_2d, act_2d);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape0[1] == shape1[0]);
        return {shape0[0], shape1[1]};
    }
};

template <typename T>
class _ScalarMul : public Layer<T>
{
public:
    double scalar;

    _ScalarMul(double scalar) : Layer<T>("_ScalarMul"), scalar(scalar)
    {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        T scalarFix = scalar * (1LL << this->scale);
        // a.printshape();
        // this->activation.printshape();
        this->backend->scalarmul(a, scalarFix, this->activation);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &shape0 = inShapes[0];
        return shape0;
    }
};

template <typename T>
class AttentionMask : public Layer<T>
{
public:
    double scalar;

    AttentionMask(double scalar) : Layer<T>("AttentionMask"), scalar(scalar) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto shape = shapes[0];
        always_assert(shape.size() == 2);
        always_assert(shape[0] == shape[1]);
    }

    void _forward(Tensor<T> &a)
    {
        T scalarFix = scalar * (1LL << this->scale);
        this->backend->attention_mask(a, scalarFix, this->activation);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        always_assert(shape.size() == 2);
        always_assert(shape[0] == shape[1]);
        return shape;
    }
};

template <typename T>
class LocalAttentionMask : public Layer<T>
{
public:
    double scalar;
    // u64 window_size;

    LocalAttentionMask(double scalar) : Layer<T>("LocalAttentionMask"), scalar(scalar) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
        auto shape = shapes[0];
        always_assert(shape.size() == 2);
        always_assert(shape[0] == shape[1]);
    }

    void _forward(Tensor<T> &a)
    {
        T scalarFix = scalar * (1LL << this->scale);
        this->backend->local_attention_mask(a, scalarFix, this->activation);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto shape = inShapes[0];
        always_assert(shape.size() == 2);
        always_assert(shape[0] == shape[1]);
        return shape;
    }
};

template <typename T>
class _Tanh : public Layer<T>
{
public:
    _Tanh() : Layer<T>("_Tanh") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->tanh(a, this->activation, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class Unsqueeze : public Layer<T>
{
public:
    Unsqueeze() : Layer<T>("Unsqueeze") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        u64 sz = a.size();
        for (u64 i = 0; i < sz; i++)
        {
            this->activation.data[i] = a.data[i];
        }
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto inShape = inShapes[0];
        inShape.insert(inShape.begin(), 1);
        return inShape;
    }
};

template <typename T>
class AttentionTriangular : public Layer<T>
{
public:
    u64 n_heads;
    AttentionTriangular(u64 n_heads) : Layer<T>("AttentionTriangular"), n_heads(n_heads) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 3);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        auto &shape2 = shapes[2];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape2.size() == 2);
        auto n_seq = shape0[0];
        auto n_embd = shape0[1];
        always_assert(shape1[0] == n_seq);
        always_assert(shape1[1] == n_embd);
        always_assert(shape2[0] == n_seq);
        always_assert(shape2[1] == n_embd);
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("single input not allowed in AttentionTriangular");
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        always_assert(a.size() == 3);
        auto &q = *a[0];
        auto q_2d = q.as_2d();
        auto &k = *a[1];
        auto k_2d = k.as_2d();
        auto &v = *a[2];
        auto v_2d = v.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->attention_triangular(q_2d, k_2d, v_2d, act_2d, this->scale, n_heads);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 3);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        auto &shape2 = inShapes[2];
        always_assert(shape0.size() == 2);
        always_assert(shape1.size() == 2);
        always_assert(shape2.size() == 2);
        auto n_seq = shape0[0];
        auto n_embd = shape0[1];
        always_assert(shape1[0] == n_seq);
        always_assert(shape1[1] == n_embd);
        always_assert(shape2[0] == n_seq);
        always_assert(shape2[1] == n_embd);
        return {shape0[0], shape0[1]};
    }
};

template <typename T>
class _Mul : public Layer<T>
{
public:
    _Mul() : Layer<T>("_Mul")
    {
        this->doTruncationForward = true;
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 2);
        auto &shape0 = shapes[0];
        auto &shape1 = shapes[1];
        always_assert(shape0.size() == shape1.size());
        for (u64 i = 0; i < shape0.size(); i++)
        {
            always_assert(shape0[i] == shape1[i]);
        }
    }

    void _forward(Tensor<T> &a)
    {
        throw std::runtime_error("single input not allowed in mul");
    }

    void _forward(std::vector<Tensor<T> *> &a)
    {
        always_assert(a.size() == 2);
        auto &a0 = *a[0];
        auto &a1 = *a[1];
        this->backend->mul(a0, a1, this->activation);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 2);
        auto &shape0 = inShapes[0];
        auto &shape1 = inShapes[1];
        always_assert(shape0.size() == shape1.size());
        for (u64 i = 0; i < shape0.size(); i++)
        {
            always_assert(shape0[i] == shape1[i]);
        }
        return shape0;
    }
};

template <typename T>
class _MHADummy : public Layer<T>
{
public:
    int n_heads, n_embed, dim_W;
    Tensor2D<T> wQKV, wProj;
    Tensor1D<T> bQKV, bProj;
    std::string attnMask;
    bool selfAttn;
    std::string qkvLayout;
    bool doNormQKt, doRotEmb;

    _MHADummy(int n_heads, int n_embed, int dim_W, std::string attnMask, std::string qkvLayout, bool doNormQKt, bool doRotEmb = false) : Layer<T>("_MHADummy"), n_heads(n_heads), n_embed(n_embed), dim_W(dim_W), attnMask(attnMask), qkvLayout(qkvLayout), wQKV(n_embed, 3 * n_heads * dim_W), wProj(n_embed, n_embed), bQKV(3 * n_heads * dim_W), bProj(n_embed), doNormQKt(doNormQKt), doRotEmb(doRotEmb)
    {
        selfAttn = (attnMask.compare("self") == 0);
    }

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        auto a_2d = a.as_2d();
        auto act_2d = this->activation.as_2d();
        this->backend->mha(n_heads, n_embed, dim_W, selfAttn, doNormQKt, doRotEmb, wQKV, bQKV, wProj, bProj, a_2d, act_2d);
        this->activation.d_data = act_2d.d_data;
        // printf("mha output=%lx\n", this->activation.d_data);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &inShape = inShapes[0];
        return inShape;
    }
};

template <typename T>
class _ScalarDiv : public Layer<T>
{
public:
    double scalar;

    _ScalarDiv(double scalar) : Layer<T>("_ScalarDiv"), scalar(scalar) {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->scalardiv(a, scalar, this->activation, this->scale, this->mode);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &shape0 = inShapes[0];
        return shape0;
    }
};

template <typename T>
class RotaryEmbedding : public Layer<T>
{
public:
    u64 base = 10000;

    RotaryEmbedding() : Layer<T>("RotaryEmbedding") {}

    void _resize(const std::vector<std::vector<u64>> &shapes)
    {
        always_assert(shapes.size() == 1);
    }

    void _forward(Tensor<T> &a)
    {
        this->backend->rotary_embedding(a, this->activation, this->scale);
    }

    std::vector<u64> get_output_dims(const std::vector<std::vector<u64>> &inShapes)
    {
        always_assert(inShapes.size() == 1);
        auto &shape0 = inShapes[0];
        return shape0;
    }
};
