#pragma once
#include <sytorch/layers/layers.h>
#include <llama/assert.h>

template <typename T>
class Sequential {
public:
    std::vector<Layer<T>*> layers;
    Tensor4D<T> activation;
    Backend<T> *backend = new ClearText<T>();
    LayerGraphNode<T> *root = nullptr;

    void initScale(u64 scale) {
        always_assert(std::is_integral<T>::value || scale == 0);
        for(auto &l : layers) {
            l->initScale(scale);
        }
    }

    void resize(u64 d1, u64 d2, u64 d3, u64 d4) {
        for(auto &l : layers) {
            l->resize(d1, d2, d3, d4);
            d1 = l->activation.d1;
            d2 = l->activation.d2;
            d3 = l->activation.d3;
            d4 = l->activation.d4;
        }
        this->activation.resize(d1, d2, d3, d4);
    }

    void genGraph()
    {
        Tensor4D<T> ip(0, 0, 0, 0);
        ip.graphNode = new LayerGraphNode<T>();
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        Layer<T>::fakeExecution = true;
        this->forward(ip);
        Layer<T>::fakeExecution = false;
        root = ip.graphNode;
    }

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale)
    {
        initScale(scale);
        resize(d1, d2, d3, d4);
        genGraph();
    }

    void optimize()
    {
        backend->optimize(root);
    }

    Sequential(std::vector<Layer<T>*> _layers) : layers(_layers), activation(0, 0, 0, 0) {
        int s = layers.size();
        // Set isFirst
        for(int i = 0; i < s; ++i) {
            if (layers[i]->name == "Conv2D" || layers[i]->name == "FC") {
                layers[i]->isFirst = true;
                break;
            }
        }
        
        // Optimization: ReLU-MaxPool
        for(int i = 0; i < s - 1; i++) {
            if (layers[i+1]->name == "MaxPool2D") {
                auto &n = layers[i]->name;
                if (n == "ReLU") {
                    std::swap(layers[i], layers[i+1]);
                }
            }
        }
    }
    
    Tensor4D<T>& forward(Tensor4D<T> &a, bool train = true) {
        layers[0]->forward(a, train);
        u64 size = layers.size();
        for(u64 i = 1; i < size; i++) {
            layers[i]->forward(layers[i-1]->activation, train);
        }
        this->activation.resize(layers[size-1]->activation.d1, layers[size-1]->activation.d2, layers[size-1]->activation.d3, layers[size-1]->activation.d4);
        this->activation.copy(layers[size-1]->activation);
        return this->activation;

    }
    void backward(const Tensor4D<T> &e) {
        int size = layers.size();
        layers[size-1]->backward(e);
        for (int i = size - 2; i >= 0; i--) {
            layers[i]->backward(layers[i+1]->inputDerivative);
        }
    }

    struct layer_dims get_output_dims(struct layer_dims &in) {
        struct layer_dims res = in;
        u64 size = layers.size();
        for(u64 i = 0; i < size; i++) {
            res = layers[i]->get_output_dims(res);
        }
        return res;
    }

    virtual void setBackend(Backend<T> *b) {
        for(auto &l : layers) {
            l->setBackend(b);
        }
        this->backend = b;
    }
};