#include <sytorch/tensor.h>
#include <fstream>
#include <filesystem>

template <typename T>
class SytorchModule {
public:

    Tensor4D<T> activation;
    Backend<T> *backend = new ClearText<T>;
    LayerGraphNode<T> *root = nullptr;
    bool debug = true;
    u64 scale;

public:

    virtual Tensor4D<T>& _forward(Tensor4D<T> &input) = 0;

    SytorchModule() : activation(0, 0, 0, 0)
    {

    }

    void init(u64 d1, u64 d2, u64 d3, u64 d4, u64 scale)
    {
        Tensor4D<T> ip(d1, d2, d3, d4);
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        Layer<T>::fakeExecution = true;
        auto &res = this->_forward(ip);
        Layer<T>::fakeExecution = false;
        root = ip.graphNode;
        activation.resize(res.d1, res.d2, res.d3, res.d4);
        
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            auto &inp = node->layer->inputDerivative;
            node->layer->init(inp.d1, inp.d2, inp.d3, inp.d4, scale);
        });

        this->scale = scale;
    }

    void zero()
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            if (node->layer->name == "Conv2D" || node->layer->name == "FC") {
                node->layer->getweights().fill(0);
                node->layer->getbias().fill(0);
            }
            else if (node->layer->name == "BatchNorm2dInference") {
                BatchNorm2dInference<T> *bn = (BatchNorm2dInference<T> *) node->layer;
                bn->A.fill(0);
                bn->B.fill(0);
            }
        });
    }

    void setBackend(Backend<T> *b)
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->setBackend(b);
        });
        backend = b;
    }

    Tensor4D<T>& forward(Tensor4D<T> &input)
    {
        if (debug) {
            auto res = this->_forward(input);
            this->activation.copy(res);
            return this->activation;
        }
        else {
            auto res = this->_forward(input); // todo: calculate using the generated graph
            this->activation.copy(res);
            return this->activation;
        }
    }

    void optimize()
    {
        backend->optimize(root);
    }

    void load(const std::string weightsFile)
    {
        size_t size_in_bytes = std::filesystem::file_size(weightsFile);
        always_assert(size_in_bytes % 4 == 0); // as it's float
        size_t numParameters = size_in_bytes / 4;
        float *floatWeights = new float[numParameters];
        
        std::ifstream file(weightsFile, std::ios::binary);
        file.read((char*) floatWeights, size_in_bytes);
        file.close();
        u64 scale = this->scale;
        
        size_t wIdx = 0;
        topologicalApply(root, [&wIdx, &floatWeights, &scale](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            auto layer = node->layer;
            if(layer->name.find("Conv2D") != std::string::npos || layer->name.find("FC") != std::string::npos) {
                auto& weights = layer->getweights();

                for (int j = 0; j < weights.d1; j++) {
                    for(int k = 0; k < weights.d2; ++k) {
                        weights(j, k) = floatWeights[wIdx + weights.d2 * j + k] * (1LL << scale);
                    }
                }
                
                auto wSize = weights.d1 * weights.d2;
                wIdx += wSize;

                auto& bias = layer->getbias();

                for (int j = 0; j < bias.size; ++j) {
                    bias(j) = floatWeights[wIdx + j] * (1LL << (2*scale));
                }

                wSize = bias.size;
                wIdx += wSize;
            }
            else if (layer->name.find("BatchNorm2dInference") != std::string::npos) {
                auto bn = (BatchNorm2dInference<T>*) layer;
                auto channel = bn->A.size;
                auto gammaPtr = floatWeights + wIdx;
                auto betaPtr = floatWeights + wIdx + channel;
                auto meanPtr = floatWeights + wIdx + 2 * channel;
                auto varPtr = floatWeights + wIdx + 3 * channel;
                for (int j = 0; j < channel; ++j) {
                    bn->A(j) = (gammaPtr[j] / std::sqrt(varPtr[j])) * (1LL << scale);
                    bn->B(j) = (betaPtr[j] - gammaPtr[j] * meanPtr[j] / std::sqrt(varPtr[j])) * (1LL << (2 * scale));
                }
                wIdx += 4 * channel;
            }
        });
    }
};