#include <sytorch/tensor.h>
#include <fstream>
#include <filesystem>
#include <map>
#include <algorithm>

template <typename T>
class SytorchModule {
public:

    Tensor<T> activation;
    Backend<T> *backend = new ClearText<T>;
    LayerGraphNode<T> *root = nullptr;
    bool debug = true;
    u64 scale;
    // std::map<std::string, LayerGraphNode<T> *> addLayerMap;
    // std::map<std::string, LayerGraphNode<T> *> concatLayerMap;
    std::vector<LayerGraphNode<T> *> allNodesInExecutionOrder;

    const std::vector<std::string> functionalLayers = {"Add", "Concat"};
    std::map<std::string, LayerGraphNode<T> *> functionalLayerMap;

public:

    virtual Tensor<T>& _forward(Tensor<T> &input) = 0;

    SytorchModule() : activation({}), allNodesInExecutionOrder(0)
    {

    }

    void generateFunctionalLayerMap()
    {
        functionalLayerMap.clear();
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            if (std::find(functionalLayers.begin(), functionalLayers.end(), node->layer->name) != functionalLayers.end()) {
                std::string id = node->layer->name;
                for(auto& parent: node->parents) {
                    id += "|" + std::to_string((uint64_t)(parent));
                }
                functionalLayerMap[id] = node;
            }
        });
    }

    LayerGraphNode<T> *getFunctionalNode(const std::string &layerName, std::vector<Tensor<T> *> ips)
    {
        std::string id = layerName;
        for(auto& ip: ips) {
            id += "|" + std::to_string((uint64_t)(ip->graphNode));
        }
        if (functionalLayerMap.find(id) == functionalLayerMap.end()) {
            std::cerr << "Layer not found" << std::endl;
            exit(1);
        }
        return functionalLayerMap[id];
    }

    void functionalGraphGen(const std::string &layerName, std::vector<Tensor<T> *> &arr, Tensor<T> &c)
    {
        for (auto &a : arr) {
            always_assert(a->graphGenMode);
        }
        c.graphNode = new LayerGraphNode<T>();
        c.graphNode->layer = new PlaceHolderLayer<T>(layerName);
        for (auto &a : arr) {
            c.graphNode->parents.push_back(a->graphNode);
            a->graphNode->children.push_back(c.graphNode);
        }
        c.graphNode->allNodesInExecutionOrderRef = arr[0]->graphNode->allNodesInExecutionOrderRef;
        c.graphNode->allNodesInExecutionOrderRef->push_back(c.graphNode);
        c.graphGenMode = true;
    }

    void genGraphAndExecutionOrder()
    {
        Tensor<T> ip({});
        ip.graphGenMode = true;
        ip.graphNode = new LayerGraphNode<T>();
        ip.graphNode->layer = new PlaceHolderLayer<T>("Input");
        ip.graphNode->allNodesInExecutionOrderRef = &allNodesInExecutionOrder;
        auto &res = this->_forward(ip);
        ip.graphGenMode = false;
        root = ip.graphNode;
    }

    void init(u64 scale)
    {
        genGraphAndExecutionOrder();
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->initScale(scale);
        });

        this->scale = scale;
        generateFunctionalLayerMap();
    }

    void zero()
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            if (node->layer->name == "Conv2D" || node->layer->name == "FC" || node->layer->name == "Conv3D" || node->layer->name == "ConvTranspose3D") {
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

    Tensor<T>& forward(Tensor<T> &input)
    {
        topologicalApply(root, [](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->numUsages = 0;
        });
        input.graphNode = root;
        input.graphNode->currTensor = &input;
        if (debug) {
            auto& res = this->_forward(input);
            this->activation.resize(res.shape);
            this->activation.copy(res);
            return this->activation;
        }
        else {
            auto& res = this->_forward(input); // todo: calculate using the generated graph
            this->activation.resize(res.shape);
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
        for (auto &node: allNodesInExecutionOrder) {
            auto layer = node->layer;
            if(layer->name.find("Conv2D") != std::string::npos || layer->name.find("FC") != std::string::npos) {
                auto& weights = layer->getweights();
                for (int j = 0; j < weights.d1; j++) {
                    for(int k = 0; k < weights.d2; ++k) {
                        weights(j, k) = i64(floatWeights[wIdx + weights.d2 * j + k] * (1LL << scale));
                    }
                }

                auto wSize = weights.d1 * weights.d2;
                wIdx += wSize;

                auto& bias = layer->getbias();
                if (layer->useBias) {

                    for (int j = 0; j < bias.size; ++j) {
                        bias(j) = i64(floatWeights[wIdx + j] * (1LL << (2*scale)));
                    }

                    wSize = bias.size;
                    wIdx += wSize;
                }
                else
                    bias.fill(0);
            }
            else if (layer->name.find("BatchNorm2dInference") != std::string::npos) {
                auto bn = (BatchNorm2dInference<T>*) layer;
                auto channel = bn->A.size;
                auto gammaPtr = floatWeights + wIdx;
                auto betaPtr = floatWeights + wIdx + channel;
                auto meanPtr = floatWeights + wIdx + 2 * channel;
                auto varPtr = floatWeights + wIdx + 3 * channel;
                for (int j = 0; j < channel; ++j) {
                    bn->A(j) = i64((gammaPtr[j] / std::sqrt(varPtr[j])) * (1LL << scale));
                    bn->B(j) = i64((betaPtr[j] - gammaPtr[j] * meanPtr[j] / std::sqrt(varPtr[j])) * (1LL << (2 * scale)));
                }
                wIdx += 4 * channel;
            }
        }

        always_assert(wIdx == numParameters);
        delete[] floatWeights;
    }

    Tensor<T> add(std::vector<Tensor<T> *> &arr)
    {
        Tensor<T> c(arr[0]->shape);
        if (arr[0]->graphGenMode) {
            functionalGraphGen("Add", arr, c);
            return c;
        }

        // check if all tensors have same dimensions
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[i]->is_same_shape(*arr[0]) == false) {
                throw std::runtime_error("All tensors must have same dimensions");
            }
        }

        auto cNode = getFunctionalNode("Add", arr);
        cNode->currTensor = &c;
        c.graphNode = cNode;

        u64 sz = c.size();
        for (int i = 0; i < sz; ++i) {
            c.data[i] = 0;
            for (auto &a : arr) {
                c.data[i] += a->data[i];
            }
        }

        for (auto &a : arr) {
            bool gcHappened = a->graphNode->incrementAndGc();
        }
        return c;
    }

    template <typename... Args>
    Tensor<T> add(Args & ... args)
    {
        auto res = collect(args...);
        return add(res);
    }

    // doesnt have assertions, don't use directly, use the variadic version only
    void concat(std::vector<Tensor<T> *> &arr, Tensor<T> &c)
    {
        if (arr[0]->graphGenMode) {
            functionalGraphGen("Concat", arr, c);
            return;
        }

        auto cNode = getFunctionalNode("Concat", arr);
        cNode->currTensor = &c;
        c.graphNode = cNode;

        u64 sz = c.size();
        for(int i = 0; i < sz; ++i)
        {
            u64 l = i % c.shape.back();
            for(auto &a : arr) {
                if(l < a->shape.back()) {
                    c.data[i] = a->data[i];
                    break;
                }
                l -= a->shape.back();
            }
        }

        for (auto &a : arr) {
            bool gcHappened = a->graphNode->incrementAndGc();
        }
    }

    Tensor<T> concat(std::vector<Tensor<T> *> &arr)
    {
        u64 channels = 0;
        for (auto &a : arr) {
            always_assert(a->shape.size() == arr[0]->shape.size());
            for (int i = 0; i < a->shape.size() - 1; ++i) {
                always_assert(a->shape[i] == arr[0]->shape[i]);
            }
            channels += a->shape.back();
        }
        std::vector<u64> shape = arr[0]->shape;
        shape.back() = channels;
        Tensor<T> c(shape);
        concat(arr, c);
        return c;
    }

    template <typename... Args>
    Tensor<T> concat(Args & ... args)
    {
        auto res = collect(args...);
        return concat(res);
    }

    void train()
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->train();
        });
    }

    void eval()
    {
        topologicalApply(root, [=](LayerGraphNode<T> *node, LayerGraphNode<T> *_root) {
            node->layer->eval();
        });
    }
};